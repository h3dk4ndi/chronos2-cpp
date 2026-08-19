import numpy as np

from tcnvol.layers import ResidualBlock, WeightNormConv1d, GELU, _glorot
from tcnvol.config import TCNConfig



# ─────────────────────────────────────────────────────────────────────
# TCN
# ─────────────────────────────────────────────────────────────────────

class TCN:
    """
    Temporal Convolutional Network for binary classification.

    Architecture: stack of ResidualBlocks with increasing dilations,
    followed by a linear readout on the last timestep and sigmoid output.

    Receptive field: 1 + 2(k-1) * Σ d_i

    Parameters
    ----------
    F            : int           number of feature channels
    k            : int           kernel size (shared across all blocks)
    dilations    : list[int]     one dilation per block, e.g. [1, 2, 4]
    dropout_rate : float         dropout rate in all blocks
    C_clean      : np.ndarray    RMT-cleaned correlation matrix for
                                 structured init of first block (optional)

    References
    ----------
    Bai, Kolter & Koltun (2018) arXiv:1803.01271
    """

    def __init__(
        self,
        F:            int,
        k:            int,
        dilations:    list[int],
        dropout_rate: float,
    ) -> None:
        self.F = F
        self.blocks = [ResidualBlock(F, k, d, dropout_rate)
                       for d in dilations]
        self.W_out  = _glorot(F, 1)
        self.b_out  = np.zeros(1, dtype=np.float32)
        self.dW_out = np.zeros_like(self.W_out)
        self.db_out = np.zeros_like(self.b_out)
        self._tcn_out: np.ndarray | None  = None
        self._H_last:  np.ndarray | None  = None

    def __repr__(self) -> str:
        blocks = "\n  ".join(repr(b) for b in self.blocks)
        rf = 1 + 2 * (self.blocks[0].conv1.k - 1) * sum(b.conv1.d for b in self.blocks)
        return (f"TCN(\n  {blocks}\n"
                f"  Linear({self.F} → 1)\n"
                f"  receptive_field={rf}\n)")

    def __call__(self, X: np.ndarray, training: bool = True) -> np.ndarray:
        return self.forward(X, training)

    def __len__(self) -> int:
        return len(self.blocks)

    def forward(self, X: np.ndarray, training: bool = True) -> np.ndarray:
        out = X
        for block in self.blocks:
            out = block.forward(out, training)
        self._tcn_out = out
        self._H_last  = out[:, -1, :]           # (N, F) — last timestep only
        logits        = self._H_last @ self.W_out + self.b_out
        # 19/08/26 -- drop the sigmoid
        return logits   

    def backward(self, dlogits: np.ndarray) -> None:
        self.dW_out = self._H_last.T @ dlogits
        self.db_out = np.sum(dlogits, axis=0)
        dH_last     = dlogits @ self.W_out.T    # (N, F)
        dout        = np.zeros_like(self._tcn_out)
        dout[:, -1, :] = dH_last               # gradient only at last timestep
        for block in reversed(self.blocks):
            dout = block.backward(dout)

    def parameters(self) -> dict:
        params = {}
        for i, block in enumerate(self.blocks):
            for name, p in block.parameters().items():
                params[f"block{i}_{name}"] = p
        params["W_out"] = self.W_out
        params["b_out"] = self.b_out
        return params

    def gradients(self) -> dict:
        grads = {}
        for i, block in enumerate(self.blocks):
            for name, g in block.gradients().items():
                grads[f"block{i}_{name}"] = g
        grads["W_out"] = self.dW_out
        grads["b_out"] = self.db_out
        return grads

    def state_dict(self) -> dict:
        """Returns a deep copy of all parameters."""
        return {k: v.copy() for k, v in self.parameters().items()}

    def load_state(self, state: dict) -> None:
        """Restores parameters from a state dict (in-place copy)."""
        for k, v in state.items():
            self.parameters()[k][:] = v
