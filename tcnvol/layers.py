import numpy as np 


# ─────────────────────────────────────────────────────────────────────
# Utility (no state — kept as functions)
# ─────────────────────────────────────────────────────────────────────

def _glorot(fan_in: int, fan_out: int) -> np.ndarray:
    limit = np.sqrt(6.0 / (fan_in + fan_out))
    return np.random.uniform(-limit, limit, (fan_in, fan_out)).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────
# Layers
# ─────────────────────────────────────────────────────────────────────

class WeightNormConv1d:
    """
    Dilated causal 1D convolution with weight normalisation.

    W = g * V / ||V||,   Z[n,t,f] = Σ_c wins[n,t,c] * W[c,f] + b[f]

    Left-padding of (k-1)*d zeros ensures strict causality.

    Parameters
    ----------
    C_in   : int           input channels
    F      : int           output filters
    k      : int           kernel size
    d      : int           dilation
    V_init : np.ndarray    optional structured initialisation (e.g. from RMT)

    References
    ----------
    van den Oord et al. (2016) WaveNet — dilated causal convolutions
    Salimans & Kingma (2016)  — weight normalisation
    Bai et al. (2018)         — TCN
    """

    def __init__(
        self,
        C_in:   int,
        F:      int,
        k:      int,
        d:      int,
        V_init: np.ndarray | None = None,
    ) -> None:
        self.k    = k
        self.d    = d
        self.C_in = C_in
        self.F    = F
        self.V    = V_init.astype(np.float32) if V_init is not None else _glorot(k * C_in, F)
        self.g    = np.linalg.norm(self.V, axis=0, keepdims=True).astype(np.float32)
        self.b    = np.zeros(F, dtype=np.float32)
        self.dV   = np.zeros_like(self.V)
        self.dg   = np.zeros_like(self.g)
        self.db   = np.zeros_like(self.b)
        self._cache: dict = {}

    def __repr__(self) -> str:
        return f"WeightNormConv1d(C_in={self.C_in}, F={self.F}, k={self.k}, d={self.d})"

    def __call__(self, X: np.ndarray) -> np.ndarray:
        return self.forward(X)

    def forward(self, X: np.ndarray) -> np.ndarray:
        col_norms = np.linalg.norm(self.V, axis=0, keepdims=True) + 1e-8
        W         = self.g * self.V / col_norms
        N, T, _   = X.shape
        pad       = (self.k - 1) * self.d
        X_pad     = np.concatenate([np.zeros((N, pad, self.C_in), np.float32), X], axis=1)
        wins      = np.empty((N, T, self.k * self.C_in), np.float32)
        for a in range(self.k):
            wins[:, :, a*self.C_in:(a+1)*self.C_in] = X_pad[:, a*self.d:a*self.d+T, :]
        Z = np.einsum('ntc,cf->ntf', wins, W, optimize=True) + self.b
        self._cache = {"W": W, "col_norms": col_norms, "wins": wins, "X_shape": X.shape}
        return Z.astype(np.float32)

    def backward(self, dZ: np.ndarray) -> np.ndarray:
        W, col_norms, wins      = self._cache["W"], self._cache["col_norms"], self._cache["wins"]
        N, T, C_in              = self._cache["X_shape"]
        dW_eff                  = np.einsum("ntc,ntf->cf", wins, dZ, optimize=True)
        self.db                 = np.sum(dZ, axis=(0, 1))
        dwins                   = np.einsum("ntf,cf->ntc", dZ, W, optimize=True)
        pad                     = (self.k - 1) * self.d
        dX_pad                  = np.zeros((N, T + pad, C_in), np.float32)
        for a in range(self.k):
            dX_pad[:, a*self.d:a*self.d+T, :] += dwins[:, :, a*C_in:(a+1)*C_in]
        dX                      = dX_pad[:, pad:, :]
        V_hat                   = self.V / col_norms
        self.dg                 = np.sum(dW_eff * V_hat, axis=0, keepdims=True)
        self.dV                 = (self.g / col_norms) * (
                                      dW_eff - V_hat * np.sum(dW_eff * V_hat, axis=0, keepdims=True)
                                  )
        return dX.astype(np.float32)

    def parameters(self) -> dict:
        return {"V": self.V, "g": self.g, "b": self.b}

    def gradients(self) -> dict:
        return {"V": self.dV, "g": self.dg, "b": self.db}


class GELU:
    """
    GELU activation (tanh approximation).

    GELU(x) ≈ 0.5x * (1 + tanh(√(2/π) * (x + 0.044715 x³)))

    Reference: Hendrycks & Gimpel (2016) arXiv:1606.08415
    """

    def __init__(self) -> None:
        self._grad: np.ndarray | None = None

    def __repr__(self) -> str:
        return "GELU()"

    def __call__(self, X: np.ndarray) -> np.ndarray:
        return self.forward(X)

    def forward(self, X: np.ndarray) -> np.ndarray:
        arg         = np.sqrt(2.0 / np.pi) * (X + 0.044715 * X**3)
        cdf         = 0.5 * (1.0 + np.tanh(arg))
        sech2       = 1.0 - np.tanh(arg)**2
        darg        = np.sqrt(2.0 / np.pi) * (1.0 + 3.0 * 0.044715 * X**2)
        self._grad  = (cdf + X * 0.5 * sech2 * darg).astype(np.float32)
        return (X * cdf).astype(np.float32)

    def backward(self, dout: np.ndarray) -> np.ndarray:
        return (dout * self._grad).astype(np.float32)


class Dropout:
    """
    Inverted dropout.

    At training time: mask ~ Bernoulli(1-p), scale by 1/(1-p).
    At test time:     identity pass.

    Reference: Srivastava et al. (2014) JMLR 15
    """

    def __init__(self, rate: float) -> None:
        self.rate         = rate
        self._mask: np.ndarray | None = None

    def __repr__(self) -> str:
        return f"Dropout(rate={self.rate})"

    def __call__(self, X: np.ndarray, training: bool = True) -> np.ndarray:
        return self.forward(X, training)

    def forward(self, X: np.ndarray, training: bool = True) -> np.ndarray:
        if not training or self.rate == 0.0:
            self._mask = None
            return X
        self._mask = (np.random.rand(*X.shape) > self.rate).astype(np.float32) / (1.0 - self.rate)
        return (X * self._mask).astype(np.float32)

    def backward(self, dout: np.ndarray) -> np.ndarray:
        return dout if self._mask is None else (dout * self._mask).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────
# ResidualBlock
# ─────────────────────────────────────────────────────────────────────

class ResidualBlock:
    """
    TCN residual block.

    out = Dropout(GELU(Conv2(Dropout(GELU(Conv1(X)))))) + X

    Both conv layers share the same dilation d. The identity shortcut
    requires C_in == F (channel dimension preserved throughout).

    Parameters
    ----------
    F            : int           number of channels
    k            : int           kernel size
    dilation     : int           dilation factor
    dropout_rate : float         dropout probability
    V1_init      : np.ndarray    optional init for first conv (RMT-based)

    References
    ----------
    He et al. (2016) Deep Residual Learning — residual connection
    Bai et al. (2018) TCN — residual block design
    """

    def __init__(
        self,
        F:            int,
        k:            int,
        dilation:     int,
        dropout_rate: float,
        V1_init:      np.ndarray | None = None,
    ) -> None:
        self.conv1 = WeightNormConv1d(F, F, k, dilation, V_init=V1_init)
        self.act1  = GELU()
        self.drop1 = Dropout(dropout_rate)
        self.conv2 = WeightNormConv1d(F, F, k, dilation)
        self.act2  = GELU()
        self.drop2 = Dropout(dropout_rate)

    def __repr__(self) -> str:
        return (f"ResidualBlock(F={self.conv1.F}, k={self.conv1.k}, "
                f"d={self.conv1.d}, dropout={self.drop1.rate})")

    def __call__(self, X: np.ndarray, training: bool = True) -> np.ndarray:
        return self.forward(X, training)

    def forward(self, X: np.ndarray, training: bool = True) -> np.ndarray:
        H = self.drop1.forward(self.act1.forward(self.conv1.forward(X)), training)
        H = self.drop2.forward(self.act2.forward(self.conv2.forward(H)), training)
        return H + X                         # identity residual

    def backward(self, dout: np.ndarray) -> np.ndarray:
        dH = self.drop2.backward(dout)
        dH = self.act2.backward(dH)
        dH = self.conv2.backward(dH)
        dH = self.drop1.backward(dH)
        dH = self.act1.backward(dH)
        dX_conv = self.conv1.backward(dH)
        return dX_conv + dout                # gradient through residual shortcut

    def parameters(self) -> dict:
        return {
            **{f"conv1_{k}": v for k, v in self.conv1.parameters().items()},
            **{f"conv2_{k}": v for k, v in self.conv2.parameters().items()},
        }

    def gradients(self) -> dict:
        return {
            **{f"conv1_{k}": v for k, v in self.conv1.gradients().items()},
            **{f"conv2_{k}": v for k, v in self.conv2.gradients().items()},
        }
