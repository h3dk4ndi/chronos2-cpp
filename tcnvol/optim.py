import numpy as np 


# ─────────────────────────────────────────────────────────────────────
# Optimiser
# ─────────────────────────────────────────────────────────────────────

class AdamW:
    """
    Adam with decoupled weight decay and global gradient clipping.

    m_t = β₁ m_{t-1} + (1-β₁) g_t
    v_t = β₂ v_{t-1} + (1-β₂) g_t²
    θ_t ← θ_t - α * (m̂_t / (√v̂_t + ε) + λ θ_t)   [weight matrices only]

    References
    ----------
    Kingma & Ba (2014)          — Adam
    Loshchilov & Hutter (2019)  — decoupled weight decay
    Pascanu et al. (2013)       — gradient clipping
    """

    def __init__(
        self,
        lr:           float = 5e-4,
        beta1:        float = 0.9,
        beta2:        float = 0.999,
        eps:          float = 1e-8,
        weight_decay: float = 1e-4,
        clip_norm:    float = 1.0,
    ) -> None:
        self.lr           = lr
        self.beta1        = beta1
        self.beta2        = beta2
        self.eps          = eps
        self.weight_decay = weight_decay
        self.clip_norm    = clip_norm
        self._m:  dict    = {}
        self._v:  dict    = {}
        self._t:  int     = 0

    def __repr__(self) -> str:
        return (f"AdamW(lr={self.lr}, wd={self.weight_decay}, "
                f"clip={self.clip_norm})")

    def step(self, params: dict, grads: dict) -> float:
        """
        Updates params in-place. Returns the pre-clip gradient norm.

        Parameters
        ----------
        params : dict   flat parameter dict from TCN.parameters()
        grads  : dict   flat gradient dict from TCN.gradients()

        Returns
        -------
        float   global gradient norm before clipping
        """
        grad_norm = float(np.sqrt(sum(np.sum(grads[k] ** 2) for k in params)))
        scale     = min(1.0, self.clip_norm / (grad_norm + 1e-12))
        self._t  += 1

        for key in params:
            if key not in self._m:
                self._m[key] = np.zeros_like(params[key])
                self._v[key] = np.zeros_like(params[key])
            g            = grads[key] * scale
            self._m[key] = self.beta1 * self._m[key] + (1 - self.beta1) * g
            self._v[key] = self.beta2 * self._v[key] + (1 - self.beta2) * g**2
            m_hat        = self._m[key] / (1 - self.beta1 ** self._t)
            v_hat        = self._v[key] / (1 - self.beta2 ** self._t)
            if key.endswith("_V") or key == "W_out":   # weight decay on matrices only
                params[key] -= self.lr * (m_hat / (np.sqrt(v_hat) + self.eps)
                                          + self.weight_decay * params[key])
            else:
                params[key] -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

        return grad_norm