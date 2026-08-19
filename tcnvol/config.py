from dataclasses import dataclass, field

# ─────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────

@dataclass
class TCNConfig:
    """
    Hyperparameter container for the TCN.

    Parameters
    ----------
    kernel_size  : int    conv kernel size (default 3)
    dilations    : list   dilation per block (default [1, 2, 4])
    dropout      : float  dropout rate (default 0.25)
    lr           : float  Adam learning rate (default 5e-4)
    weight_decay : float  AdamW weight decay (default 1e-4)
    clip_norm    : float  global gradient clip norm (default 1.0)
    epochs       : int    max training epochs (default 50)
    batch_size   : int    mini-batch size (default 64)
    patience     : int    early-stopping patience (default 10)
    seeds        : list   random seeds for ensemble (default [1..5])
    """
    kernel_size:  int   = 3
    dilations: list = field(default_factory=lambda: [1,2,4,8,16,32,64])   # RF = 509 (our context is 512)
    dropout:      float = 0.25
    lr:           float = 5e-4
    weight_decay: float = 1e-4
    clip_norm:    float = 1.0
    epochs:       int   = 50
    batch_size:   int   = 64
    patience:     int   = 10
    seeds:        list  = field(default_factory=lambda: [1, 2, 3, 4, 5])

    def __post_init__(self) -> None:
        assert 0 < self.dropout < 1,  "dropout must be in (0, 1)"
        assert self.lr > 0,            "lr must be positive"
        assert self.epochs > 0,        "epochs must be positive"
        assert len(self.dilations) > 0,"dilations must be non-empty"