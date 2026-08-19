import numpy as np
from tcnvol.tcn    import TCN
from tcnvol.optim  import AdamW
from tcnvol.config import TCNConfig
from tcnvol.losses import mse_loss, qlike


# ─────────────────────────────────────────────────────────────────────
# Trainer
# ─────────────────────────────────────────────────────────────────────

class Trainer:
    """
    Single-seed training loop with early stopping on validation QLIKE.

    Parameters
    ----------
    F       : int         number of input features
    cfg     : TCNConfig   hyperparameter config
    """

    def __init__(self, F: int, cfg: TCNConfig) -> None:
        self.F   = F
        self.cfg = cfg
        self._model: TCN | None = None

    def __repr__(self) -> str:
        return (f"Trainer(F={self.F}, epochs={self.cfg.epochs}, "
                f"batch={self.cfg.batch_size}, patience={self.cfg.patience})")

    def fit(
        self,
        X_train: np.ndarray, y_train: np.ndarray,
        X_val:   np.ndarray, y_val:   np.ndarray,
    ) -> dict:
        """
        Trains the TCN and restores the best checkpoint.

        X_train, X_val : (N, W, F)      standardised on TRAIN statistics only
        y_train, y_val : (N,)           log realised variance at t + ROLL_W

        Returns
        -------
        dict with keys train_loss, val_mse, val_qlike
        """
        model = TCN(self.F, self.cfg.kernel_size, self.cfg.dilations, self.cfg.dropout)

        # start the head at the training mean so it doesn't travel ~9 log units
        model.b_out[:] = float(np.mean(y_train))

        opt = AdamW(self.cfg.lr, weight_decay=self.cfg.weight_decay,
                    clip_norm=self.cfg.clip_norm)

        best_q, best_state = np.inf, None
        patience = 0
        history  = {"train_loss": [], "val_mse": [], "val_qlike": []}

        for epoch in range(1, self.cfg.epochs + 1):
            idx    = np.random.permutation(len(X_train))
            losses = []

            for start in range(0, len(X_train), self.cfg.batch_size):
                b             = idx[start:start + self.cfg.batch_size]
                y_hat         = model.forward(X_train[b], training=True)
                loss, dlogits = mse_loss(y_hat, y_train[b])
                model.backward(dlogits)
                opt.step(model.parameters(), model.gradients())
                losses.append(loss)

            train_loss   = float(np.mean(losses))
            y_val_hat    = self._predict(X_val, model)
            val_mse, _   = mse_loss(y_val_hat, y_val)
            val_q        = qlike(y_val.reshape(-1), y_val_hat.reshape(-1))

            history["train_loss"].append(train_loss)
            history["val_mse"].append(val_mse)
            history["val_qlike"].append(val_q)

            print(f"  Ep {epoch:03d} | train_mse={train_loss:.5f} | "
                  f"val_mse={val_mse:.5f} | val_qlike={val_q:.5f}")

            if val_q < best_q - 1e-6:
                best_q     = val_q
                best_state = model.state_dict()
                patience   = 0
            else:
                patience  += 1

            if patience >= self.cfg.patience:
                print(f"  → Early stop. Best val QLIKE: {best_q:.5f}")
                break

        if best_state is not None:
            model.load_state(best_state)

        self._model = model
        return history

    def predict(self, X: np.ndarray, batch_size: int = 256) -> np.ndarray:
        """Returns predicted log realised variance, shape (N, 1)."""
        if self._model is None:
            raise RuntimeError("Call fit() before predict().")
        return self._predict(X, self._model, batch_size)

    def _predict(self, X: np.ndarray, model: TCN, batch_size: int = 256) -> np.ndarray:
        return np.vstack([
            model.forward(X[s:s + batch_size], training=False)
            for s in range(0, len(X), batch_size)
        ])