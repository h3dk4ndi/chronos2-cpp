"""
TCN regression baseline — same origins, same target, same metric as the
C++ Chronos-2 walk-forward. Writes per-origin QLIKE to tcn_loss.csv.
"""
import numpy as np

from tcnvol.config  import TCNConfig
from tcnvol.data    import load_splits
from tcnvol.trainer import Trainer
from tcnvol.losses  import qlike

SEED = 0


def main() -> None:
    np.random.seed(SEED)

    (X_tr, y_tr), (X_va, y_va), (X_te, y_te, t_te) = load_splits()
    print(f"[tcn] train={len(y_tr)}  val={len(y_va)}  test={len(y_te)}")
    print(f"[tcn] window={X_tr.shape[1]}  features={X_tr.shape[2]}  "
          f"origins {t_te[0]}..{t_te[-1]}")

    if len(y_te) != 2321:
        raise SystemExit(f"[tcn] ABORT: {len(y_te)} test origins, expected 2321")

    cfg = TCNConfig()
    tr  = Trainer(F=X_tr.shape[2], cfg=cfg)
    print(f"[tcn] {tr}")

    tr.fit(X_tr, y_tr, X_va, y_va)

    pred = tr.predict(X_te).reshape(-1)
    if not np.isfinite(pred).all():
        raise SystemExit("[tcn] ABORT: non-finite predictions")

    u    = y_te - pred
    loss = np.exp(u) - u - 1.0                     # per-origin QLIKE

    print(f"[tcn] QLIKE  {loss.mean():.6f}")
    print(f"[tcn] bias   {u.mean():+.4f}   (positive = under-predicting)")

    np.savetxt("tcn_loss.csv", loss, fmt="%.17g")
    print(f"[tcn] wrote tcn_loss.csv  ({len(loss)} rows)")


if __name__ == "__main__":
    main()