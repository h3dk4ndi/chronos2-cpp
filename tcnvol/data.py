import sqlite3
import numpy as np

CONTEXT, ROLL_W = 512, 21
TRAIN_END, VAL_START, VAL_END, TEST_START = 4879, 4900, 5443, 5464


def safelog(x):                       # matches C++ SafeLog
    x = np.asarray(x, float)
    return np.where(x > 0, np.log(np.where(x > 0, x, 1.0)), np.nan)


def build_matrix(db="blp.db", sec="XAU Curncy"):
    """Reproduces BuildMatrix() from src/study.cpp. Returns (14, N)."""
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    q = ("SELECT close2closeRV, parkinson, garmanKlass, rogersSatchell, yangZhang, "
         "returns, negativeRealisedSemivar, positiveRealisedSemivar, bipowerVariation, "
         "signedJump, leverage, leverageMean5, jumpComponent, relativeJump "
         "FROM prep_data WHERE security = ? ORDER BY date")
    cols = np.array(con.execute(q, (sec,)).fetchall(), dtype=float).T
    con.close()

    return np.vstack([
        safelog(cols[0]), safelog(cols[1]), safelog(cols[2]),
        safelog(cols[3]), safelog(cols[4]),
        cols[5],                                            # returns, raw
        safelog(cols[6]), safelog(cols[7]), safelog(cols[8]),
        cols[9], cols[10], cols[11], cols[12], cols[13],    # raw: can be <= 0
    ])


def load_splits(db="blp.db", sec="XAU Curncy", verbose=True):
    """
    Returns (X_tr, y_tr), (X_va, y_va), (X_te, y_te, t_te).
    X : (N, CONTEXT, 14) standardised on TRAIN statistics only
    y : (N,)             RAW log realised variance at t + ROLL_W
    """
    M = build_matrix(db, sec)
    if verbose:
        nan = [int(np.sum(~np.isfinite(r))) for r in M]
        print(f"[data] {M.shape}  NaN per row: {nan}")

    y_all = np.full(M.shape[1], np.nan)
    y_all[:-ROLL_W] = M[0, ROLL_W:]                 # y[t] = M[0][t+21]

    mu    = np.nanmean(M[:, :TRAIN_END], axis=1, keepdims=True)
    sigma = np.nanstd (M[:, :TRAIN_END], axis=1, keepdims=True)
    sigma[sigma < 1e-12] = 1.0
    Xs = (M - mu) / sigma

    def windows(t_lo, t_hi):
        ts = [t for t in range(t_lo, t_hi + 1)
              if np.isfinite(Xs[:, t-CONTEXT+1:t+1]).all() and np.isfinite(y_all[t])]
        X = np.stack([Xs[:, t-CONTEXT+1:t+1].T for t in ts])
        return X.astype(np.float32), y_all[np.array(ts)].astype(np.float32), np.array(ts)

    X_tr, y_tr, _    = windows(CONTEXT - 1, TRAIN_END - ROLL_W)
    X_va, y_va, _    = windows(VAL_START,   VAL_END - ROLL_W)
    X_te, y_te, t_te = windows(TEST_START,  M.shape[1] - ROLL_W - 1)

    return (X_tr, y_tr), (X_va, y_va), (X_te, y_te, t_te)