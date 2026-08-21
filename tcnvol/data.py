import sqlite3
import numpy as np

CONTEXT, ROLL_W = 512, 21
TRAIN_END, VAL_START, VAL_END, TEST_START = 4879, 4900, 5443, 5464


def safelog(x):                       # matches C++ SafeLog
    x = np.asarray(x, float)
    return np.where(x > 0, np.log(np.where(x > 0, x, 1.0)), np.nan)


#def build_matrix(db="blp.db", sec="XAU Curncy"):
#    """Reproduces BuildMatrix() from src/study.cpp. Returns (14, N)."""
#    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
#    q = ("SELECT close2closeRV, parkinson, garmanKlass, rogersSatchell, yangZhang, "
#         "returns, negativeRealisedSemivar, positiveRealisedSemivar, bipowerVariation, "
#         "signedJump, leverage, leverageMean5, jumpComponent, relativeJump "
#         "FROM prep_data WHERE security = ? ORDER BY date")
#    cols = np.array(con.execute(q, (sec,)).fetchall(), dtype=float).T
#    con.close()

#    return np.vstack([
#        safelog(cols[0]), safelog(cols[1]), safelog(cols[2]),
#        safelog(cols[3]), safelog(cols[4]),
#        cols[5],                                            # returns, raw
#        safelog(cols[6]), safelog(cols[7]), safelog(cols[8]),
#        cols[9], cols[10], cols[11], cols[12], cols[13],    # raw: can be <= 0
#    ])


def build_matrix(db="blp.db", sec="XAU Curncy"):
    """
    Builds 14 features for each of nine securities.

    XAU is always the first block, so M[0] remains historical
    XAU log realised variance.

    Returns
    -------
    M : ndarray, shape (126, N)
    """

    securities = [
        sec,
        "XAG Curncy",
        "DXY Curncy",
        "EURUSD Curncy",
        "USDJPY Curncy",
        "USGG10YR Index",
        "CL1 Comdty",
        "HG1 Comdty",
        "VIX Index",
    ]

    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)

    # XAU supplies the master calendar.
    master_dates = [
        row[0]
        for row in con.execute(
            """
            SELECT date
            FROM prep_data
            WHERE security = ?
            ORDER BY date
            """,
            (sec,),
        ).fetchall()
    ]

    N = len(master_dates)

    date_index = {date: i for i, date in enumerate(master_dates)}

    q = """
        SELECT
            date,
            close2closeRV,
            parkinson,
            garmanKlass,
            rogersSatchell,
            yangZhang,
            returns,
            negativeRealisedSemivar,
            positiveRealisedSemivar,
            bipowerVariation,
            signedJump,
            leverage,
            leverageMean5,
            jumpComponent,
            relativeJump
        FROM prep_data
        WHERE security = ?
        ORDER BY date
    """

    blocks = []

    for security in securities:
        # 14 rows aligned to the XAU calendar.
        cols = np.full((14, N), np.nan, dtype=float)

        rows = con.execute(q, (security,)).fetchall()

        for row in rows:
            date = row[0]
            position = date_index.get(date)

            # Ignore dates not present in the XAU calendar.
            if position is None:
                continue

            cols[:, position] = [
                np.nan if value is None else float(value)
                for value in row[1:]
            ]

        block = np.vstack([
            safelog(cols[0]),
            safelog(cols[1]),
            safelog(cols[2]),
            safelog(cols[3]),
            safelog(cols[4]),
            cols[5],
            safelog(cols[6]),
            safelog(cols[7]),
            safelog(cols[8]),
            cols[9],
            cols[10],
            cols[11],
            cols[12],
            cols[13],
        ])

        blocks.append(block)

    con.close()

    M = np.vstack(blocks)

    expected_shape = (14 * len(securities), N)

    if M.shape != expected_shape:
        raise RuntimeError(
            f"matrix shape {M.shape}, "
            f"expected {expected_shape}"
        )

    return M

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

    # Zero on the standardised scale equals the training mean.
    Xs[~np.isfinite(Xs)] = 0.0

    def windows(t_lo, t_hi):
        ts = [t for t in range(t_lo, t_hi + 1)
              if np.isfinite(Xs[:, t-CONTEXT+1:t+1]).all() and np.isfinite(y_all[t])]
        X = np.stack([Xs[:, t-CONTEXT+1:t+1].T for t in ts])
        return X.astype(np.float32), y_all[np.array(ts)].astype(np.float32), np.array(ts)

    X_tr, y_tr, _    = windows(CONTEXT - 1, TRAIN_END - ROLL_W)
    X_va, y_va, _    = windows(VAL_START,   VAL_END - ROLL_W)
    X_te, y_te, t_te = windows(TEST_START,  M.shape[1] - ROLL_W - 1)

    return (X_tr, y_tr), (X_va, y_va), (X_te, y_te, t_te)