import numpy as np

def mse_loss(y_hat, y):
    y = y.reshape(-1, 1).astype(np.float32)
    d = y_hat - y
    return float(np.mean(d ** 2)), (2.0 * d) / len(y)

def qlike(actual, pred):          # both log-RV
    u = actual - pred
    return float(np.mean(np.exp(u) - u - 1.0))
