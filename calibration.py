import numpy as np
from sklearn.metrics import log_loss


def softmax(logits: np.ndarray, axis: int = 1) -> np.ndarray:
    z = logits - logits.max(axis=axis, keepdims=True)
    exps = np.exp(z)
    return exps / exps.sum(axis=axis, keepdims=True)


def find_temperature(logits_val: np.ndarray, y_val: np.ndarray,
                     tmin: float, tmax: float, steps: int) -> float:
    best_t = 1.0
    best_loss = np.inf
    grid = np.linspace(tmin, tmax, steps)
    for T in grid:
        probs = softmax(logits_val / T, axis=1)
        loss = log_loss(y_val, probs, labels=np.arange(probs.shape[1]))
        if loss < best_loss:
            best_loss = loss
            best_t = T
    return best_t
