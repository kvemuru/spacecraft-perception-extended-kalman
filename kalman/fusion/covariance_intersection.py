import numpy as np
import scipy.linalg
from kalman.ekf_core.state import State


def fuse_pair(a: State, b: State, alpha: float | None = None) -> State:
    Pa_inv = np.linalg.inv(a.P)
    Pb_inv = np.linalg.inv(b.P)
    ya = Pa_inv @ a.x
    yb = Pb_inv @ b.x
    if alpha is None:
        alpha = _optimal_alpha(a.P, b.P)
    Y = alpha * Pa_inv + (1 - alpha) * Pb_inv
    y = alpha * ya + (1 - alpha) * yb
    P = np.linalg.inv(Y)
    x = P @ y
    ts = max(a.timestamp, b.timestamp)
    return State(x, P, ts)


def fuse_many(states: list[State]) -> State:
    if len(states) == 0:
        raise ValueError("No states to fuse")
    if len(states) == 1:
        return states[0].copy()
    result = states[0]
    for s in states[1:]:
        result = fuse_pair(result, s)
    return result


def _optimal_alpha(Pa: np.ndarray, Pb: np.ndarray) -> float:
    n = Pa.shape[0]
    def det_log_alpha(a: float) -> float:
        a = np.clip(a, 1e-6, 1 - 1e-6)
        Y = a * np.linalg.inv(Pa) + (1 - a) * np.linalg.inv(Pb)
        _, logdet = np.linalg.slogdet(Y)
        return -logdet
    lo, hi = 1e-6, 1 - 1e-6
    phi = (np.sqrt(5) - 1) / 2
    for _ in range(30):
        d = hi - lo
        m1 = lo + phi * d
        m2 = hi - phi * d
        f1 = det_log_alpha(m1)
        f2 = det_log_alpha(m2)
        if f1 < f2:
            hi = m2
        else:
            lo = m1
    return (lo + hi) / 2
