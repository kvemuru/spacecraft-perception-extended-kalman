"""Covariance intersection fusion: pairwise, sequential, and batch."""

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


def fuse_batch(states: list[State]) -> State:
    if len(states) == 0:
        raise ValueError("No states to fuse")
    if len(states) == 1:
        return states[0].copy()
    if len(states) == 2:
        return fuse_pair(states[0], states[1])
    n = states[0].x.shape[0]
    m = len(states)
    invs = [np.linalg.inv(s.P) for s in states]
    ys = [inv @ s.x for inv, s in zip(invs, states)]
    w = np.full(m, 1.0 / m)
    for _ in range(50):
        w_old = w.copy()
        for i in range(m):
            fixed = sum(w[j] * invs[j] for j in range(m) if j != i)
            fixed_y = sum(w[j] * ys[j] for j in range(m) if j != i)
            def trace_alpha(a: float) -> float:
                a = np.clip(a, 1e-6, 1 - 1e-6)
                Y_total = a * invs[i] + fixed
                P_total = np.linalg.inv(Y_total)
                return np.trace(P_total)
            lo, hi = 1e-6, 1 - 1e-6
            phi = (np.sqrt(5) - 1) / 2
            for _ in range(30):
                d = hi - lo
                m1 = lo + phi * d
                m2 = hi - phi * d
                f1 = trace_alpha(m1 * (1 - sum(w[j] for j in range(m) if j != i)))
                f2 = trace_alpha(m2 * (1 - sum(w[j] for j in range(m) if j != i)))
                if f1 < f2:
                    hi = m2
                else:
                    lo = m1
            alpha_opt = (lo + hi) / 2
            w[i] = alpha_opt * (1 - sum(w[j] for j in range(m) if j != i))
        if np.max(np.abs(w - w_old)) < 1e-6:
            break
    Y_total = sum(w[j] * invs[j] for j in range(m))
    y_total = sum(w[j] * ys[j] for j in range(m))
    P = np.linalg.inv(Y_total)
    x = P @ y_total
    ts = max(s.timestamp for s in states)
    return State(x, P, ts)


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
