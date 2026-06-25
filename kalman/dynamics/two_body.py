import numpy as np
from kalman.dynamics.base import DynamicsModel
from kalman.utils.constants import MU_EARTH


class TwoBodyDynamics(DynamicsModel):
    def __init__(self, mu: float = MU_EARTH, q_scale: float = 1e-8):
        self._mu = mu
        self._q_scale = q_scale

    @property
    def state_dim(self) -> int:
        return 6

    def f(self, x: np.ndarray) -> np.ndarray:
        r = x[:3]
        v = x[3:6]
        r_norm = np.linalg.norm(r)
        acc = -self._mu / r_norm ** 3 * r
        return np.concatenate([v, acc])

    def propagate(self, x: np.ndarray, dt: float) -> np.ndarray:
        return rk4(self.f, x, dt)

    def stm_analytic(self, x: np.ndarray, dt: float) -> np.ndarray:
        r = x[:3]
        v = x[3:6]
        r_norm = np.linalg.norm(r)
        rdotv = np.dot(r, v)
        Phi = np.eye(6)
        Phi[0:3, 3:6] = np.eye(3) * dt
        mu_r3 = self._mu / r_norm ** 3
        mu_r5 = 3 * self._mu / r_norm ** 5
        G = mu_r5 * np.outer(r, r) - mu_r3 * np.eye(3)
        Phi[3:6, 0:3] = -G * dt
        return Phi

    def jacobian(self, x: np.ndarray, dt: float) -> np.ndarray:
        return self.stm_analytic(x, dt)

    def Q(self, dt: float) -> np.ndarray:
        Q3 = self._q_scale * np.eye(3)
        Q = np.zeros((6, 6))
        Q[0:3, 0:3] = Q3 * (dt ** 3 / 3)
        Q[0:3, 3:6] = Q3 * (dt ** 2 / 2)
        Q[3:6, 0:3] = Q3 * (dt ** 2 / 2)
        Q[3:6, 3:6] = Q3 * dt
        return Q


def rk4(f, x: np.ndarray, dt: float) -> np.ndarray:
    k1 = f(x)
    k2 = f(x + 0.5 * dt * k1)
    k3 = f(x + 0.5 * dt * k2)
    k4 = f(x + dt * k3)
    return x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
