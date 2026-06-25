"""Hill-Clohessy-Wiltshire relative motion for elliptical orbits (numerical STM)."""

import numpy as np
from kalman.dynamics.base import DynamicsModel
from kalman.dynamics.two_body import rk4


class HillClohessyWiltshireDynamics(DynamicsModel):
    def __init__(self, mu: float, q_scale: float = 1e-8):
        self._mu = mu
        self._q_scale = q_scale

    @property
    def state_dim(self) -> int:
        return 6

    def set_reference(self, r_ref: np.ndarray, v_ref: np.ndarray):
        self._r_ref = r_ref.copy()
        self._v_ref = v_ref.copy()

    def f(self, x_rel: np.ndarray) -> np.ndarray:
        r_ref = self._r_ref
        v_ref = self._v_ref
        r_ref_norm = np.linalg.norm(r_ref)
        r_chaser = r_ref + x_rel[:3]
        r_chaser_norm = np.linalg.norm(r_chaser)
        a_ref = -self._mu / r_ref_norm ** 3 * r_ref
        a_chaser = -self._mu / r_chaser_norm ** 3 * r_chaser
        a_rel = a_chaser - a_ref
        return np.concatenate([x_rel[3:6], a_rel])

    def propagate(self, x: np.ndarray, dt: float) -> np.ndarray:
        return rk4(self.f, x, dt)

    def jacobian(self, x: np.ndarray, dt: float) -> np.ndarray:
        return self.stm_numeric(x, dt)

    def stm_numeric(self, x: np.ndarray, dt: float, eps: float = 1e-6) -> np.ndarray:
        n = self.state_dim
        Phi = np.eye(n)
        for i in range(n):
            dx = np.zeros(n)
            dx[i] = eps
            xp = self.propagate(x + dx, dt)
            xm = self.propagate(x - dx, dt)
            Phi[:, i] = (xp - xm) / (2 * eps)
        return Phi

    def Q(self, dt: float) -> np.ndarray:
        Q3 = self._q_scale * np.eye(3)
        Q = np.zeros((6, 6))
        Q[0:3, 0:3] = Q3 * (dt ** 3 / 3)
        Q[0:3, 3:6] = Q3 * (dt ** 2 / 2)
        Q[3:6, 0:3] = Q3 * (dt ** 2 / 2)
        Q[3:6, 3:6] = Q3 * dt
        return Q
