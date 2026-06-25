import numpy as np
from kalman.dynamics.base import DynamicsModel
from kalman.utils.constants import MU_EARTH, J2, R_EARTH
from kalman.dynamics.two_body import rk4


class J2PerturbedDynamics(DynamicsModel):
    def __init__(self, mu: float = MU_EARTH, j2: float = J2,
                 r_e: float = R_EARTH, cd: float = 0.0,
                 area_mass_ratio: float = 0.0, q_scale: float = 1e-8):
        self._mu = mu
        self._j2 = j2
        self._r_e = r_e
        self._cd = cd
        self._amr = area_mass_ratio
        self._q_scale = q_scale
        self._num_states = 6

    @property
    def state_dim(self) -> int:
        return self._num_states

    def f(self, x: np.ndarray) -> np.ndarray:
        r = x[:3]
        v = x[3:6]
        r_norm = np.linalg.norm(r)
        acc = -self._mu / r_norm ** 3 * r
        z2 = r[2] ** 2
        r2 = r_norm ** 2
        factor = 1.5 * self._j2 * self._mu * self._r_e ** 2 / r_norm ** 5
        j2_acc = factor * np.array([
            r[0] * (5 * z2 / r2 - 1),
            r[1] * (5 * z2 / r2 - 1),
            r[2] * (5 * z2 / r2 - 3)
        ])
        acc = acc + j2_acc
        if self._cd != 0.0 and self._amr != 0.0:
            h = np.linalg.norm(np.cross(r, v))
            r_p = self._r_e + 0.0
            rho = 1.0
            drag = -0.5 * self._cd * self._amr * rho * np.linalg.norm(v) * v
            acc = acc + drag
        return np.concatenate([v, acc])

    def propagate(self, x: np.ndarray, dt: float) -> np.ndarray:
        return rk4(self.f, x, dt)

    def jacobian(self, x: np.ndarray, dt: float) -> np.ndarray:
        return self.stm_numeric(x, dt)

    def stm_numeric(self, x: np.ndarray, dt: float, eps: float = 1e-6) -> np.ndarray:
        n = self.state_dim
        Phi = np.eye(n)
        xf = self.propagate(x, dt)
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
