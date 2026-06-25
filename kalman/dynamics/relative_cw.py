import numpy as np
from kalman.dynamics.base import DynamicsModel


class ClohessyWiltshireDynamics(DynamicsModel):
    def __init__(self, n: float, q_scale: float = 1e-8):
        self._n = n
        self._q_scale = q_scale

    @property
    def state_dim(self) -> int:
        return 6

    def A(self) -> np.ndarray:
        n = self._n
        return np.array([
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1],
            [3 * n ** 2, 0, 0, 0, 2 * n, 0],
            [0, 0, 0, -2 * n, 0, 0],
            [0, 0, -n ** 2, 0, 0, 0]
        ])

    def propagate(self, x: np.ndarray, dt: float) -> np.ndarray:
        Phi = self.jacobian(x, dt)
        return Phi @ x

    def jacobian(self, x: np.ndarray, dt: float) -> np.ndarray:
        n = self._n
        nt = n * dt
        s, c = np.sin(nt), np.cos(nt)
        Phi = np.zeros((6, 6))
        Phi[0, 0] = 4 - 3 * c
        Phi[0, 1] = 0
        Phi[0, 2] = 0
        Phi[0, 3] = s / n
        Phi[0, 4] = 2 * (1 - c) / n
        Phi[0, 5] = 0
        Phi[1, 0] = 6 * (s - nt)
        Phi[1, 1] = 1
        Phi[1, 2] = 0
        Phi[1, 3] = -2 * (1 - c) / n
        Phi[1, 4] = (4 * s - 3 * nt) / n
        Phi[1, 5] = 0
        Phi[2, 0] = 0
        Phi[2, 1] = 0
        Phi[2, 2] = c
        Phi[2, 3] = 0
        Phi[2, 4] = 0
        Phi[2, 5] = s / n
        Phi[3, 0] = 3 * n * s
        Phi[3, 1] = 0
        Phi[3, 2] = 0
        Phi[3, 3] = c
        Phi[3, 4] = 2 * s
        Phi[3, 5] = 0
        Phi[4, 0] = 6 * n * (c - 1)
        Phi[4, 1] = 0
        Phi[4, 2] = 0
        Phi[4, 3] = -2 * s
        Phi[4, 4] = 4 * c - 3
        Phi[4, 5] = 0
        Phi[5, 0] = 0
        Phi[5, 1] = 0
        Phi[5, 2] = -n * s
        Phi[5, 3] = 0
        Phi[5, 4] = 0
        Phi[5, 5] = c
        return Phi

    def Q(self, dt: float) -> np.ndarray:
        Q3 = self._q_scale * np.eye(3)
        Q = np.zeros((6, 6))
        Q[0:3, 0:3] = Q3 * (dt ** 3 / 3)
        Q[0:3, 3:6] = Q3 * (dt ** 2 / 2)
        Q[3:6, 0:3] = Q3 * (dt ** 2 / 2)
        Q[3:6, 3:6] = Q3 * dt
        return Q
