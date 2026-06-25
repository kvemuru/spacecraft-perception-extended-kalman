import numpy as np
import scipy.linalg
from kalman.ekf_core.interfaces import DynamicsModel, MeasurementModel
from kalman.ekf_core.state import State


class EKF:
    def __init__(self, dynamics: DynamicsModel, state: State):
        self.dynamics = dynamics
        self.state = state

    def predict(self, dt: float) -> None:
        n = self.dynamics.state_dim
        Phi = self.dynamics.jacobian(self.state.x, dt)
        self.state.x = self.dynamics.propagate(self.state.x, dt)
        Q = self.dynamics.Q(dt)
        self.state.P = Phi @ self.state.P @ Phi.T + Q
        self.state.timestamp += dt

    def update(self, meas: MeasurementModel, z: np.ndarray, R: np.ndarray) -> None:
        n = self.dynamics.state_dim
        H = meas.jacobian(self.state.x)
        y = z - meas.h(self.state.x)

        S = H @ self.state.P @ H.T + R
        K = scipy.linalg.solve(S, H @ self.state.P, assume_a='pos').T
        self.state.x = self.state.x + K @ y
        I_KH = np.eye(n) - K @ H
        self.state.P = (I_KH @ self.state.P @ I_KH.T
                        + K @ R @ K.T)

    def iterate_update(self, meas: MeasurementModel, z: np.ndarray,
                       R: np.ndarray, max_iter: int = 10,
                       tol: float = 1e-8) -> None:
        for _ in range(max_iter):
            x0 = self.state.x.copy()
            H = meas.jacobian(self.state.x)
            y = z - meas.h(self.state.x)
            S = H @ self.state.P @ H.T + R
            K = scipy.linalg.solve(S, H @ self.state.P, assume_a='pos').T
            self.state.x = x0 + K @ y
            I_KH = np.eye(self.dynamics.state_dim) - K @ H
            self.state.P = (I_KH @ self.state.P @ I_KH.T
                            + K @ R @ K.T)
            if np.linalg.norm(self.state.x - x0) < tol:
                break

    def predict_to(self, target_time: float) -> None:
        dt = target_time - self.state.timestamp
        if dt > 0:
            self.predict(dt)

    def nees(self, x_true: np.ndarray) -> float:
        d = self.state.x - x_true
        return d @ scipy.linalg.solve(self.state.P, d, assume_a='pos')

    def nis(self, z: np.ndarray, meas: MeasurementModel, R: np.ndarray) -> float:
        y = z - meas.h(self.state.x)
        S = meas.jacobian(self.state.x) @ self.state.P @ meas.jacobian(self.state.x).T + R
        return y @ scipy.linalg.solve(S, y, assume_a='pos')
