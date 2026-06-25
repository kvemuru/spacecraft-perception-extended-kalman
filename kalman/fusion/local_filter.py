"""Per-sensor EKF wrapper for decentralized fusion."""

import numpy as np
from kalman.ekf_core.ekf import EKF
from kalman.ekf_core.state import State
from kalman.ekf_core.interfaces import DynamicsModel, MeasurementModel


class LocalFilter:
    def __init__(self, dynamics: DynamicsModel, state: State,
                 measurement_model: MeasurementModel,
                 R: np.ndarray, sensor_id: str):
        self.ekf = EKF(dynamics, state.copy())
        self.meas_model = measurement_model
        self.R = R
        self.sensor_id = sensor_id

    def predict(self, dt: float) -> None:
        self.ekf.predict(dt)

    def update(self, z: np.ndarray) -> None:
        self.ekf.update(self.meas_model, z, self.R)

    def predict_and_update(self, dt: float, z: np.ndarray) -> None:
        self.predict(dt)
        self.update(z)

    @property
    def state(self) -> State:
        return self.ekf.state

    @property
    def information_vector(self) -> np.ndarray:
        P = self.state.P
        return np.linalg.solve(P, self.state.x)

    @property
    def information_matrix(self) -> np.ndarray:
        return np.linalg.inv(self.state.P)
