import numpy as np
from kalman.measurements.base import MeasurementModel


class RelativePosition(MeasurementModel):
    def h(self, x: np.ndarray) -> np.ndarray:
        return x[:3]

    def jacobian(self, x: np.ndarray) -> np.ndarray:
        H = np.zeros((3, 6))
        H[0:3, 0:3] = np.eye(3)
        return H


class RelativePose(MeasurementModel):
    def h(self, x: np.ndarray) -> np.ndarray:
        return x[:6]

    def jacobian(self, x: np.ndarray) -> np.ndarray:
        return np.eye(6)
