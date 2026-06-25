import numpy as np
from kalman.measurements.base import MeasurementModel


class RangeRate(MeasurementModel):
    def h(self, x: np.ndarray) -> np.ndarray:
        r = x[:3]
        v = x[3:6]
        rho = np.linalg.norm(r)
        return np.array([np.dot(r, v) / rho])

    def jacobian(self, x: np.ndarray) -> np.ndarray:
        r = x[:3]
        v = x[3:6]
        rho = np.linalg.norm(r)
        rho3 = rho ** 3
        H = np.zeros((1, 6))
        H[0, 0:3] = v / rho - np.dot(r, v) * r / rho3
        H[0, 3:6] = r / rho
        return H
