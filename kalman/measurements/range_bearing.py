import numpy as np
from kalman.measurements.base import MeasurementModel


class RangeBearing(MeasurementModel):
    def h(self, x: np.ndarray) -> np.ndarray:
        r = x[:3]
        rho = np.linalg.norm(r)
        az = np.arctan2(r[1], r[0])
        el = np.arcsin(r[2] / rho)
        return np.array([rho, az, el])

    def jacobian(self, x: np.ndarray) -> np.ndarray:
        r = x[:3]
        rho = np.linalg.norm(r)
        rho2 = rho ** 2
        H = np.zeros((3, 6))
        H[0, 0:3] = r / rho
        H[1, 0] = -r[1] / (r[0] ** 2 + r[1] ** 2)
        H[1, 1] = r[0] / (r[0] ** 2 + r[1] ** 2)
        xy_norm = np.sqrt(r[0] ** 2 + r[1] ** 2)
        if xy_norm > 1e-12:
            H[2, 0] = -r[0] * r[2] / (rho2 * xy_norm)
            H[2, 1] = -r[1] * r[2] / (rho2 * xy_norm)
            H[2, 2] = xy_norm / rho2
        return H
