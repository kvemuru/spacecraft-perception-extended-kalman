import numpy as np
from kalman.measurements.base import MeasurementModel


class RangeBearingFromStation(MeasurementModel):
    def __init__(self, r_station: np.ndarray):
        self._r_station = r_station.copy()

    def h(self, x: np.ndarray) -> np.ndarray:
        r_rel = x[:3] - self._r_station
        rho = np.linalg.norm(r_rel)
        az = np.arctan2(r_rel[1], r_rel[0])
        el = np.arcsin(r_rel[2] / rho)
        return np.array([rho, az, el])

    def jacobian(self, x: np.ndarray) -> np.ndarray:
        r_rel = x[:3] - self._r_station
        rho = np.linalg.norm(r_rel)
        rho2 = rho ** 2
        H = np.zeros((3, 6))
        H[0, 0:3] = r_rel / rho
        denom = r_rel[0] ** 2 + r_rel[1] ** 2
        H[1, 0] = -r_rel[1] / denom
        H[1, 1] = r_rel[0] / denom
        xy_norm = np.sqrt(denom)
        if xy_norm > 1e-12:
            H[2, 0] = -r_rel[0] * r_rel[2] / (rho2 * xy_norm)
            H[2, 1] = -r_rel[1] * r_rel[2] / (rho2 * xy_norm)
            H[2, 2] = xy_norm / rho2
        return H
