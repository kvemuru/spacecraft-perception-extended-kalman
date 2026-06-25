"""Optical camera azimuth/elevation measurement model."""

import numpy as np
from kalman.measurements.base import MeasurementModel


class AnglesOnly(MeasurementModel):
    def h(self, x: np.ndarray) -> np.ndarray:
        r = x[:3]
        az = np.arctan2(r[1], r[0])
        el = np.arcsin(r[2] / np.linalg.norm(r))
        return np.array([az, el])

    def jacobian(self, x: np.ndarray) -> np.ndarray:
        r = x[:3]
        rho = np.linalg.norm(r)
        rho2 = rho ** 2
        H = np.zeros((2, 6))
        denom = r[0] ** 2 + r[1] ** 2
        H[0, 0] = -r[1] / denom
        H[0, 1] = r[0] / denom
        xy_norm = np.sqrt(denom)
        if xy_norm > 1e-12:
            H[1, 0] = -r[0] * r[2] / (rho2 * xy_norm)
            H[1, 1] = -r[1] * r[2] / (rho2 * xy_norm)
            H[1, 2] = xy_norm / rho2
        return H
