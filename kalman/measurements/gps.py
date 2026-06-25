import numpy as np
from kalman.measurements.base import MeasurementModel
from kalman.utils.coordinates import eci_to_ecef, gmst_from_jd


class GPSPosition(MeasurementModel):
    def __init__(self, jd: float):
        self._jd = jd

    def h(self, x: np.ndarray) -> np.ndarray:
        r_eci = x[:3]
        gmst = gmst_from_jd(self._jd)
        r_ecef = eci_to_ecef(r_eci, gmst)
        return r_ecef

    def jacobian(self, x: np.ndarray) -> np.ndarray:
        gmst = gmst_from_jd(self._jd)
        c, s = np.cos(gmst), np.sin(gmst)
        C = np.array([
            [c, s, 0],
            [-s, c, 0],
            [0, 0, 1]
        ])
        H = np.zeros((3, 6))
        H[0:3, 0:3] = C
        return H


class GPSPositionVelocity(MeasurementModel):
    def __init__(self, jd: float):
        self._jd = jd

    @property
    def measurement_dim(self) -> int:
        return 6

    def h(self, x: np.ndarray) -> np.ndarray:
        r_eci = x[:3]
        v_eci = x[3:6]
        gmst = gmst_from_jd(self._jd)
        gmst_dot = 7.2921150e-5
        c, s = np.cos(gmst), np.sin(gmst)
        C = np.array([
            [c, s, 0],
            [-s, c, 0],
            [0, 0, 1]
        ])
        Cdot = gmst_dot * np.array([
            [-s, c, 0],
            [-c, -s, 0],
            [0, 0, 0]
        ])
        r_ecef = C @ r_eci
        v_ecef = C @ v_eci + Cdot @ r_eci
        return np.concatenate([r_ecef, v_ecef])

    def jacobian(self, x: np.ndarray) -> np.ndarray:
        gmst = gmst_from_jd(self._jd)
        gmst_dot = 7.2921150e-5
        c, s = np.cos(gmst), np.sin(gmst)
        C = np.array([
            [c, s, 0],
            [-s, c, 0],
            [0, 0, 1]
        ])
        Cdot = gmst_dot * np.array([
            [-s, c, 0],
            [-c, -s, 0],
            [0, 0, 0]
        ])
        H = np.zeros((6, 6))
        H[0:3, 0:3] = C
        H[0:3, 3:6] = np.zeros((3, 3))
        H[3:6, 0:3] = Cdot
        H[3:6, 3:6] = C
        return H
