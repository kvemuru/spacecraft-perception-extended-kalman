import numpy as np
from kalman.utils.constants import OMEGA_EARTH


def skew(v: np.ndarray) -> np.ndarray:
    return np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0]
    ])


def r3(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([
        [c, s, 0],
        [-s, c, 0],
        [0, 0, 1]
    ])


def eci_to_ecef(r_eci: np.ndarray, gmst: float) -> np.ndarray:
    return r3(gmst) @ r_eci


def ecef_to_eci(r_ecef: np.ndarray, gmst: float) -> np.ndarray:
    return r3(gmst).T @ r_ecef


def eci_to_lvlh(r_eci: np.ndarray, v_eci: np.ndarray,
                 r_target: np.ndarray, v_target: np.ndarray):
    h = np.cross(r_target, v_target)
    x_hat = r_target / np.linalg.norm(r_target)
    z_hat = h / np.linalg.norm(h)
    y_hat = np.cross(z_hat, x_hat)
    C_eci_lvlh = np.column_stack([x_hat, y_hat, z_hat])
    dr = r_eci - r_target
    dv = v_eci - v_target
    r_lvlh = C_eci_lvlh.T @ dr
    v_lvlh = C_eci_lvlh.T @ dv
    return r_lvlh, v_lvlh, C_eci_lvlh


def lvlh_to_eci(r_lvlh: np.ndarray, v_lvlh: np.ndarray,
                 r_target: np.ndarray, v_target: np.ndarray):
    h = np.cross(r_target, v_target)
    x_hat = r_target / np.linalg.norm(r_target)
    z_hat = h / np.linalg.norm(h)
    y_hat = np.cross(z_hat, x_hat)
    C_eci_lvlh = np.column_stack([x_hat, y_hat, z_hat])
    dr = C_eci_lvlh @ r_lvlh
    dv = C_eci_lvlh @ v_lvlh
    return dr + r_target, dv + v_target


def gmst_from_jd(jd: float) -> float:
    t = (jd - 2451545.0) / 36525.0
    theta = (280.46061837 + 360.98564736629 * (jd - 2451545.0)
             + 0.000387933 * t ** 2 - t ** 3 / 38710000.0)
    return np.deg2rad(theta % 360.0)
