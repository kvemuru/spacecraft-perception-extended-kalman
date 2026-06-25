import numpy as np
from kalman.utils.constants import MU_EARTH, J2, R_EARTH, OMEGA_EARTH
from kalman.utils.time_ import datetime_to_jd, jd_to_datetime, jd_diff_seconds
from kalman.utils.coordinates import (eci_to_ecef, ecef_to_eci, eci_to_lvlh,
                                       lvlh_to_eci, gmst_from_jd, r3)
from datetime import datetime, timezone


def test_constants():
    assert MU_EARTH > 0
    assert J2 > 0
    assert R_EARTH > 0
    assert OMEGA_EARTH > 0


def test_time_roundtrip():
    dt = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    jd = datetime_to_jd(dt)
    dt2 = jd_to_datetime(jd)
    assert abs((dt - dt2).total_seconds()) < 1e-3


def test_jd_diff():
    jd1 = 2460000.0
    jd2 = 2460000.5
    assert abs(jd_diff_seconds(jd1, jd2) - (-43200)) < 1e-6


def test_eci_ecef_roundtrip():
    r_eci = np.array([7000e3, 0.0, 0.0])
    gmst = np.deg2rad(45.0)
    r_ecef = eci_to_ecef(r_eci, gmst)
    r_eci2 = ecef_to_eci(r_ecef, gmst)
    assert np.allclose(r_eci, r_eci2)


def test_r3_orthogonal():
    R = r3(0.5)
    assert np.allclose(R @ R.T, np.eye(3))
    assert np.abs(np.linalg.det(R) - 1) < 1e-15


def test_lvlh_roundtrip():
    r_t = np.array([7000e3, 0.0, 0.0])
    v_t = np.array([0.0, 7500.0, 0.0])
    r_c = r_t + np.array([100.0, 50.0, 20.0])
    v_c = v_t + np.array([1.0, 0.5, 0.2])
    r_rel, v_rel, _ = eci_to_lvlh(r_c, v_c, r_t, v_t)
    r_c2, v_c2 = lvlh_to_eci(r_rel, v_rel, r_t, v_t)
    assert np.allclose(r_c, r_c2)
    assert np.allclose(v_c, v_c2)


def test_gmst_from_jd():
    theta = gmst_from_jd(2451545.0)
    assert abs(theta - 4.894961) < 1e-3
