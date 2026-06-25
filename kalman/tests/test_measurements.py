import numpy as np
import pytest
from kalman.measurements.range_bearing import RangeBearing
from kalman.measurements.range_rate import RangeRate
from kalman.measurements.angles_only import AnglesOnly
from kalman.measurements.gps import GPSPosition, GPSPositionVelocity
from kalman.measurements.relative_pose import RelativePosition, RelativePose
from kalman.measurements.range_bearing_from_station import RangeBearingFromStation


def test_range_bearing_h():
    meas = RangeBearing()
    r = np.array([1000.0, 1000.0, 500.0])
    v = np.array([1.0, 0.0, 0.0])
    x = np.concatenate([r, v])
    z = meas.h(x)
    assert len(z) == 3
    assert z[0] == pytest.approx(np.linalg.norm(r))
    assert z[1] == pytest.approx(np.pi / 4)


def test_range_bearing_jacobian():
    meas = RangeBearing()
    r = np.array([1000.0, 1000.0, 500.0])
    v = np.array([1.0, 0.0, 0.0])
    x = np.concatenate([r, v])
    H = meas.jacobian(x)
    H_num = _numeric_jac(meas, x, eps=1e-4)
    assert np.allclose(H, H_num, atol=1e-6)


def test_range_rate():
    meas = RangeRate()
    r = np.array([1000.0, 0.0, 0.0])
    v = np.array([0.0, 10.0, 0.0])
    x = np.concatenate([r, v])
    z = meas.h(x)
    assert len(z) == 1
    assert z[0] == pytest.approx(0.0)


def test_angles_only_h():
    meas = AnglesOnly()
    r = np.array([1000.0, 1000.0, 500.0])
    v = np.array([1.0, 0.0, 0.0])
    x = np.concatenate([r, v])
    z = meas.h(x)
    assert len(z) == 2
    assert z[0] == pytest.approx(np.pi / 4)


def test_gps_position():
    jd = 2460000.0
    meas_gps = GPSPosition(jd)
    meas_pv = GPSPositionVelocity(jd)
    r = np.array([7000e3, 0.0, 0.0])
    v = np.array([0.0, 7546.0, 0.0])
    x = np.concatenate([r, v])
    z_pos = meas_gps.h(x)
    z_pv = meas_pv.h(x)
    assert len(z_pos) == 3
    assert len(z_pv) == 6


def test_gps_jacobian():
    jd = 2460000.0
    meas = GPSPosition(jd)
    r = np.array([7000e3, 100.0, 50.0])
    v = np.array([0.0, 7546.0, 10.0])
    x = np.concatenate([r, v])
    H = meas.jacobian(x)
    H_num = _numeric_jac(meas, x, eps=1e0)
    assert np.allclose(H, H_num, atol=1e-4)


def test_relative_position():
    meas = RelativePosition()
    x = np.array([100.0, 50.0, 20.0, 1.0, 0.5, 0.2])
    z = meas.h(x)
    assert np.allclose(z, x[:3])


def test_relative_pose():
    meas = RelativePose()
    x = np.array([100.0, 50.0, 20.0, 1.0, 0.5, 0.2])
    z = meas.h(x)
    assert np.allclose(z, x)


def test_range_bearing_from_station():
    r_station = np.array([6378e3, 0.0, 0.0])
    meas = RangeBearingFromStation(r_station)
    r = np.array([7000e3, 100.0, 50.0])
    v = np.array([0.0, 7546.0, 10.0])
    x = np.concatenate([r, v])
    z = meas.h(x)
    r_rel = r - r_station
    expected_rho = np.linalg.norm(r_rel)
    assert abs(z[0] - expected_rho) < 1.0
    assert len(z) == 3


def test_range_bearing_from_station_jacobian():
    r_station = np.array([6378e3, 0.0, 0.0])
    meas = RangeBearingFromStation(r_station)
    r = np.array([7000e3, 100.0, 50.0])
    v = np.array([0.0, 7546.0, 10.0])
    x = np.concatenate([r, v])
    H = meas.jacobian(x)
    H_num = _numeric_jac(meas, x, eps=1e0)
    assert np.allclose(H, H_num, atol=1e-4)


def _numeric_jac(meas, x, eps=1e-8):
    m = len(meas.h(x))
    n = len(x)
    H = np.zeros((m, n))
    for i in range(n):
        dx = np.zeros(n)
        dx[i] = eps
        H[:, i] = (meas.h(x + dx) - meas.h(x - dx)) / (2 * eps)
    return H
