import numpy as np
from kalman.ekf_core.state import State
from kalman.ekf_core.ekf import EKF
from kalman.dynamics.two_body import TwoBodyDynamics
from kalman.measurements.range_bearing import RangeBearing


def test_state_creation():
    x = np.array([1.0, 2.0, 3.0])
    P = np.eye(3)
    s = State(x, P, timestamp=10.0)
    assert np.allclose(s.x, x)
    assert np.allclose(s.P, P)
    assert s.timestamp == 10.0


def test_state_copy():
    s = State(np.array([1.0, 2.0, 3.0]), np.eye(3))
    s2 = s.copy()
    s2.x[0] = 99.0
    assert s.x[0] == 1.0


def test_ekf_predict():
    dynamics = TwoBodyDynamics()
    r0 = np.array([7000e3, 0.0, 0.0])
    v0 = np.array([0.0, 7546.0, 0.0])
    x0 = np.concatenate([r0, v0])
    P0 = np.eye(6) * 1e6
    state = State(x0, P0)
    ekf = EKF(dynamics, state)
    ekf.predict(10.0)
    assert not np.allclose(ekf.state.x, x0)
    assert ekf.state.timestamp == 10.0


def test_ekf_update():
    dynamics = TwoBodyDynamics()
    r0 = np.array([7000e3, 100.0, 50.0])
    v0 = np.array([0.0, 7546.0, 10.0])
    x0 = np.concatenate([r0, v0])
    P0 = np.eye(6) * 1e6
    state = State(x0, P0)
    ekf = EKF(dynamics, state)
    meas = RangeBearing()
    z = meas.h(x0)
    R = np.diag([1e3, np.deg2rad(0.1) ** 2, np.deg2rad(0.1) ** 2])
    ekf.update(meas, z, R)
    assert np.allclose(ekf.state.x, x0, atol=1e-6)


def test_predict_consistency():
    np.random.seed(42)
    dynamics = TwoBodyDynamics()
    r0 = np.array([7000e3, 0.0, 0.0])
    v0 = np.array([0.0, 7546.0, 0.0])
    x0 = np.concatenate([r0, v0])
    P0 = np.diag([1e6, 1e6, 1e6, 1e3, 1e3, 1e3])
    state = State(x0, P0)
    ekf = EKF(dynamics, state)
    x_true = dynamics.propagate(x0, 60.0)
    ekf.predict(60.0)
    nees = ekf.nees(x_true)
    assert nees < 10.0
