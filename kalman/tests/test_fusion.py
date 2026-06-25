import numpy as np
from kalman.fusion.local_filter import LocalFilter
from kalman.fusion.covariance_intersection import fuse_pair, fuse_many
from kalman.fusion.decentralized import DecentralizedFusion
from kalman.ekf_core.state import State
from kalman.dynamics.two_body import TwoBodyDynamics
from kalman.measurements.range_bearing import RangeBearing


def test_local_filter():
    dyn = TwoBodyDynamics()
    x0 = np.array([7000e3, 0.0, 0.0, 0.0, 7546.0, 0.0])
    P0 = np.eye(6) * 1e6
    state = State(x0, P0)
    meas = RangeBearing()
    R = np.diag([1e3, np.deg2rad(0.1) ** 2, np.deg2rad(0.1) ** 2])
    lf = LocalFilter(dyn, state, meas, R, "test_sensor")
    assert lf.sensor_id == "test_sensor"
    lf.predict(10.0)
    z = meas.h(x0)
    lf.update(z)
    assert lf.state.timestamp == 10.0


def test_information_form():
    dyn = TwoBodyDynamics()
    x0 = np.array([7000e3, 0.0, 0.0, 0.0, 7546.0, 0.0])
    P0 = np.eye(6) * 1e6
    state = State(x0, P0)
    meas = RangeBearing()
    R = np.diag([1e3, np.deg2rad(0.1) ** 2, np.deg2rad(0.1) ** 2])
    lf = LocalFilter(dyn, state, meas, R, "test")
    y = lf.information_vector
    Y = lf.information_matrix
    x_recovered = np.linalg.solve(Y, y)
    assert np.allclose(x_recovered, lf.state.x)


def test_fuse_pair():
    x1 = np.array([100.0, 50.0, 20.0])
    P1 = np.diag([10.0, 10.0, 10.0])
    x2 = np.array([102.0, 48.0, 21.0])
    P2 = np.diag([100.0, 100.0, 100.0])
    s1 = State(x1, P1)
    s2 = State(x2, P2)
    fused = fuse_pair(s1, s2)
    assert np.allclose(fused.P, fused.P.T)
    assert fused.x.shape == (3,)


def test_fuse_many():
    states = [
        State(np.array([100.0]), np.diag([10.0])),
        State(np.array([102.0]), np.diag([10.0])),
        State(np.array([98.0]), np.diag([10.0])),
    ]
    fused = fuse_many(states)
    assert abs(fused.x[0] - 100.0) < 2.0


def test_decentralized_fusion():
    dyn = TwoBodyDynamics()
    x0 = np.array([7000e3, 0.0, 0.0, 0.0, 7546.0, 0.0])
    P0 = np.eye(6) * 1e6

    state1 = State(x0.copy(), P0.copy())
    state2 = State(x0.copy(), P0.copy())
    meas = RangeBearing()
    R = np.diag([1e3, np.deg2rad(0.1) ** 2, np.deg2rad(0.1) ** 2])
    lf1 = LocalFilter(dyn, state1, meas, R, "sensor_a")
    lf2 = LocalFilter(dyn, state2, meas, R, "sensor_b")

    df = DecentralizedFusion(dyn, 6)
    df.add_filter(lf1)
    df.add_filter(lf2)
    df.predict_all(10.0)

    z = meas.h(x0)
    df.update_sensor("sensor_a", z + np.array([10.0, 0.01, 0.01]))
    df.update_sensor("sensor_b", z + np.array([-10.0, -0.01, -0.01]))

    fused = df.fuse()
    assert np.allclose(fused.P, fused.P.T)
