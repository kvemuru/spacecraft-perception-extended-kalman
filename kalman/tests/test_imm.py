import numpy as np
from kalman.estimation.imm import IMM
from kalman.dynamics.two_body import TwoBodyDynamics
from kalman.dynamics.j2_perturbed import J2PerturbedDynamics
from kalman.measurements.range_bearing import RangeBearing
from kalman.utils.constants import MU_EARTH


def test_imm_init():
    dyn_a = TwoBodyDynamics(mu=MU_EARTH, q_scale=1e-8)
    dyn_b = TwoBodyDynamics(mu=MU_EARTH, q_scale=1e-4)
    Pi = np.array([[0.95, 0.05], [0.05, 0.95]])
    imm = IMM([dyn_a, dyn_b], Pi)
    r0 = np.array([7000e3, 0.0, 0.0])
    v0 = np.array([0.0, 7546.0, 0.0])
    x0 = np.concatenate([r0, v0])
    P0 = np.diag([1e6, 1e6, 1e6, 1e2, 1e2, 1e2])
    imm.init_ekfs(x0, P0)
    s = imm.state
    assert s.x.shape == (6,)
    assert np.allclose(s.P, s.P.T)


def test_imm_predict_update():
    dyn_a = TwoBodyDynamics(mu=MU_EARTH, q_scale=1e-10)
    dyn_b = TwoBodyDynamics(mu=MU_EARTH, q_scale=1e-6)
    Pi = np.array([[0.9, 0.1], [0.1, 0.9]])
    imm = IMM([dyn_a, dyn_b], Pi)
    r0 = np.array([7000e3, 100.0, 50.0])
    v0 = np.array([0.0, 7546.0, 10.0])
    x0 = np.concatenate([r0, v0])
    P0 = np.diag([1e6, 1e6, 1e6, 1e2, 1e2, 1e2])
    imm.init_ekfs(x0, P0)
    imm.predict(10.0)
    assert abs(imm.state.timestamp - 10.0) < 1e-9
    meas = RangeBearing()
    z = meas.h(imm.state.x)
    R = np.diag([1e3, np.deg2rad(0.1) ** 2, np.deg2rad(0.1) ** 2])
    imm.update(meas, z, R)
    assert np.allclose(imm.mode_probabilities.sum(), 1.0)


def test_imm_state_reasonable_with_mismatch():
    np.random.seed(42)
    dyn_a = TwoBodyDynamics(mu=MU_EARTH, q_scale=1e-10)
    dyn_b = J2PerturbedDynamics(mu=MU_EARTH, q_scale=1e-2)
    Pi = np.array([[0.9, 0.1], [0.1, 0.9]])
    imm = IMM([dyn_a, dyn_b], Pi)
    r0 = np.array([7000e3, 0.0, 0.0])
    v0 = np.array([0.0, np.sqrt(MU_EARTH / 7000e3), 0.0])
    x0 = np.concatenate([r0, v0])
    P0 = np.diag([1e6, 1e6, 1e6, 1e2, 1e2, 1e2])
    imm.init_ekfs(x0, P0)
    truth = TwoBodyDynamics(mu=MU_EARTH)
    x_true = x0.copy()
    meas = RangeBearing()
    R = np.diag([1e3, np.deg2rad(0.1) ** 2, np.deg2rad(0.1) ** 2])
    dt = 60.0
    for _ in range(20):
        x_true = truth.propagate(x_true, dt)
        z = meas.h(x_true) + np.random.normal(0, [50.0, np.deg2rad(0.005), np.deg2rad(0.005)])
        imm.predict_and_update(dt, meas, z, R)
    state = imm.state
    pos_err = np.linalg.norm(state.x[:3] - x_true[:3])
    assert pos_err < 50_000.0


def test_imm_component_states():
    dyn_a = TwoBodyDynamics(mu=MU_EARTH, q_scale=1e-8)
    dyn_b = TwoBodyDynamics(mu=MU_EARTH, q_scale=1e-6)
    Pi = np.array([[0.9, 0.1], [0.1, 0.9]])
    imm = IMM([dyn_a, dyn_b], Pi)
    r0 = np.array([7000e3, 0.0, 0.0])
    v0 = np.array([0.0, 7546.0, 0.0])
    x0 = np.concatenate([r0, v0])
    P0 = np.eye(6)
    imm.init_ekfs(x0, P0)
    states = imm.component_states
    assert len(states) == 2
    assert np.allclose(states[0].x, x0)
