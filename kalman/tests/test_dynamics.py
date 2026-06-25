import numpy as np
from kalman.dynamics.two_body import TwoBodyDynamics, rk4
from kalman.dynamics.j2_perturbed import J2PerturbedDynamics
from kalman.dynamics.relative_cw import ClohessyWiltshireDynamics
from kalman.dynamics.relative_hcw import HillClohessyWiltshireDynamics
from kalman.utils.constants import MU_EARTH, R_EARTH


def test_two_body_propagate():
    dyn = TwoBodyDynamics()
    r0 = np.array([7000e3, 0.0, 0.0])
    v0 = np.array([0.0, 7546.0, 0.0])
    x0 = np.concatenate([r0, v0])
    x1 = dyn.propagate(x0, 100.0)
    assert len(x1) == 6
    r1_norm = np.linalg.norm(x1[:3])
    assert abs(r1_norm - 7000e3) < 1e3


def test_rk4_circular_orbit():
    mu = MU_EARTH
    r0 = 7000e3
    v0 = np.sqrt(mu / r0)
    x = np.array([r0, 0.0, 0.0, 0.0, v0, 0.0])
    period = 2 * np.pi * r0 / v0
    steps = 100
    dt = period / steps
    xf = x.copy()
    for _ in range(steps):
        xf = rk4(lambda s: np.concatenate([s[3:6], [-mu * s[0] / np.linalg.norm(s[:3]) ** 3,
                                                      -mu * s[1] / np.linalg.norm(s[:3]) ** 3,
                                                      -mu * s[2] / np.linalg.norm(s[:3]) ** 3]]), xf, dt)
    assert np.allclose(x[:3], xf[:3], atol=1e3)


def test_two_body_stm():
    dyn = TwoBodyDynamics()
    r0 = np.array([7000e3, 100.0, 50.0])
    v0 = np.array([0.0, 7546.0, 10.0])
    x0 = np.concatenate([r0, v0])
    dt = 10.0
    Phi = dyn.jacobian(x0, dt)
    assert Phi.shape == (6, 6)
    assert np.allclose(Phi[:3, 3:6], np.eye(3) * dt)


def test_cw_stm():
    n = 0.001
    dyn = ClohessyWiltshireDynamics(n=n)
    x0 = np.array([100.0, 50.0, 20.0, 1.0, 0.5, 0.2])
    Phi = dyn.jacobian(x0, 10.0)
    assert Phi.shape == (6, 6)
    x1 = dyn.propagate(x0, 10.0)
    x1_expected = Phi @ x0
    assert np.allclose(x1, x1_expected)


def test_cw_numerical_stm():
    n = 0.001
    dyn = ClohessyWiltshireDynamics(n=n)
    x0 = np.array([100.0, 50.0, 20.0, 1.0, 0.5, 0.2])
    dt = 10.0
    Phi_analytic = dyn.jacobian(x0, dt)
    Phi_numeric = _numeric_stm(dyn, x0, dt)
    assert np.allclose(Phi_analytic, Phi_numeric, atol=1e-8)


def test_j2_vs_two_body():
    dyn2 = TwoBodyDynamics()
    dyn_j2 = J2PerturbedDynamics(j2=0.0)
    r0 = np.array([7000e3, 0.0, 0.0])
    v0 = np.array([0.0, 7546.0, 0.0])
    x0 = np.concatenate([r0, v0])
    x1 = dyn2.propagate(x0, 100.0)
    x1_j2 = dyn_j2.propagate(x0, 100.0)
    assert np.allclose(x1, x1_j2)


def test_hcw_vs_cw_circular():
    r0 = 7000e3
    v0 = np.sqrt(MU_EARTH / r0)
    n = v0 / r0
    dyn_cw = ClohessyWiltshireDynamics(n=n)
    dyn_hcw = HillClohessyWiltshireDynamics(mu=MU_EARTH)
    r_ref = np.array([r0, 0.0, 0.0])
    v_ref = np.array([0.0, v0, 0.0])
    dyn_hcw.set_reference(r_ref, v_ref)
    x_rel = np.array([100.0, 50.0, 20.0, 1.0, 0.5, 0.2])
    dt = 10.0
    x_cw = dyn_cw.propagate(x_rel, dt)
    x_hcw = dyn_hcw.propagate(x_rel, dt)
    assert np.allclose(x_cw, x_hcw, atol=1.0)


def test_exponential_atm_density():
    from kalman.dynamics.atmosphere import exponential_atm_density
    assert exponential_atm_density(0.0) > 1.0
    assert exponential_atm_density(100_000.0) > 0
    assert exponential_atm_density(100_000.0) < 1.0
    assert exponential_atm_density(400_000.0) < exponential_atm_density(200_000.0)
    assert exponential_atm_density(1_000_000.0) > 0


def test_drag_with_exponential_model():
    dyn = J2PerturbedDynamics(cd=2.2, area_mass_ratio=0.01)
    r = np.array([R_EARTH + 400_000.0, 0.0, 0.0])
    v = np.array([0.0, 7666.0, 0.0])
    x = np.concatenate([r, v])
    x1 = dyn.propagate(x, 60.0)
    r1_norm = np.linalg.norm(x1[:3])
    assert r1_norm < np.linalg.norm(r)


def test_j2_mismatch_at_multiple_altitudes():
    from kalman.ekf_core.ekf import EKF
    from kalman.ekf_core.state import State
    from kalman.measurements.gps import GPSPositionVelocity
    altitudes = [400_000.0, 20_000_000.0, 35_786_000.0]
    for alt in altitudes:
        r_norm = R_EARTH + alt
        v_circ = np.sqrt(MU_EARTH / r_norm)
        r0 = np.array([r_norm, 0.0, 0.0])
        v0 = np.array([0.0, v_circ, 0.0])
        x0 = np.concatenate([r0, v0])
        truth = J2PerturbedDynamics(mu=MU_EARTH)
        filt = TwoBodyDynamics(mu=MU_EARTH, q_scale=1e-4)
        state = State(x0.copy(), np.diag([1e6, 1e6, 1e6, 1e2, 1e2, 1e2]))
        ekf = EKF(filt, state)
        jd = 2460000.0
        x_true = x0.copy()
        errors = []
        dt = 10.0
        for step in range(200):
            x_true = truth.propagate(x_true, dt)
            ekf.predict(dt)
            if step % 3 == 0:
                meas = GPSPositionVelocity(jd + step * dt / 86400.0)
                z = meas.h(x_true)
                R = np.diag([10.0 ** 2] * 3 + [0.1 ** 2] * 3)
                ekf.update(meas, z, R)
            err = np.linalg.norm(ekf.state.x[:3] - x_true[:3])
            errors.append(err)
        last_50 = np.mean(errors[-50:])
        assert last_50 < 1000.0


def test_j2_mismatch_ekf_converges():
    from kalman.ekf_core.ekf import EKF
    from kalman.ekf_core.state import State
    from kalman.measurements.gps import GPSPositionVelocity
    r0 = np.array([7000e3, 0.0, 0.0])
    v0 = np.array([0.0, np.sqrt(MU_EARTH / 7000e3), 0.0])
    x0 = np.concatenate([r0, v0])
    truth = J2PerturbedDynamics(mu=MU_EARTH)
    filt = TwoBodyDynamics(mu=MU_EARTH, q_scale=1e-4)
    state = State(x0.copy(), np.diag([1e6, 1e6, 1e6, 1e2, 1e2, 1e2]))
    ekf = EKF(filt, state)
    jd = 2460000.0
    x_true = x0.copy()
    errors = []
    dt = 10.0
    for step in range(200):
        x_true = truth.propagate(x_true, dt)
        ekf.predict(dt)
        if step % 3 == 0:
            meas = GPSPositionVelocity(jd + step * dt / 86400.0)
            z = meas.h(x_true)
            R = np.diag([10.0 ** 2] * 3 + [0.1 ** 2] * 3)
            ekf.update(meas, z, R)
        err = np.linalg.norm(ekf.state.x[:3] - x_true[:3])
        errors.append(err)
    last_50 = np.mean(errors[-50:])
    assert last_50 < 1000.0


def _numeric_stm(dyn, x, dt, eps=1e-6):
    n = len(x)
    Phi = np.eye(n)
    for i in range(n):
        dx = np.zeros(n)
        dx[i] = eps
        xp = dyn.propagate(x + dx, dt)
        xm = dyn.propagate(x - dx, dt)
        Phi[:, i] = (xp - xm) / (2 * eps)
    return Phi
