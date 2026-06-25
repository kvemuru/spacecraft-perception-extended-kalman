import numpy as np
from kalman.applications.orbit_determination import OrbitDetermination
from kalman.applications.relative_nav_docking import RelativeNavDocking
from kalman.applications.space_object_tracking import SpaceObjectTracking
from kalman.applications.multi_sensor_fusion import MultiSensorFusion
from kalman.dynamics.two_body import TwoBodyDynamics
from kalman.measurements.range_bearing import RangeBearing
from kalman.utils.constants import MU_EARTH


def test_orbit_determination_radar():
    r0 = np.array([7000e3, 0.0, 0.0])
    v0 = np.array([0.0, 7546.0, 0.0])
    x0 = np.concatenate([r0, v0])
    P0 = np.diag([1e6, 1e6, 1e6, 1e2, 1e2, 1e2])
    od = OrbitDetermination(x0, P0)
    od.predict(60.0)
    meas = RangeBearing()
    z = meas.h(od.state.x)
    R = np.diag([1e3, np.deg2rad(0.01) ** 2, np.deg2rad(0.01) ** 2])
    od.update_radar(z, R)
    assert od.r.shape == (3,)
    assert od.v.shape == (3,)


def test_relative_nav_docking():
    n = 0.001
    x_rel0 = np.array([100.0, 50.0, 20.0, 1.0, 0.5, 0.2])
    P0 = np.diag([10.0, 10.0, 10.0, 1.0, 1.0, 1.0])
    rnd = RelativeNavDocking(x_rel0, P0, n)
    rnd.predict(10.0)
    rnd.update_relative_position(x_rel0[:3], np.eye(3) * 1.0)
    assert np.allclose(rnd.dr, x_rel0[:3], atol=1.0)


def test_space_object_tracking():
    r0 = np.array([7000e3, 100.0, 50.0])
    v0 = np.array([0.0, 7546.0, 10.0])
    x0 = np.concatenate([r0, v0])
    P0 = np.diag([1e8, 1e8, 1e8, 1e4, 1e4, 1e4])
    sot = SpaceObjectTracking(x0, P0)
    sot.predict(60.0)
    meas = RangeBearing()
    z = meas.h(sot.state.x)
    R = np.diag([1e3, np.deg2rad(0.01) ** 2, np.deg2rad(0.01) ** 2])
    sot.update_radar(z, R)
    assert sot.state.x.shape == (6,)


def test_multi_sensor_fusion():
    r0 = np.array([7000e3, 0.0, 0.0])
    v0 = np.array([0.0, 7546.0, 0.0])
    x0 = np.concatenate([r0, v0])
    P0 = np.diag([1e6, 1e6, 1e6, 1e2, 1e2, 1e2])
    msf = MultiSensorFusion(x0, P0)
    msf.add_radar(np.diag([1e3, np.deg2rad(0.01) ** 2, np.deg2rad(0.01) ** 2]))
    msf.add_camera(np.diag([np.deg2rad(0.001) ** 2, np.deg2rad(0.001) ** 2]))
    msf.predict_all(10.0)
    meas = RangeBearing()
    z_radar = meas.h(x0)
    z_cam = meas.h(x0)[1:]
    msf.update_radar(z_radar)
    msf.update_camera(z_cam)
    fused = msf.fuse()
    assert fused.x.shape == (6,)
    assert np.allclose(fused.P, fused.P.T)


def test_multi_sensor_fusion_runs_and_produces_symmetric_covariance():
    from kalman.measurements.gps import GPSPositionVelocity
    from kalman.measurements.angles_only import AnglesOnly
    np.random.seed(42)
    r0 = np.array([7000e3, 0.0, 0.0])
    v0 = np.array([0.0, 7546.0, 0.0])
    x0 = np.concatenate([r0, v0])
    P0 = np.diag([1e6, 1e6, 1e6, 1e2, 1e2, 1e2])
    truth = TwoBodyDynamics(mu=MU_EARTH)
    msf = MultiSensorFusion(x0.copy(), P0, mu=MU_EARTH, q_scale=1e-10)
    msf.add_camera(np.diag([np.deg2rad(0.001) ** 2, np.deg2rad(0.001) ** 2]))
    msf.add_gps(np.diag([10.0 ** 2] * 3 + [0.1 ** 2] * 3), jd=2460000.0)
    x_true = x0.copy()
    meas_cam = AnglesOnly()
    meas_gps = GPSPositionVelocity(2460000.0)
    for step in range(50):
        x_true = truth.propagate(x_true, 10.0)
        z_cam = meas_cam.h(x_true) + np.random.normal(0, [np.deg2rad(0.001), np.deg2rad(0.001)])
        gps_jd = 2460000.0 + step * 10.0 / 86400.0
        msf.predict_all(10.0)
        msf.update_camera(z_cam)
        gps_filter = msf.fusion.filters["gps"]
        gps_filter.meas_model.jd = gps_jd
        z_gps = meas_gps.h(x_true) + np.random.normal(0, [10.0, 10.0, 10.0, 0.1, 0.1, 0.1])
        msf.update_gps(z_gps)
        fused = msf.fuse()
        assert np.allclose(fused.P, fused.P.T)
    assert fused.x.shape == (6,)
