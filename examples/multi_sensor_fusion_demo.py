"""Multi-sensor decentralized EKF fusion demo.

Truth runs with two-body dynamics. Three sensors (ground radar,
optical camera, GPS) feed separate local EKFs fused via
covariance intersection.
"""

import numpy as np
from scipy.stats import chi2
from kalman.applications.multi_sensor_fusion import MultiSensorFusion
from kalman.dynamics.two_body import TwoBodyDynamics
from kalman.measurements.range_bearing import RangeBearing
from kalman.measurements.angles_only import AnglesOnly
from kalman.measurements.gps import GPSPositionVelocity
from kalman.fusion.local_filter import LocalFilter
from kalman.ekf_core.state import State

np.random.seed(42)
MU = 3.986004418e14
INITIAL_JD = 2460000.0

R0 = np.array([7000e3, 0.0, 0.0])
V0 = np.array([0.0, np.sqrt(MU / 7000e3), 0.0])
X0 = np.concatenate([R0, V0])

truth_dyn = TwoBodyDynamics(mu=MU)

P0 = np.diag([1e6, 1e6, 1e6, 1e2, 1e2, 1e2])
Q_SCALE = 1e-10

msf = MultiSensorFusion(X0, P0, mu=MU, q_scale=Q_SCALE)
filt_dyn = TwoBodyDynamics(mu=MU, q_scale=Q_SCALE)

msf.add_sensor(LocalFilter(
    filt_dyn, State(X0.copy(), P0.copy()),
    RangeBearing(),
    np.diag([50.0 ** 2, np.deg2rad(0.005) ** 2, np.deg2rad(0.005) ** 2]),
    "radar"))

meas_cam = AnglesOnly()
R_cam = np.diag([np.deg2rad(0.001) ** 2, np.deg2rad(0.001) ** 2])
msf.add_sensor(LocalFilter(filt_dyn, State(X0.copy(), P0.copy()),
                           meas_cam, R_cam, "camera"))

R_gps = np.diag([10.0 ** 2, 10.0 ** 2, 10.0 ** 2, 0.1 ** 2, 0.1 ** 2, 0.1 ** 2])
msf.add_gps(R_gps, jd=INITIAL_JD)

meas_radar = RangeBearing()
DT = 10.0
N_STEPS = 200

x_true = X0.copy()
fused_errs = []
gps_errs = []

for step in range(N_STEPS):
    x_true = truth_dyn.propagate(x_true, DT)

    z_radar = meas_radar.h(x_true) + np.random.normal(0, [50.0, np.deg2rad(0.005), np.deg2rad(0.005)])
    z_cam = meas_cam.h(x_true) + np.random.normal(0, [np.deg2rad(0.001), np.deg2rad(0.001)])
    gps_jd = INITIAL_JD + step * DT / 86400.0

    msf.predict_all(DT)
    msf.update_radar(z_radar)

    nis = msf.fusion.filters["camera"].ekf.nis(z_cam, meas_cam, R_cam)
    if nis < chi2.ppf(0.99, 2):
        msf.update_camera(z_cam)

    gps_filter = msf.fusion.filters["gps"]
    gps_filter.meas_model.jd = gps_jd
    z_gps = gps_filter.meas_model.h(x_true) + np.random.normal(0, [10.0, 10.0, 10.0, 0.1, 0.1, 0.1])
    msf.update_gps(z_gps)

    fused = msf.fuse()
    fused_errs.append(np.linalg.norm(fused.x[:3] - x_true[:3]))
    gps_err = np.linalg.norm(gps_filter.state.x[:3] - x_true[:3])
    gps_errs.append(gps_err)

    if step % 40 == 0:
        print(f"step {step:3d}  fused: {fused_errs[-1]:6.1f} m  gps-only: {gps_err:6.1f} m")

print(f"\nMean position error — fused: {np.mean(fused_errs):.1f} m  gps-only: {np.mean(gps_errs):.1f} m")
print(f"Final  position error — fused: {fused_errs[-1]:.1f} m  gps-only: {gps_errs[-1]:.1f} m")
