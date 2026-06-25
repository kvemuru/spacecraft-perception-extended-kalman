"""Compare position errors across sensor configurations at every step.

Each filter runs persistently across all steps. Error at each step
reflects the cumulative filtering performance, not a single-shot estimate.
"""

import numpy as np
from kalman.dynamics.two_body import TwoBodyDynamics
from kalman.measurements.range_bearing import RangeBearing
from kalman.measurements.gps import GPSPosition, GPSPositionVelocity
from kalman.fusion.local_filter import LocalFilter
from kalman.fusion.decentralized import DecentralizedFusion
from kalman.fusion.covariance_intersection import fuse_many
from kalman.ekf_core.state import State
from kalman.ekf_core.ekf import EKF

np.random.seed(42)
MU = 3.986004418e14

R0 = np.array([7000e3, 0.0, 0.0])
V0 = np.array([0.0, np.sqrt(MU / 7000e3), 0.0])
X0 = np.concatenate([R0, V0])
P0 = np.diag([1e6, 1e6, 1e6, 1e2, 1e2, 1e2])
Q_SCALE = 1e-10

truth_dyn = TwoBodyDynamics(mu=MU)
filt_dyn = TwoBodyDynamics(mu=MU, q_scale=Q_SCALE)

R_radar = np.diag([50.0 ** 2, np.deg2rad(0.005) ** 2, np.deg2rad(0.005) ** 2])
R_gps_pos = np.diag([10.0 ** 2] * 3)
R_gps_pv = np.diag([10.0 ** 2] * 3 + [0.1 ** 2] * 3)

meas_radar = RangeBearing()

# --- Set up persistent filters ---
ekf_gps_only = EKF(filt_dyn, State(X0.copy(), P0.copy()))
ekf_radar_only = EKF(filt_dyn, State(X0.copy(), P0.copy()))
ekf_radar_gps = EKF(filt_dyn, State(X0.copy(), P0.copy()))

# --- Fusion system with all 3 sensors ---
msf = DecentralizedFusion(filt_dyn, 6)
for sid in ["radar", "gps"]:
    lf = LocalFilter(filt_dyn, State(X0.copy(), P0.copy()),
                     meas_radar if sid == "radar" else GPSPositionVelocity(2460000.0),
                     R_radar if sid == "radar" else R_gps_pv, sid)
    msf.add_filter(lf)

DT = 10.0
N_STEPS = 200

x_true = X0.copy()
errors = {"GPS-only": [], "Radar-only": [], "Radar+GPS EKF": [], "Fused": []}

for step in range(N_STEPS):
    x_true = truth_dyn.propagate(x_true, DT)
    z_radar = meas_radar.h(x_true) + np.random.normal(
        0, [50.0, np.deg2rad(0.005), np.deg2rad(0.005)])
    gps_jd = 2460000.0 + step * DT / 86400.0
    z_gps_pos = GPSPosition(gps_jd).h(x_true) + np.random.normal(0, [10.0, 10.0, 10.0])
    z_gps_pv = GPSPositionVelocity(gps_jd).h(x_true) + np.random.normal(
        0, [10.0, 10.0, 10.0, 0.1, 0.1, 0.1])

    # GPS-only: position-only GPS every step
    ekf_gps_only.predict(DT)
    ekf_gps_only.update(GPSPosition(gps_jd), z_gps_pos, R_gps_pos)
    errors["GPS-only"].append(np.linalg.norm(ekf_gps_only.state.x[:3] - x_true[:3]))

    # Radar-only
    ekf_radar_only.predict(DT)
    ekf_radar_only.update(meas_radar, z_radar, R_radar)
    errors["Radar-only"].append(np.linalg.norm(ekf_radar_only.state.x[:3] - x_true[:3]))

    # Radar+GPS in a single EKF
    ekf_radar_gps.predict(DT)
    ekf_radar_gps.update(meas_radar, z_radar, R_radar)
    ekf_radar_gps.update(GPSPositionVelocity(gps_jd), z_gps_pv, R_gps_pv)
    errors["Radar+GPS EKF"].append(np.linalg.norm(ekf_radar_gps.state.x[:3] - x_true[:3]))

    # Fused via CI: radar + GPS local EKFs
    msf.predict_all(DT)
    msf.filters["radar"].update(z_radar)
    msf.filters["gps"].meas_model.jd = gps_jd
    msf.filters["gps"].update(z_gps_pv)
    fused = msf.fuse()
    errors["Fused"].append(np.linalg.norm(fused.x[:3] - x_true[:3]))

# --- Print table ---
header = f"{'step':>5}  {'GPS-only':>10}  {'Radar-only':>12}  {'Radar+GPS':>12}  {'Fused':>12}"
print(header)
print("-" * len(header))
for step in range(0, N_STEPS, 25):
    print(f"{step:5d}  {errors['GPS-only'][step]:10.2f}  "
          f"{errors['Radar-only'][step]:12.1f}  "
          f"{errors['Radar+GPS EKF'][step]:12.1f}  "
          f"{errors['Fused'][step]:12.1f}")

print()
for name in errors:
    e = errors[name]
    print(f"{name:15s}  mean: {np.mean(e):8.1f} m  final: {e[-1]:8.1f} m")
