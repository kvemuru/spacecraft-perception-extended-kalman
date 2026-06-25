"""Multi-sensor decentralized EKF fusion demo.
Simulates a spacecraft in LEO with two sensors (radar + camera)
feeding separate local EKFs fused via covariance intersection.
"""

import numpy as np
from kalman.applications.multi_sensor_fusion import MultiSensorFusion
from kalman.dynamics.two_body import TwoBodyDynamics
from kalman.measurements.range_bearing import RangeBearing
from kalman.measurements.angles_only import AnglesOnly

np.random.seed(42)

MU = 3.986004418e14
R0 = np.array([7000e3, 0.0, 0.0])
V0 = np.array([0.0, 7546.0, 0.0])
X0 = np.concatenate([R0, V0])
DT = 10.0
N_STEPS = 50

dyn = TwoBodyDynamics(mu=MU)
P0 = np.diag([1e6, 1e6, 1e6, 1e2, 1e2, 1e2])

msf = MultiSensorFusion(X0, P0, mu=MU)
msf.add_radar(np.diag([1e3, np.deg2rad(0.02) ** 2, np.deg2rad(0.02) ** 2]))
msf.add_camera(np.diag([np.deg2rad(0.005) ** 2, np.deg2rad(0.005) ** 2]))

meas_radar = RangeBearing()
meas_cam = AnglesOnly()

x_true = X0.copy()
errors = []
for step in range(N_STEPS):
    x_true = dyn.propagate(x_true, DT)

    z_radar = meas_radar.h(x_true) + np.random.normal(0, [10, np.deg2rad(0.02), np.deg2rad(0.02)])
    z_cam = meas_cam.h(x_true) + np.random.normal(0, [np.deg2rad(0.005), np.deg2rad(0.005)])

    msf.predict_all(DT)
    msf.update_radar(z_radar)
    msf.update_camera(z_cam)
    fused = msf.fuse()

    err = np.linalg.norm(fused.x[:3] - x_true[:3])
    errors.append(err)
    if step % 10 == 0:
        print(f"step {step:3d}  position error: {err:.2f} m")

print(f"\nmean position error: {np.mean(errors):.2f} m")
print(f"final position error: {errors[-1]:.2f} m")
