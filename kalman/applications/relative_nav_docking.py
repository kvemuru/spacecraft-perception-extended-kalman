"""Relative navigation and docking application with CW dynamics."""

import numpy as np
from kalman.ekf_core.ekf import EKF
from kalman.ekf_core.state import State
from kalman.dynamics.relative_cw import ClohessyWiltshireDynamics
from kalman.measurements.relative_pose import RelativePosition, RelativePose


class RelativeNavDocking:
    def __init__(self, x_rel0: np.ndarray, P0: np.ndarray,
                 mean_motion: float, q_scale: float = 1.0):
        self.dynamics = ClohessyWiltshireDynamics(n=mean_motion, q_scale=q_scale)
        self.state = State(x_rel0.copy(), P0.copy())
        self.ekf = EKF(self.dynamics, self.state)

    def predict(self, dt: float) -> None:
        self.ekf.predict(dt)

    def update_relative_position(self, z: np.ndarray, R: np.ndarray) -> None:
        meas = RelativePosition()
        self.ekf.update(meas, z, R)

    def update_relative_pose(self, z: np.ndarray, R: np.ndarray) -> None:
        meas = RelativePose()
        self.ekf.update(meas, z, R)

    def update_range_bearing(self, z: np.ndarray, R: np.ndarray) -> None:
        from kalman.measurements.range_bearing import RangeBearing
        meas = RangeBearing()
        self.ekf.update(meas, z, R)

    @property
    def dr(self) -> np.ndarray:
        return self.state.x[:3]

    @property
    def dv(self) -> np.ndarray:
        return self.state.x[3:6]
