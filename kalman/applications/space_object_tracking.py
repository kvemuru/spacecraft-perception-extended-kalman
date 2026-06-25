import numpy as np
from kalman.ekf_core.ekf import EKF
from kalman.ekf_core.state import State
from kalman.dynamics.two_body import TwoBodyDynamics
from kalman.measurements.angles_only import AnglesOnly
from kalman.measurements.range_bearing import RangeBearing
from kalman.measurements.range_rate import RangeRate


class SpaceObjectTracking:
    def __init__(self, x0: np.ndarray, P0: np.ndarray,
                 mu: float = 3.986004418e14, q_scale: float = 1.0):
        self.dynamics = TwoBodyDynamics(mu=mu, q_scale=q_scale)
        self.state = State(x0.copy(), P0.copy())
        self.ekf = EKF(self.dynamics, self.state)

    def predict(self, dt: float) -> None:
        self.ekf.predict(dt)

    def update_angles(self, z: np.ndarray, R: np.ndarray) -> None:
        meas = AnglesOnly()
        self.ekf.update(meas, z, R)

    def update_radar(self, z: np.ndarray, R: np.ndarray) -> None:
        meas = RangeBearing()
        self.ekf.update(meas, z, R)

    def update_doppler(self, z: np.ndarray, R: np.ndarray) -> None:
        meas = RangeRate()
        self.ekf.update(meas, z, R)

    def iterate_update_angles(self, z: np.ndarray, R: np.ndarray) -> None:
        meas = AnglesOnly()
        self.ekf.iterate_update(meas, z, R)
