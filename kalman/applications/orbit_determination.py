import numpy as np
from kalman.ekf_core.ekf import EKF
from kalman.ekf_core.state import State
from kalman.dynamics.two_body import TwoBodyDynamics
from kalman.measurements.range_bearing import RangeBearing
from kalman.measurements.range_rate import RangeRate
from kalman.measurements.gps import GPSPositionVelocity


class OrbitDetermination:
    def __init__(self, x0: np.ndarray, P0: np.ndarray,
                 mu: float = 3.986004418e14, q_scale: float = 1.0):
        self.dynamics = TwoBodyDynamics(mu=mu, q_scale=q_scale)
        self.state = State(x0.copy(), P0.copy())
        self.ekf = EKF(self.dynamics, self.state)

    def predict(self, dt: float) -> None:
        self.ekf.predict(dt)

    def update_radar(self, z_rb: np.ndarray, R_rb: np.ndarray) -> None:
        meas = RangeBearing()
        self.ekf.update(meas, z_rb, R_rb)

    def update_radar_doppler(self, z_rr: np.ndarray, R_rr: np.ndarray) -> None:
        meas = RangeRate()
        self.ekf.update(meas, z_rr, R_rr)

    def update_gps(self, z_gps: np.ndarray, R_gps: np.ndarray, jd: float) -> None:
        meas = GPSPositionVelocity(jd)
        self.ekf.update(meas, z_gps, R_gps)

    @property
    def r(self) -> np.ndarray:
        return self.state.x[:3]

    @property
    def v(self) -> np.ndarray:
        return self.state.x[3:6]
