import numpy as np
from kalman.fusion.decentralized import DecentralizedFusion
from kalman.fusion.local_filter import LocalFilter
from kalman.fusion.covariance_intersection import fuse_pair
from kalman.ekf_core.state import State
from kalman.dynamics.two_body import TwoBodyDynamics
from kalman.dynamics.j2_perturbed import J2PerturbedDynamics
from kalman.measurements.range_bearing import RangeBearing
from kalman.measurements.angles_only import AnglesOnly
from kalman.measurements.gps import GPSPositionVelocity
from kalman.measurements.relative_pose import RelativePosition


class MultiSensorFusion:
    def __init__(self, x0: np.ndarray, P0: np.ndarray,
                 mu: float = 3.986004418e14, q_scale: float = 1.0):
        self.dynamics = TwoBodyDynamics(mu=mu, q_scale=q_scale)
        self.dynamics_high = J2PerturbedDynamics(mu=mu, q_scale=q_scale)
        self.fusion = DecentralizedFusion(self.dynamics, len(x0))
        self._x0 = x0
        self._P0 = P0

    def add_gps(self, R: np.ndarray, jd: float, sensor_id: str = "gps") -> None:
        state = State(self._x0.copy(), self._P0.copy())
        meas = GPSPositionVelocity(jd)
        lf = LocalFilter(self.dynamics, state, meas, R, sensor_id)
        self.fusion.add_filter(lf)

    def add_radar(self, R: np.ndarray, sensor_id: str = "radar") -> None:
        state = State(self._x0.copy(), self._P0.copy())
        meas = RangeBearing()
        lf = LocalFilter(self.dynamics, state, meas, R, sensor_id)
        self.fusion.add_filter(lf)

    def add_camera(self, R: np.ndarray, sensor_id: str = "camera") -> None:
        state = State(self._x0.copy(), self._P0.copy())
        meas = AnglesOnly()
        lf = LocalFilter(self.dynamics, state, meas, R, sensor_id)
        self.fusion.add_filter(lf)

    def add_sensor(self, lf: LocalFilter) -> None:
        self.fusion.add_filter(lf)

    def predict_all(self, dt: float) -> None:
        self.fusion.predict_all(dt)

    def update_gps(self, z: np.ndarray, sensor_id: str = "gps") -> None:
        self.fusion.update_sensor(sensor_id, z)

    def update_radar(self, z: np.ndarray, sensor_id: str = "radar") -> None:
        self.fusion.update_sensor(sensor_id, z)

    def update_camera(self, z: np.ndarray, sensor_id: str = "camera") -> None:
        self.fusion.update_sensor(sensor_id, z)

    def update_sensor(self, sensor_id: str, z: np.ndarray) -> None:
        self.fusion.update_sensor(sensor_id, z)

    def fuse(self) -> State:
        return self.fusion.fuse()
