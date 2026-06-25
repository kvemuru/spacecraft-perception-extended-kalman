import numpy as np
from kalman.fusion.local_filter import LocalFilter
from kalman.fusion.covariance_intersection import fuse_many
from kalman.ekf_core.state import State
from kalman.ekf_core.interfaces import DynamicsModel


class DecentralizedFusion:
    def __init__(self, dynamics: DynamicsModel, num_states: int):
        self.dynamics = dynamics
        self.filters: dict[str, LocalFilter] = {}
        self._num_states = num_states

    def add_filter(self, local_filter: LocalFilter) -> None:
        self.filters[local_filter.sensor_id] = local_filter

    def remove_filter(self, sensor_id: str) -> None:
        self.filters.pop(sensor_id, None)

    def predict_all(self, dt: float) -> None:
        for f in self.filters.values():
            f.predict(dt)

    def update_sensor(self, sensor_id: str, z: np.ndarray) -> None:
        self.filters[sensor_id].update(z)

    def fuse(self) -> State:
        states = [f.state for f in self.filters.values()]
        return fuse_many(states)

    def get_filter_state(self, sensor_id: str) -> State:
        return self.filters[sensor_id].state
