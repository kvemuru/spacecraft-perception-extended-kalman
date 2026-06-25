"""State vector dataclass with covariance and timestamp."""

import numpy as np
from dataclasses import dataclass, field


@dataclass
class State:
    x: np.ndarray
    P: np.ndarray
    timestamp: float = 0.0

    def __post_init__(self):
        assert self.x.ndim == 1, "State vector must be 1-D"
        assert self.P.ndim == 2, "Covariance must be 2-D"
        assert self.x.shape[0] == self.P.shape[0] == self.P.shape[1]

    def copy(self) -> "State":
        return State(self.x.copy(), self.P.copy(), self.timestamp)
