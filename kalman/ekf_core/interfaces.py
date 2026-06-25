"""Abstract base classes for DynamicsModel and MeasurementModel."""

import numpy as np
from abc import ABC, abstractmethod


class DynamicsModel(ABC):
    @abstractmethod
    def propagate(self, x: np.ndarray, dt: float) -> np.ndarray:
        pass

    @abstractmethod
    def jacobian(self, x: np.ndarray, dt: float) -> np.ndarray:
        pass

    @abstractmethod
    def Q(self, dt: float) -> np.ndarray:
        pass

    @property
    @abstractmethod
    def state_dim(self) -> int:
        pass


class MeasurementModel(ABC):
    @abstractmethod
    def h(self, x: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def jacobian(self, x: np.ndarray) -> np.ndarray:
        pass
