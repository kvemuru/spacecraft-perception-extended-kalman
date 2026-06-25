"""Base dynamics model class aliased from ekf_core.interfaces."""

from kalman.ekf_core.interfaces import DynamicsModel as DynamicsModelBase

class DynamicsModel(DynamicsModelBase):
    pass
