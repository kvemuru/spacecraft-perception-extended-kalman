"""Base measurement model class aliased from ekf_core.interfaces."""

from kalman.ekf_core.interfaces import MeasurementModel as MeasurementModelBase

class MeasurementModel(MeasurementModelBase):
    pass
