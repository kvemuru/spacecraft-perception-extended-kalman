# API Reference


## `kalman.applications`

Pre-built application wrappers for common use cases.


## `kalman.applications.multi_sensor_fusion`

High-level multi-sensor fusion system with radar, camera, GPS.

### `MultiSensorFusion`

- **`add_camera(self, R: 'np.ndarray', sensor_id: 'str' = 'camera') -> 'None'`**
- **`add_gps(self, R: 'np.ndarray', jd: 'float', sensor_id: 'str' = 'gps') -> 'None'`**
- **`add_radar(self, R: 'np.ndarray', sensor_id: 'str' = 'radar', meas_model: 'MeasurementModel | None' = None) -> 'None'`**
- **`add_sensor(self, lf: 'LocalFilter') -> 'None'`**
- **`fuse(self) -> 'State'`**
- **`predict_all(self, dt: 'float') -> 'None'`**
- **`update_camera(self, z: 'np.ndarray', sensor_id: 'str' = 'camera') -> 'None'`**
- **`update_gps(self, z: 'np.ndarray', sensor_id: 'str' = 'gps') -> 'None'`**
- **`update_radar(self, z: 'np.ndarray', sensor_id: 'str' = 'radar') -> 'None'`**
- **`update_sensor(self, sensor_id: 'str', z: 'np.ndarray') -> 'None'`**


## `kalman.applications.orbit_determination`

Orbit determination application with radar measurements.

### `OrbitDetermination`

- **`predict(self, dt: float) -> None`**
- **`update_gps(self, z_gps: numpy.ndarray, R_gps: numpy.ndarray, jd: float) -> None`**
- **`update_radar(self, z_rb: numpy.ndarray, R_rb: numpy.ndarray) -> None`**
- **`update_radar_doppler(self, z_rr: numpy.ndarray, R_rr: numpy.ndarray) -> None`**


## `kalman.applications.relative_nav_docking`

Relative navigation and docking application with CW dynamics.

### `RelativeNavDocking`

- **`predict(self, dt: float) -> None`**
- **`update_range_bearing(self, z: numpy.ndarray, R: numpy.ndarray) -> None`**
- **`update_relative_pose(self, z: numpy.ndarray, R: numpy.ndarray) -> None`**
- **`update_relative_position(self, z: numpy.ndarray, R: numpy.ndarray) -> None`**


## `kalman.applications.space_object_tracking`

Space object tracking application for debris / unknown objects.

### `SpaceObjectTracking`

- **`iterate_update_angles(self, z: numpy.ndarray, R: numpy.ndarray) -> None`**
- **`predict(self, dt: float) -> None`**
- **`update_angles(self, z: numpy.ndarray, R: numpy.ndarray) -> None`**
- **`update_doppler(self, z: numpy.ndarray, R: numpy.ndarray) -> None`**
- **`update_radar(self, z: numpy.ndarray, R: numpy.ndarray) -> None`**


## `kalman.dynamics`

Pluggable dynamics models for orbital and relative motion.


## `kalman.dynamics.atmosphere`

Exponential atmospheric density model for Earth orbit.

Multi-band exponential model with altitude-dependent scale heights
suitable for spacecraft drag computation from 0-1000 km altitude.
Reference densities and scale heights approximate the US Standard
Atmosphere (1976).

#### Functions

- **`exponential_atm_density(h: float) -> float`**


## `kalman.dynamics.base`

Base dynamics model class aliased from ekf_core.interfaces.

### `DynamicsModel`

Helper class that provides a standard way to create an ABC using
inheritance.

- **`Q(self, dt: float) -> numpy.ndarray`**
- **`jacobian(self, x: numpy.ndarray, dt: float) -> numpy.ndarray`**
- **`propagate(self, x: numpy.ndarray, dt: float) -> numpy.ndarray`**


## `kalman.dynamics.j2_perturbed`

J2-perturbed orbital dynamics with optional atmospheric drag and SRP.

### `J2PerturbedDynamics`

Helper class that provides a standard way to create an ABC using
inheritance.

- **`Q(self, dt: float) -> numpy.ndarray`**
- **`f(self, x: numpy.ndarray) -> numpy.ndarray`**
- **`jacobian(self, x: numpy.ndarray, dt: float) -> numpy.ndarray`**
- **`propagate(self, x: numpy.ndarray, dt: float) -> numpy.ndarray`**
- **`stm_numeric(self, x: numpy.ndarray, dt: float, eps: float = 1e-06) -> numpy.ndarray`**


## `kalman.dynamics.relative_cw`

Clohessy-Wiltshire linearized relative motion for circular orbits.

### `ClohessyWiltshireDynamics`

Helper class that provides a standard way to create an ABC using
inheritance.

- **`A(self) -> numpy.ndarray`**
- **`Q(self, dt: float) -> numpy.ndarray`**
- **`jacobian(self, x: numpy.ndarray, dt: float) -> numpy.ndarray`**
- **`propagate(self, x: numpy.ndarray, dt: float) -> numpy.ndarray`**


## `kalman.dynamics.relative_hcw`

Hill-Clohessy-Wiltshire relative motion for elliptical orbits (numerical STM).

### `HillClohessyWiltshireDynamics`

Helper class that provides a standard way to create an ABC using
inheritance.

- **`Q(self, dt: float) -> numpy.ndarray`**
- **`f(self, x_rel: numpy.ndarray) -> numpy.ndarray`**
- **`jacobian(self, x: numpy.ndarray, dt: float) -> numpy.ndarray`**
- **`propagate(self, x: numpy.ndarray, dt: float) -> numpy.ndarray`**
- **`set_reference(self, r_ref: numpy.ndarray, v_ref: numpy.ndarray)`**
- **`stm_numeric(self, x: numpy.ndarray, dt: float, eps: float = 1e-06) -> numpy.ndarray`**


## `kalman.dynamics.two_body`

Two-body Keplerian dynamics with RK4 integration and analytic STM.

### `TwoBodyDynamics`

Helper class that provides a standard way to create an ABC using
inheritance.

- **`Q(self, dt: float) -> numpy.ndarray`**
- **`f(self, x: numpy.ndarray) -> numpy.ndarray`**
- **`jacobian(self, x: numpy.ndarray, dt: float) -> numpy.ndarray`**
- **`propagate(self, x: numpy.ndarray, dt: float) -> numpy.ndarray`**
- **`stm_analytic(self, x: numpy.ndarray, dt: float) -> numpy.ndarray`**

#### Functions

- **`rk4(f, x: numpy.ndarray, dt: float) -> numpy.ndarray`**


## `kalman.ekf_core`

Generic EKF engine: interfaces, state, predict/update with Joseph form.


## `kalman.ekf_core.ekf`

Extended Kalman Filter core: predict, update (Joseph form), IEKF, gating.

### `EKF`

- **`gated_update(self, meas: kalman.ekf_core.interfaces.MeasurementModel, z: numpy.ndarray, R: numpy.ndarray, alpha: float = 0.01) -> bool`**
- **`iterate_update(self, meas: kalman.ekf_core.interfaces.MeasurementModel, z: numpy.ndarray, R: numpy.ndarray, max_iter: int = 10, tol: float = 1e-08) -> None`**
- **`nees(self, x_true: numpy.ndarray) -> float`**
- **`nis(self, z: numpy.ndarray, meas: kalman.ekf_core.interfaces.MeasurementModel, R: numpy.ndarray) -> float`**
- **`predict(self, dt: float) -> None`**
- **`predict_to(self, target_time: float) -> None`**
- **`update(self, meas: kalman.ekf_core.interfaces.MeasurementModel, z: numpy.ndarray, R: numpy.ndarray) -> None`**


## `kalman.ekf_core.interfaces`

Abstract base classes for DynamicsModel and MeasurementModel.

### `DynamicsModel`

Helper class that provides a standard way to create an ABC using
inheritance.

- **`Q(self, dt: float) -> numpy.ndarray`**
- **`jacobian(self, x: numpy.ndarray, dt: float) -> numpy.ndarray`**
- **`propagate(self, x: numpy.ndarray, dt: float) -> numpy.ndarray`**

### `MeasurementModel`

Helper class that provides a standard way to create an ABC using
inheritance.

- **`h(self, x: numpy.ndarray) -> numpy.ndarray`**
- **`jacobian(self, x: numpy.ndarray) -> numpy.ndarray`**


## `kalman.ekf_core.state`

State vector dataclass with covariance and timestamp.

### `State`

State(x: numpy.ndarray, P: numpy.ndarray, timestamp: float = 0.0)

- **`copy(self) -> 'State'`**


## `kalman.estimation`

Advanced estimation filters: IMM, etc.


## `kalman.estimation.imm`

Interacting Multiple Model (IMM) filter for maneuvering targets.

### `IMM`

- **`init_ekfs(self, x0: numpy.ndarray, P0: numpy.ndarray) -> None`**
- **`predict(self, dt: float) -> None`**
- **`predict_and_update(self, dt: float, meas: kalman.ekf_core.interfaces.MeasurementModel, z: numpy.ndarray, R: numpy.ndarray) -> None`**
- **`update(self, meas: kalman.ekf_core.interfaces.MeasurementModel, z: numpy.ndarray, R: numpy.ndarray) -> None`**


## `kalman.fusion`

Decentralized sensor fusion via covariance intersection.


## `kalman.fusion.covariance_intersection`

Covariance intersection fusion: pairwise, sequential, and batch.

#### Functions

- **`fuse_batch(states: list[kalman.ekf_core.state.State]) -> kalman.ekf_core.state.State`**
- **`fuse_many(states: list[kalman.ekf_core.state.State]) -> kalman.ekf_core.state.State`**
- **`fuse_pair(a: kalman.ekf_core.state.State, b: kalman.ekf_core.state.State, alpha: float | None = None) -> kalman.ekf_core.state.State`**


## `kalman.fusion.decentralized`

Decentralized fusion orchestrator managing multiple LocalFilters.

### `DecentralizedFusion`

- **`add_filter(self, local_filter: kalman.fusion.local_filter.LocalFilter) -> None`**
- **`fuse(self, batch: bool = False) -> kalman.ekf_core.state.State`**
- **`get_filter_state(self, sensor_id: str) -> kalman.ekf_core.state.State`**
- **`predict_all(self, dt: float) -> None`**
- **`remove_filter(self, sensor_id: str) -> None`**
- **`update_sensor(self, sensor_id: str, z: numpy.ndarray) -> None`**


## `kalman.fusion.local_filter`

Per-sensor EKF wrapper for decentralized fusion.

### `LocalFilter`

- **`predict(self, dt: float) -> None`**
- **`predict_and_update(self, dt: float, z: numpy.ndarray) -> None`**
- **`update(self, z: numpy.ndarray) -> None`**


## `kalman.measurements`

Pluggable sensor measurement models for space perception.


## `kalman.measurements.angles_only`

Optical camera azimuth/elevation measurement model.

### `AnglesOnly`

Helper class that provides a standard way to create an ABC using
inheritance.

- **`h(self, x: numpy.ndarray) -> numpy.ndarray`**
- **`jacobian(self, x: numpy.ndarray) -> numpy.ndarray`**


## `kalman.measurements.base`

Base measurement model class aliased from ekf_core.interfaces.

### `MeasurementModel`

Helper class that provides a standard way to create an ABC using
inheritance.

- **`h(self, x: numpy.ndarray) -> numpy.ndarray`**
- **`jacobian(self, x: numpy.ndarray) -> numpy.ndarray`**


## `kalman.measurements.gps`

GPS position/velocity measurement models with ECI-to-ECEF transform.

### `GPSPosition`

Helper class that provides a standard way to create an ABC using
inheritance.

- **`h(self, x: numpy.ndarray) -> numpy.ndarray`**
- **`jacobian(self, x: numpy.ndarray) -> numpy.ndarray`**

### `GPSPositionVelocity`

Helper class that provides a standard way to create an ABC using
inheritance.

- **`h(self, x: numpy.ndarray) -> numpy.ndarray`**
- **`jacobian(self, x: numpy.ndarray) -> numpy.ndarray`**


## `kalman.measurements.range_bearing`

Radar range, azimuth, elevation measurement model.

### `RangeBearing`

Helper class that provides a standard way to create an ABC using
inheritance.

- **`h(self, x: numpy.ndarray) -> numpy.ndarray`**
- **`jacobian(self, x: numpy.ndarray) -> numpy.ndarray`**


## `kalman.measurements.range_bearing_from_station`

Ground-station radar range/azimuth/elevation measurement model.

### `RangeBearingFromStation`

Helper class that provides a standard way to create an ABC using
inheritance.

- **`h(self, x: numpy.ndarray) -> numpy.ndarray`**
- **`jacobian(self, x: numpy.ndarray) -> numpy.ndarray`**


## `kalman.measurements.range_rate`

Doppler range-rate measurement model.

### `RangeRate`

Helper class that provides a standard way to create an ABC using
inheritance.

- **`h(self, x: numpy.ndarray) -> numpy.ndarray`**
- **`jacobian(self, x: numpy.ndarray) -> numpy.ndarray`**


## `kalman.measurements.relative_pose`

Relative position / full-pose measurement models for docking.

### `RelativePose`

Helper class that provides a standard way to create an ABC using
inheritance.

- **`h(self, x: numpy.ndarray) -> numpy.ndarray`**
- **`jacobian(self, x: numpy.ndarray) -> numpy.ndarray`**

### `RelativePosition`

Helper class that provides a standard way to create an ABC using
inheritance.

- **`h(self, x: numpy.ndarray) -> numpy.ndarray`**
- **`jacobian(self, x: numpy.ndarray) -> numpy.ndarray`**


## `kalman.utils`

Utility functions: constants, coordinate transforms, time conversions.


## `kalman.utils.constants`

Physical constants for Earth and solar system.


## `kalman.utils.coordinates`

Coordinate frame transforms: ECI, ECEF, LVLH, GMST.

#### Functions

- **`ecef_to_eci(r_ecef: numpy.ndarray, gmst: float) -> numpy.ndarray`**
- **`eci_to_ecef(r_eci: numpy.ndarray, gmst: float) -> numpy.ndarray`**
- **`eci_to_lvlh(r_eci: numpy.ndarray, v_eci: numpy.ndarray, r_target: numpy.ndarray, v_target: numpy.ndarray)`**
- **`gmst_from_jd(jd: float) -> float`**
- **`lvlh_to_eci(r_lvlh: numpy.ndarray, v_lvlh: numpy.ndarray, r_target: numpy.ndarray, v_target: numpy.ndarray)`**
- **`r3(angle: float) -> numpy.ndarray`**
- **`skew(v: numpy.ndarray) -> numpy.ndarray`**


## `kalman.utils.time_`

Julian Date / MJD / datetime conversions.

#### Functions

- **`datetime_to_jd(dt: datetime.datetime) -> float`**
- **`datetime_to_mjd(dt: datetime.datetime) -> float`**
- **`jd_diff_seconds(jd1: float, jd2: float) -> float`**
- **`jd_to_datetime(jd: float) -> datetime.datetime`**
- **`jd_to_mjd(jd: float) -> float`**
- **`mjd_to_datetime(mjd: float) -> datetime.datetime`**
- **`mjd_to_jd(mjd: float) -> float`**
- **`seconds_to_jd(sec: float) -> float`**
