# kalman — Modular Extended Kalman Filter for Space Perception

A modular Extended Kalman Filter framework for spacecraft perception, organized around four core space domain awareness tasks: absolute orbit determination, relative navigation & docking, space object tracking, and multi-sensor fusion. The design separates the EKF into four pluggable layers — **dynamics models** (how the state evolves), **measurement models** (how sensors observe the state), a **generic EKF core** (predict/update cycle with Joseph-form covariance updates and iterated IEKF refinement), and a **decentralized fusion layer** (per-sensor local EKFs publishing information vectors, fused via covariance intersection to handle unknown cross-correlations). Each layer defines abstract interfaces so users can swap in new motion models or sensor types without touching the filter logic. Pre-built application wrappers wire these layers together for common use cases, while utility modules handle coordinate frames (ECI↔ECEF↔LVLH), time conversions, and physical constants.

## Architecture

```
kalman/
├── ekf_core/           # Generic EKF engine
│   ├── interfaces.py   # ABCs: DynamicsModel, MeasurementModel
│   ├── state.py        # State dataclass (x, P, timestamp)
│   └── ekf.py          # predict(), update() (Joseph form), IEKF, NEES/NIS
├── dynamics/           # Pluggable motion models
│   ├── two_body.py     # Keplerian: r̈ = -μr/‖r‖³, RK4 + analytic STM
│   ├── j2_perturbed.py # J2 + optional drag & SRP
│   ├── relative_cw.py  # Clohessy-Wiltshire (circular, closed-form STM)
│   └── relative_hcw.py # Hill-CW (elliptical, numeric STM)
├── measurements/       # Pluggable sensor models
│   ├── range_bearing.py   # Radar: [ρ, az, el]
│   ├── range_rate.py      # Doppler: [ρ̇]
│   ├── angles_only.py     # Optical: [az, el]
│   ├── gps.py             # GPS position/velocity (ECI↔ECEF)
│   └── relative_pose.py   # Relative position / full 6-DOF pose
├── fusion/             # Decentralized fusion
│   ├── local_filter.py        # Per-sensor EKF wrapper
│   ├── covariance_intersection.py  # Optimal CI via golden-section search
│   └── decentralized.py       # Master orchestrator for N local EKFs
├── applications/       # Use-case wrappers
│   ├── orbit_determination.py
│   ├── relative_nav_docking.py
│   ├── space_object_tracking.py
│   └── multi_sensor_fusion.py
├── utils/
│   ├── constants.py    # μ⊕, J₂, R⊕, ω⊕, c, …
│   ├── coordinates.py  # ECI↔ECEF↔LVLH, GMST, rotation helpers
│   └── time_.py        # JD / MJD / UTC conversions
└── tests/              # pytest suite (36 tests)
    ├── test_ekf_core.py
    ├── test_dynamics.py
    ├── test_measurements.py
    ├── test_fusion.py
    ├── test_applications.py
    └── test_utils.py
```

## Quick Start

```bash
pip install -e .
python examples/multi_sensor_fusion_demo.py
```

Expected output (varies due to RNG):

```
step   0  position error: 234.86 m
step  10  position error: 471.34 m
step  20  position error: 948.85 m
step  30  position error: 2309.19 m
step  40  position error: 1238.29 m

mean position error: 1075.94 m
final position error: 1558.91 m
```

## Usage

```python
import numpy as np
from kalman.applications.multi_sensor_fusion import MultiSensorFusion

# Initial LEO state (ECI)
r0 = np.array([7000e3, 0.0, 0.0])
v0 = np.array([0.0, 7546.0, 0.0])
x0 = np.concatenate([r0, v0])
P0 = np.diag([1e6, 1e6, 1e6, 1e2, 1e2, 1e2])

# Create fusion system with radar + camera
msf = MultiSensorFusion(x0, P0)
msf.add_radar(np.diag([1e3, 1e-6, 1e-6]))
msf.add_camera(np.diag([1e-6, 1e-6]))

# Run
msf.predict_all(10.0)
msf.update_radar(np.array([rho, az, el]))
msf.update_camera(np.array([az, el]))

fused = msf.fuse()  # State with x, P
```

## Running Tests

```bash
pip install -e ".[dev]"
python -m pytest kalman/tests/ -v
```

## Models

### Dynamics

| Model | State | Equations | Use Case |
|---|---|---|---|
| `two_body` | 6 (r, v) | RK4 integration | Orbit determination, tracking |
| `j2_perturbed` | 6+ | + J2 / drag / SRP | High-precision OD |
| `relative_cw` | 6 (dr, dv) | Closed-form STM | Circular-orbit rendezvous |
| `relative_hcw` | 6 (dr, dv) | Numeric STM | Elliptical-orbit rendezvous |

### Measurements

| Model | z | h(x) | Typical Sensor |
|---|---|---|---|
| `range_bearing` | 3 | [‖r‖, az, el] | Radar, LIDAR |
| `range_rate` | 1 | (r·v)/‖r‖ | Doppler radar |
| `angles_only` | 2 | [az, el] | Optical camera |
| `gps` | 3 or 6 | ECI→ECEF transform | GPS receiver |
| `relative_pose` | 3 or 6 | Identity on dr, dv | Docking camera |

### Fusion

Three layers are implemented in `fusion/`:

1. **LocalFilter** — wraps a single EKF with one dynamics + one measurement model
2. **CovarianceIntersection** — fuses two or more State estimates while handling unknown cross-correlations; optimal weight α found via golden-section search on log-det
3. **DecentralizedFusion** — orchestrator holding N LocalFilters, each predicting and updating independently, then fusing via CI

## References

### Books

| Title | Author(s) | Area |
|---|---|---|
| *Optimal State Estimation* | Dan Simon | EKF theory, Joseph form, IEKF |
| *Fundamentals of Spacecraft Attitude Determination and Control* | Markley & Crassidis | Spacecraft navigation, sensor fusion |
| *Orbital Mechanics for Engineering Students* | Howard Curtis | Two-body problem, CW equations, coordinate frames |
| *Statistical Orbit Determination* | Tapley, Schutz, Born | Orbit determination with EKF / UKF |
| *Estimation with Applications to Tracking and Navigation* | Bar-Shalom, Li, Kirubarajan | Tracking, sensor fusion, recursive estimation |
| *Kalman and Bayesian Filters in Python* | Roger Labbé ([free online](https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python)) | Practical EKF with Python |

### Key Papers

| Title | Author(s) | Year | Relevance |
|---|---|---|---|
| [A New Approach to Linear Filtering and Prediction Problems](https://doi.org/10.1115/1.3662552) | R. E. Kalman | 1960 | Original Kalman filter |
| [A Non-divergent Filter Formulation](https://doi.org/10.2514/6.1968-115) | R. H. Battin | 1968 | Joseph-form covariance update |
| [Generalized Information Representation and CI for Decentralized Estimation](https://doi.org/10.1109/CDC.2002.1184291) | Julier & Uhlmann | 2002 | Covariance intersection theory |
| [A Modular and Multi-Sensor Fusion Approach Based on an EKF](https://doi.org/10.3929/ethz-a-010008265) | Lynen et al. | 2013 | Decentralized EKF architecture |

### Reference Implementations

| Project | Language | Description |
|---|---|---|
| [ethzasl\_msf](https://github.com/ethz-asl/ethzasl_msf) | C++ / ROS | Modular EKF sensor fusion (ETH Zurich) |
| [Orekit](https://www.orekit.org/) | Java | Orbit determination with EKF / UKF / ESKF |
| [Kalman-and-Bayesian-Filters-in-Python](https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python) | Python | Book-quality KF / EKF / UKF reference code |
| [NAIF SPICE](https://naif.jpl.nasa.gov/naif/) | C / Python | Coordinate frames, time, ephemerides |
