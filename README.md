# kalman — Modular Extended Kalman Filter for Space Perception

A modular Extended Kalman Filter framework for spacecraft perception, organized around four core space domain awareness tasks: absolute orbit determination, relative navigation & docking, space object tracking, and multi-sensor fusion. The design separates the EKF into four pluggable layers — **dynamics models** (how the state evolves), **measurement models** (how sensors observe the state), a **generic EKF core** (predict/update cycle with Joseph-form covariance updates and iterated IEKF refinement), and a **decentralized fusion layer** (per-sensor local EKFs fused via covariance intersection to handle unknown cross-correlations). Each layer defines abstract interfaces so users can swap in new motion models or sensor types without touching the filter logic. Advanced estimation via **IMM** (Interacting Multiple Model) handles maneuvering targets. Pre-built application wrappers wire these layers together for common use cases, while utility modules handle coordinate frames (ECI↔ECEF↔LVLH), time conversions, and physical constants.

## Architecture

```
kalman/
├── ekf_core/             # Generic EKF engine
│   ├── interfaces.py     # ABCs: DynamicsModel, MeasurementModel
│   ├── state.py          # State dataclass (x, P, timestamp)
│   └── ekf.py            # predict(), update() (Joseph form), IEKF, NEES/NIS, gating
├── dynamics/             # Pluggable motion models
│   ├── two_body.py       # Keplerian: r̈ = -μr/‖r‖³, RK4 + analytic STM
│   ├── j2_perturbed.py   # J2 + atmospheric drag (exponential density) & SRP
│   ├── atmosphere.py     # Multi-band exponential density model (0–1000 km)
│   ├── relative_cw.py    # Clohessy-Wiltshire (circular, closed-form STM)
│   └── relative_hcw.py   # Hill-CW (elliptical, numeric STM)
├── measurements/         # Pluggable sensor models
│   ├── range_bearing.py            # Radar: [ρ, az, el]
│   ├── range_rate.py               # Doppler: [ρ̇]
│   ├── range_bearing_from_station.py  # Ground-station radar
│   ├── angles_only.py              # Optical: [az, el]
│   ├── gps.py                      # GPS position/velocity (ECI↔ECEF)
│   └── relative_pose.py            # Relative position / full 6-DOF pose
├── fusion/               # Decentralized fusion
│   ├── local_filter.py              # Per-sensor EKF wrapper
│   ├── covariance_intersection.py   # Optimal CI: pairwise, sequential, batch
│   └── decentralized.py            # Master orchestrator for N local EKFs
├── estimation/           # Advanced estimation filters
│   └── imm.py            # Interacting Multiple Model (IMM) filter
├── applications/         # Use-case wrappers
│   ├── orbit_determination.py
│   ├── relative_nav_docking.py
│   ├── space_object_tracking.py
│   └── multi_sensor_fusion.py
├── utils/
│   ├── constants.py     # μ⊕, J₂, R⊕, ω⊕, c, …
│   ├── coordinates.py   # ECI↔ECEF↔LVLH, GMST, rotation helpers
│   └── time_.py         # JD / MJD / UTC conversions
├── tests/                # pytest suite (51 tests)
│   ├── test_ekf_core.py
│   ├── test_dynamics.py  # incl. exponential atm, drag, multi-altitude J2
│   ├── test_measurements.py
│   ├── test_fusion.py    # incl. batch CI
│   ├── test_imm.py       # IMM predict/update, mode convergence
│   ├── test_applications.py
│   └── test_utils.py
├── examples/
│   ├── multi_sensor_fusion_demo.py  # Radar + camera + GPS fusion demo
│   └── compare_errors_demo.py       # Step-by-step sensor config comparison
├── scripts/
│   └── gen_api_docs.py   # Auto-generate API_REFERENCE.md
├── API_REFERENCE.md       # Auto-generated API docs
└── README.md
```

## Quick Start

```bash
pip install -e .
python examples/multi_sensor_fusion_demo.py
```

Expected output (varies due to RNG):

```
step   0  fused:   295.7 m  gps-only:   17.7 m
step  40  fused:   218.2 m  gps-only:    3.5 m
step  80  fused:    44.8 m  gps-only:    3.0 m

Mean position error — fused: 170.1 m  gps-only: 3.3 m
```

### Sensor Comparison Demo

Compare four configurations side-by-side:

```bash
python examples/compare_errors_demo.py
```

```
 step    GPS-only    Radar-only     Radar+GPS         Fused
-----------------------------------------------------------
    0       15.58         296.5          18.4         295.6
   25        3.58         188.3           9.1         187.2
   50        3.50         200.4           2.4         199.2
   75        3.20         170.9           2.6         170.2
  100        3.90         193.7           1.9         192.9
  125        3.63          98.2           3.3          97.7
  150        1.97          71.9           3.4          71.6
  175        1.33         106.9           2.5         106.5

GPS-only         mean:      3.6 m  final:      1.7 m
Radar-only       mean:    205.3 m  final:    164.0 m
Radar+GPS EKF    mean:      3.3 m  final:      2.3 m
Fused            mean:    204.2 m  final:    163.4 m
```

GPS delivers ~3 m accuracy regardless of other sensors. Radar-only converges to ~170 m bounded error via range/bearing. Centralized EKF (Radar+GPS) slightly outperforms GPS alone. Decentralized CI fusion is conservative — it guarantees consistency under unknown cross-correlations but gets pulled by the overconfident radar filter's small covariance, producing fused errors similar to radar-only.

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
| `j2_perturbed` | 6 | J2 + exponential drag + SRP | High-precision OD |
| `atmosphere` | — | Multi-band exponential ρ(h) | Drag computation (0–1000 km) |
| `relative_cw` | 6 (dr, dv) | Closed-form STM | Circular-orbit rendezvous |
| `relative_hcw` | 6 (dr, dv) | Numeric STM | Elliptical-orbit rendezvous |

### Measurements

| Model | z | h(x) | Typical Sensor |
|---|---|---|---|
| `range_bearing` | 3 | [‖r‖, az, el] | Radar, LIDAR |
| `range_bearing_from_station` | 3 | [‖r−r₀‖, az, el] | Ground station radar |
| `range_rate` | 1 | (r·v)/‖r‖ | Doppler radar |
| `angles_only` | 2 | [az, el] | Optical camera |
| `gps` | 3 or 6 | ECI→ECEF transform | GPS receiver |
| `relative_pose` | 3 or 6 | Identity on dr, dv | Docking camera |

### Estimation

| Filter | Description |
|---|---|
| `EKF` | Core predict/update, Joseph form, IEKF, innovation gating |
| `IMM` | Interacting Multiple Model for maneuvering targets |
| `LocalFilter` | Per-sensor EKF wrapper for decentralized fusion |
| `CovarianceIntersection` | Pairwise, sequential, or batch fusion (optimal α) |
| `DecentralizedFusion` | Orchestrator holding N LocalFilters |

## References

### Books

| Title | Author(s) | Area |
|---|---|---|
| [*Optimal State Estimation*](https://onlinelibrary.wiley.com/doi/book/10.1002/0470045345) | Dan Simon | EKF theory, Joseph form, IEKF |
| [*Fundamentals of Spacecraft Attitude Determination and Control*](https://link.springer.com/book/10.1007/978-1-4939-0802-8) | Markley & Crassidis | Spacecraft navigation, sensor fusion |
| [*Orbital Mechanics for Engineering Students*](https://shop.elsevier.com/books/orbital-mechanics-for-engineering-students/curtis/978-0-443-29015-2) | Howard Curtis | Two-body problem, CW equations, coordinate frames |
| [*Statistical Orbit Determination*](https://www.amazon.com/Statistical-Orbit-Determination-Bob-Schutz/dp/0126836302) | Tapley, Schutz, Born | Orbit determination with EKF / UKF |
| [*Estimation with Applications to Tracking and Navigation*](https://onlinelibrary.wiley.com/doi/book/10.1002/0471221279) | Bar-Shalom, Li, Kirubarajan | Tracking, sensor fusion, recursive estimation |
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
