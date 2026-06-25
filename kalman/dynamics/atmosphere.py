"""Exponential atmospheric density model for Earth orbit.

Multi-band exponential model with altitude-dependent scale heights
suitable for spacecraft drag computation from 0-1000 km altitude.
Reference densities and scale heights approximate the US Standard
Atmosphere (1976).
"""

import numpy as np

BANDS = [
    (0.0, 1.225, 8.0),
    (25_000.0, 4.0e-2, 7.5),
    (50_000.0, 1.0e-3, 8.0),
    (80_000.0, 2.0e-5, 6.5),
    (100_000.0, 5.0e-7, 18.0),
    (150_000.0, 2.0e-9, 25.0),
    (200_000.0, 3.5e-10, 32.0),
    (300_000.0, 2.5e-12, 45.0),
    (500_000.0, 1.0e-13, 65.0),
    (800_000.0, 1.0e-15, 100.0),
]


def exponential_atm_density(h: float) -> float:
    if h <= 0.0:
        return BANDS[0][1]
    if h >= BANDS[-1][0]:
        val = BANDS[-1][1] * np.exp(-(h - BANDS[-1][0]) / BANDS[-1][2])
        return max(val, 1e-30)
    for i in range(len(BANDS) - 1):
        h0, rho0, H = BANDS[i]
        h1, _, _ = BANDS[i + 1]
        if h0 <= h < h1:
            return rho0 * np.exp(-(h - h0) / H)
    return 1e-30
