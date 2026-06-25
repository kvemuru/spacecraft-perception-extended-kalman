"""Julian Date / MJD / datetime conversions."""

import numpy as np
from datetime import datetime, timedelta, timezone

JD_J2000 = 2451545.0
MJD_J2000 = 51544.5


def datetime_to_jd(dt: datetime) -> float:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    mjd = datetime_to_mjd(dt)
    return mjd + 2400000.5


def datetime_to_mjd(dt: datetime) -> float:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    epoch = datetime(1858, 11, 17, tzinfo=timezone.utc)
    return (dt - epoch).total_seconds() / 86400.0


def jd_to_datetime(jd: float) -> datetime:
    mjd = jd - 2400000.5
    return mjd_to_datetime(mjd)


def mjd_to_datetime(mjd: float) -> datetime:
    epoch = datetime(1858, 11, 17, tzinfo=timezone.utc)
    return epoch + timedelta(days=mjd)


def jd_to_mjd(jd: float) -> float:
    return jd - 2400000.5


def mjd_to_jd(mjd: float) -> float:
    return mjd + 2400000.5


def jd_diff_seconds(jd1: float, jd2: float) -> float:
    return (jd1 - jd2) * 86400.0


def seconds_to_jd(sec: float) -> float:
    return sec / 86400.0
