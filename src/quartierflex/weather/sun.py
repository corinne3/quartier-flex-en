"""
sun.py: where is the sun in the sky, hour by hour?

Essential to compute what a tilted panel receives: the same
sunshine does not give the same production at noon (sun high, facing the
panel) as at 6 pm (sun low, grazing).

Method: simplified NOAA astronomical formulas (accuracy of
about 0.5°, more than enough here). No external library.

Outputs:
- zenith_deg  : angle between the sun and the vertical (0 = at zenith, > 90 = night)
- azimuth_deg : direction of the sun (0 = north, 90 = east, 180 = south, 270 = west)
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def solar_position(times: pd.DatetimeIndex, lat: float, lon: float) -> pd.DataFrame:
    """times: hourly index WITH time zone (converted to UTC internally)."""
    t = times.tz_convert("UTC")
    doy = t.dayofyear.to_numpy()
    hour = t.hour.to_numpy() + t.minute.to_numpy() / 60.0

    # Year angle (radians)
    g = 2 * np.pi / 365.0 * (doy - 1 + (hour - 12) / 24)
    # Equation of time (minutes) and declination (radians)
    eqtime = 229.18 * (0.000075 + 0.001868 * np.cos(g) - 0.032077 * np.sin(g)
                       - 0.014615 * np.cos(2 * g) - 0.040849 * np.sin(2 * g))
    decl = (0.006918 - 0.399912 * np.cos(g) + 0.070257 * np.sin(g) - 0.006758 * np.cos(2 * g)
            + 0.000907 * np.sin(2 * g) - 0.002697 * np.cos(3 * g) + 0.00148 * np.sin(3 * g))
    # True solar time -> hour angle
    true_solar_min = hour * 60 + eqtime + 4 * lon
    ha = np.radians(true_solar_min / 4 - 180)
    phi = np.radians(lat)

    cos_z = np.sin(phi) * np.sin(decl) + np.cos(phi) * np.cos(decl) * np.cos(ha)
    cos_z = np.clip(cos_z, -1, 1)
    zen = np.arccos(cos_z)
    # Azimuth (from north, clockwise)
    sin_z = np.maximum(np.sin(zen), 1e-9)
    cos_az = (np.sin(decl) - np.sin(phi) * cos_z) / (np.cos(phi) * sin_z)
    az = np.degrees(np.arccos(np.clip(cos_az, -1, 1)))
    az = np.where(ha > 0, 360 - az, az)
    return pd.DataFrame({"zenith_deg": np.degrees(zen), "azimuth_deg": az}, index=times)


def clear_sky_ghi(zenith_deg: np.ndarray) -> np.ndarray:
    """
    Horizontal irradiance under a perfectly clear sky (Haurwitz model), in W/m².
    Used as a reference: "today's weather is worth X % of a clear sky".
    """
    cz = np.cos(np.radians(zenith_deg))
    return np.where(cz > 0.01, 1098.0 * cz * np.exp(-0.057 / np.maximum(cz, 0.01)), 0.0)
