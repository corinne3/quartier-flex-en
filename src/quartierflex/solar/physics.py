"""
physics.py: how much do the panels produce, hour by hour? (physics model)

Calculation chain, from weather to kWh:

  1. Sun position (weather/sun.py)
  2. Irradiance received BY THE tilted PANEL (POA, "plane of array"):
        POA = direct × cos(angle of incidence)       <- the sun "head-on"
            + diffuse × (1 + cos tilt) / 2           <- light from the sky
            + global × albedo × (1 - cos tilt) / 2   <- light reflected by the ground
  3. Cell temperature (hotter = lower efficiency):
        T_cell = T_air + (NOCT - 20) / 800 × POA
  4. Power:
        P = kWp × POA / 1000 × (1 + γ × (T_cell - 25)) × PR
     with γ ≈ -0.4 %/°C and PR (performance ratio) ≈ 0.86 for losses
     (inverter, cables, average soiling...).

Output unit: kWh produced during each hour (= average kW over the hour).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import PVConfig, Site
from ..weather.sun import solar_position


def _erbs(ghi: np.ndarray, zenith_deg: np.ndarray, doy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Splits direct / diffuse when only global is known (Erbs model)."""
    cz = np.cos(np.radians(zenith_deg))
    i0 = 1367 * (1 + 0.033 * np.cos(2 * np.pi * doy / 365))
    kt = np.where(cz > 0.05, ghi / (i0 * np.maximum(cz, 0.05)), 0)
    kt = np.clip(kt, 0, 1)
    fd = np.where(kt <= 0.22, 1 - 0.09 * kt,
                  np.where(kt <= 0.8, 0.9511 - 0.1604 * kt + 4.388 * kt**2 - 16.638 * kt**3 + 12.336 * kt**4, 0.165))
    dhi = ghi * fd
    dni = np.where(cz > 0.05, (ghi - dhi) / np.maximum(cz, 0.05), 0)
    return np.clip(dni, 0, 1100), dhi


def plane_of_array(weather: pd.DataFrame, site: Site, pv: PVConfig) -> pd.DataFrame:
    sp = solar_position(weather.index + pd.Timedelta("30min"), site.lat, site.lon)  # middle of the hour
    zen = sp["zenith_deg"].to_numpy()
    az = sp["azimuth_deg"].to_numpy()
    ghi = weather["ghi_wm2"].to_numpy()
    dni = weather["dni_wm2"].to_numpy()
    dhi = weather["dhi_wm2"].to_numpy()
    if np.nanmax(dni) <= 1 and np.nanmax(ghi) > 50:        # no direct provided -> Erbs
        dni, dhi = _erbs(ghi, zen, weather.index.dayofyear.to_numpy())

    tilt, paz = np.radians(pv.tilt_deg), np.radians(pv.azimuth_deg)
    z, a = np.radians(zen), np.radians(az)
    cos_aoi = np.cos(z) * np.cos(tilt) + np.sin(z) * np.sin(tilt) * np.cos(a - paz)
    beam = dni * np.clip(cos_aoi, 0, None)
    sky = dhi * (1 + np.cos(tilt)) / 2
    ground = ghi * pv.albedo * (1 - np.cos(tilt)) / 2
    poa = np.where(zen < 90, beam + sky + ground, 0.0)
    return pd.DataFrame({"poa_wm2": poa, "zenith_deg": zen, "azimuth_deg": az}, index=weather.index)


def pv_production_kwh(weather: pd.DataFrame, site: Site, pv: PVConfig) -> pd.Series:
    """Hourly production (kWh) of the panel array with peak power pv.kwc."""
    poa = plane_of_array(weather, site, pv)["poa_wm2"].to_numpy()
    t_cell = weather["temp_c"].to_numpy() + (pv.noct_c - 20) / 800 * poa
    p = pv.kwc * poa / 1000 * (1 + pv.gamma_per_c * (t_cell - 25)) * pv.performance_ratio
    p = np.clip(p, 0, pv.kwc)          # the inverter caps at peak power
    return pd.Series(p, index=weather.index, name="pv_kwh")
