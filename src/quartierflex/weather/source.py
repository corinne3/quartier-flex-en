"""
source.py: fetch the weather, OBSERVED and FORECAST.

Two datasets, both free and keyless (Open-Meteo):

1. OBSERVED weather ("archive" API): what actually happened.
   -> used to compute the panels' actual production.
2. FORECAST weather ("historical forecast" API): what the weather models
   predicted the day before. -> this is what a forecasting AI is allowed
   to use (it does not know the future!).

Columns produced (one row per hour, index = START of the hour, Paris time zone):
    ghi_wm2    global horizontal irradiance (W/m²)
    dni_wm2    direct normal irradiance, facing the sun (W/m²)
    dhi_wm2    diffuse irradiance, from the whole sky (W/m²)
    temp_c     air temperature (°C)
    cloud_pct  cloud cover (%)

Fallback: if offline, SYNTHETIC weather (clear sky × random clouds),
always flagged in df.attrs["source"].
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import CACHE_DIR, Site
from .sun import clear_sky_ghi, solar_position

log = logging.getLogger(__name__)

HOURLY = "temperature_2m,shortwave_radiation,direct_normal_irradiance,diffuse_radiation,cloud_cover"
URLS = {
    "observed": "https://archive-api.open-meteo.com/v1/archive",
    "forecast": "https://historical-forecast-api.open-meteo.com/v1/forecast",
}
RENAME = {
    "temperature_2m": "temp_c", "shortwave_radiation": "ghi_wm2",
    "direct_normal_irradiance": "dni_wm2", "diffuse_radiation": "dhi_wm2", "cloud_cover": "cloud_pct",
}


def _index(t0: pd.Timestamp, t1: pd.Timestamp) -> pd.DatetimeIndex:
    return pd.date_range(t0, t1, freq="h", inclusive="left", tz="Europe/Paris")


def _download(kind: str, site: Site, t0: pd.Timestamp, t1: pd.Timestamp) -> pd.DataFrame:
    import httpx

    r = httpx.get(URLS[kind], params={
        "latitude": site.lat, "longitude": site.lon,
        "start_date": f"{(t0 - pd.Timedelta('1D')):%Y-%m-%d}", "end_date": f"{(t1 + pd.Timedelta('1D')):%Y-%m-%d}",
        "hourly": HOURLY, "timezone": "UTC",
    }, timeout=60.0)
    r.raise_for_status()
    js = r.json()["hourly"]
    df = pd.DataFrame({RENAME[k]: js[k] for k in RENAME if k in js})
    # Open-Meteo gives the average of the PREVIOUS hour: the 10:00 value
    # covers 09:00-10:00. We shift the index back to the START of the hour.
    ts = pd.to_datetime(js["time"]).tz_localize("UTC") - pd.Timedelta("1h")
    df.index = ts.tz_convert("Europe/Paris")
    return df


def synthetic_weather(site: Site, t0: pd.Timestamp, t1: pd.Timestamp, seed: int = 0) -> pd.DataFrame:
    """Plausible weather: clear sky × random cloud factor (persistent from one day to the next)."""
    rng = np.random.default_rng(seed)
    idx = _index(t0, t1)
    sp = solar_position(idx + pd.Timedelta("30min"), site.lat, site.lon)
    cs = clear_sky_ghi(sp["zenith_deg"].to_numpy())
    n_days = len(idx) // 24 + 2
    daily_cloud = np.clip(np.convolve(rng.beta(1.2, 1.6, n_days), [0.3, 0.4, 0.3], "same"), 0, 1)
    day_i = ((idx - idx[0]) / pd.Timedelta("1D")).astype(int)
    cloud = np.clip(daily_cloud[day_i] + rng.normal(0, 0.12, len(idx)), 0, 1)
    ghi = cs * (1 - 0.75 * cloud ** 2)
    doy = idx.dayofyear.to_numpy()
    temp = 13 - 9 * np.cos(2 * np.pi * (doy - 15) / 365) + 5 * np.sin((idx.hour.to_numpy() - 9) / 24 * 2 * np.pi)
    cz = np.cos(np.radians(sp["zenith_deg"].to_numpy()))
    dhi = ghi * (0.2 + 0.7 * cloud)
    dni = np.where(cz > 0.05, (ghi - dhi) / np.maximum(cz, 0.05), 0)
    df = pd.DataFrame({"temp_c": temp, "ghi_wm2": ghi, "dni_wm2": np.clip(dni, 0, 1000),
                       "dhi_wm2": dhi, "cloud_pct": cloud * 100}, index=idx)
    df.attrs["source"] = "SYNTHETIC (fallback, do not publish)"
    return df


def degrade_to_forecast(obs: pd.DataFrame, seed: int = 1) -> pd.DataFrame:
    """
    Fallback for FORECAST weather: observed + realistic forecast error
    (clouds are what forecasts miss the most).
    """
    rng = np.random.default_rng(seed)
    n = len(obs)
    err = np.convolve(rng.normal(0, 1, n + 12), np.ones(12) / np.sqrt(12), "valid")[:n]
    fc = obs.copy()
    fc["cloud_pct"] = np.clip(obs["cloud_pct"] + 18 * err, 0, 100)
    factor = np.clip(1 - 0.25 * err, 0.3, 1.5)
    for c in ("ghi_wm2", "dni_wm2", "dhi_wm2"):
        fc[c] = obs[c] * factor
    fc["temp_c"] = obs["temp_c"] + rng.normal(0, 1.2, n)
    fc.attrs["source"] = "SIMULATED forecast (observed + error)"
    return fc


def load_weather(site: Site, start: str, days: int, history_days: int = 0, kind: str = "observed",
                 offline: bool = False, seed: int = 0, cache_dir: Path = CACHE_DIR) -> pd.DataFrame:
    """
    Hourly weather over [start - history_days, start + days[.
    kind = "observed" or "forecast".
    """
    t0 = pd.Timestamp(start).normalize() - pd.Timedelta(days=history_days)
    t1 = pd.Timestamp(start).normalize() + pd.Timedelta(days=days)
    idx = _index(t0, t1)
    cache = cache_dir / f"weather_{kind}_{site.lat}_{site.lon}_{t0:%Y%m%d}_{t1:%Y%m%d}.csv"
    df, source = None, f"Open-Meteo ({'archive' if kind == 'observed' else 'archived forecasts'})"

    if cache.exists():
        df = pd.read_csv(cache, index_col=0)
        df.index = pd.to_datetime(df.index, utc=True).tz_convert("Europe/Paris")
        source += " (cache)"
    elif not offline:
        try:
            df = _download(kind, site, t0, t1)
            cache.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(cache)
        except Exception as e:
            log.warning("Weather %s unavailable (%s): fallback.", kind, e)

    if df is None:
        obs = synthetic_weather(site, t0, t1, seed=seed)
        if kind == "forecast":
            return degrade_to_forecast(obs, seed=seed + 1)
        return obs

    df = df.reindex(idx).interpolate(limit_direction="both")
    for c in RENAME.values():
        if c not in df:
            df[c] = 12.0 if c == "temp_c" else 0.0
    rad = ["ghi_wm2", "dni_wm2", "dhi_wm2", "cloud_pct"]
    df[rad] = df[rad].clip(lower=0)      # no negative irradiance (temperature, however, can be negative)
    df.attrs["source"] = source
    return df


def load_weather_pair(site: Site, start: str, days: int, history_days: int, offline: bool = False,
                      seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(observed, forecast). If archived forecasts are missing: simulated forecast."""
    obs = load_weather(site, start, days, history_days, "observed", offline, seed)
    try:
        fc = load_weather(site, start, days, history_days, "forecast", offline or "SYNTH" in obs.attrs["source"], seed)
        if "SYNTH" in fc.attrs.get("source", "") and "SYNTH" not in obs.attrs["source"]:
            fc = degrade_to_forecast(obs, seed + 1)
    except Exception:
        fc = degrade_to_forecast(obs, seed + 1)
    return obs, fc
