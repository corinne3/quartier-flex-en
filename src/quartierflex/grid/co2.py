"""
co2.py: the national electricity grid, hour by hour (RTE éCO2mix).

RTE is the French transmission system operator. Reused from the Bilan Net project. Two pieces of information:
- consumption_mw: France's consumption (used to approximate Tempo days);
- co2_g_per_kwh : AVERAGE carbon intensity of a grid kWh at that hour.

When the building draws from the grid, it is assigned this CO2. When it consumes
its own solar power, we count 0 in operation (the embodied energy of the
panels is a separate parameter, see assessment/kpi.py).

Source: ODRÉ API (Open Data Réseaux Énergies), dataset eco2mix-national-cons-def.
Fallback: synthetic data, flagged as such.
"""

from __future__ import annotations

import io
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import CACHE_DIR

log = logging.getLogger(__name__)
URL = "https://odre.opendatasoft.com/api/explore/v2.1/catalog/datasets/eco2mix-national-cons-def/exports/csv"


def _download(t0: pd.Timestamp, t1: pd.Timestamp) -> pd.DataFrame:
    import httpx

    r = httpx.get(URL, params={
        "select": "date_heure,consommation,taux_co2",
        "where": f"date_heure >= date'{t0:%Y-%m-%d}' AND date_heure < date'{t1:%Y-%m-%d}'",
        "order_by": "date_heure", "delimiter": ";", "timezone": "UTC",
    }, timeout=90.0, follow_redirects=True)
    r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text), sep=";")
    ts = pd.to_datetime(df["date_heure"], utc=True, errors="coerce").dt.tz_convert("Europe/Paris")
    out = pd.DataFrame({"consumption_mw": pd.to_numeric(df["consommation"], errors="coerce").to_numpy(),
                        "co2_g_per_kwh": pd.to_numeric(df["taux_co2"], errors="coerce").to_numpy()}, index=ts)
    out = out[~out.index.isna()].sort_index()
    return out.groupby(out.index.floor("h")).mean()


def synthetic_grid(index: pd.DatetimeIndex, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    h = index.hour.to_numpy()
    doy = index.dayofyear.to_numpy()
    season = 1 + 0.25 * np.cos(2 * np.pi * (doy - 15) / 365)
    shape = 0.8 + 0.15 * np.exp(-((h - 9) ** 2) / 8) + 0.22 * np.exp(-((h - 19) ** 2) / 6) - 0.1 * np.exp(-((h - 4) ** 2) / 6)
    cons = 55_000 * season * shape * rng.lognormal(0, 0.03, len(index))
    load = (cons - cons.min()) / (cons.max() - cons.min() + 1e-9)
    co2 = 15 + 60 * load**2 + rng.normal(0, 4, len(index))
    df = pd.DataFrame({"consumption_mw": cons, "co2_g_per_kwh": np.clip(co2, 8, None)}, index=index)
    df.attrs["source"] = "SYNTHETIC (fallback, do not publish)"
    return df


def approx_tempo(consumption_mw: pd.Series) -> pd.Series:
    """APPROXIMATE Tempo color from the day's consumption (the real calendar is published by RTE)."""
    daily = consumption_mw.groupby(consumption_mw.index.date).mean()
    wd = pd.to_datetime(daily.index).dayofweek < 5
    col = pd.Series("BLUE", index=daily.index)
    col[daily.to_numpy() > 64_000] = "WHITE"
    col[(daily.to_numpy() > 72_000) & wd] = "RED"
    return pd.Series(col.reindex(consumption_mw.index.date).to_numpy(), index=consumption_mw.index)


def load_grid(start: str, days: int, history_days: int = 0, offline: bool = False, seed: int = 0,
              cache_dir: Path = CACHE_DIR) -> pd.DataFrame:
    t0 = pd.Timestamp(start).normalize() - pd.Timedelta(days=history_days)
    t1 = pd.Timestamp(start).normalize() + pd.Timedelta(days=days)
    idx = pd.date_range(t0, t1, freq="h", inclusive="left", tz="Europe/Paris")
    cache = cache_dir / f"rte_{t0:%Y%m%d}_{t1:%Y%m%d}.csv"
    df, source = None, "RTE éCO2mix (ODRÉ)"
    if cache.exists():
        df = pd.read_csv(cache, index_col=0)
        df.index = pd.to_datetime(df.index, utc=True).tz_convert("Europe/Paris")
        source += " (cache)"
    elif not offline:
        try:
            df = _download(t0 - pd.Timedelta("1D"), t1 + pd.Timedelta("1D"))
            if len(df) < 24:
                raise ValueError("too little data")
            cache.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(cache)
        except Exception as e:
            log.warning("RTE unavailable (%s): synthetic data.", e)
            df = None
    if df is None:
        df = synthetic_grid(idx, seed)
        source = df.attrs["source"]
    df = df.reindex(idx).interpolate(limit_direction="both")
    df["tempo"] = approx_tempo(df["consumption_mw"])
    df.attrs["source"] = source
    return df
