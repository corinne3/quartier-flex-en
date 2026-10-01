"""
forecast.py: the AI that forecasts tomorrow's solar production.

The problem
-----------
To control the battery well, you need to know TODAY how much the
panels will produce TOMORROW. We do not know tomorrow's actual weather:
only the weather FORECAST.

Three forecasts compared (from cheapest to most expensive in compute)
---------------------------------------------------------------------
1. Persistence      : "tomorrow = today". Near-zero cost.
2. Physics          : the physics model (physics.py) applied to the forecast weather.
3. AI (ML)          : a gradient boosting model that learns, from history, the relationship
                      between FORECAST weather and MEASURED production.
                      It corrects what physics ignores: soiling, weather forecast
                      bias, site specifics.

So that the model works whatever the size of the installation, it
learns production PER kWp, then we multiply by the kWp.

Metric: nMAE = mean absolute error / mean production (daylight hours).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from ..config import PVConfig, Site
from ..weather.sun import clear_sky_ghi
from .physics import plane_of_array, pv_production_kwh


def features(fc_weather: pd.DataFrame, site: Site, pv: PVConfig) -> pd.DataFrame:
    poa = plane_of_array(fc_weather, site, pv)
    one_kwc = PVConfig(**{**pv.__dict__, "kwc": 1.0})
    X = pd.DataFrame(index=fc_weather.index)
    X["hour"] = fc_weather.index.hour
    doy = fc_weather.index.dayofyear
    X["doy_sin"], X["doy_cos"] = np.sin(2 * np.pi * doy / 365), np.cos(2 * np.pi * doy / 365)
    for c in ("ghi_wm2", "dni_wm2", "dhi_wm2", "cloud_pct", "temp_c"):
        X[c] = fc_weather[c].to_numpy()
    X["poa_wm2"] = poa["poa_wm2"].to_numpy()
    X["clear_sky"] = clear_sky_ghi(poa["zenith_deg"].to_numpy())
    X["phys_per_kwc"] = pv_production_kwh(fc_weather, site, one_kwc).to_numpy()
    return X


class PVForecaster:
    """ML forecast of production (kWh/h)."""

    def __init__(self, site: Site, pv: PVConfig, max_iter: int = 300):
        self.site, self.pv = site, pv
        self.model = HistGradientBoostingRegressor(max_iter=max_iter, learning_rate=0.06, max_leaf_nodes=31)

    def fit(self, fc_weather_hist: pd.DataFrame, measured_hist_kwh: pd.Series) -> "PVForecaster":
        X = features(fc_weather_hist, self.site, self.pv)
        y = measured_hist_kwh.reindex(X.index).to_numpy() / self.pv.kwc
        ok = ~np.isnan(y)
        self.model.fit(X[ok], y[ok])
        return self

    def predict(self, fc_weather: pd.DataFrame) -> pd.Series:
        X = features(fc_weather, self.site, self.pv)
        p = np.clip(self.model.predict(X), 0, None) * self.pv.kwc
        p[X["clear_sky"].to_numpy() <= 0] = 0.0      # at night, zero, period
        return pd.Series(p, index=fc_weather.index, name="pv_forecast_ml_kwh")


def persistence(measured: pd.Series, index: pd.DatetimeIndex) -> pd.Series:
    """'Tomorrow = today' forecast: the value measured 24 h earlier."""
    return measured.reindex(index - pd.Timedelta("24h")).set_axis(index).fillna(0).rename("pv_persistence_kwh")


def nmae(pred: pd.Series, truth: pd.Series) -> float:
    day = truth > 0.01 * max(truth.max(), 1e-9)
    if not day.any():
        return float("nan")
    return float(np.abs(pred[day] - truth[day]).mean() / truth[day].mean())


def compare_forecasts(obs: pd.DataFrame, fc: pd.DataFrame, measured: pd.Series, sim_index: pd.DatetimeIndex,
                      site: Site, pv: PVConfig, meter=None) -> tuple[pd.DataFrame, dict]:
    """
    Trains on history (before sim_index), evaluates on sim_index.
    Returns (hourly table of forecasts, dict of nMAE errors).
    `meter` (optional) measures the AI's compute cost.
    """
    from contextlib import nullcontext

    hist = fc.index < sim_index[0]
    ctx = meter.measure("setup") if meter else nullcontext()
    with ctx:
        model = PVForecaster(site, pv).fit(fc[hist], measured[hist])
    ctx = meter.measure("shared") if meter else nullcontext()
    with ctx:
        ml = model.predict(fc.loc[sim_index])
    phys = pv_production_kwh(fc.loc[sim_index], site, pv).rename("pv_physics_kwh")
    pers = persistence(measured, sim_index)
    truth = measured.loc[sim_index]
    table = pd.concat([truth, pers, phys, ml], axis=1)
    scores = {"persistence": nmae(pers, truth), "physics": nmae(phys, truth), "ai_ml": nmae(ml, truth)}
    return table, scores
