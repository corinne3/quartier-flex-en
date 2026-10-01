"""
learning.py: learning from consumption measurements (the occupant-side AI).

Two learning tasks
------------------
1. RECOGNIZE HABITS (unsupervised learning, k-means)
   Each curve is cut into days, normalized (the shape of the day,
   not its volume), and similar days are grouped together.
   Each home is assigned the group of the majority of its days.
   The algorithm does NOT know the true types: we then check whether it
   recovers them (adjusted Rand index: 1 = perfect, 0 = chance).
   Purpose: know who consumes during the day (ideal for solar) and who
   consumes in the evening (ideal for the battery), without a questionnaire.

2. FORECAST the building's CONSUMPTION for the next 24 h
   (gradient boosting), compared with two naive forecasts:
     - "same as yesterday" (D-1);
     - "same as the same day last week" (D-7).
   Features: hour, day of week, season, consumption at D-1 and D-7,
   mean of the last 7 days at the same hour, forecast temperature. No access to the future (lags ≥ 24 h).

Privacy (worth telling the jury): this processing can run LOCALLY
in the building; only aggregated data leaves it.
"""

from __future__ import annotations

from contextlib import nullcontext

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import adjusted_rand_score


# =============================================================================
# 1. Habits
# =============================================================================
def daily_profiles(loads: pd.DataFrame) -> pd.DataFrame:
    """One row per (home, day), 24 columns = share of the day's consumption at each hour."""
    rows = []
    for col in loads.columns:
        s = loads[col]
        for day, g in s.groupby(s.index.date):
            if len(g) != 24 or g.sum() <= 0:
                continue
            rows.append([col, day] + list(g.to_numpy() / g.sum()))
    return pd.DataFrame(rows, columns=["home", "day"] + list(range(24)))


def _name_cluster(profile: np.ndarray) -> str:
    day = profile[9:17].sum()
    evening = profile[18:23].sum()
    night = profile[list(range(0, 6)) + [23]].sum()
    if day > 0.40:
        return "home during the day"
    if night > 0.25:
        return "night owl"
    if evening > 0.40:
        return "evening peak"
    return "mixed profile"


def cluster_habits(loads: pd.DataFrame, true_types: dict | None = None, k: int = 4, seed: int = 0) -> dict:
    prof = daily_profiles(loads)
    X = prof[list(range(24))].to_numpy()
    km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(X)
    prof["group"] = km.labels_
    per_house = prof.groupby("home")["group"].agg(lambda s: int(s.mode().iloc[0]))
    names = {g: _name_cluster(km.cluster_centers_[g]) for g in range(k)}
    table = pd.DataFrame({"group": per_house, "habit": per_house.map(names)})
    out = {"table": table, "centres": pd.DataFrame(km.cluster_centers_.T, columns=[names[g] + f" ({g})" for g in range(k)]),
           "ari": np.nan}
    if true_types:
        table["type_real"] = [true_types[h] for h in table.index]
        out["ari"] = float(adjusted_rand_score(table["type_real"], table["group"]))
    return out


# =============================================================================
# 2. Consumption forecasting
# =============================================================================
def _features(total: pd.Series, index: pd.DatetimeIndex, temp: pd.Series | None) -> pd.DataFrame:
    X = pd.DataFrame(index=index)
    X["hour"] = index.hour
    X["dow"] = index.dayofweek
    doy = index.dayofyear
    X["doy_sin"], X["doy_cos"] = np.sin(2 * np.pi * doy / 365), np.cos(2 * np.pi * doy / 365)
    X["lag24"] = total.reindex(index - pd.Timedelta("24h")).to_numpy()
    X["lag168"] = total.reindex(index - pd.Timedelta("168h")).to_numpy()
    # Mean of the last 7 days at the same hour: smooths out the randomness of a single day
    X["avg7d"] = np.nanmean([total.reindex(index - pd.Timedelta(hours=24 * d)).to_numpy() for d in range(1, 8)], axis=0)
    X["temp"] = temp.reindex(index).to_numpy() if temp is not None else 12.0
    return X


class LoadForecaster:
    def __init__(self, max_iter: int = 200):
        self.model = HistGradientBoostingRegressor(max_iter=max_iter, learning_rate=0.08)

    def fit(self, total_hist: pd.Series, temp: pd.Series | None = None) -> "LoadForecaster":
        idx = total_hist.index[168:]
        X = _features(total_hist, idx, temp)
        self.model.fit(X, total_hist.loc[idx].to_numpy())
        return self

    def predict(self, total_known: pd.Series, index: pd.DatetimeIndex, temp: pd.Series | None = None) -> pd.Series:
        X = _features(total_known, index, temp).ffill().bfill()
        return pd.Series(np.clip(self.model.predict(X), 0, None), index=index, name="consumption_forecast_ml_kwh")


def nmae(pred: pd.Series, truth: pd.Series) -> float:
    return float(np.abs(pred - truth).mean() / truth.mean())


def compare_load_forecasts(total: pd.Series, sim_index: pd.DatetimeIndex, temp_fc: pd.Series | None = None,
                           meter=None) -> tuple[pd.DataFrame, dict]:
    """
    Trains on the history, forecasts the simulated period.
    For each hour of the period, the "ML" forecast only uses
    values from at least 24 h earlier (same constraint as the naive ones).
    """
    hist = total[total.index < sim_index[0]]
    with (meter.measure("setup") if meter else nullcontext()):
        model = LoadForecaster().fit(hist, temp_fc)
    with (meter.measure("shared") if meter else nullcontext()):
        ml = model.predict(total, sim_index, temp_fc)
    j1 = total.reindex(sim_index - pd.Timedelta("24h")).set_axis(sim_index).rename("load_d-1_kwh")
    j7 = total.reindex(sim_index - pd.Timedelta("168h")).set_axis(sim_index).rename("load_d-7_kwh")
    truth = total.loc[sim_index].rename("consumption_measured_kwh")
    table = pd.concat([truth, j1, j7, ml], axis=1)
    return table, {"d-1": nmae(j1, truth), "d-7": nmae(j7, truth), "ai_ml": nmae(ml, truth)}
