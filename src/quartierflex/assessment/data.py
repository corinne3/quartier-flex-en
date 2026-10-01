"""
data.py: gathering ALL the inputs of a simulation into a single object.

This is the "meeting point" of modules 1 to 6:
    weather  -> observed and forecast weather
    solar    -> true and measured production
    usage    -> consumption of each home
    grid     -> CO2, Tempo colors, purchase and resale prices

Time window:
    [start - history ; start + duration + 1 day[
    - the history is used to train the AI models (never the future);
    - the extra day lets the optimizers "see" 24 h ahead
      up to the last simulated hour.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from ..config import Scenario
from ..weather import load_weather_pair
from ..grid import buy_price, load_grid, sell_price
from ..solar import measured_production, pv_production_kwh
from ..usage import make_building


@dataclass
class Inputs:
    scenario: Scenario
    index: pd.DatetimeIndex          # the whole window
    sim_index: pd.DatetimeIndex      # the evaluated period
    weather: pd.DataFrame            # observed
    weather_fc: pd.DataFrame         # forecast
    pv_true: pd.Series               # real production (physical model on observed weather)
    pv_meas: pd.Series               # what the meter measures
    loads: pd.DataFrame              # one column per home
    types: dict                      # type of each home
    grid: pd.DataFrame               # co2, France consumption, tempo
    price_buy: pd.Series
    price_sell: pd.Series

    @property
    def load_total(self) -> pd.Series:
        return self.loads.sum(axis=1).rename("consumption_kwh")

    def sources(self) -> dict:
        return {
            "observed weather": self.weather.attrs.get("source"),
            "forecast weather": self.weather_fc.attrs.get("source"),
            "measured production": self.pv_meas.attrs.get("source"),
            "consumptions": self.loads.attrs.get("source"),
            "grid (CO2)": self.grid.attrs.get("source"),
        }


def load_inputs(scn: Scenario) -> Inputs:
    days_total = scn.days + 1
    obs, fc = load_weather_pair(scn.site, scn.start, days_total, scn.history_days, scn.offline, scn.seed)
    idx = obs.index
    fc = fc.reindex(idx).interpolate(limit_direction="both")
    fc.attrs["source"] = fc.attrs.get("source") or "Open-Meteo"
    pv_true = pv_production_kwh(obs, scn.site, scn.pv)
    pv_meas = measured_production(pv_true, seed=scn.seed)
    loads, types = make_building(scn.building, idx, seed=scn.seed)
    grid = load_grid(scn.start, days_total, scn.history_days, scn.offline, scn.seed).reindex(idx).ffill().bfill()
    t0 = pd.Timestamp(scn.start).tz_localize("Europe/Paris")
    sim_index = idx[(idx >= t0) & (idx < t0 + pd.Timedelta(days=scn.days))]
    return Inputs(scn, idx, sim_index, obs, fc, pv_true, pv_meas, loads, types, grid,
                  buy_price(idx, scn.tariff, grid["tempo"]), sell_price(idx, scn.tariff))
