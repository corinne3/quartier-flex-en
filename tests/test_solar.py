"""Module 2 — solar. Run on its own: pytest tests/test_solar.py -v"""
import numpy as np
import pandas as pd

from quartierflex.config import PVConfig, Site
from quartierflex.weather import load_weather_pair
from quartierflex.solar import compare_forecasts, measured_production, pv_production_kwh


def test_annual_production_plausible():
    obs, _ = load_weather_pair(Site(), "2024-01-01", 366, 0, offline=True)
    p = pv_production_kwh(obs, Site(), PVConfig(kwc=1))
    assert 1100 < p.sum() < 1700          # order of magnitude for Valence: ~1,300-1,450 kWh/kWp/year
    assert (p[obs.index.hour == 2] == 0).all()   # nothing at night
    assert p.max() <= 1.0                  # capped at peak power


def test_meter_close_to_model():
    obs, _ = load_weather_pair(Site(), "2024-05-06", 30, 0, offline=True)
    true = pv_production_kwh(obs, Site(), PVConfig())
    meas = measured_production(true, seed=1)
    assert 0.85 < meas.sum() / true.sum() < 1.02   # soiling and faults: slightly less, never more


def test_ai_forecast_works():
    obs, fc = load_weather_pair(Site(), "2024-05-06", 3, 30, offline=True)
    true = pv_production_kwh(obs, Site(), PVConfig())
    meas = measured_production(true)
    sim = obs.index[obs.index >= pd.Timestamp("2024-05-06", tz="Europe/Paris")]
    table, scores = compare_forecasts(obs, fc, meas, sim, Site(), PVConfig())
    assert all(np.isfinite(v) and v < 1 for v in scores.values())
    assert (table["pv_forecast_ml_kwh"] >= 0).all()
