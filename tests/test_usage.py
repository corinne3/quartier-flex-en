"""Module 4 — usage. Run on its own: pytest tests/test_usage.py -v"""
import pandas as pd

from quartierflex.config import BuildingConfig
from quartierflex.usage import TYPES, cluster_habits, compare_load_forecasts, household_load, make_building

IDX = pd.date_range("2024-01-01", periods=24 * 366, freq="h", tz="Europe/Paris")


def test_annual_consumption_per_type():
    import numpy as np

    for kind, (annual, *_) in TYPES.items():
        s = household_load(kind, IDX, np.random.default_rng(0)).sum()
        assert 0.85 * annual < s < 1.15 * annual, kind


def test_habits_recovered():
    loads, types = make_building(BuildingConfig(n_homes=16), IDX[: 24 * 60], seed=3)
    r = cluster_habits(loads, types, k=4)
    assert r["ari"] > 0.3
    assert set(r["table"].index) == set(loads.columns)


def test_consumption_forecast():
    loads, _ = make_building(BuildingConfig(), IDX[: 24 * 70], seed=1)
    total = loads.sum(axis=1)
    sim = total.index[-24 * 7:]
    _, scores = compare_load_forecasts(total, sim)
    assert scores["ai_ml"] < 0.5
