"""
MODULE 4 — USAGE: home consumption (simulated) and learning (habits, forecasting).

    from quartierflex.usage import make_building, cluster_habits, compare_load_forecasts

Demo: quartier demo usage      Tests: pytest tests/test_usage.py
Doc: docs/modules/04_usage.md
"""

from .learning import LoadForecaster, cluster_habits, compare_load_forecasts, daily_profiles  # noqa: F401
from .profiles import TYPES, household_load, make_building  # noqa: F401
