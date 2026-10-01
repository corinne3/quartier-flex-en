"""
MODULE 2 — SOLAR: panel production (physics), meter (measurement), forecast (AI).

    from quartierflex.solar import pv_production_kwh, measured_production, compare_forecasts

Demo: quartier demo solar      Tests: pytest tests/test_solar.py
Sheet: docs/modules/02_solar.md
"""

from .meter import measured_production  # noqa: F401
from .physics import plane_of_array, pv_production_kwh  # noqa: F401
from .forecast import PVForecaster, compare_forecasts, nmae, persistence  # noqa: F401
