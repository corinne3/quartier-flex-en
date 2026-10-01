"""
MODULE 5 — CONNECTED APPLIANCES: heating, water heater, EV charging.

    from quartierflex.appliances import make_portfolio, simulate_without_shedding, learn_thermal

- portfolio.py : who has what (occupant type × appliances × connected), sheddable groups
- physics.py   : how each appliance consumes (1R1C thermal model, hot-water storage, charging)
- learning.py  : the AI learns each home's insulation and inertia from measurements

Demo: quartier demo appliances      Tests: pytest tests/test_appliances.py
Module sheet: docs/modules/05_appliances.md
"""

from .learning import (ThermalModel, learn_thermal, error_learning,  # noqa: F401
                            weather_at_step, simulate_without_shedding)
from .portfolio import APPLIANCES, NAMES_APPLIANCES, Portfolio, setpoints, make_portfolio  # noqa: F401
from .physics import HomeState, gains_kw, cut_allowed, time_step, power_sheddable, thermostat  # noqa: F401
