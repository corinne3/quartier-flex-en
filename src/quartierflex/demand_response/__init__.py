"""
MODULE 7 — DEMAND RESPONSE: RTE's requests, and what the district declares to RTE.

    from quartierflex.demand_response import create_requests, report_flexibility

- signals.py   : when and how much RTE (the French transmission system operator) asks to shed
                 (stress days, peak slots)
- reporting.py : the flexibility declared to RTE (sheddable groups + battery state)

Demo: quartier demo demand_response      Tests: pytest tests/test_demand_response.py
Module sheet: docs/modules/07_demand_response.md
"""

from .reporting import (SHARE_SHEDDABLE, flex_batteries, forecast_consumption, hourly_profiles,  # noqa: F401
                       report_flexibility)
from .signals import Request, create_requests, stress_days  # noqa: F401
