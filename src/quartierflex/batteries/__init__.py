"""
MODULE 3 — BATTERIES (second life): physical model, aging, BMS (measurement + estimation).

    from quartierflex.batteries import BatteryBank, run_bms, estimate_soh

Demo: quartier demo batteries      Tests: pytest tests/test_batteries.py
Doc: docs/modules/03_batteries.md
"""

from .bms import Sensors, demo_profile, estimate_soh, ocv_cell, run_bms, soc_from_voltage  # noqa: F401
from .model import BatteryBank, BatteryPack  # noqa: F401
