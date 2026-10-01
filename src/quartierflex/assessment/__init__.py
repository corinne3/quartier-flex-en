"""
MODULE 11 — ASSESSMENT: hourly simulation (self-consumption), indicators, net AI balance, sizing.

    from quartierflex.assessment import load_inputs, simulate, kpis, run_all, sweep

Demo: quartier demo assessment      Tests: pytest tests/test_assessment.py
Doc: docs/modules/11_assessment.md
"""

from .bench import run_all, summary_table  # noqa: F401
from .sizing import sweep  # noqa: F401
from .data import Inputs, load_inputs  # noqa: F401
from .kpi import kpis, net_ai_balance  # noqa: F401
from .simulation import simulate  # noqa: F401
