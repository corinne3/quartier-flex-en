"""
MODULE 9 — AGGREGATOR: the AI that allocates RTE's request across the district's homes.

    from quartierflex.aggregator import load_district, run_demand_response, table

- data.py       : the district at a 15-min step (weather, solar, grid, portfolio, history, learning)
- simulation.py : runs the district with a strategy (comfort rule shared by all)
- strategies.py : none / cut everything / round-robin / AI optimizer / LLM agent
- assessment.py : delivered shedding, rebound, comfort, €, CO2, and NET AI BALANCE

Demo: quartier demo aggregator      Tests: pytest tests/test_aggregator.py
Module sheet: docs/modules/09_aggregator.md
"""

from .assessment import kpis_demand_response, run_demand_response, table  # noqa: F401
from .data import DistrictContext, load_district  # noqa: F401
from .simulation import Action, Observation, run_simulation  # noqa: F401
from .strategies import ALL, AgentLLM, AIOptimizer, set_volumes, make_strategy  # noqa: F401
