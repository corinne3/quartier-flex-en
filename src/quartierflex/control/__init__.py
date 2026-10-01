"""
MODULE 8 — BATTERY CONTROL: who decides when to charge / discharge the battery.

    Without AI: NoBattery, SelfConsumptionRule, TariffRule
    With AI   : Optimizer (ML forecasts + linear programming), AgentLLM (hourly / routed)

Demo: quartier demo control      Tests: pytest tests/test_control.py
Doc: docs/modules/08_control.md
"""

from .agent import AgentLLM  # noqa: F401
from .base import Controller, Decision, State  # noqa: F401
from .optimizer import Optimizer, solve_plan, wear_cost_eur_per_kwh  # noqa: F401
from .forecasts import ai_forecasts, naive_forecasts  # noqa: F401
from .rules import SelfConsumptionRule, TariffRule, NoBattery  # noqa: F401


def make_controller(key: str, llm=None) -> Controller:
    """Factory from a short name (used by the command line and the interface)."""
    table = {
        "no_battery": lambda: NoBattery(),
        "rule": lambda: SelfConsumptionRule(),
        "tariff_rule": lambda: TariffRule(),
        "optimizer": lambda: Optimizer(),
        "agent_hourly": lambda: AgentLLM(llm=llm, mode="hourly"),
        "agent_route": lambda: AgentLLM(llm=llm, mode="route"),
    }
    return table[key]()


ALL = ["no_battery", "rule", "tariff_rule", "optimizer", "agent_hourly", "agent_route"]
LLM_KEYS = ["agent_hourly", "agent_route"]
