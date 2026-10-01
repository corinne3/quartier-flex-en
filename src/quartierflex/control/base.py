"""
base.py: the contract between the simulator and a control strategy.

Every hour, the strategy receives the STATE (hour, battery level,
forecasts...) and returns a DECISION made of two numbers:

    grid_charge_kw   : power to charge FROM THE GRID (e.g. at night during off-peak hours)
    max_discharge_kw : maximum power the battery is allowed to deliver
                       this hour (0 = keep the energy for later)

Solar surplus, on the other hand, is ALWAYS stored if there is room left (it is
free and sells poorly). This "real-time" layer is shared by all
strategies: what sets them apart is WHEN they hold on to energy
and WHEN they buy from the grid to fill up.

Why this split? That is how real systems work:
a local controller reacts within the second (surplus / shortfall), a "brain"
(rule, optimizer, AI) sets the setpoints every hour.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

INF = 1e9


@dataclass
class Decision:
    grid_charge_kw: float = 0.0
    max_discharge_kw: float = INF
    reason: str = ""


@dataclass
class State:
    t: pd.Timestamp
    soc: float                 # average state of charge of the bank (0-1)
    energy_kwh: float          # stored energy
    capacity_kwh: float        # current capacity (SOH included)
    usable_kwh: float          # usable capacity (between SOC min and max)
    max_charge_kw: float
    max_discharge_kw: float
    horizon: pd.DataFrame      # forecasts and prices for the next 24 h (index = hours)


class Controller:
    name = "base"
    label = "base"
    family = "?"                # "reference", "rule", "optimization + AI", "agent LLM"
    uses_ai = False
    uses_llm = False
    description = ""

    def __init__(self, **kw):
        self.meter = None
        self.stats: dict = {}

    def setup(self, ctx, meter) -> None:
        self.meter = meter

    def decide(self, state: State) -> Decision:  # pragma: no cover
        raise NotImplementedError

    def bump(self, k: str, n: int = 1) -> None:
        self.stats[k] = self.stats.get(k, 0) + n
