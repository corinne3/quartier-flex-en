"""
agent.py: an LLM agent controls the building's battery.

Reuses the Bilan Net harness (quartierflex/agent/): loop, typed tools,
budget, validation, self-correction, safe fallback.

Two ways to use it (harness engineering, again):
- "hourly": the LLM is consulted EVERY HOUR (24 calls / day).
- "route" : the LLM is only consulted at the 2 moments when a real decision is
            made: 9pm (should we recharge from the grid tonight?) and 6am
            (how should the battery be spread over the day?). In between, the
            last setpoint is kept. 2 calls / day.

Tool provided: get_forecast(hours) -> AI forecasts of production,
consumption and prices for the coming hours.
Fallback: if the agent fails, self-consumption rule.
"""

from __future__ import annotations

import json

from pydantic import BaseModel, ConfigDict, Field

from ..agent.harness import AgentHarness, Budget
from ..agent.tools import Tool, ToolRegistry
from .base import INF, Controller, Decision, State
from .forecasts import ai_forecasts


class AgentDecision(BaseModel):
    """
    REQUIRED fields and unknown fields FORBIDDEN: an answer that does not really
    contain a decision is rejected, and the harness asks the LLM to correct it.
    (With default values, any JSON would pass for a decision.)
    """

    model_config = ConfigDict(extra="forbid")
    grid_charge_kw: float = Field(ge=0, description="Charging power from the grid (kW)")
    max_discharge_kw: float = Field(ge=0, description="Maximum allowed discharge (kW)")
    reason: str = ""


SYSTEM = """You control the shared battery of a building equipped with solar panels.
Solar surplus is always stored automatically. You only decide:
- grid_charge_kw: charge from the grid (worth it during off-peak hours if tomorrow's sun will not be enough),
- max_discharge_kw: power the battery may deliver (0 to keep it for more expensive hours).
Goal: minimize the bill. Buying costs much more than selling earns.
On RED days, peak hours (6am-10pm) cost about 4 times more than night off-peak hours:
if the sun will not be enough, charging at night to discharge during the day is very profitable.
Use the get_forecast tool to see prices and forecasts. Then answer ONLY with this JSON, no other text:
{"grid_charge_kw": <number>, "max_discharge_kw": <number>, "reason": "<10 words max>"}
Example: {"grid_charge_kw": 8, "max_discharge_kw": 0, "reason": "cheap night before red day"}"""


class ForecastArgs(BaseModel):
    hours: int = Field(12, ge=1, le=24, description="Number of hours to forecast")


class AgentLLM(Controller):
    family = "agent LLM"
    uses_ai = True
    uses_llm = True

    def __init__(self, llm=None, mode: str = "hourly", trace_limit: int = 20, **kw):
        super().__init__(**kw)
        self.llm, self.mode = llm, mode
        self.name = "4_agent_llm_hourly" if mode == "hourly" else "5_agent_llm_route"
        self.label = "LLM agent (every hour)" if mode == "hourly" else "Routed LLM agent (twice a day)"
        self.description = ("The LLM decides the battery setpoint " +
                            ("every hour." if mode == "hourly" else "at 9pm and 6am only; the setpoint is kept in between."))
        self.traces, self.trace_limit = [], trace_limit
        self._last = Decision(reason="start: self-consumption")
        self._state = None

    def setup(self, ctx, meter) -> None:
        super().setup(ctx, meter)
        self.fc = ai_forecasts(ctx, meter)

        def get_forecast(a: ForecastArgs) -> str:
            hz = self._state.horizon.iloc[: a.hours]
            return "; ".join(f"{t:%Hh}: pv={r.pv_fc:.1f} load={r.load_fc:.1f} kWh price={r.price_buy:.2f}€"
                             for t, r in hz.iterrows())

        tools = ToolRegistry([Tool("get_forecast", "Production/consumption/price forecasts for the coming hours",
                                   get_forecast, ForecastArgs)])
        self.harness = AgentHarness(self.llm, SYSTEM, AgentDecision, tools=tools, max_steps=3,
                                    budget=Budget(max_llm_calls=3, max_tokens=3000), meter=meter)

    def decide(self, state: State) -> Decision:
        self._state = state
        if self.mode == "route" and state.t.hour not in (21, 6):
            return self._last
        obs = {"hour": f"{state.t:%Y-%m-%d %H:%M}", "level_battery_pct": round(100 * state.soc),
               "energy_stored_kwh": round(state.energy_kwh, 1), "capacity_useful_kwh": round(state.usable_kwh, 1),
               "charge_max_kw": round(state.max_charge_kw, 1), "discharge_max_kw": round(state.max_discharge_kw, 1),
               "price_current_eur_kwh": round(float(state.horizon["price_buy"].iloc[0]), 3)}
        res = self.harness.run(json.dumps(obs, ensure_ascii=False))
        self.bump("llm_runs")
        if len(self.traces) < self.trace_limit:
            self.traces.append({"obs": obs["hour"], "stop": res.stop_reason, "trace": res.trace})
        if res.output is None:
            self.bump(f"fallback_{res.stop_reason}")
            self._last = Decision(reason="fallback: self-consumption")
        else:
            o = res.output
            self._last = Decision(min(o.grid_charge_kw, state.max_charge_kw), o.max_discharge_kw, o.reason)
        return self._last
