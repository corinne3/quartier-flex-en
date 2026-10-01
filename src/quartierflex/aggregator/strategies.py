"""
strategies.py: how the AGGREGATOR allocates RTE's request across the homes.

This is where the AI "juggles the sheddable groups". Five strategies, from the simplest
to the most compute-intensive:

| strategy         | AI?               | what it does                                                           |
|------------------|-------------------|------------------------------------------------------------------------|
| 0 none           | no                | baseline: RTE's request is ignored                                     |
| 1 cut everything | no                | cuts EVERYTHING allowed during the request (everyone at the same       |
|                  |                   | time) + battery at full power                                          |
| 2 round-robin    | no                | battery at full power, EV charging delayed, water heaters cut, then    |
|                  |                   | heaters cut in rotation, just enough to reach the volume               |
| 2b prepared      | no                | + common-sense rules: battery recharged the night before, spread       |
|    round-robin   |                   | over the request, 30 % margin (the best strategy WITHOUT AI)           |
| 3 AI optimizer   | yes (light)       | learned forecasts + linear optimization: recharges the battery the     |
|                  |                   | night before, preheats, cuts first those with comfort margin           |
|                  |                   | (absent, warm homes), spares retirees, smooths the rebound             |
| 4 LLM agent      | yes (heavy)       | an LLM reads the declared flexibility and, the day before, picks the   |
|                  |                   | optimizer settings: protected groups, order, preheating, recharging;   |
|                  |                   | the optimizer executes                                                 |

All follow the same comfort rule (heating cut for 30 min max).
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field
from scipy.optimize import linprog

from ..agent.harness import AgentHarness, Budget
from ..agent.tools import NoArgs, Tool, ToolRegistry
from ..appliances import APPLIANCES
from ..demand_response import forecast_consumption, report_flexibility
from .simulation import Action, Observation, max_charge, max_discharge


class Strategy:
    name = "base"
    label = "base"
    family = "no AI"
    uses_ai = False
    uses_llm = False
    description = ""

    def setup(self, ctx, meter) -> None:
        self.ctx, self.meter = ctx, meter
        self.counters: dict = {}

    def bump(self, key: str, n: int = 1) -> None:
        self.counters[key] = self.counters.get(key, 0) + n

    def day_before(self, request, ctx) -> None:
        """Called when a request is announced (the day before at 5 pm)."""

    def decide(self, obs: Observation) -> Action:
        return Action()


# =============================================================================
class NoShedding(Strategy):
    name, label = "0_none", "No shedding (baseline)"
    description = "Ignores RTE's request. Serves as the baseline curve."


# =============================================================================
class CutEverything(Strategy):
    name, label = "1_cut_all", "Cut everything at once"
    description = "During the request, cuts everything the comfort rule allows, with the battery at full power."

    def decide(self, obs: Observation) -> Action:
        if obs.request is None:
            return Action()
        return Action(cut={a: obs.allowed[a] & (obs.p_now[a] > 0) for a in APPLIANCES},
                      battery_kw=-max_discharge(obs.bank, obs.dt), reason="cut everything")


# =============================================================================
def fill(need: float, candidates: np.ndarray, power: np.ndarray, order: np.ndarray) -> tuple[np.ndarray, float]:
    """Cuts homes in the given order until `need` kW is covered. Returns (mask, kW obtained)."""
    cut = np.zeros(len(candidates), bool)
    obtained = 0.0
    for i in order:
        if obtained >= need:
            break
        if candidates[i] and power[i] > 0.01:
            cut[i] = True
            obtained += power[i]
    return cut, obtained


class RoundRobin(Strategy):
    name, label = "2_round_robin", "Round-robin (simple rule)"
    description = ("Battery at full power, EV charging delayed, water heaters cut, then heaters cut "
                   "in rotation up to the requested volume. No forecasting.")

    def setup(self, ctx, meter) -> None:
        super().setup(ctx, meter)
        self.pointer = 0

    def decide(self, obs: Observation) -> Action:
        if obs.request is None:
            return Action()
        need = obs.request.volume_kw
        bat = max_discharge(obs.bank, obs.dt)
        need -= min(bat, max(obs.net_kw, 0))
        cut = {}
        for a in ("ev", "water_heater"):
            cut[a] = obs.allowed[a] & (obs.p_now[a] > 0)
            need -= float(obs.p_now[a][cut[a]].sum())
        n = len(obs.t_in_measure)
        order = (np.arange(n) + self.pointer) % n                   # rotation
        cut["heating"], _ = fill(need, obs.allowed["heating"], obs.p_now["heating"], order)
        if cut["heating"].any():
            self.pointer = (int(np.where(cut["heating"])[0].max()) + 1) % n
        return Action(cut=cut, battery_kw=-bat, reason="round_robin")


# =============================================================================
class PreparedRoundRobin(RoundRobin):
    name, label = "2b_prepared_round_robin", "Prepared round-robin (rules, no AI)"
    description = ("Like round-robin, plus three common-sense rules: batteries recharged the night before and "
                   "kept for the request, battery spread over the whole duration, 30 % margin on the volume "
                   "(to offset the catch-up of heaters switched back on).")
    MARGIN = 1.3

    def decide(self, obs: Observation) -> Action:
        cfg = self.ctx.scenario.battery
        d = obs.upcoming
        if obs.request is None:
            if d is None:
                return Action()
            if obs.off_peak and obs.bank.soc < cfg.soc_max - 0.01:
                return Action(battery_kw=max_charge(obs.bank, obs.dt), reason="overnight recharge (rule)")
            return Action(battery_kw=max(0.0, -obs.net_kw), reason="battery kept (rule)")
        e_bat = max(0.0, obs.bank.energy_kwh - cfg.soc_min * obs.bank.capacity_kwh) * cfg.eta_discharge
        bat = min(max_discharge(obs.bank, obs.dt), e_bat / max(obs.steps_left * obs.dt, obs.dt))
        need = self.MARGIN * obs.request.volume_kw - min(bat, max(obs.net_kw, 0))
        cut = {}
        for a in ("ev", "water_heater"):
            cut[a] = obs.allowed[a] & (obs.p_now[a] > 0)
            need -= float(obs.p_now[a][cut[a]].sum())
        n = len(obs.t_in_measure)
        order = (np.arange(n) + self.pointer) % n
        cut["heating"], _ = fill(need, obs.allowed["heating"], obs.p_now["heating"], order)
        if cut["heating"].any():
            self.pointer = (int(np.where(cut["heating"])[0].max()) + 1) % n
        return Action(cut=cut, battery_kw=-bat, reason="prepared round-robin")


# =============================================================================
class AIOptimizer(Strategy):
    name, label = "3_optimizer_ai", "AI optimizer (forecasts + optimization)"
    family, uses_ai = "AI (light)", True
    description = ("Thermal model learned for each home, forecasts, linear optimization of the "
                   "allocation across groups, batteries recharged the night before, preheating, "
                   "retirees spared, soft restart after the request.")
    COST_COMFORT = 0.25        # €/kWh shed per discomfort "risk unit" (internal trade-off)
    PENALTY = 0.30            # €/kWh not shed (≈ RTE penalty)
    MARGIN = 1.0                # safety margin on the volume (tested: 1.1 does worse, the battery runs out too early)
    PREHEAT_C = 1.0       # °C added to the setpoint 1 h before the request (occupied homes)

    def setup(self, ctx, meter) -> None:
        super().setup(ctx, meter)
        self.plans = []        # [(request, homes to preheat, settings)]
        self._fallback = RoundRobin()
        self._fallback.setup(ctx, meter)

    def day_before(self, request, ctx) -> None:
        # Who will be home during the request? They will be preheated 1 h before (+1 °C).
        from ..appliances import setpoints

        _, sched = setpoints(ctx.portfolio, pd.DatetimeIndex([request.start]))   # EXPECTED presence (schedule)
        present = sched[0] & ctx.portfolio.heating & ctx.portfolio.connected
        self.plans.append((request, present, self.settings(request, ctx)))
        self.bump("plans_day_before")

    def settings(self, request, ctx) -> dict:
        """The strategic choices for a request. The optimizer fixes them; the LLM agent CHOOSES them."""
        return {"preheat_c": self.PREHEAT_C, "precharge": True, "excluded": set(), "rank": {}}

    def _prep(self, obs: Observation):
        return next(((d, pres, r) for d, pres, r in self.plans if obs.t < d.end), (None, None, None))

    def decide(self, obs: Observation) -> Action:
        d, pres, sett = self._prep(obs)
        cfg = self.ctx.scenario.battery
        if obs.request is None:
            act = Action()
            if d is not None and obs.t >= d.announce:
                # --- Preparation: recharge the battery overnight (off-peak hours), then keep it full.
                h_before = (d.start - obs.t).total_seconds() / 3600
                # Energy useful during the request (no more: midday sun must still be storable,
                # otherwise it would be sold back at a low price)
                need = min(sum(pk.p_max_kw for pk in obs.bank.packs), d.volume_kw) * d.duration_h / cfg.eta_discharge
                target_soc = min(cfg.soc_max, cfg.soc_min + need / max(obs.bank.capacity_kwh, 1e-9))
                if not sett["precharge"]:
                    act = Action()
                elif obs.off_peak and obs.bank.soc < target_soc - 0.01:
                    act = Action(battery_kw=min(max_charge(obs.bank, obs.dt),
                                                 (target_soc - obs.bank.soc) * obs.bank.capacity_kwh / obs.dt / cfg.eta_charge),
                                 reason="overnight recharge before the request")
                else:
                    act = Action(battery_kw=max(0.0, -obs.net_kw), reason="battery kept for the request")
                if 0 < h_before <= 1.0 and sett["preheat_c"] > 0:
                    act.delta_setpoint = np.where(pres, sett["preheat_c"], 0.0)
                    act.reason += " + preheating"
            # --- Soft restart: after a request, EV charging waits for off-peak hours
            #     (otherwise it all restarts at 8 pm: a new peak, at the highest price).
            finished = any(dd.end <= obs.t < dd.end.normalize() + pd.Timedelta(hours=31) for dd, _, _ in self.plans)
            charge = obs.allowed["ev"] & (obs.p_now["ev"] > 0)
            if finished and not obs.off_peak and 12 <= obs.t.hour < 22:
                act.cut = {"ev": charge}
                act.reason = (act.reason + " ; " if act.reason else "") + "EV charging delayed to off-peak hours"
            elif finished and obs.off_peak and charge.any():
                # ... then SPREAD over the night (otherwise they all start at 10 pm: a new peak)
                h_departure = ((7 - obs.t.hour) % 24) - obs.t.minute / 60
                budget = 1.3 * float(obs.ev_need[charge].sum()) / max(h_departure, obs.dt)
                order = np.argsort(-obs.ev_need)                      # largest needs first
                kept, _ = fill(budget, charge, obs.p_now["ev"], order)
                act.cut = {"ev": charge & ~kept}
                act.reason = (act.reason + " ; " if act.reason else "") + "EV charging spread over the night"
            return act
        return self._allocate(obs, sett or self.settings(obs.request, self.ctx))

    def _allocate(self, obs: Observation, sett: dict) -> Action:
        """Linear optimization, at GROUP level, over the remaining steps of the request."""
        ctx, portfolio = self.ctx, self.ctx.portfolio
        k0, H = obs.k, max(1, obs.steps_left)
        idx = ctx.index[k0:k0 + H]
        prev = forecast_consumption(portfolio, ctx.model, obs.t_in_measure, idx, ctx.weather_fc, ctx.profiles)
        # "SHADOW" homes: for each home, the AI simulates with its learned model what it would consume
        # IF IT HAD NOT BEEN CUT (same setpoint, same weather). Homes switched back on after 30 min
        # consume more (catch-up): the actual − shadow gap is this catch-up, and we cut that much more.
        d = obs.request
        cfg_a = ctx.scenario.appliances
        m = ctx.model
        heats = portfolio.heating & portfolio.connected
        gains = m.g0 + m.g1 * obs.presence + m.g2 * obs.ghi
        if getattr(self, "_ref_pour", None) is not d:
            self._ref_pour = d
            self._t_shadow = obs.t_in_measure.copy()
            p0 = m.power(self._t_shadow, obs.t_ext, obs.setpoint, obs.presence, obs.ghi, portfolio.p_heating_max, obs.dt)
            self._bias = float(obs.p_now["heating"].sum() - p0[heats].sum())   # calibration at the 1st step
            self._ev_ref = obs.ev_need.copy()        # car needs "as if nothing were cut"
            self._ev_seen = obs.ev_need.copy()
        p_shadow = m.power(self._t_shadow, obs.t_ext, obs.setpoint, obs.presence, obs.ghi, portfolio.p_heating_max, obs.dt)
        self._t_shadow = self._t_shadow + obs.dt / m.c * (p_shadow + gains - m.ua * (self._t_shadow - obs.t_ext))
        ref_now = float(p_shadow[heats].sum()) + self._bias
        catchup = max(0.0, float(obs.p_now["heating"].sum()) - ref_now)
        # Cars: the baseline is charging WITHOUT cuts (they would have finished charging earlier).
        arrival = np.clip(obs.ev_need - self._ev_seen, 0, None)   # newly plugged-in cars
        self._ev_ref = self._ev_ref + arrival
        p_ev_ref = np.minimum(cfg_a.ev_kw, self._ev_ref / obs.dt)
        self._ev_ref = np.maximum(0.0, self._ev_ref - p_ev_ref * obs.dt)
        self._ev_seen = obs.ev_need.copy()
        # Cars and water heaters: cut EVERYTHING that is drawing (no discomfort); what this actually sheds
        # versus the baseline is deducted from the volume to find on heating and the battery.
        cut = {a: np.zeros(portfolio.n, bool) for a in APPLIANCES}
        for a in ("ev", "water_heater"):
            cut[a] = obs.allowed[a] & (obs.p_now[a] > 0)
        credit = float(p_ev_ref[cut["ev"]].sum() + obs.p_now["water_heater"][cut["water_heater"]].sum())
        # Comfort margin of each home (°C above the tolerated minimum) and risk if cut for 15 min
        drop = ctx.model.drop_if_cut(obs.t_in_measure, obs.t_ext, obs.presence, obs.ghi, obs.dt)
        margin = obs.t_in_measure - (obs.setpoint - portfolio.tolerance)
        risk = np.where(obs.presence, np.clip(drop / np.maximum(margin, 0.05), 0, 5), 0.0)
        groups, cap, cost, members = [], [], [], []
        for a in ("heating",):
            grp = portfolio.group(a)
            for g in sorted(set(grp) - {""}):
                cols = grp == g
                now = float(obs.p_now[a][cols & obs.allowed[a]].sum())
                future = prev[a][:, cols].sum(axis=1) * (0.5 if a == "heating" else 1.0)
                c = np.concatenate([[now], future[1:]]) if H > 1 else np.array([now])
                if g in sett["excluded"]:
                    c = np.zeros_like(c)                  # protected group: never cut
                groups.append((a, g))
                cap.append(c)
                members.append(cols)
                r = float(risk[cols].mean()) if a == "heating" else 0.0
                cost.append(0.02 + self.COST_COMFORT * r + 0.01 * sett["rank"].get(g, 0))
        G = len(groups)
        pb_max = max_discharge(obs.bank, obs.dt)
        e_bat = max(0.0, obs.bank.energy_kwh - ctx.scenario.battery.soc_min * obs.bank.capacity_kwh) * \
            ctx.scenario.battery.eta_discharge
        target = max(0.0, self.MARGIN * obs.request.volume_kw + catchup - credit)
        # Variables: x[g,t] (G*H), b[t] (H), s[t] shortfall (H)
        nv = G * H + 2 * H
        c = np.concatenate([np.repeat(cost, H), np.full(H, 0.01), np.full(H, self.PENALTY)])
        A, bnd = [], []
        for t in range(H):                                   # shed sum + battery + shortfall >= volume
            row = np.zeros(nv)
            row[[g * H + t for g in range(G)]] = -1
            row[G * H + t] = -1
            row[G * H + H + t] = -1
            A.append(row)
            bnd.append(-target)
        row = np.zeros(nv)                                   # battery energy over the request
        row[G * H:G * H + H] = obs.dt
        A.append(row)
        bnd.append(e_bat)
        bounds = [(0, max(0.0, float(cap[g][t]))) for g in range(G) for t in range(H)]
        bounds += [(0, min(pb_max, max(obs.net_kw, 0)))] + [(0, pb_max)] * (H - 1) + [(0, None)] * H
        res = linprog(c, A_ub=np.array(A), b_ub=np.array(bnd), bounds=bounds, method="highs")
        self.bump("optimizations")
        if not res.success:
            self.bump("failures_optimization")
            return self._fallback.decide(obs)
        x = res.x
        for gi, (a, g) in enumerate(groups):
            need = x[gi * H]
            if need <= 0.01:
                continue
            cols = members[gi] & obs.allowed[a]
            # Within the group: first those with the most margin (or absent), those already cut continue
            prio = np.where(obs.presence, margin, 10.0)
            order = np.argsort(-prio)
            m, _ = fill(need, cols, obs.p_now[a], order)
            cut[a] |= m
        return Action(cut=cut, battery_kw=-float(x[G * H]),
                      reason=f"optimization: battery {x[G * H]:.0f} kW, catch-up offset {catchup:.0f} kW, cars+water heaters {credit:.0f} kW")


# =============================================================================
class SheddingPlan(BaseModel):
    """MANDATORY agent output: validated by Pydantic (otherwise the harness asks for a correction)."""

    model_config = ConfigDict(extra="forbid")
    order: list[str] = Field(min_length=1, description="groups to cut, from first to last")
    exclude: list[str] = Field(default_factory=list, description="groups never to cut")
    preheat: bool
    precharge_battery: bool
    reason: str = ""


SYSTEM = """You are the demand-response aggregator of a district. The grid operator (RTE) asks to shed a volume (kW) during a time slot.
You have sheddable groups (occupant type/appliance) and second-life electric-car batteries.
Rules: a heater is cut for 30 min at most, then back on for 30 min; retirees are vulnerable to cold;
delaying a car charge (ev) or cutting a water heater (water_heater) bothers nobody; batteries must be
recharged the night before to be useful. Call the tool view_flexibility, then answer ONLY with this JSON:
{"order": ["<group>", ...], "exclude": ["<group>", ...], "preheat": true/false, "precharge_battery": true/false, "reason": "<10 words>"}
Example: {"order": ["family/ev", "family/water_heater", "working_couple/heating"], "exclude": ["retirees/heating"], "preheat": true, "precharge_battery": true, "reason": "comfort first"}"""


class AgentLLM(AIOptimizer):
    """
    The LLM agent does not control homes one by one (that would be thousands of calls):
    the day before, it CHOOSES the optimizer settings (which groups to protect, in which order
    to cut, whether to preheat, whether to recharge the batteries). Execution every 15 min
    remains the optimizer's. This is "harness engineering": the LLM at the strategic level,
    reliable and cheap tools at the operational level.
    """

    name, label = "4_agent_llm", "LLM agent (tunes the optimizer the day before)"
    family, uses_ai, uses_llm = "AI (generative)", True, True
    description = ("An LLM reads the declared flexibility (tool) and chooses the day before: protected groups, cut "
                   "order, preheating, battery recharging. The optimizer executes.")

    def __init__(self, llm=None):
        self.llm = llm
        self.traces = []

    def setup(self, ctx, meter) -> None:
        super().setup(ctx, meter)
        self._report = ""
        tools = ToolRegistry([Tool("view_flexibility", "Forecast flexibility per group, battery state, requested volume",
                                   lambda a: self._report, NoArgs)])
        self.harness = AgentHarness(self.llm, SYSTEM, SheddingPlan, tools=tools, max_steps=4,
                                    budget=Budget(max_llm_calls=4, max_tokens=6000), meter=meter)

    def settings(self, request, ctx) -> dict:
        rep = getattr(request, "report", None)
        lines = [f"Request: {request.volume_kw:.0f} kW from {request.start:%d/%m %Hh} to {request.end:%Hh}."]
        if rep is not None:
            for _, r in rep["groups"].iterrows():
                tol = "" if np.isnan(r["tolerance °C"]) else f", tolerance {r['tolerance °C']} °C"
                lines.append(f"{r['group']}: {r['homes']} homes, {r['kW sheddable']} kW{tol}")
            b = rep["batteries"]
            lines.append(f"batteries: {len(b['packs'])} packs, mean SOH "
                          f"{np.mean([p['SOH %'] for p in b['packs']]):.0f} %, up to {b['kw_held_over_slot']:.0f} kW "
                          f"if recharged; currently {b['energy_current_kwh']:.0f} kWh available")
        self._report = "\n".join(lines)
        res = self.harness.run(json.dumps({"request_kw": round(request.volume_kw), "start": f"{request.start:%Y-%m-%d %H:%M}",
                                           "end": f"{request.end:%H:%M}"}, ensure_ascii=False))
        self.bump("agent_calls")
        self.traces.append({"request": f"{request.start:%Y-%m-%d %H:%M}", "stop": res.stop_reason, "trace": res.trace})
        known = set(ctx.portfolio.groups())
        if res.output is None:                              # safe fallback: optimizer default settings
            self.bump(f"fallback_{res.stop_reason}")
            return super().settings(request, ctx)
        plan = res.output
        unknown = [g for g in plan.order + plan.exclude if g not in known]
        if unknown:
            self.bump("groups_unknown_ignores", len(unknown))
        order = [g for g in plan.order if g in known]
        return {"preheat_c": 1.0 if plan.preheat else 0.0, "precharge": plan.precharge_battery,
                "excluded": {g for g in plan.exclude if g in known}, "rank": {g: i for i, g in enumerate(order)},
                "reason": plan.reason}


def make_strategy(key: str, llm=None) -> Strategy:
    return {"none": NoShedding, "cut_all": CutEverything, "round_robin": RoundRobin, "prepared_round_robin": PreparedRoundRobin,
            "optimizer": AIOptimizer}[key]() if key != "agent" else AgentLLM(llm)


ALL = ["none", "cut_all", "round_robin", "prepared_round_robin", "optimizer", "agent"]


def set_volumes(ctx, series_ref, bank_ref=None) -> list:
    """
    RTE sets the volume of each request: share_of_flex × flexibility DECLARED at the announcement.
    The declaration uses the AI forecast (learned thermal model) from the state measured at the announcement
    (the no-shedding trajectory is used: the same request for all strategies).
    """
    from ..batteries import BatteryBank

    portfolio = ctx.portfolio
    bank = bank_ref or BatteryBank(ctx.scenario.battery, soc0=ctx.scenario.battery.soc_min)
    out = []
    for d in ctx.requests:
        k = int(np.searchsorted(ctx.index, d.announce))
        t_in = series_ref["t_in"][min(k, len(series_ref["t_in"]) - 1)] if "t_in" in series_ref else ctx.state0.t_in
        t_in = t_in + np.random.default_rng(k).normal(0, 0.1, len(t_in))      # box measurement (noisy)
        idx = ctx.weather_fc.index[(ctx.weather_fc.index >= d.announce) & (ctx.weather_fc.index < d.end)]
        prev = forecast_consumption(portfolio, ctx.model, t_in, idx, ctx.weather_fc, ctx.profiles)
        mask = (idx >= d.start) & (idx < d.end)
        rep = report_flexibility(portfolio, prev, mask, bank, d.duration_h)
        d.volume_kw = round(ctx.scenario.demand_response.share_of_flex * rep["total_kw"], 1)
        d.report = rep
        out.append(rep)
    return out
