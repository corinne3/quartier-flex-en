"""
simulation.py: run the district every 15 minutes with an aggregator strategy.

At each step:
    1. the day before a request (5 pm), the strategy receives the announcement (it can prepare:
       recharge the batteries overnight, plan preheating...);
    2. the strategy sees the MEASURED state (home temperatures with sensor noise,
       power of each appliance, batteries) and decides: which appliances to cut,
       what the battery does;
    3. the COMFORT RULE applies to all: a heater is never cut for more than
       30 min, then stays on for 30 min; a water heater for 2 h at most;
    4. physics advances (appliances module models); the battery applies the command and ages;
    5. the district's grid draw, comfort and service failures are recorded.

DELIVERED shedding is then measured as the difference with the "no shedding" simulation
(this is RTE's "baseline curve"; in reality it is ESTIMATED, here we know it).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..appliances import APPLIANCES, cut_allowed, time_step, power_sheddable
from ..batteries import BatteryBank

MAX_STEP = {"heating": 2, "water_heater": 8, "ev": 10**6}     # 30 min, 2 h, unlimited (charging is delayed)
REST_STEP = {"heating": 2, "water_heater": 4, "ev": 0}


@dataclass
class Observation:
    k: int
    t: pd.Timestamp
    dt: float
    request: object | None          # request IN PROGRESS (or None)
    upcoming: object | None        # next announced request, not yet finished
    steps_left: int               # steps left in the current request
    t_in_measure: np.ndarray
    p_now: dict                     # power each appliance is requesting now (kW)
    allowed: dict                  # cut allowed by the comfort rule (bool)
    setpoint: np.ndarray
    presence: np.ndarray
    t_ext: float
    ghi: float
    net_kw: float                   # consumption without cuts − solar production (kW)
    bank: BatteryBank
    off_peak: bool
    ev_need: np.ndarray = None    # kWh left to charge in each car (known to the smart charger)


@dataclass
class Action:
    cut: dict = field(default_factory=dict)   # appliance -> bool array
    battery_kw: float | None = None            # + charge, − discharge; None = self-consumption
    delta_setpoint: np.ndarray | None = None    # preheating (°C added to the setpoint)
    reason: str = ""


def max_discharge(bank: BatteryBank, dt: float) -> float:
    return sum(p.max_discharge_kw(dt) for p in bank.packs)


def max_charge(bank: BatteryBank, dt: float) -> float:
    return sum(p.max_charge_kw(dt) for p in bank.packs)


def run_simulation(ctx, strat, meter, seed: int = 0) -> dict:
    scn = ctx.scenario
    portfolio = ctx.portfolio
    state = ctx.state0.copy()
    cfg = scn.battery
    bank = BatteryBank(cfg, soc0=cfg.soc_min)             # batteries empty at start (nothing for free)
    rng = np.random.default_rng(seed + 11)
    dt = ctx.dt
    n = portfolio.n
    has = {"heating": portfolio.heating, "water_heater": portfolio.water_heater, "ev": portfolio.ev}
    announced = set()
    base = portfolio.base_kw.reindex(ctx.index).to_numpy()
    rows, n_cuts = [], {a: 0 for a in APPLIANCES}
    t_hist = np.zeros((len(ctx.index), n))
    for k, t in enumerate(ctx.index):
        t_hist[k] = state.t_in
        # ------------------------------------------------ current / announced requests
        req = next((d for d in ctx.requests if d.active(t)), None)
        upcoming = next((d for d in ctx.requests if d.announce <= t < d.end), None)
        for i, d in enumerate(ctx.requests):
            if i not in announced and d.announce <= t:
                announced.add(i)
                with meter.measure("shared"):
                    strat.day_before(d, ctx)
        steps_left = int(round((req.end - t).total_seconds() / 60 / scn.demand_response.step_min)) if req else 0
        t_ext, ghi = float(ctx.weather["temp_c"].iloc[k]), float(ctx.weather["ghi_wm2"].iloc[k])
        sp, pres = ctx.setpoint[k], ctx.presence[k]
        p_now = power_sheddable(portfolio, state, t_ext, ghi, sp, pres, dt)
        allowed = {a: has[a] & portfolio.connected & cut_allowed(state, a, MAX_STEP[a], REST_STEP[a])
                    for a in APPLIANCES}
        net_without = float(base[k].sum() + sum(v.sum() for v in p_now.values()) - ctx.pv_kw[k])
        obs = Observation(k, t, dt, req, upcoming, steps_left, state.t_in + rng.normal(0, 0.1, n), p_now,
                          allowed, sp, pres, t_ext, ghi, net_without, bank, bool(ctx.off_peak[k]),
                          state.ev_need.copy())
        with meter.measure("shared"):
            act = strat.decide(obs)
        # ------------------------------------------------ comfort rule + physics
        cut = {a: np.asarray(act.cut.get(a, np.zeros(n, bool)), bool) & allowed[a] for a in APPLIANCES}
        sp_eff = sp + (act.delta_setpoint if act.delta_setpoint is not None else 0.0)
        p = time_step(portfolio, state, t_ext, ghi, sp_eff, pres, t.hour, t.minute, bool(ctx.off_peak[k]),
                         t.dayofweek >= 5, cut, dt)
        for a in APPLIANCES:
            n_cuts[a] += int(cut[a].sum())
        consumption = float(base[k].sum() + sum(v.sum() for v in p.values()))
        net = consumption - ctx.pv_kw[k]
        # ------------------------------------------------ battery
        if act.battery_kw is None:
            cmd = -net                                   # self-consumption: store the surplus, cover the shortfall
        else:
            cmd = act.battery_kw
        if cmd < 0:
            cmd = -min(-cmd, max(net, 0.0))              # never discharge to sell back
        pb = bank.step(cmd, dt)
        grid_draw = net + pb
        # ------------------------------------------------ comfort (electrically heated homes, occupied)
        gap = (sp - portfolio.tolerance) - state.t_in
        cold = np.where(portfolio.heating & pres, np.clip(gap, 0, None), 0.0)
        rows.append({
            "time": t, "grid_draw_kw": grid_draw, "consumption_kw": consumption, "pv_kw": ctx.pv_kw[k], "battery_kw": pb,
            "soc": bank.soc, "heating_kw": float(p["heating"].sum()), "water_heater_kw": float(p["water_heater"].sum()),
            "ev_kw": float(p["ev"].sum()), "base_kw": float(base[k].sum()),
            "request_kw": req.volume_kw if req else 0.0, "in_request": req is not None,
            "discomfort_degh": float(cold.sum() * dt), "cold_homes": int((cold > 0).sum()),
            "t_in_avg": float(state.t_in[portfolio.heating].mean()) if portfolio.heating.any() else np.nan,
            "t_in_min": float(state.t_in[portfolio.heating & pres].min()) if (portfolio.heating & pres).any() else np.nan,
            "cuts": int(sum(c.sum() for c in cut.values())),
            "price_buy": ctx.price_buy[k], "price_sell": ctx.price_sell[k], "co2_g_kwh": ctx.co2[k],
            "reason": act.reason,
        })
    return {"series": pd.DataFrame(rows).set_index("time"), "bank": bank, "n_cuts": n_cuts,
            "cold_water_kwh": state.cold_water_kwh, "ev_missing_kwh": state.ev_missing_kwh,
            "t_in": t_hist}
