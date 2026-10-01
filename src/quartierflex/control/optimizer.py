"""
optimizer.py: planning the battery over 24 h with AI forecasts (model predictive control).

Principle (MPC, "Model Predictive Control")
-------------------------------------------
Every hour:
  1. take the AI forecasts for the next 24 hours (production, consumption)
     and the prices (purchase, resale);
  2. compute the BEST charge/discharge plan over these 24 h by solving
     a linear optimization problem;
  3. apply ONLY the first hour of the plan, and start again the next hour
     with fresh information.

The problem solved (linear programming, HiGHS solver via scipy)
---------------------------------------------------------------
Variables per hour: purchase g, resale e, charge c, discharge d, stored energy E.
    minimize    Σ buy_price·g − sell_price·e + wear·(c + d) − terminal_value·E_end
    subject to  production − consumption + g − e − c + d = 0   (electrical balance)
                E(t+1) = E(t) + η_c·c − d/η_d                    (battery balance)
                E_min ≤ E ≤ E_max ; 0 ≤ c, d ≤ P_max ; g, e ≥ 0

Wear cost: what 1 kWh flowing through the battery "costs", derived from its
price and its state of health (batteries module). A worn battery is more
expensive to work: the optimizer spares it. This is how
"battery condition" enters the decision.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import linprog

from .base import Controller, Decision, State
from .forecasts import ai_forecasts


def wear_cost_eur_per_kwh(ctx) -> float:
    b = ctx.scenario.battery
    soh = float(np.mean(b.soh_init)) if b.soh_init else 0.7
    efc_left = max((soh - 0.60) / max(b.aging_per_efc, 1e-9), 50.0)   # cycles left before the end of second life (60%)
    return b.price_eur_per_kwh / (2 * efc_left)


def solve_plan(pv, load, pbuy, psell, e0, e_min, e_max, p_ch, p_dis, eta_c, eta_d, wear, terminal_value):
    """Solves the plan over H hours. Returns (c, d, g, e, E), or None on failure."""
    H = len(pv)
    # Variable order: g(0..H-1), e, c, d, E(1..H)
    n = 5 * H
    ig, ie, ic, idd, iE = (np.arange(H) + k * H for k in range(5))
    cost = np.zeros(n)
    # Slight preference for "now" (0.2% per hour): without it, when several hours
    # have the same price, the solver may postpone discharging until later... at every
    # replanning, and never discharge (receding-horizon "procrastination").
    disc = 1 - 0.002 * np.arange(H)
    cost[ig] = np.asarray(pbuy) * disc
    cost[ie] = -np.asarray(psell) * disc
    cost[ic] = wear
    cost[idd] = wear
    cost[iE[-1]] = -terminal_value
    A = np.zeros((2 * H, n))
    b = np.zeros(2 * H)
    for t in range(H):
        # balance: g - e - c + d = load - pv
        A[t, ig[t]], A[t, ie[t]], A[t, ic[t]], A[t, idd[t]] = 1, -1, -1, 1
        b[t] = load[t] - pv[t]
        # battery: E_t - E_{t-1} - eta_c c + d/eta_d = 0
        r = H + t
        A[r, iE[t]] = 1
        if t > 0:
            A[r, iE[t - 1]] = -1
        A[r, ic[t]] = -eta_c
        A[r, idd[t]] = 1 / eta_d
        b[r] = e0 if t == 0 else 0.0
    bounds = ([(0, None)] * H + [(0, None)] * H + [(0, p_ch)] * H + [(0, p_dis)] * H + [(e_min, e_max)] * H)
    res = linprog(cost, A_eq=A, b_eq=b, bounds=bounds, method="highs")
    if not res.success:
        return None
    x = res.x
    return x[ic], x[idd], x[ig], x[ie], x[iE]


class Optimizer(Controller):
    name = "3_optimizer_ai"
    label = "Optimizer + AI forecasts"
    family = "optimization + AI"
    uses_ai = True
    description = ("Model predictive control: every hour, an optimal 24 h plan (linear programming) "
                   "based on ML forecasts of production and consumption, prices and wear.")

    def __init__(self, replan_every_h: int = 1, **kw):
        super().__init__(**kw)
        self.replan_every_h = replan_every_h

    def setup(self, ctx, meter) -> None:
        super().setup(ctx, meter)
        self.fc = ai_forecasts(ctx, meter)
        self.cfg = ctx.scenario.battery
        self.wear = wear_cost_eur_per_kwh(ctx)
        self._plan = None
        self._plan_t = None

    def decide(self, state: State) -> Decision:
        hz = state.horizon
        if self._plan is None or (state.t - self._plan_t).total_seconds() / 3600 >= self.replan_every_h:
            cap = state.capacity_kwh
            e_min, e_max = self.cfg.soc_min * cap, self.cfg.soc_max * cap
            e0 = float(np.clip(state.energy_kwh, e_min, e_max))
            # Value of the energy left at the end of the horizon: what it will be worth at the worst time
            # to use it later (lowest price × efficiency − wear). Too high, and the
            # battery would never empty; zero, and it would pointlessly empty before midnight.
            terminal = max(0.0, float(hz["price_buy"].min()) * self.cfg.eta_discharge - self.wear)
            # NOMINAL power of the bank (not the current margin: an empty battery
            # cannot discharge NOW, but will be able to after charging).
            p_name = sum(self.cfg.p_max_kw * min(1.0, s / 0.8) for s in self.cfg.soh_init)
            plan = solve_plan(hz["pv_fc"].to_numpy(), hz["load_fc"].to_numpy(), hz["price_buy"].to_numpy(),
                              hz["price_sell"].to_numpy(), e0, e_min, e_max, p_name, p_name,
                              self.cfg.eta_charge, self.cfg.eta_discharge, self.wear, terminal)
            self.bump("plans")
            if plan is None:
                self.bump("plans_failed")
                return Decision(reason="solver failure: self-consumption")
            self._plan, self._plan_t = plan, state.t
        k = int((state.t - self._plan_t).total_seconds() // 3600)
        c, d, g, e, E = self._plan
        surplus_fc = max(0.0, float(hz["pv_fc"].iloc[0] - hz["load_fc"].iloc[0]))
        grid_charge = max(0.0, float(c[k]) - surplus_fc)
        max_dis = float(d[k]) * 1.25 if d[k] > 0.05 else 0.0     # small margin for forecast error
        # If the current hour is (nearly) the most expensive of the horizon, discharging now is worth
        # as much as later: allow it, so as not to buy at a high price while holding on to energy.
        prices = hz["price_buy"].to_numpy()
        if prices[0] >= 0.98 * prices.max():
            max_dis = max(max_dis, state.max_discharge_kw)
        return Decision(grid_charge_kw=grid_charge if grid_charge > 0.05 else 0.0, max_discharge_kw=max_dis,
                        reason=f"plan: charge {c[k]:.1f} / discharge {d[k]:.1f} kW")
