"""
sizing.py: how many panels and how many batteries for this building?

We vary:
  - the panels' peak power (kWp);
  - the number of second-life battery packs;
and for each combination we simulate the period, then extrapolate to a full year:
  - self-sufficiency rate;
  - annual bill, and savings compared with "buying everything from the grid";
  - investment (panels + batteries) and simple PAYBACK TIME;
  - CO2 avoided.

⚠️ Annual extrapolation from a short period: should be repeated over
several periods (one per season) for a real sizing study.
Investment prices are orders of magnitude TO BE CHECKED.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd

from ..measure.meter import Meter
from ..control import SelfConsumptionRule, NoBattery
from .kpi import kpis
from .simulation import simulate


def sweep(ctx, kwc_list=(0, 10, 20, 30, 40, 60), packs_list=(0, 1, 2, 3, 4), controller_factory=None) -> pd.DataFrame:
    scn = ctx.scenario
    base_kwc = scn.pv.kwc if scn.pv.kwc > 0 else 1.0
    b = scn.battery
    cap0 = b.capacities_kwh[0] if b.capacities_kwh else 40.0
    soh0 = float(np.mean(b.soh_init)) if b.soh_init else 0.75
    sim = ctx.sim_index
    year = 365 / (len(sim) / 24)
    load = ctx.load_total.loc[sim]
    grid_only_cost = float((load * ctx.price_buy.loc[sim]).sum()) * year
    grid_only_co2 = float((load * ctx.grid["co2_g_per_kwh"].loc[sim]).sum()) / 1000 * year
    rows = []
    for kwc in kwc_list:
        for n in packs_list:
            bcfg = replace(b, capacities_kwh=[cap0] * n, soh_init=[soh0] * n)
            ctrl = (controller_factory() if controller_factory else SelfConsumptionRule()) if n > 0 else NoBattery()
            ctrl.setup(ctx, None)
            s, bank = simulate(ctx, ctrl, Meter(scn.compute, watch=()), battery=bcfg, pv_scale=kwc / base_kwc)
            k = kpis(s, ctx, bank, None, bcfg)
            capex = kwc * scn.pv.price_eur_per_kwc + n * cap0 * soh0 * b.price_eur_per_kwh
            cost_y = (k["bill_eur"] + k["wear_eur"]) * year
            saving = grid_only_cost - cost_y
            rows.append({
                "kwc": kwc, "packs": n, "storage_kwh": round(n * cap0 * soh0, 1),
                "self_sufficiency_pct": 100 * (k["self_sufficiency"] if not np.isnan(k["self_sufficiency"]) else 0),
                "self_consumption_pct": 100 * k["self_consumption"] if kwc > 0 else np.nan,
                "bill_year_eur": cost_y, "savings_year_eur": saving, "investment_eur": capex,
                "payback_years": capex / saving if saving > 0 else np.inf,
                "co2_avoided_kg_year": grid_only_co2 - k["co2_kg"] * year,
            })
    return pd.DataFrame(rows)
