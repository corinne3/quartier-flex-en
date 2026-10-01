"""
simulation.py: running the building hour by hour with a control strategy.

At each hour t:
    1. real solar production, real consumption of all homes;
    2. the strategy receives the state + the forecasts for the next 24 h -> Decision;
    3. "real-time" layer (identical for all strategies):
         - if the decision requests grid charging: charge (surplus + grid);
         - otherwise, if there is solar surplus: store it;
         - otherwise: the battery supplies the shortfall, within the allowed limit;
    4. the battery applies it (bounded by SOC, power, state of health) and ages;
    5. any shortfall is bought from the grid, any remainder is sold.

All quantities are in kWh per hour (= average kW).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..batteries import BatteryBank
from ..config import BatteryConfig
from ..control.base import State
from ..control.forecasts import naive_forecasts


def horizon_frame(ctx, fc: pd.DataFrame, pv_scale: float = 1.0) -> pd.DataFrame:
    H = fc.copy()
    H["pv_fc"] = H["pv_fc"] * pv_scale
    H["price_buy"] = ctx.price_buy.reindex(H.index).to_numpy()
    H["price_sell"] = ctx.price_sell.reindex(H.index).to_numpy()
    H["co2"] = ctx.grid["co2_g_per_kwh"].reindex(H.index).to_numpy()
    return H.ffill().bfill()


def simulate(ctx, controller, meter, battery: BatteryConfig | None = None, pv_scale: float = 1.0,
             sim_index: pd.DatetimeIndex | None = None) -> tuple[pd.DataFrame, BatteryBank | None]:
    sim_index = sim_index if sim_index is not None else ctx.sim_index
    cfg = battery if battery is not None else ctx.scenario.battery
    has_batt = controller.family != "reference" and len(cfg.capacities_kwh) > 0
    # Start with an EMPTY battery (SOC min): the initial energy must not be counted as free.
    bank = BatteryBank(cfg, soc0=cfg.soc_min) if has_batt else None
    fc = getattr(controller, "fc", None)
    if fc is None:
        fc = naive_forecasts(ctx)
    H = horizon_frame(ctx, fc, pv_scale)
    pv_all = ctx.pv_true * pv_scale
    load_all = ctx.load_total
    rows = []
    for t in sim_index:
        pv, load = float(pv_all[t]), float(load_all[t])
        hz = H.loc[t: t + pd.Timedelta("23h")]
        if bank is not None:
            state = State(t, bank.soc, bank.energy_kwh, bank.capacity_kwh, bank.usable_kwh,
                          bank.max_charge_kw(), bank.max_discharge_kw(), hz)
            with meter.measure("shared"):
                dec = controller.decide(state)
            surplus = pv - load
            if dec.grid_charge_kw > 0:
                cmd = max(surplus, 0.0) + dec.grid_charge_kw
            elif surplus >= 0:
                cmd = surplus
            else:
                cmd = -min(-surplus, dec.max_discharge_kw)
            p = bank.step(cmd)
            soc, soh = bank.soc, float(np.mean([pk.soh for pk in bank.packs]))
            reason = dec.reason
        else:
            p, soc, soh, reason = 0.0, np.nan, np.nan, "no battery"
        net = load - pv + p
        rows.append({
            "time": t, "pv_kwh": pv, "consumption_kwh": load, "battery_kw": p, "soc": soc, "soh_average": soh,
            "purchase_kwh": max(net, 0.0), "resale_kwh": max(-net, 0.0),
            "price_buy": float(ctx.price_buy[t]), "price_sell": float(ctx.price_sell[t]),
            "co2_g_kwh": float(ctx.grid["co2_g_per_kwh"][t]), "reason": reason,
        })
    return pd.DataFrame(rows).set_index("time"), bank
