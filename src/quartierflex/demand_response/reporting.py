"""
reporting.py: what the district DECLARES to RTE (its flexibility), batteries included.

Why declare?
------------
RTE can only request what exists. So the day before, the aggregator sends
a forecast of what it can shed in each slot:
    - per sheddable GROUP (occupant type × appliance): kW forecast;
    - the BATTERY STATE: state of charge (SOC), state of health (SOH), real
      capacity, max power (reduced if the battery is worn) -> energy and
      power available if it is recharged before the event.
RTE then sets its request: a share (60 % by default) of the declared total.

How we forecast (the forecasting AI, appliances module)
-------------------------------------------------------
- heating: each home is SIMULATED with its LEARNED thermal model,
  from the temperature measured at the announcement, with the FORECAST weather;
  because of the rotation (30 min cut / 30 min on), only HALF of a group's
  heating power can be shed continuously;
- water heater and car: average profile learned from the measured history (hour by hour).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..appliances import APPLIANCES, Portfolio, setpoints

SHARE_SHEDDABLE = {"heating": 0.5, "water_heater": 1.0, "ev": 1.0}   # 30/30 rotation for heating


def hourly_profiles(hist: dict) -> dict:
    """Average power per hour of day and per home (water heater, car), learned from the history."""
    hours = pd.DatetimeIndex(hist["index"]).hour
    out = {}
    for a in ("water_heater", "ev"):
        df = pd.DataFrame(hist["p"][a])
        out[a] = df.groupby(hours).mean().reindex(range(24)).fillna(0).to_numpy()   # [24 × homes]
    return out


def forecast_consumption(portfolio: Portfolio, model, t_in_measure: np.ndarray, index: pd.DatetimeIndex,
                         weather_fc: pd.DataFrame, profiles: dict) -> dict:
    """Consumption forecast per appliance and per home over `index` [steps × homes]."""
    dt = (index[1] - index[0]).total_seconds() / 3600 if len(index) > 1 else 0.25
    sp, pres = setpoints(portfolio, index)
    w = weather_fc.reindex(index)
    p_ch, t_pred = model.forecast(t_in_measure, w["temp_c"].to_numpy(), sp, pres, w["ghi_wm2"].to_numpy(),
                                  portfolio.p_heating_max, dt)
    h = index.hour.to_numpy()
    return {"heating": np.where(portfolio.heating, p_ch, 0.0), "water_heater": profiles["water_heater"][h], "ev": profiles["ev"][h],
            "t_in": t_pred, "setpoint": sp, "presence": pres}


def flex_batteries(bank, duration_h: float) -> dict:
    """Battery state reported to RTE, and what they can hold over a slot."""
    cfg = bank.cfg
    packs = [{"pack": p.name, "SOC %": round(100 * p.soc, 1), "SOH %": round(100 * p.soh, 1),
              "real capacity kWh": round(p.capacity_kwh, 1), "max power kW": round(p.p_max_kw, 1)}
             for p in bank.packs]
    e_mob = sum((cfg.soc_max - cfg.soc_min) * p.capacity_kwh for p in bank.packs) * cfg.eta_discharge
    p_max = sum(p.p_max_kw for p in bank.packs)
    return {"packs": packs, "energy_available_kwh": e_mob, "power_max_kw": p_max,
            "kw_held_over_slot": min(p_max, e_mob / max(duration_h, 1e-9)),
            "energy_current_kwh": max(0.0, bank.energy_kwh - cfg.soc_min * bank.capacity_kwh) * cfg.eta_discharge}


def report_flexibility(portfolio: Portfolio, prev: dict, mask_slot: np.ndarray, bank, duration_h: float) -> dict:
    """
    The report sent to RTE for a slot: table per group + batteries.
    `prev` = forecast_consumption(...); mask_slot = steps of `prev` that fall within the slot.
    """
    rows = []
    for a in APPLIANCES:
        grp = portfolio.group(a)
        p = prev[a][mask_slot]
        for g in sorted(set(grp) - {""}):
            cols = grp == g
            kw = float(p[:, cols].sum(axis=1).mean())
            rows.append({"group": g, "appliance": a, "type": g.split("/")[0], "homes": int(cols.sum()),
                         "kW forecast": round(kw, 1), "kW sheddable": round(SHARE_SHEDDABLE[a] * kw, 1),
                         "tolerance °C": float(portfolio.tolerance[cols].min()) if a == "heating" else np.nan})
    table = pd.DataFrame(rows)
    bat = flex_batteries(bank, duration_h)
    total_app = float(table["kW sheddable"].sum()) if len(table) else 0.0
    return {"groups": table, "batteries": bat, "total_appliances_kw": total_app,
            "total_kw": total_app + bat["kw_held_over_slot"]}
