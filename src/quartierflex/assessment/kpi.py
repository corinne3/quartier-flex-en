"""
kpi.py: the indicators of a simulation, and the NET AI BALANCE.

Physical and economic indicators (over the period, then annualized)
-------------------------------------------------------------------
- SELF-CONSUMPTION rate  = share of solar production consumed on site
                           (directly or via the battery)
- SELF-SUFFICIENCY rate  = share of consumption covered by solar
- grid purchase / resale (kWh), net bill (€), CO2 of purchases (kg)
- battery wear: loss of state of health, converted to € (lost capacity × price)

Cost of the AI (measurement module)
-----------------------------------
Energy consumed by the forecasts, the optimizer and the LLM, converted to
€ (at the average purchase price) and to CO2 (at the average grid intensity).

Net AI balance (THE question of the challenge)
----------------------------------------------
    AI gain = (bill of the best strategy WITHOUT AI − bill of the AI strategy)
              − cost of the energy consumed by the AI
Same calculation in purchased kWh and in CO2. Everything is also expressed per home.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def kpis(sim: pd.DataFrame, ctx, bank, meter_report: dict | None, battery_cfg=None) -> dict:
    days = max(len(sim) / 24, 1e-9)
    year = 365 / days
    pv, load = sim["pv_kwh"].sum(), sim["consumption_kwh"].sum()
    imp, exp = sim["purchase_kwh"].sum(), sim["resale_kwh"].sum()
    cost = float((sim["purchase_kwh"] * sim["price_buy"]).sum() - (sim["resale_kwh"] * sim["price_sell"]).sum())
    co2 = float((sim["purchase_kwh"] * sim["co2_g_kwh"]).sum() / 1000)
    cfg = battery_cfg or ctx.scenario.battery
    wear_eur, soh_loss = 0.0, 0.0
    if bank is not None:
        losses = bank.soh_loss()
        soh_loss = float(np.mean(losses))
        wear_eur = float(sum(l * p.cap_name_kwh for l, p in zip(losses, bank.packs)) * cfg.price_eur_per_kwh)
    comp_kwh = llm_calls = tokens = 0.0
    if meter_report:
        comp_kwh = sum(b["energy_kwh"] for b in meter_report.values())
        llm_calls = sum(b["llm_calls"] for b in meter_report.values())
        tokens = sum(b["tokens_in"] + b["tokens_out"] for b in meter_report.values())
    mean_price = float(sim["price_buy"].mean())
    mean_co2 = float(sim["co2_g_kwh"].mean())
    n = ctx.scenario.building.n_homes
    return {
        "days": days,
        "production_kwh": pv, "consumption_kwh": load, "purchase_kwh": imp, "resale_kwh": exp,
        "self_consumption": (pv - exp) / pv if pv > 0 else np.nan,
        "self_sufficiency": (load - imp) / load if load > 0 else np.nan,
        "bill_eur": cost, "co2_kg": co2,
        "wear_soh_pts": 100 * soh_loss, "wear_eur": wear_eur,
        "ai_kwh": comp_kwh, "ai_eur": comp_kwh * mean_price, "ai_co2_kg": comp_kwh * mean_co2 / 1000,
        "llm_calls": llm_calls, "tokens": tokens,
        # Annualized and per home
        "bill_eur_year_home": cost * year / n,
        "purchase_kwh_year_home": imp * year / n,
        "co2_kg_year_home": co2 * year / n,
        "ai_kwh_year_home": comp_kwh * year / n,
    }


def net_ai_balance(k_ai: dict, k_ref: dict) -> dict:
    """Net balance of an AI strategy compared with the best strategy without AI (same period)."""
    save_eur = (k_ref["bill_eur"] + k_ref["wear_eur"]) - (k_ai["bill_eur"] + k_ai["wear_eur"])
    save_kwh = k_ref["purchase_kwh"] - k_ai["purchase_kwh"]
    save_co2 = k_ref["co2_kg"] - k_ai["co2_kg"]
    return {
        "gain_raw_eur": save_eur, "cost_ai_eur": k_ai["ai_eur"], "gain_net_eur": save_eur - k_ai["ai_eur"],
        "gain_raw_kwh": save_kwh, "cost_ai_kwh": k_ai["ai_kwh"], "gain_net_kwh": save_kwh - k_ai["ai_kwh"],
        "gain_raw_co2_kg": save_co2, "cost_ai_co2_kg": k_ai["ai_co2_kg"], "gain_net_co2_kg": save_co2 - k_ai["ai_co2_kg"],
        "ratio_gain_over_cost_kwh": (save_kwh / k_ai["ai_kwh"]) if k_ai["ai_kwh"] > 0 else np.inf,
    }
