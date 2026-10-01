"""
assessment.py: did demand response work, at what cost to comfort, and is the AI worth what it consumes?

Indicators (for each strategy, compared with "no shedding" over the same period)
--------------------------------------------------------------------------------
- DELIVERED shedding = grid draw without shedding − grid draw with it, during the request (kWh)
- fulfillment rate = delivered / requested; hold rate = share of 15-min steps holding ≥ 90 % of the volume
- REBOUND = overconsumption in the 2 h after the request (kWh), and peak reached (kW)
- COMFORT: degree-hours below the tolerated minimum (occupied, electric heating), on top of the
  baseline; cold water (kWh missing); cars not charged at departure (kWh missing)
- MONEY: RTE payment (− penalties if not held), bill savings (Tempo),
  battery wear -> PROFIT
- CO2: Σ (avoided grid draw × grid intensity at that hour), rebound and recharging included
- AI COST: measured energy (learning + forecasts + optimization + LLM), in kWh, €, CO2

Net AI balance = profit of the AI strategy − profit of the best strategy WITHOUT AI − AI cost.
"""

from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd

from ..config import RESULTS_DIR, Scenario
from ..measure import Meter
from .data import load_district
from .simulation import run_simulation
from .strategies import ALL, set_volumes, make_strategy


def _bill(s: pd.DataFrame, dt: float) -> float:
    imp = s["grid_draw_kw"].clip(lower=0)
    exp = (-s["grid_draw_kw"]).clip(lower=0)
    return float(((imp * s["price_buy"]) - (exp * s["price_sell"])).sum() * dt)


def _wear_eur(bank, cfg) -> float:
    return float(sum(l * p.cap_name_kwh for l, p in zip(bank.soh_loss(), bank.packs)) * cfg.price_eur_per_kwh)


def kpis_demand_response(res: dict, ref: dict, ctx, ai_kwh: float = 0.0, llm_calls: int = 0) -> dict:
    scn, dt = ctx.scenario, ctx.dt
    s, r = res["series"], ref["series"]
    eff = r["grid_draw_kw"] - s["grid_draw_kw"]
    E_req = E_delivered = E_real = rebound = 0.0
    hold, peaks, per_request = [], [], []
    for d in ctx.requests:
        m = (s.index >= d.start) & (s.index < d.end)
        e = float(eff[m].sum() * dt)
        req = d.energy_kwh
        # Checked at each 15-min step (like RTE, which verifies power at every step):
        # an "excess" in one step does not pay for a "shortfall" in another.
        delivered = float(eff[m].clip(lower=0, upper=d.volume_kw).sum() * dt)
        E_req, E_delivered, E_real = E_req + req, E_delivered + delivered, E_real + e
        hold.append(float((eff[m] >= 0.9 * d.volume_kw).mean()) if m.any() else np.nan)
        w = (s.index >= d.end) & (s.index < d.end + pd.Timedelta("2h"))
        rb = float((-eff[w]).clip(lower=0).sum() * dt)
        rebound += rb
        day = s.index.date == d.start.date()
        peaks.append(float(s["grid_draw_kw"][w].max() - r["grid_draw_kw"][day].max()) if w.any() else 0.0)
        per_request.append({"slot": f"{d.start:%a %d/%m %Hh}-{d.end:%Hh}", "level": d.level,
                            "requested kW": d.volume_kw, "avg shed kW": round(e / d.duration_h, 1),
                            "delivered %": round(100 * delivered / req, 0) if req else np.nan,
                            "rebound kWh": round(rb, 1)})
    cfg_e = scn.demand_response
    revenue = (cfg_e.payment_eur_mwh * E_delivered - cfg_e.penalty_eur_mwh * (E_req - E_delivered)) / 1000
    if res is ref:
        revenue = 0.0                         # the baseline committed to nothing
    fact, fact_ref = _bill(s, dt), _bill(r, dt)
    us, us_ref = _wear_eur(res["bank"], scn.battery), _wear_eur(ref["bank"], scn.battery)
    price_avg, co2_avg = float(s["price_buy"].mean()), float(s["co2_g_kwh"].mean())
    profit = revenue + (fact_ref - fact) - (us - us_ref)
    return {
        "requests": len(ctx.requests), "request_kwh": E_req, "shed_kwh": E_real, "delivered_kwh": E_delivered,
        "fulfilment": E_delivered / E_req if E_req else np.nan,
        "hold": float(np.nanmean(hold)) if hold else np.nan,
        "rebound_kwh": rebound, "overpeak_after_kw": max(peaks) if peaks else 0.0,
        "discomfort_added_degh": float(s["discomfort_degh"].sum() - r["discomfort_degh"].sum()),
        "t_min_c": float(s.loc[s["in_request"] | (s.index.hour.isin([18, 19, 20])), "t_in_min"].min()),
        "cold_water_kwh": res["cold_water_kwh"] - ref["cold_water_kwh"],
        "ev_missing_kwh": res["ev_missing_kwh"] - ref["ev_missing_kwh"],
        "revenue_rte_eur": revenue, "savings_bill_eur": fact_ref - fact, "wear_eur": us - us_ref,
        "profit_eur": profit,
        "co2_avoided_kg": float((eff * s["co2_g_kwh"]).sum() * dt / 1000),
        "consumption_total_kwh": float(s["grid_draw_kw"].clip(lower=0).sum() * dt),
        "ai_kwh": ai_kwh, "ai_eur": ai_kwh * price_avg, "ai_co2_kg": ai_kwh * co2_avg / 1000, "llm_calls": llm_calls,
        "per_request": per_request,
    }


def run_demand_response(scn: Scenario, keys=None, llm_kind: str = "fake", llm_model: str = "qwen2.5:1.5b",
                   ctx=None, save: bool = True, log=print) -> dict:
    from ..agent.llm import make_llm

    keys = [k for k in (keys or ALL) if k != "none"]
    t0 = time.time()
    ctx = ctx or load_district(scn)
    log(f"District: {ctx.portfolio.n} homes, {len(ctx.index)} 15-min steps, {len(ctx.requests)} RTE request(s). {ctx.note}")
    # 1. Baseline without shedding, then RTE sets the volumes from the declared flexibility
    ref_strat = make_strategy("none")
    m0 = Meter(scn.compute, watch=())
    ref_strat.setup(ctx, m0)
    ref = run_simulation(ctx, ref_strat, m0, seed=scn.seed)
    m_decl = Meter(scn.compute, watch=())
    with m_decl.measure("shared"):
        reports = set_volumes(ctx, ref)
    decl_kwh = m_decl.report()["shared"]["energy_kwh"]
    ref["series"]["request_kw"] = [next((d.volume_kw for d in ctx.requests if d.active(t)), 0.0) for t in ref["series"].index]
    out = {"meta": {"scenario": {"start": scn.start, "days": scn.days, "homes": ctx.portfolio.n,
                                 "packs": len(scn.battery.capacities_kwh), "kwc": scn.pv.kwc},
                    "sources": ctx.sources(), "note": ctx.note, "llm": llm_kind if "agent" in keys else None,
                    "cost_learning_kwh": ctx.cost_learning["energy_kwh"], "cost_declaration_kwh": decl_kwh},
           "requests": [d.dict() for d in ctx.requests], "results": {}, "series": {}, "traces": {},
           "reports": [{"groups": r["groups"].to_dict("records"), "batteries": r["batteries"]} for r in reports]}
    out["series"]["0_none"] = ref["series"].drop(columns=["reason"]).reset_index().astype({"time": str}).to_dict("list")
    ref_k = kpis_demand_response(ref, ref, ctx)
    out["results"]["0_none"] = {"label": ref_strat.label, "family": "reference", "kpi": ref_k, "counters": {}}
    # 2. Each strategy
    for key in keys:
        llm = make_llm(llm_kind, llm_model) if key == "agent" else None
        st = make_strategy(key, llm)
        meter = Meter(scn.compute, watch=("ollama",) if (key == "agent" and llm_kind == "ollama") else ())
        with meter.measure("setup"):
            st.setup(ctx, meter)
        t1 = time.time()
        res = run_simulation(ctx, st, meter, seed=scn.seed)
        rep = meter.report()
        ai = sum(max(b["energy_kwh"], b["energy_kwh_llm_active"]) for b in rep.values()) + decl_kwh
        if st.uses_ai:
            ai += ctx.cost_learning["energy_kwh"]
        calls = int(sum(b["llm_calls"] for b in rep.values()))
        k = kpis_demand_response(res, ref, ctx, ai, calls)
        out["results"][st.name] = {"label": st.label, "family": st.family, "description": st.description,
                                     "kpi": k, "counters": st.counters, "duration_s": round(time.time() - t1, 1)}
        out["series"][st.name] = res["series"].drop(columns=["reason"]).reset_index().astype({"time": str}).to_dict("list")
        if getattr(st, "traces", None):
            out["traces"][st.name] = st.traces
        log(f"  {st.label}: delivered {100 * k['fulfilment']:.0f} %, profit {k['profit_eur']:.2f} €, "
            f"AI {1000 * ai:.3f} Wh, {calls} LLM calls ({time.time() - t1:.0f} s)")
    # 3. Net AI balance: against the best strategy WITHOUT AI
    without_ai = {n: r for n, r in out["results"].items() if r["family"] == "no AI"}
    if without_ai:
        best = max(without_ai, key=lambda n: without_ai[n]["kpi"]["profit_eur"])
        kb = without_ai[best]["kpi"]
        for n, r in out["results"].items():
            if r["family"].startswith("AI"):
                kk = r["kpi"]
                r["reference_ai"] = without_ai[best]["label"]
                r["assessment_ai"] = {
                    "gain_raw_eur": kk["profit_eur"] - kb["profit_eur"], "cost_ai_eur": kk["ai_eur"],
                    "gain_net_eur": kk["profit_eur"] - kb["profit_eur"] - kk["ai_eur"],
                    "comfort_wins_degh": kb["discomfort_added_degh"] - kk["discomfort_added_degh"],
                    "co2_net_kg": kk["co2_avoided_kg"] - kb["co2_avoided_kg"] - kk["ai_co2_kg"],
                    "kwh_shed_per_wh_ai": kk["delivered_kwh"] / max(1000 * kk["ai_kwh"], 1e-9),
                }
    out["meta"]["duration_total_s"] = round(time.time() - t0, 1)
    if save:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        (RESULTS_DIR / "demand_response.json").write_text(json.dumps(out, ensure_ascii=False, indent=1, default=float),
                                                     encoding="utf-8")
    out["_ctx"] = ctx
    return out


def table(out: dict) -> pd.DataFrame:
    rows = []
    for n, r in out["results"].items():
        k = r["kpi"]
        b = r.get("assessment_ai", {})
        rows.append({
            "strategy": r["label"], "delivered %": round(100 * k["fulfilment"], 0) if n != "0_none" else 0,
            "hold %": round(100 * k["hold"], 0) if n != "0_none" else 0,
            "rebound kWh": round(k["rebound_kwh"], 1), "overpeak after kW": round(k["overpeak_after_kw"], 1),
            "added discomfort °C·h": round(k["discomfort_added_degh"], 1), "min T °C": round(k["t_min_c"], 1),
            "cold water kWh": round(k["cold_water_kwh"], 1), "EV not charged kWh": round(k["ev_missing_kwh"], 1),
            "RTE revenue €": round(k["revenue_rte_eur"], 2), "bill savings €": round(k["savings_bill_eur"], 2),
            "battery wear €": round(k["wear_eur"], 2), "PROFIT €": round(k["profit_eur"], 2),
            "CO2 avoided kg": round(k["co2_avoided_kg"], 1), "AI mWh": round(1e6 * k["ai_kwh"], 2),
            "LLM calls": k["llm_calls"], "NET AI gain €": round(b["gain_net_eur"], 2) if b else "",
        })
    return pd.DataFrame(rows).set_index("strategy")
