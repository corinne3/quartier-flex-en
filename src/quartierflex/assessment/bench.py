"""
bench.py: running all the strategies and producing the full assessment.

    results = run_all(Scenario(), keys=["no_battery", "rule", "optimizer", "agent_route"])

- Strategies without an LLM: over the whole period.
- LLM agents: over the first `llm_days` days only (one LLM call takes
  several seconds on CPU); they are compared with the rule ON THE SAME DAYS.
Output: results/latest.json (indicators, net balances, hourly series, traces).
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime

import numpy as np
import pandas as pd

from ..agent.llm import make_llm
from ..config import RESULTS_DIR, Scenario
from ..measure.meter import Meter
from ..control import LLM_KEYS, make_controller
from .data import load_inputs
from .kpi import kpis, net_ai_balance
from .simulation import simulate

log = logging.getLogger(__name__)


def _clean(x):
    if isinstance(x, (np.floating, float)):
        return None if (np.isnan(x) or np.isinf(x)) else float(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    return x


def run_all(scn: Scenario, keys: list[str], llm_kind: str = "fake", llm_model: str = "qwen2.5:1.5b",
            llm_days: int = 2, ctx=None, out_dir=RESULTS_DIR) -> dict:
    t_start = time.perf_counter()
    ctx = ctx or load_inputs(scn)
    llm = make_llm(llm_kind, llm_model) if any(k in LLM_KEYS for k in keys) else None
    if llm is not None and llm_kind == "ollama" and not llm.ping():
        raise RuntimeError(f"Ollama is not responding or model '{llm_model}' is missing: ollama pull {llm_model}")
    sub = ctx.sim_index[: 24 * llm_days]

    results, series, traces = {}, {}, {}
    ref_sub = None
    for key in keys:
        ctrl = make_controller(key, llm)
        is_llm = key in LLM_KEYS
        meter = Meter(scn.compute, watch=("ollama",) if (is_llm and llm_kind == "ollama") else ())
        log.info("▶ %s", ctrl.label)
        ctrl.setup(ctx, meter)
        idx = sub if is_llm else ctx.sim_index
        sim, bank = simulate(ctx, ctrl, meter, sim_index=idx)
        k = kpis(sim, ctx, bank, meter.report())
        k.update(name=ctrl.name, label=ctrl.label, family=ctrl.family, description=ctrl.description,
                 uses_ai=ctrl.uses_ai, uses_llm=ctrl.uses_llm, stats=dict(ctrl.stats), meter=meter.report(),
                 period="LLM sub-period" if is_llm else "full period")
        results[ctrl.name] = k
        series[ctrl.name] = {"time": sim.index.strftime("%Y-%m-%d %H:%M").tolist(),
                             **{c: sim[c].round(3).tolist() for c in ("pv_kwh", "consumption_kwh", "battery_kw", "soc",
                                                                       "purchase_kwh", "resale_kwh")}}
        if getattr(ctrl, "traces", None):
            traces[ctrl.name] = ctrl.traces
        if is_llm and ref_sub is None:  # "rule" reference on the same days
            r = make_controller("rule")
            r.setup(ctx, None)
            s2, b2 = simulate(ctx, r, Meter(scn.compute, watch=()), sim_index=sub)
            ref_sub = kpis(s2, ctx, b2, None)
        log.info("   bill %.2f € | purchase %.1f kWh | AI %.2e kWh", k["bill_eur"], k["purchase_kwh"], k["ai_kwh"])

    # Net balances: each AI strategy against the best strategy WITHOUT AI over the same period
    no_ai = [k for k in results.values() if not k["uses_ai"] and k["family"] != "reference"]
    best_ref = min(no_ai, key=lambda k: k["bill_eur"] + k["wear_eur"]) if no_ai else None
    for k in results.values():
        if not k["uses_ai"]:
            continue
        ref = ref_sub if k["uses_llm"] else best_ref
        if ref is not None:
            k["assessment_ai"] = net_ai_balance(k, ref)
            k["reference_ai"] = "Self-consumption rule (same days)" if k["uses_llm"] else best_ref["label"]

    out = {
        "meta": {"date": datetime.now().isoformat(timespec="seconds"), "start": scn.start, "days": scn.days,
                 "llm_days": llm_days, "n_homes": scn.building.n_homes, "kwc": scn.pv.kwc,
                 "batteries_kwh_new": scn.battery.capacities_kwh, "soh_init": scn.battery.soh_init,
                 "tariff": scn.tariff.option, "llm": getattr(llm, "name", None), "sources": ctx.sources(),
                 "runtime_s": round(time.perf_counter() - t_start, 1)},
        "results": {n: {k: (v if isinstance(v, (dict, str, bool)) else _clean(v)) for k, v in r.items()}
                      for n, r in results.items()},
        "series": series, "traces": traces,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "latest.json").write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    return out


def summary_table(out: dict) -> pd.DataFrame:
    rows = []
    for r in out["results"].values():
        b = r.get("assessment_ai") or {}
        rows.append({
            "strategy": r["label"], "period": r["period"],
            "self-sufficiency %": None if r["self_sufficiency"] is None else round(100 * r["self_sufficiency"], 1),
            "bill €": round(r["bill_eur"], 2), "battery wear €": round(r["wear_eur"], 2),
            "purchase kWh": round(r["purchase_kwh"], 1), "AI kWh": r["ai_kwh"], "LLM calls": r["llm_calls"],
            "net AI gain €": None if not b else round(b["gain_net_eur"], 3),
            "net AI gain kWh": None if not b else round(b["gain_net_kwh"], 3),
        })
    return pd.DataFrame(rows)
