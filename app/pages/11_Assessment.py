"""Page 11 · Assessment — brings all the modules together and answers the challenge question."""

from _common import run_button, scenario_sidebar, show_result, test_box

import pandas as pd
import streamlit as st

from quartierflex.aggregator import run_demand_response, table
from quartierflex.assessment import load_inputs, run_all, summary_table
from quartierflex.demo import demo_assessment

st.set_page_config(page_title="11 · Assessment: does the AI save more than it consumes?", layout="wide")
scn = scenario_sidebar()
st.title("11 · Assessment: does the AI save more than it consumes?")
st.markdown(
    """
> **net AI gain** = profit of the strategy **with AI** − profit of the best strategy **without AI**
> − **cost of the energy consumed by the AI** (measured on this PC)

Profit = RTE payment (− penalties) + savings on the bill − battery wear. We also look at
**comfort** (extra degree-hours of cold), **rebound** and **CO2**.
"""
)

tab1, tab2, tab3 = st.tabs(["Demand response: net AI balance", "Self-consumption (battery control)", "Sizing"])

with tab1:
    c1, c2 = st.columns(2)
    llm_kind = c1.selectbox("Agent's LLM", ["fake", "ollama"], help="fake = dummy LLM (quick test, costs not real)")
    model = c2.text_input("Ollama model", "qwen2.5:1.5b")
    keys = st.multiselect("Strategies", ["none", "cut_all", "round_robin", "prepared_round_robin", "optimizer", "agent"],
                          default=["none", "cut_all", "round_robin", "prepared_round_robin", "optimizer", "agent"])
    if llm_kind == "ollama" and "agent" in keys:
        st.warning("The agent makes 2 to 4 LLM calls per RTE request: expect a few minutes. Close other applications.")
    out = run_button("assessment_eff", "Run the assessment", fn=lambda: run_demand_response(scn, keys, llm_kind=llm_kind,
                                                                               llm_model=model, log=lambda *a: None))
    if out:
        st.caption("Sources: " + " · ".join(f"{k}: {v}" for k, v in out["meta"]["sources"].items()))
        if out["meta"].get("note"):
            st.info(out["meta"]["note"])
        if any("SYNTH" in str(v) for v in out["meta"]["sources"].values()) or (llm_kind == "fake" and "agent" in keys):
            st.warning("Synthetic data or dummy LLM: illustrative results, not publishable.")
        st.markdown("### RTE requests")
        st.dataframe(pd.DataFrame(out["requests"])[["start", "end", "level", "volume_kw"]])
        st.markdown("### Strategy comparison")
        st.dataframe(table(out).T)
        rows = []
        for r in out["results"].values():
            b = r.get("assessment_ai")
            if b:
                rows.append({"AI strategy": r["label"], "compared with": r.get("reference_ai"),
                             "gross gain €": round(b["gain_raw_eur"], 2), "AI cost €": round(b["cost_ai_eur"], 6),
                             "NET gain €": round(b["gain_net_eur"], 2), "comfort gained °C·h": round(b["comfort_wins_degh"], 1),
                             "CO2 net kg": round(b["co2_net_kg"], 2), "AI mWh": round(1e6 * r["kpi"]["ai_kwh"], 2),
                             "kWh shed per Wh of AI": round(b["kwh_shed_per_wh_ai"]),
                             "verdict": "✓ the AI pays off" if b["gain_net_eur"] > 0 else "✕ the AI does not pay off"})
        if rows:
            st.markdown("### Net AI balance")
            st.dataframe(pd.DataFrame(rows))
        name = st.selectbox("Show a strategy's curve", list(out["series"]))
        s = pd.DataFrame(out["series"][name]).set_index("time")
        ref = pd.DataFrame(out["series"]["0_none"]).set_index("time")
        s.index = ref.index = pd.to_datetime(s.index, utc=True).tz_convert("Europe/Paris")
        st.line_chart(pd.DataFrame({"grid draw (strategy)": s["grid_draw_kw"], "grid draw without shedding": ref["grid_draw_kw"],
                                    "target": ref["grid_draw_kw"] - ref["request_kw"].where(ref["request_kw"] > 0)}))
        if out.get("traces"):
            with st.expander("Inside the LLM agent's head (traces)"):
                st.json(out["traces"])

with tab2:
    st.markdown("The previous project: batteries serving **self-consumption** (hour by hour, outside demand response).")
    c1, c2, c3 = st.columns(3)
    llm2 = c1.selectbox("LLM", ["fake", "ollama"], key="llm2")
    model2 = c2.text_input("Ollama model", "qwen2.5:1.5b", key="model2")
    llm_days = c3.slider("Days tested for the LLM agents", 1, 3, 1)
    keys2 = st.multiselect("Strategies", ["no_battery", "rule", "tariff_rule", "optimizer", "agent_hourly", "agent_route"],
                           default=["no_battery", "rule", "tariff_rule", "optimizer"], key="keys2")
    out2 = run_button("assessment_ai", "Run the self-consumption assessment",
                      fn=lambda: run_all(scn, keys2, llm_kind=llm2, llm_model=model2, llm_days=llm_days, ctx=load_inputs(scn)))
    if out2:
        st.dataframe(summary_table(out2))

with tab3:
    st.markdown("How many panels and batteries? (self-consumption rule, extrapolated to a full year)")
    kwc = st.multiselect("Panel peak powers tested (kWp)", [0, 30, 60, 90, 120, 180, 240], default=[0, 60, 120, 180])
    packs = st.multiselect("Numbers of packs tested", list(range(0, 11)), default=[0, 2, 4, 6, 8])
    res = run_button("dim", "Run the sizing",
                     fn=lambda: demo_assessment(scn, tuple(sorted(kwc)), tuple(sorted(packs))))
    if res is not None:
        show_result(res)

test_box("assessment")
