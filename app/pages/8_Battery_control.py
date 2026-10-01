"""Page 8 · Battery control — see docs/modules/. Launch: quartier interface"""

from _common import run_button, scenario_sidebar, show_result, test_box

import streamlit as st

from quartierflex.demo import demo_control

st.set_page_config(page_title="8 · Battery control (self-consumption)", layout="wide")
scn = scenario_sidebar()
st.title("8 · Battery control (self-consumption)")
st.markdown("""**What this module does**: compares the "brains" that decide when to charge and discharge:
- **No battery** (baseline);
- **Self-consumption rule**: stores the surplus, gives it back when there is a shortfall. No forecast;
- **Tariff rule**: charges at night during off-peak hours if tomorrow's sun will not be enough;
- **Optimizer + AI**: AI forecasts + linear optimization over 24 h, which takes into account prices and
  **battery wear** (the more worn a battery is, the more it costs to make it work).

LLM agents are tested on page **11 · Assessment** (they need Ollama and computing time).""")
keys = st.multiselect("Strategies", ["no_battery", "rule", "tariff_rule", "optimizer"],
                      default=["no_battery", "rule", "tariff_rule", "optimizer"])
show = st.selectbox("Strategy to show in detail", keys or ["rule"])
res = run_button("control", fn=lambda: demo_control(scn, keys=tuple(keys), show=show))
if res is not None:
    show_result(res)
test_box("control")
