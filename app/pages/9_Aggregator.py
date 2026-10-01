"""Page 9 · Aggregator — see docs/modules/09_aggregator.md. Launch: quartier interface"""

from _common import run_button, scenario_sidebar, show_result, test_box

import streamlit as st

from quartierflex.demo import demo_aggregator

st.set_page_config(page_title="9 · Aggregator: the AI juggling sheddable groups", layout="wide")
scn = scenario_sidebar()
st.title("9 · Aggregator: the AI juggling sheddable groups")
st.markdown("""**What this module does**: receives RTE's request and decides, **every 15 minutes**, which appliances
to cut and what the batteries do. The same **comfort rule** for everyone: a heater is never cut for more than
30 min in a row, then stays on for 30 min.

| Strategy | AI? | Idea |
|---|---|---|
| Cut everything | no | everyone at the same time, battery at full power |
| Round-robin | no | battery, cars, water heaters, then heaters in turn until the volume is reached |
| Prepared round-robin | no | + battery charged the night before, 30% margin |
| **AI optimizer** | yes, light | learned thermal model + "shadow homes" (what they would consume without a cut) + linear optimization; preheating; retirees spared; EV charging shifted to off-peak hours |
| LLM agent | yes, heavy | page **11 · Assessment** (requires Ollama) |

We measure: **delivered** volume, **rebound** after the request, **comfort**, € (RTE payment, Tempo bill,
battery wear), CO2… and the energy consumed by the AI.""")
keys = st.multiselect("Strategies", ["none", "cut_all", "round_robin", "prepared_round_robin", "optimizer"],
                      default=["none", "cut_all", "round_robin", "prepared_round_robin", "optimizer"])
res = run_button("aggregator", fn=lambda: demo_aggregator(scn, keys=tuple(keys)))
if res is not None:
    show_result(res)
test_box("aggregator")
