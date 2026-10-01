"""Page 7 · Demand response: RTE requests — see docs/modules/07_demand_response.md. Launch: quartier interface"""

from _common import run_button, scenario_sidebar, show_result, test_box

import streamlit as st

from quartierflex.demo import demo_demand_response

st.set_page_config(page_title="7 · Demand response: what RTE asks", layout="wide")
scn = scenario_sidebar()
st.title("7 · Demand response: what RTE asks, what the district declares")
st.markdown("""**How it works in France**
- On very cold days (**stress** days: Tempo white/red, EcoWatt orange/red), national consumption
  rises, especially **in the evening (6 pm - 8 pm)** and in the morning (7 am - 9 am).
- **RTE does not control homes**: it activates an **aggregator** (a "demand-response operator") for a **volume**
  (kW) over a time slot. The aggregator splits that volume among its homes (module 9).
- So that RTE can rely on it, the district **declares** its flexibility the day before: sheddable kW **per group**
  (forecast by the AI of module 5) and **the state of the batteries** (charge, health, power) → RTE requests a share
  (60% by default) of that total.

⚠️ The Tempo color is **approximated** from real national consumption (RTE éCO2mix). If there is no stress
day in the period, a request is simulated on the busiest day (this is flagged).""")
res = run_button("demand_response", fn=lambda: demo_demand_response(scn))
if res is not None:
    show_result(res)
test_box("demand_response")
