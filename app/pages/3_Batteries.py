"""Page 3 · Second-life batteries — see docs/modules/. Launch: quartier interface"""

from _common import run_button, scenario_sidebar, show_result, test_box

import streamlit as st

from quartierflex.demo import demo_batteries

st.set_page_config(page_title="3 · Second-life batteries", layout="wide")
scn = scenario_sidebar()
st.title("3 · Second-life batteries")
st.markdown("""**What this module does**: simulates a reused car battery pack over several months.
- **SOC** (state of charge): the BMS estimates it by counting energy, which **drifts**, then corrects it using the
  voltage measured at rest.
- **SOH** (state of health): poorly known when the pack is recovered; the AI **learns** it from the measurements
  (robust regression) and predicts the end of second life (SOH 60%).
- Aging from use (cycles) and from time; reduced power when the battery is worn.""")
c1, c2, c3 = st.columns(3)
days = c1.slider("Simulated duration (days)", 30, 730, 365, step=30)
soh_true = c2.slider("Actual SOH of the pack", 0.60, 0.90, 0.75, step=0.01)
soh_guess = c3.slider("SOH assumed at installation", 0.60, 0.95, 0.80, step=0.01)
res = run_button("batteries", fn=lambda: demo_batteries(scn, days=days, soh_true=soh_true, soh_guess=soh_guess))
if res is not None:
    show_result(res)
test_box("batteries")
