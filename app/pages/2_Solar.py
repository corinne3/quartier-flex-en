"""Page 2 · Solar — see docs/modules/. Launch: quartier interface"""

from _common import run_button, scenario_sidebar, show_result, test_box

import streamlit as st

from quartierflex.demo import demo_solar

st.set_page_config(page_title="2 · Solar", layout="wide")
scn = scenario_sidebar()
st.title("2 · Solar")
st.markdown("""**What this module does**:
- computes the panels' output from the weather (physical model: sun, tilt, temperature);
- simulates the **meter** (real measurements drift from theory: soiling, faults, noise);
- compares 3 production forecasts: "same as yesterday", physics on forecast weather, and **AI** (gradient boosting
  that learns from the measurement history).

**Sizing variable**: the panels' peak power (kWp), in the left sidebar.""")

res = run_button("solar", fn=lambda: demo_solar(scn))
if res is not None:
    show_result(res)
test_box("solar")
