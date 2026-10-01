"""Page 1 · Weather — see docs/modules/. Launch: quartier interface"""

from _common import run_button, scenario_sidebar, show_result, test_box

import streamlit as st

from quartierflex.demo import demo_weather

st.set_page_config(page_title="1 · Weather", layout="wide")
scn = scenario_sidebar()
st.title("1 · Weather")
st.markdown("""**What this module does**: fetches hourly weather in Valence, both **observed** (what actually happened)
and **forecast the day before** (what an AI is allowed to know). It also computes the sun's position.

**Why it matters**: solar production depends on sunshine; weather forecasts are never perfect,
and that error is exactly what the forecasting AI will have to deal with.

**What to look at**: the gap between the observed and forecast curves, especially on cloudy days.""")

res = run_button("weather", fn=lambda: demo_weather(scn))
if res is not None:
    show_result(res)
test_box("weather")
