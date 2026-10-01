"""Page 10 · Measuring the AI's energy — see docs/modules/. Launch: quartier interface"""

from _common import run_button, scenario_sidebar, show_result, test_box

import streamlit as st

from quartierflex.demo import demo_measure

st.set_page_config(page_title="10 · Measuring the AI's energy", layout="wide")
scn = scenario_sidebar()
st.title("10 · Measuring the AI's energy")
st.markdown("""**What this module does**: measures the energy consumed by the AI's computations **on this PC**
(CPU time × processor power), separating what is paid once (training) from what is
paid every hour (forecasts, optimization, LLM calls).
For an LLM running in Ollama, the "whole machine" measurement is used (method validated on Bilan Net).""")

res = run_button("measure", fn=lambda: demo_measure(scn))
if res is not None:
    show_result(res)
test_box("measure")
