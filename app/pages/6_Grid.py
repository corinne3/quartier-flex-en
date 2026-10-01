"""Page 6 · Grid: prices and CO2 — see docs/modules/. Launch: quartier interface"""

from _common import run_button, scenario_sidebar, show_result, test_box

import streamlit as st

from quartierflex.demo import demo_grid

st.set_page_config(page_title="6 · Grid: prices and CO2", layout="wide")
scn = scenario_sidebar()
st.title("6 · Grid: prices and CO2")
st.markdown("""**What this module does**: gives, hour by hour, the **purchase price** per kWh (Base, Off-peak hours, Tempo),
the **feed-in price** for surplus, and the **CO2** of the French grid (RTE éCO2mix). Tempo white and red days are also the days when RTE requests demand response (module 7).

**Battery economics in one sentence**: buying costs ~5 times more than reselling earns,
so storing your surplus to consume it later pays off.""")

res = run_button("grid", fn=lambda: demo_grid(scn))
if res is not None:
    show_result(res)
test_box("grid")
