"""Page 4 · Occupant usage — see docs/modules/. Launch: quartier interface"""

from _common import run_button, scenario_sidebar, show_result, test_box

import streamlit as st

from quartierflex.demo import demo_usage

st.set_page_config(page_title="4 · Occupant usage", layout="wide")
scn = scenario_sidebar()
st.title("4 · Occupant usage")
st.markdown("""**What this module does**:
- generates each home's consumption according to its type (family, working couple, retirees, remote worker,
  student), with weekdays / weekends, seasons, appliances, absences (**simulated** data);
- **recognizes habits** without knowing the types (k-means on the shape of the days);
- **forecasts the building's consumption** 24 h ahead (AI) and compares it with "same as yesterday" and "same as last week".""")

res = run_button("usage", fn=lambda: demo_usage(scn))
if res is not None:
    show_result(res)
test_box("usage")
