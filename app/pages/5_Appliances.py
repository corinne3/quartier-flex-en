"""Page 5 · Connected appliances — see docs/modules/05_appliances.md. Launch: quartier interface"""

from _common import run_button, scenario_sidebar, show_result, test_box

import streamlit as st

from quartierflex.demo import demo_appliances

st.set_page_config(page_title="5 · Connected appliances", layout="wide")
scn = scenario_sidebar()
st.title("5 · Connected appliances")
st.markdown("""**What this module does**: describes what each home can **shed**.
- Each home = an **occupant type** (module 4) × **appliances**: electric heating, water heater,
  electric car. Only homes with a **connected box** (smart controller) can be controlled.
- **Sheddable group** = occupant type × appliance (e.g. "retirees/heating"): this is the aggregator's unit.
- **Heating**: "1R1C" thermal model (insulation + inertia). Cutting it for 30 min lowers the temperature by
  a few tenths of a degree, then the thermostat **catches up**: the energy is mostly **shifted**.
- **Water heater**: a store of hot water, can be cut for 2 h. **Car**: charging that can be shifted before departure.
- **Lighting, cooking, fridge**: not sheddable (low power, comfort, safety).
- **The AI learns** each home's insulation and inertia from the box's measurements (regression),
  then **forecasts** its consumption. A single home is hard to predict (variable schedules), the district much
  less so: this is the **diversity effect** (load aggregation).""")
hour = st.slider("Hour of the experimental cut (heater off for 30 min)", 6, 21, 18)
res = run_button("appliances", fn=lambda: demo_appliances(scn, hour_cut=hour))
if res is not None:
    show_result(res)
test_box("appliances")
