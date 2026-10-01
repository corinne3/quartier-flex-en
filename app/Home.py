"""
Home.py: home page of the interface. Launch: quartier interface
(or: python -m streamlit run app/Home.py)

Streamlit automatically builds the left-hand menu from the files in app/pages/:
one page per module, in numerical order.
"""

from _common import scenario_sidebar  # noqa: F401  (adds src/ to the path)

import streamlit as st

st.set_page_config(page_title="Quartier Flex", page_icon="⚡", layout="wide")
scenario_sidebar()

st.title("⚡🏘️ Quartier Flex: demand response, second-life batteries and solar")
st.markdown(
    """
**The question**: when RTE (the French transmission system operator) asks a district to **shed** its consumption
(on very cold days), does the artificial intelligence that spreads the effort across homes and batteries **save more
than it consumes**?

**The 3 levels**
- **RTE** sends a request to the district: "shed X kW from 6 pm to 8 pm".
- **The aggregator (our AI)** juggles the **sheddable groups** (occupant type × appliance) and the **batteries**.
- **The home**: an appliance is cut for 30 min at most, without losing comfort.

**How to use this interface**
1. Set the scenario in the left sidebar (kept from one page to the next). Default: January 2024, 60 homes.
2. Try the modules **one by one**, in menu order.
3. Finish with **11 · Assessment**.
"""
)

st.markdown("### The modules")
st.table({
    "Module": ["1 · Weather", "2 · Solar", "3 · Batteries", "4 · Usage", "5 · Appliances", "6 · Grid",
               "7 · Demand response", "8 · Battery control", "9 · Aggregator", "10 · Measurement", "11 · Assessment"],
    "Role": [
        "Observed and forecast weather (Open-Meteo), sun position",
        "Solar panel output, meter, production forecast",
        "Second-life batteries: charge, health, aging, BMS estimation",
        "Home consumption (occupant types), habit recognition, forecast",
        "Connected appliances (heating, water heater, car), sheddable groups, learned thermal model",
        "Tariffs (Tempo), grid CO2, national consumption (RTE)",
        "RTE requests (stress days) and declared flexibility, batteries included",
        "Batteries serving self-consumption (hour by hour)",
        "The AI that splits the request between groups and batteries, every 15 min",
        "Energy consumed by the AI itself",
        "Net AI balance: demand response, self-consumption, sizing",
    ],
    "AI used": ["–", "Gradient boosting", "Robust regression (health)", "k-means + gradient boosting",
                "System identification (regression)", "–", "Forecasting (learned model)",
                "Linear optimization + LLM", "Linear optimization + LLM agent", "–", "–"],
})
st.info("Consumption, appliances and meters are SIMULATED (no personal data available); "
        "weather, CO2 and national consumption are real when a connection is available.")
