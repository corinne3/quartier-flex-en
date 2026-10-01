"""
_common.py: what all the pages of the interface share.

- scenario_sidebar(): the common settings (period, district, panels,
  batteries, tariff), in the left sidebar, kept from one page to the next.
- show_result():      displays a DemoResult (sentences, charts, tables).
- test_box():         reminds how to test the module on its own.
"""

from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT / "src", ROOT / "app"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import streamlit as st  # noqa: E402

from quartierflex.config import Scenario  # noqa: E402

DEFAULTS = {"start": dt.date(2024, 1, 15), "days": 7, "homes": 60, "kwc": 120.0, "packs": 4, "soh": 0.75,
            "tariff": "TEMPO", "offline": False}


def scenario_sidebar() -> Scenario:
    ss = st.session_state
    for k, v in DEFAULTS.items():
        ss.setdefault(k, v)
    sb = st.sidebar
    sb.header("Scenario")
    sb.date_input("Start of the period", key="start", min_value=dt.date(2023, 3, 1), max_value=dt.date(2025, 6, 30))
    sb.slider("Duration (days)", 1, 21, key="days")
    sb.slider("Homes in the district", 5, 300, step=5, key="homes")
    sb.slider("Solar panels (kWp)", 0.0, 600.0, step=10.0, key="kwc")
    sb.slider("Car battery packs (second life)", 0, 20, key="packs")
    sb.slider("Average battery state of health (SOH)", 0.60, 0.90, step=0.01, key="soh")
    sb.selectbox("Tariff", ["TEMPO", "HPHC", "BASE"], key="tariff")
    sb.checkbox("Offline (synthetic data)", key="offline",
                help="Checked: no downloads. Results are then illustrative only, not publishable.")
    scn = Scenario(start=str(ss.start), days=int(ss.days), offline=bool(ss.offline))
    scn.building.n_homes = int(ss.homes)
    scn.pv.kwc = float(ss.kwc)
    cap = scn.battery.capacities_kwh[0]
    scn.battery.capacities_kwh = [cap] * int(ss.packs)
    # SOH varies from pack to pack around the average (batteries from different cars)
    spread = [0.04, -0.04, 0.02, -0.02, 0.0, 0.03, -0.03, 0.01]
    scn.battery.soh_init = [round(min(0.95, max(0.55, ss.soh + spread[i % 8])), 2) for i in range(int(ss.packs))]
    scn.tariff.option = ss.tariff
    return scn


def show_result(r) -> None:
    st.subheader(r.title)
    for line in r.lines:
        if line:
            st.markdown(f"- {line}")
    for fig in r.figs:
        st.pyplot(fig)
    for name, t in r.tables.items():
        st.markdown(f"**{name}**")
        st.dataframe(t)


def test_box(module: str) -> None:
    with st.expander("Test this module on its own (command line)"):
        test_file = ROOT / "tests" / f"test_{module}.py"
        pytest_line = (f"pytest tests/test_{module}.py -v" if test_file.exists() or module != "measure"
                       else "# (no dedicated test file)")
        st.code(f"quartier demo {module}\n{pytest_line}", language="bash")
        st.markdown(f"Module guide: `docs/modules/` (module **{module}**).")


def run_button(key: str, label: str = "Run the module", fn=None):
    """Button that runs fn() and keeps the result in memory for the page."""
    if st.button(label, key=f"btn_{key}", type="primary"):
        with st.spinner("Computing..."):
            try:
                st.session_state[f"res_{key}"] = fn()
            except Exception as e:  # show the error rather than crash the page
                st.session_state[f"res_{key}"] = None
                st.error(f"Error: {e}")
    return st.session_state.get(f"res_{key}")
