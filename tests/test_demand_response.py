"""Tests for module 7 — RTE requests and declared flexibility (offline)."""
import pandas as pd

from quartierflex.aggregator import load_district, set_volumes, make_strategy, run_simulation
from quartierflex.demand_response import create_requests, stress_days
from quartierflex.measure import Meter


def test_stress_days_and_slots(scn):
    idx = pd.date_range("2024-01-15", periods=24 * 5, freq="h", tz="Europe/Paris")
    grid = pd.DataFrame({"consumption_mw": 60000.0, "co2_g_per_kwh": 30.0,
                         "tempo": ["RED"] * 24 + ["WHITE"] * 24 + ["BLUE"] * 72}, index=idx)
    t, note = stress_days(grid, sorted(set(idx.date)))
    assert list(t.values()) == ["RED", "WHITE"] and note == ""
    fine = pd.date_range(idx[0], periods=96 * 5, freq="15min")
    req, _ = create_requests(grid, fine, scn.demand_response)
    # red day: morning + evening; white day: evening only
    assert [(d.start.hour, d.level) for d in req] == [(7, "RED"), (18, "RED"), (18, "WHITE")]


def test_always_a_request_in_demo(scn):
    idx = pd.date_range("2024-01-15", periods=24 * 3, freq="h", tz="Europe/Paris")
    grid = pd.DataFrame({"consumption_mw": [50000.0] * 24 + [55000.0] * 24 + [52000.0] * 24, "tempo": "BLUE"}, index=idx)
    t, note = stress_days(grid, sorted(set(idx.date)))
    assert list(t.values()) == ["FORCED"] and note


def test_declared_flexibility_includes_batteries(scn):
    ctx = load_district(scn)
    m = Meter(scn.compute, watch=())
    st = make_strategy("none")
    st.setup(ctx, m)
    ref = run_simulation(ctx, st, m)
    reps = set_volumes(ctx, ref)
    assert reps and all(d.volume_kw > 0 for d in ctx.requests)
    r = reps[0]
    assert r["batteries"]["kw_held_over_slot"] > 0 and len(r["batteries"]["packs"]) == len(scn.battery.capacities_kwh)
    assr = abs(ctx.requests[0].volume_kw - scn.demand_response.share_of_flex * r["total_kw"]) < 0.2
    assert assr
