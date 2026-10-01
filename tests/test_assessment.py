"""Module 11 — assessment. Run on its own: pytest tests/test_assessment.py -v"""
import numpy as np

from quartierflex.assessment import kpis, run_all, simulate, sweep
from quartierflex.measure import Meter
from quartierflex.control import make_controller


def test_energy_conservation(ctx):
    c = make_controller("rule")
    c.setup(ctx, None)
    sim, _ = simulate(ctx, c, Meter(watch=()))
    # production + purchase = consumption + resale + net battery charge, every hour
    lhs = sim["pv_kwh"] + sim["purchase_kwh"]
    rhs = sim["consumption_kwh"] + sim["resale_kwh"] + sim["battery_kw"]
    assert np.allclose(lhs, rhs, atol=1e-6)


def test_battery_improves_self_sufficiency(ctx):
    res = {}
    for key in ("no_battery", "rule"):
        c = make_controller(key)
        c.setup(ctx, None)
        s, b = simulate(ctx, c, Meter(watch=()))
        res[key] = kpis(s, ctx, b, None)
    assert res["rule"]["self_sufficiency"] > res["no_battery"]["self_sufficiency"]


def test_sizing(ctx):
    t = sweep(ctx, kwc_list=(0, 30), packs_list=(0, 2))
    assert len(t) == 4 and t.loc[(t.kwc == 0) & (t.packs == 0), "savings_year_eur"].iloc[0] == 0


def test_run_all_writes_results(scn, ctx, tmp_path):
    out = run_all(scn, ["no_battery", "rule", "optimizer", "agent_route"], ctx=ctx, llm_days=1, out_dir=tmp_path)
    assert (tmp_path / "latest.json").exists()
    assert "assessment_ai" in out["results"]["3_optimizer_ai"]
