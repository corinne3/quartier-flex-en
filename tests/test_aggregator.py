"""Tests for module 9 — aggregator (offline, dummy LLM)."""
import pytest

from quartierflex.aggregator import run_demand_response, table


@pytest.fixture(scope="module")
def out(scn):
    return run_demand_response(scn, llm_kind="fake", save=False, log=lambda *a: None)


def test_all_strategies_run(out):
    assert set(out["results"]) == {"0_none", "1_cut_all", "2_round_robin", "2b_prepared_round_robin",
                                     "3_optimizer_ai", "4_agent_llm"}
    assert len(table(out)) == 6


def test_comfort_rule_and_services(out):
    for r in out["results"].values():
        k = r["kpi"]
        assert k["ev_missing_kwh"] <= 1.0          # cars are charged by departure time
        assert k["cold_water_kwh"] <= 1.0


def test_optimizer_beats_simple_round_robin(out):
    opt, tq = out["results"]["3_optimizer_ai"]["kpi"], out["results"]["2_round_robin"]["kpi"]
    assert opt["fulfilment"] > tq["fulfilment"]
    assert opt["discomfort_added_degh"] < tq["discomfort_added_degh"]
    assert opt["rebound_kwh"] < tq["rebound_kwh"]


def test_ai_assessment_and_agent(out):
    for n in ("3_optimizer_ai", "4_agent_llm"):
        b = out["results"][n]["assessment_ai"]
        assert b["cost_ai_eur"] >= 0 and "gain_net_eur" in b
    assert out["results"]["4_agent_llm"]["kpi"]["llm_calls"] >= 1
    assert out["traces"]["4_agent_llm"][0]["stop"] == "ok"      # the FakeLLM honors the JSON contract
