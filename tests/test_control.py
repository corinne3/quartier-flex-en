"""Module 8 — battery control. Run on its own: pytest tests/test_control.py -v"""
import numpy as np

from quartierflex.agent.llm import FakeLLM
from quartierflex.measure import Meter
from quartierflex.control import AgentLLM, make_controller, solve_plan


def test_optimizer_buys_when_cheap():
    # 2 cheap hours then 2 expensive hours, no sun, empty battery:
    # the plan must charge first and discharge afterwards.
    pv = np.zeros(4)
    load = np.full(4, 2.0)
    price = np.array([0.10, 0.10, 0.60, 0.60])
    c, d, g, e, E = solve_plan(pv, load, price, np.full(4, 0.0), 1.0, 1.0, 20.0, 5.0, 5.0, 0.95, 0.95, 0.01, 0.0)
    assert c[:2].sum() > 1 and d[2:].sum() > 1


def test_all_rules_simulate(ctx):
    from quartierflex.assessment import simulate

    for key in ("no_battery", "rule", "tariff_rule", "optimizer"):
        c = make_controller(key)
        m = Meter(watch=())
        c.setup(ctx, m)
        sim, _ = simulate(ctx, c, m)
        assert len(sim) == len(ctx.sim_index)


def test_llm_agent_with_fake_llm(ctx):
    from quartierflex.assessment import simulate

    a = AgentLLM(llm=FakeLLM(), mode="route")
    m = Meter(watch=())
    a.setup(ctx, m)
    sim, _ = simulate(ctx, a, m, sim_index=ctx.sim_index[:24])
    assert a.stats.get("llm_runs", 0) >= 1 and len(sim) == 24
