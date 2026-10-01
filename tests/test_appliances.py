"""Tests for module 5 — connected appliances (offline)."""
import numpy as np
import pandas as pd

from quartierflex.appliances import (HomeState, learn_thermal, setpoints, make_portfolio, time_step,
                                    simulate_without_shedding)
from quartierflex.weather import load_weather


def _portfolio(scn, days=14):
    idx = pd.date_range("2024-01-01", periods=24 * days, freq="h", tz="Europe/Paris")
    return make_portfolio(scn.building, scn.appliances, idx, seed=1), idx


def test_portfolio_consistent(scn):
    portfolio, _ = _portfolio(scn)
    assert portfolio.n == scn.building.n_homes
    t = portfolio.table()
    assert t.loc["TOTAL", "homes"] == portfolio.n
    # a group only exists for connected, equipped homes
    for g in portfolio.groups():
        typ, app = g.split("/")
        cols = portfolio.group(app) == g
        assert (portfolio.types[cols] == typ).all() and portfolio.connected[cols].all()


def test_types_stable_whatever_the_duration(scn):
    a, _ = _portfolio(scn, 7)
    b, _ = _portfolio(scn, 30)
    assert (a.types == b.types).all()


def test_cut_lowers_temperature_then_catches_up(scn):
    portfolio, idx = _portfolio(scn, 2)
    i = int(np.argmax(portfolio.heating))
    sp, pres = setpoints(portfolio, idx[:1])
    res = {}
    for case_name, cut in (("normal", False), ("cut", True)):
        hs = HomeState.initial(portfolio)
        hs.t_in[:] = 20.0
        c = np.zeros(portfolio.n, bool)
        c[i] = cut
        p1 = time_step(portfolio, hs, 0.0, 0.0, np.full(portfolio.n, 20.0), np.ones(portfolio.n, bool), 18, 0, False, False,
                          {"heating": c}, 0.25)
        t1 = hs.t_in[i]
        p2 = time_step(portfolio, hs, 0.0, 0.0, np.full(portfolio.n, 20.0), np.ones(portfolio.n, bool), 18, 15, False, False, {}, 0.25)
        res[case_name] = (p1["heating"][i], t1, p2["heating"][i])
    assert res["cut"][0] == 0.0 and res["normal"][0] > 0      # cut = 0 kW
    assert res["cut"][1] < res["normal"][1]                   # the temperature drops
    assert res["cut"][2] > res["normal"][2]                   # then the thermostat catches up


def test_thermal_learning(scn):
    portfolio, idx = _portfolio(scn, 14)
    w = load_weather(scn.site, "2024-01-01", 14, 0, "observed", True, 0)
    fine = portfolio.base_kw.index
    h = simulate_without_shedding(portfolio, w, fine, seed=1)
    m = learn_thermal(portfolio, h)
    sel = portfolio.heating
    err_c = np.median(np.abs(m.c[sel] / portfolio.c_kwh_k[sel] - 1))
    assert err_c < 0.35          # inertia is learned to within ±35% at least
