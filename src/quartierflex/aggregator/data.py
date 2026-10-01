"""
data.py: prepare the district's world at a 15-minute step.

Gathers:
    weather (observed + forecast), solar production, grid (CO2, Tempo, prices) -> modules 1, 2, 6
    portfolio of homes and appliances                                          -> modules 4, 5
    HISTORY measured by the boxes (no shedding) + learning                     -> module 5
    RTE requests (stress days, slots)                                          -> module 7

The compute cost of learning is measured (it will be counted in the assessment
of the strategies that use AI).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..appliances import HomeState, Portfolio, learn_thermal, setpoints, make_portfolio, weather_at_step, simulate_without_shedding
from ..assessment import load_inputs
from ..config import Scenario
from ..demand_response import create_requests, hourly_profiles
from ..measure import Meter


@dataclass
class DistrictContext:
    scenario: Scenario
    inputs: object                  # hourly inputs (assessment.load_inputs)
    portfolio: Portfolio
    index: pd.DatetimeIndex         # simulated period, at a 15-min step
    dt: float
    weather: pd.DataFrame             # observed, at step
    weather_fc: pd.DataFrame          # forecast, at step (index: history + period + 1 day)
    setpoint: np.ndarray            # [steps × homes] ACTUAL setpoint (schedule + real-life variations)
    presence: np.ndarray            # ACTUAL presence
    pv_kw: np.ndarray
    price_buy: np.ndarray
    price_sell: np.ndarray
    co2: np.ndarray
    off_peak: np.ndarray
    hist: dict                      # measured history (appliances)
    state0: HomeState                     # state of the homes at the start of the period
    model: object                  # learned thermal model
    profiles: dict                   # learned hourly profiles (water heater, car)
    requests: list                  # RTE requests (volume set later)
    note: str = ""
    cost_learning: dict = field(default_factory=dict)

    def sources(self) -> dict:
        return self.inputs.sources()


def load_district(scn: Scenario) -> DistrictContext:
    inp = load_inputs(scn)
    step = scn.demand_response.step_min
    portfolio = make_portfolio(scn.building, scn.appliances, inp.index, seed=scn.seed, step_min=step)
    full = portfolio.base_kw.index
    t0 = inp.sim_index[0]
    t1 = inp.sim_index[-1] + pd.Timedelta("1h")
    hist_idx = full[full < t0]
    sim_idx = full[(full >= t0) & (full < t1)]
    dt = step / 60

    # History without shedding (what the boxes measured) + learning, cost measured
    hist = simulate_without_shedding(portfolio, inp.weather, hist_idx, seed=scn.seed)
    meter = Meter(scn.compute, watch=())
    with meter.measure("setup"):
        model = learn_thermal(portfolio, hist)
        profiles = hourly_profiles(hist)
    rep = meter.report()["setup"]

    weather = weather_at_step(inp.weather, sim_idx)
    weather_fc = weather_at_step(inp.weather_fc, full)
    sp, pres = setpoints(portfolio, sim_idx, real_life=True, seed=scn.seed + 1)   # real life (unknown in advance)
    hourly = sim_idx.floor("h")
    requests, note = create_requests(inp.grid, sim_idx, scn.demand_response)
    return DistrictContext(
        scenario=scn, inputs=inp, portfolio=portfolio, index=sim_idx, dt=dt, weather=weather, weather_fc=weather_fc,
        setpoint=sp, presence=pres,
        pv_kw=inp.pv_true.reindex(hourly).to_numpy(),
        price_buy=inp.price_buy.reindex(hourly).to_numpy(), price_sell=inp.price_sell.reindex(hourly).to_numpy(),
        co2=inp.grid["co2_g_per_kwh"].reindex(hourly).to_numpy(),
        off_peak=np.isin(sim_idx.hour, scn.tariff.hc_hours),
        hist=hist, state0=hist["state_final"], model=model, profiles=profiles, requests=requests, note=note,
        cost_learning={"energy_kwh": rep["energy_kwh"], "duration_s": rep["wall_s"]},
    )
