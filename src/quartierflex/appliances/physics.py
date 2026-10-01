"""
physics.py: how each appliance consumes, time step by time step.

1. HEATING: "1R1C" thermal model (one resistance, one capacitance)
------------------------------------------------------------------
A home = a "heat bathtub":
    - INERTIA C (kWh/°C): the energy needed to raise the home by 1 °C;
    - LOSSES UA (kW/°C): the heat that leaks out, proportional to the indoor-outdoor difference.
    T(t+dt) = T(t) + dt / C × (P_heating + gains − UA × (T − T_ext))
The thermostat offsets the losses and GRADUALLY closes the gap to the setpoint
(proportional control: a 1 °C gap is closed in ~1.5 h), within the limit
of the installed power. If heating is CUT for 30 min, the temperature drops
a little (a few tenths of a degree); when it comes back on, the thermostat catches up:
this is the REBOUND EFFECT (consumption is shifted, not removed).

2. WATER HEATER: an energy store
--------------------------------
Hot water is drawn (morning, noon, evening); the tank reheats either during off-peak
hours (day/night contactor), or as soon as it drops below 85 %. Cutting the water heater
for 2 h has no effect if the store is sufficient. Failure = "cold water" (kWh missing).

3. ELECTRIC CAR: charging to be delayed
---------------------------------------
Plugged in when coming home in the evening, must be charged at departure (7 am). Without
control it charges on arrival... right in the 6-8 pm peak. Delaying causes no discomfort
as long as the car is full in the morning. Failure = kWh missing at departure.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .portfolio import Portfolio

# Distribution of hot-water draws over the day (share of the day, per hour)
DRAW = np.zeros(24)
DRAW[[6, 7]] = 0.2
DRAW[[12, 13]] = 0.05
DRAW[[19, 20, 21]] = 0.5 / 3


TAU_REGULATION_H = 1.5     # thermostat catch-up time (proportional control)


def thermostat(ua, c, t_in, setpoint, t_ext, gains, p_max):
    """Power requested by the thermostat: losses to offset + gradual closing of the gap."""
    return np.clip(ua * (t_in - t_ext) - gains + c * (setpoint - t_in) / TAU_REGULATION_H, 0.0, p_max)


@dataclass
class HomeState:
    """Physical state of all homes (arrays)."""

    t_in: np.ndarray          # indoor temperature (°C)
    water_heater_kwh: np.ndarray        # energy in the water heater
    water_heater_on: np.ndarray         # bool: water heater element on (hysteresis)
    ev_need: np.ndarray      # kWh left to charge
    ev_plugged: np.ndarray     # bool
    # Cut counters (in time steps), for the "30 min max then 30 min rest" rule
    cut_since: dict         # appliance -> consecutive steps cut
    rest_since: dict         # appliance -> consecutive steps NOT cut
    # Cumulative service failures
    cold_water_kwh: float = 0.0
    ev_missing_kwh: float = 0.0

    @classmethod
    def initial(cls, portfolio: Portfolio, t_ext0: float = 5.0) -> "HomeState":
        n = portfolio.n
        return cls(t_in=np.full(n, 19.0), water_heater_kwh=np.full(n, 0.8 * portfolio.cfg.water_heater_kwh_max),
                   water_heater_on=np.zeros(n, bool), ev_need=np.zeros(n), ev_plugged=np.zeros(n, bool),
                   cut_since={a: np.zeros(n, int) for a in ("heating", "water_heater", "ev")},
                   rest_since={a: np.full(n, 99, int) for a in ("heating", "water_heater", "ev")})

    def copy(self) -> "HomeState":
        return HomeState(self.t_in.copy(), self.water_heater_kwh.copy(), self.water_heater_on.copy(), self.ev_need.copy(),
                    self.ev_plugged.copy(), {k: v.copy() for k, v in self.cut_since.items()},
                    {k: v.copy() for k, v in self.rest_since.items()}, self.cold_water_kwh, self.ev_missing_kwh)


def cut_allowed(state: HomeState, appliance: str, max_step: int, rest_step: int) -> np.ndarray:
    """Can this appliance be cut now? (comfort rule: max duration, then rest)"""
    in_progress = state.cut_since[appliance]
    return ((in_progress > 0) & (in_progress < max_step)) | ((in_progress == 0) & (state.rest_since[appliance] >= rest_step))


def gains_kw(portfolio: Portfolio, presence: np.ndarray, ghi_wm2: float) -> np.ndarray:
    """Free gains: occupants and appliances (0.15 to 0.4 kW) + sun through the windows."""
    return 0.15 + 0.25 * presence + ghi_wm2 / 1000 * portfolio.floor_area * 0.004


def time_step(portfolio: Portfolio, state: HomeState, t_ext: float, ghi: float, setpoint: np.ndarray, presence: np.ndarray,
                 hour: int, minute: int, off_peak: bool, weekend: bool, cut: dict, dt_h: float) -> dict:
    """
    Advances the whole district by one step. `cut` = {appliance: bool array} (cut requests,
    ALREADY filtered by the comfort rule). Returns the power (kW) of each home.
    """
    cfg = portfolio.cfg
    n = portfolio.n
    # ---------------- Heating (ideal thermostat + 1R1C model)
    gains = gains_kw(portfolio, presence, ghi)
    p_ch = thermostat(portfolio.ua_kw_k, portfolio.c_kwh_k, state.t_in, setpoint, t_ext, gains, portfolio.p_heating_max)
    p_ch = np.where(cut.get("heating", np.zeros(n, bool)), 0.0, p_ch)
    state.t_in = state.t_in + dt_h / portfolio.c_kwh_k * (p_ch + gains - portfolio.ua_kw_k * (state.t_in - t_ext))
    p_ch_electric = np.where(portfolio.heating, p_ch, 0.0)          # the others heat with gas/wood: out of scope

    # ---------------- Water heater
    draw = portfolio.water_heater_kwh_day * DRAW[hour] * dt_h
    state.water_heater_kwh = state.water_heater_kwh - np.where(portfolio.water_heater, draw, 0.0)
    shortfall = np.clip(-state.water_heater_kwh, 0, None)
    state.cold_water_kwh += float(shortfall.sum())
    state.water_heater_kwh = np.maximum(state.water_heater_kwh, 0.0)
    full = state.water_heater_kwh >= cfg.water_heater_kwh_max - 1e-6
    state.water_heater_on = np.where(portfolio.water_heater_hc, off_peak & ~full,
                           (state.water_heater_kwh < 0.85 * cfg.water_heater_kwh_max) | (state.water_heater_on & ~full))
    p_water_heater = np.where(state.water_heater_on & portfolio.water_heater, np.minimum(cfg.water_heater_kw, (cfg.water_heater_kwh_max - state.water_heater_kwh) / dt_h), 0.0)
    p_water_heater = np.where(cut.get("water_heater", np.zeros(n, bool)), 0.0, p_water_heater)
    state.water_heater_kwh = state.water_heater_kwh + p_water_heater * dt_h

    # ---------------- Electric car
    if minute == 0 and hour == 7:                              # morning departure
        state.ev_missing_kwh += float(state.ev_need[state.ev_plugged].sum())
        state.ev_need[:] = 0.0
        state.ev_plugged[:] = False
    arrives = portfolio.ev & (portfolio.ev_arrival == hour) & (minute == 0)
    state.ev_plugged = state.ev_plugged | arrives
    state.ev_need = state.ev_need + np.where(arrives, cfg.ev_kwh_per_day * (0.6 if weekend else 1.0), 0.0)
    p_ev = np.where(state.ev_plugged, np.minimum(cfg.ev_kw, state.ev_need / dt_h), 0.0)
    p_ev = np.where(cut.get("ev", np.zeros(n, bool)), 0.0, p_ev)
    state.ev_need = np.maximum(state.ev_need - p_ev * dt_h, 0.0)

    # ---------------- Cut counters
    for a in ("heating", "water_heater", "ev"):
        c = cut.get(a, np.zeros(n, bool))
        state.cut_since[a] = np.where(c, state.cut_since[a] + 1, 0)
        state.rest_since[a] = np.where(c, 0, state.rest_since[a] + 1)
    return {"heating": p_ch_electric, "water_heater": p_water_heater, "ev": p_ev}


def power_sheddable(portfolio: Portfolio, state: HomeState, t_ext: float, ghi: float, setpoint: np.ndarray,
                        presence: np.ndarray, dt_h: float) -> dict:
    """
    What each home would consume NOW without a cut (= what can be shed), per appliance.
    "Dry run" computation: does not modify the state.
    """
    cfg = portfolio.cfg
    gains = gains_kw(portfolio, presence, ghi)
    p_ch = thermostat(portfolio.ua_kw_k, portfolio.c_kwh_k, state.t_in, setpoint, t_ext, gains, portfolio.p_heating_max)
    full = state.water_heater_kwh >= cfg.water_heater_kwh_max - 1e-6
    p_water_heater = np.where(state.water_heater_on & ~full & portfolio.water_heater, cfg.water_heater_kw, 0.0)
    p_ev = np.where(state.ev_plugged, np.minimum(cfg.ev_kw, state.ev_need / dt_h), 0.0)
    ok = portfolio.connected
    return {"heating": np.where(portfolio.heating & ok, p_ch, 0.0), "water_heater": np.where(ok, p_water_heater, 0.0),
            "ev": np.where(ok, p_ev, 0.0)}
