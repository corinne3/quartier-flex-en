"""
learning.py: the AI LEARNS each home from the controller box measurements.

The problem
-----------
To cut a heater without disturbing anyone, the aggregator must know, for EACH home:
    - how much it is consuming right now (what gets shed);
    - how much the temperature will drop if heating is cut for 15 or 30 min.
Yet it knows neither the insulation (UA) nor the inertia (C) of the homes.

The solution: system identification (linear regression)
-------------------------------------------------------
Every 15 min the box measures the indoor temperature (with sensor noise)
and the heating power. The 1R1C model reads:
    ΔT/dt = (UA/C) × (T_ext − T) + (1/C) × P + gains/C
It is LINEAR in (UA/C, 1/C, gains): a least-squares regression on
2 weeks of history gives UA and C for each home. No cut is needed
to learn: setpoint changes (night, absence) are enough.

This is a "light" AI: a few milliseconds of compute for the whole district.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .portfolio import Portfolio, setpoints
from .physics import HomeState, gains_kw, time_step, thermostat


def weather_at_step(weather: pd.DataFrame, index: pd.DatetimeIndex) -> pd.DataFrame:
    """Hourly weather -> weather at simulation step (linear interpolation)."""
    w = weather[["temp_c", "ghi_wm2"]].reindex(weather.index.union(index)).interpolate(limit_direction="both")
    return w.reindex(index)


def simulate_without_shedding(portfolio: Portfolio, weather: pd.DataFrame, index: pd.DatetimeIndex, seed: int = 0,
                            noise_sensor_c: float = 0.1) -> dict:
    """
    Runs the district WITHOUT any cut (the history measured by the boxes).
    Returns the measurements: measured temperature, power per appliance, weather.
    """
    rng = np.random.default_rng(seed + 7)
    dt = (index[1] - index[0]).total_seconds() / 3600
    w = weather_at_step(weather, index)
    sp, pres = setpoints(portfolio, index, real_life=True, seed=seed)          # real life (variable schedules)
    sp_sched, pres_sched = setpoints(portfolio, index)                       # the known schedule
    state = HomeState.initial(portfolio)
    n_t, n = len(index), portfolio.n
    t_in = np.zeros((n_t + 1, n))
    t_in[0] = state.t_in
    p = {a: np.zeros((n_t, n)) for a in ("heating", "heating_raw", "water_heater", "ev")}
    hc = np.isin(index.hour, (22, 23, 0, 1, 2, 3, 4, 5))
    for k, ts in enumerate(index):
        t_ext, ghi = float(w["temp_c"].iloc[k]), float(w["ghi_wm2"].iloc[k])
        # raw thermostat power (even for gas-heated homes: used for learning)
        g = gains_kw(portfolio, pres[k], ghi)
        raw = thermostat(portfolio.ua_kw_k, portfolio.c_kwh_k, state.t_in, sp[k], t_ext, g, portfolio.p_heating_max)
        out = time_step(portfolio, state, t_ext, ghi, sp[k], pres[k], ts.hour, ts.minute, bool(hc[k]),
                           ts.dayofweek >= 5, {}, dt)
        p["heating_raw"][k] = raw
        for a in ("heating", "water_heater", "ev"):
            p[a][k] = out[a]
        t_in[k + 1] = state.t_in
    measure = t_in + rng.normal(0, noise_sensor_c, t_in.shape)
    return {"index": index, "dt": dt, "t_ext": w["temp_c"].to_numpy(), "ghi": w["ghi_wm2"].to_numpy(),
            "setpoint": sp_sched, "presence": pres_sched, "setpoint_true": sp, "presence_true": pres, "t_in_true": t_in, "t_in_measure": measure, "p": p,
            "base_kw": portfolio.base_kw.reindex(index).to_numpy(), "state_final": state}


@dataclass
class ThermalModel:
    """UA and C LEARNED for each home (+ gains)."""

    ua: np.ndarray
    c: np.ndarray
    g0: np.ndarray      # base gains (kW)
    g1: np.ndarray      # presence-related gains (kW)
    g2: np.ndarray      # solar gains (kW per W/m²)

    def power(self, t_in, t_ext, setpoint, presence, ghi, p_max, dt_h):
        """Power the thermostat will request (forecast)."""
        gains = self.g0 + self.g1 * presence + self.g2 * ghi
        return thermostat(self.ua, self.c, t_in, setpoint, t_ext, gains, p_max)

    def forecast(self, t_int0, t_ext, setpoint, presence, ghi, p_max, dt_h):
        """
        Simulates the home WITH THE LEARNED MODEL over several steps (e.g. the day-before forecast
        for the next day). Starts from the measured temperature; returns (powers, temperatures).
        Simulating the model beats a "holding power" estimate: it captures heating RESTARTS
        (end of night, return from work), which create the 7 am and 6 pm peaks.
        """
        T = np.array(t_int0, dtype=float)
        P_out, T_out = np.zeros((len(t_ext), len(T))), np.zeros((len(t_ext), len(T)))
        for k in range(len(t_ext)):
            gains = self.g0 + self.g1 * presence[k] + self.g2 * ghi[k]
            P = thermostat(self.ua, self.c, T, setpoint[k], t_ext[k], gains, p_max)
            T = T + dt_h / self.c * (P + gains - self.ua * (T - t_ext[k]))
            P_out[k], T_out[k] = P, T
        return P_out, T_out

    def drop_if_cut(self, t_in, t_ext, presence, ghi, dt_h):
        """Temperature drop (°C) if heating is cut for dt_h."""
        gains = self.g0 + self.g1 * presence + self.g2 * ghi
        return np.maximum(0.0, dt_h / self.c * (self.ua * (t_in - t_ext) - gains))


def learn_thermal(portfolio: Portfolio, hist: dict) -> ThermalModel:
    """Linear regression, home by home, on the measured history."""
    dt = hist["dt"]
    T = hist["t_in_measure"]
    # Light smoothing of the measurement (3-step rolling mean): reduces the effect of sensor noise
    Ts = pd.DataFrame(T).rolling(3, center=True, min_periods=1).mean().to_numpy()
    y = (Ts[1:] - Ts[:-1]) / dt
    n = portfolio.n
    ua, c, g0, g1, g2 = (np.zeros(n) for _ in range(5))
    for i in range(n):
        X = np.column_stack([hist["t_ext"] - Ts[:-1, i], hist["p"]["heating_raw"][:, i], np.ones(len(y)),
                             hist["presence"][:, i].astype(float), hist["ghi"]])
        coef, *_ = np.linalg.lstsq(X, y[:, i], rcond=None)
        a, b = coef[0], max(coef[1], 1e-3)
        c[i] = 1 / b
        ua[i] = max(a, 1e-4) * c[i]
        g0[i], g1[i], g2[i] = coef[2] * c[i], coef[3] * c[i], coef[4] * c[i]
    return ThermalModel(ua, c, g0, g1, g2)


def error_learning(portfolio: Portfolio, m: ThermalModel) -> pd.DataFrame:
    """Compares learned parameters with the true ones (electrically heated homes)."""
    sel = portfolio.heating
    return pd.DataFrame({
        "home": np.array(portfolio.names)[sel], "type": portfolio.types[sel],
        "UA true (W/°C)": (1000 * portfolio.ua_kw_k[sel]).round(0), "UA learned (W/°C)": (1000 * m.ua[sel]).round(0),
        "C true (kWh/°C)": portfolio.c_kwh_k[sel].round(2), "C learned (kWh/°C)": m.c[sel].round(2),
        "true time constant (h)": (portfolio.c_kwh_k[sel] / portfolio.ua_kw_k[sel]).round(0),
    })
