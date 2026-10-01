"""
bms.py: MEASURE the state of the batteries, and ESTIMATE it when it cannot be measured.

The problem
-----------
State of charge (SOC) and state of health (SOH) are NOT measured
directly: no sensor displays "78%". The BMS (Battery Management
System) ESTIMATES them from what it really measures:
  - the current / energy flowing in and out (with a small sensor error);
  - the voltage (which depends on the state of charge, especially at rest).

For second-life batteries this is crucial: their real SOH is poorly
known when they are recovered, and it keeps decreasing.

Estimators
----------
1. SOC by coulomb counting: add up the energy going in and out.
   Simple, but the slightest sensor error (here an offset of
   30 Wh per hour) ACCUMULATES hour after hour: it drifts.
2. SOC corrected by the rest voltage: when the battery is idle, its
   open-circuit voltage (OCV) indicates its level. The counter is re-anchored.
   (The principle of a Kalman filter, in simplified form.)
3. Estimated SOH (the "AI / learning" part): between two rest periods, the
   level at the start and at the end is known (from the voltage), as is the
   energy exchanged (from counting) -> real capacity = energy / change in level.
   These estimates are noisy: a robust (Huber) regression on the history
   extracts the current state of health and its TREND, hence a forecast
   of the end of second life (SOH 60%).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import HuberRegressor

from .model import BatteryPack

N_CELLS_SERIES = 96          # cells in series in a car pack (~350 V)
SOC_GRID = np.linspace(0, 1, 201)


def ocv_cell(soc: np.ndarray | float) -> np.ndarray:
    """Open-circuit voltage of a lithium-ion (NMC) cell as a function of its state of charge. Typical shape."""
    soc = np.asarray(soc, dtype=float)
    return 3.30 + 0.85 * soc + 0.10 * soc**4 - 0.25 * np.exp(-20 * soc)


def soc_from_voltage(pack_voltage: float) -> float:
    """Inverse of the OCV curve: rest voltage -> state of charge."""
    v = ocv_cell(SOC_GRID) * N_CELLS_SERIES
    return float(np.interp(pack_voltage, v, SOC_GRID))


@dataclass
class Sensors:
    energy_gain_error: float = 0.02   # the sensor overestimates energy by 2% (systematic error)
    energy_noise_kwh: float = 0.02    # random noise per hour
    energy_offset_kwh: float = 0.03   # current-sensor offset: it "sees" 30 Wh/h even at rest
    voltage_noise_v: float = 0.4      # voltage measurement noise (on ~350 V)


def run_bms(pack: BatteryPack, power_kw: np.ndarray, index: pd.DatetimeIndex, sensors: Sensors | None = None,
            seed: int = 0, soh_guess: float | None = None) -> pd.DataFrame:
    """
    Runs a pack through a power profile, simulating what the BMS
    measures and what it estimates. Returns an hourly table (truth vs estimates).

    soh_guess: SOH assumed at installation (often poorly known in second life).
    """
    s = sensors or Sensors()
    rng = np.random.default_rng(seed)
    cap_name = pack.cap_name_kwh
    soh_believed = soh_guess if soh_guess is not None else pack.soh
    soc_cc = pack.soc       # pure counting
    soc_est = pack.soc      # counting + voltage re-anchoring
    rows = []
    rest_h = 0
    for t, p_cmd in zip(index, power_kw):
        soc_before = pack.soc
        p = pack.step(float(p_cmd))
        stored_true = (pack.soc - soc_before) * pack.capacity_kwh
        stored_meas = (stored_true * (1 + s.energy_gain_error) + s.energy_offset_kwh
                       + rng.normal(0, s.energy_noise_kwh))
        cap_believed = cap_name * soh_believed
        soc_cc = float(np.clip(soc_cc + stored_meas / cap_believed, 0, 1))
        soc_est = float(np.clip(soc_est + stored_meas / cap_believed, 0, 1))
        rest_h = rest_h + 1 if abs(p) < 0.1 else 0
        v_meas = np.nan
        if rest_h >= 1:  # at rest: the measured voltage reflects the state of charge
            v_meas = float(ocv_cell(pack.soc) * N_CELLS_SERIES + rng.normal(0, s.voltage_noise_v))
            soc_est = 0.3 * soc_est + 0.7 * soc_from_voltage(v_meas)
        rows.append({
            "time": t, "power_kw": p, "soc_true": pack.soc, "soh_true": pack.soh, "efc": pack.efc,
            "energy_measured_kwh": stored_meas, "tension_rest_v": v_meas,
            "soc_counting": soc_cc, "soc_estimated": soc_est,
        })
    return pd.DataFrame(rows).set_index("time")


def estimate_soh(bms_log: pd.DataFrame, cap_name_kwh: float, min_delta_soc: float = 0.3) -> dict:
    """
    Estimates the real capacity between two rest-voltage measurements:
        capacity ≈ energy measured between them / (end SOC - start SOC)
    then fits a robust trend SOH = a + b × cycles.
    """
    rest = bms_log["tension_rest_v"].dropna()
    points = []
    times = list(rest.index)
    for i in range(1, len(times)):
        t0, t1 = times[i - 1], times[i]
        soc0 = soc_from_voltage(rest[t0])
        soc1 = soc_from_voltage(rest[t1])
        if abs(soc1 - soc0) < min_delta_soc:
            continue
        seg = bms_log.loc[(bms_log.index > t0) & (bms_log.index <= t1)]
        energy = seg["energy_measured_kwh"].sum()
        cap = abs(energy / (soc1 - soc0))
        points.append({"time": t1, "efc": float(seg["efc"].iloc[-1]), "soh_measure": cap / cap_name_kwh})
    pts = pd.DataFrame(points)
    out = {"points": pts, "soh_estimated": np.nan, "slope_per_cycle": np.nan, "cycles_before_60pct": np.nan}
    if len(pts) >= 5:
        X = pts[["efc"]].to_numpy()
        y = pts["soh_measure"].to_numpy()
        reg = HuberRegressor().fit(X, y)
        last = float(pts["efc"].iloc[-1])
        soh_now = float(reg.predict([[last]])[0])
        slope = float(reg.coef_[0])
        out.update(soh_estimated=soh_now, slope_per_cycle=slope,
                   cycles_before_60pct=((soh_now - 0.60) / -slope) if slope < 0 else np.inf)
    return out


def demo_profile(days: int, charge_kw: float = 6.0, discharge_kw: float = 5.0) -> np.ndarray:
    """Typical self-consumption profile: charge in the sun (10am-3pm), discharge in the evening (6-10pm), rest otherwise."""
    day = np.zeros(24)
    day[10:16] = charge_kw
    day[18:23] = -discharge_kw
    return np.tile(day, days)
