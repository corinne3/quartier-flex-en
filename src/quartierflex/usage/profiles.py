"""
profiles.py: the electricity consumption of each home, hour by hour.

Why simulate?
Real individual load curves (Linky smart meter) are personal data:
not available as open data at the level of a single home. So we
generate REALISTIC consumption, with the characteristics that matter
for the project:
  - household TYPES with different habits (who consumes when?);
  - weekdays ≠ weekends;
  - more consumption in winter (lighting, cooking);
  - randomness: busier or quieter days, washing machines,
    absences (holidays).
Excluding heating and hot water (assumed non-electric or not controlled here).

⚠️ Annual consumption = orders of magnitude TO BE CHECKED (the team can
   look up the averages published by ADEME / Enedis).

Each type = a baseload (fridge, internet box...) + "bumps"
(peak hour, width, power) on weekdays and weekends.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import BuildingConfig

H = np.arange(24)

# type: (kWh/year, baseload kW, weekday bumps, weekend bumps); bump = (hour, width h, kW)
TYPES = {
    "family": (4000, 0.25,
                [(7.0, 0.8, 1.0), (12.5, 1.0, 0.3), (19.5, 1.5, 2.0), (21.5, 1.0, 0.8)],
                [(9.0, 1.2, 0.8), (12.5, 1.2, 1.3), (19.5, 1.5, 1.8)]),
    "working_couple": (2600, 0.18,
                     [(7.0, 0.7, 0.7), (19.5, 1.2, 1.4), (21.5, 1.0, 0.7)],
                     [(10.0, 1.5, 0.7), (13.0, 1.2, 0.8), (20.0, 1.5, 1.2)]),
    "retirees": (2800, 0.20,
                  [(8.5, 1.2, 0.6), (12.0, 1.2, 1.2), (16.0, 2.0, 0.4), (19.0, 1.2, 1.0)],
                  [(8.5, 1.2, 0.6), (12.0, 1.2, 1.2), (16.0, 2.0, 0.4), (19.0, 1.2, 1.0)]),
    "remote_worker": (3200, 0.20,
                    [(8.0, 0.8, 0.5), (13.0, 3.0, 0.35), (12.5, 0.8, 0.8), (19.5, 1.2, 1.3)],
                    [(10.0, 1.5, 0.7), (13.0, 1.2, 0.8), (20.0, 1.5, 1.2)]),
    "student": (1400, 0.10,
                 [(9.0, 1.0, 0.3), (13.0, 1.0, 0.3), (20.0, 1.5, 0.6), (23.0, 1.0, 0.4)],
                 [(11.0, 1.5, 0.3), (14.0, 1.5, 0.3), (21.0, 1.5, 0.6), (23.5, 1.0, 0.5)]),
}


def _shape(base_kw: float, bumps: list) -> np.ndarray:
    s = np.full(24, base_kw)
    for c, w, a in bumps:
        d = np.minimum(np.abs(H - c), 24 - np.abs(H - c))  # circular distance (11pm is close to midnight)
        s += a * np.exp(-0.5 * (d / w) ** 2)
    return s


def household_load(kind: str, index: pd.DatetimeIndex, rng: np.random.Generator) -> np.ndarray:
    annual, base, wd, we = TYPES[kind]
    shape_wd, shape_we = _shape(base, wd), _shape(base, we)
    hours = index.hour.to_numpy()
    weekend = index.dayofweek.to_numpy() >= 5
    load = np.where(weekend, shape_we[hours], shape_wd[hours])
    # Season: +25% in January, -25% in July (lighting, cooking, standby)
    doy = index.dayofyear.to_numpy()
    load = load * (1 + 0.25 * np.cos(2 * np.pi * (doy - 15) / 365))
    # Daily randomness: busier or quieter days, absences
    days = ((index - index[0]) / pd.Timedelta("1D")).astype(int)
    n_days = int(days.max()) + 1
    day_factor = rng.lognormal(0, 0.15, n_days)
    away = rng.random(n_days) < 0.03
    day_factor[away] = 0.0
    load = load * day_factor[days] + np.where(away[days], base, 0.0)
    # Hourly noise + machines (washing machine / dishwasher: ~1 kWh over 2 h)
    load = load * rng.gamma(20, 1 / 20, len(index))
    for d in range(n_days):
        if not away[d] and rng.random() < {"family": 0.6, "student": 0.15}.get(kind, 0.35):
            start = d * 24 + int(rng.choice([9, 10, 14, 19, 20, 21] if kind != "retirees" else [9, 10, 11, 14]))
            load[start:start + 2] += 0.5
    # Scale to the target annual consumption
    per_hour_target = annual / 8760
    return load * per_hour_target / _shape_mean(kind)


def _shape_mean(kind: str) -> float:
    """Mean "raw" hourly consumption of the type, used to calibrate the annual scale."""
    _, base, wd, we = TYPES[kind]
    return (5 * _shape(base, wd).mean() + 2 * _shape(base, we).mean()) / 7 * 1.0 + 0.5 * 2 * {
        "family": 0.6, "student": 0.15}.get(kind, 0.35) / 24


def make_building(cfg: BuildingConfig, index: pd.DatetimeIndex, seed: int = 0) -> tuple[pd.DataFrame, dict]:
    """
    Returns (hourly consumption in kWh, one column per home; {home: type}).
    Types are drawn according to cfg.mix (reproducible with seed).
    """
    rng = np.random.default_rng(seed)
    kinds = list(cfg.mix)
    probs = np.array([cfg.mix[k] for k in kinds], dtype=float)
    probs /= probs.sum()
    # Types are drawn BEFORE consumption (separate generator): this way a home keeps
    # the same type whatever the simulated duration (consistency from one interface page to another).
    draw_types = rng.choice(len(kinds), size=cfg.n_homes, p=probs)
    rng_load = np.random.default_rng(seed + 1000)
    types = {}
    cols = {}
    for i in range(cfg.n_homes):
        name = f"L{i + 1:02d}"
        k = kinds[int(draw_types[i])]
        types[name] = k
        cols[name] = household_load(k, index, rng_load)
    df = pd.DataFrame(cols, index=index).round(4)
    df.attrs["source"] = "SIMULATED consumption (typical profiles + randomness)"
    return df, types
