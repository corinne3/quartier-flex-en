"""
meter.py: what the production meter MEASURES (≠ theory).

In reality, measured production deviates from the physics model:
- measurement noise and micro-variations (± 2 %);
- soiling (dust, pollen) that builds up and is then washed off by rain;
- short outages (inverter tripping);
- occasional shading.

Lacking a real meter during the hackathon, we SIMULATE these deviations. This is
precisely what an AI can learn and a physics model ignores:
that is the point of comparing the two (see forecast.py).
⚠️ To be replaced by real measurements if the team finds some (inverter export, Linky in feed-in mode).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def measured_production(true_kwh: pd.Series, seed: int = 0, noise: float = 0.02,
                        soiling_per_day: float = 0.0015, outage_prob: float = 0.003) -> pd.Series:
    rng = np.random.default_rng(seed)
    n = len(true_kwh)
    days = ((true_kwh.index - true_kwh.index[0]) / pd.Timedelta("1D")).astype(int)
    # Soiling: loss that grows every day, reset to zero by a random "rain" (~once every 12 days)
    n_days = int(days.max()) + 1
    loss = np.zeros(n_days)
    for d in range(1, n_days):
        loss[d] = 0.0 if rng.random() < 1 / 12 else min(loss[d - 1] + soiling_per_day, 0.08)
    soiling = 1 - loss[days]
    # Outages: an outage lasts a few hours
    ok = np.ones(n)
    i = 0
    while i < n:
        if rng.random() < outage_prob:
            k = rng.integers(1, 5)
            ok[i:i + k] = 0
            i += k
        i += 1
    meas = true_kwh.to_numpy() * soiling * ok * (1 + rng.normal(0, noise, n))
    s = pd.Series(np.clip(meas, 0, None).round(3), index=true_kwh.index, name="pv_measure_kwh")
    s.attrs["source"] = "SIMULATED meter (physics model + soiling + outages + noise)"
    return s
