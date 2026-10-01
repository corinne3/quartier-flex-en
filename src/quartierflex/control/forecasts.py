"""
forecasts.py: preparing the forecasts used by the strategies.

Two families:
- WITHOUT AI: production "same as yesterday" (persistence), consumption "same as
  last week" (D-7). Compute cost ≈ 0.
- WITH AI: ML forecast of production (solar/forecast.py) and of
  consumption (usage/learning.py). Cost measured by the `meter`.

Forecasts are computed for the whole period at once (this is
equivalent: each value only uses information available 24 h
earlier), and the compute cost is filed in the right buckets.
"""

from __future__ import annotations

from contextlib import nullcontext

import pandas as pd

from ..solar import PVForecaster, persistence
from ..usage import LoadForecaster


def naive_forecasts(ctx) -> pd.DataFrame:
    idx = ctx.index[ctx.index >= ctx.sim_index[0]]
    pv = persistence(ctx.pv_meas, idx)
    load = ctx.load_total.reindex(idx - pd.Timedelta("168h")).set_axis(idx)
    return pd.DataFrame({"pv_fc": pv.to_numpy(), "load_fc": load.to_numpy()}, index=idx)


def ai_forecasts(ctx, meter=None) -> pd.DataFrame:
    idx = ctx.index[ctx.index >= ctx.sim_index[0]]
    hist = ctx.index < ctx.sim_index[0]
    scn = ctx.scenario
    m_setup = meter.measure("setup") if meter else nullcontext()
    with m_setup:
        pv_model = PVForecaster(scn.site, scn.pv).fit(ctx.weather_fc[hist], ctx.pv_meas[hist]) if scn.pv.kwc > 0 else None
        load_model = LoadForecaster().fit(ctx.load_total[hist], ctx.weather_fc["temp_c"])
    m_shared = meter.measure("shared") if meter else nullcontext()
    with m_shared:
        pv = pv_model.predict(ctx.weather_fc.loc[idx]) if pv_model else pd.Series(0.0, index=idx)
        load = load_model.predict(ctx.load_total, idx, ctx.weather_fc["temp_c"])
    return pd.DataFrame({"pv_fc": pv.to_numpy(), "load_fc": load.to_numpy()}, index=idx)
