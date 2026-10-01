"""
signals.py: WHEN does RTE request demand response, and HOW MUCH?

In reality
----------
- RTE (the French transmission system operator) monitors the supply/demand balance.
  On very cold days, national consumption rises (electric heating): these are
  "stress" days (EcoWatt orange/red signal, white/red Tempo days).
- Winter peaks are in the morning (7-9 am) and above all in the evening (6-8 pm).
- RTE does not talk to homes: it ACTIVATES an aggregator, the day before or
  the same day, for a volume (kW) over a slot.

In the simulation
-----------------
- Stress day = weekday with a RED or WHITE Tempo color (color approximated
  from RTE's actual national consumption, grid module).
  If there is none in the period, the busiest working day is used
  (so the demo always has at least one event; this is flagged).
- Slots: 6-8 pm on stress days, + 7-9 am on red days.
- Announcement: the day before at 5 pm.
- Volume = a share of the FLEXIBILITY declared by the district (see reporting.py).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import pandas as pd

from ..config import DemandResponseConfig


@dataclass
class Request:
    """A demand-response request sent by RTE to the district aggregator."""

    start: pd.Timestamp
    end: pd.Timestamp
    announce: pd.Timestamp
    level: str                 # "RED" or "WHITE" (or "FORCED" in the demo)
    volume_kw: float = 0.0      # set by RTE from the declared flexibility

    @property
    def duration_h(self) -> float:
        return (self.end - self.start).total_seconds() / 3600

    @property
    def energy_kwh(self) -> float:
        return self.volume_kw * self.duration_h

    def active(self, t: pd.Timestamp) -> bool:
        return self.start <= t < self.end

    def dict(self) -> dict:
        d = asdict(self)
        return {k: (str(v) if isinstance(v, pd.Timestamp) else v) for k, v in d.items()}


def stress_days(grid: pd.DataFrame, days: list) -> tuple[dict, str]:
    """{date: level} of the stress days among `days`, and a note if one had to be forced."""
    color = grid["tempo"].groupby(grid.index.date).first()
    consumption = grid["consumption_mw"].groupby(grid.index.date).mean()
    out = {}
    for d in days:
        if pd.Timestamp(d).dayofweek < 5 and color.get(d, "BLUE") in ("RED", "WHITE"):
            out[d] = color[d]
    note = ""
    if not out:
        workdays = [d for d in days if pd.Timestamp(d).dayofweek < 5 and d in consumption.index]
        if workdays:
            d = max(workdays, key=lambda x: consumption[x])
            out[d] = "FORCED"
            note = (f"No stress day in the period: simulating a request on the busiest day ({d}). "
                    "Pick a very cold period (e.g. January) for real events.")
    return out, note


def create_requests(grid: pd.DataFrame, sim_index: pd.DatetimeIndex, cfg: DemandResponseConfig) -> tuple[list, str]:
    """List of requests (volume still 0: it is set after the flexibility declaration)."""
    days = sorted(set(sim_index.date))
    stressed, note = stress_days(grid, days)
    tz = sim_index.tz
    requests = []
    for d, level in stressed.items():
        day = pd.Timestamp(d).tz_localize(tz)
        for h0, h1 in cfg.slots:
            if h0 < 12 and level != "RED":
                continue                                   # morning peak: red days only
            start = day + pd.Timedelta(hours=h0)
            announce = day - pd.Timedelta(hours=7)          # the day before at 5 pm
            if announce < sim_index[0]:
                announce = sim_index[0]
            requests.append(Request(start, day + pd.Timedelta(hours=h1), announce, level))
    return requests, note
