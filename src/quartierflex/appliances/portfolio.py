"""
portfolio.py: the PORTFOLIO of the district's homes and their connected appliances.

Each home = an occupant TYPE (usage module: family, retirees...)
          × EQUIPMENT (electric heating? water heater? electric car?)
          × connected or not (smart controller box: otherwise, no demand response possible).

The appliances, and why they are (or are not) sheddable
-------------------------------------------------------
| appliance           | sheddable?                         | what limits it                       |
|---------------------|------------------------------------|--------------------------------------|
| heating             | yes, 30 min max, in rotation       | comfort (minimum temperature)        |
| water heater        | yes, up to 2 h                     | available hot water                  |
| EV charging         | yes: charging is DELAYED           | being charged at morning departure   |
| lighting, cooking,  | no                                 | low power, comfort, safety           |
| cold, IT            |                                    | (the fridge: food-safety risk)       |

The "non-sheddable loads" come from the usage module (consumption profiles).

Sheddable group = (occupant type, appliance), e.g. "retirees/heating".
This is the level at which the aggregator reasons: each group has its own sheddable
power and "tolerance" (retirees are home during the day and more vulnerable
to cold; working people are away: heating already turned down).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..config import AppliancesConfig, BuildingConfig
from ..usage import make_building

APPLIANCES = ("heating", "water_heater", "ev")
NAMES_APPLIANCES = {"heating": "heating", "water_heater": "water heater", "ev": "EV charging"}

# Awake presence (hours), weekday / weekend. Night (sleep) is handled separately.
PRESENCE = {
    "family": (list(range(6, 8)) + list(range(17, 23)), list(range(7, 23))),
    "working_couple": (list(range(6, 8)) + list(range(18, 23)), list(range(8, 23))),
    "retirees": (list(range(7, 23)), list(range(7, 23))),
    "remote_worker": (list(range(7, 23)), list(range(8, 23))),
    "student": ([7] + list(range(18, 24)), list(range(10, 24))),
}
NIGHT = {k: list(range(0, 6)) + [23] for k in PRESENCE} | {"student": list(range(0, 7))}
SETPOINT = {"present": 20.0, "night": 18.0, "absent": 16.5}
SETPOINT_RETIREES = 21.0
FLOOR_AREA_M2 = {"family": 85, "working_couple": 60, "retirees": 70, "remote_worker": 65, "student": 25}
WATER_HEATER_KWH_DAY = {"family": 7.0, "working_couple": 4.5, "retirees": 4.0, "remote_worker": 4.5, "student": 1.8}
EV_ARRIVAL = {"family": 18, "working_couple": 19, "retirees": 17, "remote_worker": 18, "student": 19}


@dataclass
class Portfolio:
    """The whole district, as arrays (one element per home): fast to simulate."""

    names: list
    types: np.ndarray          # occupant type
    floor_area: np.ndarray        # m²
    connected: np.ndarray       # bool: smart controller box
    heating: np.ndarray      # bool: electric heating
    water_heater: np.ndarray            # bool: electric water heater
    water_heater_hc: np.ndarray         # bool: water heater on off-peak contactor
    ev: np.ndarray             # bool: electric car
    ua_kw_k: np.ndarray        # heat loss (kW per °C of difference)  — TRUE values (unknown to the AI)
    c_kwh_k: np.ndarray        # inertia (kWh per °C)                    — same
    p_heating_max: np.ndarray
    tolerance: np.ndarray      # °C below setpoint accepted (when present)
    water_heater_kwh_day: np.ndarray
    ev_arrival: np.ndarray     # car arrival hour
    base_kw: pd.DataFrame      # non-sheddable loads (kW), at simulation step
    cfg: AppliancesConfig

    @property
    def n(self) -> int:
        return len(self.names)

    def group(self, appliance: str) -> np.ndarray:
        """Sheddable group name of each home for this appliance ('' if not applicable)."""
        has = {"heating": self.heating, "water_heater": self.water_heater, "ev": self.ev}[appliance] & self.connected
        return np.where(has, np.char.add(self.types.astype(str), f"/{appliance}"), "")

    def groups(self) -> list:
        out = []
        for a in APPLIANCES:
            out += sorted(set(self.group(a)) - {""})
        return out

    def table(self) -> pd.DataFrame:
        """Cross table: number of connected homes per occupant type × appliance."""
        rows = {}
        for k in sorted(set(self.types)):
            m = (self.types == k)
            rows[k] = {"homes": int(m.sum()), "connected": int((m & self.connected).sum()),
                       **{NAMES_APPLIANCES[a]: int((m & self.connected & getattr(self, a)).sum()) for a in APPLIANCES}}
        df = pd.DataFrame(rows).T
        df.loc["TOTAL"] = df.sum()
        return df


def _tables(portfolio: "Portfolio"):
    """Schedule of each home: setpoint and presence by (weekend?, hour)."""
    sp = np.full((portfolio.n, 2, 24), SETPOINT["absent"])
    pres = np.zeros((portfolio.n, 2, 24), dtype=bool)
    for i, k in enumerate(portfolio.types):
        for we in (0, 1):
            hrs = PRESENCE[k][we]
            present_sp = SETPOINT_RETIREES if k == "retirees" else SETPOINT["present"]
            for h in range(24):
                if h in hrs:
                    sp[i, we, h], pres[i, we, h] = present_sp, True
                elif h in NIGHT[k]:
                    sp[i, we, h] = SETPOINT["night"]
    return sp, pres


def setpoints(portfolio: "Portfolio", index: pd.DatetimeIndex, real_life: bool = False, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """
    Temperature setpoint (°C) and awake presence (bool), arrays [time steps × homes].

    real_life=False: the thermostat SCHEDULE (what the aggregator knows).
    real_life=True:  REAL life: each day, each household gets up / comes home up to 1 h earlier or later,
                     and is sometimes away all day. The aggregator does not know this: it is what makes a
                     single home unpredictable, and the district (where deviations cancel out) predictable.
    """
    sp_t, pres_t = _tables(portfolio)
    hours = index.hour.to_numpy()
    we = (index.dayofweek.to_numpy() >= 5).astype(int)
    n_t = len(index)
    i = np.arange(portfolio.n)[None, :]
    if not real_life:
        h = np.broadcast_to(hours[:, None], (n_t, portfolio.n))
        return sp_t[i, we[:, None], h], pres_t[i, we[:, None], h]
    rng = np.random.default_rng(seed + 303)
    days = pd.DatetimeIndex(index).normalize()
    uniq, j = np.unique(days, return_inverse=True)
    shift = rng.choice([-1, 0, 0, 0, 1], size=(len(uniq), portfolio.n))     # schedule shift for the day (h)
    absent = rng.random((len(uniq), portfolio.n)) < 0.04                  # day away
    h = (hours[:, None] - shift[j]) % 24
    sp = sp_t[i, we[:, None], h]
    pres = pres_t[i, we[:, None], h]
    abs_k = absent[j] & ~np.isin(hours, list(range(0, 7)) + [23])[:, None]
    sp = np.where(abs_k, SETPOINT["absent"], sp)
    pres = pres & ~abs_k
    return sp, pres


def make_portfolio(bcfg: BuildingConfig, acfg: AppliancesConfig, index: pd.DatetimeIndex, seed: int = 0,
              step_min: int = 15) -> Portfolio:
    """
    Draws the portfolio at random (reproducible) and prepares the non-sheddable loads
    at simulation step (hourly index -> steps of step_min minutes, in kW).
    """
    loads, types_d = make_building(bcfg, index, seed=seed)       # kWh per hour = average kW
    names = list(loads.columns)
    types = np.array([types_d[n] for n in names])
    rng = np.random.default_rng(seed + 101)
    n = len(names)
    floor_area = np.array([FLOOR_AREA_M2[t] for t in types]) * rng.uniform(0.8, 1.25, n)
    insulation = rng.lognormal(0, 0.25, n)                         # some homes are poorly insulated
    fine = pd.date_range(index[0], index[-1] + pd.Timedelta("1h"), freq=f"{step_min}min", inclusive="left")
    base = loads.reindex(fine, method="ffill")
    base.attrs["source"] = loads.attrs.get("source")
    tol = np.where(types == "retirees", acfg.tolerance_vulnerable_c, acfg.tolerance_c)
    return Portfolio(
        names=names, types=types, floor_area=floor_area,
        connected=rng.random(n) < acfg.share_connected,
        heating=rng.random(n) < acfg.share_heating_electric,
        water_heater=rng.random(n) < acfg.share_water_heater_electric,
        water_heater_hc=rng.random(n) < acfg.share_water_heater_off_peak,
        ev=(rng.random(n) < acfg.share_ev) & (types != "student"),
        ua_kw_k=floor_area * acfg.heat_loss_w_per_k_m2 * insulation / 1000,
        c_kwh_k=floor_area * acfg.inertia_kwh_per_k_m2 * rng.uniform(0.8, 1.2, n),
        p_heating_max=floor_area * acfg.p_heating_kw_per_m2,
        tolerance=tol,
        water_heater_kwh_day=np.array([WATER_HEATER_KWH_DAY[t] for t in types]) * rng.uniform(0.7, 1.3, n),
        ev_arrival=np.array([EV_ARRIVAL[t] for t in types]) + rng.integers(-1, 2, n),
        base_kw=base, cfg=acfg,
    )
