"""
tariffs.py: how much does a kWh bought from the grid cost, and how much does a kWh sold back earn?

Three tariff options (price per kWh, excluding the subscription):
- BASE : same price at every hour;
- HPHC : off-peak hours (10pm - 6am) cheaper than peak hours;
- TEMPO: 3 day colors (blue, white, red) × off-peak / peak hours.
         Peak hours on red days are very expensive: that is where
         a full battery earns the most.

Surplus resale: fixed price per kWh fed in (much lower than the purchase
price) -> it is almost always better to consume your own solar power (or
store it) than to sell it. That is the whole economics of the battery.

⚠️ Prices = orders of magnitude TO BE CHECKED (see config.TariffConfig).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import TariffConfig


def buy_price(index: pd.DatetimeIndex, cfg: TariffConfig, tempo: pd.Series | None = None) -> pd.Series:
    """Purchase price (€/kWh) for each hour."""
    hc = np.isin(index.hour, cfg.hc_hours)
    if cfg.option == "BASE":
        p = np.full(len(index), cfg.base)
    elif cfg.option == "HPHC":
        p = np.where(hc, cfg.hc, cfg.hp)
    elif cfg.option == "TEMPO":
        colors = tempo.reindex(index).fillna("BLUE").to_numpy() if tempo is not None else np.array(["BLUE"] * len(index))
        p = np.array([cfg.tempo[c][0 if is_hc else 1] for c, is_hc in zip(colors, hc)])
    else:
        raise ValueError(f"Unknown tariff option: {cfg.option}")
    return pd.Series(p, index=index, name="price_buy_eur_kwh")


def sell_price(index: pd.DatetimeIndex, cfg: TariffConfig) -> pd.Series:
    return pd.Series(cfg.sell_surplus, index=index, name="price_sell_eur_kwh")
