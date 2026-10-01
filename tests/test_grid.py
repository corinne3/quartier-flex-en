"""Module 6 — grid. Run on its own: pytest tests/test_grid.py -v"""
import pandas as pd

from quartierflex.config import TariffConfig
from quartierflex.grid import buy_price, load_grid, sell_price

IDX = pd.date_range("2024-01-15", periods=48, freq="h", tz="Europe/Paris")


def test_off_peak_cheaper():
    p = buy_price(IDX, TariffConfig(option="HPHC"))
    assert p.iloc[3] == TariffConfig().hc and p.iloc[12] == TariffConfig().hp


def test_tempo_red():
    tempo = pd.Series("RED", index=IDX)
    p = buy_price(IDX, TariffConfig(option="TEMPO"), tempo)
    assert p.iloc[12] == TariffConfig().tempo["RED"][1]


def test_resale_cheaper_than_purchase():
    assert (sell_price(IDX, TariffConfig()) < buy_price(IDX, TariffConfig())).all()


def test_offline_grid_flagged():
    g = load_grid("2024-01-15", 2, 0, offline=True)
    assert "SYNTH" in g.attrs["source"] and set(g["tempo"]) <= {"BLUE", "WHITE", "RED"}
