"""
MODULE 6 — GRID: purchase/resale tariffs and CO2 of the national grid (RTE).

    from quartierflex.grid import load_grid, buy_price, sell_price

Demo: quartier demo grid      Tests: pytest tests/test_grid.py
Doc: docs/modules/06_grid.md
"""

from .co2 import approx_tempo, load_grid  # noqa: F401
from .tariffs import buy_price, sell_price  # noqa: F401
