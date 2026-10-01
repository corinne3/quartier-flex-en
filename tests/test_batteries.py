"""Module 3 — batteries. Run on its own: pytest tests/test_batteries.py -v"""
import pandas as pd
import pytest

from quartierflex.batteries import BatteryBank, BatteryPack, demo_profile, estimate_soh, run_bms
from quartierflex.config import BatteryConfig


def test_efficiency_and_bounds():
    cfg = BatteryConfig()
    p = BatteryPack("p", 40, 0.75, cfg, soc=0.5)
    e0 = p.energy_kwh
    got = p.step(5.0)
    assert got == pytest.approx(5.0)
    # 5 kWh in -> 4.75 stored (give or take a hair: the battery aged during the hour)
    assert p.energy_kwh - e0 == pytest.approx(5.0 * cfg.eta_charge, rel=1e-3)
    for _ in range(50):
        p.step(100.0)
    assert p.soc <= cfg.soc_max + 1e-9                                     # never above 90%
    for _ in range(50):
        p.step(-100.0)
    assert p.soc >= cfg.soc_min - 1e-9                                     # never below 10%


def test_worn_battery_is_less_powerful_and_ages():
    cfg = BatteryConfig()
    new, worn = BatteryPack("a", 40, 0.9, cfg), BatteryPack("b", 40, 0.6, cfg)
    assert worn.p_max_kw < new.p_max_kw
    soh0 = worn.soh
    for _ in range(24 * 30):
        worn.step(5.0 if worn.soc < 0.8 else -5.0)
    assert worn.soh < soh0


def test_bank_spares_the_most_worn_pack():
    bank = BatteryBank(BatteryConfig(capacities_kwh=[40, 40], soh_init=[0.85, 0.62]), soc0=0.5)
    e = [p.energy_kwh for p in bank.packs]
    bank.step(-6.0)
    given = [e0 - p.energy_kwh for e0, p in zip(e, bank.packs)]
    assert given[1] < given[0]


def test_bms_corrects_drift_and_estimates_health():
    cfg = BatteryConfig()
    pack = BatteryPack("p", 40, 0.75, cfg, soc=0.5)
    days = 200
    idx = pd.date_range("2024-01-01", periods=24 * days, freq="h")
    log = run_bms(pack, demo_profile(days), idx, soh_guess=0.80)
    err_cc = (log["soc_counting"] - log["soc_true"]).abs().mean()
    err_est = (log["soc_estimated"] - log["soc_true"]).abs().mean()
    assert err_est < err_cc / 2
    est = estimate_soh(log, 40)
    assert abs(est["soh_estimated"] - pack.soh) < 0.05
