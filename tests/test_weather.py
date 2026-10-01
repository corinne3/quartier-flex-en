"""Module 1 — weather. Run on its own: pytest tests/test_weather.py -v"""
import numpy as np
import pandas as pd

from quartierflex.config import Site
from quartierflex.weather import load_weather_pair, solar_position


def test_sun_solar_noon_summer():
    # June 21, 1:30 pm Paris time ≈ solar noon in Valence: sun high (zenith ≈ 44.9 - 23.4 ≈ 21.5°)
    t = pd.DatetimeIndex([pd.Timestamp("2024-06-21 13:30", tz="Europe/Paris")])
    sp = solar_position(t, 44.93, 4.89)
    assert 19 < sp["zenith_deg"].iloc[0] < 24
    assert 170 < sp["azimuth_deg"].iloc[0] < 190      # due south


def test_sun_down_at_night():
    t = pd.DatetimeIndex([pd.Timestamp("2024-06-21 02:00", tz="Europe/Paris")])
    assert solar_position(t, 44.93, 4.89)["zenith_deg"].iloc[0] > 90


def test_offline_weather_flagged_and_consistent():
    obs, fc = load_weather_pair(Site(), "2024-05-06", 3, 2, offline=True)
    assert "SYNTH" in obs.attrs["source"]
    assert len(obs) == 5 * 24 and set(obs.columns) >= {"ghi_wm2", "dni_wm2", "dhi_wm2", "temp_c", "cloud_pct"}
    assert (obs["ghi_wm2"] >= 0).all()
    assert not np.allclose(obs["ghi_wm2"], fc["ghi_wm2"])   # the forecast is not reality
