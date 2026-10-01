"""
MODULE 1 — WEATHER: observed and forecast weather, and the sun's position.

    from quartierflex.weather import load_weather_pair, solar_position
    obs, forecast = load_weather_pair(Site(), "2024-05-06", days=7, history_days=60)

Demo: quartier demo weather      Tests: pytest tests/test_weather.py
Sheet: docs/modules/01_weather.md
"""

from .sun import clear_sky_ghi, solar_position  # noqa: F401
from .source import load_weather, load_weather_pair  # noqa: F401
