# Module 1 — Weather

**Role**: provide hourly weather in Valence, both **observed** and **forecast the day before**, plus the sun's position.

## Inputs / outputs
- Inputs: location (`Site`), start date, number of days, days of history.
- Outputs: hourly table (index = start of the hour, Paris time zone):
  - `ghi_wm2`: global horizontal irradiance (W/m²);
  - `dni_wm2`: direct normal irradiance (W/m²);
  - `dhi_wm2`: diffuse irradiance (W/m²);
  - `temp_c`: temperature (°C);
  - `cloud_pct`: cloud cover (%).

## Method
- **Observed**: Open-Meteo "archive" API, free, no key.
- **Forecast**: Open-Meteo "historical forecast" API, which contains forecasts as they were published. This is what an AI is allowed to use: it does not know the future.
- **Sun position**: simplified NOAA formulas (`sun.py`), accurate to ~0.5°.
- Open-Meteo gives the average of the **previous** hour: the index is shifted by 1 h to represent the **start** of the hour.
- **Offline fallback**: clear sky (Haurwitz model) × persistent random clouds. Always flagged in `attrs["source"]`.

## Test on its own
```bash
quartier demo weather
pytest tests/test_weather.py -v
```
The tests check:
- the sun's elevation on 21 June at solar noon (zenith ≈ 21.5°);
- that the sun is down at night;
- that offline weather is flagged and consistent.

## Limits and ideas
- A single station (Open-Meteo grid point) for the whole building.
- Idea for the team: compare with a real local station, or try other cities (`Site(lat, lon)`).
