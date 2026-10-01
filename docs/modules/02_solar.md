# Module 2 — Solar

**Role**: compute the panels' output, simulate what the meter measures, and **forecast** production.

## Physics (`physics.py`)
1. Sun position (module 1).
2. Irradiance received by the tilted panel (POA):
   `POA = direct × cos(incidence) + diffuse × (1+cos i)/2 + global × albedo × (1−cos i)/2`
3. Cell temperature: `T_cell = T_air + (NOCT − 20)/800 × POA`
4. Power: `P = kWp × POA/1000 × (1 + γ (T_cell − 25)) × PR`, capped at peak power.
   - γ = −0.4 %/°C;
   - PR (performance ratio) = 0.86.

## Meter (`meter.py`)
Real measurements differ from theory. We **simulate** this:
- ±2 % noise;
- soiling that builds up, then gets washed off;
- outages lasting a few hours.

⚠️ To be replaced with a real inverter export if the team finds one.

## Forecast (`forecast.py`), the AI part
| Method | Principle | Compute cost |
|---|---|---|
| Persistence | "tomorrow = today" | ~0 |
| Physics | physical model applied to the **forecast** weather | very low |
| **AI (ML)** | gradient boosting trained on history: forecast weather → **measured** production, per kWp | low, measured |

- Metric: **nMAE** = mean absolute error / mean production, over daylight hours.
- The AI can correct what physics ignores: soiling, weather-forecast bias.

## Sizing variable
`PVConfig.kwc` (plus tilt, orientation). Rule of thumb: 1 kWp ≈ 2.5 panels ≈ 5 m².

## Test on its own
```bash
quartier demo solar --kwc 30
pytest tests/test_solar.py -v
```
The tests check:
- a plausible annual yield (1,100–1,700 kWh/kWp/year);
- zero at night, and never more than peak power;
- that the meter measures slightly less than the model;
- that the AI forecast works.

## For the team
Check Valence's expected yield with **PVGIS** (the European Commission's free tool) and compare it with the model.
