# Module 4 — Home usage

**Role**: produce each home's consumption, then **learn** from these measurements.

## Generation (`profiles.py`), SIMULATED data
Individual Linky (smart meter) load curves are personal data and not public. So we simulate them.
- 5 household types, each with its annual consumption (excluding heating, to be checked) and its schedule:
  - family: 4,000 kWh;
  - working couple: 2,600 kWh;
  - retirees: 2,800 kWh;
  - remote worker: 3,200 kWh;
  - student: 1,400 kWh.
- Weekday ≠ weekend, +25 % in winter, −25 % in summer (lighting, cooking).
- Randomness: busier or quieter days, absences (3 %), washing machine and dishwasher.
- Adjustable building mix (`BuildingConfig.mix`).

## Learning (`learning.py`), the AI part
1. **Recognize habits** (k-means, unsupervised):
   - normalized shape of each day → clusters;
   - each home gets the cluster of the majority of its days;
   - automatic names: "home during the day", "evening peak", "night owl";
   - quality: ARI (1 = perfectly recovers the true types, 0 = chance).
2. **Forecast the building's consumption** 24 h ahead (gradient boosting), compared with "same as yesterday" (D-1) and "same as last week" (D-7).
   - Features: hour, day, season, D-1, D-7, 7-day average, forecast temperature.

Why it matters for control: "home during the day" households consume solar directly, "evening peak" ones need the battery.

## Test on its own
```bash
quartier demo usage --homes 20
pytest tests/test_usage.py -v
```

## Privacy
These computations can run **locally** in the building. Only aggregates leave it.
