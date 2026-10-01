# Module 6 — Grid: tariffs and CO2

**Role**: give, hour by hour, the cost of a purchased kWh, the revenue from a resold kWh, and the grid's CO2.

## Tariffs (`tariffs.py`), orders of magnitude to be checked
| Option | Principle |
|---|---|
| BASE | same price all day |
| HPHC | cheaper off-peak hours 10 pm–6 am |
| TEMPO | blue, white, red days × off-peak/peak hours; red peak hours are very expensive |

- Resale of surplus: fixed feed-in price, much lower than the purchase price.
- Storing surplus to use it later is therefore profitable: that is the battery's business case.
- Subscription not counted (identical for all strategies).

## CO2 (`co2.py`)
- RTE éCO2mix via the ODRÉ API: hourly **average** carbon intensity (reused from the Bilan Net project).
- Tempo colors **approximated** from national consumption.

## Test on its own
```bash
quartier demo grid --tariff TEMPO --start 2024-01-15
pytest tests/test_grid.py -v
```

## For the team
Look up current prices (Base, peak/off-peak, Tempo) and the feed-in price for surplus, then put them in `config.TariffConfig`.
