# Module 5 — Connected appliances

**Role**: describe what each home can **shed**, and **learn** how it reacts.

## Who has what (`portfolio.py`)

Each home = an **occupant type** (module 4) × **equipment** × connected or not.

| Parameter (`config.AppliancesConfig`) | Default | To be checked |
|---|---|---|
| electric heating | 55 % of homes | yes (CEREN, ADEME) |
| electric water heater | 65 % (70 % of them on off-peak control) | yes |
| electric car charged at home | 25 % (except students) | yes |
| controllable box (smart controller) | 80 % | assumption |

**Sheddable group** = (occupant type, appliance), for example `retirees/heating`. This is the granularity the
aggregator reasons at (module 9): the "occupant type × appliance type" cross.

| Appliance | Sheddable? | Limit |
|---|---|---|
| heating | yes, **30 min max** then 30 min on | comfort: temperature ≥ setpoint − 1 °C (− 0.5 °C for retirees) |
| water heater | yes, 2 h max | hot water available |
| EV charging | yes: it is **delayed** | charged by departure (7 am) |
| lighting, cooking, fridge, IT | **no** | low power, comfort, food safety |

## The physics (`physics.py`)

- **Heating: 1R1C model.** The home is a "bathtub of heat":
  `T(t+dt) = T(t) + dt/C × (P_heating + gains − UA × (T − T_out))`
  - C = thermal inertia (kWh/°C), UA = losses (kW/°C), gains = occupants, appliances, sun.
  - The thermostat offsets the losses and gradually catches up with the gap to the setpoint (in ~1 h 30).
- **Key takeaway**: cutting a heater for 30 min lowers the temperature by a few tenths of a degree,
  then the thermostat **catches up**. The energy is mostly **shifted**, not removed: this is the **rebound effect**.
- **Water heater**: an energy store; hot-water draws in the morning, at noon and in the evening.
- **Car**: plugged in in the evening, needs ~9 kWh, 7 kW charger.
- **Real life**: every day, each household shifts its schedule (± 1 h) and is sometimes away. The aggregator only
  knows the thermostat's **schedule**.

## The AI (`learning.py`)

- **System identification**: the box measures temperature (sensor noise 0.1 °C) and power every
  15 min. The 1R1C model is **linear** in (UA/C, 1/C, gains/C): a least-squares regression
  gives UA and C for **each home**, without ever cutting it.
- **Forecast**: each home is simulated with its learned model and the forecast weather. This captures the heating
  **restarts** (after the night setback, when people get home from work), which create the 7 am and 6 pm peaks.
- **Diversity (aggregation effect)**: the forecast error is large for a single home (unpredictable schedules), much
  smaller for the district (the deviations cancel out). That is why RTE deals with an **aggregator**.

## Test

```powershell
quartier demo appliances
pytest tests/test_appliances.py -v
```

The demo shows: the district's consumption by use, the heating forecast, and **the experiment of a heater cut for
30 min** (temperature, then catch-up).

## Limits

- Consumption, appliances and behavior are **simulated** (no public individual Linky data).
- 1R1C model: simple but standard; no separate rooms, no heat pump (COP).
