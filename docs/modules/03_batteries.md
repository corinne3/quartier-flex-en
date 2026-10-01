# Module 3 — Second-life batteries

**Role**: simulate **reused** electric-car battery packs, their aging, and what the BMS measures and estimates.

## Vocabulary
- **Nominal capacity**: what the pack stored when new (e.g. 40 kWh).
- **SOH** (state of health): current capacity / new capacity. In second life: ~70–80 % at the start, end of life around 60 %.
- **SOC** (state of charge): fill level from 0 to 100 % of the **current** capacity.
- **Equivalent full cycle (EFC)**: energy that went through the battery, expressed as one full charge + one full discharge.

## Model (`model.py`)
- Usable capacity = nominal × SOH, between SOC 10 % and 90 %.
- Charge and discharge efficiency: 95 % each (round trip ≈ 90 %).
- Max power reduced if the battery is worn: `P_max × min(1, SOH/0.8)`.
- Aging:
  - per cycle: `ΔSOH = 0.012 % × EFC`;
  - over time: `1.5 %/year × (0.5 + SOC)`, faster if the battery stays highly charged.
- **Bench of several packs**: power is shared in proportion to what each pack can still deliver. The most worn ones work less.

## BMS (`bms.py`): measure and estimate
No sensor displays the SOC or SOH: they have to be **estimated**.

| Estimator | Principle | Limit |
|---|---|---|
| SOC by coulomb counting | add up energy in and out | sensor error accumulates: **drift** |
| Corrected SOC | recalibrate using the **resting voltage** (OCV curve) | needs rest periods |
| **Learned SOH** | between 2 rests: capacity = energy / ΔSOC; then robust regression (Huber) over the history | biased if the energy sensor is biased (here +2 %) |

- AI output: current SOH, trend per cycle, **cycles left before 60 %**.
- Battery control (module 8) uses the state of health to compute the **wear cost** of 1 kWh.

## Test on its own
```bash
quartier demo batteries
pytest tests/test_batteries.py -v
```
The tests check:
- efficiency and SOC bounds;
- that a worn battery is less powerful and ages;
- that the bench spares the most worn pack;
- that the BMS cuts the SOC error by more than 2× and estimates the SOH to within ±5 points.

## Parameters for the team to check (`config.BatteryConfig`)
Typical SOH at recovery, aging speed in second life, price per kWh of a reused pack.
