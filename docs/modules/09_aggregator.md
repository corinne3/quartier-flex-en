# Module 9 — Aggregator: the AI juggling sheddable groups

**Role**: receive RTE's request and decide, **every 15 minutes**, which appliances to cut and what the batteries
do. Then compute the **net AI balance**.

## The simulation (`simulation.py`)

At each 15-min step:
1. the day before a request, the strategy receives the announcement (it can prepare);
2. it sees the **measured** state (noisy temperature, power of each appliance, cars' needs, batteries);
3. the **comfort rule** applies to everyone: heating cut for 30 min max then 30 min on, water heater 2 h max;
4. the physics moves forward, the battery applies the command and ages;
5. we record the district's grid draw, comfort, and service failures.

**Delivered** demand response = grid draw without shedding − grid draw with shedding (the baseline curve is known
here; it is estimated in reality). Checked **at each 15-min step**, like RTE does: a missed step cannot be offset by
an over-delivered one.

## The strategies (`strategies.py`)

| Strategy | AI? | What it does |
|---|---|---|
| 0 None | no | baseline |
| 1 Cut everything | no | everyone at the same time + battery at full power → big rebound, discomfort |
| 2 Round-robin | no | battery, cars, water heaters, then heaters in turn until the volume is reached |
| 2b Prepared round-robin | no | + battery charged the night before, spread over the request, 30 % margin |
| 3 **AI optimizer** | light | see below |
| 4 **LLM agent** | heavy | the day before, the LLM chooses the optimizer's **settings**, which then executes |

**The AI optimizer**:
- charges the batteries the night before, **just enough** (the midday sun can still be stored);
- **preheats** occupied homes by 1 °C, 1 h before;
- **shadow homes**: for each home, it simulates with the learned model what it would consume **without being
  cut**; the gap with reality is the **catch-up** of the heaters switched back on, which it offsets;
- cars: it knows their needs, hence what they would have consumed without being cut;
- **linear optimization** (SciPy/HiGHS) between heating groups and battery: discomfort cost (measured comfort
  margin, retirees protected), battery wear, penalty if the volume is not held;
- **soft restart**: after the request, EV charging waits for off-peak hours (otherwise a new peak at 8 pm, at the
  highest price), then is **spread over the night** (otherwise everything starts at 10 pm: another peak).
- accepted limit: preheating creates a small bump **before** the request (5–6 pm).

**The LLM agent** (harness engineering): one call per request (2 to 4 with the `view_flexibility` tool), JSON output
validated by Pydantic (`SheddingPlan`): group order, protected groups, preheating, charging. Safe fallback:
the optimizer's default settings. The LLM works at the **strategic** level (the day before); execution every
15 min is left to reliable, sober tools.

## The assessment (`assessment.py`)

| Indicator | Meaning |
|---|---|
| delivered %, hold rate % | energy shed / requested; share of 15-min steps held at ≥ 90 % |
| rebound kWh, over-peak kW | extra consumption in the 2 h that follow; peak created after the request |
| added discomfort °C·h | degree-hours below the tolerated minimum (occupants present), compared with the baseline |
| cold water, uncharged car | service failures (kWh) |
| RTE revenue, bill savings, wear | €; profit = revenue + savings − wear |
| CO2 avoided | avoided grid draw × grid intensity, rebound included |
| AI mWh | measured energy: learning + forecasts + optimization + LLM |

**Net AI gain** = profit (AI strategy) − profit (best strategy without AI) − cost of the AI.

## Test

```powershell
quartier demo aggregator                 # without LLM
quartier flex                            # everything, fake LLM
quartier flex --llm ollama               # with the real LLM
pytest tests/test_aggregator.py -v
```
