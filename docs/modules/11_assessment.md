# Module 11 — Assessment (self-consumption) and sizing

**Role**: assemble all the modules, run the building hour by hour, and answer the challenge's question.

## Simulation (`simulation.py`)
Every hour:
1. actual production and consumption;
2. the strategy's decision;
3. real-time layer (store the surplus, cover the shortfall within the allowed limit, or charge from the grid);
4. the battery applies the command and ages;
5. purchase or resale of the remainder.

Checked by a test: **energy conservation** at every hour.

## Indicators (`kpi.py`)
- Self-consumption, self-sufficiency, purchases and resales (kWh), bill (€), CO2 (kg).
- Battery wear (SOH points lost, and in €).
- AI energy (kWh, €, CO2), LLM calls, tokens.
- Everything also expressed **per year and per home**.

## Net AI balance
```
net AI gain = (bill + wear of the best strategy WITHOUT AI)
            − (bill + wear of the AI strategy)
            − cost of the energy consumed by the AI
```
Computed in €, in kWh and in CO2. The LLM agents are compared with the rule **on the same days**.

## Sizing (`sizing.py`)
- Sweep: kWp × number of packs.
- For each combination: self-sufficiency, annual bill, savings vs "everything from the grid", investment, **payback time**, CO2 avoided.
- ⚠️ Extrapolated to the full year from the chosen period: redo it in winter AND in summer.

## Test
```bash
quartier run                                            # all strategies without LLM
quartier run --llm ollama --llm-days 1                  # with the LLM agents
quartier size --homes 20
pytest tests/test_assessment.py -v
```

## The DEMAND-RESPONSE assessment (Quartier Flex project)

The main assessment in this repository is the demand-response one: see [09_aggregator.md](09_aggregator.md).

```powershell
quartier flex                            # all strategies, fake LLM agent
quartier flex --llm ollama               # with the real LLM (a few minutes)
```
