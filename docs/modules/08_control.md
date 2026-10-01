# Module 8 — Battery control (self-consumption)

**Role**: decide, every hour, when to charge and when to discharge the battery.

## The contract (`base.py`)
Each strategy returns 2 numbers:
- `grid_charge_kw`: charge from the grid;
- `max_discharge_kw`: allowed discharge (0 = keep the energy for later).

A shared "real-time" layer **always** stores solar surplus. Only the "brain" changes.

## The strategies
| Strategy | Family | Principle |
|---|---|---|
| No battery | baseline | panels only |
| Self-consumption rule | rule | stores the surplus, gives it back when there is a shortfall |
| Tariff rule | rule | + charges at night during off-peak hours if tomorrow's sun ("same as yesterday") will not be enough |
| **Optimizer + AI** | optimization + AI | ML forecasts (modules 2 and 4) + linear programming over 24 h, re-planned every hour |
| **Hourly LLM agent** | LLM agent | the LLM decides every hour (tool `get_forecast`) |
| **Routed LLM agent** | LLM agent | the LLM decides only at 9 pm and 6 am |

## The optimizer (`optimizer.py`)
- Minimizes: purchases − resales + **wear** × energy going through the battery − value of the remaining energy.
- Constraints: electrical balance, battery balance, SOC and power bounds.
- **Wear cost** = pack price / (2 × cycles left before SOH 60 %). A more worn battery costs more to use: the optimizer spares it.
- Solver: HiGHS (via SciPy), a few milliseconds per plan.

## LLM agent (`agent.py`)
- Agentic harness from Bilan Net: loop, typed tools, budget, validation, fallback to the rule.
- Local LLM via Ollama (`qwen2.5:1.5b`); `FakeLLM` for the tests.

## Test on its own
```bash
quartier demo control
pytest tests/test_control.py -v
```
The tests check:
- that the optimizer charges when it is cheap and discharges when it is expensive;
- that all the rules simulate;
- that the agent works with the fake LLM.
