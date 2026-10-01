# Module 10 — Measuring the AI's compute energy

**Role**: measure the energy consumed by the AI computations, for the net balance.

- Reused from **Bilan Net**, a validated approach: 3 cross-checked methods.
  - CPU of the Python process;
  - CPU of the whole machine (essential for an LLM running in Ollama);
  - compute time reported by Ollama.
- Energy = CPU seconds × (TDP / logical cores). Your PC's TDP: 28 W (`--tdp`).
- Cost buckets:
  - `setup`: training, once;
  - `shared`: every hour, for the building (forecasts, optimization, LLM calls).
- Conversion: to € at the average purchase price, and to CO2 at the grid's average intensity.

## Test on its own
```bash
quartier demo measure
```

Order of magnitude measured with Bilan Net on this PC: 1 call to Qwen 2.5 1.5B ≈ 0.07 Wh.
