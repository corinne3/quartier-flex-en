"""
MODULE 10 — MEASUREMENT: how much energy does the AI itself consume?

Reused from Bilan Net (validated method): Python CPU time, CPU time of the
whole machine (for the LLM running in Ollama), duration reported by
Ollama; converted to energy using the processor's TDP.
Three buckets: setup (once), shared (every hour, for the building),
per_house (every hour, per home — not used here: decisions are
made for the whole building).

    from quartierflex.measure import Meter
    m = Meter(); with m.measure("shared"): ...; m.report()

Demo: quartier demo measure      Sheet: docs/modules/10_measurement.md
"""

from .meter import Meter, maybe_codecarbon  # noqa: F401
