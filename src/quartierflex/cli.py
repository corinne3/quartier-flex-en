"""
cli.py: the "quartier" command line.

    quartier check                      checks Python, internet access (weather, RTE), Ollama
    quartier demo <module>              demo of ONE module (text + image in results/)
         modules: weather solar batteries usage appliances grid demand_response control aggregator measure assessment
    quartier flex [options]             THE ASSESSMENT: RTE requests, all aggregator strategies, net AI balance
    quartier run [options]              battery control for self-consumption (previous project)
    quartier size [options]             sizing sweep: panels × batteries
    quartier interface                  opens the interface (one page per module)
    quartier video [options]            builds the demo video (synthetic voice) + the conclusion slide

Common options: --start 2024-01-15 --days 7 --kwc 120 --homes 60 --packs 4 --tariff TEMPO --offline --model
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys

from .config import RESULTS_DIR, ROOT, Scenario


def scenario_from(args) -> Scenario:
    scn = Scenario(start=args.start, days=args.days, offline=args.offline)
    scn.pv.kwc = args.kwc
    scn.building.n_homes = args.homes
    scn.tariff.option = args.tariff
    scn.compute.cpu_tdp_w = args.tdp
    if args.packs is not None:
        cap = scn.battery.capacities_kwh[0]
        soh = sum(scn.battery.soh_init) / len(scn.battery.soh_init)
        scn.battery.capacities_kwh = [cap] * args.packs
        scn.battery.soh_init = [soh] * args.packs
    return scn


def cmd_check(args) -> int:
    import platform

    print(f"Python {platform.python_version()} — {platform.system()} {platform.release()}")
    ok = True
    for mod in ("numpy", "pandas", "sklearn", "scipy", "pydantic", "httpx", "psutil", "matplotlib", "streamlit"):
        try:
            m = __import__(mod)
            print(f"  ✅ {mod} {getattr(m, '__version__', '')}")
        except ImportError:
            print(f"  ❌ {mod} missing")
            ok = ok and mod == "streamlit"
    import pandas as pd

    from .config import Site
    from .weather.source import _download
    from .grid.co2 import _download as rte

    for name, f in [("Observed weather", lambda: _download("observed", Site(), pd.Timestamp("2024-05-06"), pd.Timestamp("2024-05-07"))),
                    ("Forecast weather", lambda: _download("forecast", Site(), pd.Timestamp("2024-05-06"), pd.Timestamp("2024-05-07"))),
                    ("RTE CO2", lambda: rte(pd.Timestamp("2024-05-06"), pd.Timestamp("2024-05-07")))]:
        try:
            d = f()
            print(f"  ✅ {name}: {len(d)} hours")
        except Exception as e:
            print(f"  ⚠️ {name}: {e} (fallback used)")
    from .agent.llm import OllamaClient

    print("  ✅ Ollama ready" if OllamaClient(model=args.model).ping() else f"  ⚪ Ollama not found or model {args.model} missing (optional)")
    return 0 if ok else 1


def cmd_demo(args) -> int:
    from .demo import DEMOS

    scn = scenario_from(args)
    r = DEMOS[args.module](scn)
    r.print()
    for p in r.save(args.module):
        print(f"📊 {p}")
    return 0


def cmd_run(args) -> int:
    from .assessment import run_all, summary_table
    from .control import ALL

    scn = scenario_from(args)
    keys = args.strategies or [k for k in ALL if not k.startswith("agent")] + (
        ["agent_hourly", "agent_route"] if args.llm == "ollama" else [])
    out = run_all(scn, keys, llm_kind=args.llm, llm_model=args.model, llm_days=args.llm_days)
    print("\nSources:", out["meta"]["sources"])
    print(summary_table(out).to_string())
    print(f"\nFull results: {RESULTS_DIR / 'latest.json'}")
    return 0


def cmd_demand_response(args) -> int:
    import pandas as pd

    from .aggregator import ALL, run_demand_response, table

    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 30)
    scn = scenario_from(args)
    keys = args.strategies or [k for k in ALL if k != "agent" or args.llm != "none"]
    out = run_demand_response(scn, keys, llm_kind=args.llm if args.llm != "none" else "fake", llm_model=args.model)
    print("\nSources:", out["meta"]["sources"])
    print("\nRTE requests:")
    for d in out["requests"]:
        print(f"  {d['start'][:16]} -> {d['end'][11:16]}  {d['level']:6s} {d['volume_kw']:.0f} kW")
    print("\n" + table(out).T.to_string())
    for n, r in out["results"].items():
        b = r.get("assessment_ai")
        if b:
            print(f"\n{r['label']} vs {r['reference_ai']}: NET gain {b['gain_net_eur']:+.2f} €, comfort "
                  f"{b['comfort_wins_degh']:+.1f} °C·h, net CO2 {b['co2_net_kg']:+.1f} kg, "
                  f"{b['kwh_shed_per_wh_ai']:.0f} kWh shed per Wh of AI, counters {r['counters']}")
    print(f"\nFull results: {RESULTS_DIR / 'demand_response.json'}")
    if args.llm == "fake" and "agent" in keys:
        print("⚠️ Fake LLM: the agent's costs are not real. Use --llm ollama.")
    return 0


def cmd_dim(args) -> int:
    from .demo import demo_assessment

    r = demo_assessment(scenario_from(args))
    r.print()
    r.tables["Sweep"].to_csv(RESULTS_DIR / "sizing.csv", index=False)
    for p in r.save("sizing"):
        print(f"📊 {p}")
    return 0


def cmd_video(args) -> int:
    from .video import make_video

    p = make_video(scenario_from(args), mode_voice=args.voice)
    print(f"🎬 Video: {p}")
    print(f"🖼️ Conclusion slide: {RESULTS_DIR / 'slide_conclusion.png'}")
    return 0


def cmd_interface(args) -> int:
    return subprocess.call([sys.executable, "-m", "streamlit", "run", str(ROOT / "app" / "Home.py")])


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="quartier", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp):
        sp.add_argument("--start", default="2024-01-15")
        sp.add_argument("--days", type=int, default=7)
        sp.add_argument("--kwc", type=float, default=120.0)
        sp.add_argument("--packs", type=int, default=None, help="number of battery packs (default 4)")
        sp.add_argument("--homes", type=int, default=60)
        sp.add_argument("--tariff", choices=["BASE", "HPHC", "TEMPO"], default="TEMPO")
        sp.add_argument("--tdp", type=float, default=28.0)
        sp.add_argument("--offline", action="store_true", help="no downloads (synthetic data)")
        sp.add_argument("--model", default="qwen2.5:1.5b")

    c = sub.add_parser("check")
    c.add_argument("--model", default="qwen2.5:1.5b")
    c.set_defaults(func=cmd_check)
    d = sub.add_parser("demo")
    d.add_argument("module", choices=["weather", "solar", "batteries", "usage", "appliances", "grid", "demand_response",
                                      "control", "aggregator", "measure", "assessment"])
    common(d)
    d.set_defaults(func=cmd_demo)
    r = sub.add_parser("run")
    common(r)
    r.add_argument("--strategies", nargs="*", help="no_battery rule tariff_rule optimizer agent_hourly agent_route")
    r.add_argument("--llm", choices=["fake", "ollama"], default="fake")
    r.add_argument("--llm-days", type=int, default=2)
    r.set_defaults(func=cmd_run)
    e = sub.add_parser("flex")
    common(e)
    e.add_argument("--strategies", nargs="*",
                   help="none cut_all round_robin prepared_round_robin optimizer agent")
    e.add_argument("--llm", choices=["fake", "ollama", "none"], default="fake",
                   help="agent LLM: fake (test), ollama (real, slow), none (no agent)")
    e.set_defaults(func=cmd_demand_response)
    dm = sub.add_parser("size")
    common(dm)
    dm.set_defaults(func=cmd_dim)
    v = sub.add_parser("video")
    common(v)
    v.add_argument("--voice", choices=["auto", "edge", "windows", "none"], default="auto",
                   help="auto: neural voice (internet), else Windows voice, else no voice")
    v.set_defaults(func=cmd_video)
    i = sub.add_parser("interface")
    i.set_defaults(func=cmd_interface)
    return p


def main(argv=None) -> int:
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(errors="replace")
        except Exception:
            pass
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    for noisy in ("httpx", "httpcore", "matplotlib"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
