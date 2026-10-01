"""
demo.py: one demonstration per module. Used both by:
  - the command line       : quartier demo <module>   (text + PNG image in results/)
  - the Streamlit interface: one page per module (same computations, same charts)

Each demo returns a DemoResult: summary sentences, charts, tables.
Each demo uses ONLY its own module (and earlier ones when essential),
so the project can be tested module by module.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .config import RESULTS_DIR, Scenario  # noqa: E402

C1, C2, C3, C4 = "#2a78d6", "#eb6834", "#1baf7a", "#898781"   # colorblind-friendly palette


@dataclass
class DemoResult:
    title: str
    lines: list = field(default_factory=list)
    figs: list = field(default_factory=list)
    tables: dict = field(default_factory=dict)

    def print(self) -> None:
        print("=" * 70 + f"\n{self.title}\n" + "=" * 70)
        for line in self.lines:
            print("•", line)
        for name, t in self.tables.items():
            print(f"\n{name}\n{t.to_string()}")

    def save(self, stem: str) -> list:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        paths = []
        for i, f in enumerate(self.figs):
            p = RESULTS_DIR / f"demo_{stem}{'' if i == 0 else f'_{i + 1}'}.png"
            f.savefig(p, dpi=130, bbox_inches="tight")
            paths.append(p)
        return paths


def _ax(title, ylabel, figsize=(10, 3.4)):
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_title(title, loc="left", fontsize=11)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    return fig, ax


def _sim_slice(idx, scn):
    t0 = pd.Timestamp(scn.start).tz_localize("Europe/Paris")
    return idx[(idx >= t0) & (idx < t0 + pd.Timedelta(days=scn.days))]


# =============================================================================
# 1. Weather
# =============================================================================
def demo_weather(scn: Scenario) -> DemoResult:
    from .weather import load_weather_pair

    obs, fc = load_weather_pair(scn.site, scn.start, scn.days, 0, scn.offline, scn.seed)
    r = DemoResult(f"Module 1 — Weather in {scn.site.name} ({scn.start}, {scn.days} days)")
    r.lines += [f"Observed source: {obs.attrs['source']}", f"Forecast source: {fc.attrs['source']}",
                f"Mean irradiance: {obs['ghi_wm2'].mean():.0f} W/m² (max {obs['ghi_wm2'].max():.0f})",
                f"Temperature: {obs['temp_c'].min():.1f} to {obs['temp_c'].max():.1f} °C",
                f"Mean error of the irradiance forecast: {np.abs(fc['ghi_wm2'] - obs['ghi_wm2']).mean():.0f} W/m²"]
    fig, ax = _ax("Horizontal irradiance: observed vs forecast the day before", "W/m²")
    ax.plot(obs.index, obs["ghi_wm2"], color=C1, lw=1.6, label="observed")
    ax.plot(fc.index, fc["ghi_wm2"], color=C2, lw=1.2, ls="--", label="forecast")
    ax.legend(frameon=False)
    r.figs.append(fig)
    r.tables["Daily summary"] = obs.resample("D").agg({"ghi_wm2": "mean", "temp_c": "mean", "cloud_pct": "mean"}).round(1)
    return r


# =============================================================================
# 2. Solar
# =============================================================================
def demo_solar(scn: Scenario) -> DemoResult:
    from .weather import load_weather_pair
    from .solar import compare_forecasts, measured_production, pv_production_kwh

    obs, fc = load_weather_pair(scn.site, scn.start, scn.days, scn.history_days, scn.offline, scn.seed)
    true = pv_production_kwh(obs, scn.site, scn.pv)
    meas = measured_production(true, seed=scn.seed)
    sim = _sim_slice(obs.index, scn)
    table, scores = compare_forecasts(obs, fc, meas, sim, scn.site, scn.pv)
    r = DemoResult(f"Module 2 — Solar: {scn.pv.kwc:.0f} kWp, tilt {scn.pv.tilt_deg:.0f}°, azimuth {scn.pv.azimuth_deg:.0f}°")
    prod = meas.loc[sim].sum()
    r.lines += [f"Measured production over the period: {prod:.0f} kWh ({prod / scn.pv.kwc / scn.days:.2f} kWh/kWp/day)" if scn.pv.kwc else "No panels",
                f"Meter vs physical model gap: {100 * (meas.loc[sim].sum() / max(true.loc[sim].sum(), 1e-9) - 1):+.1f} %",
                "Forecast error (nMAE, lower = better): " +
                ", ".join(f"{k} {100 * v:.0f} %" for k, v in scores.items())]
    fig, ax = _ax("Production: measured vs forecasts", "kWh per hour")
    ax.plot(table.index, table.iloc[:, 0], color="black", lw=1.8, label="measured")
    ax.plot(table.index, table["pv_persistence_kwh"], color=C4, lw=1, label="persistence (same as yesterday)")
    ax.plot(table.index, table["pv_physics_kwh"], color=C2, lw=1.2, ls="--", label="physics (forecast weather)")
    ax.plot(table.index, table["pv_forecast_ml_kwh"], color=C1, lw=1.4, label="AI (ML)")
    ax.legend(frameon=False, ncol=4, fontsize=8)
    r.figs.append(fig)
    r.tables["Forecast error (nMAE)"] = pd.DataFrame({"nMAE %": {k: round(100 * v, 1) for k, v in scores.items()}})
    return r


# =============================================================================
# 3. Batteries
# =============================================================================
def demo_batteries(scn: Scenario, days: int = 365, soh_true: float = 0.75, soh_guess: float = 0.80,
                   charge_kw: float = 6.0, discharge_kw: float = 5.0) -> DemoResult:
    from .batteries import BatteryBank, BatteryPack, demo_profile, estimate_soh, run_bms

    cfg = scn.battery
    cap = cfg.capacities_kwh[0] if cfg.capacities_kwh else 40.0
    pack = BatteryPack("pack_test", cap, soh_true, cfg, soc=0.5)
    idx = pd.date_range("2024-01-01", periods=24 * days, freq="h")
    log = run_bms(pack, demo_profile(days, charge_kw, discharge_kw), idx, seed=scn.seed, soh_guess=soh_guess)
    est = estimate_soh(log, cap)
    e_cc = (log["soc_counting"] - log["soc_true"]).abs().mean()
    e_est = (log["soc_estimated"] - log["soc_true"]).abs().mean()
    r = DemoResult(f"Module 3 — Second-life battery: {cap:.0f} kWh when new, actual SOH {100 * soh_true:.0f}%, "
                   f"assumed SOH {100 * soh_guess:.0f}%")
    r.lines += [f"After {days} days: {pack.efc:.0f} equivalent cycles, SOH {100 * soh_true:.1f}% → {100 * pack.soh:.1f}%",
                f"Mean state-of-charge error: coulomb counting alone {100 * e_cc:.1f} pts, "
                f"counting + rest voltage {100 * e_est:.1f} pts",
                f"SOH estimated by learning: {100 * est['soh_estimated']:.1f}% (actual {100 * pack.soh:.1f}%)"
                if not np.isnan(est["soh_estimated"]) else "Not enough points to estimate the SOH",
                (f"End of second life (SOH 60%) estimated in ~{est['cycles_before_60pct']:.0f} cycles "
                 f"(≈ {est['cycles_before_60pct'] / max(pack.efc / days * 365, 1e-9):.1f} years at this rate)")
                if np.isfinite(est.get("cycles_before_60pct", np.nan)) else ""]
    w = log.iloc[: 24 * 7]
    fig, ax = _ax("State of charge (1st week): true vs BMS estimates", "SOC")
    ax.plot(w.index, w["soc_true"], color="black", lw=1.8, label="true")
    ax.plot(w.index, w["soc_counting"], color=C2, lw=1.2, ls="--", label="counting alone")
    ax.plot(w.index, w["soc_estimated"], color=C1, lw=1.4, label="counting + voltage")
    ax.legend(frameon=False, ncol=3)
    r.figs.append(fig)
    fig2, ax2 = _ax("State of health: true vs measurements and learned trend", "SOH")
    ax2.plot(log["efc"], log["soh_true"], color="black", lw=1.8, label="true")
    if len(est["points"]):
        ax2.scatter(est["points"]["efc"], est["points"]["soh_measure"], s=6, color=C4, alpha=0.5, label="measurements")
    if not np.isnan(est["soh_estimated"]):
        x = np.array([0, log["efc"].iloc[-1]])
        ax2.plot(x, est["soh_estimated"] + est["slope_per_cycle"] * (x - x[-1]), color=C1, lw=2, label="learned trend")
    ax2.set_xlabel("equivalent full cycles")
    ax2.legend(frameon=False)
    r.figs.append(fig2)
    bank = BatteryBank(cfg)
    r.tables["Scenario battery bank"] = pd.DataFrame([{
        "pack": p.name, "new capacity kWh": p.cap_name_kwh, "SOH %": round(100 * p.soh, 1),
        "current capacity kWh": round(p.capacity_kwh, 1), "max power kW": round(p.p_max_kw, 1)} for p in bank.packs])
    return r


# =============================================================================
# 4. Usage
# =============================================================================
def demo_usage(scn: Scenario) -> DemoResult:
    from .usage import cluster_habits, compare_load_forecasts, make_building

    t0 = pd.Timestamp(scn.start).normalize() - pd.Timedelta(days=scn.history_days)
    idx = pd.date_range(t0, periods=24 * (scn.history_days + scn.days), freq="h", tz="Europe/Paris")
    loads, types = make_building(scn.building, idx, seed=scn.seed)
    total = loads.sum(axis=1)
    sim = _sim_slice(idx, scn)
    hab = cluster_habits(loads.loc[: sim[0]], types, k=4, seed=scn.seed)
    table, scores = compare_load_forecasts(total, sim)
    r = DemoResult(f"Module 4 — Usage: {scn.building.n_homes} homes")
    year = 365 / (len(idx) / 24)
    r.lines += [f"Mean consumption: {loads.sum().mean() * year:.0f} kWh/year per home (simulated)",
                f"Habits recognized without knowing the types: agreement with reality (ARI) = {hab['ari']:.2f} "
                "(1 = perfect, 0 = random)",
                "Consumption forecast error (nMAE): " + ", ".join(f"{k} {100 * v:.0f} %" for k, v in scores.items())]
    fig, ax = _ax("Daily shape of the habit groups found by the AI", "share of daily consumption")
    for i, c in enumerate(hab["centres"].columns):
        ax.plot(range(24), hab["centres"][c], lw=2, label=c, color=[C1, C2, C3, C4][i % 4])
    ax.set_xlabel("hour")
    ax.legend(frameon=False, fontsize=8)
    r.figs.append(fig)
    fig2, ax2 = _ax("Building consumption: measured vs forecasts", "kWh per hour")
    ax2.plot(table.index, table["consumption_measured_kwh"], color="black", lw=1.8, label="measured")
    ax2.plot(table.index, table["load_d-7_kwh"], color=C4, lw=1, label="same as last week")
    ax2.plot(table.index, table["consumption_forecast_ml_kwh"], color=C1, lw=1.4, label="AI (ML)")
    ax2.legend(frameon=False, ncol=3)
    r.figs.append(fig2)
    r.tables["Habits per home"] = hab["table"]
    return r


# =============================================================================
# 6. Grid
# =============================================================================
def demo_grid(scn: Scenario) -> DemoResult:
    from .grid import buy_price, load_grid, sell_price

    g = load_grid(scn.start, scn.days, 0, scn.offline, scn.seed)
    pb, ps = buy_price(g.index, scn.tariff, g["tempo"]), sell_price(g.index, scn.tariff)
    r = DemoResult(f"Module 6 — Grid: {scn.tariff.option} tariff")
    r.lines += [f"CO2 source: {g.attrs['source']}",
                f"Grid CO2: {g['co2_g_per_kwh'].min():.0f} to {g['co2_g_per_kwh'].max():.0f} g/kWh (mean {g['co2_g_per_kwh'].mean():.0f})",
                f"Purchase price: {pb.min():.2f} to {pb.max():.2f} €/kWh; surplus resale: {ps.iloc[0]:.2f} €/kWh",
                f"Tempo days (approximate): {dict(g['tempo'].groupby(g.index.date).first().value_counts())}"]
    fig, ax = _ax("Purchase price per kWh", "€/kWh")
    ax.step(pb.index, pb, where="post", color=C1, lw=1.6, label="purchase")
    ax.step(ps.index, ps, where="post", color=C3, lw=1.2, label="resale")
    ax.legend(frameon=False)
    r.figs.append(fig)
    fig2, ax2 = _ax("Grid carbon intensity (RTE)", "g CO2/kWh")
    ax2.plot(g.index, g["co2_g_per_kwh"], color=C2, lw=1.4)
    r.figs.append(fig2)
    return r


# =============================================================================
# 8. Battery control
# =============================================================================
def demo_control(scn: Scenario, keys=("no_battery", "rule", "tariff_rule", "optimizer"), show="optimizer",
                  ctx=None) -> DemoResult:
    from .assessment import load_inputs, simulate
    from .assessment.kpi import kpis
    from .measure import Meter
    from .control import make_controller

    ctx = ctx or load_inputs(scn)
    rows, sims = [], {}
    for k in keys:
        c = make_controller(k)
        m = Meter(scn.compute, watch=())
        c.setup(ctx, m)
        s, bank = simulate(ctx, c, m)
        kp = kpis(s, ctx, bank, m.report())
        sims[k] = s
        rows.append({"strategy": c.label, "self-sufficiency %": round(100 * kp["self_sufficiency"], 1),
                     "purchase kWh": round(kp["purchase_kwh"], 1), "resale kWh": round(kp["resale_kwh"], 1),
                     "bill €": round(kp["bill_eur"], 2), "battery wear €": round(kp["wear_eur"], 2),
                     "AI (Wh)": round(1000 * kp["ai_kwh"], 3)})
    r = DemoResult(f"Module 8 — Battery control ({scn.days} days, {scn.tariff.option} tariff)")
    t = pd.DataFrame(rows)
    best = t.loc[(t["bill €"] + t["battery wear €"]).idxmin(), "strategy"]
    r.lines += [f"Best bill (wear included): {best}", "Sources: " + "; ".join(f"{k} = {v}" for k, v in ctx.sources().items())]
    r.tables["Strategy comparison"] = t
    s = sims.get(show, next(iter(sims.values())))
    d = s.iloc[:72]
    fig, ax = _ax(f"First 3 days — {show}", "kWh per hour", figsize=(10, 3.8))
    ax.plot(d.index, d["pv_kwh"], color=C3, lw=1.6, label="production")
    ax.plot(d.index, d["consumption_kwh"], color="black", lw=1.4, label="consumption")
    ax.bar(d.index, d["battery_kw"], width=0.035, color=C1, alpha=0.6, label="battery (+ charge / − discharge)")
    ax.plot(d.index, d["purchase_kwh"], color=C2, lw=1.2, ls="--", label="grid purchase")
    ax.legend(frameon=False, ncol=4, fontsize=8)
    r.figs.append(fig)
    if any(sk["soc"].notna().any() for sk in sims.values()):
        fig2, ax2 = _ax("Battery state of charge", "SOC")
        for k, sk in sims.items():
            if sk["soc"].notna().any():
                ax2.plot(sk.index, sk["soc"], lw=1.3, label=k)
        ax2.legend(frameon=False, fontsize=8)
        r.figs.append(fig2)
    return r


# =============================================================================
# 10. Measurement
# =============================================================================
def demo_measure(scn: Scenario, ctx=None) -> DemoResult:
    from .agent.llm import FakeLLM
    from .assessment import load_inputs
    from .measure import Meter
    from .control import ai_forecasts
    from .control.optimizer import solve_plan

    ctx = ctx or load_inputs(scn)
    m = Meter(scn.compute, watch=())
    ai_forecasts(ctx, m)
    with m.measure("shared"):
        for _ in range(24):
            solve_plan(np.full(24, 5.0), np.full(24, 4.0), np.full(24, 0.2), np.full(24, 0.04), 20, 5, 50, 10, 10,
                       0.95, 0.95, 0.03, 0.15)
    rep = m.report()
    r = DemoResult("Module 10 — Measuring the AI's compute cost (on this PC)")
    r.lines += [f"Training the 2 ML models (once): {1000 * rep['setup']['energy_kwh']:.3f} Wh, "
                f"{rep['setup']['wall_s']:.2f} s",
                f"Forecasts + 24 optimization plans (1 day): {1000 * rep['shared']['energy_kwh']:.3f} Wh",
                f"Assumed TDP: {scn.compute.cpu_tdp_w} W. For an LLM, use Ollama: whole-machine measurement is enabled.",
                "Order of magnitude measured with Bilan Net: 1 call to Qwen 2.5 1.5B on this PC ≈ 0.07 Wh."]
    r.tables["Measurement per bucket"] = pd.DataFrame(rep).T[["wall_s", "cpu_s_self", "energy_kwh"]]
    _ = FakeLLM
    return r


# =============================================================================
# 11. Assessment + sizing
# =============================================================================
def demo_assessment(scn: Scenario, kwc_list=(0, 15, 30, 45, 60), packs_list=(0, 1, 2, 3, 4), ctx=None) -> DemoResult:
    from .assessment import load_inputs, sweep

    ctx = ctx or load_inputs(scn)
    t = sweep(ctx, kwc_list, packs_list)
    r = DemoResult(f"Module 11 — Sizing ({scn.building.n_homes} homes, self-consumption rule, "
                   f"extrapolated to a year from {scn.days} days)")
    ok = t[np.isfinite(t["payback_years"])]
    if len(ok):
        b = ok.loc[ok["payback_years"].idxmin()]
        r.lines.append(f"Best payback time: {b['kwc']:.0f} kWp + {int(b['packs'])} pack(s) "
                       f"→ {b['payback_years']:.1f} years, self-sufficiency {b['self_sufficiency_pct']:.0f}%")
    r.lines.append("⚠️ Extrapolated from a short period: rerun in winter and in summer before drawing conclusions.")
    pv = t.pivot(index="packs", columns="kwc", values="self_sufficiency_pct")
    fig, ax = plt.subplots(figsize=(7, 3.6))
    im = ax.imshow(pv.to_numpy(), cmap="Blues", aspect="auto", origin="lower")
    ax.set_xticks(range(len(pv.columns)), [f"{c:.0f}" for c in pv.columns])
    ax.set_yticks(range(len(pv.index)), [str(i) for i in pv.index])
    ax.set_xlabel("panels (kWp)")
    ax.set_ylabel("battery packs")
    ax.set_title("Self-sufficiency rate (%)", loc="left", fontsize=11)
    for i in range(pv.shape[0]):
        for j in range(pv.shape[1]):
            v = pv.to_numpy()[i, j]
            ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=9, color="white" if v > 50 else "black")
    fig.colorbar(im, ax=ax)
    r.figs.append(fig)
    r.tables["Sweep"] = t.round(1)
    return r


# =============================================================================
# 5. Connected appliances
# =============================================================================
def demo_appliances(scn: Scenario, hour_cut: int = 18, home: str | None = None) -> DemoResult:
    from .appliances import (NAMES_APPLIANCES, HomeState, learn_thermal, setpoints, error_learning, make_portfolio,
                            weather_at_step, time_step, simulate_without_shedding)
    from .demand_response import forecast_consumption, hourly_profiles
    from .weather import load_weather_pair

    hist_days = scn.history_days
    obs, fc = load_weather_pair(scn.site, scn.start, scn.days, hist_days, scn.offline, scn.seed)
    portfolio = make_portfolio(scn.building, scn.appliances, obs.index, seed=scn.seed, step_min=scn.demand_response.step_min)
    idx = portfolio.base_kw.index
    t0 = pd.Timestamp(scn.start).tz_localize("Europe/Paris")
    h_all = simulate_without_shedding(portfolio, obs, idx, seed=scn.seed)
    cut = int((idx < t0).sum())
    hist = {**h_all, "index": idx[:cut], "t_ext": h_all["t_ext"][:cut], "ghi": h_all["ghi"][:cut],
            "presence": h_all["presence"][:cut], "t_in_measure": h_all["t_in_measure"][:cut + 1],
            "p": {a: v[:cut] for a, v in h_all["p"].items()}}
    model = learn_thermal(portfolio, hist)
    profiles = hourly_profiles(hist)
    sim = idx[cut:]
    prev = forecast_consumption(portfolio, model, h_all["t_in_measure"][cut], sim, weather_at_step(fc, sim), profiles)
    true = h_all["p"]["heating"][cut:]
    sel = portfolio.heating
    peak = np.isin(sim.hour, [18, 19])
    err_log = np.abs(prev["heating"] - true)[peak][:, sel].sum() / max(true[peak][:, sel].sum(), 1e-9)
    q_prev, q_true = prev["heating"][:, sel].sum(1), true[:, sel].sum(1)
    err_q = np.abs(q_prev - q_true)[peak].sum() / max(q_true[peak].sum(), 1e-9)
    e = error_learning(portfolio, model)
    err_c = float((100 * (e["C learned (kWh/°C)"] / e["C true (kWh/°C)"] - 1).abs()).median()) if len(e) else np.nan

    # Experiment: cut ONE heater for 30 min and watch the temperature and the catch-up
    i = (list(portfolio.names).index(home) if home in portfolio.names else
         int(np.argmax(portfolio.heating & (portfolio.types == "family"))))
    day = sim[sim.dayofweek < 5][0].normalize() if (sim.dayofweek < 5).any() else sim[0].normalize()
    win = pd.date_range(day + pd.Timedelta(hours=hour_cut - 2), periods=4 * 6, freq=f"{scn.demand_response.step_min}min")
    k0 = int(np.searchsorted(idx, win[0]))
    sp, pres = setpoints(portfolio, win)
    w = weather_at_step(obs, win)
    curves = {}
    for case_name, cut_h in (("no cut", None), ("cut 30 min", hour_cut)):
        hs = HomeState.initial(portfolio)
        hs.t_in = h_all["t_in_true"][k0].copy()
        T, P = [], []
        for k, t in enumerate(win):
            c = np.zeros(portfolio.n, bool)
            if cut_h is not None and t.hour == cut_h and t.minute < 30:
                c[i] = True
            out = time_step(portfolio, hs, float(w["temp_c"].iloc[k]), float(w["ghi_wm2"].iloc[k]), sp[k], pres[k],
                               t.hour, t.minute, t.hour in scn.tariff.hc_hours, t.dayofweek >= 5, {"heating": c}, 0.25)
            T.append(hs.t_in[i])
            P.append(out["heating"][i] if portfolio.heating[i] else 0.0)
        curves[case_name] = (np.array(T), np.array(P))
    drop = float((curves["no cut"][0] - curves["cut 30 min"][0]).max())
    gap = (curves["cut 30 min"][1] - curves["no cut"][1]) * 0.25
    shed = -gap[gap < 0].sum()
    catches_up = gap[gap > 0].sum()

    r = DemoResult(f"Module 5 — Connected appliances: {portfolio.n} homes in the district")
    r.lines += [
        f"Connected homes (controllable box): {int(portfolio.connected.sum())} / {portfolio.n}; "
        f"electric heating: {int(portfolio.heating.sum())}, water heater: {int(portfolio.water_heater.sum())}, EV: {int(portfolio.ev.sum())}",
        f"Mean consumption per home per day: non-sheddable uses {h_all['base_kw'][cut:].sum() * 0.25 / portfolio.n / scn.days:.1f} kWh, "
        + ", ".join(f"{NAMES_APPLIANCES[a]} {h_all['p'][a][cut:].sum() * 0.25 / portfolio.n / scn.days:.1f} kWh" for a in NAMES_APPLIANCES),
        f"The AI learned each home's inertia to within ±{err_c:.0f}% (median) from {hist_days} days of measurements",
        f"Heating forecast for 6-8 pm ({scn.days} days ahead): error {100 * err_log:.0f}% per home, "
        f"{100 * err_q:.0f}% for the whole district (errors cancel out: this is DIVERSITY / aggregation)",
        f"Experiment ({portfolio.names[i]}, {portfolio.types[i]}): heater cut for 30 min at {hour_cut}:00 -> the temperature drops by "
        f"{drop:.2f} °C; {shed:.2f} kWh shed, then {catches_up:.2f} kWh of CATCH-UP: the energy is mostly SHIFTED",
    ]
    df = pd.DataFrame({"non-sheddable uses": h_all["base_kw"][cut:].sum(1),
                       **{NAMES_APPLIANCES[a]: h_all["p"][a][cut:].sum(1) for a in NAMES_APPLIANCES}}, index=sim)
    j = df.loc[df.index.normalize() == day]
    days_fr = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    fig, ax = _ax(f"District consumption by use — {days_fr[day.dayofweek]} {day:%d/%m}", "kW", figsize=(10, 3.6))
    ax.stackplot(j.index, j.T.to_numpy(), labels=j.columns, colors=[C4, C2, C1, C3], alpha=0.85)
    ax.legend(frameon=False, ncol=4, fontsize=8, loc="upper left")
    r.figs.append(fig)
    fig2, ax2 = _ax("District heating: actual vs forecast by the learned model", "kW")
    ax2.plot(sim, q_true, color="black", lw=1.5, label="actual")
    ax2.plot(sim, q_prev, color=C1, lw=1.2, ls="--", label="forecast (learned thermal model)")
    ax2.legend(frameon=False)
    r.figs.append(fig2)
    fig3, (a1, a2) = plt.subplots(2, 1, figsize=(10, 4.6), sharex=True)
    for case_name, col in (("no cut", C4), ("cut 30 min", C2)):
        a1.plot(win, curves[case_name][0], color=col, lw=1.8, label=case_name)
        a2.step(win, curves[case_name][1], where="post", color=col, lw=1.8, label=case_name)
    a1.plot(win, sp[:, i], color="black", lw=1, ls=":", label="setpoint")
    a1.set_ylabel("°C")
    a2.set_ylabel("kW")
    a1.set_title(f"Cutting a heater for 30 min ({portfolio.names[i]}): temperature drop, then catch-up", loc="left", fontsize=11)
    a1.legend(frameon=False, fontsize=8)
    for a in (a1, a2):
        a.grid(alpha=0.3)
    r.figs.append(fig3)
    r.tables["Who has what (connected homes by occupant type × appliance)"] = portfolio.table()
    r.tables["Learning: learned vs true parameters (excerpt)"] = e.head(10).set_index("home")
    return r


# =============================================================================
# 7. Demand response: RTE requests and declared flexibility
# =============================================================================
def demo_demand_response(scn: Scenario, ctx=None) -> DemoResult:
    from .aggregator import load_district, set_volumes, make_strategy, run_simulation
    from .measure import Meter

    ctx = ctx or load_district(scn)
    m = Meter(scn.compute, watch=())
    st = make_strategy("none")
    st.setup(ctx, m)
    ref = run_simulation(ctx, st, m, seed=scn.seed)
    reports = set_volumes(ctx, ref)
    g = ctx.inputs.grid.loc[ctx.inputs.sim_index]
    r = DemoResult(f"Module 7 — Demand response: what RTE requests, what the district declares ({scn.start}, {scn.days} days)")
    r.lines += [f"Grid source: {ctx.inputs.grid.attrs.get('source')}",
                f"Tempo days (approximate): {dict(g['tempo'].groupby(g.index.date).first().value_counts())}",
                f"RTE requests: {len(ctx.requests)} " + (f"({ctx.note})" if ctx.note else ""),
                "RTE deals with the DISTRICT (via the aggregator), not with each home: a single home is "
                "unpredictable, a district much less so (diversity / aggregation, see module 5)",
                "Declared batteries: SOC, SOH, actual capacity and power (reduced if worn) -> available energy"]
    if ctx.requests:
        r.tables["RTE requests"] = pd.DataFrame([{
            "slot": f"{d.start:%a %d/%m %Hh}-{d.end:%Hh}", "level": d.level, "announced": f"{d.announce:%d/%m %Hh}",
            "declared flex kW": round(rep["total_kw"], 1), "of which batteries kW": round(rep["batteries"]["kw_held_over_slot"], 1),
            "requested volume kW": d.volume_kw} for d, rep in zip(ctx.requests, reports)])
        rep = reports[0]
        r.tables[f"Declared flexibility for {ctx.requests[0].start:%d/%m %Hh} (per sheddable group)"] = rep["groups"]
        r.tables["Battery status reported to RTE"] = pd.DataFrame(rep["batteries"]["packs"])
        fig, ax = _ax("Declared flexibility per request (kW)", "kW")
        labels = [f"{d.start:%d/%m %Hh}" for d in ctx.requests]
        bottom = np.zeros(len(labels))
        for a, col in (("heating", C2), ("water_heater", C1), ("ev", C3)):
            v = np.array([rp["groups"].loc[rp["groups"]["appliance"] == a, "kW sheddable"].sum() for rp in reports])
            ax.bar(labels, v, bottom=bottom, color=col, label={"water_heater": "water_heaters", "ev": "cars"}.get(a, a))
            bottom += v
        vb = np.array([rp["batteries"]["kw_held_over_slot"] for rp in reports])
        ax.bar(labels, vb, bottom=bottom, color=C4, label="batteries")
        ax.plot(labels, [d.volume_kw for d in ctx.requests], "k_", markersize=30, mew=2, label="requested volume")
        ax.legend(frameon=False, fontsize=8, ncol=5)
        r.figs.append(fig)
    fig2, ax2 = _ax("French national consumption (RTE) and stress days", "GW")
    ax2.plot(g.index, g["consumption_mw"] / 1000, color="black", lw=1.3)
    for d in ctx.requests:
        ax2.axvspan(d.start, d.end, color=C2 if d.level == "RED" else C1, alpha=0.3)
    r.figs.append(fig2)
    return r


# =============================================================================
# 9. Aggregator
# =============================================================================
def demo_aggregator(scn: Scenario, keys=("none", "cut_all", "round_robin", "prepared_round_robin", "optimizer"),
                    llm_kind: str = "fake", llm_model: str = "qwen2.5:1.5b", ctx=None, out=None) -> DemoResult:
    from .aggregator import run_demand_response, table

    # save=False: do not overwrite results/demand_response.json (the one from "quartier flex", LLM agent included)
    out = out or run_demand_response(scn, list(keys), llm_kind=llm_kind, llm_model=llm_model, ctx=ctx, log=lambda *a: None,
                                save=False)
    ctx = out["_ctx"]
    t = table(out)
    r = DemoResult(f"Module 9 — Aggregator: allocating the RTE request across {ctx.portfolio.n} homes")
    if not ctx.requests:
        r.lines.append("No RTE request over the period.")
        return r
    ai = [n for n, v in out["results"].items() if v.get("assessment_ai")]
    r.lines += [f"{len(ctx.requests)} request(s); mean volume {np.mean([d.volume_kw for d in ctx.requests]):.0f} kW. {ctx.note}"]
    for n in ai:
        v = out["results"][n]
        b = v["assessment_ai"]
        r.lines.append(f"{v['label']} vs {v['reference_ai']}: NET gain {b['gain_net_eur']:+.2f} €, "
                       f"comfort {b['comfort_wins_degh']:+.1f} °C·h, AI {1e6 * v['kpi']['ai_kwh']:.2f} mWh "
                       f"({b['kwh_shed_per_wh_ai']:,.0f} kWh shed per Wh of AI)".replace(",", "\u202f"))
    r.tables["Strategy comparison"] = t.T
    r.tables["Per request (best AI strategy)" if ai else "Per request"] = pd.DataFrame(
        out["results"][ai[0] if ai else list(out["results"])[-1]]["kpi"]["per_request"])
    d = ctx.requests[0]
    win = (d.start - pd.Timedelta("3h"), d.end + pd.Timedelta("3h"))
    fig, ax = _ax(f"District grid draw around the request of {d.start:%d/%m %Hh}", "kW", figsize=(10, 4))
    cols = {"0_none": "black", "1_cut_all": C2, "2_round_robin": C4, "2b_prepared_round_robin": "#9467bd",
            "3_optimizer_ai": C1, "4_agent_llm": C3}
    for n, sdict in out["series"].items():
        s = pd.DataFrame(sdict).set_index("time")
        s.index = pd.to_datetime(s.index, utc=True).tz_convert("Europe/Paris")
        w = s.loc[win[0]:win[1]]
        ax.plot(w.index, w["grid_draw_kw"], lw=2 if n == "0_none" else 1.4, color=cols.get(n),
                label=out["results"][n]["label"])
        if n == "0_none":
            ax.plot(w.index, w["grid_draw_kw"] - w["request_kw"], color="black", lw=1, ls=":", label="target (reference − volume)")
    ax.axvspan(d.start, d.end, color=C2, alpha=0.1)
    ax.legend(frameon=False, fontsize=7, ncol=2)
    r.figs.append(fig)
    return r


DEMOS = {"weather": demo_weather, "solar": demo_solar, "batteries": demo_batteries, "usage": demo_usage,
         "appliances": demo_appliances, "grid": demo_grid, "demand_response": demo_demand_response, "control": demo_control,
         "aggregator": demo_aggregator, "measure": demo_measure, "assessment": demo_assessment}
