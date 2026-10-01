"""
video.py: AUTOMATICALLY builds the demo video (and the conclusion slide).

    quartier video                  # Microsoft neural voice (internet), else Windows voice, else no voice
    quartier video --voice windows  # installed Windows voice (offline)
    quartier video --voice none     # no voice (captions only)

Steps:
1. reruns the demos (appliances, demand response, aggregator) to get the charts;
2. reads the assessment from the last "quartier flex" command (results/demand_response.json), LLM agent included;
3. draws 7 images at 1920×1080 (title, chart, caption);
4. has the narration read by a text-to-speech voice;
5. assembles everything with ffmpeg -> results/quartier_flex.mp4 (+ results/slide_conclusion.png).
"""

from __future__ import annotations

import asyncio
import io
import json
import logging
import subprocess
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .config import RESULTS_DIR, Scenario  # noqa: E402

log = logging.getLogger(__name__)
BLUE, ORANGE, GREEN, GREY, BLACK = "#2a78d6", "#eb6834", "#1baf7a", "#6b6a66", "#1d1d1b"
W, H, DPI = 16, 9, 120   # 1920 × 1080


# =============================================================================
# 1. The figures
# =============================================================================
def figures(scn: Scenario) -> dict:
    """Assessment figures: from results/demand_response.json (last run), otherwise recomputed (without LLM)."""
    p = RESULTS_DIR / "demand_response.json"
    if p.exists():
        out = json.loads(p.read_text(encoding="utf-8"))
    else:
        from .aggregator import run_demand_response

        out = run_demand_response(scn, ["none", "cut_all", "round_robin", "prepared_round_robin", "optimizer"],
                             log=lambda *a: None)
    R = out["results"]
    opt = R.get("3_optimizer_ai", {})
    ag = R.get("4_agent_llm")
    ref_label = opt.get("reference_ai", "")
    ref = next((r for r in R.values() if r["label"] == ref_label), None)
    c = {"n_requests": len(out["requests"]), "homes": out["meta"]["scenario"]["homes"],
         "gain_opt": opt.get("assessment_ai", {}).get("gain_net_eur", np.nan),
         "ai_opt_mwh": 1e6 * opt.get("kpi", {}).get("ai_kwh", np.nan),
         "rebound_opt": opt.get("kpi", {}).get("rebound_kwh", np.nan),
         "rebound_ref": ref["kpi"]["rebound_kwh"] if ref else np.nan,
         "discomfort_ref": ref["kpi"]["discomfort_added_degh"] if ref else np.nan,
         "discomfort_opt": opt.get("kpi", {}).get("discomfort_added_degh", np.nan),
         "achieved_opt": 100 * opt.get("kpi", {}).get("fulfilment", np.nan),
         "agent": ag is not None and out["meta"].get("llm") == "ollama"}
    if ag is not None:
        c.update({"gain_ag": ag["assessment_ai"]["gain_net_eur"], "ai_ag_mwh": 1e6 * ag["kpi"]["ai_kwh"],
                  "calls": ag["kpi"]["llm_calls"]})
        c["ratio"] = c["ai_ag_mwh"] / max(c["ai_opt_mwh"], 1e-9)
    return c


def _rebound_division(c: dict) -> tuple[str, str]:
    """(short text for the card, text for the voice): 'divided by 6.5', or 'almost eliminated' if ~0."""
    ref, opt = c["rebound_ref"], c["rebound_opt"]
    if not np.isfinite(ref) or ref <= 0:
        return "–", "limits the rebound"
    if opt < 0.05 * ref:
        return "≈ 0", "eliminates almost all of the rebound"
    r = f"{ref / opt:.1f}"
    return f"÷ {r}", f"divides the rebound by {r}"


def _nb(x: float) -> str:
    """Rounded number, readable by a text-to-speech voice (comma as thousands separator)."""
    return f"{x:,.0f}"


# =============================================================================
# 2. The narration (written to be READ by a voice: no symbols)
# =============================================================================
def narration(c: dict) -> list[str]:
    t = [
        f"This is Quartier Flex. A district of {c['homes']} homes, solar panels, reused electric car "
        "batteries, and an artificial intelligence that answers demand response requests "
        "from the power grid.",
        "We cut a heater for thirty minutes. The temperature drops by less than one degree, then the heater "
        "catches up. The energy is shifted, not removed. That is the rebound effect, and that is the whole problem.",
        "Each home combines an occupant type, family, retirees, students, and its connected appliances: "
        "heating, water heater, electric car. These are our sheddable groups.",
        f"January twenty twenty-four, with real data from the national grid: {c['n_requests']} demand response "
        "requests in one week. The day before, the district declares what it can shed, and the state of its "
        "batteries. The grid operator sets its request based on that.",
        "In black, the district without demand response. The simple rules cut at the right time, but everything "
        "restarts at once at the end: a new peak. The optimizer, in blue, holds the request, with no rebound and "
        "nobody getting cold.",
    ]
    assessment = (f"The verdict. The optimizer earns {_nb(c['gain_opt'])} euros more in one week, "
             f"{_rebound_division(c)[1]}, "
             "for less than one thousandth of a watt-hour of compute.")
    if c.get("agent"):
        assessment += (f" The large language model agent uses {_nb(c['ratio'])} times more energy, "
                  "for a slightly worse result.")
    t.append(assessment)
    t.append("The lesson: the most sober artificial intelligence is the most effective. The language model only "
             "decides once, the day before, and every one of its answers is checked before being applied.")
    return t


# =============================================================================
# 3. The images
# =============================================================================
def _page(title: str, subtitle: str | None):
    fig = plt.figure(figsize=(W, H), dpi=DPI, facecolor="white")
    fig.text(0.04, 0.93, title, fontsize=30, weight="bold", color=BLACK, va="center")
    fig.add_artist(plt.Line2D([0.04, 0.96], [0.885, 0.885], color=BLUE, lw=3))
    fig.text(0.96, 0.03, "Quartier Flex · Aclimakathon 2026 · github.com/corinne3/quartier-flex-en", fontsize=11,
             color=GREY, ha="right")
    if subtitle:
        fig.patches.append(plt.Rectangle((0, 0.07), 1, 0.13, transform=fig.transFigure, color="#f2f1ee", zorder=-1))
        fig.text(0.5, 0.135, "\n".join(textwrap.wrap(subtitle, 125)), fontsize=17, ha="center", va="center",
                 color=BLACK)
    return fig


def _image(fig, src_fig, box=(0.06, 0.22, 0.88, 0.64)):
    buf = io.BytesIO()
    src_fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    ax = fig.add_axes(box)
    ax.imshow(plt.imread(buf))
    ax.axis("off")


def _table(fig, df, box=(0.2, 0.25, 0.72, 0.58)):
    ax = fig.add_axes(box)
    ax.axis("off")
    tb = ax.table(cellText=df.values, colLabels=list(df.columns), rowLabels=list(df.index), loc="center",
                  cellLoc="center")
    tb.auto_set_font_size(False)
    tb.set_fontsize(16)
    tb.scale(1, 2.6)
    for (r, col), cell in tb.get_celld().items():
        cell.set_edgecolor("#dddddd")
        if r == 0 or col == -1:
            cell.set_text_props(weight="bold")
            cell.set_facecolor("#eef4fb")


def _cards(fig, cards, y=0.5, h=0.3):
    n = len(cards)
    for i, (large, small, col) in enumerate(cards):
        x = 0.06 + i * (0.88 / n)
        fig.patches.append(plt.Rectangle((x + 0.01, y - h / 2), 0.88 / n - 0.02, h, transform=fig.transFigure,
                                         color=col, alpha=0.12, zorder=-1))
        fig.text(x + 0.44 / n, y + 0.05, large, fontsize=44 if len(large) < 9 else 34, weight="bold", color=col, ha="center", va="center")
        fig.text(x + 0.44 / n, y - 0.07, "\n".join(textwrap.wrap(small, 34)), fontsize=16, color=BLACK, ha="center",
                 va="center")


def slide_conclusion(c: dict, subtitle: str | None = None):
    fig = _page("The sober AI wins. The chatty AI costs.", subtitle)
    cards = [(f"+{_nb(c['gain_opt'])} €", f"net gain in one week ({c['homes']} homes), "
                                            "AI energy deducted", GREEN),
              (_rebound_division(c)[0],
               f"rebound ({_nb(c['rebound_ref'])} → {_nb(c['rebound_opt'])} kWh), and nobody gets colder", BLUE)]
    if c.get("agent"):
        cards.append((f"× {_nb(c['ratio'])}", f"energy for the LLM agent ({_nb(c['ai_ag_mwh'])} mWh vs "
                                               f"{c['ai_opt_mwh']:.1f}" + "), for a slightly worse result", ORANGE))
    else:
        cards.append((f"{c['ai_opt_mwh']:.1f} mWh", "of compute energy for the AI optimizer", ORANGE))
    _cards(fig, cards, y=0.58 if subtitle else 0.55)
    fig.text(0.5, 0.33 if subtitle else 0.27,
             "To control flexibility: forecasting + optimization, day to day.\n"
             "The LLM only where a decision is needed, once, the day before.",
             fontsize=20, ha="center", va="center", color=BLACK, style="italic")
    if not subtitle:
        fig.text(0.5, 0.12, "Real data from RTE éCO2mix and Open-Meteo · simulated homes and appliances · "
                            "AI measured on a laptop without a GPU", fontsize=13, ha="center", color=GREY)
    return fig


def images(scn: Scenario, c: dict, texts: list[str]) -> list:
    from .demo import demo_aggregator, demo_appliances, demo_demand_response

    log.info("Demos (charts)...")
    app = demo_appliances(scn)
    eff = demo_demand_response(scn)
    agr = demo_aggregator(scn)
    figs = []
    f = _page("Quartier Flex", texts[0])
    f.text(0.5, 0.66, "Demand response in a residential district", fontsize=34, ha="center", color=BLACK)
    f.text(0.5, 0.55, "connected appliances · second-life car batteries · solar", fontsize=22,
           ha="center", color=GREY)
    f.text(0.5, 0.40, "Does the controlling AI earn more than it consumes?", fontsize=26, ha="center",
           color=BLUE, weight="bold")
    figs.append(f)
    f = _page("Cutting a heater for 30 min", texts[1])
    _image(f, app.figs[2])
    figs.append(f)
    f = _page("Sheddable groups: occupant type × appliance", texts[2])
    t = next(iter(app.tables.values())).rename(columns={"EV charging": "EV"})
    t.index = [i.replace("_", " ") for i in t.index]
    _table(f, t)
    figs.append(f)
    f = _page("What RTE requests, what the district declares", texts[3])
    _image(f, eff.figs[0] if eff.figs else app.figs[0])
    figs.append(f)
    f = _page("The aggregator: holding the request without rebound", texts[4])
    _image(f, agr.figs[0])
    figs.append(f)
    f = slide_conclusion(c, texts[5])
    f.texts[0].set_text("The verdict")
    figs.append(f)
    f = _page("The lesson", texts[6])
    f.text(0.5, 0.62, "The most sober AI is the most effective.", fontsize=40, weight="bold", ha="center", color=GREEN)
    f.text(0.5, 0.45, "Forecasting + optimization every 15 minutes.\nThe LLM: one decision the day before, "
                      "checked before being applied.", fontsize=24, ha="center", color=BLACK)
    figs.append(f)
    return figs


# =============================================================================
# 4. The voice
# =============================================================================
def _voice_edge(text: str, path: Path, voice: str = "en-US-AriaNeural") -> bool:
    try:
        import edge_tts

        asyncio.run(edge_tts.Communicate(text, voice).save(str(path)))
        return path.exists() and path.stat().st_size > 1000
    except Exception as e:  # no internet, package missing...
        log.warning("Neural voice unavailable (%s)", e)
        return False


def _voice_windows(text: str, path: Path) -> bool:
    try:
        import pyttsx3

        eng = pyttsx3.init()
        for v in eng.getProperty("voices"):
            key = (v.id + " " + v.name).lower()
            if any(k in key for k in ("en-us", "en_us", "en-gb", "en_gb", "english", "zira", "david")):
                eng.setProperty("voice", v.id)
                break
        eng.setProperty("rate", 175)
        eng.save_to_file(text, str(path))
        eng.runAndWait()
        return path.exists() and path.stat().st_size > 1000
    except Exception as e:
        log.warning("Windows voice unavailable (%s)", e)
        return False


def voice(text: str, path_base: Path, mode: str) -> Path | None:
    if mode in ("auto", "edge") and _voice_edge(text, path_base.with_suffix(".mp3")):
        return path_base.with_suffix(".mp3")
    if mode in ("auto", "windows") and _voice_windows(text, path_base.with_suffix(".wav")):
        return path_base.with_suffix(".wav")
    return None


# =============================================================================
# 5. Assembly (ffmpeg)
# =============================================================================
def _ffmpeg() -> str:
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        return "ffmpeg"


def make_video(scn: Scenario, mode_voice: str = "auto", output: Path | None = None) -> Path:
    tmp = RESULTS_DIR / "video_tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    output = output or RESULTS_DIR / "quartier_flex.mp4"
    c = figures(scn)
    texts = narration(c)
    slide_conclusion(c).savefig(RESULTS_DIR / "slide_conclusion.png", dpi=DPI)
    figs = images(scn, c, texts)
    ff = _ffmpeg()
    segments = []
    for i, (fig, txt) in enumerate(zip(figs, texts)):
        img = tmp / f"s{i}.png"
        fig.savefig(img, dpi=DPI)
        plt.close(fig)
        audio = voice(txt, tmp / f"s{i}", mode_voice) if mode_voice != "none" else None
        seg = tmp / f"s{i}.mp4"
        cmd = [ff, "-y", "-loglevel", "error", "-loop", "1", "-framerate", "25", "-i", str(img)]
        if audio is not None:
            cmd += ["-i", str(audio), "-af", "apad=pad_dur=1.0", "-shortest"]
        else:   # no voice: duration estimated from reading pace (~2.5 words/s)
            dur = len(txt.split()) / 2.5 + 1.0
            cmd += ["-f", "lavfi", "-i", "anullsrc=r=44100:cl=mono", "-t", f"{dur:.1f}"]
        cmd += ["-c:v", "libx264", "-tune", "stillimage", "-pix_fmt", "yuv420p", "-r", "25",
                "-c:a", "aac", "-ar", "44100", "-ac", "1", str(seg)]
        subprocess.run(cmd, check=True)
        segments.append(seg)
        log.info("Segment %d/%d ready%s", i + 1, len(figs), "" if audio else " (no voice)")
    list_file = tmp / "list.txt"
    list_file.write_text("".join(f"file '{s.name}'\n" for s in segments), encoding="utf-8")
    subprocess.run([ff, "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(list_file), "-c", "copy",
                    str(output)], check=True, cwd=tmp)
    return output
