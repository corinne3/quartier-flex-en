"""
llm.py: talking to an LLM.

Two implementations of the same interface `chat(messages, tools, json_mode)`:

1. OllamaClient: a real open-source LLM running ON YOUR PC (CPU).
   Ollama is a local server (http://localhost:11434) that loads a
   quantized (compressed) model and runs it. No credits, no data sent
   over the internet, and above all: we can MEASURE its consumption (it runs on
   our machine, in a process we monitor).

2. FakeLLM: a deterministic fake LLM (a few rules). It is used to:
   - test all the plumbing without Ollama (tests, GitHub CI);
   - develop fast (no 5 s wait per call).
   ⚠️ Its energy measurements are WORTHLESS: never publish
   "fake" results.

Recommended models on CPU (download them BEFORE the hackathon):
   ollama pull qwen2.5:1.5b     (~1 GB, fast, decent tool calling)
   ollama pull qwen2.5:3b       (~2 GB, better reasoning, 2x slower)
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ToolCall:
    name: str
    arguments: dict


@dataclass
class LLMResponse:
    content: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    tokens_in: int = 0      # prompt tokens (what we send)
    tokens_out: int = 0     # generated tokens (what the model writes)
    duration_s: float = 0.0 # wait time seen by Python
    active_s: float = 0.0   # compute time reported by Ollama (prompt reading + generation)


# =============================================================================
# Ollama (real local LLM)
# =============================================================================
class OllamaClient:
    """
    Minimal client for the Ollama /api/chat API (documented at github.com/ollama/ollama,
    file docs/api.md).

    Parameters that matter for SOBRIETY:
    - temperature=0     : stable (reproducible) answers.
    - num_ctx           : max context size (tokens). Smaller = less memory,
                          faster. 4096: enough for the "naive" prompt (~2000 tokens).
    - num_predict       : max number of generated tokens. We cap it to keep
                          a small model from "rambling" (every token costs).
    - keep_alive        : keeps the model loaded in memory between two calls
                          (otherwise we pay for reloading every time!).
    """

    def __init__(
        self,
        model: str = "qwen2.5:1.5b",
        host: str = "http://localhost:11434",
        num_ctx: int = 4096,
        num_predict: int = 256,
        timeout_s: float = 300.0,
    ):
        import httpx

        self.model = model
        self.host = host.rstrip("/")
        self.options = {"temperature": 0, "num_ctx": num_ctx, "num_predict": num_predict}
        self._http = httpx.Client(timeout=timeout_s)
        self.name = f"ollama:{model}"

    def chat(self, messages: list[dict], tools: list[dict] | None = None, json_mode: bool = False,
             num_predict: int | None = None) -> LLMResponse:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "keep_alive": "30m",
            "options": dict(self.options, **({"num_predict": num_predict} if num_predict else {})),
        }
        if tools:
            payload["tools"] = tools
        elif json_mode:
            payload["format"] = "json"   # forces valid JSON output
        t0 = time.perf_counter()
        r = self._http.post(f"{self.host}/api/chat", json=payload)
        r.raise_for_status()
        js = r.json()
        msg = js.get("message", {})
        calls = []
        for tc in msg.get("tool_calls") or []:
            fn = tc.get("function", {})
            args = fn.get("arguments", {})
            if isinstance(args, str):  # some models return a JSON string
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {}
            calls.append(ToolCall(fn.get("name", ""), args or {}))
        return LLMResponse(
            content=msg.get("content", "") or "",
            tool_calls=calls,
            tokens_in=int(js.get("prompt_eval_count", 0) or 0),
            tokens_out=int(js.get("eval_count", 0) or 0),
            duration_s=time.perf_counter() - t0,
            # Ollama returns its durations in nanoseconds.
            active_s=((js.get("prompt_eval_duration") or 0) + (js.get("eval_duration") or 0)) / 1e9,
        )

    def ping(self) -> bool:
        """Checks that Ollama is running and the model is downloaded."""
        try:
            r = self._http.get(f"{self.host}/api/tags", timeout=5)
            names = [m.get("name", "") for m in r.json().get("models", [])]
            return any(n.startswith(self.model) for n in names)
        except Exception:
            return False


# =============================================================================
# FakeLLM (to test without Ollama)
# =============================================================================
def _estimate_tokens(text: str) -> int:
    """Classic approximation: ~4 characters per token."""
    return max(1, len(text) // 4)


def _last_json(text: str) -> dict:
    """Extracts the last JSON object {...} from a text (or {} if none)."""
    for m in reversed(list(re.finditer(r"\{[^{}]*\}", text, flags=re.S))):
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            continue
    return {}


FAKE_POLICY_CODE = '''
def policy(obs):
    # Generated policy (FakeLLM): water heater during the lowest-carbon forecast
    # hours, heating shed at the peak of stress days.
    forecast = obs.get("co2_forecast_24h") or []
    hour = obs["hour"]
    need_hours = int(obs["water_remaining_kwh"] / max(obs["water_heater_kw"], 0.1) + 0.999)
    water_on = False
    if need_hours > 0 and forecast:
        remaining = forecast[: obs["hours_left_today"]]
        ranked = sorted(range(len(remaining)), key=lambda i: remaining[i])
        water_on = 0 in ranked[:need_hours]
    offset = 0.0
    tense = obs["tempo_today"] in ("ROUGE", "BLANC")
    if tense and hour in (18, 19, 20):
        offset = -1.5 if obs["t_in_c"] > obs["comfort_min_c"] + 0.8 else 0.0
    elif tense and hour in (15, 16, 17):
        offset = 0.5
    return {"heating_offset_c": offset, "water_heater_on": water_on, "reason": "politique generee"}
'''


class FakeLLM:
    """
    Fake LLM: mimics the BEHAVIOR of an agent (tool calls, JSON,
    code generation) with simple rules. See the warning at the top
    of the file.
    """

    name = "fake"

    def chat(self, messages: list[dict], tools: list[dict] | None = None, json_mode: bool = False,
             num_predict: int | None = None) -> LLMResponse:
        prompt = "\n".join(str(m.get("content", "")) for m in messages)
        tin = _estimate_tokens(prompt) + (_estimate_tokens(json.dumps(tools)) if tools else 0)

        # 1) Code generation request ("designer" strategy)
        if "def policy(obs)" in messages[0].get("content", ""):
            content = f"```python\n{FAKE_POLICY_CODE}\n```"
            return LLMResponse(content, tokens_in=tin, tokens_out=_estimate_tokens(content))

        # 2) Tool-using agent: if it has not called a tool yet, it calls one.
        already_called = any(m.get("role") == "tool" for m in messages)
        if tools and not already_called:
            names = [t["function"]["name"] for t in tools]
            call = ToolCall(names[0], {"hours": 12} if "forecast" in names[0] else {})
            return LLMResponse("", tool_calls=[call], tokens_in=tin, tokens_out=15)

        # 3) Final decision in JSON, from the observation found in the prompt.
        obs = {}
        for m in messages:
            if m.get("role") == "user":
                obs = _last_json(str(m.get("content", ""))) or obs
        forecast_min = None
        for m in messages:
            if m.get("role") == "tool":
                nums = [float(x) for x in re.findall(r"\d+(?:\.\d+)?", str(m.get("content", "")))]
                if nums:
                    forecast_min = min(nums)
        co2 = float(obs.get("co2_now_g_per_kwh", 50))
        hour = int(obs.get("hour", 12))
        tempo = str(obs.get("tempo_today", "BLUE"))
        water_on = hour in (1, 2, 3, 4) or (forecast_min is not None and co2 <= forecast_min + 2)
        offset = -1.5 if (tempo in ("RED", "WHITE") and hour in (18, 19, 20)) else 0.0
        decision = {"heating_offset_c": offset, "water_heater_on": water_on, "reason": "fake"}
        if "demand-response aggregator" in str(messages[0].get("content", "")):   # demand-response domain
            tool_txt = "\n".join(str(m.get("content", "")) for m in messages if m.get("role") == "tool")
            groups = re.findall(r"^([a-z_]+/(?:heating|water_heater|ev)):", tool_txt, flags=re.M)
            heating = [g for g in groups if g.endswith("/heating") and not g.startswith("retirees")]
            decision = {"order": [g for g in groups if not g.endswith("/heating")] + heating,
                        "exclude": [g for g in groups if g == "retirees/heating"],
                        "preheat": True, "precharge_battery": True, "reason": "fake: protect the retirees"}
            content = json.dumps(decision)
            return LLMResponse(content, tokens_in=tin, tokens_out=_estimate_tokens(content))
        if "battery" in str(messages[0].get("content", "")):   # control domain: battery decision
            decision = {"grid_charge_kw": 0.0, "max_discharge_kw": 50.0, "reason": "fake"}
        content = json.dumps(decision)
        return LLMResponse(content, tokens_in=tin, tokens_out=_estimate_tokens(content))

    def ping(self) -> bool:
        return True


def make_llm(kind: str = "fake", model: str = "qwen2.5:1.5b", **kw):
    """Factory: make_llm("ollama", "qwen2.5:3b") or make_llm("fake")."""
    if kind == "ollama":
        return OllamaClient(model=model, **kw)
    return FakeLLM()
