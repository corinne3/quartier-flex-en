"""
harness.py: the agent loop.

Loop outline (one "run" = one agent decision)
---------------------------------------------

    messages = [system, user]
    repeat (at most max_steps times):
        [budget exceeded?] -> STOP, safe fallback
        response = LLM(messages, tools)
        if the response asks for tools:
            run each tool, append the results to the messages
            -> loop again (the LLM will read the results)
        else:
            extract the JSON from the response, validate it (Pydantic)
            valid   -> END, return the decision
            invalid -> explain the error to the LLM, loop again (self-correction)
    -> STOP (too many steps), safe fallback

The 4 "harness engineering" levers visible here
-----------------------------------------------
1. max_steps : limits the number of round trips (each turn = one LLM call).
2. Budget    : cap on tokens and calls. An agent without a budget can
               loop and consume without limit. Here, it is an ENERGY BUDGET
               in disguise: tokens ≈ compute ≈ kWh.
3. Validation: an invalid output is never applied as-is.
4. Trace     : every step is recorded (debugging + demo to the jury:
               "here is what the agent thought and how much it cost").

The safe fallback is NOT here: it is the strategy that decides what
to do when the agent fails (usually: apply the simple rule).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, ValidationError

from .tools import ToolRegistry


@dataclass
class Budget:
    """Caps PER RUN (per decision)."""

    max_llm_calls: int = 4
    max_tokens: int = 4000

    def exceeded(self, calls: int, tokens: int) -> bool:
        return calls >= self.max_llm_calls or tokens >= self.max_tokens


@dataclass
class AgentResult:
    output: BaseModel | None           # the validated decision (None on failure)
    stop_reason: str                   # "ok", "budget", "max_steps", "llm_error"
    llm_calls: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    trace: list[dict] = field(default_factory=list)


def extract_json(text: str) -> dict | None:
    """
    Small models often wrap the JSON in text or in ```json ...```.
    We look for the first decodable JSON object in the response.
    """
    text = text.strip()
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        pass
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.S)
    candidates = [fence.group(1)] if fence else []
    candidates += re.findall(r"\{.*\}", text, flags=re.S)
    for c in candidates:
        try:
            obj = json.loads(c)
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            continue
    return None


class AgentHarness:
    def __init__(
        self,
        llm,
        system_prompt: str,
        output_model: type[BaseModel],
        tools: ToolRegistry | None = None,
        max_steps: int = 4,
        budget: Budget | None = None,
        meter=None,
        keep_trace: bool = True,
    ):
        self.llm = llm
        self.system_prompt = system_prompt
        self.output_model = output_model
        self.tools = tools
        self.max_steps = max_steps
        self.budget = budget or Budget()
        self.meter = meter
        self.keep_trace = keep_trace

    def run(self, user_message: str) -> AgentResult:
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_message},
        ]
        res = AgentResult(output=None, stop_reason="max_steps")
        schemas = self.tools.schemas() if self.tools else None

        for step in range(self.max_steps):
            if self.budget.exceeded(res.llm_calls, res.tokens_in + res.tokens_out):
                res.stop_reason = "budget"
                break
            try:
                resp = self.llm.chat(messages, tools=schemas, json_mode=not schemas)
            except Exception as e:
                res.stop_reason = "llm_error"
                res.trace.append({"step": step, "error": str(e)})
                break

            res.llm_calls += 1
            res.tokens_in += resp.tokens_in
            res.tokens_out += resp.tokens_out
            if self.meter is not None:
                self.meter.record_llm(resp.tokens_in, resp.tokens_out, resp.duration_s, resp.active_s)
            if self.keep_trace:
                res.trace.append({
                    "step": step,
                    "content": resp.content[:500],
                    "tool_calls": [{"name": c.name, "args": c.arguments} for c in resp.tool_calls],
                    "tokens_in": resp.tokens_in,
                    "tokens_out": resp.tokens_out,
                    "duration_s": round(resp.duration_s, 3),
                })

            # --- Case 0: tool call written AS TEXT (common with small models):
            # the model replies {"name": "get_forecast", "arguments": {...}} in the content instead
            # of using the "tool_calls" channel. Without this guard, that reply passed validation
            # (every decision field having a default value): the agent "decided"...
            # to decide nothing. Real bug observed during the hackathon.
            if not resp.tool_calls and self.tools is not None:
                maybe = extract_json(resp.content) or {}
                if isinstance(maybe.get("name"), str) and maybe["name"] in self.tools._tools:
                    from .llm import ToolCall

                    args = maybe.get("arguments") or maybe.get("parameters") or {}
                    resp.tool_calls = [ToolCall(maybe["name"], args if isinstance(args, dict) else {})]
                    if self.keep_trace:
                        res.trace.append({"step": step, "note": "tool call written as text, converted by the harness"})

            # --- Case 1: the LLM wants to use tools
            if resp.tool_calls and self.tools is not None:
                messages.append({
                    "role": "assistant",
                    "content": resp.content,
                    "tool_calls": [
                        {"function": {"name": c.name, "arguments": c.arguments}} for c in resp.tool_calls
                    ],
                })
                for c in resp.tool_calls:
                    result = self.tools.call(c.name, c.arguments)
                    messages.append({"role": "tool", "content": result, "tool_name": c.name})
                    if self.keep_trace:
                        res.trace.append({"step": step, "tool_result": {c.name: result[:300]}})
                continue

            # --- Case 2: final answer -> validate
            data = extract_json(resp.content)
            if data is not None:
                try:
                    res.output = self.output_model.model_validate(data)
                    res.stop_reason = "ok"
                    return res
                except ValidationError as e:
                    err = f"Invalid JSON: {e.errors()[:2]}"
            else:
                err = "No JSON found in your reply."
            # Self-correction: send the error back to the LLM.
            messages.append({"role": "assistant", "content": resp.content})
            messages.append({
                "role": "user",
                "content": f"{err} Reply ONLY with a valid JSON object matching the requested format.",
            })

        return res
