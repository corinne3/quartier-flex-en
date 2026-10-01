"""
agent/: the agentic harness, reused as-is from Bilan Net (loop, typed tools, budget, validation).
See the corinne3/bilan-net repository, docs/03_HARNESS_AGENTS.md.
"""

from .harness import AgentHarness, AgentResult, Budget  # noqa: F401
from .llm import FakeLLM, LLMResponse, OllamaClient, make_llm  # noqa: F401
from .tools import Tool, ToolRegistry  # noqa: F401
