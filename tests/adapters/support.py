"""Shared helpers of the adapter tests: a small weather catalog and scripted Jev answers.

Scripted answers are fixtures, not model outputs: they say nothing about Jev's accuracy.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from jevtools.backends.scripted import ScriptedBackend
from jevtools.context import Context
from jevtools.router import Router
from jevtools.wire import ChoiceQuestion, DecisionRequest
from tests.scenario.fixtures import SCENARIO_NOW

WEATHER_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a city.",
        "parameters": {
            "type": "object",
            "required": ["city"],
            "properties": {
                "city": {"type": "string", "description": "The city name"},
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"], "default": "celsius",
                         "description": "The temperature unit"},
            },
        },
    },
}  # fmt: skip
SEARCH_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "search_web",
        "description": "Search the web for pages about a topic.",
        "parameters": {"type": "object", "required": ["query"],
                       "properties": {"query": {"type": "string", "description": "The search query"}}},
    },
}  # fmt: skip
WEATHER_REQUEST = "What's the weather like in Zurich in Fahrenheit?"
WEATHER: dict[str, Any] = {
    "tool": {"get_weather": 0.97, "search_web": 0.01, "NO_TOOL": 0.01, "UNSUPPORTED": 0.01},
    "get_weather.city": {"Zurich": 0.96, "NOT_STATED": 0.02, "NONE_OF_THESE": 0.02},
    "get_weather.unit": {"fahrenheit": 0.97, "celsius": 0.01, "NOT_STATED": 0.01, "NONE_OF_THESE": 0.01},
    "search_web.query.accept.*": 0.2,
}


def tool_options(request: DecisionRequest) -> list[str]:
    """The labels of the ``tool`` Choice of a sent request (empty when there is none)."""
    question = request.questions.get("tool")
    return list(question.criteria) if isinstance(question, ChoiceQuestion) else []


def loop_script(request: DecisionRequest) -> Mapping[str, Any]:
    """Weather first; once an observation exists (``DONE`` is offered), the loop is done."""
    if "DONE" in tool_options(request):
        return {**WEATHER, "tool": {"DONE": 0.95, "get_weather": 0.03, "NO_TOOL": 0.01, "UNSUPPORTED": 0.01}}
    return WEATHER


def weather_context() -> Context:
    """A fixed clock (the §13 ``now``) and no sources."""
    return Context(now=SCENARIO_NOW, locale="en-CH", user={"name": "Sam Muster"})


def weather_router(script: Any = None, **kw: Any) -> tuple[Router, ScriptedBackend]:
    """A router over the weather and search tools answered by ``script`` (default :data:`WEATHER`)."""
    backend = ScriptedBackend(WEATHER if script is None else script)
    return Router([WEATHER_TOOL, SEARCH_TOOL], backend=backend, context=weather_context(), **kw), backend


__all__ = ["SEARCH_TOOL", "WEATHER", "WEATHER_REQUEST", "WEATHER_TOOL", "loop_script", "tool_options",
           "weather_context", "weather_router"]  # fmt: skip
