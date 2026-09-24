"""Anthropic interop (spec §3.10): ``tool_use`` blocks, Messages responses, Anthropic tools → Catalog and
Anthropic messages → OpenAI-style messages. Answers are scripted fixtures."""

from __future__ import annotations

from typing import Any

import pytest

from jevtools.adapters import anthropic as ja
from jevtools.adapters.pending import InMemoryPendingStore
from jevtools.policy import Tier
from tests.adapters.support import WEATHER_REQUEST, loop_script, weather_router
from tests.scenario import scripts
from tests.scenario.fixtures import scenario_messages, scenario_router

ANTHROPIC_TOOLS: list[dict[str, Any]] = [
    {"name": "get_weather", "description": "Get the current weather for a city.",
     "input_schema": {"type": "object", "required": ["city"],
                      "properties": {"city": {"type": "string", "description": "The city name"},
                                     "unit": {"type": "string", "enum": ["celsius", "fahrenheit"],
                                              "default": "celsius"}}}},
    {"name": "search_web", "description": "Search the web for pages about a topic.",
     "input_schema": {"type": "object", "required": ["query"],
                      "properties": {"query": {"type": "string", "description": "The search query"}}}},
]  # fmt: skip


def test_to_tool_use_shares_the_call_digest() -> None:
    router, _ = weather_router()
    decision = router.decide(WEATHER_REQUEST)
    (block,) = ja.to_tool_use(decision)
    assert block == {"type": "tool_use", "id": "toolu_jev_" + decision.tool_calls[0].id.removeprefix("call_jev_"),
                     "name": "get_weather", "input": {"city": "Zurich", "unit": "fahrenheit"}}  # fmt: skip
    message = ja.to_message(decision)
    assert message["stop_reason"] == "tool_use" and message["content"] == [block]
    assert message["type"] == "message" and message["usage"]["output_tokens"] == 0
    assert message["x_jev"]["outcome"] == "execute"


def test_non_execute_outcomes_have_no_tool_use() -> None:
    router, _ = scenario_router(scripts.R2)
    confirm = router.decide(scenario_messages(scripts.R2_REQUEST, history=True))
    assert ja.to_tool_use(confirm) == []
    message = ja.to_message(confirm)
    assert message["stop_reason"] == "end_turn" and message["content"][0]["type"] == "text"
    assert message["x_jev"]["pending_id"] == confirm.pending_id
    abstain = scenario_router(scripts.R7)[0].decide(scenario_messages(scripts.R7_REQUEST))
    assert ja.to_message(abstain)["content"] == []


def test_anthropic_tools_become_a_catalog() -> None:
    catalog = ja.catalog_from_anthropic(ANTHROPIC_TOOLS)
    assert catalog.names == ("get_weather", "search_web") and catalog.get("get_weather").tier is Tier.READ
    assert catalog.get("get_weather").slot("unit").kind == "enum"
    with pytest.raises(ValueError, match="input_schema"):
        ja.tool_to_openai({"type": "web_search_20250305", "name": "web_search"})


def test_tool_choice_mapping() -> None:
    assert ja.tool_choice_to_openai(None) == "auto"
    assert ja.tool_choice_to_openai({"type": "any"}) == "required"
    assert ja.tool_choice_to_openai({"type": "none"}) == "none"
    assert ja.tool_choice_to_openai({"type": "tool", "name": "get_weather"}) == {
        "type": "function", "function": {"name": "get_weather"}}  # fmt: skip
    with pytest.raises(ValueError):
        ja.tool_choice_to_openai({"type": "bogus"})


def test_messages_convert_to_openai_style() -> None:
    messages = [
        {"role": "user", "content": WEATHER_REQUEST},
        {"role": "assistant", "content": [{"type": "text", "text": "Checking."},
                                          {"type": "tool_use", "id": "toolu_1", "name": "get_weather",
                                           "input": {"city": "Zurich"}}]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "toolu_1",
                                      "content": [{"type": "text", "text": "61F"}]},
                                     {"type": "text", "text": "thanks"}]},
    ]  # fmt: skip
    out = ja.to_openai_messages(messages, system="Be brief.")
    assert out == [
        {"role": "system", "content": "Be brief."},
        {"role": "user", "content": WEATHER_REQUEST},
        {"role": "assistant", "content": "Checking.", "tool_calls": [
            {"id": "toolu_1", "type": "function",
             "function": {"name": "get_weather", "arguments": '{"city":"Zurich"}'}}]},
        {"role": "tool", "tool_call_id": "toolu_1", "content": "61F"},
        {"role": "user", "content": "thanks"},
    ]  # fmt: skip


def test_complete_runs_a_loop_and_resumes_prompts() -> None:
    router, backend = weather_router(loop_script)
    first = ja.complete([{"role": "user", "content": WEATHER_REQUEST}], ANTHROPIC_TOOLS, router=router)
    (block,) = first["content"]
    second = ja.complete([
        {"role": "user", "content": WEATHER_REQUEST},
        {"role": "assistant", "content": first["content"]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": block["id"], "content": "61F"}]},
    ], ANTHROPIC_TOOLS, router=router)  # fmt: skip
    assert second["x_jev"]["outcome"] == "done" and second["stop_reason"] == "end_turn"

    router2, backend2 = scenario_router(scripts.R2)
    store = InMemoryPendingStore()
    history = scenario_messages(scripts.R2_REQUEST, history=True)
    card = ja.complete(history, router=router2, store=store)
    done = ja.complete([*history, {"role": "assistant", "content": card["content"]},
                        {"role": "user", "content": "ok"}], router=router2, store=store)  # fmt: skip
    assert done["stop_reason"] == "tool_use" and len(backend2.requests) == 1
