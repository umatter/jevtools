"""Pydantic AI adapter (spec §7.2.5, §10.6): ``JevModel.request`` inside a real ``Agent`` with function tools,
structured output elected through the output tool, prompts resumed from the message history, text handoffs.

Answers are scripted fixtures: nothing here is evidence about Jev's accuracy."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal

import pytest

pytest.importorskip("pydantic_ai")

from pydantic import BaseModel  # noqa: E402
from pydantic_ai import Agent  # noqa: E402
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart  # noqa: E402
from pydantic_ai.models.function import AgentInfo, FunctionModel  # noqa: E402

from jevtools.adapters.pydantic_ai import ABSTAIN_TEXT, JevModel, to_openai_messages  # noqa: E402
from jevtools.backends.scripted import ScriptedBackend  # noqa: E402
from jevtools.router import Router  # noqa: E402
from jevtools.wire import DecisionRequest  # noqa: E402
from tests.adapters.support import (  # noqa: E402
    WEATHER,
    WEATHER_REQUEST,
    loop_script,
    tool_options,
    weather_context,
    weather_router,
)

calls: list[dict[str, Any]] = []


def _agent(model: JevModel, **kw: Any) -> Agent[None, Any]:
    agent: Agent[None, Any] = Agent(model, **kw)

    @agent.tool_plain
    def get_weather(city: str, unit: Literal["celsius", "fahrenheit"] = "celsius") -> str:
        """Get the current weather for a city."""
        calls.append({"city": city, "unit": unit})
        return f"61 degrees {unit} in {city}"

    @agent.tool_plain
    def search_web(query: str) -> str:
        """Search the web for pages about a topic."""
        return f"results for {query}"

    return agent


def test_agent_runs_the_elected_tool_and_finishes() -> None:
    calls.clear()
    router, backend = weather_router(loop_script)
    result = _agent(JevModel(router)).run_sync(WEATHER_REQUEST)
    assert calls == [{"city": "Zurich", "unit": "fahrenheit"}]
    assert result.output == "61 degrees fahrenheit in Zurich"  # the receipt of the finished loop
    responses = [m for m in result.all_messages() if isinstance(m, ModelResponse)]
    assert [r.finish_reason for r in responses] == ["tool_call", "stop"]
    assert responses[0].provider_details is not None and responses[0].provider_details["jev"]["outcome"] == "execute"
    assert responses[1].provider_details is not None and responses[1].provider_details["jev"]["outcome"] == "done"
    assert "DONE" in tool_options(backend.requests[1]) and len(backend.requests) == 2


def test_abstain_without_a_text_model_answers_the_fixed_text() -> None:
    router, _ = weather_router({**WEATHER, "tool": {"NO_TOOL": 0.95, "get_weather": 0.03, "search_web": 0.01,
                                                    "UNSUPPORTED": 0.01}})  # fmt: skip
    assert _agent(JevModel(router)).run_sync("Tell me a joke").output == ABSTAIN_TEXT


def test_abstain_hands_off_to_the_text_model_without_function_tools() -> None:
    seen: list[AgentInfo] = []

    def joker(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        seen.append(info)
        return ModelResponse(parts=[TextPart("A joke.")])

    router, _ = weather_router({**WEATHER, "tool": {"NO_TOOL": 0.95, "get_weather": 0.03, "search_web": 0.01,
                                                    "UNSUPPORTED": 0.01}})  # fmt: skip
    result = _agent(JevModel(router, text_model=FunctionModel(joker))).run_sync("Tell me a joke")
    assert result.output == "A joke." and seen[0].function_tools == []


class Triage(BaseModel):
    """Triage the request."""

    kind: Literal["weather", "news", "other"]
    urgent: bool


def test_structured_output_is_elected_through_the_output_tool() -> None:
    sent: list[DecisionRequest] = []

    def script(request: DecisionRequest) -> Mapping[str, Any]:
        sent.append(request)
        return {"tool": {"final_result": 0.95, "UNSUPPORTED": 0.05},
                "final_result.kind": {"weather": 0.9, "news": 0.05, "other": 0.03, "NOT_STATED": 0.01,
                                      "NONE_OF_THESE": 0.01},
                "final_result.urgent": 0.1}  # fmt: skip

    router = Router([], backend=ScriptedBackend(script), context=weather_context())
    result = Agent(JevModel(router), output_type=Triage).run_sync(WEATHER_REQUEST)
    assert result.output == Triage(kind="weather", urgent=False)
    # output tools are side-effect free (declared risk: read): no `authorized` gate; NO_TOOL removed (text disallowed)
    assert list(sent[0].questions) == ["tool", "final_result.kind", "final_result.urgent"]
    assert "NO_TOOL" not in tool_options(sent[0])


def _ambiguous(request: DecisionRequest) -> Mapping[str, Any]:
    if "DONE" in tool_options(request):
        return {"tool": {"DONE": 0.95, "get_weather": 0.03, "NO_TOOL": 0.01, "UNSUPPORTED": 0.01}}
    return {**WEATHER, "get_weather.city": {"Zurich": 0.5, "Bern": 0.46, "NOT_STATED": 0.02, "NONE_OF_THESE": 0.02}}


def test_a_clarify_prompt_is_resumed_from_the_message_history() -> None:
    calls.clear()
    router, backend = weather_router(_ambiguous)
    agent = _agent(JevModel(router))
    first = agent.run_sync("What's the weather in Zurich or Bern in Fahrenheit?")
    assert first.output.startswith("Which") and calls == []
    response = first.all_messages()[-1]
    assert isinstance(response, ModelResponse) and response.provider_details is not None
    assert response.provider_details["jev"]["outcome"] == "clarify"
    second = agent.run_sync("1", message_history=first.all_messages())
    assert calls == [{"city": "Zurich", "unit": "fahrenheit"}]
    assert second.output == "61 degrees fahrenheit in Zurich"
    assert len(backend.requests) == 2  # the click itself made no Jev call; the second request is the DONE step


def test_message_conversion_keeps_the_pending_id() -> None:
    router, _ = weather_router(_ambiguous)
    first = _agent(JevModel(router)).run_sync("What's the weather in Zurich or Bern in Fahrenheit?")
    converted = to_openai_messages(first.all_messages())
    assert [m["role"] for m in converted] == ["user", "assistant"]
    assert converted[1]["x_jev"]["pending_id"].startswith("pnd_") and converted[1]["content"] == first.output


# -- review-edges regressions -----------------------------------------------------------------------------------------


def test_a_prompt_under_structured_output_raises_with_the_card_and_resumes() -> None:
    """With ``output_type=Model`` a text prompt would be rejected and re-decided (a wasted Jev round, then
    UnexpectedModelBehavior); the prompt surfaces as JevPromptRequired and resumes from its messages (#4)."""
    from jevtools.adapters.pydantic_ai import JevPromptRequired

    def script(request: DecisionRequest) -> Mapping[str, Any]:
        if "DONE" in tool_options(request):
            return {"tool": {"final_result": 0.95, "DONE": 0.03, "UNSUPPORTED": 0.02},
                    "final_result.kind": {"weather": 0.9, "news": 0.05, "other": 0.03, "NOT_STATED": 0.01,
                                          "NONE_OF_THESE": 0.01}, "final_result.urgent": 0.1}  # fmt: skip
        return {"tool": {"get_weather": 0.95, "final_result": 0.04, "UNSUPPORTED": 0.01},
                "get_weather.city": {"Zurich": 0.5, "Bern": 0.46, "NOT_STATED": 0.02, "NONE_OF_THESE": 0.02},
                "get_weather.unit": {"fahrenheit": 0.97, "celsius": 0.01, "NOT_STATED": 0.01, "NONE_OF_THESE": 0.01},
                "final_result.kind": {"weather": 0.9, "news": 0.05, "other": 0.03, "NOT_STATED": 0.01,
                                      "NONE_OF_THESE": 0.01}, "final_result.urgent": 0.1}  # fmt: skip

    calls.clear()
    backend = ScriptedBackend(script)
    agent = _agent(JevModel(Router([], backend=backend, context=weather_context())), output_type=Triage)
    with pytest.raises(JevPromptRequired) as info:
        agent.run_sync("What's the weather in Zurich or Bern in Fahrenheit?")
    exc = info.value
    assert exc.decision.outcome.value == "clarify" and exc.pending_id and exc.text.startswith("Which")
    assert len(backend.requests) == 1  # no second (re-decided) round
    result = agent.run_sync("1", message_history=exc.messages)  # the user's answer resumes the prompt
    assert calls == [{"city": "Zurich", "unit": "fahrenheit"}]
    assert result.output == Triage(kind="weather", urgent=False)


def test_an_abstain_under_structured_output_raises_once() -> None:
    from jevtools.adapters.pydantic_ai import JevPromptRequired

    backend = ScriptedBackend({"tool": {"UNSUPPORTED": 0.9, "final_result": 0.1}})
    agent = Agent(JevModel(Router([], backend=backend, context=weather_context())), output_type=Triage)
    with pytest.raises(JevPromptRequired) as info:
        agent.run_sync("Book me a flight")
    assert info.value.decision.outcome.value in ("abstain", "escalate") and len(backend.requests) == 1
    # a text-capable output type still answers the prompt as text
    text_agent = Agent(JevModel(Router([], backend=ScriptedBackend({"tool": {"UNSUPPORTED": 0.9, "final_result": 0.1}}),
                                       context=weather_context())), output_type=[Triage, str])  # fmt: skip
    assert isinstance(text_agent.run_sync("Book me a flight").output, str)


async def test_run_stream_replays_the_decision() -> None:
    """``agent.run_stream`` works: the decision is replayed as one streamed response (#13)."""
    calls.clear()
    router, _ = weather_router(loop_script)
    agent = _agent(JevModel(router))
    async with agent.run_stream(WEATHER_REQUEST) as streamed:
        output = await streamed.get_output()
    assert output == "61 degrees fahrenheit in Zurich" and calls == [{"city": "Zurich", "unit": "fahrenheit"}]
