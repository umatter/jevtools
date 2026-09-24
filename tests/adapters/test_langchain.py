"""LangChain adapter (spec §7.2.3, §10.6): ``JevChatModel.bind_tools`` + a fake ``ToolNode`` loop, per-call
context through ``configurable``, pending prompts, and ``confirm_node`` with a stand-in ``interrupt``.

Answers are scripted fixtures: nothing here is evidence about Jev's accuracy."""

from __future__ import annotations

import sys
import types
from typing import Any, Literal

import pytest

pytest.importorskip("langchain_core")

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage  # noqa: E402
from langchain_core.tools import tool  # noqa: E402

from jevtools.adapters.langchain import (  # noqa: E402
    JevChatModel,
    confirm_node,
    needs_confirmation,
    to_openai_messages,
    tool_choice_to_openai,
)
from tests.adapters.support import WEATHER_REQUEST, loop_script, tool_options, weather_router  # noqa: E402
from tests.scenario import scripts  # noqa: E402
from tests.scenario.fixtures import R2_HISTORY, scenario_router  # noqa: E402


@tool
def get_weather(city: str, unit: Literal["celsius", "fahrenheit"] = "celsius") -> str:
    """Get the current weather for a city."""
    return f"61 degrees {unit} in {city}"


@tool
def search_web(query: str) -> str:
    """Search the web for pages about a topic."""
    return f"results for {query}"


TOOLS = {"get_weather": get_weather, "search_web": search_web}


def _tool_node(message: AIMessage) -> list[ToolMessage]:
    """A minimal stand-in for LangGraph's ``ToolNode``: run every tool call, answer with ``ToolMessage``s."""
    out: list[ToolMessage] = []
    for call in message.tool_calls:
        result = TOOLS[call["name"]].invoke(call)
        assert isinstance(result, ToolMessage)
        out.append(result)
    return out


def test_bind_tools_and_a_tool_node_loop() -> None:
    router, backend = weather_router(loop_script)
    llm = JevChatModel(router=router).bind_tools([get_weather, search_web])
    messages: list[BaseMessage] = [HumanMessage(WEATHER_REQUEST)]
    for _ in range(4):
        ai = llm.invoke(messages)
        assert isinstance(ai, AIMessage)
        messages.append(ai)
        if not ai.tool_calls:
            break
        messages += _tool_node(ai)
    first, tool_msg, last = messages[1], messages[2], messages[3]
    assert isinstance(first, AIMessage) and first.tool_calls[0]["name"] == "get_weather"
    assert first.tool_calls[0]["args"] == {"city": "Zurich", "unit": "fahrenheit"}
    assert first.tool_calls[0]["id"].startswith("call_jev_") and first.content == ""
    assert first.response_metadata["jev"]["outcome"] == "execute"
    assert isinstance(tool_msg, ToolMessage) and tool_msg.content == "61 degrees fahrenheit in Zurich"
    assert isinstance(last, AIMessage) and not last.tool_calls
    assert last.response_metadata["jev"]["outcome"] == "done" and len(messages) == 4
    state = backend.requests[1].state
    assert isinstance(state, dict) and state["observations"][0]["preview"] == "61 degrees fahrenheit in Zurich"
    assert first.usage_metadata is not None and first.usage_metadata["output_tokens"] == 0


async def test_agenerate_and_bound_tool_subset() -> None:
    router, backend = weather_router()
    llm = JevChatModel(router=router).bind_tools([get_weather])
    ai = await llm.ainvoke([HumanMessage(WEATHER_REQUEST)])
    assert isinstance(ai, AIMessage) and ai.tool_calls[0]["name"] == "get_weather"
    assert tool_options(backend.requests[0]) == ["get_weather", "NO_TOOL", "UNSUPPORTED"]


def test_tool_choice_mapping() -> None:
    router, backend = weather_router()
    JevChatModel(router=router).bind_tools([get_weather, search_web], tool_choice="get_weather").invoke(
        [HumanMessage(WEATHER_REQUEST)])  # fmt: skip
    assert "tool" not in backend.requests[0].questions
    assert tool_choice_to_openai("any") == "required" and tool_choice_to_openai(None) == "auto"
    assert tool_choice_to_openai(False) == "none"
    with pytest.raises(ValueError, match="not a bound tool"):
        tool_choice_to_openai("nope", [{"type": "function", "function": {"name": "get_weather"}}])


def test_per_call_context_through_configurable() -> None:
    router, backend = weather_router()
    llm = JevChatModel(router=router, context={"user": {"name": "Ada"}})
    llm.invoke([HumanMessage(WEATHER_REQUEST)])
    llm.invoke([HumanMessage(WEATHER_REQUEST)], config={"configurable": {"jev_context": {"user": {"name": "Bo"}}}})
    users = [r.state["user"] for r in backend.requests if isinstance(r.state, dict)]
    assert users == [{"name": "Ada"}, {"name": "Bo"}]


def test_message_conversion() -> None:
    ai = AIMessage(content="", tool_calls=[{"name": "get_weather", "args": {"city": "Bern"}, "id": "c1"}],
                   response_metadata={"jev": {"pending_id": "pnd_1"}})  # fmt: skip
    out = to_openai_messages([SystemMessage("sys"), HumanMessage("hi"), ai,
                              ToolMessage(content="ok", tool_call_id="c1", name="get_weather")])  # fmt: skip
    assert out == [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": None, "x_jev": {"pending_id": "pnd_1"}, "tool_calls": [
            {"id": "c1", "type": "function", "function": {"name": "get_weather", "arguments": '{"city": "Bern"}'}}]},
        {"role": "tool", "tool_call_id": "c1", "content": "ok", "name": "get_weather"},
    ]  # fmt: skip


def _r2_messages() -> list[BaseMessage]:
    history: list[BaseMessage] = [HumanMessage(R2_HISTORY[0]["content"]), AIMessage(R2_HISTORY[1]["content"])]
    return [*history, HumanMessage(scripts.R2_REQUEST)]


def test_a_confirm_is_resumed_by_the_next_human_message() -> None:
    router, backend = scenario_router(scripts.R2)
    llm = JevChatModel(router=router)
    messages = _r2_messages()
    card = llm.invoke(messages)
    assert isinstance(card, AIMessage) and not card.tool_calls and card.content
    assert card.response_metadata["jev"]["outcome"] == "confirm" and needs_confirmation({"messages": [card]})
    done = llm.invoke([*messages, card, HumanMessage("ok")])
    assert isinstance(done, AIMessage) and done.tool_calls[0]["name"] == "send_email"
    assert len(backend.requests) == 1  # the click made no Jev call


def test_abstain_goes_to_the_text_llm() -> None:
    class _Joker:
        def invoke(self, messages: Any) -> AIMessage:
            return AIMessage("Why did the tool cross the road?")

    router, _ = scenario_router(scripts.R7)
    ai = JevChatModel(router=router, text_llm=_Joker()).invoke([HumanMessage(scripts.R7_REQUEST)])
    assert ai.content == "Why did the tool cross the road?" and ai.response_metadata["jev"]["outcome"] == "abstain"


def test_confirm_node_interrupts_and_returns_the_reply() -> None:
    router, _ = scenario_router(scripts.R2)
    card = JevChatModel(router=router).invoke(_r2_messages())
    asked: list[Any] = []

    def fake_interrupt(value: Any) -> Any:
        asked.append(value)
        return {"selection": "ok"}

    update = confirm_node({"messages": [card]}, interrupt=fake_interrupt)
    assert asked[0]["kind"] == "jevtools" and asked[0]["pending_id"] == card.response_metadata["jev"]["pending_id"]
    (reply,) = update["messages"]
    ok_text = next(o["text"] for o in asked[0]["prompt"]["options"] if o["id"] == "ok")
    assert isinstance(reply, HumanMessage) and reply.content == ok_text
    assert confirm_node({"messages": [AIMessage("plain")]}, interrupt=fake_interrupt) == {}


def test_confirm_node_imports_langgraph_lazily(monkeypatch: pytest.MonkeyPatch) -> None:
    router, _ = scenario_router(scripts.R2)
    card = JevChatModel(router=router).invoke(_r2_messages())
    fake = types.ModuleType("langgraph.types")
    fake.interrupt = lambda value: "cancel"  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "langgraph", types.ModuleType("langgraph"))
    monkeypatch.setitem(sys.modules, "langgraph.types", fake)
    assert confirm_node({"messages": [card]})["messages"][0].content == "cancel"


def test_confirm_node_without_langgraph_says_so(monkeypatch: pytest.MonkeyPatch) -> None:
    router, _ = scenario_router(scripts.R2)
    card = JevChatModel(router=router).invoke(_r2_messages())
    monkeypatch.setitem(sys.modules, "langgraph.types", None)
    with pytest.raises(ImportError, match="needs LangGraph"):
        confirm_node({"messages": [card]})


# -- review-edges regressions -----------------------------------------------------------------------------------------


def test_with_structured_output_elects_the_schema_at_the_read_tier() -> None:
    """``with_structured_output`` binds the schema as an output tool: side-effect free (``risk: read``), so no
    ``authorized`` gate and no confirm band; it returns the model instead of ``None`` (#9)."""
    from pydantic import BaseModel, Field

    from jevtools.backends.scripted import ScriptedBackend
    from jevtools.router import Router
    from jevtools.wire import DecisionRequest
    from tests.adapters.support import weather_context

    class Triage(BaseModel):
        """Triage the request."""

        kind: Literal["weather", "news", "other"]
        urgent: bool

    sent: list[DecisionRequest] = []

    def script(request: DecisionRequest) -> dict[str, Any]:
        sent.append(request)
        name = next(iter(request.questions["tool"].criteria))  # type: ignore[union-attr, arg-type]
        return {"tool": {name: 0.95, "UNSUPPORTED": 0.05}, "*authorized*": 0.9, "*.urgent": 0.1,
                "*.kind": {"weather": 0.9, "news": 0.05, "other": 0.03, "NOT_STATED": 0.01,
                           "NONE_OF_THESE": 0.01}}  # fmt: skip

    llm = JevChatModel(router=Router([], backend=ScriptedBackend(script), context=weather_context()))
    assert llm.with_structured_output(Triage).invoke(WEATHER_REQUEST) == Triage(kind="weather", urgent=False)
    assert not any(q.endswith(".authorized") for q in sent[0].questions)

    class Risky(BaseModel):
        """Record the triage."""

        model_config = {"json_schema_extra": {"x-jev": {"risk": "external"}}}
        kind: Literal["weather", "news", "other"] = Field(description="The kind")

    sent.clear()
    llm.with_structured_output(Risky).invoke(WEATHER_REQUEST)
    assert any(q.endswith(".authorized") for q in sent[0].questions)  # an explicit risk is kept
