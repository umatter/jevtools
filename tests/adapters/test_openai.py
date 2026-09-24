"""OpenAI adapter (spec §3.10, §7.2.1, §7.2.2, §7.2.4): message shapes per outcome, ``wrap`` pass-through,
``tool_choice`` mapping, drop-in loops and pending prompts matched by prefix hash or ``pending_id``.

Answers are scripted: nothing here is evidence about Jev's accuracy."""

from __future__ import annotations

import functools
import json
import sys
from datetime import datetime, timedelta, timezone
from typing import Any

import pytest

from jevtools.adapters import openai as jo
from jevtools.adapters._router import router_for
from jevtools.adapters.pending import InMemoryPendingStore, pending_scope, prefix_key
from jevtools.backends.errors import (
    BackendConfigError,
    BackendError,
    JevAuthError,
    JevNotFound,
    JevProtocolError,
    JevRateLimited,
    JevUnavailable,
    JevValidationError,
)
from jevtools.errors import BallotError, CatalogError
from jevtools.router import Router
from jevtools.wire import DecisionRequest, DecisionResponse
from tests.adapters.support import (
    SEARCH_TOOL,
    WEATHER_REQUEST,
    WEATHER_TOOL,
    loop_script,
    tool_options,
    weather_context,
    weather_router,
)
from tests.scenario import scripts
from tests.scenario.fixtures import scenario_messages, scenario_router

# --------------------------------------------------------------------------------------------------------------------
# complete: shapes per outcome
# --------------------------------------------------------------------------------------------------------------------


def test_execute_is_a_tool_calls_completion() -> None:
    router, _ = weather_router()
    doc = jo.complete([{"role": "user", "content": WEATHER_REQUEST}], router=router)
    assert doc["object"] == "chat.completion" and doc["model"] == "jevtools" and doc["id"].startswith("chatcmpl-dec_")
    choice = doc["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    message = choice["message"]
    assert message["role"] == "assistant" and message["content"] is None
    (call,) = message["tool_calls"]
    assert call["type"] == "function" and call["id"].startswith("call_jev_")
    assert call["function"] == {"name": "get_weather", "arguments": '{"city":"Zurich","unit":"fahrenheit"}'}
    assert message["x_jev"]["outcome"] == "execute" and message["x_jev"]["idempotency_key"].startswith("idem_")
    usage = doc["usage"]
    assert usage["completion_tokens"] == 0 and usage["prompt_tokens"] == usage["total_tokens"] > 0
    assert usage["x_jev"]["jev_calls"] == 1 and usage["x_jev"]["outcome"] == "execute"


def test_a_bare_string_is_one_user_turn_and_the_router_is_built_from_tools() -> None:
    from jevtools.backends.scripted import ScriptedBackend
    from tests.adapters.support import WEATHER

    doc = jo.complete(WEATHER_REQUEST, [WEATHER_TOOL, SEARCH_TOOL], backend=ScriptedBackend(WEATHER),
                      context=weather_context())  # fmt: skip
    assert doc["choices"][0]["message"]["tool_calls"][0]["function"]["name"] == "get_weather"


def test_router_options_need_no_router() -> None:
    router, _ = weather_router()
    with pytest.raises(TypeError, match="only accepted without a router"):
        jo.complete(WEATHER_REQUEST, router=router, policy=None)


def test_abstain_is_an_empty_stop_message() -> None:
    router, backend = scenario_router(scripts.R7)
    doc = jo.complete(scenario_messages(scripts.R7_REQUEST), router=router)
    choice = doc["choices"][0]
    assert choice["finish_reason"] == "stop" and "tool_calls" not in choice["message"]
    assert choice["message"]["content"] == "" and choice["message"]["x_jev"]["outcome"] == "abstain"
    assert len(backend.requests) == 1


def test_confirm_carries_the_prompt_options_and_pending_id() -> None:
    router, _ = scenario_router(scripts.R2)
    store = InMemoryPendingStore()
    messages = scenario_messages(scripts.R2_REQUEST, history=True)
    doc = jo.complete(messages, router=router, store=store)
    choice = doc["choices"][0]
    message = choice["message"]
    assert choice["finish_reason"] == "stop" and "tool_calls" not in message
    x = message["x_jev"]
    assert x["outcome"] == "confirm" and x["pending_id"].startswith("pnd_")
    assert [o["id"] for o in x["options"]] == ["ok", "change", "cancel"]
    assert isinstance(message["content"], str) and message["content"]
    # stored under the pending id and under the prefix hash of the requester's scope, the conversation and the card
    assert x["pending_id"] in store and prefix_key([*messages, message], pending_scope(router)) in store
    assert prefix_key([*messages, message]) not in store


# --------------------------------------------------------------------------------------------------------------------
# pending prompts (§7.2.4)
# --------------------------------------------------------------------------------------------------------------------


def _echo(message: dict[str, Any]) -> dict[str, Any]:
    """What a plain OpenAI client echoes: role and content (no ``x_jev``), plus SDK noise."""
    return {"role": "assistant", "content": message["content"], "refusal": None}


def test_confirm_is_resumed_by_prefix_hash_without_a_jev_call() -> None:
    router, backend = scenario_router(scripts.R2)
    store = InMemoryPendingStore()
    messages = scenario_messages(scripts.R2_REQUEST, history=True)
    card = jo.complete(messages, router=router, store=store)["choices"][0]["message"]
    doc = jo.complete([*messages, _echo(card), {"role": "user", "content": "ok"}], router=router, store=store)
    message = doc["choices"][0]["message"]
    assert doc["choices"][0]["finish_reason"] == "tool_calls"
    assert json.loads(message["tool_calls"][0]["function"]["arguments"])["to"] == "anna.keller@acme.com"
    assert len(backend.requests) == 1  # the click costs no Jev call
    assert card["x_jev"]["pending_id"] not in store and len(store) == 0  # consumed


def test_pending_is_matched_by_the_echoed_pending_id() -> None:
    router, backend = scenario_router(scripts.R2)
    store = InMemoryPendingStore()
    messages = scenario_messages(scripts.R2_REQUEST, history=True)
    card = jo.complete(messages, router=router, store=store)["choices"][0]["message"]
    edited = {"role": "assistant", "content": "(card shown as a widget)", "x_jev": card["x_jev"]}
    doc = jo.complete([*messages, edited, {"role": "user", "content": "yes"}], router=router, store=store)
    assert doc["choices"][0]["message"]["x_jev"]["outcome"] == "execute" and len(backend.requests) == 1


def test_an_explicit_pending_id_wins() -> None:
    router, _ = scenario_router(scripts.R2)
    store = InMemoryPendingStore()
    messages = scenario_messages(scripts.R2_REQUEST, history=True)
    card = jo.complete(messages, router=router, store=store)["choices"][0]["message"]
    edited = {"role": "assistant", "content": "something else"}
    doc = jo.complete([*messages, edited, {"role": "user", "content": "cancel"}], router=router, store=store,
                      pending_id=card["x_jev"]["pending_id"])  # fmt: skip
    assert doc["choices"][0]["message"]["x_jev"]["outcome"] == "abstain"


def test_expired_pending_compiles_a_fresh_turn() -> None:
    now = [datetime(2026, 9, 24, 12, 0, tzinfo=timezone.utc)]
    store = InMemoryPendingStore(clock=lambda: now[0])
    router, backend = scenario_router(scripts.R2)
    messages = scenario_messages(scripts.R2_REQUEST, history=True)
    card = jo.complete(messages, router=router, store=store)["choices"][0]["message"]
    now[0] += timedelta(minutes=16)
    jo.complete([*messages, _echo(card), {"role": "user", "content": "ok"}], router=router, store=store)
    assert len(backend.requests) == 2  # a miss simply compiles the whole conversation again
    state = backend.requests[1].state
    assert isinstance(state, dict) and state["request"] == "ok"
    assert [t["text"] for t in state["history"]][-1] == card["content"]


def test_a_pending_is_only_resumed_by_its_own_requester() -> None:
    router, backend = scenario_router(scripts.R2)
    store = InMemoryPendingStore()
    messages = scenario_messages(scripts.R2_REQUEST, history=True)
    card = jo.complete(messages, router=router, store=store, context={"user": {"name": "Sam Muster"}})
    message = card["choices"][0]["message"]
    reply = [*messages, {**_echo(message), "x_jev": message["x_jev"]}, {"role": "user", "content": "yes"}]
    other = jo.complete(reply, router=router, store=store, context={"user": {"name": "Bob"}})
    assert len(backend.requests) == 2 and other["choices"][0]["message"]["x_jev"].get("trace_id")
    assert message["x_jev"]["pending_id"] in store  # the other requester's miss keeps the handle
    own = jo.complete(reply, router=router, store=store, context={"user": {"name": "Sam Muster"}})
    assert own["choices"][0]["finish_reason"] == "tool_calls" and len(backend.requests) == 2


async def test_the_pending_scope_never_fetches_a_lazy_source() -> None:
    """The scope hashes given rows only: a ToolSource is named, not fetched (an async caller cannot run here)."""
    from jevtools.context import Context
    from jevtools.sources import Registry
    from jevtools.sources.toolsource import ToolSource

    fetched: list[str] = []

    async def caller(tool: str, args: dict[str, Any]) -> list[dict[str, str]]:
        fetched.append(tool)
        return [{"id": "a"}]

    router, _ = weather_router()
    lazy = ToolSource("list_things", key="id", name="things", call=caller)
    ctx = Context(sources=[lazy, Registry("teams", [{"id": "x"}], key="id")])
    scope = pending_scope(router, ctx)
    assert fetched == [] and lazy.stale and scope == pending_scope(router, ctx)
    other = Context(sources=[lazy, Registry("teams", [{"id": "y"}], key="id")])
    assert pending_scope(router, other) != scope  # given rows are part of the requester's identity


@pytest.mark.parametrize("tool_choice", ["none", {"type": "function", "function": {"name": "get_weather"}}])
def test_tool_choice_is_honoured_on_a_resumed_turn(tool_choice: Any) -> None:
    router, backend = scenario_router(scripts.R2)
    store = InMemoryPendingStore()
    messages = scenario_messages(scripts.R2_REQUEST, history=True)
    card = jo.complete(messages, router=router, store=store)["choices"][0]["message"]
    reply = [*messages, _echo(card), {"role": "user", "content": "yes"}]
    doc = jo.complete(reply, router=router, store=store, tool_choice=tool_choice)
    calls = doc["choices"][0]["message"].get("tool_calls") or []
    assert all(c["function"]["name"] != "send_email" for c in calls)  # never the pending call
    if tool_choice == "none":
        assert doc["choices"][0]["finish_reason"] == "stop" and len(backend.requests) == 1  # no Jev call
    assert card["x_jev"]["pending_id"] in store  # not consumed: an `auto` request can still resume it
    named = {"type": "function", "function": {"name": "send_email"}}
    done = jo.complete(reply, router=router, store=store, tool_choice=named)
    assert done["choices"][0]["message"]["tool_calls"][0]["function"]["name"] == "send_email"


def test_a_non_reply_conversation_never_matches() -> None:
    router, backend = scenario_router(scripts.R2)
    store = InMemoryPendingStore()
    messages = scenario_messages(scripts.R2_REQUEST, history=True)
    card = jo.complete(messages, router=router, store=store)["choices"][0]["message"]
    jo.complete([*messages, _echo(card)], router=router, store=store)  # ends with the assistant: a fresh turn
    assert len(backend.requests) == 2


# --------------------------------------------------------------------------------------------------------------------
# tool_choice / parallel_tool_calls (§7.2.2)
# --------------------------------------------------------------------------------------------------------------------


def test_tool_choice_none_makes_no_jev_call() -> None:
    router, backend = weather_router()
    doc = jo.complete(WEATHER_REQUEST, router=router, tool_choice="none")
    assert doc["choices"][0]["message"]["x_jev"]["outcome"] == "abstain" and backend.requests == []


def test_tool_choice_required_removes_no_tool() -> None:
    router, backend = weather_router()
    jo.complete(WEATHER_REQUEST, router=router, tool_choice="required")
    assert "NO_TOOL" not in tool_options(backend.requests[0]) and "UNSUPPORTED" in tool_options(backend.requests[0])


def test_named_tool_choice_asks_no_tool_question() -> None:
    router, backend = weather_router()
    doc = jo.complete(WEATHER_REQUEST, router=router, tool_choice={"type": "function",
                                                                   "function": {"name": "get_weather"}})  # fmt: skip
    assert "tool" not in backend.requests[0].questions
    assert not any(q.startswith("search_web.") for q in backend.requests[0].questions)
    assert doc["choices"][0]["message"]["tool_calls"][0]["function"]["name"] == "get_weather"


def test_parallel_tool_calls_is_accepted_and_one_call_is_decided() -> None:
    router, _ = weather_router()
    doc = jo.complete(WEATHER_REQUEST, router=router, parallel_tool_calls=True)
    assert len(doc["choices"][0]["message"]["tool_calls"]) == 1


# --------------------------------------------------------------------------------------------------------------------
# drop-in loops: role:tool messages are observations
# --------------------------------------------------------------------------------------------------------------------


def test_tool_messages_become_observations_and_the_loop_finishes() -> None:
    router, backend = weather_router(loop_script)
    messages: list[dict[str, Any]] = [{"role": "user", "content": WEATHER_REQUEST}]
    first = jo.complete(messages, router=router)["choices"][0]["message"]
    call = first["tool_calls"][0]
    messages += [first, {"role": "tool", "tool_call_id": call["id"], "content": '{"temp": 61, "unit": "F"}'}]
    second = jo.complete(messages, router=router)
    assert second["choices"][0]["message"]["x_jev"]["outcome"] == "done"
    assert second["choices"][0]["finish_reason"] == "stop"
    state = backend.requests[1].state
    assert isinstance(state, dict) and state["observations"][0]["tool"] == "get_weather"
    assert state["progress"][0].startswith('Step 1: get_weather(city="Zurich"')
    assert "DONE" in tool_options(backend.requests[1])


# --------------------------------------------------------------------------------------------------------------------
# tools of the request vs the router's catalog
# --------------------------------------------------------------------------------------------------------------------


def test_router_for_uses_the_router_for_its_own_tools_and_derives_otherwise() -> None:
    router, backend = weather_router()
    assert router_for(router, [SEARCH_TOOL, WEATHER_TOOL]) is router
    subset = router_for(router, [WEATHER_TOOL])
    assert subset is not router and subset.catalog.names == ("get_weather",)
    assert router_for(router, [WEATHER_TOOL]) is subset  # cached: click resumes find their live state
    assert subset.backend is backend and subset.policy is router.policy
    extra = {"type": "function", "function": {"name": "get_time", "description": "Get the time in a city.",
                                              "parameters": {"type": "object", "properties": {}}}}  # fmt: skip
    wider = router_for(router, [WEATHER_TOOL, extra])
    assert wider.catalog.names == ("get_weather", "get_time")
    # the router's compiled definition wins for a known tool, even if the request's copy differs
    changed = {"type": "function", "function": {**WEATHER_TOOL["function"], "description": "Weather."}}
    assert router_for(router, [changed]).catalog.get("get_weather").description == "Get the current weather for a city."


def test_complete_decides_over_the_request_tools() -> None:
    router, backend = weather_router()
    jo.complete(WEATHER_REQUEST, [WEATHER_TOOL], router=router)
    assert tool_options(backend.requests[0]) == ["get_weather", "NO_TOOL", "UNSUPPORTED"]


def test_context_mapping_is_laid_over_the_router_context() -> None:
    router, backend = weather_router()
    jo.complete(WEATHER_REQUEST, router=router, context={"user": {"name": "Ada"}, "tz": "America/New_York"})
    state = backend.requests[0].state
    assert isinstance(state, dict) and state["user"] == {"name": "Ada"} and "America/New_York" in state["now"]
    with pytest.raises(ValueError, match="unsupported context field"):
        jo.complete(WEATHER_REQUEST, router=router, context={"sources": []})


# --------------------------------------------------------------------------------------------------------------------
# wrap
# --------------------------------------------------------------------------------------------------------------------


class _FakeCompletions:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> str:
        self.calls.append(kwargs)
        return "from-the-real-model"


class _FakeAsyncCompletions:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def create(self, **kwargs: Any) -> str:
        self.calls.append(kwargs)
        return "from-the-real-model"


class _FakeClient:
    def __init__(self, completions: Any) -> None:
        self.chat = type("Chat", (), {"completions": completions})()
        self.api_key = "sk-test-not-used"
        self.models = "models-endpoint"


def test_wrap_passes_other_models_through() -> None:
    router, backend = weather_router()
    inner = _FakeCompletions()
    client = jo.wrap(_FakeClient(inner), router)
    assert client.chat.completions.create(model="gpt-4o", messages=[]) == "from-the-real-model"
    assert inner.calls == [{"model": "gpt-4o", "messages": []}] and backend.requests == []
    assert client.models == "models-endpoint"  # other attributes pass through too


@pytest.fixture(params=["attrdict", "sdk"])
def response_shape(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> str:
    """Both shapes ``wrap`` returns (docs/DECISIONS.md, OpenAI adapter): ``attrdict`` hides the SDK (the fallback
    :class:`~jevtools.adapters.openai.AttrDict`); ``sdk`` needs it (``openai.types.chat.ChatCompletion``, the shape
    users of ``wrap(OpenAI(), …)`` get; installed by the ``dev`` extra)."""
    if request.param == "attrdict":
        monkeypatch.setitem(sys.modules, "openai.types.chat", None)  # import_module raises ImportError
    else:
        pytest.importorskip("openai.types.chat")
    return str(request.param)


def _shape_of(response: Any) -> str:
    return "attrdict" if isinstance(response, jo.AttrDict) else "sdk"


def test_wrap_intercepts_its_model(response_shape: str) -> None:
    router, _ = weather_router()
    inner = _FakeCompletions()
    client = jo.wrap(_FakeClient(inner), router)
    resp = client.chat.completions.create(model="jevtools", messages=[{"role": "user", "content": WEATHER_REQUEST}],
                                          tools=[WEATHER_TOOL, SEARCH_TOOL], temperature=0)  # fmt: skip
    assert inner.calls == [] and _shape_of(resp) == response_shape
    assert resp.choices[0].finish_reason == "tool_calls"
    assert resp.choices[0].message.tool_calls[0].function.name == "get_weather"
    # attribute access and model_dump() read both shapes; subscripting works on the AttrDict fallback only
    assert resp.usage.x_jev["jev_calls"] == 1 and resp.model_dump()["usage"]["x_jev"]["jev_calls"] == 1
    assert resp.model_dump()["model"] == "jevtools"


def test_wrap_reads_the_context_from_extra_body() -> None:
    router, backend = weather_router()
    client = jo.wrap(_FakeClient(_FakeCompletions()), router, model_name="jev")
    client.chat.completions.create(model="jev", messages=[{"role": "user", "content": WEATHER_REQUEST}],
                                   extra_body={"jevtools": {"context": {"user": {"name": "Ada"}}}})  # fmt: skip
    state = backend.requests[0].state
    assert isinstance(state, dict) and state["user"] == {"name": "Ada"}


def test_wrap_streams_two_chunks(response_shape: str) -> None:
    router, _ = weather_router()
    client = jo.wrap(_FakeClient(_FakeCompletions()), router)
    chunks = list(client.chat.completions.create(model="jevtools", stream=True,
                                                 messages=[{"role": "user", "content": WEATHER_REQUEST}]))  # fmt: skip
    assert [c.object for c in chunks] == ["chat.completion.chunk", "chat.completion.chunk"]
    assert [_shape_of(c) for c in chunks] == [response_shape, response_shape]
    assert chunks[0].choices[0].delta.tool_calls[0].index == 0
    assert chunks[1].choices[0].finish_reason == "tool_calls" and chunks[1].usage.x_jev["jev_calls"] == 1


async def test_wrap_async_client() -> None:
    router, _ = weather_router()
    inner = _FakeAsyncCompletions()
    client = jo.wrap(_FakeClient(inner), router)
    assert await client.chat.completions.create(model="other", messages=[]) == "from-the-real-model"
    resp = await client.chat.completions.create(model="jevtools",
                                                messages=[{"role": "user", "content": WEATHER_REQUEST}])  # fmt: skip
    assert resp.choices[0].message.x_jev["outcome"] == "execute"
    stream = await client.chat.completions.create(model="jevtools", stream=True,
                                                  messages=[{"role": "user", "content": WEATHER_REQUEST}])  # fmt: skip
    assert len([c async for c in stream]) == 2


def _required_args(func: Any) -> Any:
    """Like openai's ``@required_args(...)``: a plain sync ``functools.wraps`` wrapper, also around async ``create``."""

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        return func(*args, **kwargs)

    return wrapper


class _DecoratedAsyncCompletions(_FakeAsyncCompletions):
    """``AsyncCompletions`` as the openai SDK defines it: ``create`` is async under a sync decorator."""

    @_required_args
    async def create(self, **kwargs: Any) -> str:
        self.calls.append(kwargs)
        return "from-the-real-model"


async def test_wrap_detects_a_decorated_async_create() -> None:
    router, _ = weather_router()
    client = jo.wrap(_FakeClient(_DecoratedAsyncCompletions()), router)
    assert client.chat.completions._async
    resp = await client.chat.completions.create(model="jevtools",
                                                messages=[{"role": "user", "content": WEATHER_REQUEST}])  # fmt: skip
    assert resp.choices[0].message.x_jev["outcome"] == "execute"
    stream = await client.chat.completions.create(model="jevtools", stream=True,
                                                  messages=[{"role": "user", "content": WEATHER_REQUEST}])  # fmt: skip
    assert len([c async for c in stream]) == 2
    assert not jo.wrap(_FakeClient(_FakeCompletions()), router).chat.completions._async


def test_wrap_detects_the_sdk_clients() -> None:
    openai = pytest.importorskip("openai")
    router, _ = weather_router()
    kw = {"api_key": "sk-unused", "base_url": "http://127.0.0.1:9/v1"}
    assert jo.wrap(openai.AsyncOpenAI(**kw), router).chat.completions._async
    assert not jo.wrap(openai.OpenAI(**kw), router).chat.completions._async


async def test_acomplete_matches_pending_prompts() -> None:
    router, backend = scenario_router(scripts.R2)
    store = InMemoryPendingStore()
    messages = scenario_messages(scripts.R2_REQUEST, history=True)
    card = (await jo.acomplete(messages, router=router, store=store))["choices"][0]["message"]
    doc = await jo.acomplete([*messages, _echo(card), {"role": "user", "content": "1"}], router=router, store=store)
    assert doc["choices"][0]["message"]["x_jev"]["outcome"] == "execute" and len(backend.requests) == 1


# --------------------------------------------------------------------------------------------------------------------
# error mapping (§7.2.4)
# --------------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("exc", "status", "code"),
    [
        (CatalogError("get_weather: invalid x-jev"), 400, "jevtools_bad_tool"),
        (BallotError("tool", "choice.options"), 400, "jevtools_bad_tool"),
        (JevValidationError([{"loc": ["body", "state"], "msg": "bad"}]), 502, "jev_invalid_request"),
        (BackendError("HTTP 400: bad", status=400), 502, "jev_invalid_request"),
        (JevNotFound("HTTP 404", status=404), 502, "jev_invalid_request"),
        (JevAuthError("https://x: HTTP 401: Bearer sk-secret-123456789", status=401), 502, "jev_auth"),
        (BackendConfigError("no key"), 502, "jev_auth"),
        (JevRateLimited("HTTP 429", retry_after=2.5), 429, "jev_rate_limited"),
        (JevUnavailable("HTTP 503", status=503), 503, "jev_unavailable"),
        (JevUnavailable("timeout"), 503, "jev_unavailable"),
        (JevProtocolError("missing answer"), 502, "jev_protocol_error"),
        (RuntimeError("boom"), 500, "jevtools_internal"),
    ],
)
def test_error_mapping(exc: BaseException, status: int, code: str) -> None:
    error = jo.error_response(exc)
    assert (error.status, error.code) == (status, code)
    assert "sk-secret" not in json.dumps(error.body)
    if isinstance(exc, JevRateLimited):
        assert error.headers == {"Retry-After": "3"}


def test_redact_masks_keys() -> None:
    assert jo.redact("Authorization: Bearer abc.def") == "Authorization: Bearer [redacted]"
    assert jo.redact("url?key=sk-or-v1-abcdef123456&x=1") == "url?key=[redacted]&x=1"
    assert "sk-proj" not in jo.redact("sent sk-proj-ABCDEFGHIJKLMNOP")


class _FailingBackend:
    name = "failing"
    model = "jev-latest"

    def __init__(self, error: BackendError) -> None:
        self.error = error

    def decide(self, request: DecisionRequest) -> DecisionResponse:
        raise self.error

    async def adecide(self, request: DecisionRequest) -> DecisionResponse:
        raise self.error


@pytest.mark.parametrize(
    ("error", "status", "code"),
    [
        (JevAuthError("https://api: HTTP 401: invalid key sk-live-abcdefghijkl", status=401), 502, "jev_auth"),
        (JevRateLimited("https://api: HTTP 429: slow down"), 429, "jev_rate_limited"),
        (JevUnavailable("https://api: HTTP 502: bad gateway", status=502), 503, "jev_unavailable"),
        (JevUnavailable("connect timeout"), 503, "jev_unavailable"),
    ],
)
def test_fail_closed_decisions_map_to_errors(error: BackendError, status: int, code: str) -> None:
    router = Router([WEATHER_TOOL], backend=_FailingBackend(error), context=weather_context())
    doc = jo.complete(WEATHER_REQUEST, router=router)
    assert "tool_calls" not in doc["choices"][0]["message"]  # never a call on an error path
    decision = router.decide(WEATHER_REQUEST)
    mapped = jo.decision_error(decision)
    assert mapped is not None and (mapped.status, mapped.code) == (status, code)
    assert "sk-live" not in json.dumps(mapped.body)


def test_decision_error_is_none_for_served_decisions() -> None:
    router, _ = weather_router()
    assert jo.decision_error(router.decide(WEATHER_REQUEST)) is None


def test_without_a_router_a_reply_is_recompiled_safely() -> None:
    from jevtools.backends.scripted import ScriptedBackend
    from tests.scenario.fixtures import scenario_context, scenario_tools

    backend = ScriptedBackend(scripts.R2)
    ctx = scenario_context()
    messages = scenario_messages(scripts.R2_REQUEST, history=True)
    card = jo.complete(messages, scenario_tools(), backend=backend, context=ctx)["choices"][0]["message"]
    assert card["x_jev"]["outcome"] == "confirm"
    doc = jo.complete([*messages, _echo(card), {"role": "user", "content": "ok"}], scenario_tools(), backend=backend,
                      context=ctx)  # fmt: skip
    # a fresh router has no in-memory state for the click: the pending is re-compiled (one more Jev round) over the
    # conversation with the card and the reply as evidence; this script then asks for the subject (never executes)
    assert len(backend.requests) == 2 and "tool_calls" not in doc["choices"][0]["message"]
    state = backend.requests[1].state
    assert isinstance(state, dict) and state["request"] == "ok" and state["history"][-1]["text"] == card["content"]
