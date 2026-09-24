"""The OpenAI-compatible proxy (spec §7.2.4, §10.6) with Starlette's TestClient and scripted backends: execute,
pending CONFIRM resumed by prefix hash and by ``pending_id``, expiry → fresh compile, the error-mapping table,
``fallback_llm`` pass-through with x-jev stripped, streaming and per-request context/sources."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from typing import Any

import httpx
import pytest

pytest.importorskip("starlette")

from starlette.testclient import TestClient  # noqa: E402

from jevtools.adapters.pending import InMemoryPendingStore  # noqa: E402
from jevtools.backends.errors import (  # noqa: E402
    BackendError,
    JevAuthError,
    JevRateLimited,
    JevUnavailable,
    JevValidationError,
)
from jevtools.backends.scripted import ScriptedBackend  # noqa: E402
from jevtools.serve import app as serve_app  # noqa: E402
from jevtools.serve.app import create_app  # noqa: E402
from jevtools.serve.config import ServeConfig  # noqa: E402
from jevtools.wire import DecisionRequest, DecisionResponse  # noqa: E402
from tests.scenario import scripts  # noqa: E402
from tests.scenario.fixtures import SCENARIO_CONTACTS, scenario_messages, scenario_tools  # noqa: E402
from tests.serve.support import SCENARIO_CONTEXT, scenario_config  # noqa: E402

MODEL = "~typesafe/jev-latest"
FALLBACK = {"base_url": "https://llm.example/v1", "model": "gpt-fallback", "api_key": "sk-fallback-key"}


class Failing:
    """A backend that always raises ``exc``."""

    model = MODEL
    name = "failing"

    def __init__(self, exc: BackendError) -> None:
        self.exc = exc
        self.calls = 0

    def decide(self, request: DecisionRequest) -> DecisionResponse:
        self.calls += 1
        raise self.exc

    async def adecide(self, request: DecisionRequest) -> DecisionResponse:
        return self.decide(request)


class FallbackLLM:
    """An OpenAI-compatible endpoint behind ``httpx.MockTransport`` that records what it receives."""

    def __init__(self) -> None:
        self.requests: list[httpx.Request] = []
        self.transport = httpx.MockTransport(self.handle)

    def handle(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        return httpx.Response(200, json={
            "id": "chatcmpl-llm", "object": "chat.completion", "created": 1, "model": "gpt-fallback",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "LLM answer"},
                         "finish_reason": "stop"}],
        })  # fmt: skip

    def body(self, i: int = -1) -> dict[str, Any]:
        return json.loads(self.requests[i].content)  # type: ignore[no-any-return]


def client(script: Any = scripts.R2, *, config: ServeConfig | None = None, **kw: Any) -> tuple[TestClient, Any]:
    backend = kw.pop("backend", None) or ScriptedBackend(script, model=MODEL)
    app = create_app(config or scenario_config(), backend=backend, **kw)
    return TestClient(app), backend


def chat(c: TestClient, messages: list[dict[str, Any]], **body: Any) -> httpx.Response:
    return c.post("/v1/chat/completions", json={"model": "jevtools", "messages": messages,
                                                "tools": scenario_tools(), **body})  # fmt: skip


def message(response: httpx.Response) -> dict[str, Any]:
    assert response.status_code == 200, response.text
    return response.json()["choices"][0]["message"]  # type: ignore[no-any-return]


def test_models_and_health() -> None:
    c, _ = client()
    assert c.get("/v1/models").json()["data"][0]["id"] == "jevtools"
    health = c.get("/healthz").json()
    assert health["status"] == "ok" and health["backend"] == "scripted" and health["model"] == MODEL
    assert health["policy"] == "jevtools-default-0.1" and health["fallback"] is False


def test_execute_path() -> None:
    c, backend = client(scripts.R1)
    response = chat(c, scenario_messages(scripts.R1_REQUEST))
    doc = response.json()
    msg = message(response)
    assert doc["object"] == "chat.completion" and doc["choices"][0]["finish_reason"] == "tool_calls"
    [call] = msg["tool_calls"]
    assert call["function"]["name"] == "get_weather"
    assert json.loads(call["function"]["arguments"]) == {"city": "Zurich", "unit": "fahrenheit"}
    assert msg["x_jev"]["outcome"] == "execute" and msg["x_jev"]["idempotency_key"].startswith("idem_")
    assert doc["usage"]["prompt_tokens"] > 0 and doc["usage"]["completion_tokens"] == 0
    assert len(backend.requests) == 1


def test_pending_confirm_resumed_by_prefix_hash() -> None:
    c, backend = client(scripts.R2)
    history = scenario_messages(scripts.R2_REQUEST, history=True)
    card = message(chat(c, history))
    assert card["x_jev"]["outcome"] == "confirm" and card["content"].startswith("Send an email")
    echoed = {"role": "assistant", "content": card["content"]}  # a plain client drops x_jev
    done = message(chat(c, [*history, echoed, {"role": "user", "content": "yes"}]))
    assert done["x_jev"]["outcome"] == "execute" and done["tool_calls"][0]["function"]["name"] == "send_email"
    assert len(backend.requests) == 1  # the click cost no Jev call
    message(chat(c, [*history, echoed, {"role": "user", "content": "yes"}]))
    assert len(backend.requests) == 2  # a consumed handle is forgotten: the same request compiles afresh


def test_pending_confirm_resumed_by_pending_id() -> None:
    c, backend = client(scripts.R2)
    history = scenario_messages(scripts.R2_REQUEST, history=True)
    card = message(chat(c, history))
    edited = {"role": "assistant", "content": "(card shown)", "x_jev": card["x_jev"]}  # text changed: no prefix hit
    done = message(chat(c, [*history, edited, {"role": "user", "content": "ok"}]))
    assert done["x_jev"]["outcome"] == "execute" and len(backend.requests) == 1

    c2, backend2 = client(scripts.R2)
    card2 = message(chat(c2, history))
    plain = {"role": "assistant", "content": "(card shown)"}
    done2 = message(chat(c2, [*history, plain, {"role": "user", "content": "ok"}],
                         jevtools={"pending_id": card2["x_jev"]["pending_id"]}))  # fmt: skip
    assert done2["x_jev"]["outcome"] == "execute" and len(backend2.requests) == 1


def test_expired_pending_compiles_a_fresh_turn() -> None:
    clock = [datetime.now(timezone.utc)]
    store = InMemoryPendingStore(clock=lambda: clock[0])
    c, backend = client(scripts.R2, store=store)
    history = scenario_messages(scripts.R2_REQUEST, history=True)
    card = message(chat(c, history))
    assert len(store) == 2  # under the pending id and the prefix key
    clock[0] += timedelta(minutes=16)
    reply = message(chat(c, [*history, {"role": "assistant", "content": card["content"]},
                             {"role": "user", "content": "yes"}]))  # fmt: skip
    assert len(backend.requests) == 2  # a miss compiles the conversation again (one more round)
    assert reply["x_jev"]["trace_id"] != card["x_jev"]["trace_id"]


def error_of(response: httpx.Response) -> dict[str, Any]:
    body = response.json()
    assert "choices" not in body  # a tool call is never produced on an error path
    return body["error"]  # type: ignore[no-any-return]


@pytest.mark.parametrize(
    ("exc", "status", "code"),
    [
        (JevAuthError("https://api.typesafe.ai/v1/systemone: HTTP 401: bad key sk-live-abcdef123456", status=401),
         502, "jev_auth"),
        (JevUnavailable("https://api.typesafe.ai/v1/systemone: HTTP 503: down", status=503), 503, "jev_unavailable"),
        (JevUnavailable("timeout after retries"), 503, "jev_unavailable"),
        (JevValidationError([{"loc": ["body", "state"], "msg": "too long", "type": "value_error"}]), 502,
         "jev_invalid_request"),
        (BackendError("https://api.typesafe.ai/v1/systemone: HTTP 400: bad", status=400), 502,
         "jev_invalid_request"),
    ],
)  # fmt: skip
def test_error_mapping(exc: BackendError, status: int, code: str) -> None:
    c, _ = client(backend=Failing(exc))
    response = chat(c, scenario_messages(scripts.R1_REQUEST))
    assert response.status_code == status
    error = error_of(response)
    assert error["code"] == code and "sk-live-abcdef123456" not in response.text


def test_rate_limit_passes_retry_after() -> None:
    c, _ = client(backend=Failing(JevRateLimited("HTTP 429: slow down", retry_after=7)))
    response = chat(c, scenario_messages(scripts.R1_REQUEST))
    assert response.status_code == 429 and response.headers["retry-after"] == "7"
    assert error_of(response)["code"] == "jev_rate_limited"


def test_bad_tools_and_bad_requests() -> None:
    c, backend = client()
    bad_tool = {"type": "function", "function": {"name": "t", "parameters": {"type": "object", "properties": {
        "a": {"type": "string", "x-jev": {"no_such_key": 1}}}}}}  # fmt: skip
    response = c.post("/v1/chat/completions", json={"model": "jevtools", "messages": [{"role": "user",
                                                    "content": "hi"}], "tools": [bad_tool]})  # fmt: skip
    assert response.status_code == 400 and error_of(response)["code"] == "jevtools_bad_tool"
    malformed = c.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "hi"}],
                                                     "tools": [{"type": "function"}]})  # fmt: skip
    assert malformed.status_code == 400 and error_of(malformed)["code"] == "jevtools_bad_tool"
    assert c.post("/v1/chat/completions", content=b"{nope").json()["error"]["code"] == "invalid_json"
    assert c.post("/v1/chat/completions", json={"messages": []}).json()["error"]["code"] == "missing_messages"
    bad_ctx = chat(c, scenario_messages(scripts.R1_REQUEST), jevtools={"context": {"sources": []}})
    assert bad_ctx.status_code == 400 and error_of(bad_ctx)["code"] == "jevtools_bad_context"
    bad_src = chat(c, scenario_messages(scripts.R1_REQUEST), jevtools={"sources": {"contacts": "x"}})
    assert bad_src.status_code == 400 and error_of(bad_src)["code"] == "jevtools_bad_sources"
    assert chat(c, scenario_messages("hi"), jevtools=[1]).json()["error"]["code"] == "jevtools_bad_extra"
    assert backend.requests == []


def test_fallback_on_jev_failure_strips_x_jev() -> None:
    llm = FallbackLLM()
    config = scenario_config(fallback_llm=FALLBACK)
    c, failing = client(config=config, backend=Failing(JevUnavailable("HTTP 503", status=503)),
                        fallback_transport=llm.transport)  # fmt: skip
    history = [*scenario_messages(scripts.R2_REQUEST, history=True)]
    history[1] = {**history[1], "x_jev": {"outcome": "confirm", "pending_id": "pnd_x"}}
    response = chat(c, history, jevtools={"context": {"locale": "en-CH"}}, temperature=0.1)
    msg = message(response)
    assert msg["content"] == "LLM answer" and msg["x_jev"] == {"outcome": "fallback", "reason": "jev_unavailable"}
    forwarded = llm.body()
    assert "jevtools" not in forwarded and forwarded["model"] == "gpt-fallback" and forwarded["temperature"] == 0.1
    assert "x-jev" not in json.dumps(forwarded["tools"]) and "x_jev" not in json.dumps(forwarded["messages"])
    assert [t["function"]["name"] for t in forwarded["tools"]] == [t["function"]["name"] for t in scenario_tools()]
    assert llm.requests[0].headers["authorization"] == "Bearer sk-fallback-key"
    assert llm.requests[0].url == "https://llm.example/v1/chat/completions" and failing.calls >= 1


def test_fallback_on_bad_tools_and_unsupported() -> None:
    llm = FallbackLLM()
    c, _ = client(scripts.R7, config=scenario_config(fallback_llm=FALLBACK), fallback_transport=llm.transport)
    bad = c.post("/v1/chat/completions", json={"model": "jevtools", "messages": [{"role": "user", "content": "hi"}],
                                               "tools": [{"type": "function"}]})  # fmt: skip
    assert message(bad)["x_jev"]["reason"] == "bad_tool"
    unsupported = {**scripts.R7, "tool": {"UNSUPPORTED": 0.9, "NO_TOOL": 0.05, "search_web": 0.05}}
    c2, _ = client(unsupported, config=scenario_config(fallback_llm=FALLBACK), fallback_transport=llm.transport)
    assert message(chat(c2, scenario_messages("Book me a flight to Rome")))["x_jev"]["reason"] == "unsupported"
    c3, _ = client(scripts.R7, config=scenario_config(fallback_llm=FALLBACK), fallback_transport=llm.transport)
    abstain = message(chat(c3, scenario_messages(scripts.R7_REQUEST)))
    assert abstain["x_jev"]["outcome"] == "abstain" and len(llm.requests) == 2  # NO_TOOL is served, not forwarded


def test_fallback_failure_returns_the_mapped_error() -> None:
    def down(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("unreachable")

    c, _ = client(config=scenario_config(fallback_llm=FALLBACK), backend=Failing(JevUnavailable("HTTP 503")),
                  fallback_transport=httpx.MockTransport(down))  # fmt: skip
    response = chat(c, scenario_messages(scripts.R1_REQUEST))
    assert response.status_code == 503 and error_of(response)["code"] == "jev_unavailable"


def test_streaming() -> None:
    c, _ = client(scripts.R1)
    with c.stream("POST", "/v1/chat/completions", json={"model": "jevtools", "stream": True,
                                                        "messages": scenario_messages(scripts.R1_REQUEST),
                                                        "tools": scenario_tools()}) as response:  # fmt: skip
        assert response.headers["content-type"].startswith("text/event-stream")
        lines = [line for line in response.iter_lines() if line.startswith("data: ")]
    assert lines[-1] == "data: [DONE]"
    first, last = (json.loads(line[6:]) for line in lines[:2])
    assert first["object"] == "chat.completion.chunk" and first["choices"][0]["delta"]["tool_calls"][0]["index"] == 0
    assert last["choices"][0]["finish_reason"] == "tool_calls"


def test_per_request_context_and_sources() -> None:
    config = ServeConfig.from_dict({"backend": "simulator", "context": SCENARIO_CONTEXT})
    c, _ = client(scripts.R2, config=config)
    contacts = {"rows": SCENARIO_CONTACTS, "key": "email", "label": "{name} <{email}>", "describe": "{notes}",
                "match": ["name", "aliases", "team"], "provides": ["email", "person"]}  # fmt: skip
    history = scenario_messages(scripts.R2_REQUEST, history=True)
    card = message(chat(c, history, jevtools={"sources": {"contacts": contacts}, "conversation_id": "conv-1",
                                              "context": {"user": {"name": "Sam Muster"}}}))  # fmt: skip
    assert card["x_jev"]["outcome"] == "confirm" and "anna.keller@acme.com" in card["content"]
    state = c.app.state.jevtools  # type: ignore[attr-defined]
    assert len(state._routers) == 1
    message(chat(c, history, jevtools={"sources": {"contacts": contacts}}))
    assert len(state._routers) == 1  # the same sources reuse the cached base router


def test_run_starts_uvicorn(monkeypatch: pytest.MonkeyPatch, tmp_path: Any) -> None:
    uvicorn = pytest.importorskip("uvicorn")

    seen: dict[str, Any] = {}
    monkeypatch.setattr(uvicorn, "run", lambda app, **kw: seen.update(app=app, **kw))
    path = tmp_path / "jevtools.toml"
    path.write_text('backend = "simulator"\nmodel_name = "jt"\n', encoding="utf-8")
    serve_app.run(path, host="0.0.0.0", port=9999)
    assert seen["host"] == "0.0.0.0" and seen["port"] == 9999
    assert seen["app"].state.jevtools.config.model_name == "jt"
    from jevtools import serve

    assert serve.run is serve_app.run and serve.create_app is create_app
