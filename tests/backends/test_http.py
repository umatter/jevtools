"""HTTP backends over ``httpx.MockTransport`` (spec §8.2, §8.4, §10.4): URLs, headers, model ids, parsing, retries and
typed errors."""

from __future__ import annotations

import json
import math
from collections.abc import Callable
from typing import Any

import httpx
import pytest

from jevtools.backends.errors import (
    BackendConfigError,
    BackendError,
    JevAuthError,
    JevNotFound,
    JevProtocolError,
    JevRateLimited,
    JevUnavailable,
    JevValidationError,
    error_for_status,
)
from jevtools.backends.http import (
    HTTPBackend,
    OpenRouterDecisions,
    OpenRouterSystemOne,
    TypeSafe,
    backoff,
    retry_after,
)
from jevtools.backends.scripted import ScriptedBackend
from jevtools.wire import ChoiceQuestion, DecisionRequest, NoulQuestion

REQUEST = DecisionRequest(
    model="",
    state={"request": "hi"},
    questions={"tool": ChoiceQuestion(criteria={"a": None, "NO_TOOL": "nothing"}), "t.authorized": NoulQuestion()},
)
ANSWERS = {
    "tool": {"type": "choice", "choice": "a", "confidence": 0.8, "probabilities": {"a": 0.9, "NO_TOOL": 0.1}},
    "t.authorized": {"type": "noul", "noul": 0.93},
}
DECISIONS_BODY = {"model": "jev-1.13.0", "id": "dec-1", "provider": "TypeSafe", "answers": ANSWERS,
                  "usage": {"input_tokens": 120, "output_tokens": 2, "cost": 0.000005}}  # fmt: skip

Handler = Callable[[httpx.Request], httpx.Response]


def recording(responses: list[httpx.Response]) -> tuple[Handler, list[httpx.Request]]:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return responses[min(len(seen) - 1, len(responses) - 1)]

    return handler, seen


def backend(factory: Callable[..., HTTPBackend], handler: Handler, **kw: Any) -> tuple[HTTPBackend, list[float]]:
    sleeps: list[float] = []
    transport = httpx.MockTransport(handler)
    b = factory("key-123", transport=transport, async_transport=transport, sleep=sleeps.append, **kw)
    return b, sleeps


@pytest.mark.parametrize(
    ("factory", "url", "model", "name"),
    [
        (HTTPBackend.typesafe, "https://api.typesafe.ai/v1/systemone", "jev-latest", "typesafe"),
        (HTTPBackend.openrouter, "https://openrouter.ai/api/v1/systemone", "~typesafe/jev-latest",
         "openrouter_systemone"),
        (HTTPBackend.openrouter_decisions, "https://openrouter.ai/api/alpha/decisions", "~typesafe/jev-latest",
         "openrouter_decisions"),
    ],
)  # fmt: skip
def test_urls_headers_models_and_parsing(
    factory: Callable[..., HTTPBackend], url: str, model: str, name: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("JEVTOOLS_MODEL", raising=False)
    monkeypatch.delenv("TYPESAFE_BASE_URL", raising=False)
    handler, seen = recording([httpx.Response(200, json=DECISIONS_BODY)])
    b, _ = backend(factory, handler, referer="https://example.org", title="Example")
    response = b.decide(REQUEST)
    assert (b.name, b.model) == (name, model)
    request = seen[0]
    assert str(request.url) == url
    assert request.headers["authorization"] == "Bearer key-123"
    assert request.headers["accept"] == "application/json"
    assert request.headers["http-referer"] == "https://example.org" and request.headers["x-title"] == "Example"
    body = json.loads(request.content)
    assert body["model"] == model and list(body["questions"]) == ["tool", "t.authorized"]
    assert response.usage.cost == pytest.approx(0.000005) and response.id == "dec-1"
    assert response.provider == "TypeSafe" and response.answers["t.authorized"].noul == 0.93  # type: ignore[union-attr]


def test_model_override_and_explicit_model(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("JEVTOOLS_MODEL", "typesafe/jev-1.13")
    assert OpenRouterDecisions("k").model == "typesafe/jev-1.13"
    assert OpenRouterSystemOne("k", model="custom").model == "custom"
    monkeypatch.setenv("TYPESAFE_BASE_URL", "http://localhost:9999/")
    assert TypeSafe("k").url == "http://localhost:9999/v1/systemone"


def test_missing_key_is_a_config_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(BackendConfigError):
        HTTPBackend.openrouter()
    assert issubclass(BackendConfigError, BackendError)


def test_client_is_reused() -> None:
    handler, seen = recording([httpx.Response(200, json=DECISIONS_BODY)])
    b, _ = backend(HTTPBackend.typesafe, handler)
    b.decide(REQUEST)
    client = b.client
    b.decide(REQUEST)
    assert b.client is client and len(seen) == 2
    b.close()


def test_422_is_typed_with_qids() -> None:
    detail = [{"loc": ["body", "questions", "t.authorized", "noul", "criteria"], "msg": "bad", "type": "value_error"}]
    handler, _ = recording([httpx.Response(422, json={"detail": detail})])
    b, sleeps = backend(HTTPBackend.typesafe, handler)
    with pytest.raises(JevValidationError) as info:
        b.decide(REQUEST)
    assert info.value.qids() == {"t.authorized"} and not info.value.request_level() and sleeps == []
    state_error = JevValidationError([{"loc": ["body", "state"], "msg": "too long"}])
    assert state_error.request_level() and state_error.qids() == set()


def test_429_honours_retry_after_ms_then_retry_after() -> None:
    handler, seen = recording([
        httpx.Response(429, headers={"retry-after-ms": "250"}, json={}),
        httpx.Response(429, headers={"retry-after": "2"}, json={}),
        httpx.Response(200, json=DECISIONS_BODY),
    ])  # fmt: skip
    b, sleeps = backend(HTTPBackend.typesafe, handler)
    assert b.decide(REQUEST).model == "jev-1.13.0"
    assert sleeps == [0.25, 2.0] and len(seen) == 3


def test_429_after_retries_is_rate_limited() -> None:
    handler, _ = recording([httpx.Response(429, headers={"retry-after": "1"}, json={"error": "slow down"})])
    b, sleeps = backend(HTTPBackend.typesafe, handler, max_retries=1)
    with pytest.raises(JevRateLimited) as info:
        b.decide(REQUEST)
    assert info.value.retry_after == 1.0 and info.value.status == 429 and sleeps == [1.0]


def test_5xx_backoff_then_unavailable() -> None:
    handler, seen = recording([httpx.Response(503, text="down")])
    b, sleeps = backend(HTTPBackend.openrouter_decisions, handler)
    with pytest.raises(JevUnavailable):
        b.decide(REQUEST)
    assert sleeps == [backoff(0), backoff(1)] == [0.5, 1.0] and len(seen) == 3
    assert backoff(10) == 5.0


def test_timeout_is_unavailable() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("slow", request=request)

    b, sleeps = backend(HTTPBackend.typesafe, handler, max_retries=1)
    with pytest.raises(JevUnavailable, match="timeout"):
        b.decide(REQUEST)
    assert sleeps == [0.5]


@pytest.mark.parametrize(("status", "error"), [(401, JevAuthError), (403, JevAuthError), (404, JevNotFound),
                                               (400, BackendError)])  # fmt: skip
def test_non_retried_statuses(status: int, error: type[BackendError]) -> None:
    handler, seen = recording([httpx.Response(status, json={"error": {"message": "nope"}})])
    b, sleeps = backend(HTTPBackend.typesafe, handler)
    with pytest.raises(error, match="nope"):
        b.decide(REQUEST)
    assert len(seen) == 1 and sleeps == []


def test_non_json_success_is_a_protocol_error() -> None:
    handler, _ = recording([httpx.Response(200, text="<html>")])
    b, _ = backend(HTTPBackend.typesafe, handler)
    with pytest.raises(JevProtocolError):
        b.decide(REQUEST)


def test_error_for_status_mapping() -> None:
    assert isinstance(error_for_status(422, {"detail": "x"}, url="u", message="m"), BackendError)
    assert isinstance(error_for_status(408, None, url="u", message="m"), JevUnavailable)
    assert isinstance(error_for_status(500, None, url="u", message="m"), JevUnavailable)


async def test_async_decide_reuses_one_client_per_loop() -> None:
    handler, seen = recording([httpx.Response(503, json={}), httpx.Response(200, json=DECISIONS_BODY)])
    sleeps: list[float] = []

    async def asleep(delay: float) -> None:
        sleeps.append(delay)

    transport = httpx.MockTransport(handler)
    b = HTTPBackend.typesafe("k", async_transport=transport, asleep=asleep)
    response = await b.adecide(REQUEST)
    assert response.usage.input_tokens == 120 and sleeps == [0.5] and len(seen) == 2
    client = b._aclient()
    await b.adecide(REQUEST)
    assert b._aclient() is client
    await b.aclose()


# -- review-edges regressions -----------------------------------------------------------------------------------------


@pytest.mark.parametrize("raw", ["inf", "1e400", "Infinity", "nan", "-inf"])
def test_non_finite_retry_hints_fall_back_to_backoff(raw: str) -> None:
    handler, seen = recording([httpx.Response(429, headers={"retry-after": raw}, json={}),
                               httpx.Response(200, json=DECISIONS_BODY)])  # fmt: skip
    b, sleeps = backend(HTTPBackend.typesafe, handler)
    assert b.decide(REQUEST).model == "jev-1.13.0" and sleeps == [backoff(0)] and len(seen) == 2
    assert retry_after(httpx.Response(429, headers={"retry-after": raw})) is None


@pytest.mark.parametrize("headers", [{"retry-after": "86400"}, {"retry-after-ms": "3600000"}])
def test_a_retry_hint_above_the_cap_fails_now_and_reports_the_hint(headers: dict[str, str]) -> None:
    handler, seen = recording([httpx.Response(429, headers=headers, json={"error": "quota"})])
    b, sleeps = backend(HTTPBackend.typesafe, handler)
    with pytest.raises(JevRateLimited) as info:
        b.decide(REQUEST)
    assert sleeps == [] and len(seen) == 1 and info.value.retry_after in (86400.0, 3600.0)
    capped, _ = backend(HTTPBackend.typesafe, recording([httpx.Response(429, headers={"retry-after": "3"}, json={}),
                                                        httpx.Response(200, json=DECISIONS_BODY)])[0],
                        max_retry_wait=2.0)  # fmt: skip
    with pytest.raises(JevRateLimited):
        capped.decide(REQUEST)


async def test_async_retry_hints_are_capped_and_finite() -> None:
    slept: list[float] = []

    async def asleep(delay: float) -> None:
        slept.append(delay)

    for raw, expect in (("inf", [0.5]), ("86400", [])):
        handler, _ = recording([httpx.Response(429, headers={"retry-after": raw}, json={}),
                                httpx.Response(200, json=DECISIONS_BODY)])  # fmt: skip
        slept.clear()
        b = HTTPBackend.typesafe("k", async_transport=httpx.MockTransport(handler), asleep=asleep)
        if expect:
            assert (await b.adecide(REQUEST)).model == "jev-1.13.0"
        else:
            with pytest.raises(JevRateLimited):
                await b.adecide(REQUEST)
        assert slept == expect
        await b.aclose()


def test_a_non_finite_hint_fails_closed_through_the_router() -> None:
    from jevtools import Router

    def ping(x: str) -> str:
        """Ping."""
        return x

    sleeps: list[float] = []
    transport = httpx.MockTransport(lambda r: httpx.Response(429, headers={"retry-after": "inf"}, json={}))
    b = HTTPBackend.typesafe("k", transport=transport, sleep=sleeps.append)  # time.sleep(inf) raises OverflowError
    assert Router([ping], backend=b).decide("ping hello").rule == "P0.backend.fail_closed"
    assert sleeps and all(math.isfinite(s) and s <= 5.0 for s in sleeps)


class _KeepAlive:
    """A local HTTP/1.1 keep-alive server answering every POST with one Noul answer."""

    BODY = json.dumps({"model": "m", "answers": {"q": {"type": "noul", "noul": 0.9}}}).encode()

    def __enter__(self) -> str:
        import threading
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

        body = self.BODY

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def do_POST(self) -> None:  # noqa: N802 - http.server API
                self.rfile.read(int(self.headers["Content-Length"]))
                self.send_response(200)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args: Any) -> None:
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        return f"http://127.0.0.1:{self.server.server_address[1]}/v1/systemone"

    def __exit__(self, *exc: object) -> None:
        self.server.shutdown()
        self.server.server_close()


def test_async_clients_of_closed_loops_are_dropped_and_aclose_never_crashes() -> None:
    import asyncio

    request = DecisionRequest(model="m", state="s", questions={"q": NoulQuestion()})
    with _KeepAlive() as url:
        b = HTTPBackend(url=url, api_key="k", model="m")
        for _ in range(3):
            asyncio.run(b.adecide(request))  # each run binds a client to a loop that is then closed
        assert len(b._aclients) == 1  # the clients of closed loops were dropped, not kept for the backend's life

        async def main() -> httpx.AsyncClient:
            async with b:
                await b.adecide(request)
                return b._aclient()

        current = asyncio.run(main())  # exiting `async with` used to raise "Event loop is closed"
        assert current.is_closed and b._aclients == {}
        asyncio.run(b.aclose())  # idempotent


def test_the_sync_client_is_created_once_under_concurrent_splits(monkeypatch: pytest.MonkeyPatch) -> None:
    import time
    from concurrent.futures import ThreadPoolExecutor

    created: list[httpx.Client] = []
    real = httpx.Client

    class SlowClient(real):  # type: ignore[misc,valid-type]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            time.sleep(0.02)  # the default transport builds an SSL context: the race window of the real client
            super().__init__(*args, **kwargs)
            created.append(self)

    monkeypatch.setattr(httpx, "Client", SlowClient)
    handler, seen = recording([httpx.Response(200, json=DECISIONS_BODY)])
    b, _ = backend(HTTPBackend.typesafe, handler)
    with ThreadPoolExecutor(max_workers=4) as pool:  # Router._perform's split path
        list(pool.map(b.decide, [REQUEST] * 4))
    assert len(created) == 1 and len(seen) == 4
    b.close()
    assert all(c.is_closed for c in created)


def test_scripted_backend_has_name_and_model() -> None:
    scripted = ScriptedBackend(model="m", name="fixture")
    assert (scripted.name, scripted.model) == ("fixture", "m")
    assert ScriptedBackend().name == "scripted"
