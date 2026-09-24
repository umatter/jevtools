"""``jevtools serve``: an OpenAI-compatible proxy (spec §7.2.4), so any language (R on day one) can use jevtools by
pointing an OpenAI SDK's ``base_url`` at it.

```
POST /v1/chat/completions   OpenAI Chat Completions with tools
GET  /v1/models             lists the configured model name ("jevtools")
GET  /healthz
```

- **Per-request context**: ``extra_body={"jevtools": {"context": {...}, "sources": {"contacts": [...rows]},
  "conversation_id": "…", "pending_id": "…"}}`` (the OpenAI SDK merges ``extra_body`` into the JSON body).
  ``context`` overrides ``now``/``tz``/``locale``/``user``/``shareable``/``include_system``; ``sources`` adds
  registries for this request (rows of a configured source reuse its settings).
- **Pending CONFIRM/CLARIFY** live server-side in a :class:`~jevtools.adapters.pending.PendingStore` (in memory by
  default, TTL 15 min), keyed by ``x_jev.pending_id`` and by the prefix hash of the conversation up to the prompt.
  The next request resumes it (a click — ``yes``, an option number or text — costs no Jev call; free text runs a
  resume round). A miss (expiry, restart, another worker) compiles a fresh turn, which is still correct at the cost
  of one round.
- **Graceful degradation**: with ``fallback_llm`` configured, a request jevtools cannot serve — a bad tool schema,
  Jev failure (P0), or a decision the policy would escalate while no escalator is configured (``UNSUPPORTED``, a
  diffuse tool or slot) — is forwarded unchanged to it with ``x-jev`` stripped; the answer is marked
  ``x_jev.outcome = "fallback"``.
- **Error mapping** without a fallback (§7.2.4, :func:`jevtools.adapters.openai.error_response`): bad tool 400
  ``jevtools_bad_tool``; Jev 400/422 → 502 ``jev_invalid_request``; 401/403 → 502 ``jev_auth`` (never the key);
  429 → 429 with ``Retry-After``; 5xx/timeout → 503 ``jev_unavailable``. A tool call is never produced on an error
  path.
- ``stream: true`` is answered as server-sent events: one chunk with the whole message, one with the finish reason.
"""

from __future__ import annotations

import json
import os
import threading
from collections import OrderedDict
from collections.abc import AsyncIterator, Mapping, Sequence
from contextvars import ContextVar
from datetime import timedelta
from typing import Any

import httpx
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response, StreamingResponse
from starlette.routing import Route

from jevtools._version import SPEC_VERSION, __version__
from jevtools.adapters._router import merge_context, router_for
from jevtools.adapters.openai import (
    ErrorResponse,
    api_error,
    chat_completion,
    completion_chunks,
    decision_error,
    error_response,
)
from jevtools.adapters.pending import InMemoryPendingStore, PendingStore, adecide_turn
from jevtools.backends.base import Backend
from jevtools.backends.errors import BackendError
from jevtools.canonical import jsonable, sha256_of
from jevtools.confidence import IsotonicCalibrator
from jevtools.decision import Decision
from jevtools.fallback import Escalator, Filler, TextLLM
from jevtools.policy import RULE_FAIL_CLOSED, RULE_UNSUPPORTED, Outcome, Policy
from jevtools.router import Router
from jevtools.serve.config import ServeConfig, request_sources
from jevtools.spec.catalog import strip_xjev

FALLBACK = "fallback"
"""``x_jev.outcome`` of an answer produced by ``fallback_llm``."""
_CAPTURED: ContextVar[list[BackendError] | None] = ContextVar("jevtools_serve_backend_errors", default=None)


# --------------------------------------------------------------------------------------------------------------------
# Backend error capture
# --------------------------------------------------------------------------------------------------------------------


class CapturingBackend:
    """Wraps the Jev backend and records the typed errors of the current request.

    The router fails closed on a backend error (rule P0) and keeps only its text in the trace; the proxy needs the
    exception itself for the error mapping (``Retry-After`` of a 429, the status of a 401/403).
    """

    def __init__(self, inner: Backend) -> None:
        self.inner = inner
        self.model = inner.model
        self.name = inner.name

    @staticmethod
    def _record(exc: BackendError) -> None:
        errors = _CAPTURED.get()
        if errors is not None:
            errors.append(exc)

    def decide(self, request: Any) -> Any:
        try:
            return self.inner.decide(request)
        except BackendError as exc:
            self._record(exc)
            raise

    async def adecide(self, request: Any) -> Any:
        try:
            return await self.inner.adecide(request)
        except BackendError as exc:
            self._record(exc)
            raise


# --------------------------------------------------------------------------------------------------------------------
# State
# --------------------------------------------------------------------------------------------------------------------


def _error_json(err: ErrorResponse) -> JSONResponse:
    return JSONResponse(err.body, status_code=err.status, headers=err.headers or None)


class ServeState:
    """Everything the proxy holds: configuration, backend, policy, LLM endpoints, the pending store and the base
    routers (one per distinct set of per-request sources; tool sets are derived from them)."""

    def __init__(
        self,
        config: ServeConfig,
        *,
        backend: Backend | None = None,
        policy: Policy | None = None,
        store: PendingStore | None = None,
        calibrators: Mapping[str, IsotonicCalibrator] | None = None,
        text_llm: TextLLM | None = None,
        filler: Filler | None = None,
        escalator: Escalator | None = None,
        fallback_transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.config = config
        self.backend = CapturingBackend(backend if backend is not None else config.build_backend())
        self.policy = policy if policy is not None else config.build_policy()
        self.store: PendingStore = store if store is not None else InMemoryPendingStore(
            ttl=timedelta(seconds=config.pending_ttl_s))  # fmt: skip
        self.calibrators = dict(calibrators) if calibrators is not None else config.build_calibrators()
        self.text_llm = text_llm if text_llm is not None else config.build_text_llm()
        self.filler = filler if filler is not None else config.build_filler()
        self.escalator = escalator if escalator is not None else config.build_escalator()
        self.fallback_transport = fallback_transport
        self.sources = config.build_sources()
        self._routers: OrderedDict[str, Router] = OrderedDict()
        self._lock = threading.Lock()

    def base_router(self, extra: Mapping[str, Any] | None = None) -> Router:
        """The base router for a request's sources (configured sources, replaced by the request's of the same
        name), cached by the sources' content."""
        key = sha256_of(jsonable(dict(extra or {})))
        with self._lock:
            router = self._routers.get(key)
            if router is not None:
                self._routers.move_to_end(key)
                return router
        added = request_sources(extra, self.config.sources) if extra else []
        names = {s.name for s in added}
        sources = [s for s in self.sources if s.name not in names] + added
        router = Router([], backend=self.backend, policy=self.policy, context=self.config.build_context(sources),
                        filler=self.filler, escalator=self.escalator, text_llm=self.text_llm,
                        calibrators=self.calibrators)  # fmt: skip
        with self._lock:
            router = self._routers.setdefault(key, router)
            while len(self._routers) > max(1, self.config.max_routers):
                self._routers.popitem(last=False)
        return router

    # -- degradation --------------------------------------------------------------------------------------------

    def wants_escalation(self, decision: Decision) -> bool:
        """The policy would escalate and nothing answered: Jev failure (P0), ``UNSUPPORTED`` (P2) or a diffuse
        tool or slot distribution, unless a configured escalator produced a handoff (text or a proposed call)."""
        if decision.outcome is Outcome.ESCALATE and (decision.content or decision.call is not None):
            return False
        shape = (getattr(decision.trace, "outcome", None) or {}).get("shape")
        return decision.rule in (RULE_FAIL_CLOSED, RULE_UNSUPPORTED) or decision.rule.endswith(".diffuse") \
            or shape == "diffuse"  # fmt: skip

    async def forward(self, body: Mapping[str, Any], *, reason: str) -> tuple[int, Any]:
        """Send the request unchanged (``jevtools`` extras dropped, ``x-jev`` stripped from tools and messages) to
        ``fallback_llm``; returns the status and the JSON document, marked ``x_jev.outcome = "fallback"``."""
        cfg = self.config.fallback_llm
        assert cfg is not None
        payload = {k: v for k, v in body.items() if k not in ("jevtools", "stream", "stream_options")}
        payload["messages"] = [{k: v for k, v in m.items() if k not in ("x_jev", "x-jev")} if isinstance(m, Mapping)
                               else m for m in body.get("messages") or []]  # fmt: skip
        if body.get("tools") is not None:
            payload["tools"] = strip_xjev(body["tools"])
        if cfg.model:
            payload["model"] = cfg.model
        payload.update(cfg.options)
        headers = {"Content-Type": "application/json", "Accept": "application/json", **cfg.headers}
        key = cfg.key()
        if key:
            headers["Authorization"] = f"Bearer {key}"
        async with httpx.AsyncClient(timeout=cfg.timeout, transport=self.fallback_transport) as client:
            response = await client.post(cfg.url, json=payload, headers=headers)
        doc = response.json()
        if response.is_success and isinstance(doc, dict):
            for choice in doc.get("choices") or []:
                message = choice.get("message") if isinstance(choice, dict) else None
                if isinstance(message, dict):
                    message["x_jev"] = {"outcome": FALLBACK, "reason": reason}
        return response.status_code, doc


# --------------------------------------------------------------------------------------------------------------------
# Responses
# --------------------------------------------------------------------------------------------------------------------


def _sse(chunks: Sequence[Mapping[str, Any]]) -> StreamingResponse:
    async def events() -> AsyncIterator[str]:
        for chunk in chunks:
            yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(events(), media_type="text/event-stream")


def _respond(doc: Mapping[str, Any], *, stream: bool, status: int = 200) -> Response:
    if stream and status == 200 and doc.get("choices"):
        head = {"id": doc.get("id") or "chatcmpl-fallback", "created": doc.get("created") or 0,
                "model": doc.get("model"), "system_fingerprint": doc.get("system_fingerprint")}  # fmt: skip
        choice = dict(doc["choices"][0])
        choice.setdefault("finish_reason", "stop")
        return _sse(completion_chunks({**doc, **head, "choices": [choice]}))
    return JSONResponse(dict(doc), status_code=status)


async def _degrade(state: ServeState, body: Mapping[str, Any], err: ErrorResponse | None, *, reason: str,
                   stream: bool) -> Response:  # fmt: skip
    """Forward to ``fallback_llm`` when configured, else return ``err`` (a fallback failure also returns ``err``,
    or a 502 ``fallback_unavailable``)."""
    if state.config.fallback_llm is not None:
        try:
            status, doc = await state.forward(body, reason=reason)
        except (httpx.HTTPError, ValueError) as exc:
            return _error_json(err or api_error(502, "upstream_error", "fallback_unavailable",
                                             f"fallback_llm failed: {type(exc).__name__}: {exc}"))  # fmt: skip
        return _respond(doc, stream=stream, status=status)
    assert err is not None
    return _error_json(err)


# --------------------------------------------------------------------------------------------------------------------
# Endpoints
# --------------------------------------------------------------------------------------------------------------------


def _state(request: Request) -> ServeState:
    return request.app.state.jevtools  # type: ignore[no-any-return]


async def chat_completions(request: Request) -> Response:
    """``POST /v1/chat/completions``."""
    state = _state(request)
    try:
        body = await request.json()
    except ValueError:
        return _error_json(api_error(400, "invalid_request_error", "invalid_json", "the request body is not JSON"))
    if not isinstance(body, dict) or not isinstance(body.get("messages"), list) or not body["messages"]:
        return _error_json(api_error(400, "invalid_request_error", "missing_messages", "`messages` must be a "
                                  "non-empty list"))  # fmt: skip
    extras = body.get("jevtools") or {}
    if not isinstance(extras, Mapping):
        return _error_json(
            api_error(400, "invalid_request_error", "jevtools_bad_extra", "`jevtools` must be an object")
        )
    stream = bool(body.get("stream"))
    try:
        base = state.base_router(extras.get("sources"))
    except (ValueError, TypeError) as exc:
        return _error_json(api_error(400, "invalid_request_error", "jevtools_bad_sources", str(exc)))
    try:
        router = router_for(base, body.get("tools") or None)
    except Exception as exc:  # noqa: BLE001 - any failure to compile the request's tools is a bad tool
        err = error_response(exc)
        if err.status != 400:
            err = api_error(400, "invalid_request_error", "jevtools_bad_tool", f"{type(exc).__name__}: {exc}")
        return await _degrade(state, body, err, reason="bad_tool", stream=stream)
    try:
        ctx = merge_context(router.context, extras.get("context"))
    except (ValueError, TypeError) as exc:
        return _error_json(api_error(400, "invalid_request_error", "jevtools_bad_context", str(exc)))
    captured: list[BackendError] = []
    token = _CAPTURED.set(captured)
    try:
        decision = await adecide_turn(
            router, body["messages"], context=ctx, tool_choice=body.get("tool_choice") or "auto",
            parallel_tool_calls=bool(body.get("parallel_tool_calls")), store=state.store,
            pending_id=extras.get("pending_id"),
        )  # fmt: skip
    except Exception as exc:  # noqa: BLE001 - mapped to an OpenAI error (or forwarded)
        return await _degrade(state, body, error_response(exc), reason="error", stream=stream)
    finally:
        _CAPTURED.reset(token)
    if state.wants_escalation(decision):
        failure: ErrorResponse | None = None
        if decision.rule == RULE_FAIL_CLOSED:
            failure = error_response(captured[0]) if captured else decision_error(decision)
        if state.config.fallback_llm is not None:
            return await _degrade(state, body, failure, reason=_reason(decision), stream=stream)
        if failure is not None:
            return _error_json(failure)
    return _respond(chat_completion(decision, model=str(body.get("model") or state.config.model_name)), stream=stream)


def _reason(decision: Decision) -> str:
    if decision.rule == RULE_FAIL_CLOSED:
        return "jev_unavailable"
    if decision.rule == RULE_UNSUPPORTED:
        return "unsupported"
    return "escalate"


async def models(request: Request) -> Response:
    """``GET /v1/models``: the configured model name."""
    name = _state(request).config.model_name
    return JSONResponse({"object": "list", "data": [{"id": name, "object": "model", "created": 0,
                                                     "owned_by": "jevtools"}]})  # fmt: skip


async def healthz(request: Request) -> Response:
    """``GET /healthz``: liveness plus what answers (backend, model, policy)."""
    state = _state(request)
    return JSONResponse({"status": "ok", "version": __version__, "spec": SPEC_VERSION, "backend": state.backend.name,
                         "model": state.backend.model, "policy": state.policy.version,
                         "fallback": state.config.fallback_llm is not None})  # fmt: skip


def create_app(
    config: ServeConfig | str | os.PathLike[str] | Mapping[str, Any] | None = None,
    *,
    backend: Backend | None = None,
    policy: Policy | None = None,
    store: PendingStore | None = None,
    calibrators: Mapping[str, IsotonicCalibrator] | None = None,
    text_llm: TextLLM | None = None,
    filler: Filler | None = None,
    escalator: Escalator | None = None,
    fallback_transport: httpx.AsyncBaseTransport | None = None,
) -> Starlette:
    """The proxy as a Starlette ASGI app.

    ``config`` is a :class:`ServeConfig`, a path to ``jevtools.toml`` or a mapping; keyword arguments replace what
    the configuration would build (tests pass a ``ScriptedBackend``; ``fallback_transport`` replaces the network of
    ``fallback_llm`` calls, e.g. an ``httpx.MockTransport``).
    """
    if config is None:
        cfg = ServeConfig()
    elif isinstance(config, ServeConfig):
        cfg = config
    elif isinstance(config, Mapping):
        cfg = ServeConfig.from_dict(config)
    else:
        cfg = ServeConfig.from_toml(config)
    app = Starlette(routes=[
        Route("/v1/chat/completions", chat_completions, methods=["POST"]),
        Route("/v1/models", models, methods=["GET"]),
        Route("/healthz", healthz, methods=["GET"]),
    ])  # fmt: skip
    app.state.jevtools = ServeState(cfg, backend=backend, policy=policy, store=store, calibrators=calibrators,
                                    text_llm=text_llm, filler=filler, escalator=escalator,
                                    fallback_transport=fallback_transport)  # fmt: skip
    return app


def run(
    config: ServeConfig | str | os.PathLike[str] | Mapping[str, Any] | None = None,
    host: str = "127.0.0.1",
    port: int = 8787,
    **uvicorn_options: Any,
) -> None:
    """Serve the proxy with uvicorn (what ``jevtools serve --config … --host … --port …`` calls); ``config`` is
    what :func:`create_app` takes (a ``jevtools.toml`` path, a :class:`ServeConfig`, a mapping or ``None``)."""
    try:
        import uvicorn
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ImportError("jevtools serve needs the `serve` extra: pip install 'jevtools[serve]'") from exc
    uvicorn.run(create_app(config), host=host, port=port, **uvicorn_options)


__all__ = [
    "FALLBACK",
    "CapturingBackend",
    "InMemoryPendingStore",
    "PendingStore",
    "ServeState",
    "chat_completions",
    "create_app",
    "healthz",
    "models",
    "run",
]
