"""OpenAI Chat Completions interop (spec §3.10, §7.2.1, §7.2.2, §7.2.4).

- :func:`complete` / :func:`acomplete`: one decision over OpenAI-style ``messages`` and ``tools``, returned as a
  ``ChatCompletion``-shaped dict whose ``choices[0].message`` is :meth:`Decision.to_openai_message
  <jevtools.decision.Decision.to_openai_message>` (``finish_reason`` ``tool_calls`` or ``stop``) and whose
  ``usage`` carries ``x_jev``.
- :func:`wrap`: a drop-in proxy around an OpenAI client (sync or async, duck-typed: no ``openai`` dependency) that
  intercepts ``client.chat.completions.create(model="jevtools", …)`` only; every other model passes through.
- ``role: tool`` messages become observations (``Context.all_observations``), which is how drop-in loops get
  multi-step behaviour. Pending CONFIRM/CLARIFY prompts are matched by ``x_jev.pending_id`` or by the prefix hash
  (:mod:`jevtools.adapters.pending`).
- ``tool_choice`` (``none``/``auto``/``required``/named) goes to the router unchanged (§7.2.2);
  ``parallel_tool_calls`` is accepted (``ext.parallel`` is not part of this version: one call is decided).
- :func:`error_response` / :func:`decision_error`: the §7.2.4 error-mapping table (used by ``jevtools serve``). A
  tool call is never produced on an error path.
"""

from __future__ import annotations

import importlib
import inspect
import math
import re
import time
from collections.abc import AsyncIterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from jevtools.adapters._router import merge_context, router_for
from jevtools.adapters.pending import (
    InMemoryPendingStore,
    MessageLike,
    PendingStore,
    adecide_turn,
    decide_turn,
    default_store,
)
from jevtools.backends.base import Backend
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
from jevtools.canonical import jsonable
from jevtools.context import Context
from jevtools.decision import Decision
from jevtools.errors import BallotError, CatalogError, ConstraintError
from jevtools.policy import RULE_FAIL_CLOSED
from jevtools.router import Router

MODEL_NAME = "jevtools"
"""The model name ``wrap`` intercepts and ``jevtools serve`` lists."""


# --------------------------------------------------------------------------------------------------------------------
# ChatCompletion documents
# --------------------------------------------------------------------------------------------------------------------


def usage_doc(decision: Decision) -> dict[str, Any]:
    """OpenAI ``usage`` extended with ``x_jev``. Jev generates no tokens, so ``completion_tokens`` is 0 and
    ``prompt_tokens`` counts the Jev input tokens."""
    tokens = decision.usage.jev_input_tokens
    return {
        "prompt_tokens": tokens,
        "completion_tokens": 0,
        "total_tokens": tokens,
        "x_jev": {**jsonable(decision.usage), "rounds": decision.rounds, "outcome": decision.outcome.value},
    }


def _created(decision: Decision) -> int:
    created = getattr(decision.trace, "created_at", None)
    return int(created.timestamp()) if created is not None else int(time.time())


def chat_completion(decision: Decision, *, model: str = MODEL_NAME) -> dict[str, Any]:
    """The ``ChatCompletion`` document of a decision (``choices[0].message`` as in §3.10)."""
    return {
        "id": "chatcmpl-" + decision.decision_id,
        "object": "chat.completion",
        "created": _created(decision),
        "model": model,
        "system_fingerprint": None,
        "choices": [
            {
                "index": 0,
                "message": decision.to_openai_message(),
                "finish_reason": decision.finish_reason,
                "logprobs": None,
            }
        ],
        "usage": usage_doc(decision),
    }


def completion_chunks(completion: Mapping[str, Any]) -> list[dict[str, Any]]:
    """A ``ChatCompletion`` document as ``chat.completion.chunk`` documents (``stream=True``): one delta with the
    whole message, then the finish reason with ``usage``. Jev decides the whole message at once."""
    choice = completion["choices"][0]
    message = choice["message"]
    delta: dict[str, Any] = {"role": "assistant", "content": message.get("content")}
    if message.get("tool_calls"):
        delta["tool_calls"] = [{"index": i, **call} for i, call in enumerate(message["tool_calls"])]
    if "x_jev" in message:
        delta["x_jev"] = message["x_jev"]
    head = {k: completion[k] for k in ("id", "created", "model", "system_fingerprint")}
    return [
        {**head, "object": "chat.completion.chunk",
         "choices": [{"index": 0, "delta": delta, "finish_reason": None, "logprobs": None}]},
        {**head, "object": "chat.completion.chunk",
         "choices": [{"index": 0, "delta": {}, "finish_reason": choice["finish_reason"], "logprobs": None}],
         "usage": completion.get("usage")},
    ]  # fmt: skip


# --------------------------------------------------------------------------------------------------------------------
# complete
# --------------------------------------------------------------------------------------------------------------------


_SHARED_STORE = InMemoryPendingStore()
"""Pending prompts of calls without a router (each builds a fresh router; a later reply is re-compiled safely)."""


def _store(store: PendingStore | None, router: Router | None, built: Router) -> PendingStore:
    if store is not None:
        return store
    return default_store(built) if router is not None else _SHARED_STORE


def _router(
    tools: Sequence[Any] | None,
    router: Router | None,
    backend: Backend | None,
    context: Context | None,
    router_kw: Mapping[str, Any],
) -> Router:
    if router is not None:
        if router_kw:
            raise TypeError(f"router options {sorted(router_kw)} are only accepted without a router")
        return router_for(router, tools)
    if backend is None:
        from jevtools.backends.auto import auto

        backend = auto()
    return Router(list(tools or ()), backend=backend, context=context, **router_kw)


def complete(
    messages: str | Sequence[MessageLike],
    tools: Sequence[Any] | None = None,
    *,
    context: Context | Mapping[str, Any] | None = None,
    router: Router | None = None,
    backend: Backend | None = None,
    tool_choice: Any = "auto",
    parallel_tool_calls: bool = False,
    store: PendingStore | None = None,
    pending_id: str | None = None,
    model: str = MODEL_NAME,
    **router_kw: Any,
) -> dict[str, Any]:
    """Decide one turn and return a ``ChatCompletion``-shaped dict (spec §7.2.1).

    ``router`` is the host's router (``tools`` may name a subset or add tools, see
    :func:`~jevtools.adapters._router.router_for`); without one, a router is built from ``tools``, ``backend``
    (default :func:`jevtools.backends.auto`), ``context`` and ``router_kw`` (``policy``, ``filler``…). ``context``
    may be a mapping of fields (``now``, ``tz``, ``user``…) laid over the router's context. Pending prompts are
    kept in ``store`` (default: one in-memory store per router) and matched by ``pending_id``, ``x_jev.pending_id``
    or the prefix hash.
    """
    turns = [{"role": "user", "content": messages}] if isinstance(messages, str) else list(messages)
    base = context if isinstance(context, Context) else None
    r = _router(tools, router, backend, base, router_kw)
    ctx = merge_context(r.context, context)
    decision = decide_turn(r, turns, context=ctx, tool_choice=tool_choice, parallel_tool_calls=parallel_tool_calls,
                           store=_store(store, router, r), pending_id=pending_id)  # fmt: skip
    return chat_completion(decision, model=model)


async def acomplete(
    messages: str | Sequence[MessageLike],
    tools: Sequence[Any] | None = None,
    *,
    context: Context | Mapping[str, Any] | None = None,
    router: Router | None = None,
    backend: Backend | None = None,
    tool_choice: Any = "auto",
    parallel_tool_calls: bool = False,
    store: PendingStore | None = None,
    pending_id: str | None = None,
    model: str = MODEL_NAME,
    **router_kw: Any,
) -> dict[str, Any]:
    """Async :func:`complete`."""
    turns = [{"role": "user", "content": messages}] if isinstance(messages, str) else list(messages)
    base = context if isinstance(context, Context) else None
    r = _router(tools, router, backend, base, router_kw)
    ctx = merge_context(r.context, context)
    decision = await adecide_turn(r, turns, context=ctx, tool_choice=tool_choice,
                                  parallel_tool_calls=parallel_tool_calls,
                                  store=_store(store, router, r),
                                  pending_id=pending_id)  # fmt: skip
    return chat_completion(decision, model=model)


# --------------------------------------------------------------------------------------------------------------------
# wrap
# --------------------------------------------------------------------------------------------------------------------


class AttrDict(dict[str, Any]):
    """A dict with attribute access (``resp.choices[0].message.tool_calls``), used when the ``openai`` package is
    not installed. ``model_dump()`` returns a plain copy, like the SDK's models."""

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name) from None

    def model_dump(self, **_: Any) -> dict[str, Any]:
        """A plain JSON copy."""
        return _plain(self)  # type: ignore[no-any-return]


def _attr(value: Any) -> Any:
    if isinstance(value, Mapping):
        return AttrDict({k: _attr(v) for k, v in value.items()})
    if isinstance(value, list):
        return [_attr(v) for v in value]
    return value


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_plain(v) for v in value]
    return value


def to_response(doc: Mapping[str, Any], *, chunk: bool = False) -> Any:
    """``openai.types.chat.ChatCompletion`` (or ``ChatCompletionChunk``) when the SDK is installed (extra fields
    such as ``x_jev`` are kept), else an :class:`AttrDict`."""
    try:
        module = importlib.import_module("openai.types.chat")
    except ImportError:
        return _attr(doc)
    cls = getattr(module, "ChatCompletionChunk" if chunk else "ChatCompletion")
    return cls.model_validate(dict(doc))


@dataclass
class _Interceptor:
    router: Router
    model_name: str
    store: PendingStore
    context: Context | Mapping[str, Any] | None

    def arguments(self, kwargs: Mapping[str, Any]) -> dict[str, Any]:
        extra = kwargs.get("extra_body") or {}
        jev = (extra.get("jevtools") or {}) if isinstance(extra, Mapping) else {}
        context: Any = self.context
        if jev.get("context") is not None:
            context = merge_context(merge_context(self.router.context, self.context), jev["context"])
        return {
            "tools": kwargs.get("tools"),
            "context": context,
            "tool_choice": kwargs.get("tool_choice") or "auto",
            "parallel_tool_calls": bool(kwargs.get("parallel_tool_calls") or False),
            "store": self.store,
            "pending_id": jev.get("pending_id"),
            "model": self.model_name,
        }


class _Completions:
    def __init__(self, inner: Any, interceptor: _Interceptor, is_async: bool) -> None:
        self._inner = inner
        self._icpt = interceptor
        self._async = is_async

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    def create(self, **kwargs: Any) -> Any:
        if kwargs.get("model") != self._icpt.model_name:
            return self._inner.create(**kwargs)
        if self._async:
            return self._acreate(kwargs)
        doc = complete(kwargs["messages"], router=self._icpt.router, **self._icpt.arguments(kwargs))
        if kwargs.get("stream"):
            return iter([to_response(c, chunk=True) for c in completion_chunks(doc)])
        return to_response(doc)

    async def _acreate(self, kwargs: dict[str, Any]) -> Any:
        doc = await acomplete(kwargs["messages"], router=self._icpt.router, **self._icpt.arguments(kwargs))
        if kwargs.get("stream"):
            return _achunks([to_response(c, chunk=True) for c in completion_chunks(doc)])
        return to_response(doc)


async def _achunks(chunks: list[Any]) -> AsyncIterator[Any]:
    for chunk in chunks:
        yield chunk


class _Chat:
    def __init__(self, inner: Any, completions: _Completions) -> None:
        self._inner = inner
        self.completions = completions

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


class WrappedClient:
    """An OpenAI client whose ``chat.completions.create`` is decided by jevtools for ``model_name`` only; every
    other attribute and model passes through to the wrapped client."""

    def __init__(self, client: Any, chat: _Chat) -> None:
        self._client = client
        self.chat = chat

    def __getattr__(self, name: str) -> Any:
        return getattr(self._client, name)

    @property
    def wrapped(self) -> Any:
        """The original client."""
        return self._client


def wrap(
    client: Any,
    router: Router,
    model_name: str = MODEL_NAME,
    *,
    context: Context | Mapping[str, Any] | None = None,
    store: PendingStore | None = None,
    is_async: bool | None = None,
) -> WrappedClient:
    """Wrap an OpenAI (or OpenAI-compatible) client: ``client.chat.completions.create(model=model_name, …)`` is
    answered by ``router`` (spec §7.2.1); any other model goes to the wrapped client unchanged.

    Async clients are detected from ``chat.completions.create`` being a coroutine function (``is_async``
    overrides). Per-request context: ``extra_body={"jevtools": {"context": {...}, "pending_id": "…"}}``.
    ``stream=True`` yields two chunks (the whole message, then the finish reason).
    """
    completions = client.chat.completions
    if is_async is None:
        is_async = inspect.iscoroutinefunction(getattr(completions, "create", None))
    interceptor = _Interceptor(router=router, model_name=model_name,
                               store=store if store is not None else default_store(router),
                               context=context)  # fmt: skip
    return WrappedClient(client, _Chat(client.chat, _Completions(completions, interceptor, bool(is_async))))


# --------------------------------------------------------------------------------------------------------------------
# Error mapping (§7.2.4)
# --------------------------------------------------------------------------------------------------------------------

_SECRET = re.compile(r"(?i)(bearer\s+)\S+|\b(sk|or|ts)-[A-Za-z0-9_\-]{8,}|((?:api[_-]?key|key|token)=)[^&\s]+")
_STATUS = re.compile(r"\bHTTP (\d{3})\b")


def redact(text: str) -> str:
    """``text`` with bearer tokens, ``sk-…``-style keys and ``key=`` query values masked."""

    def mask(m: re.Match[str]) -> str:
        if m.group(1):
            return m.group(1) + "[redacted]"
        if m.group(3):
            return m.group(3) + "[redacted]"
        return "[redacted]"

    return _SECRET.sub(mask, text)


@dataclass(frozen=True)
class ErrorResponse:
    """An HTTP error in OpenAI's shape: ``status``, ``{"error": {type, code, message, param}}`` and headers."""

    status: int
    body: dict[str, Any]
    headers: dict[str, str] = field(default_factory=dict)

    @property
    def code(self) -> str:
        """``error.code``."""
        return str(self.body["error"]["code"])


def api_error(status: int, type_: str, code: str, message: str, headers: Mapping[str, str] | None = None) -> (
        ErrorResponse):  # fmt: skip
    """An OpenAI-style error document (``{"error": {"type", "code", "message", "param"}}``, message redacted)."""
    body = {"error": {"type": type_, "code": code, "message": redact(message), "param": None}}
    return ErrorResponse(status=status, body=body, headers=dict(headers or {}))


def _retry_headers(retry_after: float | None) -> dict[str, str]:
    return {"Retry-After": str(max(0, math.ceil(retry_after)))} if retry_after is not None else {}


def status_error(status: int | None, message: str = "", *, retry_after: float | None = None) -> ErrorResponse:
    """Map a Jev HTTP status (``None`` = transport failure or timeout) to the proxy's error (§7.2.4)."""
    if status in (400, 404, 422):
        return api_error(
            502, "upstream_error", "jev_invalid_request", message or f"Jev rejected the request ({status})"
        )
    if status in (401, 403):
        return api_error(502, "upstream_error", "jev_auth", f"Jev rejected the credentials (HTTP {status})")
    if status == 429:
        return api_error(429, "rate_limit_error", "jev_rate_limited", message or "Jev rate limit reached",
                      _retry_headers(retry_after))  # fmt: skip
    return api_error(503, "upstream_error", "jev_unavailable", message or "Jev is unavailable")


def error_response(exc: BaseException) -> ErrorResponse:
    """The §7.2.4 error for an exception raised while serving a request.

    ======================================  ======  =============================================
    Condition                               HTTP    ``error.code``
    ======================================  ======  =============================================
    bad tool schema / compile error         400     ``jevtools_bad_tool`` (``invalid_request_error``)
    Jev 400/422 (after isolation), 404      502     ``jev_invalid_request``
    Jev 401/403, no API key configured      502     ``jev_auth`` (the key is never echoed)
    Jev 429 after retries                   429     ``jev_rate_limited`` + ``Retry-After``
    Jev 5xx/timeout after retries           503     ``jev_unavailable``
    malformed Jev response                  502     ``jev_protocol_error``
    anything else                           500     ``jevtools_internal``
    ======================================  ======  =============================================
    """
    if isinstance(exc, (CatalogError, ConstraintError, BallotError)):
        return api_error(400, "invalid_request_error", "jevtools_bad_tool", str(exc))
    if isinstance(exc, (JevAuthError, BackendConfigError)):
        text = "no Jev API key configured" if isinstance(exc, BackendConfigError) else ""
        return api_error(502, "upstream_error", "jev_auth", text or f"Jev rejected the credentials (HTTP {exc.status})")
    if isinstance(exc, JevRateLimited):
        return status_error(429, str(exc), retry_after=exc.retry_after)
    if isinstance(exc, (JevValidationError, JevNotFound)):
        return status_error(exc.status or 422, str(exc))
    if isinstance(exc, JevUnavailable):
        return status_error(exc.status, str(exc))
    if isinstance(exc, JevProtocolError):
        return api_error(502, "upstream_error", "jev_protocol_error", str(exc))
    if isinstance(exc, BackendError):
        return status_error(exc.status, str(exc))
    return api_error(500, "server_error", "jevtools_internal", f"{type(exc).__name__}: {exc}")


def decision_error(decision: Decision) -> ErrorResponse | None:
    """The §7.2.4 error for a fail-closed decision (rule ``P0.backend.fail_closed``), else ``None``.

    The router records backend failures as text in the trace's call records; the HTTP status is read back from
    it (``… HTTP 401 …``); a failure without one (transport error, timeout) maps to ``jev_unavailable``.
    """
    if decision.rule != RULE_FAIL_CLOSED:
        return None
    errors = [c.error for r in getattr(decision.trace, "rounds", []) for c in r.calls if c.error]
    text = errors[0] if errors else "Jev call failed"
    found = _STATUS.search(text)
    status = int(found.group(1)) if found else None
    if status is not None and 500 <= status:
        status = None
    return status_error(status, text)


__all__ = [
    "MODEL_NAME",
    "AttrDict",
    "ErrorResponse",
    "WrappedClient",
    "acomplete",
    "api_error",
    "chat_completion",
    "complete",
    "completion_chunks",
    "decision_error",
    "error_response",
    "redact",
    "status_error",
    "to_response",
    "usage_doc",
    "wrap",
]
