"""HTTP backends for the three Jev endpoints (spec §8.2; same request/response contract, only URL and model differ).

- One reused :class:`httpx.Client` (and one :class:`httpx.AsyncClient` per event loop) per backend, so split calls
  share connections.
- Typed errors (:mod:`jevtools.backends.errors`); retries on ``{408, 429, 500, 502, 503, 504}`` and transport
  errors, honouring ``retry-after-ms`` / ``retry-after``, else exponential backoff ``0.5·2ⁿ s`` (≤ 5 s).
- ``usage.cost``, ``id`` and ``provider`` (OpenRouter Decisions) are parsed into the response.
- Optional OpenRouter attribution headers ``HTTP-Referer`` and ``X-Title``.
- ``JEVTOOLS_MODEL`` overrides the default model id when no model is passed.
"""

from __future__ import annotations

import asyncio
import os
import time
from collections.abc import Awaitable, Callable, Mapping
from typing import Any

import httpx

from jevtools.backends.errors import (
    BackendConfigError,
    BackendError,
    JevProtocolError,
    JevRateLimited,
    JevUnavailable,
    error_for_status,
)
from jevtools.wire import DecisionRequest, DecisionResponse

RETRY_STATUSES = frozenset({408, 429, 500, 502, 503, 504})
DEFAULT_TIMEOUT = 30.0
DEFAULT_MAX_RETRIES = 2
TYPESAFE_MODEL = "jev-latest"
OPENROUTER_MODEL = "~typesafe/jev-latest"
OPENROUTER_SYSTEMONE_URL = "https://openrouter.ai/api/v1/systemone"
OPENROUTER_DECISIONS_URL = "https://openrouter.ai/api/alpha/decisions"


class HTTPBackend:
    """Calls a Jev endpoint over HTTPS.

    Prefer the constructors :meth:`typesafe`, :meth:`openrouter` and :meth:`openrouter_decisions`. ``transport`` /
    ``async_transport`` replace the network (``httpx.MockTransport`` in tests); ``sleep`` / ``asleep`` replace the
    backoff sleeps.
    """

    def __init__(
        self,
        *,
        url: str,
        api_key: str,
        model: str,
        name: str = "http",
        timeout: float = DEFAULT_TIMEOUT,
        max_retries: int = DEFAULT_MAX_RETRIES,
        headers: Mapping[str, str] | None = None,
        referer: str | None = None,
        title: str | None = None,
        transport: httpx.BaseTransport | None = None,
        async_transport: httpx.AsyncBaseTransport | None = None,
        sleep: Callable[[float], None] | None = None,
        asleep: Callable[[float], Awaitable[None]] | None = None,
    ) -> None:
        if not api_key:
            raise BackendConfigError(f"No API key for {url}")
        self.url = url
        self.model = model
        self.name = name
        self.timeout = timeout
        self.max_retries = max_retries
        attribution = {"HTTP-Referer": referer, "X-Title": title}
        self._headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
            **{k: v for k, v in attribution.items() if v},
            **(headers or {}),
        }
        self._transport = transport
        self._async_transport = async_transport
        self._sleep = sleep or time.sleep
        self._asleep = asleep or asyncio.sleep
        self._client: httpx.Client | None = None
        self._aclients: dict[int, httpx.AsyncClient] = {}

    # -- constructors -------------------------------------------------------------------------------------------

    @classmethod
    def typesafe(cls, api_key: str | None = None, *, model: str | None = None, **kw: Any) -> HTTPBackend:
        """TypeSafe direct: ``POST $TYPESAFE_BASE_URL/v1/systemone`` (default ``https://api.typesafe.ai``)."""
        base = os.environ.get("TYPESAFE_BASE_URL", "https://api.typesafe.ai")
        key = api_key or os.environ.get("TYPESAFE_API_KEY", "")
        return cls(url=f"{base.rstrip('/')}/v1/systemone", api_key=key, model=_model(model, TYPESAFE_MODEL),
                   name="typesafe", **kw)  # fmt: skip

    @classmethod
    def openrouter(cls, api_key: str | None = None, *, model: str | None = None, **kw: Any) -> HTTPBackend:
        """OpenRouter System One surface: ``POST https://openrouter.ai/api/v1/systemone``."""
        key = api_key or os.environ.get("OPENROUTER_API_KEY", "")
        return cls(url=OPENROUTER_SYSTEMONE_URL, api_key=key, model=_model(model, OPENROUTER_MODEL),
                   name="openrouter_systemone", **kw)  # fmt: skip

    @classmethod
    def openrouter_decisions(cls, api_key: str | None = None, *, model: str | None = None, **kw: Any) -> HTTPBackend:
        """OpenRouter Decisions API: ``POST https://openrouter.ai/api/alpha/decisions`` (reports ``usage.cost``)."""
        key = api_key or os.environ.get("OPENROUTER_API_KEY", "")
        return cls(url=OPENROUTER_DECISIONS_URL, api_key=key, model=_model(model, OPENROUTER_MODEL),
                   name="openrouter_decisions", **kw)  # fmt: skip

    # -- clients ------------------------------------------------------------------------------------------------

    @property
    def client(self) -> httpx.Client:
        """The reused synchronous client."""
        if self._client is None:
            self._client = httpx.Client(timeout=self.timeout, transport=self._transport)
        return self._client

    def _aclient(self) -> httpx.AsyncClient:
        """The reused async client of the running event loop (connection pools are loop-bound)."""
        key = id(asyncio.get_running_loop())
        client = self._aclients.get(key)
        if client is None:
            client = httpx.AsyncClient(timeout=self.timeout, transport=self._async_transport)
            self._aclients[key] = client
        return client

    def close(self) -> None:
        """Close the synchronous client (async clients are closed by :meth:`aclose`)."""
        if self._client is not None:
            self._client.close()
            self._client = None

    async def aclose(self) -> None:
        """Close every client."""
        self.close()
        clients, self._aclients = list(self._aclients.values()), {}
        for client in clients:
            await client.aclose()

    def __enter__(self) -> HTTPBackend:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    async def __aenter__(self) -> HTTPBackend:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()

    # -- calls --------------------------------------------------------------------------------------------------

    def _body(self, request: DecisionRequest) -> dict[str, Any]:
        body = request.to_wire()
        if not body.get("model"):
            body["model"] = self.model
        return body

    def decide(self, request: DecisionRequest) -> DecisionResponse:
        body = self._body(request)
        for attempt in range(self.max_retries + 1):
            try:
                response = self.client.post(self.url, json=body, headers=self._headers)
            except httpx.TransportError as exc:
                self._sleep(self._transport_retry(exc, attempt))
                continue
            delay = self._retry_delay(response, attempt)
            if delay is None:
                return self._parse(response)
            self._sleep(delay)
        raise AssertionError("unreachable")  # pragma: no cover

    async def adecide(self, request: DecisionRequest) -> DecisionResponse:
        body = self._body(request)
        for attempt in range(self.max_retries + 1):
            try:
                response = await self._aclient().post(self.url, json=body, headers=self._headers)
            except httpx.TransportError as exc:
                await self._asleep(self._transport_retry(exc, attempt))
                continue
            delay = self._retry_delay(response, attempt)
            if delay is None:
                return self._parse(response)
            await self._asleep(delay)
        raise AssertionError("unreachable")  # pragma: no cover

    def _transport_retry(self, exc: httpx.TransportError, attempt: int) -> float:
        """Backoff before the next attempt, or :class:`JevUnavailable` once retries are exhausted."""
        if attempt >= self.max_retries:
            kind = "timeout" if isinstance(exc, httpx.TimeoutException) else "transport error"
            raise JevUnavailable(f"{self.url}: {kind}: {exc}") from exc
        return backoff(attempt)

    def _retry_delay(self, response: httpx.Response, attempt: int) -> float | None:
        """``None`` to parse this response now, else the delay before retrying a retryable status."""
        if response.status_code not in RETRY_STATUSES or attempt >= self.max_retries:
            return None
        return retry_after(response) or backoff(attempt)

    def _parse(self, response: httpx.Response) -> DecisionResponse:
        try:
            payload = response.json()
        except ValueError:
            payload = None
        if not response.is_success:
            raise error_for_status(
                response.status_code, payload, url=self.url, message=_message(payload, response),
                retry_after=retry_after(response),
            )  # fmt: skip
        if not isinstance(payload, dict):
            raise JevProtocolError(f"{self.url}: response body is not a JSON object", status=response.status_code)
        try:
            return DecisionResponse.from_json(payload)
        except ValueError as exc:
            raise JevProtocolError(f"{self.url}: malformed response: {exc}", status=response.status_code) from exc

    def __repr__(self) -> str:
        return f"HTTPBackend(name={self.name!r}, url={self.url!r}, model={self.model!r})"


def TypeSafe(api_key: str | None = None, **kw: Any) -> HTTPBackend:  # noqa: N802 - spec name
    """``HTTPBackend.typesafe()``."""
    return HTTPBackend.typesafe(api_key, **kw)


def OpenRouterSystemOne(api_key: str | None = None, **kw: Any) -> HTTPBackend:  # noqa: N802 - spec name
    """``HTTPBackend.openrouter()``."""
    return HTTPBackend.openrouter(api_key, **kw)


def OpenRouterDecisions(api_key: str | None = None, **kw: Any) -> HTTPBackend:  # noqa: N802 - spec name
    """``HTTPBackend.openrouter_decisions()``."""
    return HTTPBackend.openrouter_decisions(api_key, **kw)


def _model(model: str | None, default: str) -> str:
    return model or os.environ.get("JEVTOOLS_MODEL") or default


def backoff(attempt: int) -> float:
    """Exponential backoff ``0.5·2ⁿ`` seconds, capped at 5 s."""
    return float(min(0.5 * 2**attempt, 5.0))


def retry_after(response: httpx.Response) -> float | None:
    """The server's retry hint in seconds: ``retry-after-ms`` first, then ``retry-after``."""
    for header, scale in (("retry-after-ms", 1000.0), ("retry-after", 1.0)):
        raw = response.headers.get(header)
        if raw is None:
            continue
        try:
            return max(0.0, float(raw) / scale)
        except ValueError:
            continue
    return None


def _message(payload: Any, response: httpx.Response) -> str:
    if isinstance(payload, dict):
        error = payload.get("error")
        if isinstance(error, dict) and isinstance(error.get("message"), str):
            return str(error["message"])
        if isinstance(error, str):
            return error
        detail = payload.get("detail")
        if isinstance(detail, list):
            parts = []
            for entry in detail:
                if isinstance(entry, dict) and isinstance(entry.get("msg"), str):
                    loc = ".".join(str(p) for p in entry.get("loc", []) if p != "body")
                    parts.append(f"{loc}: {entry['msg']}" if loc else entry["msg"])
            if parts:
                return "; ".join(parts)
        if isinstance(detail, str):
            return detail
        if isinstance(payload.get("message"), str):
            return str(payload["message"])
    if payload is None:
        return response.text[:200] or "non-JSON response"
    return str(payload)[:200]


__all__ = [
    "OPENROUTER_DECISIONS_URL",
    "OPENROUTER_MODEL",
    "OPENROUTER_SYSTEMONE_URL",
    "RETRY_STATUSES",
    "TYPESAFE_MODEL",
    "BackendError",
    "HTTPBackend",
    "JevRateLimited",
    "OpenRouterDecisions",
    "OpenRouterSystemOne",
    "TypeSafe",
    "backoff",
    "retry_after",
]
