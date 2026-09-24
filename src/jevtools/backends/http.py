"""HTTP backends for the three Jev endpoints (same request/response contract)."""

from __future__ import annotations

import asyncio
import os
import time
from collections.abc import Mapping
from typing import Any

import httpx

from jevtools.backends.base import BackendError
from jevtools.wire import DecisionRequest, DecisionResponse

_RETRY_STATUSES = frozenset({408, 429, 500, 502, 503, 504})


class HTTPBackend:
    """Calls a Jev endpoint over HTTPS.

    Use the constructors :meth:`typesafe`, :meth:`openrouter` and
    :meth:`openrouter_decisions` rather than instantiating directly.
    """

    def __init__(
        self,
        *,
        url: str,
        api_key: str,
        model: str,
        timeout: float = 30.0,
        max_retries: int = 2,
        headers: Mapping[str, str] | None = None,
    ) -> None:
        if not api_key:
            raise BackendError(f"No API key for {url}")
        self.url = url
        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries
        self._headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
            **(headers or {}),
        }

    @classmethod
    def typesafe(cls, api_key: str | None = None, *, model: str = "jev-latest", **kw: Any) -> HTTPBackend:
        """TypeSafe direct: ``POST https://api.typesafe.ai/v1/systemone``."""
        base = os.environ.get("TYPESAFE_BASE_URL", "https://api.typesafe.ai")
        key = api_key or os.environ.get("TYPESAFE_API_KEY", "")
        return cls(url=f"{base.rstrip('/')}/v1/systemone", api_key=key, model=model, **kw)

    @classmethod
    def openrouter(cls, api_key: str | None = None, *, model: str = "~typesafe/jev-latest", **kw: Any) -> HTTPBackend:
        """OpenRouter System One surface: ``POST https://openrouter.ai/api/v1/systemone``."""
        key = api_key or os.environ.get("OPENROUTER_API_KEY", "")
        return cls(url="https://openrouter.ai/api/v1/systemone", api_key=key, model=model, **kw)

    @classmethod
    def openrouter_decisions(
        cls, api_key: str | None = None, *, model: str = "~typesafe/jev-latest", **kw: Any
    ) -> HTTPBackend:
        """OpenRouter Decisions API: ``POST https://openrouter.ai/api/alpha/decisions``."""
        key = api_key or os.environ.get("OPENROUTER_API_KEY", "")
        return cls(url="https://openrouter.ai/api/alpha/decisions", api_key=key, model=model, **kw)

    def _body(self, request: DecisionRequest) -> dict[str, Any]:
        body = request.to_wire()
        if not body.get("model"):
            body["model"] = self.model
        return body

    def _parse(self, response: httpx.Response) -> DecisionResponse:
        try:
            payload = response.json()
        except ValueError as exc:
            raise BackendError(f"{self.url}: non-JSON response ({response.status_code})") from exc
        if not response.is_success:
            raise BackendError(f"{self.url}: HTTP {response.status_code}: {_message(payload)}", status=response.status_code)
        if not isinstance(payload, dict):
            raise BackendError(f"{self.url}: unexpected response body")
        return DecisionResponse.from_json(payload)

    def decide(self, request: DecisionRequest) -> DecisionResponse:
        body = self._body(request)
        with httpx.Client(timeout=self.timeout) as client:
            for attempt in range(self.max_retries + 1):
                try:
                    response = client.post(self.url, json=body, headers=self._headers)
                except httpx.TransportError as exc:
                    if attempt == self.max_retries:
                        raise BackendError(f"{self.url}: {exc}") from exc
                    time.sleep(_backoff(attempt))
                    continue
                if response.status_code in _RETRY_STATUSES and attempt < self.max_retries:
                    time.sleep(_retry_after(response) or _backoff(attempt))
                    continue
                return self._parse(response)
        raise AssertionError("unreachable")

    async def adecide(self, request: DecisionRequest) -> DecisionResponse:
        body = self._body(request)
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            for attempt in range(self.max_retries + 1):
                try:
                    response = await client.post(self.url, json=body, headers=self._headers)
                except httpx.TransportError as exc:
                    if attempt == self.max_retries:
                        raise BackendError(f"{self.url}: {exc}") from exc
                    await asyncio.sleep(_backoff(attempt))
                    continue
                if response.status_code in _RETRY_STATUSES and attempt < self.max_retries:
                    await asyncio.sleep(_retry_after(response) or _backoff(attempt))
                    continue
                return self._parse(response)
        raise AssertionError("unreachable")


def _backoff(attempt: int) -> float:
    return min(0.5 * 2**attempt, 5.0)


def _retry_after(response: httpx.Response) -> float | None:
    raw = response.headers.get("retry-after-ms")
    if raw is not None:
        try:
            return max(0.0, float(raw) / 1000)
        except ValueError:
            pass
    raw = response.headers.get("retry-after")
    if raw is not None:
        try:
            return max(0.0, float(raw))
        except ValueError:
            return None
    return None


def _message(payload: Any) -> str:
    if isinstance(payload, dict):
        error = payload.get("error")
        if isinstance(error, dict) and isinstance(error.get("message"), str):
            return error["message"]
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
            return payload["message"]
    return str(payload)[:200]
