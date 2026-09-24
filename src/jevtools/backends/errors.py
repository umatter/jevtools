"""Typed backend errors (spec §8.4).

Every failure a backend can report is a :class:`BackendError` (``status`` is the HTTP status when there is one).
The router maps all of them to rule P0 (fail closed, §5.6) after retries and 422 isolation:

- :class:`JevValidationError` (422): ``detail[].loc`` names the offending question(s); :meth:`~JevValidationError.qids`
  drives isolation, :meth:`~JevValidationError.request_level` says the state/model itself was rejected.
- :class:`JevRateLimited` (429) and :class:`JevUnavailable` (408, 5xx, transport errors, timeouts): retried with
  backoff, raised once retries are exhausted.
- :class:`JevAuthError` (401, 403) and :class:`JevNotFound` (404): never retried.
- :class:`JevProtocolError`: a malformed body or an answer that breaks the answer-shape guards (§8.2).
- :class:`BackendConfigError`: the backend cannot be built (no key, unknown backend name).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


class BackendError(RuntimeError):
    """A backend failed to produce a usable response."""

    def __init__(self, message: str, *, status: int | None = None) -> None:
        super().__init__(message)
        self.status = status


class JevValidationError(BackendError):
    """HTTP 422 with an ``HTTPValidationError`` body: ``{"detail": [{"loc": [...], "msg", "type"}]}``."""

    def __init__(self, detail: Sequence[Mapping[str, Any]], *, message: str | None = None, status: int = 422) -> None:
        self.detail = [dict(entry) for entry in detail]
        super().__init__(message or f"HTTP {status}: {_detail_text(self.detail)}", status=status)

    def qids(self) -> set[str]:
        """Question ids named by ``loc[2]`` of entries whose ``loc`` starts with ``["body", "questions"]``."""
        found: set[str] = set()
        for entry in self.detail:
            loc = list(entry.get("loc") or ())
            if len(loc) >= 3 and loc[:2] == ["body", "questions"]:
                found.add(str(loc[2]))
        return found

    def request_level(self) -> bool:
        """Whether any entry points outside a single question (state, model, the body itself): not isolatable."""
        for entry in self.detail:
            loc = list(entry.get("loc") or ())
            if not (len(loc) >= 3 and loc[:2] == ["body", "questions"]):
                return True
        return not self.detail


class JevRateLimited(BackendError):
    """HTTP 429 after retries. ``retry_after`` is the server's hint in seconds (if any)."""

    def __init__(self, message: str, *, retry_after: float | None = None, status: int = 429) -> None:
        super().__init__(message, status=status)
        self.retry_after = retry_after


class JevUnavailable(BackendError):
    """408, 5xx, transport errors or timeouts after retries."""


class JevAuthError(BackendError):
    """401 or 403: the key is missing, invalid or not allowed to use the model."""


class JevNotFound(BackendError):
    """404: wrong URL or unknown model."""


class JevProtocolError(BackendError):
    """The response breaks the wire contract: non-JSON body, missing answer for a sent question, type mismatch."""


class BackendConfigError(BackendError):
    """The backend cannot be configured (no API key, unknown backend name)."""


def _detail_text(detail: Sequence[Mapping[str, Any]]) -> str:
    parts = []
    for entry in detail:
        loc = ".".join(str(p) for p in entry.get("loc") or () if p != "body")
        msg = str(entry.get("msg", ""))
        parts.append(f"{loc}: {msg}" if loc else msg)
    return "; ".join(parts) or "validation error"


def error_for_status(
    status: int, payload: Any, *, url: str, message: str, retry_after: float | None = None
) -> BackendError:
    """The typed error for a non-success HTTP status (``payload`` is the parsed body, if any)."""
    text = f"{url}: HTTP {status}: {message}"
    if status == 422 and isinstance(payload, Mapping) and isinstance(payload.get("detail"), list):
        entries = [e for e in payload["detail"] if isinstance(e, Mapping)]
        return JevValidationError(entries, message=text, status=status)
    if status == 429:
        return JevRateLimited(text, retry_after=retry_after, status=status)
    if status in (401, 403):
        return JevAuthError(text, status=status)
    if status == 404:
        return JevNotFound(text, status=status)
    if status == 408 or status >= 500:
        return JevUnavailable(text, status=status)
    return BackendError(text, status=status)


__all__ = [
    "BackendConfigError",
    "BackendError",
    "JevAuthError",
    "JevNotFound",
    "JevProtocolError",
    "JevRateLimited",
    "JevUnavailable",
    "JevValidationError",
    "error_for_status",
]
