"""``Cassette``: record and replay Jev calls (spec §8.7).

A cassette is a JSONL file with one record per distinct request::

    {"request_sha256": "sha256:…", "backend": "openrouter_decisions", "model": "~typesafe/jev-latest",
     "request": {…wire request…}, "response": {…wire response…}, "recorded_at": "2026-09-24T12:05:00Z"}

The key is ``sha256(canonical request)`` (:func:`jevtools.canonical.sha256_of` of the wire body, model included), the
same key the router's memoization uses (§3.8.5). Modes:

- ``replay``: answer only from the file; an unknown request raises :class:`CassetteMiss`, so CI fails loudly. It
  deliberately is **not** a :class:`~jevtools.backends.errors.BackendError`: the router turns backend errors into
  rule P0 (fail closed), which would hide a stale cassette behind an ``abstain``.
- ``record``: call the ``inner`` (live) backend and append every exchange to the file (a later line for the same key
  wins when the file is loaded).
- ``passthrough``: call ``inner`` without reading or writing the file.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Iterator
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from jevtools.backends.base import Backend
from jevtools.backends.errors import BackendConfigError
from jevtools.canonical import canonical_str, sha256_of
from jevtools.errors import JevtoolsError
from jevtools.wire import DecisionRequest, DecisionResponse

CassetteMode = Literal["replay", "record", "passthrough"]
MODES: tuple[str, ...] = ("replay", "record", "passthrough")


class CassetteMiss(JevtoolsError, LookupError):
    """A replayed request is not in the cassette (``request_sha256`` names the missing key)."""

    def __init__(self, request_sha256: str, path: Path) -> None:
        self.request_sha256 = request_sha256
        self.path = path
        super().__init__(f"{path}: no recorded response for request {request_sha256} (re-record the cassette)")


def request_key(request: DecisionRequest) -> str:
    """``sha256:<hex>`` of the canonical wire request (model, state and questions)."""
    return sha256_of(request.to_wire())


class Cassette:
    """A :class:`~jevtools.backends.base.Backend` that records or replays another backend's calls.

    ``Cassette(path, mode="replay"|"record"|"passthrough", inner=None)``. ``record`` and ``passthrough`` need
    ``inner``. ``model`` and ``name`` default to the inner backend's; a replay-only cassette takes them from its
    records (the model must match, since it is part of every key), else ``model`` / ``name`` arguments, else
    ``"cassette"``. Thread-safe (split calls may run concurrently).
    """

    def __init__(
        self,
        path: str | Path,
        mode: CassetteMode = "replay",
        inner: Backend | None = None,
        *,
        model: str | None = None,
        name: str | None = None,
    ) -> None:
        if mode not in MODES:
            raise BackendConfigError(f"unknown cassette mode {mode!r} (expected one of {', '.join(MODES)})")
        if mode != "replay" and inner is None:
            raise BackendConfigError(f"a {mode!r} cassette needs an inner backend")
        self.path = Path(path)
        self.mode: CassetteMode = mode
        self.inner = inner
        self._lock = threading.Lock()
        self._records: dict[str, dict[str, Any]] = {}
        if mode != "passthrough" and self.path.exists():
            for record in _read(self.path):
                self._records[str(record["request_sha256"])] = record
        recorded = next(iter(self._records.values()), {})
        self.model: str = model or (inner.model if inner is not None else str(recorded.get("model") or "cassette"))
        self.name: str = name or (inner.name if inner is not None else str(recorded.get("backend") or "cassette"))
        self.hits = 0
        self.misses = 0

    def __repr__(self) -> str:
        return f"Cassette({str(self.path)!r}, mode={self.mode!r}, inner={self.inner!r})"

    def __len__(self) -> int:
        return len(self._records)

    def __contains__(self, request: object) -> bool:
        return isinstance(request, DecisionRequest) and request_key(request) in self._records

    @property
    def records(self) -> list[dict[str, Any]]:
        """The loaded and recorded records (one per key, last recording wins)."""
        with self._lock:
            return list(self._records.values())

    # -- Backend protocol -----------------------------------------------------------------------------------------

    def decide(self, request: DecisionRequest) -> DecisionResponse:
        if self.mode == "replay":
            return self._replay(request)
        assert self.inner is not None
        response = self.inner.decide(request)
        if self.mode == "record":
            self._record(request, response)
        return response

    async def adecide(self, request: DecisionRequest) -> DecisionResponse:
        if self.mode == "replay":
            return self._replay(request)
        assert self.inner is not None
        response = await self.inner.adecide(request)
        if self.mode == "record":
            self._record(request, response)
        return response

    # -- internals ------------------------------------------------------------------------------------------------

    def _replay(self, request: DecisionRequest) -> DecisionResponse:
        key = request_key(request)
        with self._lock:
            record = self._records.get(key)
            if record is None:
                self.misses += 1
                raise CassetteMiss(key, self.path)
            self.hits += 1
        return DecisionResponse.from_json(dict(record["response"]))

    def _record(self, request: DecisionRequest, response: DecisionResponse) -> None:
        key = request_key(request)
        record = {
            "request_sha256": key,
            "backend": self.name,
            "model": request.model,
            "request": request.to_wire(),
            "response": response.model_dump(mode="json", exclude_none=True),
            "recorded_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
        line = canonical_str(record)
        with self._lock:
            self._records[key] = record
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")


def _read(path: Path) -> Iterator[dict[str, Any]]:
    """The records of a cassette file (blank lines skipped; a malformed line is a configuration error)."""
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except ValueError as exc:
                raise BackendConfigError(f"{path}:{number}: not a JSON line: {exc}") from exc
            if not isinstance(record, dict) or "request_sha256" not in record or "response" not in record:
                raise BackendConfigError(f"{path}:{number}: a cassette record needs request_sha256 and response")
            yield record


__all__ = ["MODES", "Cassette", "CassetteMiss", "CassetteMode", "request_key"]
