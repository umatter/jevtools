"""Token estimation with a running correction factor (spec §5.5).

``est(req) = chars(canonical JSON) / chars_per_token × r``. After each response,
``r ← EMA_α(usage.input_tokens / est_uncorrected)`` with ``α = 0.3``, clamped to ``[0.7, 1.6]`` and kept per
``(backend, model)``. The planner reads :meth:`TokenEstimator.ratio` into :attr:`jevtools.validate.Limits.token_ratio`.
"""

from __future__ import annotations

import math
import threading
from collections.abc import Mapping
from typing import Any

from jevtools.canonical import canonical_str
from jevtools.wire import DecisionRequest, Usage

EMA_ALPHA = 0.3
RATIO_MIN = 0.7
RATIO_MAX = 1.6

Key = tuple[str, str]


class TokenEstimator:
    """Estimates request tokens and learns the tokenizer ratio per backend and model (thread-safe)."""

    def __init__(
        self,
        chars_per_token: float = 3.5,
        *,
        alpha: float = EMA_ALPHA,
        clamp: tuple[float, float] = (RATIO_MIN, RATIO_MAX),
        ratios: Mapping[Key, float] | None = None,
    ) -> None:
        if chars_per_token <= 0:
            raise ValueError("chars_per_token must be positive")
        self.chars_per_token = chars_per_token
        self.alpha = alpha
        self.clamp = clamp
        self._ratios: dict[Key, float] = dict(ratios or {})
        self._lock = threading.Lock()

    def ratio(self, backend: str = "", model: str = "") -> float:
        """The current correction factor ``r`` for ``(backend, model)`` (1.0 before any observation)."""
        with self._lock:
            return self._ratios.get((backend, model), 1.0)

    @property
    def ratios(self) -> dict[Key, float]:
        """A snapshot of every learned ratio."""
        with self._lock:
            return dict(self._ratios)

    def raw(self, request: DecisionRequest | Mapping[str, Any]) -> float:
        """Uncorrected estimate: ``chars(canonical JSON) / chars_per_token``."""
        body = request.to_wire() if isinstance(request, DecisionRequest) else request
        return len(canonical_str(body)) / self.chars_per_token

    def est(self, request: DecisionRequest | Mapping[str, Any], *, backend: str = "", model: str = "") -> int:
        """Corrected estimate ``ceil(raw × r)``."""
        return math.ceil(self.raw(request) * self.ratio(backend, model))

    def observe(
        self, request: DecisionRequest | Mapping[str, Any], usage: Usage | None, *, backend: str = "", model: str = ""
    ) -> float:
        """Fold one response's ``usage.input_tokens`` into ``r`` (EMA, clamped); returns the new ratio.

        Responses without ``input_tokens`` leave the ratio unchanged.
        """
        if usage is None or not usage.input_tokens:
            return self.ratio(backend, model)
        raw = self.raw(request)
        if raw <= 0:
            return self.ratio(backend, model)
        sample = usage.input_tokens / raw
        low, high = self.clamp
        with self._lock:
            current = self._ratios.get((backend, model), 1.0)
            updated = min(high, max(low, (1 - self.alpha) * current + self.alpha * sample))
            self._ratios[(backend, model)] = updated
            return updated


__all__ = ["EMA_ALPHA", "RATIO_MAX", "RATIO_MIN", "TokenEstimator"]
