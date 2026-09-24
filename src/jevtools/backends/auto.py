"""``auto(allow_offline=False)``: pick a backend from the environment (spec §8.3).

1. ``JEVTOOLS_BACKEND`` if set: ``typesafe | openrouter_systemone | openrouter_decisions | simulator |
   cassette:<path>``;
2. else ``TYPESAFE_API_KEY`` → :func:`~jevtools.backends.http.TypeSafe`;
3. else ``OPENROUTER_API_KEY`` → :func:`~jevtools.backends.http.OpenRouterDecisions` (it reports cost);
4. else, if ``allow_offline``, :class:`~jevtools.backends.simulator.LexicalSimulator` with a :class:`UserWarning`;
5. else :class:`~jevtools.backends.errors.BackendConfigError`. There is no silent offline mode in production.

``JEVTOOLS_MODEL`` overrides the model id of whichever backend is chosen. A ``cassette:<path>`` replays by default;
``JEVTOOLS_CASSETTE_MODE=record|passthrough`` wraps the live backend that steps 2–3 select instead.
"""

from __future__ import annotations

import os
import warnings
from collections.abc import Mapping
from typing import Any

from jevtools.backends.base import Backend
from jevtools.backends.cassette import MODES, Cassette, CassetteMode
from jevtools.backends.errors import BackendConfigError
from jevtools.backends.http import HTTPBackend
from jevtools.backends.simulator import LexicalSimulator

BACKEND_NAMES: tuple[str, ...] = ("typesafe", "openrouter_systemone", "openrouter_decisions", "simulator")
"""Values of ``JEVTOOLS_BACKEND`` besides ``cassette:<path>``."""

OFFLINE_WARNING = (
    "jevtools: no Jev API key (TYPESAFE_API_KEY / OPENROUTER_API_KEY) and no JEVTOOLS_BACKEND; using the offline "
    "LexicalSimulator. It is a lexical test double, not a model: its answers are never evidence about Jev."
)


def auto(allow_offline: bool = False, *, env: Mapping[str, str] | None = None, **kwargs: Any) -> Backend:
    """The backend the environment asks for (spec §8.3).

    ``env`` replaces :data:`os.environ` (tests); ``kwargs`` go to the HTTP backend constructors (``timeout``,
    ``transport``, ``referer``…). Raises :class:`BackendConfigError` for an unknown ``JEVTOOLS_BACKEND``, a named
    HTTP backend without its key, or no configuration at all with ``allow_offline=False``.
    """
    environ = os.environ if env is None else env
    model = environ.get("JEVTOOLS_MODEL") or None
    choice = (environ.get("JEVTOOLS_BACKEND") or "").strip()
    if choice:
        return _named(choice, environ, model, kwargs)
    live = _live(environ, model, kwargs)
    if live is not None:
        return live
    if allow_offline:
        warnings.warn(OFFLINE_WARNING, UserWarning, stacklevel=2)
        return _simulator(model)
    raise BackendConfigError(
        "no Jev backend configured: set TYPESAFE_API_KEY or OPENROUTER_API_KEY, or JEVTOOLS_BACKEND "
        f"({' | '.join(BACKEND_NAMES)} | cassette:<path>); auto(allow_offline=True) allows the offline simulator"
    )


def _named(choice: str, environ: Mapping[str, str], model: str | None, kwargs: dict[str, Any]) -> Backend:
    lowered = choice.lower()
    if lowered.startswith("cassette:"):
        return _cassette(choice.split(":", 1)[1].strip(), environ, model, kwargs)
    if lowered == "typesafe":
        return HTTPBackend.typesafe(_key(environ, "TYPESAFE_API_KEY", lowered), model=model, **kwargs)
    if lowered == "openrouter_systemone":
        return HTTPBackend.openrouter(_key(environ, "OPENROUTER_API_KEY", lowered), model=model, **kwargs)
    if lowered == "openrouter_decisions":
        return HTTPBackend.openrouter_decisions(_key(environ, "OPENROUTER_API_KEY", lowered), model=model, **kwargs)
    if lowered == "simulator":
        return _simulator(model)
    raise BackendConfigError(
        f"unknown JEVTOOLS_BACKEND {choice!r} (expected {' | '.join(BACKEND_NAMES)} | cassette:<path>)"
    )


def _key(environ: Mapping[str, str], variable: str, backend: str) -> str:
    key = environ.get(variable) or ""
    if not key:
        raise BackendConfigError(f"JEVTOOLS_BACKEND={backend} needs {variable}")
    return key


def _live(environ: Mapping[str, str], model: str | None, kwargs: dict[str, Any]) -> Backend | None:
    if environ.get("TYPESAFE_API_KEY"):
        return HTTPBackend.typesafe(environ["TYPESAFE_API_KEY"], model=model, **kwargs)
    if environ.get("OPENROUTER_API_KEY"):
        return HTTPBackend.openrouter_decisions(environ["OPENROUTER_API_KEY"], model=model, **kwargs)
    return None


def _simulator(model: str | None) -> LexicalSimulator:
    return LexicalSimulator(model=model) if model else LexicalSimulator()


def _cassette(path: str, environ: Mapping[str, str], model: str | None, kwargs: dict[str, Any]) -> Cassette:
    if not path:
        raise BackendConfigError("JEVTOOLS_BACKEND=cassette:<path> needs a path")
    mode = (environ.get("JEVTOOLS_CASSETTE_MODE") or "replay").strip().lower()
    if mode not in MODES:
        raise BackendConfigError(f"unknown JEVTOOLS_CASSETTE_MODE {mode!r} (expected one of {', '.join(MODES)})")
    cassette_mode: CassetteMode = mode  # type: ignore[assignment]
    if cassette_mode == "replay":
        return Cassette(path, "replay", model=model)
    inner = _live(environ, model, kwargs)
    if inner is None:
        raise BackendConfigError(f"a {mode!r} cassette needs TYPESAFE_API_KEY or OPENROUTER_API_KEY")
    return Cassette(path, cassette_mode, inner)


__all__ = ["BACKEND_NAMES", "OFFLINE_WARNING", "auto"]
