"""``jevtools serve``: the OpenAI-compatible proxy (spec §7.2.4).

:class:`ServeConfig` (``jevtools.toml``) needs no extra; :func:`create_app` and :func:`run` need the ``serve``
extra (``starlette``, ``uvicorn``) and are imported on first use. Pending CONFIRM/CLARIFY handles live in a
:class:`PendingStore` (the protocol the adapters use too, :mod:`jevtools.adapters.pending`); pass your own
(e.g. Redis-backed) to ``create_app(store=…)`` to share it across workers.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from jevtools.adapters.pending import InMemoryPendingStore, PendingStore
from jevtools.serve.config import BackendConfig, EndpointConfig, ServeConfig, SourceConfig, build_sources

if TYPE_CHECKING:
    from jevtools.serve.app import create_app, run

__all__ = [
    "BackendConfig",
    "EndpointConfig",
    "InMemoryPendingStore",
    "PendingStore",
    "ServeConfig",
    "SourceConfig",
    "build_sources",
    "create_app",
    "run",
]


def __getattr__(name: str) -> Any:
    if name in ("create_app", "run"):
        from jevtools.serve import app

        return getattr(app, name)
    raise AttributeError(f"module 'jevtools.serve' has no attribute {name!r}")
