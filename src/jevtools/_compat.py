"""Small shims so the package runs on Python 3.10 as well as 3.11+."""

from __future__ import annotations

import asyncio
import inspect
import sys
from enum import Enum
from typing import Any, cast

if sys.version_info >= (3, 11):  # pragma: no cover - version dependent
    from enum import StrEnum
else:  # pragma: no cover - version dependent

    class StrEnum(str, Enum):
        """Backport of :class:`enum.StrEnum`: members are strings and print as their value."""

        def __str__(self) -> str:
            return str(self.value)


def load_toml(text: str) -> dict[str, Any]:
    """Parse TOML text with :mod:`tomllib` (3.11+) or ``tomli`` (3.10)."""
    if sys.version_info >= (3, 11):
        import tomllib

        return tomllib.loads(text)
    try:  # pragma: no cover - Python 3.10 only
        import tomli  # type: ignore[import-not-found, unused-ignore]
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise ImportError("Reading TOML on Python 3.10 needs `pip install tomli`.") from exc
    return cast("dict[str, Any]", tomli.loads(text))  # pragma: no cover


async def _await(value: Any) -> Any:
    return await value


def run_sync(value: Any, hint: str) -> Any:
    """Resolve a possibly awaitable ``value`` from sync code: a non-awaitable is returned unchanged; an awaitable
    runs on a fresh event loop (:func:`asyncio.run`). Inside a running loop that is refused: a coroutine is closed
    (never left un-awaited) and ``RuntimeError(hint)`` is raised (``hint`` names the async API to use instead)."""
    if not inspect.isawaitable(value):
        return value
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(_await(value))
    if inspect.iscoroutine(value):
        value.close()
    raise RuntimeError(hint)


__all__ = ["StrEnum", "load_toml", "run_sync"]
