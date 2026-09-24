"""Small shims so the package runs on Python 3.10 as well as 3.11+."""

from __future__ import annotations

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


__all__ = ["StrEnum", "load_toml"]
