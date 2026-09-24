"""External ``x-jev`` annotations: sidecar files and ``jt.hints`` (spec §3.2 layer 3, §7.1).

A sidecar maps ``tool`` to tool-level keys and ``tool.param`` (``tool.param.sub``, ``tool.list[].field``) to
parameter-level keys, leaving third-party schemas untouched::

    {"send_email": {"risk": "external"}, "send_email.to": {"source": "contacts"}}

The structured form ``{"tools": {...flat mapping...}, "sources": [...]}`` also carries source registrations,
which :mod:`jevtools.sources` consumes. YAML sidecars need PyYAML (the ``yaml`` extra).
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from jevtools.errors import CatalogError


class Sidecar(BaseModel):
    """Parsed sidecar or hints: ``entries`` keyed ``tool`` / ``tool.param``; optional ``sources`` registrations."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    entries: dict[str, dict[str, Any]] = Field(default_factory=dict)
    sources: list[dict[str, Any]] = Field(default_factory=list)

    def for_tool(self, name: str, *, all_names: Sequence[str] = ()) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
        """``(tool-level keys, {param path key: keys})`` for tool ``name``.

        Tool names may contain dots, so an entry belongs to the *longest* tool name (among ``all_names``) that
        prefixes it.
        """
        names = sorted({*all_names, name}, key=len, reverse=True)
        tool: dict[str, Any] = {}
        params: dict[str, dict[str, Any]] = {}
        for key, value in self.entries.items():
            owner = next((n for n in names if key == n or key.startswith(n + ".")), None)
            if owner != name:
                continue
            if key == name:
                tool = dict(value)
            else:
                params[key[len(name) + 1 :]] = dict(value)
        return tool, params

    def override(self, other: Sidecar | None) -> Sidecar:
        """A sidecar where ``other``'s keys win over this one's (key-wise per entry)."""
        if other is None:
            return self
        entries = {k: dict(v) for k, v in self.entries.items()}
        for key, value in other.entries.items():
            entries[key] = {**entries.get(key, {}), **value}
        return Sidecar(entries=entries, sources=[*self.sources, *other.sources])


def parse_sidecar(data: Mapping[str, Any]) -> Sidecar:
    """Build a :class:`Sidecar` from its flat or structured JSON form."""
    if not isinstance(data, Mapping):
        raise CatalogError("a sidecar must be a JSON object")
    structured = set(data) <= {"tools", "sources"} and isinstance(data.get("tools", {}), Mapping)
    if structured and ("tools" in data or "sources" in data):
        entries, sources = data.get("tools", {}), data.get("sources", [])
    else:
        entries, sources = data, []
    for key, value in entries.items():
        if not isinstance(value, Mapping):
            raise CatalogError(f"sidecar entry {key!r} must be an object of x-jev keys")
    if not isinstance(sources, list):
        raise CatalogError("sidecar 'sources' must be a list")
    return Sidecar(entries={k: dict(v) for k, v in entries.items()}, sources=[dict(s) for s in sources])


def load_sidecar(path: str | os.PathLike[str]) -> Sidecar:
    """Load ``jevtools.json`` or ``jevtools.yaml``/``.yml`` (YAML needs PyYAML)."""
    file = Path(path)
    text = file.read_text(encoding="utf-8")
    if file.suffix.lower() in (".yaml", ".yml"):
        try:
            import yaml  # type: ignore[import-untyped]
        except ModuleNotFoundError as exc:
            raise CatalogError("YAML sidecars need PyYAML: pip install 'jevtools[yaml]'") from exc
        data = yaml.safe_load(text) or {}
    else:
        data = json.loads(text)
    return parse_sidecar(data)


def hints(mapping: Mapping[str, Mapping[str, Any]]) -> Sidecar:
    """External hints in code, same as a sidecar: ``jt.hints({"send_email.to": {"source": "contacts"}})``."""
    return parse_sidecar(mapping)


def coerce_sidecar(value: Sidecar | Mapping[str, Any] | str | os.PathLike[str] | None) -> Sidecar | None:
    """Accept a :class:`Sidecar`, a mapping, or a path."""
    if value is None or isinstance(value, Sidecar):
        return value
    if isinstance(value, Mapping):
        return parse_sidecar(value)
    return load_sidecar(value)


__all__ = ["Sidecar", "coerce_sidecar", "hints", "load_sidecar", "parse_sidecar"]
