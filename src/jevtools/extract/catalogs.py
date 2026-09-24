"""Packaged reference data under ``jevtools/extract/data`` (spec §9): ISO 4217 (with minor units), ISO 3166,
ISO 639, the compact city gazetteer and the template packs.

Catalog files are JSON arrays of ``{"value", "text"?, "aliases"?}`` objects (the format the ``enum`` resolver
loads); ``iso4217`` entries add ``minor`` (the ISO minor unit), cities are ``{"name", "country", "admin"?,
"aliases"?, "pop"?}`` objects.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from functools import cache
from importlib import resources
from typing import Any

from jevtools.extract.tokens import fold

DEFAULT_MINOR_UNITS = 2


@cache
def load_data(name: str) -> Any:
    """Parse ``jevtools/extract/data/<name>.json`` (``name`` may contain ``/``: ``packs/email``)."""
    node = resources.files("jevtools").joinpath("extract").joinpath("data")
    for part in f"{name}.json".split("/"):
        node = node.joinpath(part)
    try:
        return json.loads(node.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise LookupError(f"no packaged data file jevtools/extract/data/{name}.json") from None


@cache
def currencies() -> Mapping[str, Mapping[str, Any]]:
    """ISO 4217 entries by code."""
    return {entry["value"]: entry for entry in load_data("iso4217")}


def minor_units(code: str | None) -> int:
    """ISO 4217 minor unit of a currency (``CHF`` → 2, ``JPY`` → 0); 2 for unknown codes."""
    if code is None:
        return DEFAULT_MINOR_UNITS
    entry = currencies().get(code.upper())
    return int(entry.get("minor", DEFAULT_MINOR_UNITS)) if entry else DEFAULT_MINOR_UNITS


@cache
def countries() -> Mapping[str, Mapping[str, Any]]:
    """ISO 3166-1 alpha-2 entries by code."""
    return {entry["value"]: entry for entry in load_data("iso3166")}


def country_name(code: str) -> str:
    """English short name of a country code (the code itself when unknown)."""
    entry = countries().get(code.upper())
    return str(entry["text"]) if entry else code


@dataclass(frozen=True)
class Place:
    """A gazetteer entry (a city, or a country from ISO 3166)."""

    name: str
    country: str
    admin: str | None = None
    kind: str = "city"
    aliases: tuple[str, ...] = ()
    pop: int = 0
    """Population in thousands (for ordering readings; approximate)."""

    @property
    def canonical(self) -> str:
        """Canonical value used with ``canon``: ``Zürich, CH`` / ``Zurich, Ontario, CA``; countries by name."""
        if self.kind == "country":
            return self.name
        return ", ".join(p for p in (self.name, self.admin, self.country) if p)

    def describe(self) -> str:
        """``a city in Switzerland`` / ``a city in Ontario, Canada``."""
        if self.kind == "country":
            return "a country"
        where = ", ".join(p for p in (self.admin, country_name(self.country)) if p)
        return f"a city in {where}"


@cache
def gazetteer() -> Mapping[str, tuple[Place, ...]]:
    """Folded name/alias → places (cities from ``cities.json`` plus countries from ``iso3166``)."""
    index: dict[str, list[Place]] = {}
    for raw in load_data("cities"):
        place = Place(
            name=raw["name"],
            country=raw["country"],
            admin=raw.get("admin"),
            aliases=tuple(raw.get("aliases", ())),
            pop=int(raw.get("pop", 0)),
        )
        for term in dict.fromkeys(fold(t) for t in (place.name, *place.aliases)):
            index.setdefault(term, []).append(place)
    for code, raw in countries().items():
        place = Place(name=raw["text"], country=code, kind="country", aliases=tuple(raw.get("aliases", ())))
        for term in (place.name, *place.aliases):
            bucket = index.setdefault(fold(term), [])
            if all(p.kind != "country" or p.country != code for p in bucket):
                bucket.append(place)
    return {key: tuple(sorted(places, key=lambda p: -p.pop)) for key, places in index.items()}


@cache
def template_pack(name: str) -> tuple[Mapping[str, Any], ...]:
    """Entries of a template pack (``email.subject`` → ``packs/email.json`` key ``subject``)."""
    family, _, member = name.partition(".")
    data = load_data(f"packs/{family}")
    return tuple(data.get(member, ()))


__all__ = [
    "DEFAULT_MINOR_UNITS",
    "Place",
    "countries",
    "country_name",
    "currencies",
    "gazetteer",
    "load_data",
    "minor_units",
    "template_pack",
]
