"""Normalizers (spec §4.3): deterministic, versioned; the trace records ``name@version``.

Every candidate value is ``normalize(raw)`` for the kind's normalizer at pool time, so decoding copies the
elected value verbatim (I1). The ``normalizer`` a binding records names the function that produced its value:
:data:`NORMALIZERS` maps every recorded ``name@version`` to it (``get_normalizer``), for replays and external
verifiers. ``jt.verify`` checks bindings against the stored options' values (§3.9 step 3); it does not re-run them.
"""

from __future__ import annotations

import posixpath
import re
import unicodedata
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import date, datetime, time
from decimal import ROUND_HALF_EVEN, Decimal, InvalidOperation, localcontext
from typing import Any

from jevtools.extract.catalogs import minor_units
from jevtools.extract.numbers import convert, decimal_str
from jevtools.extract.patterns import normalize_email

_WRAPPING_QUOTES = ('"', "'", "“", "”", "‘", "’", "«", "»", "„")
_TRAILING = ".,;:!?"


class NormalizationError(ValueError):
    """The raw value cannot be normalized for this slot (the candidate is dropped at pool time)."""


@dataclass(frozen=True)
class Normalizer:
    """A named, versioned normalizer ``fn(raw, schema) -> value``."""

    name: str
    fn: Callable[[Any, Mapping[str, Any]], Any]

    def __call__(self, raw: Any, schema: Mapping[str, Any] | None = None) -> Any:
        return self.fn(raw, schema or {})


def _type(schema: Mapping[str, Any]) -> str | None:
    types = schema.get("type")
    if isinstance(types, list):
        return next((t for t in types if t != "null"), None)
    return types if isinstance(types, str) else None


def normalize_string(raw: Any, schema: Mapping[str, Any] | None = None) -> str:
    """NFC; trim; collapse internal whitespace (``string@1``)."""
    return " ".join(unicodedata.normalize("NFC", str(raw)).split())


def normalize_span(raw: Any, schema: Mapping[str, Any] | None = None) -> str:
    """``string@1`` plus: strip wrapping quotes and trailing ``.,;:!?``; case preserved (``span@1``)."""
    text = normalize_string(raw)
    while True:
        stripped = text.rstrip(_TRAILING).rstrip()
        if len(stripped) >= 2 and stripped[0] in _WRAPPING_QUOTES and stripped[-1] in _WRAPPING_QUOTES:
            stripped = stripped[1:-1].strip()
        if stripped == text:
            return text
        text = stripped


def normalize_text(
    raw: Any, schema: Mapping[str, Any] | None = None, *, capital: bool = True, punctuation: bool = True
) -> str:
    """``text@1``: NFC, trim, collapse spaces (newlines kept), sentence-initial capital, final punctuation."""
    lines = [" ".join(line.split()) for line in unicodedata.normalize("NFC", str(raw)).strip().split("\n")]
    text = "\n".join(lines)
    if capital and text[:1].isalpha():
        text = text[:1].upper() + text[1:]
    if punctuation and text and text[-1] not in ".!?…⟩":
        text += "."
    return text


def normalize_title(raw: Any, schema: Mapping[str, Any] | None = None) -> str:
    """Cosmetic titles and subjects: ``text@1`` without added final punctuation."""
    return normalize_text(raw, schema, punctuation=False)


def normalize_query(raw: Any, schema: Mapping[str, Any] | None = None) -> str:
    """Queries: whitespace only (no capitalization, no punctuation)."""
    return normalize_text(raw, schema, capital=False, punctuation=False)


def to_decimal(raw: Any) -> Decimal:
    if isinstance(raw, Decimal):
        return raw
    try:
        return Decimal(str(raw))
    except InvalidOperation as exc:
        raise NormalizationError(f"not a number: {raw!r}") from exc


def emit_number(value: Decimal, schema: Mapping[str, Any]) -> Any:
    """A decimal as the schema's type: ``integer`` → int (must be integral), ``string`` → decimal text,
    else int when integral, float otherwise."""
    kind = _type(schema)
    integral = value == value.to_integral_value()
    if kind == "integer":
        if not integral:
            raise NormalizationError(f"{value} is not an integer")
        return int(value)
    if kind == "string":
        return format(value, "f")
    return int(value) if integral else float(value)


def normalize_quantity(
    raw: Any, schema: Mapping[str, Any] | None = None, *, from_unit: str | None = None, to_unit: str | None = None
) -> Any:
    """``quantity@1``: ``Decimal`` of the raw value, converted to the declared unit, cast to the schema type."""
    value = to_decimal(raw)
    if from_unit and to_unit and from_unit != to_unit:
        converted = convert(value, from_unit, to_unit)
        if converted is None:
            raise NormalizationError(f"cannot convert {from_unit} to {to_unit}")
        value = converted
    value = Decimal(decimal_str(value))
    return emit_number(value, schema or {})


def quantize_money(raw: Any, currency: str | None) -> Decimal:
    """``Decimal`` quantized half-even to the currency's ISO 4217 minor unit (2 when unknown)."""
    places = minor_units(currency)
    value = to_decimal(raw)
    with localcontext() as ctx:  # a 29+ digit amount must not raise InvalidOperation under the 28-digit default
        ctx.prec = max(ctx.prec, value.adjusted() + places + 3)
        return value.quantize(Decimal(1).scaleb(-places), rounding=ROUND_HALF_EVEN)


def normalize_money(raw: Any, schema: Mapping[str, Any] | None = None, *, currency: str | None = None) -> Any:
    """``money@1``: quantized to the ISO minor unit (``250`` → ``250.00`` CHF; JPY → 0 decimals), emitted as the
    schema's type (string ``"250.00"`` or number)."""
    amount = quantize_money(raw, currency)
    if _type(schema or {}) == "string":
        return format(amount, "f")
    return emit_number(amount, schema or {})


def money_display(raw: Any, currency: str | None) -> str:
    """Label of a money amount: always with the minor unit (``250.00``)."""
    return format(quantize_money(raw, currency), "f")


def normalize_temporal(raw: Any, schema: Mapping[str, Any] | None = None) -> str:
    """``temporal.iso8601@1``: aware datetime → ISO 8601 with offset; date → ``YYYY-MM-DD``; time → ``HH:MM:SS``."""
    fmt = (schema or {}).get("format")
    if isinstance(raw, datetime):
        if fmt == "date":
            return raw.date().isoformat()
        if fmt == "time":
            return raw.strftime("%H:%M:%S")
        if raw.tzinfo is None:
            raise NormalizationError("a datetime value needs a time zone")
        return raw.isoformat(timespec="seconds")
    if isinstance(raw, date):
        return raw.isoformat()
    if isinstance(raw, time):
        return raw.strftime("%H:%M:%S")
    return str(raw)


def normalize_duration(raw: Any, schema: Mapping[str, Any] | None = None) -> Any:
    """Durations (``temporal.duration@1``): minutes as seconds → ISO 8601 (``PT45M``) for ``format: duration``,
    else a number of minutes."""
    seconds = to_decimal(raw)
    if (schema or {}).get("format") == "duration" or _type(schema or {}) == "string":
        return iso_duration(seconds)
    return emit_number(Decimal(decimal_str(seconds / 60)), schema or {})


def iso_duration(seconds: Decimal) -> str:
    """``PT1H30M`` from a number of seconds (days as ``P<n>D``)."""
    total = int(seconds)
    days, rest = divmod(total, 86400)
    hours, rest = divmod(rest, 3600)
    minutes, secs = divmod(rest, 60)
    out = "P" + (f"{days}D" if days else "")
    clock = (f"{hours}H" if hours else "") + (f"{minutes}M" if minutes else "") + (f"{secs}S" if secs else "")
    if clock:
        out += "T" + clock
    return out if out != "P" else "PT0S"


def normalize_email_value(raw: Any, schema: Mapping[str, Any] | None = None) -> str:
    """``email@1``: lowercase the domain, keep the local part verbatim."""
    return normalize_email(normalize_string(raw))


_ESCAPING_PREFIX = re.compile(r"^(?:~|\$\{?\w|%\w+%|[A-Za-z]:(?:/|$))")
"""Home (``~/.ssh``), environment (``$HOME/x``, ``%APPDATA%``) and drive (``C:/Windows``) prefixes."""


def normalize_path(raw: Any, schema: Mapping[str, Any] | None = None, *, known: bool = False) -> str:
    """``path@1``: POSIX-normalize; ``..``, absolute, home (``~``), environment (``$HOME``) and drive (``C:/``)
    paths are rejected unless present in the source (``known``)."""
    text = normalize_string(raw).replace("\\", "/")
    path = posixpath.normpath(text)
    escapes = path.startswith("/") or path == ".." or path.startswith("../") or "/../" in f"/{path}/"
    if not known and (escapes or _ESCAPING_PREFIX.match(path)):
        raise NormalizationError(f"path {text!r} escapes the workspace")
    return path


def normalize_ref(raw: Any, schema: Mapping[str, Any] | None = None) -> Any:
    """``ref@1``: the source row's key, verbatim (never the label)."""
    return raw


def normalize_list(raw: Any, schema: Mapping[str, Any] | None = None) -> list[Any]:
    """``list@1``: dedupe preserving mention order; apply ``maxItems``."""
    out: list[Any] = []
    for item in raw or ():
        if item not in out:
            out.append(item)
    limit = (schema or {}).get("maxItems")
    return out[:limit] if isinstance(limit, int) else out


def identity(raw: Any, schema: Mapping[str, Any] | None = None) -> Any:
    """``enum@1``: identity."""
    return raw


def normalize_flag(raw: Any, schema: Mapping[str, Any] | None = None) -> bool:
    """``flag@1``: bool."""
    if isinstance(raw, str):
        return raw.strip().lower() in ("true", "yes", "1")
    return bool(raw)


NORMALIZERS: dict[str, Normalizer] = {
    n.name: n
    for n in (
        Normalizer("string@1", normalize_string),
        Normalizer("span@1", normalize_span),
        Normalizer("text@1", normalize_text),
        Normalizer("text.title@1", normalize_title),
        Normalizer("text.query@1", normalize_query),
        Normalizer("text.template@1", normalize_title),
        Normalizer("quantity@1", normalize_quantity),
        Normalizer("money@1", normalize_money),
        Normalizer("temporal.iso8601@1", normalize_temporal),
        Normalizer("temporal.duration@1", normalize_duration),
        Normalizer("email@1", normalize_email_value),
        Normalizer("path@1", normalize_path),
        Normalizer("ref@1", normalize_ref),
        Normalizer("list@1", normalize_list),
        Normalizer("enum@1", identity),
        Normalizer("record@1", identity),
        Normalizer("flag@1", normalize_flag),
    )
}
"""Every normalizer by ``name@version``."""

_NAME_RE = re.compile(r"^[a-z0-9_.]+@\d+$")


def get_normalizer(name: str) -> Normalizer:
    """A normalizer by ``name@version`` (``KeyError`` for unknown names)."""
    if not _NAME_RE.match(name):
        raise KeyError(f"malformed normalizer name {name!r} (expected name@version)")
    return NORMALIZERS[name]


__all__ = [
    "NORMALIZERS",
    "NormalizationError",
    "Normalizer",
    "emit_number",
    "get_normalizer",
    "identity",
    "iso_duration",
    "money_display",
    "normalize_duration",
    "normalize_email_value",
    "normalize_flag",
    "normalize_list",
    "normalize_money",
    "normalize_path",
    "normalize_quantity",
    "normalize_query",
    "normalize_ref",
    "normalize_span",
    "normalize_string",
    "normalize_temporal",
    "normalize_text",
    "normalize_title",
    "quantize_money",
    "to_decimal",
]
