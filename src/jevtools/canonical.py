"""Canonical JSON serialization and hashing (spec §3.1).

Canonical JSON is used for hashing, golden fixtures and cassettes:

- UTF-8; every string (keys included) NFC-normalized; non-ASCII kept as is (``ensure_ascii=false``).
- Separators ``,`` and ``:`` with no whitespace.
- Object keys keep their **insertion order** (the normative order of each document); they are never sorted.
- Numbers are printed as the shortest round-trip decimal with no exponent. Integral floats print without a
  fractional part (``1.0`` → ``1``), so equal JSON numbers hash equally in every port.
- With ``round_floats=True`` (Decision and Trace documents) every float is first rounded half-even to 4 decimals.
- Digests carry a ``sha256:`` prefix.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import unicodedata
from collections.abc import Mapping
from datetime import date, datetime, time
from decimal import ROUND_HALF_EVEN, Decimal
from enum import Enum
from typing import Any

from pydantic import BaseModel

_FOUR_DP = Decimal("0.0001")
_MAX_EXACT_INT = 2**53


def nfc(s: str) -> str:
    """Return ``s`` in Unicode normalization form C."""
    return unicodedata.normalize("NFC", s)


def round4(x: float) -> float:
    """Round half-even to 4 decimals, based on the shortest repr of ``x`` (``0.12345`` → ``0.1234``)."""
    value = float(x)
    if not math.isfinite(value):
        raise ValueError(f"cannot round non-finite number {value!r}")
    rounded = float(Decimal(repr(value)).quantize(_FOUR_DP, rounding=ROUND_HALF_EVEN))
    return rounded + 0.0  # normalizes -0.0 to 0.0


def format_number(x: int | float | Decimal) -> str:
    """Print a JSON number: shortest round-trip decimal, no exponent, integral floats without ``.0``."""
    if isinstance(x, bool):
        raise TypeError("booleans are not numbers")
    if isinstance(x, int):
        return str(x)
    if isinstance(x, Decimal):
        if not x.is_finite():
            raise ValueError(f"cannot serialize non-finite number {x!r}")
        if x == x.to_integral_value():
            return str(int(x))
        return format(x.normalize(), "f")
    if not math.isfinite(x):
        raise ValueError(f"cannot serialize non-finite number {x!r}")
    if x == int(x) and abs(x) < _MAX_EXACT_INT:
        return str(int(x))
    text = repr(x)
    if "e" in text or "E" in text:
        text = format(Decimal(text), "f")
    return text


def jsonable(obj: Any) -> Any:
    """Convert ``obj`` to JSON-native Python values (dict/list/str/int/float/bool/None), keeping key order.

    Pydantic models are dumped in JSON mode, enums become their values, datetimes their ISO form, dataclasses
    their fields in declaration order and tuples lists. Sets are rejected because they have no order.
    """
    if obj is None or isinstance(obj, (bool, int, float, str, Decimal)):
        return obj
    if isinstance(obj, Enum):
        return jsonable(obj.value)
    if isinstance(obj, BaseModel):
        return jsonable(obj.model_dump(mode="json"))
    if isinstance(obj, Mapping):
        return {_key(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    if isinstance(obj, (datetime, date, time)):
        return obj.isoformat()
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: jsonable(getattr(obj, f.name)) for f in dataclasses.fields(obj)}
    raise TypeError(f"cannot serialize {type(obj).__name__} to canonical JSON")


def _key(key: Any) -> str:
    if isinstance(key, Enum):
        key = key.value
    if isinstance(key, str):
        return key
    if isinstance(key, int) and not isinstance(key, bool):
        return str(key)
    raise TypeError(f"JSON object keys must be strings, got {type(key).__name__}")


def _encode(value: Any, out: list[str], round_floats: bool) -> None:
    if value is None:
        out.append("null")
    elif value is True:
        out.append("true")
    elif value is False:
        out.append("false")
    elif isinstance(value, str):
        out.append(json.dumps(nfc(value), ensure_ascii=False))
    elif isinstance(value, float):
        out.append(format_number(round4(value) if round_floats else value))
    elif isinstance(value, (int, Decimal)):
        out.append(format_number(value))
    elif isinstance(value, dict):
        out.append("{")
        for i, (k, v) in enumerate(value.items()):
            if i:
                out.append(",")
            out.append(json.dumps(nfc(k), ensure_ascii=False))
            out.append(":")
            _encode(v, out, round_floats)
        out.append("}")
    elif isinstance(value, list):
        out.append("[")
        for i, v in enumerate(value):
            if i:
                out.append(",")
            _encode(v, out, round_floats)
        out.append("]")
    else:  # pragma: no cover - jsonable() already rejected everything else
        raise TypeError(f"cannot serialize {type(value).__name__}")


def canonical_str(obj: Any, *, round_floats: bool = False) -> str:
    """Canonical JSON as text (see the module docstring)."""
    out: list[str] = []
    _encode(jsonable(obj), out, round_floats)
    return "".join(out)


def canonical_json(obj: Any, *, round_floats: bool = False) -> bytes:
    """Canonical JSON as UTF-8 bytes (spec §3.1). ``round_floats`` rounds floats half-even to 4 decimals."""
    return canonical_str(obj, round_floats=round_floats).encode("utf-8")


def sha256_hex(data: bytes | str) -> str:
    """Hex SHA-256 of raw bytes (strings are NFC-normalized and UTF-8 encoded first)."""
    raw = nfc(data).encode("utf-8") if isinstance(data, str) else data
    return hashlib.sha256(raw).hexdigest()


def sha256_of(obj: Any, *, round_floats: bool = False) -> str:
    """``"sha256:<hex>"`` of the canonical JSON of ``obj``."""
    return "sha256:" + hashlib.sha256(canonical_json(obj, round_floats=round_floats)).hexdigest()


def short_id(prefix: str, *parts: str, n: int = 16) -> str:
    """``prefix + sha256(concatenated parts)[:n]``: the id scheme of spec §3.10 and §6.5."""
    return prefix + sha256_hex("".join(parts))[:n]


__all__ = [
    "canonical_json",
    "canonical_str",
    "format_number",
    "jsonable",
    "nfc",
    "round4",
    "sha256_hex",
    "sha256_of",
    "short_id",
]
