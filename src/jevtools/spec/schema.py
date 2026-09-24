"""A small, dependency-free JSON Schema validator for the keywords jevtools relies on (spec §9 ``schema.py``).

Supported: ``type``, ``enum``, ``const``, ``format`` (asserted), ``pattern``, ``minimum``/``maximum``/
``exclusiveMinimum``/``exclusiveMaximum``/``multipleOf``, ``minLength``/``maxLength``, ``items``/``prefixItems``/
``minItems``/``maxItems``/``uniqueItems``, ``required``/``properties``/``patternProperties``/
``additionalProperties``/``minProperties``/``maxProperties``, ``oneOf``/``anyOf``/``allOf``/``not``,
``if``/``then``/``else``, ``dependentRequired`` and local ``$ref`` (``#/...``). ``x-jev`` and unknown keywords
are ignored. Formats are asserted because an emitted argument must be usable, not merely annotated.
"""

from __future__ import annotations

import ipaddress
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal, InvalidOperation
from typing import Any

Path = tuple[str | int, ...]


@dataclass(frozen=True)
class SchemaError:
    """One validation failure at ``path`` (argument path) for ``keyword``."""

    path: Path
    keyword: str
    message: str

    def __str__(self) -> str:
        where = ".".join(str(p) for p in self.path) or "<root>"
        return f"{where}: {self.keyword}: {self.message}"


# --------------------------------------------------------------------------------------------------------------------
# Formats
# --------------------------------------------------------------------------------------------------------------------

_EMAIL = re.compile(r"[^@\s]+@[^@\s.]+(\.[^@\s.]+)+")
_URI = re.compile(r"[A-Za-z][A-Za-z0-9+.\-]*:[^\s]*")
_UUID = re.compile(r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}")
_HOSTNAME = re.compile(
    r"(?=.{1,253}$)([A-Za-z0-9]([A-Za-z0-9\-]{0,61}[A-Za-z0-9])?)(\.[A-Za-z0-9]([A-Za-z0-9\-]{0,61}[A-Za-z0-9])?)*"
)
_DATE = re.compile(r"\d{4}-\d{2}-\d{2}")
_TIME = re.compile(r"\d{2}:\d{2}(:\d{2}(\.\d+)?)?([Zz]|[+\-]\d{2}:\d{2})?")
_DATETIME = re.compile(r"\d{4}-\d{2}-\d{2}[Tt ]\d{2}:\d{2}(:\d{2}(\.\d+)?)?([Zz]|[+\-]\d{2}:\d{2})")
_DURATION = re.compile(r"P(?!$)(\d+Y)?(\d+M)?(\d+W)?(\d+D)?(T(?=\d)(\d+H)?(\d+M)?(\d+(\.\d+)?S)?)?")
_DECIMAL = re.compile(r"-?\d+(\.\d+)?")


def _is_ip(text: str, version: int) -> bool:
    try:
        return ipaddress.ip_address(text).version == version
    except ValueError:
        return False


def _is_date(text: str) -> bool:
    if not _DATE.fullmatch(text):
        return False
    try:
        date.fromisoformat(text)
    except ValueError:
        return False
    return True


def _is_datetime(text: str) -> bool:
    if not _DATETIME.fullmatch(text):
        return False
    try:
        datetime.fromisoformat(text.replace("Z", "+00:00").replace("z", "+00:00"))
    except ValueError:
        return False
    return True


FORMAT_CHECKS: dict[str, Callable[[str], bool]] = {
    "email": lambda s: _EMAIL.fullmatch(s) is not None,
    "uri": lambda s: _URI.fullmatch(s) is not None,
    "uuid": lambda s: _UUID.fullmatch(s) is not None,
    "ipv4": lambda s: _is_ip(s, 4),
    "ipv6": lambda s: _is_ip(s, 6),
    "hostname": lambda s: _HOSTNAME.fullmatch(s) is not None,
    "date": _is_date,
    "time": lambda s: _TIME.fullmatch(s) is not None,
    "date-time": _is_datetime,
    "duration": lambda s: _DURATION.fullmatch(s) is not None,
    "decimal": lambda s: _DECIMAL.fullmatch(s) is not None,
}
"""Asserted string formats; unknown formats are ignored."""


# --------------------------------------------------------------------------------------------------------------------
# JSON value semantics
# --------------------------------------------------------------------------------------------------------------------


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float, Decimal)) and not isinstance(value, bool)


def _is_integer(value: Any) -> bool:
    if isinstance(value, bool):
        return False
    if isinstance(value, int):
        return True
    if isinstance(value, float):
        return value.is_integer()
    if isinstance(value, Decimal):
        return value.is_finite() and value == value.to_integral_value()
    return False


_TYPE_CHECKS: dict[str, Callable[[Any], bool]] = {
    "null": lambda v: v is None,
    "boolean": lambda v: isinstance(v, bool),
    "integer": _is_integer,
    "number": _is_number,
    "string": lambda v: isinstance(v, str),
    "array": lambda v: isinstance(v, (list, tuple)),
    "object": lambda v: isinstance(v, Mapping),
}


def json_equal(a: Any, b: Any) -> bool:
    """JSON equality: ``1 == 1.0`` but ``true != 1``; arrays by position, objects by key set."""
    if isinstance(a, bool) or isinstance(b, bool):
        return isinstance(a, bool) and isinstance(b, bool) and a == b
    if _is_number(a) and _is_number(b):
        return Decimal(str(a)) == Decimal(str(b))
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(json_equal(x, y) for x, y in zip(a, b, strict=True))
    if isinstance(a, Mapping) and isinstance(b, Mapping):
        return a.keys() == b.keys() and all(json_equal(a[k], b[k]) for k in a)
    return type(a) is type(b) and a == b


def _decimal(value: Any) -> Decimal:
    return Decimal(str(value))


# --------------------------------------------------------------------------------------------------------------------
# $ref resolution
# --------------------------------------------------------------------------------------------------------------------


def resolve_pointer(root: Mapping[str, Any], ref: str) -> Any:
    """Resolve a local JSON pointer ``#/a/b`` against ``root``; raises ``KeyError`` for anything else."""
    if not ref.startswith("#"):
        raise KeyError(f"only local $ref is supported: {ref!r}")
    node: Any = root
    for raw in ref[1:].split("/")[1:] if ref != "#" else ():
        part = raw.replace("~1", "/").replace("~0", "~")
        if isinstance(node, Mapping) and part in node:
            node = node[part]
        elif isinstance(node, list) and part.isdigit() and int(part) < len(node):
            node = node[int(part)]
        else:
            raise KeyError(f"unresolvable $ref {ref!r}")
    return node


def inline_refs(schema: Any, root: Mapping[str, Any] | None = None, *, _depth: int = 0) -> Any:
    """Return a copy of ``schema`` with local ``$ref`` replaced by their targets (siblings of ``$ref`` win).

    ``$defs``/``definitions`` are dropped from the result. Recursion deeper than 32 levels raises ``ValueError``.
    """
    if _depth > 32:
        raise ValueError("schema $ref nesting too deep (recursive schema?)")
    if root is None:
        root = schema if isinstance(schema, Mapping) else {}
    if isinstance(schema, list):
        return [inline_refs(s, root, _depth=_depth + 1) for s in schema]
    if not isinstance(schema, Mapping):
        return schema
    if "$ref" in schema:
        target = inline_refs(resolve_pointer(root, schema["$ref"]), root, _depth=_depth + 1)
        merged = dict(target) if isinstance(target, Mapping) else {}
        merged.update({k: inline_refs(v, root, _depth=_depth + 1) for k, v in schema.items() if k != "$ref"})
        return merged
    return {k: inline_refs(v, root, _depth=_depth + 1) for k, v in schema.items() if k not in ("$defs", "definitions")}


# --------------------------------------------------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------------------------------------------------


class _Validator:
    def __init__(self, root: Mapping[str, Any] | bool) -> None:
        self.root = root if isinstance(root, Mapping) else {}

    def run(self, value: Any, schema: Any, path: Path) -> list[SchemaError]:
        if schema is True or schema is None:
            return []
        if schema is False:
            return [SchemaError(path, "false", "no value is allowed")]
        if not isinstance(schema, Mapping):
            return []
        errors: list[SchemaError] = []
        if "$ref" in schema:
            errors += self.run(value, resolve_pointer(self.root, schema["$ref"]), path)
        for check in (self._type, self._enum, self._const, self._string, self._number, self._array, self._object):
            errors += check(value, schema, path)
        errors += self._combinators(value, schema, path)
        return errors

    def _type(self, value: Any, schema: Mapping[str, Any], path: Path) -> list[SchemaError]:
        types = schema.get("type")
        if types is None:
            return []
        names = [types] if isinstance(types, str) else list(types)
        if any(name in _TYPE_CHECKS and _TYPE_CHECKS[name](value) for name in names):
            return []
        return [SchemaError(path, "type", f"expected {' or '.join(names)}")]

    def _enum(self, value: Any, schema: Mapping[str, Any], path: Path) -> list[SchemaError]:
        if "enum" in schema and not any(json_equal(value, member) for member in schema["enum"]):
            return [SchemaError(path, "enum", f"{value!r} is not one of {schema['enum']!r}")]
        return []

    def _const(self, value: Any, schema: Mapping[str, Any], path: Path) -> list[SchemaError]:
        if "const" in schema and not json_equal(value, schema["const"]):
            return [SchemaError(path, "const", f"expected {schema['const']!r}")]
        return []

    def _string(self, value: Any, schema: Mapping[str, Any], path: Path) -> list[SchemaError]:
        if not isinstance(value, str):
            return []
        errors: list[SchemaError] = []
        if "minLength" in schema and len(value) < schema["minLength"]:
            errors.append(SchemaError(path, "minLength", f"shorter than {schema['minLength']}"))
        if "maxLength" in schema and len(value) > schema["maxLength"]:
            errors.append(SchemaError(path, "maxLength", f"longer than {schema['maxLength']}"))
        if "pattern" in schema and re.search(schema["pattern"], value) is None:
            errors.append(SchemaError(path, "pattern", f"does not match {schema['pattern']!r}"))
        check = FORMAT_CHECKS.get(schema.get("format", ""))
        if check is not None and not check(value):
            errors.append(SchemaError(path, "format", f"not a valid {schema['format']}"))
        return errors

    def _number(self, value: Any, schema: Mapping[str, Any], path: Path) -> list[SchemaError]:
        if not _is_number(value):
            return []
        try:
            number = _decimal(value)
        except InvalidOperation:
            return [SchemaError(path, "type", "not a finite number")]
        errors: list[SchemaError] = []
        bounds: tuple[tuple[str, Callable[[Decimal], bool], str], ...] = (
            ("minimum", lambda b: number < b, "less than"),
            ("maximum", lambda b: number > b, "greater than"),
            ("exclusiveMinimum", lambda b: number <= b, "not greater than"),
            ("exclusiveMaximum", lambda b: number >= b, "not less than"),
        )
        for keyword, violated, words in bounds:
            bound = schema.get(keyword)
            if _is_number(bound) and violated(_decimal(bound)):
                errors.append(SchemaError(path, keyword, f"{words} {bound}"))
        step = schema.get("multipleOf")
        if _is_number(step) and _decimal(step) > 0 and number % _decimal(step) != 0:
            errors.append(SchemaError(path, "multipleOf", f"not a multiple of {step}"))
        return errors

    def _array(self, value: Any, schema: Mapping[str, Any], path: Path) -> list[SchemaError]:
        if not isinstance(value, (list, tuple)):
            return []
        errors: list[SchemaError] = []
        if "minItems" in schema and len(value) < schema["minItems"]:
            errors.append(SchemaError(path, "minItems", f"fewer than {schema['minItems']} items"))
        if "maxItems" in schema and len(value) > schema["maxItems"]:
            errors.append(SchemaError(path, "maxItems", f"more than {schema['maxItems']} items"))
        if schema.get("uniqueItems"):
            for i, item in enumerate(value):
                if any(json_equal(item, other) for other in value[:i]):
                    errors.append(SchemaError((*path, i), "uniqueItems", "duplicate item"))
        prefix = schema.get("prefixItems")
        items = schema.get("items")
        if isinstance(items, list):  # draft-07 tuple form
            prefix, items = items, schema.get("additionalItems", True)
        prefix = prefix or []
        for i, item in enumerate(value):
            sub = prefix[i] if i < len(prefix) else items
            if sub is not None:
                errors += self.run(item, sub, (*path, i))
        return errors

    def _object(self, value: Any, schema: Mapping[str, Any], path: Path) -> list[SchemaError]:
        if not isinstance(value, Mapping):
            return []
        errors: list[SchemaError] = []
        for name in schema.get("required", ()):
            if name not in value:
                errors.append(SchemaError((*path, name), "required", "missing required property"))
        for keyword, words in (("minProperties", "fewer"), ("maxProperties", "more")):
            if keyword in schema:
                bad = len(value) < schema[keyword] if words == "fewer" else len(value) > schema[keyword]
                if bad:
                    errors.append(SchemaError(path, keyword, f"{words} than {schema[keyword]} properties"))
        for name, needed in (schema.get("dependentRequired") or {}).items():
            if name in value:
                for other in needed:
                    if other not in value:
                        errors.append(SchemaError((*path, other), "dependentRequired", f"required when {name} is set"))
        properties = schema.get("properties") or {}
        patterns = schema.get("patternProperties") or {}
        extra = schema.get("additionalProperties", True)
        for name, item in value.items():
            matched = False
            if name in properties:
                matched = True
                errors += self.run(item, properties[name], (*path, name))
            for pattern, sub in patterns.items():
                if re.search(pattern, str(name)):
                    matched = True
                    errors += self.run(item, sub, (*path, name))
            if not matched:
                if extra is False:
                    errors.append(SchemaError((*path, name), "additionalProperties", "unexpected property"))
                elif isinstance(extra, Mapping):
                    errors += self.run(item, extra, (*path, name))
        return errors

    def _combinators(self, value: Any, schema: Mapping[str, Any], path: Path) -> list[SchemaError]:
        errors: list[SchemaError] = []
        for sub in schema.get("allOf", ()):
            errors += self.run(value, sub, path)
        if "anyOf" in schema and not any(not self.run(value, sub, path) for sub in schema["anyOf"]):
            errors.append(SchemaError(path, "anyOf", "matches no alternative"))
        if "oneOf" in schema:
            matches = sum(1 for sub in schema["oneOf"] if not self.run(value, sub, path))
            if matches != 1:
                errors.append(SchemaError(path, "oneOf", f"matches {matches} alternatives, expected exactly 1"))
        if "not" in schema and not self.run(value, schema["not"], path):
            errors.append(SchemaError(path, "not", "matches a forbidden schema"))
        if "if" in schema:
            branch = "then" if not self.run(value, schema["if"], path) else "else"
            if branch in schema:
                errors += self.run(value, schema[branch], path)
        return errors


def validate(
    value: Any, schema: Mapping[str, Any] | bool, *, root: Mapping[str, Any] | None = None
) -> list[SchemaError]:
    """Validate ``value`` against ``schema``; an empty list means valid.

    ``root`` is the document that local ``$ref`` pointers resolve against (default: ``schema``).
    """
    validator = _Validator(root if root is not None else schema)
    return validator.run(value, schema, ())


def is_valid(value: Any, schema: Mapping[str, Any] | bool, *, root: Mapping[str, Any] | None = None) -> bool:
    """``True`` when :func:`validate` reports no error."""
    return not validate(value, schema, root=root)


__all__ = [
    "FORMAT_CHECKS",
    "SchemaError",
    "inline_refs",
    "is_valid",
    "json_equal",
    "resolve_pointer",
    "validate",
]
