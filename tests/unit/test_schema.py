"""A JSON-Schema-Test-Suite-style table for the supported keywords (plus agreement with ``jsonschema`` if installed)."""

from __future__ import annotations

from decimal import Decimal
from typing import Any

import pytest

from jevtools.spec.schema import inline_refs, is_valid, json_equal, validate

CASES: list[tuple[str, dict[str, Any], Any, bool]] = [
    ("type string", {"type": "string"}, "x", True),
    ("type string rejects int", {"type": "string"}, 1, False),
    ("integer accepts 1.0", {"type": "integer"}, 1.0, True),
    ("integer rejects 1.5", {"type": "integer"}, 1.5, False),
    ("bool is not a number", {"type": "number"}, True, False),
    ("type list", {"type": ["string", "null"]}, None, True),
    ("enum json equality", {"enum": [1, "a"]}, 1.0, True),
    ("enum bool vs int", {"enum": [1]}, True, False),
    ("const", {"const": {"a": [1]}}, {"a": [1.0]}, True),
    ("pattern", {"pattern": r"^\d+(\.\d{1,2})?$"}, "250.00", True),
    ("pattern miss", {"pattern": r"^\d+$"}, "25a", False),
    ("minLength counts code points", {"minLength": 2}, "é", False),
    ("maxLength", {"maxLength": 3}, "abcd", False),
    ("minimum", {"minimum": 5}, 5, True),
    ("exclusiveMinimum", {"exclusiveMinimum": 5}, 5, False),
    ("maximum", {"maximum": 480}, 481, False),
    ("exclusiveMaximum", {"exclusiveMaximum": 1}, 0.5, True),
    ("multipleOf decimal", {"multipleOf": 0.01}, 0.07, True),
    ("multipleOf miss", {"multipleOf": 15}, 20, False),
    ("items", {"items": {"type": "integer"}}, [1, 2, "x"], False),
    ("prefixItems", {"prefixItems": [{"type": "string"}], "items": {"type": "integer"}}, ["a", 1], True),
    ("minItems", {"minItems": 1}, [], False),
    ("maxItems", {"maxItems": 1}, [1, 2], False),
    ("uniqueItems", {"uniqueItems": True}, [1, 1.0], False),
    ("required", {"required": ["a"]}, {"b": 1}, False),
    ("properties", {"properties": {"a": {"type": "string"}}}, {"a": 1}, False),
    ("additionalProperties false", {"properties": {"a": {}}, "additionalProperties": False}, {"a": 1, "b": 2}, False),
    ("additionalProperties schema", {"additionalProperties": {"type": "integer"}}, {"x": 1}, True),
    ("patternProperties", {"patternProperties": {"^x_": {"type": "integer"}}}, {"x_a": "no"}, False),
    ("minProperties", {"minProperties": 1}, {}, False),
    ("anyOf", {"anyOf": [{"type": "string"}, {"type": "integer"}]}, 3, True),
    ("anyOf miss", {"anyOf": [{"type": "string"}, {"type": "integer"}]}, 3.5, False),
    ("oneOf exactly one", {"oneOf": [{"type": "integer"}, {"type": "number"}]}, 3, False),
    ("oneOf one", {"oneOf": [{"type": "integer"}, {"type": "string"}]}, 3, True),
    ("allOf", {"allOf": [{"minimum": 1}, {"maximum": 2}]}, 3, False),
    ("not", {"not": {"type": "null"}}, None, False),
    ("if/then", {"if": {"properties": {"k": {"const": "card"}}}, "then": {"required": ["n"]}}, {"k": "card"}, False),
    ("if/else", {"if": {"properties": {"k": {"const": "card"}}}, "else": {"required": ["iban"]}}, {"k": "x"}, False),
    ("dependentRequired", {"dependentRequired": {"a": ["b"]}}, {"a": 1}, False),
    ("dependentRequired ok", {"dependentRequired": {"a": ["b"]}}, {"c": 1}, True),
    ("ref", {"$defs": {"P": {"type": "integer"}}, "properties": {"p": {"$ref": "#/$defs/P"}}}, {"p": "x"}, False),
    ("false schema", {"properties": {"a": False}}, {"a": 1}, False),
    ("unknown keywords ignored", {"x-jev": {"kind": "span"}, "foo": 1}, "x", True),
]


@pytest.mark.parametrize(("name", "schema", "value", "valid"), CASES, ids=[c[0] for c in CASES])
def test_keywords(name: str, schema: dict[str, Any], value: Any, valid: bool) -> None:
    assert is_valid(value, schema) is valid


@pytest.mark.parametrize(
    ("fmt", "good", "bad"),
    [
        ("email", "anna.keller@acme.com", "anna@"),
        ("uri", "https://example.com/x?y=1", "not a uri"),
        ("uuid", "123e4567-e89b-12d3-a456-426614174000", "123e4567"),
        ("ipv4", "10.0.0.1", "10.0.0.256"),
        ("ipv6", "::1", "10.0.0.1"),
        ("hostname", "api.typesafe.ai", "-bad-.com"),
        ("date", "2026-09-29", "2026-02-30"),
        ("date-time", "2026-09-29T15:00:00+02:00", "2026-09-29T15:00:00"),
        ("time", "15:00:00+02:00", "3pm"),
        ("duration", "PT45M", "45 minutes"),
        ("decimal", "250.00", "2.5e3"),
    ],
)
def test_formats(fmt: str, good: str, bad: str) -> None:
    assert is_valid(good, {"format": fmt}) and not is_valid(bad, {"format": fmt})


def test_error_paths_and_decimal_values() -> None:
    errors = validate({"to": 1, "items": [{"qty": "x"}]}, {
        "required": ["to", "body"],
        "properties": {"to": {"type": "string"}, "items": {"items": {"properties": {"qty": {"type": "integer"}}}}},
    })  # fmt: skip
    assert sorted((e.path, e.keyword) for e in errors) == [
        (("body",), "required"), (("items", 0, "qty"), "type"), (("to",), "type"),
    ]  # fmt: skip
    assert str(errors[0]).startswith("body: required") or ":" in str(errors[0])
    assert is_valid(Decimal("250.00"), {"type": "number", "multipleOf": 0.01})
    assert json_equal({"a": [1, 2]}, {"a": [1.0, 2]}) and not json_equal([1], [True])


def test_inline_refs() -> None:
    schema = {
        "$defs": {"U": {"type": "string", "enum": ["c"]}},
        "properties": {"u": {"$ref": "#/$defs/U", "default": "c"}},
    }
    assert inline_refs(schema) == {"properties": {"u": {"type": "string", "enum": ["c"], "default": "c"}}}
    with pytest.raises(KeyError):
        inline_refs({"properties": {"u": {"$ref": "#/$defs/Missing"}}})
    with pytest.raises(ValueError):
        inline_refs({"$defs": {"N": {"properties": {"n": {"$ref": "#/$defs/N"}}}}, "$ref": "#/$defs/N"})


DECIMAL_EXACT = {"multipleOf decimal"}
"""Cases where jevtools deliberately uses decimal arithmetic (float validators say 0.07 is no multiple of 0.01)."""


def test_agrees_with_jsonschema_when_installed() -> None:
    jsonschema = pytest.importorskip("jsonschema")
    for name, schema, value, _ in CASES:
        if name in DECIMAL_EXACT:
            continue
        reference = jsonschema.Draft202012Validator(schema).is_valid(value)
        assert is_valid(value, schema) is reference, name
