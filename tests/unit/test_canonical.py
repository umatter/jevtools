from __future__ import annotations

import unicodedata
from decimal import Decimal
from enum import Enum

import pytest
from pydantic import BaseModel

from jevtools.canonical import canonical_json, format_number, nfc, round4, sha256_of, short_id


def test_separators_and_key_order_are_kept() -> None:
    assert (
        canonical_json({"b": 1, "a": [1, 2], "c": {"z": None, "y": True}})
        == b'{"b":1,"a":[1,2],"c":{"z":null,"y":true}}'
    )


def test_strings_are_nfc_and_not_ascii_escaped() -> None:
    decomposed = unicodedata.normalize("NFD", "Zürich")
    assert len(decomposed) == 7
    assert nfc(decomposed) == "Zürich"
    assert canonical_json({decomposed: decomposed}) == '{"Zürich":"Zürich"}'.encode()
    assert canonical_json('⟨x⟩ → "q"\n') == '"⟨x⟩ → \\"q\\"\\n"'.encode()


@pytest.mark.parametrize(
    ("value", "text"),
    [
        (0.55, "0.55"),
        (0.7857, "0.7857"),
        (1.0, "1"),
        (1e-05, "0.00001"),
        (1.5e22, "15000000000000000000000"),
        (-0.0, "0"),
        (45, "45"),
        (Decimal("250.00"), "250"),
        (Decimal("250.50"), "250.5"),
        (0.1 + 0.2, "0.30000000000000004"),
    ],
)
def test_number_formatting(value: float, text: str) -> None:
    assert format_number(value) == text


def test_round4_half_even_and_rounded_serialization() -> None:
    assert round4(0.12345) == 0.1234
    assert round4(0.12355) == 0.1236
    assert round4(0.55000001) == 0.55
    assert canonical_json({"p": 0.785714}, round_floats=True) == b'{"p":0.7857}'
    with pytest.raises(ValueError):
        round4(float("nan"))
    with pytest.raises(ValueError):
        canonical_json(float("inf"))


def test_models_enums_and_rejections() -> None:
    class Color(Enum):
        RED = "red"

    class M(BaseModel):
        b: int = 2
        a: Color = Color.RED

    assert canonical_json(M()) == b'{"b":2,"a":"red"}'
    assert canonical_json({1: "x"}) == b'{"1":"x"}'
    with pytest.raises(TypeError):
        canonical_json({1, 2})
    with pytest.raises(TypeError):
        canonical_json(True + 0j)


def test_hash_stability_and_ids() -> None:
    doc = {"a": 1, "b": [0.5, "x"]}
    assert sha256_of(doc) == sha256_of({"a": 1.0, "b": [0.5, "x"]})
    assert sha256_of(doc).startswith("sha256:") and len(sha256_of(doc)) == 7 + 64
    assert sha256_of({"b": 1, "a": 2}) != sha256_of({"a": 2, "b": 1})
    assert short_id("tr_", "abc") == "tr_" + "ba7816bf8f01cfea"
