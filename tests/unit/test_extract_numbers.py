"""Numbers, quantities (dimensions), money and patterns (spec §4.1, §4.2.1, §4.2.4)."""

from __future__ import annotations

from decimal import Decimal

import pytest

from jevtools.candidates import Channel
from jevtools.extract import money, numbers, patterns
from jevtools.extract.base import Mention, SourceText, claim
from jevtools.extract.locales import get_locale
from jevtools.extract.numbers import canonical_unit, convert, decimal_str, parse_number, unit_dimension
from jevtools.extract.tokens import tokenize


def source(text: str) -> SourceText:
    return SourceText("request", text, Channel.USER, tuple(tokenize(text)))


def extract(text: str, locale: str = "en") -> list[Mention]:
    src = source(text)
    nums = numbers.extract(src, get_locale(locale))
    return claim(nums + money.extract(src, get_locale(locale), nums))


@pytest.mark.parametrize(
    ("text", "decimal", "expected"),
    [
        ("1'250.50", ".", "1250.50"),
        ("1’250.50", ".", "1250.50"),
        ("1.250,50", ",", "1250.50"),
        ("1 250,50", ",", "1250.50"),
        ("1,250.50", ".", "1250.50"),
        ("1.250", ".", "1.250"),
        ("1.250", ",", "1250"),
        ("1,250", ".", "1250"),
        ("1,5", ",", "1.5"),
        ("12.5", ".", "12.5"),
        ("1.000.000", ",", "1000000"),
    ],
)
def test_parse_number_locale_separators(text: str, decimal: str, expected: str) -> None:
    assert parse_number(text, decimal) == Decimal(expected)


def test_decimal_str_and_units() -> None:
    assert decimal_str(Decimal("45.0")) == "45" and decimal_str(Decimal("1.50")) == "1.5"
    assert canonical_unit("minutes") == "minute" and canonical_unit("%") == "percent" and canonical_unit("x") is None
    assert unit_dimension("hour") == "time" and unit_dimension("gigabyte") == "data" and unit_dimension(None) is None
    assert convert(Decimal("1.5"), "hour", "minute") == Decimal(90)
    assert convert(Decimal(1), "hour", "byte") is None


@pytest.mark.parametrize(
    ("text", "locale", "value", "unit"),
    [
        ("45 min", "en", "45", "minute"),
        ("a 45-minute call", "en", "45", "minute"),
        ("2h", "en", "2", "hour"),
        ("half an hour", "en", "0.5", "hour"),
        ("an hour and a half", "en", "1.5", "hour"),
        ("one and a half hours", "en", "1.5", "hour"),
        ("forty-five minutes", "en", "45", "minute"),
        ("1.5 hours", "en", "1.5", "hour"),
        ("3 days", "en", "3", "day"),
        ("20%", "en", "20", "percent"),
        ("5 GB", "en", "5", "gigabyte"),
        ("eine halbe Stunde", "de", "0.5", "hour"),
        ("zwei Stunden", "de", "2", "hour"),
        ("une demi-heure", "fr", "0.5", "hour"),
    ],
)
def test_quantities_carry_dimension(text: str, locale: str, value: str, unit: str) -> None:
    quantities = [m for m in extract(text, locale) if m.kind == "quantity"]
    assert len(quantities) == 1
    q = quantities[0]
    assert q.value == value and q.attrs["unit"] == unit and q.dim == unit_dimension(unit) and q.free


def test_bare_numbers_and_words() -> None:
    found = {m.text: m.value for m in extract("Order 12 tickets and twenty five mugs") if m.kind == "number"}
    assert found == {"12": "12", "twenty five": "25"}
    assert [m.kind for m in extract("a 45 min sync") if m.free] == ["quantity"]  # "a" is not a number


@pytest.mark.parametrize(
    ("text", "locale", "amount", "currency"),
    [
        ("CHF 250", "en", "250", "CHF"),
        ("250 CHF", "en", "250", "CHF"),
        ("CHF 1'250.50", "en", "1250.5", "CHF"),
        ("Fr. 250.–", "en", "250", "CHF"),
        ("250 francs", "en", "250", "CHF"),
        ("€ 12.50", "en", "12.5", "EUR"),
        ("12,50 €", "de", "12.5", "EUR"),
        ("1.250,50 Euro", "de", "1250.5", "EUR"),
        ("1 250,50 €", "fr", "1250.5", "EUR"),
        ("$12", "en", "12", "USD"),
        ("1,250.50 USD", "en", "1250.5", "USD"),
        ("fünfzig Franken", "de", "50", "CHF"),
        ("£3", "en", "3", "GBP"),
    ],
)
def test_money_markers(text: str, locale: str, amount: str, currency: str) -> None:
    found = [m for m in extract(text, locale) if m.kind == "money"]
    assert [(m.value["amount"], m.value["currency"]) for m in found] == [(amount, currency)]
    assert all(m.claimed_by == "money" for m in extract(text, locale) if m.kind == "number")


def test_ten_minutes_is_time_not_money() -> None:
    mentions = extract("Email Anna that I'll be 10 minutes late")
    assert not [m for m in mentions if m.kind == "money"]
    number = next(m for m in mentions if m.kind == "number")
    assert number.claimed_by == "quantity"  # never a bare-number candidate for an amount


def test_unknown_currency_code_is_not_money() -> None:
    assert not [m for m in extract("ABC 250") if m.kind == "money"]


@pytest.mark.parametrize(
    ("text", "kind", "value"),
    [
        ("mail Anna.Keller@ACME.com now", "email", "Anna.Keller@acme.com"),
        ("see https://example.org/a?b=1.", "url", "https://example.org/a?b=1"),
        ("id 123e4567-e89b-12d3-a456-426614174000", "uuid", "123e4567-e89b-12d3-a456-426614174000"),
        ("ping 192.168.0.1 please", "ipv4", "192.168.0.1"),
    ],
)
def test_patterns(text: str, kind: str, value: str) -> None:
    found = [m for m in patterns.extract(source(text)) if m.kind == kind]
    assert [m.value for m in found] == [value]


def test_invalid_ipv4_and_regex_extractor() -> None:
    assert not [m for m in patterns.extract(source("999.1.1.1")) if m.kind == "ipv4"]
    found = patterns.extract(source("ticket PROJ-123 and PROJ-7"), [r"PROJ-\d+", r"(unclosed"])
    assert [m.value for m in found if m.kind == "regex"] == ["PROJ-123", "PROJ-7"]
