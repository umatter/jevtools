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


# -- review regressions ------------------------------------------------------------------------------------------------


def test_decimal_str_big_numbers_are_exact() -> None:
    """A 29+ digit number (a wei amount, a long id) raised InvalidOperation under the 28-digit context."""
    assert decimal_str(Decimal("1" * 29)) == "1" * 29
    assert decimal_str(Decimal("1" * 60)) == "1" * 60
    assert decimal_str(Decimal("1E+3")) == "1000" and decimal_str(Decimal("-0.50")) == "-0.5"
    found = {m.value for m in extract("order 123456789012345678901234567890 shipped") if m.kind == "number"}
    assert found == {"123456789012345678901234567890"}


def test_big_numbers_never_crash_decide() -> None:
    import jevtools as jt
    from jevtools.context import Observation

    @jt.tool
    def get_weather(city: str) -> dict[str, str]:
        """Get the current weather for a city."""
        return {"city": city}

    router = jt.Router([get_weather], backend=jt.backends.LexicalSimulator())
    decision = router.decide([{"role": "user", "content": "Weather in Zurich, order 12345678901234567890123456789"}])
    assert decision.outcome == "execute"
    obs = Observation(step=1, tool="lookup", arguments={}, content={"station_id": 10**30, "wei": "1" * 31})
    ctx = jt.Context(messages="Weather in Zurich", observations=[obs])
    assert router.decide("Weather in Zurich", context=ctx).outcome in ("execute", "confirm", "clarify")


@pytest.mark.parametrize(
    ("text", "value", "number_text"),
    [
        ("Set the offset to -3", "-3", "-3"),
        ("Set the offset to −3", "-3", "−3"),
        ("It is -12.5 outside", "-12.5", "-12.5"),
        ("Set the freezer to -18", "-18", "-18"),
    ],
)
def test_negative_numbers_keep_their_sign(text: str, value: str, number_text: str) -> None:
    [number] = [m for m in extract(text) if m.kind == "number"]
    assert (number.value, number.text) == (value, number_text)


def test_hyphens_between_digits_are_not_signs() -> None:
    assert [m.value for m in extract("pick 5-10 items") if m.kind == "number"] == ["5", "10"]
    assert [m.value for m in extract("ticket A-3") if m.kind == "number"] == ["3"]
    assert [m.value for m in extract("on 2026-09-24") if m.kind == "number"] == ["2026", "9", "24"]


@pytest.mark.parametrize(
    ("text", "amount", "currency"),
    [
        ("Refund CHF -50 to Anna", "-50", "CHF"),
        ("Refund -50 CHF", "-50", "CHF"),
        ("Refund -CHF 50", "-50", "CHF"),
        ("Pay CHF 10 000 to Anna", "10000", "CHF"),
        ("Pay 10 000 CHF", "10000", "CHF"),
        ("Pay CHF1'250.50", "1250.5", "CHF"),
        ("Transfer $2k", "2000", "USD"),
        ("Transfer $2M", "2000000", "USD"),
        ("Pay Anna 1.5 million CHF", "1500000", "CHF"),
        ("Pay Anna CHF 3 Mio.", "3000000", "CHF"),
        ("Pay two thousand five hundred francs", "2500", "CHF"),
        ("Send one hundred twenty dollars", "120", "USD"),
        ("Move one thousand two hundred francs", "1200", "CHF"),
        ("Send one hundred and twenty dollars", "120", "USD"),
        ("Show the TOP 10.00 price", "10", "TOP"),
        ("Transfer CAD 500 to Anna", "500", "CAD"),  # a major currency, not a word code
        ("Pay CHF 250 five times", "250", "CHF"),
    ],
)
def test_money_amounts_are_whole_and_signed(text: str, amount: str, currency: str) -> None:
    found = [m for m in extract(text) if m.kind == "money"]
    assert [(m.value["amount"], m.value["currency"]) for m in found] == [(amount, currency)]


def test_zwei_millionen_franken() -> None:
    found = [m for m in extract("Zahle zwei Millionen Franken", "de") if m.kind == "money"]
    assert [m.value["amount"] for m in found] == ["2000000"]


def test_space_grouping_never_joins_phone_numbers() -> None:
    assert [m.value for m in extract("call 079 123 45 67") if m.kind == "number"] == ["79", "123", "45", "67"]
    assert not [m for m in extract("10 0000 CHF") if m.kind == "money"]  # a fragment is never an amount


@pytest.mark.parametrize(
    "text", ["Upgrade to PHP 8.2", "Show the TOP 10 customers", "Try ALL 5 variants", "Pay 10 TOP"]
)
def test_word_like_currency_codes_do_not_claim_numbers(text: str) -> None:
    mentions = extract(text)
    assert not [m for m in mentions if m.kind == "money"]
    assert all(m.free for m in mentions if m.kind == "number")


@pytest.mark.parametrize(
    ("text", "value", "unit"),
    [
        ("an hour and a half", "1.5", "hour"),
        ("1 hour 30 minutes", "90", "minute"),
        ("2 hours and 15 minutes", "135", "minute"),
        ("1h 30m", "90", "minute"),
        ("1h30", "90", "minute"),
        ("1:30 hours", "90", "minute"),
        ("an hour and 20 minutes", "80", "minute"),
    ],
)
def test_compound_durations_are_summed(text: str, value: str, unit: str) -> None:
    quantities = [m for m in extract(text) if m.kind == "quantity"]
    assert [(q.value, q.attrs["unit"]) for q in quantities] == [(value, unit)]
    assert all(not m.free for m in extract(text) if m.kind == "number")


def test_durations_of_different_scales_stay_separate() -> None:
    quantities = [(m.value, m.attrs["unit"]) for m in extract("3 days and 2 hours") if m.kind == "quantity"]
    assert quantities == [("3", "day"), ("2", "hour")]
