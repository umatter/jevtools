"""Numbers, number words, fractions and unit quantities (spec §4.1 quantity, §4.2.1 dimensions).

- Digits with locale separators: ``1'250.50``, ``1.250,50``, ``1 250,50``, ``1,250.50``. A single ambiguous
  separator followed by exactly three digits follows the locale (``1.250`` is 1.25 in ``en``, 1250 in ``de``).
- Number words (``forty-five``, ``zwölf``, ``trois``), ``a dozen``.
- Quantities: a number followed by a unit gets a dimension: ``time`` (``45 min``, ``2h``, ``half an hour``,
  ``an hour and a half``), ``percent`` (``20%``) or ``data`` (``5 GB``).

Every number is also emitted as a bare ``number`` mention; when a quantity covers it, claiming marks it
``claimed_by="quantity"`` so it never enters another numeric pool (§4.2.1).
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation

from jevtools.extract.base import Mention, SourceText
from jevtools.extract.locales import Locale, all_locales
from jevtools.extract.tokens import Token, fold

_DIGITS_RE = re.compile(r"(?<![\w.,])\d+(?:['’.,]\d+)*(?!\d)")
_SPACED_RE = re.compile(r"(?<![\w.,])\d{1,3}(?:[   ]\d{3})+(?:,\d+)?(?![\d.,])")
"""Space-grouped thousands (``1 250,50``): accepted with a decimal comma or in locales whose decimal is ``,``."""
_SPACE_CHARS = "   '’"

TIME_UNITS: dict[str, Decimal] = {
    "millisecond": Decimal("0.001"),
    "second": Decimal(1),
    "minute": Decimal(60),
    "hour": Decimal(3600),
    "day": Decimal(86400),
    "week": Decimal(604800),
    "month": Decimal(2592000),
    "year": Decimal(31536000),
}
"""Seconds per time unit (months = 30 days, years = 365 days; only used for unit conversion)."""
DATA_UNITS: dict[str, Decimal] = {
    "byte": Decimal(1),
    "kilobyte": Decimal(1000),
    "megabyte": Decimal(10**6),
    "gigabyte": Decimal(10**9),
    "terabyte": Decimal(10**12),
}
PERCENT_UNITS: dict[str, Decimal] = {"percent": Decimal(1)}
DIMENSIONS: dict[str, dict[str, Decimal]] = {"time": TIME_UNITS, "data": DATA_UNITS, "percent": PERCENT_UNITS}
UNIT_ALIASES: dict[str, str] = {
    "ms": "millisecond",
    "milliseconds": "millisecond",
    "b": "byte",
    "bytes": "byte",
    "kb": "kilobyte",
    "mb": "megabyte",
    "gb": "gigabyte",
    "tb": "terabyte",
    "%": "percent",
    "percent": "percent",
    "pct": "percent",
    "prozent": "percent",
    "pourcent": "percent",
}
_ATTACHED_ONLY = frozenset({"h", "ms", "b"})
"""Unit abbreviations accepted only when written right after the digits (``2h``), never as a separate word."""
_NEVER_UNITS = frozenset({"m", "s"})
"""Too ambiguous (metres, plural s): never read as time units."""


def unit_dimension(unit: str | None) -> str | None:
    """Dimension of a canonical unit (``minute`` → ``time``); ``None`` for unknown or missing units."""
    if unit is None:
        return None
    for dim, table in DIMENSIONS.items():
        if unit in table:
            return dim
    return None


def canonical_unit(word: str | None) -> str | None:
    """Canonical unit of a unit word or slot unit (``minutes`` → ``minute``, ``%`` → ``percent``)."""
    if word is None:
        return None
    key = fold(word)
    if unit_dimension(key) is not None:
        return key
    if key in UNIT_ALIASES:
        return UNIT_ALIASES[key]
    for loc in all_locales():
        if key in loc.duration_units:
            return loc.duration_units[key]
    return None


def convert(amount: Decimal, from_unit: str, to_unit: str) -> Decimal | None:
    """Convert between units of one dimension (``1.5 hour`` → ``90 minute``); ``None`` across dimensions."""
    dim = unit_dimension(from_unit)
    if dim is None or dim != unit_dimension(to_unit):
        return None
    table = DIMENSIONS[dim]
    return amount * table[from_unit] / table[to_unit]


def parse_number(text: str, decimal: str = ".") -> Decimal | None:
    """Parse a digit string with locale separators; ``None`` if it is not a number."""
    raw = "".join(ch for ch in text.strip() if ch not in _SPACE_CHARS)
    if not raw:
        return None
    dots, commas = raw.count("."), raw.count(",")
    if dots and commas:
        dec = "." if raw.rfind(".") > raw.rfind(",") else ","
        raw = raw.replace("," if dec == "." else ".", "").replace(dec, ".")
    elif dots + commas:
        sep = "." if dots else ","
        head, _, tail = raw.rpartition(sep)
        if dots + commas > 1:
            raw = raw.replace(sep, "")
        elif len(tail) == 3 and sep != decimal and head.isdigit() and len(head) <= 3:
            raw = raw.replace(sep, "")
        else:
            raw = raw.replace(sep, ".")
    try:
        value = Decimal(raw)
    except InvalidOperation:
        return None
    return value if value.is_finite() else None


def decimal_str(value: Decimal) -> str:
    """Canonical decimal text: no exponent, no trailing zeros (``Decimal("45.0")`` → ``"45"``)."""
    normalized = value.normalize()
    if normalized == normalized.to_integral_value():
        return str(normalized.quantize(Decimal(1)))
    return format(normalized, "f")


@dataclass(frozen=True)
class _Num:
    value: Decimal
    start: int
    end: int
    after: int
    """Index of the first token after the number (where a unit may follow)."""
    words: bool


def _word_value(token: Token, locales: Sequence[Locale]) -> tuple[str, int] | None:
    key = token.folded
    for loc in locales:
        if key in loc.number_words:
            return "unit", loc.number_words[key]
        if key in loc.scale_words:
            return "scale", loc.scale_words[key]
    return None


def _hyphen_words(token: Token, locales: Sequence[Locale]) -> int | None:
    """``forty-five`` → 45."""
    if "-" not in token.text:
        return None
    total = 0
    for part in token.text.split("-"):
        found = next((loc.number_words[fold(part)] for loc in locales if fold(part) in loc.number_words), None)
        if found is None:
            return None
        total += found
    return total


def _word_numbers(tokens: Sequence[Token], locales: Sequence[Locale]) -> list[_Num]:
    out: list[_Num] = []
    i = 0
    while i < len(tokens):
        hyphen = _hyphen_words(tokens[i], locales)
        if hyphen is not None:
            out.append(_Num(Decimal(hyphen), tokens[i].start, tokens[i].end, i + 1, True))
            i += 1
            continue
        found = _word_value(tokens[i], locales)
        if found is None or found[0] == "scale":
            i += 1
            continue
        total, current, j = 0, found[1], i + 1
        while j < len(tokens):
            nxt = _word_value(tokens[j], locales)
            if nxt is None:
                break
            kind, value = nxt
            if kind == "scale":
                current = max(current, 1) * value
                if value >= 1000:
                    total, current = total + current, 0
            elif current >= 20 and current % 10 == 0 and 0 < value < 10:
                current += value
            else:
                break
            j += 1
        out.append(_Num(Decimal(total + current), tokens[i].start, tokens[j - 1].end, j, True))
        i = j
    return out


def _digit_numbers(source: SourceText, locale: Locale) -> list[_Num]:
    spans: list[tuple[int, int]] = []
    for match in _SPACED_RE.finditer(source.text):
        if locale.decimal == "," or "," in match.group():
            spans.append(match.span())
    for match in _DIGITS_RE.finditer(source.text):
        if not any(start <= match.start() < end for start, end in spans):
            spans.append(match.span())
    out: list[_Num] = []
    for start, end in sorted(spans):
        value = parse_number(source.text[start:end], locale.decimal)
        if value is not None:
            after = next((i for i, t in enumerate(source.tokens) if t.start >= end), len(source.tokens))
            out.append(_Num(value, start, end, after, False))
    return out


def _unit_at(tokens: Sequence[Token], i: int, attached: bool, locales: Sequence[Locale]) -> tuple[str, int] | None:
    """Canonical unit starting at token ``i`` (``min``, ``hours``, ``%``, ``GB``) and the index after it."""
    if i >= len(tokens):
        return None
    word = tokens[i].folded
    if word == "-" and i + 1 < len(tokens) and tokens[i + 1].start == tokens[i].end:
        return _unit_at(tokens, i + 1, True, locales)
    if word in _NEVER_UNITS or (word in _ATTACHED_ONLY and not attached):
        return None
    if word in UNIT_ALIASES:
        return UNIT_ALIASES[word], i + 1
    for loc in locales:
        if word in loc.duration_units:
            return loc.duration_units[word], i + 1
    return None


def _and_a_half(tokens: Sequence[Token], end_index: int, amount: Decimal) -> tuple[Decimal, int]:
    """``2 hours and a half`` / ``2 heures et demie``: add 0.5 and extend the span."""
    tail = [t.folded for t in tokens[end_index : end_index + 3]]
    if tail == ["and", "a", "half"]:
        return amount + Decimal("0.5"), tokens[end_index + 2].end
    if tail[:2] == ["et", "demie"]:
        return amount + Decimal("0.5"), tokens[end_index + 1].end
    return amount, tokens[end_index - 1].end


def _quantity_mention(source: SourceText, start: int, end: int, amount: Decimal, unit: str, extractor: str) -> Mention:
    return Mention(
        "quantity",
        source.text[start:end],
        (start, end),
        decimal_str(amount),
        unit_dimension(unit),
        source.channel,
        source.ref,
        extractor,
        attrs={"unit": unit, "amount": decimal_str(amount)},
    )


def _quantity(source: SourceText, num: _Num, locales: Sequence[Locale], extractor: str) -> Mention | None:
    tokens = source.tokens
    attached = num.after < len(tokens) and tokens[num.after].start == num.end
    found = _unit_at(tokens, num.after, attached, locales)
    if found is None:
        return None
    unit, end_index = found
    amount, end = _and_a_half(tokens, end_index, num.value)
    return _quantity_mention(source, num.start, end, amount, unit, extractor)


_PHRASES: tuple[tuple[tuple[str, ...], Decimal], ...] = (
    (("half", "an"), Decimal("0.5")),
    (("half", "a"), Decimal("0.5")),
    (("a", "quarter", "of", "an"), Decimal("0.25")),
    (("a", "quarter", "of", "a"), Decimal("0.25")),
    (("a", "couple", "of"), Decimal(2)),
    (("one", "and", "a", "half"), Decimal("1.5")),
    (("an",), Decimal(1)),
    (("a",), Decimal(1)),
    (("eine", "halbe"), Decimal("0.5")),
    (("einer", "halben"), Decimal("0.5")),
    (("eine",), Decimal(1)),
    (("einer",), Decimal(1)),
    (("une",), Decimal(1)),
)
"""Word phrases before a unit (longest first within each family)."""


def _phrase_at(tokens: Sequence[Token], i: int, locales: Sequence[Locale]) -> tuple[Decimal, str, int] | None:
    for phrase, amount in _PHRASES:
        n = len(phrase)
        if tuple(t.folded for t in tokens[i : i + n]) != phrase:
            continue
        found = _unit_at(tokens, i + n, False, locales)
        if found is not None:
            return amount, found[0], found[1]
        if i + n < len(tokens) and tokens[i + n].folded.startswith("demi-"):  # une demi-heure
            word = tokens[i + n].folded.split("-", 1)[1]
            unit = next((loc.duration_units[word] for loc in locales if word in loc.duration_units), None)
            if unit is not None:
                return Decimal("0.5"), unit, i + n + 1
    return None


def _phrase_quantities(source: SourceText, locales: Sequence[Locale], extractor: str) -> list[Mention]:
    """``half an hour``, ``an hour``, ``a couple of days``, ``eine halbe Stunde``, ``une demi-heure``."""
    tokens = source.tokens
    out: list[Mention] = []
    i = 0
    while i < len(tokens):
        found = _phrase_at(tokens, i, locales)
        if found is None:
            i += 1
            continue
        amount, unit, end_index = found
        total, end = _and_a_half(tokens, end_index, amount)
        out.append(_quantity_mention(source, tokens[i].start, end, total, unit, extractor))
        i = end_index
    return out


def extract(source: SourceText, locale: Locale) -> list[Mention]:
    """Numbers (``number``, dimensionless) and unit quantities (``quantity`` with ``dim``) in one text."""
    locales = (locale,) + tuple(loc for loc in all_locales() if loc.code != locale.code)
    extractor = f"numbers/{locale.code}"
    word_locales = (locale,) if locale.code == "en" else (locale, locales[1])
    nums = _digit_numbers(source, locale) + _word_numbers(source.tokens, word_locales)
    out: list[Mention] = []
    for num in nums:
        text = source.text[num.start : num.end]
        out.append(
            Mention(
                "number",
                text,
                (num.start, num.end),
                decimal_str(num.value),
                None,
                source.channel,
                source.ref,
                extractor,
                attrs={"words": num.words},
            )
        )
        quantity = _quantity(source, num, locales, extractor)
        if quantity is not None:
            out.append(quantity)
    out += _phrase_quantities(source, locales, extractor)
    return out


__all__ = [
    "DATA_UNITS",
    "DIMENSIONS",
    "TIME_UNITS",
    "UNIT_ALIASES",
    "canonical_unit",
    "convert",
    "decimal_str",
    "extract",
    "parse_number",
    "unit_dimension",
]
