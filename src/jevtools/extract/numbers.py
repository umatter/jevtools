"""Numbers, number words, fractions and unit quantities (spec §4.1 quantity, §4.2.1 dimensions).

- Digits with locale separators: ``1'250.50``, ``1.250,50``, ``1 250,50``, ``10 000``, ``1,250.50``. A single
  ambiguous separator followed by exactly three digits follows the locale (``1.250`` is 1.25 in ``en``, 1250 in
  ``de``). A minus sign directly before the digits is kept (``-3``, ``−18``; ``5-10`` stays unsigned). Digits glued
  to an ISO currency code are read whole (``CHF1'250.50``).
- Magnitudes after a digit number multiply it (``2k``, ``$2M``, ``1.5 million``, ``3 Mio.``).
- Number words (``forty-five``, ``zwölf``, ``trois``, ``two thousand five hundred``, ``one hundred twenty``).
- Quantities: a number followed by a unit gets a dimension: ``time`` (``45 min``, ``2h``, ``half an hour``,
  ``an hour and a half``), ``percent`` (``20%``) or ``data`` (``5 GB``). Compound durations are summed into one
  quantity in their smallest unit (``1 hour 30 minutes``, ``2 hours and 15 minutes``, ``1h30``, ``1h 30m``,
  ``1:30 hours`` → 90 minutes); :func:`duration_at` is shared with the temporal parser's relative offsets.

Every number is also emitted as a bare ``number`` mention; when a quantity covers it, claiming marks it
``claimed_by="quantity"`` so it never enters another numeric pool (§4.2.1).
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation, localcontext

from jevtools.extract.base import Mention, SourceText
from jevtools.extract.locales import Locale, all_locales
from jevtools.extract.tokens import Token, fold

_SIGN = r"(?:(?<![\w.,'’\-−+])[-−](?=\d))?"
"""An optional minus sign directly before the digits (``-3``, ``−3``); never after a word character, a digit or
another sign, so ``5-10``, ``A-3`` and ``2026-09-24`` stay unsigned."""
_DIGITS_RE = re.compile(_SIGN + r"(?<![\w.,])(?<!\d['’])\d+(?:['’.,]\d+)*(?!\d)")
_GROUP_SPACES = " \u00a0\u202f"
_SPACED_RE = re.compile(
    _SIGN
    + rf"(?<![\w.,])(?<!\d['’])(?<!\d[{_GROUP_SPACES}])\d{{1,3}}(?:[{_GROUP_SPACES}]\d{{3}})+(?:[.,]\d+)?"
    + rf"(?!\d|[.,]\d)(?![{_GROUP_SPACES}]\d)"
)
"""Space-grouped thousands (``10 000``, ``1 250,50``) in every locale; a run with another digit group right before
or after it (``079 123 45 67``) is not one number."""
_GLUED_CODE_RE = re.compile(r"\b([A-Z]{3})(\d+(?:['’.,]\d+)*)(?!\d)")
"""Digits glued to a 3-letter code (``CHF1'250.50``): one number when the code is an ISO 4217 currency."""
_MAGNITUDE_ATTACHED: dict[str, int] = {"k": 10**3, "K": 10**3, "M": 10**6, "bn": 10**9, "Mio": 10**6, "Mrd": 10**9}
"""Magnitude suffixes written right after the digits (``2k``, ``$2M``); case-sensitive (``5m`` is minutes)."""
_MAGNITUDE_WORDS: dict[str, int] = {
    "thousand": 10**3, "million": 10**6, "millions": 10**6, "billion": 10**9, "billions": 10**9, "bn": 10**9,
    "tausend": 10**3, "mio": 10**6, "millionen": 10**6, "milliarde": 10**9, "milliarden": 10**9, "mrd": 10**9,
    "mille": 10**3, "milliard": 10**9, "milliards": 10**9,
}  # fmt: skip
"""Magnitude words after a digit number (``1.5 million``, ``3 Mio.``, ``2 thousand``)."""
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
    """Parse a digit string with locale separators and an optional leading minus sign (``-``/``−``); ``None`` if it
    is not a number."""
    raw = "".join(ch for ch in text.strip() if ch not in _SPACE_CHARS)
    if raw[:1] in ("-", "−"):
        value = parse_number(raw[1:], decimal) if raw[1:2].isdigit() else None
        return -value if value is not None else None
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
    # A 29+ digit number (a wei amount, a long id in a tool output) must neither raise ``InvalidOperation`` nor be
    # rounded by the default 28-digit context: normalize with enough precision, format without an exponent.
    digits = len(value.as_tuple().digits)
    with localcontext() as ctx:
        ctx.prec = max(ctx.prec, digits + 2)
        return format(value.normalize(), "f")


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
        total, current, j, last = 0, found[1], i + 1, "unit"
        while j < len(tokens):
            step, nxt = 1, _word_value(tokens[j], locales)
            if nxt is None and last == "scale":
                nxt = _hyphen_unit(tokens[j], locales)  # "two hundred forty-five"
            if nxt is None and last == "scale" and tokens[j].folded in _WORD_JOINERS and j + 1 < len(tokens):
                step, nxt = 2, _word_value(tokens[j + 1], locales) or _hyphen_unit(tokens[j + 1], locales)
                if nxt is not None and nxt[0] != "unit":  # "one hundred and twenty", never "... and thousand"
                    nxt = None
            if nxt is None:
                break
            kind, value = nxt
            if kind == "scale":
                if value < 1000 and current >= value:  # "five hundred hundred"
                    break
                current = max(current, 1) * value
                if value >= 1000:
                    total, current = total + current, 0
            elif last == "scale" and current % 100 == 0 and value < 100:  # "two thousand five", "hundred twenty"
                current += value
            elif current >= 20 and current % 10 == 0 and 0 < value < 10:
                current += value
            else:
                break
            last = kind
            j += step
        out.append(_Num(Decimal(total + current), tokens[i].start, tokens[j - 1].end, j, True))
        i = j
    return out


_WORD_JOINERS = frozenset({"and", "und", "et"})


def _hyphen_unit(token: Token, locales: Sequence[Locale]) -> tuple[str, int] | None:
    """``forty-five`` after a scale word (``two hundred forty-five``)."""
    value = _hyphen_words(token, locales)
    return ("unit", value) if value is not None and value < 100 else None


def _digit_numbers(source: SourceText, locale: Locale) -> list[_Num]:
    from jevtools.extract.catalogs import currencies

    text, tokens = source.text, source.tokens
    spans: list[tuple[int, int]] = [m.span() for m in _SPACED_RE.finditer(text)]
    spans += [m.span(2) for m in _GLUED_CODE_RE.finditer(text) if m.group(1) in currencies()]
    for match in _DIGITS_RE.finditer(text):
        if not any(start < match.end() and match.start() < end for start, end in spans):
            spans.append(match.span())
    out: list[_Num] = []
    for start, end in sorted(spans):
        value = parse_number(text[start:end], locale.decimal)
        if value is None:
            continue
        after = next((i for i, t in enumerate(tokens) if t.start >= end), len(tokens))
        if any(t.start < end < t.end for t in tokens):
            after = len(tokens)  # the hour of "1:30": no unit attaches to a digit run inside a larger token
        value, end, after = _magnitude(tokens, value, end, after)
        out.append(_Num(value, start, end, after, False))
    return out


def _magnitude(tokens: Sequence[Token], value: Decimal, end: int, after: int) -> tuple[Decimal, int, int]:
    """``2k`` / ``1.5 million`` / ``3 Mio.``: the multiplied value, the new end and the token after it."""
    if after >= len(tokens):
        return value, end, after
    token = tokens[after]
    factor = _MAGNITUDE_ATTACHED.get(token.text) if token.start == end else None
    if factor is None and token.kind == "word" and token.start > end:
        factor = _MAGNITUDE_WORDS.get(token.folded)
    if factor is None:
        return value, end, after
    end, after = token.end, after + 1
    abbreviation = token.folded in ("mio", "mrd")
    if abbreviation and after < len(tokens) and tokens[after].text == "." and tokens[after].start == end:
        end, after = tokens[after].end, after + 1
    return value * factor, end, after


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
    """``2 hours and a half`` / ``2 heures et demie``: add 0.5; returns the token index after the phrase."""
    tail = [t.folded for t in tokens[end_index : end_index + 3]]
    if tail == ["and", "a", "half"]:
        return amount + Decimal("0.5"), end_index + 3
    if tail[:2] == ["et", "demie"]:
        return amount + Decimal("0.5"), end_index + 2
    return amount, end_index


COMPOUND_UNITS = ("hour", "minute", "second")
"""Time units a compound duration sums (``1 hour 30 minutes``); days and longer stay separate."""
_DURATION_JOINERS = frozenset({"and", ",", "et", "und"})


def _pair_at(
    tokens: Sequence[Token], i: int, locales: Sequence[Locale], decimal: str
) -> tuple[Decimal, str, int] | None:
    """One ``amount unit`` starting at token ``i`` (``3 days``, ``an hour``, ``zwei Stunden``) and the index after."""
    if i >= len(tokens):
        return None
    phrase = _phrase_at(tokens, i, locales)
    if phrase is not None:
        return phrase
    token = tokens[i]
    amount: Decimal | None = None
    if token.kind == "number":
        amount = parse_number(token.text, decimal)
    else:
        found = next((loc.number_words[token.folded] for loc in locales if token.folded in loc.number_words), None)
        if found is None:
            found = _hyphen_words(token, locales)
        amount = Decimal(found) if found is not None else None
    if amount is None:
        return None
    attached = i + 1 < len(tokens) and tokens[i + 1].start == token.end
    unit = _unit_at(tokens, i + 1, attached, locales)
    return (amount, unit[0], unit[1]) if unit is not None else None


def _minutes_after_h(tokens: Sequence[Token], j: int) -> tuple[Decimal, int] | None:
    """The minutes of ``1h30`` / ``1h 30m`` / ``1h30min`` right after an ``h`` component (token ``j``)."""
    if j >= len(tokens) or tokens[j].kind != "number" or not tokens[j].text.isdigit():
        return None
    minutes = int(tokens[j].text)
    if minutes >= 60:
        return None
    nxt = tokens[j + 1] if j + 1 < len(tokens) else None
    if nxt is not None and nxt.start == tokens[j].end and nxt.folded in ("m", "min", "mins"):
        return Decimal(minutes), j + 2
    if tokens[j].start == tokens[j - 1].end and (nxt is None or nxt.start > tokens[j].end):
        return Decimal(minutes), j + 1  # "1h30"
    return None


def extend_duration(
    tokens: Sequence[Token], j: int, amount: Decimal, unit: str, locales: Sequence[Locale], decimal: str = "."
) -> tuple[Decimal, str, int]:
    """Extend ``amount unit`` (ending before token ``j``) with ``and a half`` and with following ``amount unit``
    pairs of strictly smaller time units (optionally joined by ``and``/``,``/``et``/``und``), summed in the
    smallest unit: ``1 hour 30 minutes`` → ``(90, minute, j')``."""
    amount, j = _and_a_half(tokens, j, amount)
    if unit not in COMPOUND_UNITS:
        return amount, unit, j
    total, smallest = amount * TIME_UNITS[unit], unit
    after_h = unit == "hour" and tokens[j - 1].folded == "h"
    while j < len(tokens):
        if after_h and smallest == "hour":
            minutes = _minutes_after_h(tokens, j)
            if minutes is not None:
                total, smallest, j = total + minutes[0] * TIME_UNITS["minute"], "minute", minutes[1]
                continue
        k = j + 1 if tokens[j].folded in _DURATION_JOINERS else j
        pair = _pair_at(tokens, k, locales, decimal)
        if pair is None or pair[1] not in COMPOUND_UNITS or TIME_UNITS[pair[1]] >= TIME_UNITS[smallest]:
            break
        total, smallest, j = total + pair[0] * TIME_UNITS[pair[1]], pair[1], pair[2]
    if smallest == unit:
        return amount, unit, j
    return Decimal(decimal_str(total / TIME_UNITS[smallest])), smallest, j


def duration_at(
    tokens: Sequence[Token], i: int, locales: Sequence[Locale], decimal: str = "."
) -> tuple[Decimal, str, int] | None:
    """A whole duration starting at token ``i`` (``an hour and a half``, ``1 hour 30 minutes``): amount, unit and
    the token index after it. Shared by quantities and the temporal parser's offsets (``in 1 hour 30 minutes``)."""
    pair = _pair_at(tokens, i, locales, decimal)
    if pair is None:
        return None
    return extend_duration(tokens, pair[2], pair[0], pair[1], locales, decimal)


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


def _quantity(
    source: SourceText, num: _Num, locales: Sequence[Locale], extractor: str, decimal: str = "."
) -> Mention | None:
    tokens = source.tokens
    attached = num.after < len(tokens) and tokens[num.after].start == num.end
    found = _unit_at(tokens, num.after, attached, locales)
    if found is None:
        return None
    unit, end_index = found
    amount, unit, j = extend_duration(tokens, end_index, num.value, unit, locales, decimal)
    return _quantity_mention(source, num.start, tokens[j - 1].end, amount, unit, extractor)


_COLON_DURATION_RE = re.compile(r"^(\d{1,3}):([0-5]\d)$")


def _colon_durations(source: SourceText, locales: Sequence[Locale], extractor: str) -> list[Mention]:
    """``1:30 hours`` / ``1:30h``: hours and minutes of a clock-style duration (90 minutes)."""
    tokens = source.tokens
    out: list[Mention] = []
    for i, token in enumerate(tokens):
        match = _COLON_DURATION_RE.match(token.text) if token.kind == "number" else None
        if match is None:
            continue
        attached = i + 1 < len(tokens) and tokens[i + 1].start == token.end
        found = _unit_at(tokens, i + 1, attached, locales)
        if found is None or found[0] != "hour":
            continue
        minutes = Decimal(int(match.group(1)) * 60 + int(match.group(2)))
        out.append(_quantity_mention(source, token.start, tokens[found[1] - 1].end, minutes, "minute", extractor))
    return out


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


def _phrase_quantities(
    source: SourceText, locales: Sequence[Locale], extractor: str, decimal: str = "."
) -> list[Mention]:
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
        total, unit, j = extend_duration(tokens, end_index, amount, unit, locales, decimal)
        out.append(_quantity_mention(source, tokens[i].start, tokens[j - 1].end, total, unit, extractor))
        i = end_index
    return out


def _outermost(quantities: Sequence[Mention]) -> list[Mention]:
    """Drop quantities inside a longer one (``30 minutes`` of ``1 hour 30 minutes``)."""
    return [
        q
        for q in quantities
        if not any(o.span != q.span and o.span[0] <= q.span[0] and q.span[1] <= o.span[1] for o in quantities)
    ]


def extract(source: SourceText, locale: Locale) -> list[Mention]:
    """Numbers (``number``, dimensionless) and unit quantities (``quantity`` with ``dim``) in one text."""
    locales = (locale,) + tuple(loc for loc in all_locales() if loc.code != locale.code)
    extractor = f"numbers/{locale.code}"
    word_locales = (locale,) if locale.code == "en" else (locale, locales[1])
    nums = _digit_numbers(source, locale) + _word_numbers(source.tokens, word_locales)
    out: list[Mention] = []
    quantities: list[Mention] = []
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
        quantity = _quantity(source, num, locales, extractor, locale.decimal)
        if quantity is not None:
            quantities.append(quantity)
    quantities += _phrase_quantities(source, locales, extractor, locale.decimal)
    quantities += _colon_durations(source, locales, extractor)
    return out + _outermost(quantities)


__all__ = [
    "DATA_UNITS",
    "DIMENSIONS",
    "TIME_UNITS",
    "UNIT_ALIASES",
    "COMPOUND_UNITS",
    "canonical_unit",
    "convert",
    "decimal_str",
    "duration_at",
    "extend_duration",
    "extract",
    "parse_number",
    "unit_dimension",
]
