"""Money amounts (spec §4.1 money, §4.2.4): ``CHF 250``, ``Fr. 250.–``, ``250 CHF``, ``€ 12.50``, ``$12``,
``250 francs``, ``CHF 1'250.50``, ``fünfzig Franken``.

A money mention wraps a number mention with a currency marker (ISO code, symbol, ``Fr.``, a currency word) or the
Swiss ``.–`` suffix. Its value is ``{"amount": "<decimal>", "currency": "<ISO code>" | None}``; quantization to the
currency's minor unit happens at pool time (§4.3).

- A minus sign before the amount or the marker is kept (``CHF -50``, ``-CHF 50``, ``-50 CHF``).
- ISO codes that are also common words or tech terms (``PHP 8.2``, ``TOP 10``, ``ALL 5``, :data:`WORD_CODES`) make
  money only with a second money cue: a symbol or currency word on the other side, the ``.–`` suffix, or an amount
  written with the currency's minor-unit decimals (``TOP 10.00``).
- A number that is only part of a longer figure (another digit group or number word right next to it) never
  becomes money: a truncated amount would be a wrong candidate, no candidate is the safe failure.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from decimal import Decimal, InvalidOperation

from jevtools.extract.base import Mention, SourceText
from jevtools.extract.catalogs import currencies, minor_units
from jevtools.extract.locales import Locale, all_locales
from jevtools.extract.tokens import fold

SYMBOLS: dict[str, str] = {"€": "EUR", "$": "USD", "£": "GBP", "¥": "JPY", "₣": "CHF"}
_LEFT_RE = re.compile(r"(?:(?P<code>\b[A-Z]{3})|(?P<sym>[€$£¥₣])|(?P<fr>\bS?[Ff]r\.?))[  ]?$")
_RIGHT_RE = re.compile(
    r"^(?P<dash>\.[–—-]{1,2})?[  ]?(?:(?P<code>[A-Z]{3})(?![A-Za-z])|(?P<sym>[€$£¥₣])|(?P<word>[^\W\d_]+\.?))?"
)


WORD_CODES = frozenset({
    "ALL", "AMD", "BOB", "CUP", "CVE", "GEL", "MAD", "MOP", "PEN", "PHP", "SOS", "TOP", "TRY",
})  # fmt: skip
"""ISO 4217 codes that are also English words or tech terms (``PHP 8.2``, ``TOP 10``, ``ALL 5``): money only with a
second money cue. Major currencies that merely look like acronyms (``CAD``, ``SAR``, ``RON``) are not listed."""
_ADJACENT_FIGURE_RE = re.compile("^[ \u00a0\u202f'\u2019]?\\d")
_FIGURE_BEFORE_RE = re.compile("\\d[ \u00a0\u202f'\u2019]?$")


def _currency_word(word: str, locales: Sequence[Locale]) -> str | None:
    key = fold(word.rstrip("."))
    for loc in locales:
        if key in loc.currency_words:
            return loc.currency_words[key]
    return None


def _left(text: str, start: int) -> tuple[str | None, int] | None:
    match = _LEFT_RE.search(text[max(0, start - 6) : start])
    if match is None:
        return None
    offset = max(0, start - 6) + match.start()
    if match.group("code"):
        code = match.group("code")
        return (code, offset) if code in currencies() else None
    if match.group("sym"):
        return SYMBOLS[match.group("sym")], offset
    return "CHF", offset


def _right(text: str, end: int, locales: Sequence[Locale]) -> tuple[str | None, int, bool] | None:
    match = _RIGHT_RE.match(text[end : end + 24])
    if match is None:
        return None
    dash = bool(match.group("dash"))
    if match.group("code") and match.group("code") in currencies():
        return match.group("code"), end + match.end(), dash
    if match.group("sym"):
        return SYMBOLS[match.group("sym")], end + match.end(), dash
    if match.group("word"):
        code = _currency_word(match.group("word"), locales)
        if code is not None:
            return code, end + match.end(), dash
    if dash:
        return None, end + match.end("dash"), True
    return None


def _minor_digits(written: str, currency: str | None) -> bool:
    """Whether an amount is written with exactly the currency's minor-unit decimals (``10.00`` for TOP)."""
    places = minor_units(currency)
    decimals = re.search(r"[.,](\d+)$", written)
    return places > 0 and decimals is not None and len(decimals.group(1)) == places


def _part_of_longer_figure(source: SourceText, number: Mention, words: frozenset[str]) -> bool:
    """A digit group or scale word (``hundred``, ``Tausend``) right next to the number: it is a fragment of a figure
    the extractor could not read whole (``10 0000``, ``5 hundred``), so any amount built from it would be
    truncated."""
    text = source.text
    start, end = number.span
    if _ADJACENT_FIGURE_RE.match(text[end : end + 2]) or _FIGURE_BEFORE_RE.search(text[max(0, start - 2) : start]):
        return True
    before = [t for t in source.tokens if t.end <= start]
    after = [t for t in source.tokens if t.start >= end]
    return bool(before and before[-1].folded in words) or bool(after and after[0].folded in words)


_Left = tuple[str | None, int]
_Right = tuple[str | None, int, bool]


def _word_code(
    text: str, left: _Left | None, right: _Right | None, number: Mention
) -> tuple[_Left | None, _Right | None]:
    """Drop a bare :data:`WORD_CODES` marker that has no second money cue (``PHP 8.2`` is a version)."""
    end = number.span[1]
    left_word = left is not None and left[0] in WORD_CODES and text[left[1] : left[1] + 3] == left[0]
    right_word = right is not None and right[0] in WORD_CODES and text[end : right[1]].strip() == right[0]
    if left is not None and left_word and not _minor_digits(number.text, left[0]) and (right is None or right_word):
        left = None
    if right is not None and right_word and left is None and not right[2] and not _minor_digits(number.text, right[0]):
        right = None
    return left, right


def extract(source: SourceText, locale: Locale, numbers: Sequence[Mention]) -> list[Mention]:
    """Money mentions built around the ``number`` mentions of one text."""
    locales = (locale,) + tuple(loc for loc in all_locales() if loc.code != locale.code)
    words = frozenset().union(*(frozenset(loc.scale_words) for loc in locales))
    text = source.text
    out: list[Mention] = []
    for number in numbers:
        if number.kind != "number":
            continue
        start, end = number.span
        amount = str(number.value)
        currency: str | None = None
        left, right = _word_code(text, _left(text, start), _right(text, end, locales), number)
        if left is None and right is None:
            continue
        if _part_of_longer_figure(source, number, words):
            continue
        if left is not None:
            currency, start = left
        if right is not None:
            right_code, right_end, _ = right
            if currency is None or right_code in (None, currency):
                currency = currency or right_code
                end = right_end
        signed = left is not None and start > 0 and text[start - 1] in "-\u2212"
        if signed and (start < 2 or not text[start - 2].isalnum()):
            start -= 1  # "-CHF 50": the sign before the marker
            amount = _negate(amount)
        value = {"amount": amount, "currency": currency}
        out.append(
            Mention(
                "money",
                text[start:end],
                (start, end),
                value,
                "money",
                source.channel,
                source.ref,
                f"money/{locale.code}",
                attrs={"currency": currency, "amount": amount},
            )
        )
    return out


def _negate(amount: str) -> str:
    try:
        value = -Decimal(amount)
    except InvalidOperation:  # pragma: no cover - number mentions always carry a decimal string
        return amount
    return format(value, "f")


__all__ = ["SYMBOLS", "WORD_CODES", "extract"]
