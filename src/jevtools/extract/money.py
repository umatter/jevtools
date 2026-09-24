"""Money amounts (spec §4.1 money, §4.2.4): ``CHF 250``, ``Fr. 250.–``, ``250 CHF``, ``€ 12.50``, ``$12``,
``250 francs``, ``CHF 1'250.50``, ``fünfzig Franken``.

A money mention wraps a number mention with a currency marker (ISO code, symbol, ``Fr.``, a currency word) or the
Swiss ``.–`` suffix. Its value is ``{"amount": "<decimal>", "currency": "<ISO code>" | None}``; quantization to the
currency's minor unit happens at pool time (§4.3).
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from jevtools.extract.base import Mention, SourceText
from jevtools.extract.catalogs import currencies
from jevtools.extract.locales import Locale, all_locales
from jevtools.extract.tokens import fold

SYMBOLS: dict[str, str] = {"€": "EUR", "$": "USD", "£": "GBP", "¥": "JPY", "₣": "CHF"}
_LEFT_RE = re.compile(r"(?:(?P<code>\b[A-Z]{3})|(?P<sym>[€$£¥₣])|(?P<fr>\bS?[Ff]r\.?))[  ]?$")
_RIGHT_RE = re.compile(
    r"^(?P<dash>\.[–—-]{1,2})?[  ]?(?:(?P<code>[A-Z]{3})(?![A-Za-z])|(?P<sym>[€$£¥₣])|(?P<word>[^\W\d_]+\.?))?"
)


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


def extract(source: SourceText, locale: Locale, numbers: Sequence[Mention]) -> list[Mention]:
    """Money mentions built around the ``number`` mentions of one text."""
    locales = (locale,) + tuple(loc for loc in all_locales() if loc.code != locale.code)
    text = source.text
    out: list[Mention] = []
    for number in numbers:
        if number.kind != "number":
            continue
        start, end = number.span
        currency: str | None = None
        left = _left(text, start)
        right = _right(text, end, locales)
        if left is None and right is None:
            continue
        if left is not None:
            currency, start = left
        if right is not None:
            right_code, right_end, _ = right
            if currency is None or right_code in (None, currency):
                currency = currency or right_code
                end = right_end
        value = {"amount": number.value, "currency": currency}
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
                attrs={"currency": currency, "amount": number.value},
            )
        )
    return out


__all__ = ["SYMBOLS", "extract"]
