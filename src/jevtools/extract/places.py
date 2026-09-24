"""Place extractor (spec §4.2.6): gazetteer matches (compact city gazetteer plus ISO 3166 country names).

A place mention keeps the user's text as its value ("Zurich"); ``attrs["places"]`` lists every gazetteer entry the
name can denote, so a slot with ``canon: "cities"`` can expand "Zurich" into Zürich (CH) and Zurich (Ontario, CA).
Matches must start with a capital letter unless the whole text is lowercase (so "nice weather" is not Nice).
"""

from __future__ import annotations

from jevtools.extract.base import Mention, SourceText
from jevtools.extract.catalogs import gazetteer
from jevtools.extract.tokens import fold

MAX_WORDS = 4
COMMON_WORDS = frozenset(
    {
        "nice",
        "reading",
        "bath",
        "mobile",
        "orange",
        "split",
        "chad",
        "jersey",
        "turkey",
        "chile",
        "guinea",
        "salvador",
        "victoria",
        "lima",
        "rio",
        "la",
        "sf",
    }
)
"""Place names that are also common words: matched only when written with a capital letter."""


def extract(source: SourceText) -> list[Mention]:
    """Place mentions in one text (longest match first, non-overlapping)."""
    index = gazetteer()
    tokens = [t for t in source.tokens if t.kind == "word" or t.text in (".", "-")]
    lowercase_text = source.text == source.text.lower()
    out: list[Mention] = []
    i = 0
    while i < len(tokens):
        if not (tokens[i].is_capitalized or lowercase_text):
            i += 1
            continue
        if not tokens[i].is_capitalized and tokens[i].folded in COMMON_WORDS:
            i += 1
            continue
        for n in range(min(MAX_WORDS, len(tokens) - i), 0, -1):
            start, end = tokens[i].start, tokens[i + n - 1].end
            text = source.text[start:end]
            places = index.get(fold(" ".join(text.split())))
            if places and not text.endswith((".", "-")):
                out.append(
                    Mention(
                        "place",
                        text,
                        (start, end),
                        text,
                        None,
                        source.channel,
                        source.ref,
                        "place",
                        attrs={"places": places},
                    )
                )
                i += n
                break
        else:
            i += 1
    return out


__all__ = ["MAX_WORDS", "extract"]
