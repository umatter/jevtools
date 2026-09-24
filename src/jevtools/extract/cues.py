"""Cue words (spec §4.2.1, §4.2.8): negation, superlative, hedge, chit-chat and anaphor cues.

Cues are ``cue`` mentions with ``attrs["cue"]`` naming the family and ``attrs["canonical"]`` the normalized cue
(``most recent`` → ``latest``). They never claim text and are never claimed.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from jevtools.extract.base import Mention, SourceText
from jevtools.extract.locales import all_locales
from jevtools.extract.tokens import Token

_TEMPORAL_NEXT = frozenset(
    {
        "time",
        "week",
        "month",
        "year",
        "monday",
        "tuesday",
        "wednesday",
        "thursday",
        "friday",
        "saturday",
        "sunday",
        "night",
        "day",
    }
)
"""``last week`` / ``first Monday`` are temporal, not superlative."""


def _positions(tokens: Sequence[Token], phrase: Sequence[str]) -> Iterable[int]:
    n = len(phrase)
    for i in range(len(tokens) - n + 1):
        if all(tokens[i + j].folded == phrase[j] for j in range(n)):
            yield i


def _cue(source: SourceText, tokens: Sequence[Token], i: int, n: int, family: str, canonical: str) -> Mention:
    start, end = tokens[i].start, tokens[i + n - 1].end
    return Mention(
        "cue",
        source.text[start:end],
        (start, end),
        canonical,
        None,
        source.channel,
        source.ref,
        f"cues/{family}",
        attrs={"cue": family, "canonical": canonical},
    )


def extract(source: SourceText) -> list[Mention]:
    """All cue mentions of one text."""
    tokens = source.tokens
    out: list[Mention] = []
    for loc in all_locales():
        for i, token in enumerate(tokens):
            if token.folded in loc.negations:
                out.append(_cue(source, tokens, i, 1, "negation", token.folded))
            if token.folded in loc.hedges:
                out.append(_cue(source, tokens, i, 1, "hedge", token.folded))
        for phrase, canonical in loc.superlatives.items():
            for i in _positions(tokens, phrase):
                after = tokens[i + len(phrase)].folded if i + len(phrase) < len(tokens) else ""
                if after not in _TEMPORAL_NEXT:
                    out.append(_cue(source, tokens, i, len(phrase), "superlative", canonical))
        for family, phrases in (("chitchat", loc.chitchat), ("anaphor", loc.anaphors)):
            for phrase in phrases:
                for i in _positions(tokens, phrase):
                    out.append(_cue(source, tokens, i, len(phrase), family, " ".join(phrase)))
    unique: dict[tuple[str, tuple[int, int]], Mention] = {}
    for m in out:
        unique.setdefault((str(m.attrs["cue"]), m.span), m)
    return sorted(unique.values(), key=lambda m: m.span)


__all__ = ["extract"]
