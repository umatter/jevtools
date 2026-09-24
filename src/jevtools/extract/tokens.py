"""Tokenization shared by every extractor (spec §4.2.1).

Tokens keep their character offsets into the source text so mentions can report exact spans. ``fold`` gives the
accent-free lowercase form used for lexicon lookups (``Zürich`` → ``zurich``, ``nächsten`` → ``nachsten``).
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Literal

TokenKind = Literal["word", "number", "punct"]

_TOKEN_RE = re.compile(r"\d+(?:[.,:'’]\d+)*|[^\W\d_]+(?:['’\-][^\W\d_]+)*|[^\w\s]", re.UNICODE)
_SENTENCE_END_RE = re.compile(r"[.!?;]+(?=\s|$)|\n+")
_CAMEL_RE = re.compile(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")


@dataclass(frozen=True)
class Token:
    """One token with its offsets into the source text."""

    text: str
    start: int
    end: int
    kind: TokenKind

    @property
    def lower(self) -> str:
        """Casefolded text (apostrophes normalized to ``'``)."""
        return self.text.replace("’", "'").casefold()

    @property
    def folded(self) -> str:
        """Accent-free lowercase text (lexicon key)."""
        return fold(self.text)

    @property
    def is_word(self) -> bool:
        return self.kind == "word"

    @property
    def is_capitalized(self) -> bool:
        """First character is an uppercase letter."""
        return self.kind == "word" and self.text[:1].isupper()


def fold(text: str) -> str:
    """Casefold and strip accents (NFKD, combining marks removed); ``’`` becomes ``'``."""
    decomposed = unicodedata.normalize("NFKD", text.replace("’", "'"))
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch)).casefold()


def tokenize(text: str) -> list[Token]:
    """Split ``text`` into word, number and punctuation tokens (whitespace dropped)."""
    tokens: list[Token] = []
    for match in _TOKEN_RE.finditer(text):
        piece = match.group()
        if piece[0].isdigit():
            kind: TokenKind = "number"
        elif piece[0].isalpha():
            kind = "word"
        else:
            kind = "punct"
        tokens.append(Token(piece, match.start(), match.end(), kind))
    return tokens


def sentence_spans(text: str) -> list[tuple[int, int]]:
    """Character spans of sentences/clauses separated by ``. ! ? ;`` (before whitespace) or newlines."""
    spans: list[tuple[int, int]] = []
    start = 0
    for match in _SENTENCE_END_RE.finditer(text):
        _append_span(spans, text, start, match.start())
        start = match.end()
    _append_span(spans, text, start, len(text))
    return spans


def _append_span(spans: list[tuple[int, int]], text: str, start: int, end: int) -> None:
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    if end > start:
        spans.append((start, end))


def crosses_boundary(text: str, start: int, end: int) -> bool:
    """Whether ``text[start:end]`` spans a sentence/clause boundary (such spans are dropped, §4.2.6)."""
    inner = text[start:end].rstrip(".!?;")
    return bool(_SENTENCE_END_RE.search(inner))


def split_identifier(text: str) -> list[str]:
    """Lowercase tokens of an identifier or path: split on ``/ _ - .``, whitespace and camelCase."""
    parts: list[str] = []
    for chunk in re.split(r"[/_\-.\s\\]+", text):
        if chunk:
            parts.extend(p.lower() for p in _CAMEL_RE.split(chunk) if p)
    return parts


def words(text: str) -> list[str]:
    """Folded word/number tokens of ``text`` (no punctuation)."""
    return [t.folded for t in tokenize(text) if t.kind != "punct"]


def overlaps(a: tuple[int, int], b: tuple[int, int]) -> bool:
    """Half-open spans share at least one character."""
    return a[0] < b[1] and b[0] < a[1]


def covers(outer: tuple[int, int], inner: tuple[int, int]) -> bool:
    """``outer`` contains ``inner`` entirely."""
    return outer[0] <= inner[0] and inner[1] <= outer[1]


__all__ = [
    "Token",
    "TokenKind",
    "covers",
    "crosses_boundary",
    "fold",
    "overlaps",
    "sentence_spans",
    "split_identifier",
    "tokenize",
    "words",
]
