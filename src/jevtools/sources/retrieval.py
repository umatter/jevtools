"""Pure-Python retrieval (spec §4.2.7, §4.4, §9): fuzzy anchor matching, trigram similarity and BM25.

- **Fuzzy anchors** (``fuzzy_matches``): a user mention matches a row when one of its ``match`` field tokens is an
  exact token match, a prefix of ≥ 3 characters, trigram similarity ≥ 0.5, or an alias.
- **Trigram similarity** (``trigram``): Jaccard over padded character trigrams of the folded strings.
- **BM25** (``BM25``): Okapi BM25 (k1 = 1.5, b = 0.75) over token lists.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from jevtools.extract.tokens import fold, split_identifier, words

MatchHow = Literal["exact", "alias", "prefix", "trigram", "group"]
ALIAS_FIELDS: frozenset[str] = frozenset({"alias", "aliases"})
"""Match fields whose values are aliases (``Contact whose alias is "Bob"``)."""
MIN_PREFIX = 3
MIN_TRIGRAM = 0.5
SCORES: dict[str, float] = {"exact": 1.0, "alias": 0.95, "group": 0.9}


@dataclass(frozen=True)
class Match:
    """One row matched by a mention."""

    index: int
    """Row index in the source."""
    score: float
    how: MatchHow
    field: str
    term: str
    """The row term that matched (a token of the field value, or the whole alias)."""
    mention: str


def trigrams(text: str) -> set[str]:
    """Padded character trigrams of the folded text (``"anna"`` → ``{"  a", " an", "ann", "nna", "na "}``)."""
    padded = f"  {fold(text)} "
    return {padded[i : i + 3] for i in range(len(padded) - 2)}


def trigram(a: str, b: str) -> float:
    """Trigram (Jaccard) similarity of two strings in [0, 1]."""
    ta, tb = trigrams(a), trigrams(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def field_terms(value: Any) -> list[str]:
    """String terms of a field value (a string or a list of strings)."""
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple)):
        return [v for v in value if isinstance(v, str)]
    return []


def match_term(mention: str, term: str, *, alias: bool, fuzzy: bool = True) -> tuple[float, MatchHow] | None:
    """How a (single- or multi-word) mention matches one field term, if at all.

    Alias fields match the whole alias; other fields match any of their word tokens. ``fuzzy`` enables the prefix
    (≥ 3 characters) and trigram (≥ 0.5) rules.
    """
    m = fold(mention).strip()
    if not m:
        return None
    candidates = [fold(term)] if alias else [fold(term), *words(term)]
    for candidate in candidates:
        if m == candidate:
            return (SCORES["alias"], "alias") if alias else (SCORES["exact"], "exact")
    if not fuzzy:
        return None
    best: tuple[float, MatchHow] | None = None
    for candidate in candidates:
        options: list[tuple[float, MatchHow]] = []
        if len(m) >= MIN_PREFIX and candidate.startswith(m):
            options.append((0.6 + 0.3 * len(m) / len(candidate), "prefix"))
        similarity = trigram(m, candidate)
        if similarity >= MIN_TRIGRAM:
            options.append((0.9 * similarity, "trigram"))
        for option in options:
            if best is None or option[0] > best[0]:
                best = option
    return best


def fuzzy_matches(
    mention: str,
    rows: Sequence[Mapping[str, Any]],
    fields: Sequence[str],
    *,
    alias_fields: Iterable[str] = ALIAS_FIELDS,
    group_field: str | None = None,
    fuzzy: bool = True,
) -> list[Match]:
    """Rows matched by ``mention`` on ``fields`` (best match per row), highest score first."""
    aliases = frozenset(alias_fields)
    out: list[Match] = []
    for index, row in enumerate(rows):
        best: Match | None = None
        for field in fields:
            for term in field_terms(row.get(field)):
                found = match_term(mention, term, alias=field in aliases, fuzzy=fuzzy)
                if found is None:
                    continue
                score, how = found
                if field == group_field and how == "exact":
                    score, how = SCORES["group"], "group"
                if best is None or score > best.score:
                    best = Match(index, score, how, field, term, mention)
        if best is not None:
            out.append(best)
    out.sort(key=lambda m: (-m.score, m.index))
    return out


def plural_stem(token: str) -> str:
    """Strip a final ``s`` from tokens longer than 3 characters (``invoices`` → ``invoice``)."""
    return token[:-1] if len(token) > 3 and token.endswith("s") and not token.endswith("ss") else token


def path_tokens(path: str) -> list[str]:
    """BM25 tokens of a path: split on ``/ _ - .`` and camelCase, folded, plural-stemmed."""
    return [plural_stem(fold(t)) for t in split_identifier(path)]


def synonym_table(synonyms: Mapping[str, Sequence[str]] | None) -> dict[str, str]:
    """Folded, stemmed synonym → its head term (``{"config": ["cfg", "yaml"]}`` → ``{"cfg": "config", …}``)."""
    table: dict[str, str] = {}
    for head, others in (synonyms or {}).items():
        stem = plural_stem(fold(head))
        table[stem] = stem
        for other in others:
            table.setdefault(plural_stem(fold(other)), stem)
    return table


def canonical_terms(tokens: Iterable[str], table: Mapping[str, str]) -> list[str]:
    """Stem each token and map synonyms onto their head term, so a synonym group counts as one term in BM25."""
    out: list[str] = []
    for token in tokens:
        stem = plural_stem(fold(token))
        out.append(table.get(stem, stem))
    return out


class BM25:
    """Okapi BM25 over pre-tokenized documents."""

    def __init__(self, docs: Sequence[Sequence[str]], *, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.docs = [Counter(doc) for doc in docs]
        self.lengths = [len(doc) for doc in docs]
        self.avg = (sum(self.lengths) / len(self.lengths)) if self.lengths else 0.0
        df: Counter[str] = Counter()
        for doc in self.docs:
            df.update(doc.keys())
        n = len(self.docs)
        self.idf = {term: math.log(1 + (n - f + 0.5) / (f + 0.5)) for term, f in df.items()}

    def __len__(self) -> int:
        return len(self.docs)

    def score(self, query: Sequence[str], index: int) -> float:
        """BM25 score of document ``index`` for the query tokens."""
        doc, length = self.docs[index], self.lengths[index]
        total = 0.0
        for term in dict.fromkeys(query):
            tf = doc.get(term, 0)
            if tf:
                norm = tf + self.k1 * (1 - self.b + self.b * length / (self.avg or 1))
                total += self.idf[term] * tf * (self.k1 + 1) / norm
        return total

    def scores(self, query: Sequence[str]) -> list[float]:
        return [self.score(query, i) for i in range(len(self.docs))]

    def top(self, query: Sequence[str], k: int | None = None, *, min_score: float = 0.0) -> list[tuple[int, float]]:
        """``(index, score)`` of the best documents with score > ``min_score``, best first (ties: lower index)."""
        ranked = sorted(
            ((i, s) for i, s in enumerate(self.scores(query)) if s > min_score), key=lambda item: (-item[1], item[0])
        )
        return ranked if k is None else ranked[:k]

    def matched(self, query: Sequence[str], index: int) -> list[str]:
        """Query terms present in document ``index``."""
        return [t for t in dict.fromkeys(query) if t in self.docs[index]]


__all__ = [
    "ALIAS_FIELDS",
    "BM25",
    "MIN_PREFIX",
    "MIN_TRIGRAM",
    "Match",
    "MatchHow",
    "canonical_terms",
    "field_terms",
    "fuzzy_matches",
    "match_term",
    "path_tokens",
    "plural_stem",
    "synonym_table",
    "trigram",
    "trigrams",
]
