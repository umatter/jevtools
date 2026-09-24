"""``Registry``: an app-owned list of rows (contacts, accounts…) as a candidate source (spec §3.2, §4.2.7, §4.4).

- **Anchors.** User words that match ≥ 1 row on the ``match`` fields: exact token, prefix ≥ 3 characters, trigram
  similarity ≥ 0.5 (prefix/trigram only for capitalized words, or when the whole message is lowercase), or an
  alias field. Adjacent matching words form one anchor ("Anna Keller").
- **Pool.** The union of the anchors' rows, ranked by match score then by the ``recency`` attribute, cut to K.
  A registry with at most ``send_whole_if_under`` (12) rows is sent whole (anchored rows keep their match note).
- **Labels and descriptions.** ``label`` renders the WYSIWYG label (``"{name} <{email}>"``); the description is a
  match note (``Contact matching "Anna"``) plus the ``describe`` template. The value is the ``key`` field.
- ``attrs`` (and the key, the label, the match fields) travel with candidates for constraints, ``render`` and
  ``order_by``; they are never sent to Jev.
"""

from __future__ import annotations

from collections.abc import Callable, Collection, Iterable, Mapping, Sequence
from typing import Any

from jevtools.candidates import Candidate, Channel
from jevtools.canonical import round4, sha256_of
from jevtools.extract.base import Mention
from jevtools.extract.locales import all_locales
from jevtools.extract.tokens import Token, fold, tokenize, words
from jevtools.sources.base import SourceQuery, item_noun
from jevtools.sources.retrieval import BM25, Match, fuzzy_matches
from jevtools.templates import NEGATION_NOTE

Template = str | Callable[[Mapping[str, Any]], str]
Retriever = str | Callable[[SourceQuery, Sequence[Mapping[str, Any]]], Iterable[tuple[int, float]]]

HOW_PHRASE: dict[str, str] = {
    "exact": "matching",
    "bm25": "matching",
    "group": "in the group",
    "alias": "whose alias is",
    "prefix": "similar to",
    "trigram": "similar to",
}
_HOW_RANK = {"exact": 0, "alias": 1, "group": 2, "bm25": 3, "prefix": 4, "trigram": 5}
_STOP = frozenset().union(*(loc.stopwords for loc in all_locales()))
_VERBS = frozenset().union(*(loc.command_verbs for loc in all_locales()))


class _Row(dict[str, Any]):
    def __missing__(self, key: str) -> str:
        return ""


def render_template(template: Template | None, row: Mapping[str, Any], default: str = "") -> str:
    """Render a ``{field}`` template (missing fields render empty) or call a template function."""
    if template is None:
        return default
    if callable(template):
        return str(template(row))
    rendered = template.format_map(_Row({k: "" if v is None else v for k, v in row.items()}))
    return " ".join(rendered.split())


class Registry:
    """An app-owned registry of rows (spec §4.4 ``jt.Registry``)."""

    def __init__(
        self,
        name: str,
        rows: Iterable[Mapping[str, Any]],
        key: str,
        label: Template | None = None,
        describe: Template | None = None,
        match: Sequence[str] = (),
        provides: Collection[str] = (),
        attrs: Sequence[str] = (),
        retriever: Retriever = "fuzzy",
        send_whole_if_under: int = 12,
        synonyms: Mapping[str, Sequence[str]] | None = None,
        *,
        channel: Channel | str = Channel.REGISTRY,
        item: str | None = None,
        recency: str | None = None,
        hierarchy: str | None = None,
        groups: str | None = None,
    ) -> None:
        self.name = name
        self.rows: tuple[Mapping[str, Any], ...] = tuple(dict(r) for r in rows)
        self.key = key
        self.label = label
        self.describe = describe
        self.match: tuple[str, ...] = tuple(match) or (key,)
        self.provides: frozenset[str] = frozenset(provides)
        self.attrs: tuple[str, ...] = tuple(attrs)
        self.retriever = retriever
        self.send_whole_if_under = send_whole_if_under
        self.synonyms = {fold(k): [fold(v) for v in vs] for k, vs in (synonyms or {}).items()}
        self.channel = Channel(channel)
        self.item = item or item_noun(name)
        self.recency = recency
        self.hierarchy = hierarchy
        self.groups = groups
        self._by_key = {str(r.get(key)): i for i, r in enumerate(self.rows)}
        missing = [i for i, r in enumerate(self.rows) if r.get(key) in (None, "")]
        if missing:
            raise ValueError(f"registry {name!r}: rows {missing[:5]} have no {key!r} value")

    def __len__(self) -> int:
        return len(self.rows)

    def __repr__(self) -> str:
        return f"Registry({self.name!r}, {len(self.rows)} rows, key={self.key!r})"

    # -- rows -----------------------------------------------------------------------------------------------------

    @property
    def whole(self) -> bool:
        """Small registries (``≤ send_whole_if_under`` rows) are sent whole."""
        return len(self.rows) <= self.send_whole_if_under

    def lookup(self, key: Any) -> Mapping[str, Any] | None:
        """The row whose key is ``key`` (TOCTOU re-resolution, §3.8.5)."""
        index = self._by_key.get(str(key))
        return self.rows[index] if index is not None else None

    def label_of(self, row: Mapping[str, Any]) -> str:
        return render_template(self.label, row, default=str(row[self.key]))

    def describe_of(self, row: Mapping[str, Any]) -> str:
        return render_template(self.describe, row).strip().rstrip(".")

    def attrs_of(self, row: Mapping[str, Any]) -> dict[str, Any]:
        """Row attributes that travel with a candidate (never sent to Jev)."""
        fields = dict.fromkeys((self.key, *self.match, *self.attrs, *(f for f in (self.hierarchy,) if f)))
        out = {f: row.get(f) for f in fields if f in row}
        out.setdefault("label", self.label_of(row))
        return out

    def attribute_names(self) -> frozenset[str]:
        """Attributes candidates carry (usable by ``order_by``, ``render`` and constraints)."""
        return frozenset((self.key, "label", *self.match, *self.attrs, *(f for f in (self.hierarchy,) if f)))

    def content_sha256(self) -> str:
        """Hash of the registry content (the Context document references sources by name and hash)."""
        return sha256_of({"name": self.name, "key": self.key, "rows": list(self.rows)})

    # -- anchors --------------------------------------------------------------------------------------------------

    def _matches(self, text: str, *, fuzzy: bool) -> list[Match]:
        found = fuzzy_matches(text, self.rows, self.match, group_field=self.groups, fuzzy=fuzzy)
        for synonym in self.synonyms.get(fold(text), ()):
            found += [
                Match(m.index, m.score * 0.95, m.how, m.field, m.term, text)
                for m in fuzzy_matches(synonym, self.rows, self.match, group_field=self.groups, fuzzy=False)
            ]
        best: dict[int, Match] = {}
        for m in found:
            if m.index not in best or m.score > best[m.index].score:
                best[m.index] = m
        return sorted(best.values(), key=lambda m: (-m.score, m.index))

    def _candidate_token(self, tokens: Sequence[Token], i: int) -> bool:
        token = tokens[i]
        if not token.is_word or len(token.text) < 2:
            return False
        initial = i == 0 or tokens[i - 1].text in ".!?:\n"
        if token.folded in _VERBS and initial:
            return False
        return token.folded not in _STOP or (token.is_capitalized and not initial)

    def find_anchors(
        self,
        text: str,
        *,
        tokens: Sequence[Token] | None = None,
        source_ref: str = "request",
        channel: Channel = Channel.USER,
    ) -> list[Mention]:
        """Anchor mentions of this registry in one user text (called by ``run_extractors``)."""
        if self.retriever != "fuzzy" and self.retriever != "exact":
            return []
        tokens = list(tokens) if tokens is not None else tokenize(text)
        lowercase = text == text.lower()
        per_token: list[tuple[int, list[Match]]] = []
        for i in range(len(tokens)):
            if not self._candidate_token(tokens, i):
                continue
            fuzzy = self.retriever == "fuzzy" and (tokens[i].is_capitalized or lowercase)
            matches = self._matches(tokens[i].text, fuzzy=fuzzy)
            if matches:
                per_token.append((i, matches))
        return [self._anchor(text, tokens, run, source_ref, channel) for run in self._runs(tokens, per_token)]

    def _runs(
        self, tokens: Sequence[Token], per_token: list[tuple[int, list[Match]]]
    ) -> list[list[tuple[int, list[Match]]]]:
        runs: list[list[tuple[int, list[Match]]]] = []
        for i, matches in per_token:
            if runs:
                j, previous = runs[-1][-1]
                adjacent = i == j + 1 and tokens[i].start - tokens[j].end <= 1
                shared = {m.index for m in matches} & {m.index for m in previous}
                if adjacent and shared:
                    runs[-1].append((i, matches))
                    continue
            runs.append([(i, matches)])
        return runs

    def _anchor(
        self, text: str, tokens: Sequence[Token], run: list[tuple[int, list[Match]]], source_ref: str, channel: Channel
    ) -> Mention:
        start, end = tokens[run[0][0]].start, tokens[run[-1][0]].end
        mention_text = text[start:end]
        rows: dict[int, list[Match]] = {}
        for _, matches in run:
            for m in matches:
                rows.setdefault(m.index, []).append(m)
        merged = [
            Match(
                index,
                sum(m.score for m in ms) / len(run),
                min((m.how for m in ms), key=lambda h: _HOW_RANK[h]),
                ms[0].field,
                ms[0].term,
                mention_text,
            )
            for index, ms in rows.items()
        ]
        whole = self._matches(mention_text, fuzzy=False) if len(run) > 1 else []
        for m in whole:
            merged = [x for x in merged if x.index != m.index] + [m]
        merged.sort(key=lambda m: (-m.score, m.index))
        return Mention(
            "anchor",
            mention_text,
            (start, end),
            None,
            None,
            channel,
            source_ref,
            f"registry/{self.name}",
            attrs={"source": self.name, "matches": tuple(merged)},
        )

    # -- candidates -----------------------------------------------------------------------------------------------

    def _anchor_mentions(self, q: SourceQuery) -> list[Mention]:
        if q.mentions:
            return [
                m
                for m in q.mentions
                if m.kind == "anchor" and m.attrs.get("source") == self.name and m.channel is Channel.USER
            ]
        return self.find_anchors(q.request) if q.request else []

    def matched(self, q: SourceQuery) -> dict[int, tuple[Match, Mention | None]]:
        """Best match per row over the query's anchors (or the retriever's hits)."""
        best: dict[int, tuple[Match, Mention | None]] = {}
        if callable(self.retriever) or self.retriever == "bm25":
            for index, score in self._retrieve(q):
                best[index] = (Match(index, score, "exact", self.key, "", q.retrieval_text), None)
            return best
        for mention in self._anchor_mentions(q):
            for m in mention.attrs.get("matches", ()):
                if m.index not in best or m.score > best[m.index][0].score:
                    best[m.index] = (m, mention)
        return best

    def _retrieve(self, q: SourceQuery) -> list[tuple[int, float]]:
        if callable(self.retriever):
            return [(int(i), float(s)) for i, s in self.retriever(q, self.rows)]
        docs = [[w for f in self.match for w in words(" ".join(_strings(row.get(f))))] for row in self.rows]
        query = [w for w in words(q.retrieval_text) if w not in _STOP and w not in _VERBS]
        return BM25(docs).top(query)

    def _rank_key(self, index: int, match: Match) -> tuple[Any, ...]:
        recency = self.rows[index].get(self.recency) if self.recency else None
        return (-match.score, _neg(recency), self.label_of(self.rows[index]).casefold(), index)

    def ranked(self, q: SourceQuery) -> list[Candidate]:
        """Every matched row ranked (score, recency, label); with ``q.widen`` the unmatched rows follow by label."""
        matched = self.matched(q)
        order = sorted(matched, key=lambda i: self._rank_key(i, matched[i][0]))
        out = [self.candidate(self.rows[i], *matched[i]) for i in order]
        if q.widen:
            rest = sorted(
                (i for i in range(len(self.rows)) if i not in matched),
                key=lambda i: self.label_of(self.rows[i]).casefold(),
            )
            out += [self.candidate(self.rows[i]) for i in rest]
        return out

    def candidates(self, q: SourceQuery) -> list[Candidate]:
        """The whole registry when small, else the page ``[offset, offset + k)`` of the anchored ranking."""
        if self.whole and not q.offset:
            matched = self.matched(q)
            return [self.candidate(row, *matched.get(i, (None, None)), whole=True) for i, row in enumerate(self.rows)]
        return self.ranked(q)[q.offset : q.offset + q.k]

    def candidate(
        self, row: Mapping[str, Any], match: Match | None = None, mention: Mention | None = None, *, whole: bool = False
    ) -> Candidate:
        """The candidate of one row (value = key field, label from the template, match note + describe)."""
        key = row[self.key]
        prov: dict[str, Any] = {"source": self.name, "key": key}
        describe = self.describe_of(row)
        noun = self.item[:1].upper() + self.item[1:]
        if match is not None:
            shown = mention.text if mention is not None else match.mention
            head = f'{noun} {HOW_PHRASE[match.how]} "{shown}"'
            prov.update({"anchor": shown, "match": match.how, "score": round4(match.score)})
            if mention is not None:
                prov["mention"] = mention.prov()
            if mention is not None and mention.negated:
                head += "; " + NEGATION_NOTE.format(quote=mention.attrs.get("negation", shown))
        else:
            head = noun
        if whole:
            prov["whole"] = True
        text = f"{head}: {describe}." if describe else f"{head}."
        return Candidate(
            label=self.label_of(row), value=key, text=text, channel=self.channel, prov=prov, attrs=self.attrs_of(row)
        )

    def group_of(self, candidate: Candidate) -> str | None:
        """Hierarchy group of a candidate (the ``hierarchy`` attribute), for widen rounds."""
        if self.hierarchy is None:
            return None
        value = candidate.attrs.get(self.hierarchy)
        return None if value is None else str(value)


def _strings(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple)):
        return [str(v) for v in value]
    return [] if value is None else [str(value)]


def _neg(value: Any) -> Any:
    """Sort key putting larger (more recent) values first; ``None`` last."""
    if value is None:
        return (1, 0)
    if isinstance(value, (int, float)):
        return (0, -value)
    return (0, "".join(chr(0x10FFFF - ord(c)) for c in str(value)))


__all__ = ["HOW_PHRASE", "Registry", "render_template"]
