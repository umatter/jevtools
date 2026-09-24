"""``FileIndex``: workspace paths as a candidate source (spec §4.4, §4.2.8, §13.1 ``files``).

A BM25 index over path tokens (split on ``/ _ - .`` and camelCase, folded, plural-stemmed), queried with the
request's content words expanded by ``synonyms``. Each path carries attributes: ``date`` (``YYYY-MM-DD`` in the
file name, else ``mtime``), ``dirname``, ``name`` and the hierarchy ``group`` (``dirname``: the first two
directory levels) used by widen rounds.
"""

from __future__ import annotations

import posixpath
import re
from collections.abc import Callable, Collection, Mapping, Sequence
from datetime import date, datetime, timezone
from functools import cached_property
from typing import Any

from jevtools.candidates import Candidate, Channel
from jevtools.canonical import round4, sha256_of
from jevtools.extract.locales import all_locales
from jevtools.extract.tokens import words
from jevtools.sources.base import SourceQuery
from jevtools.sources.retrieval import BM25, canonical_terms, path_tokens, plural_stem, synonym_table

_DATE_RE = re.compile(r"(\d{4})-(\d{2})-(\d{2})")
_STOP = frozenset().union(*(loc.stopwords for loc in all_locales()))
_VERBS = frozenset().union(*(loc.command_verbs for loc in all_locales()))
_GENERIC = frozenset({"file", "document", "doc", "folder", "directory", "workspace"})
"""Words that name the kind of item rather than the item; they never make a hit on their own."""

Hierarchy = str | Callable[[str], str] | None


def date_attr(path: str, mtime: Any = None) -> str | None:
    """``YYYY-MM-DD`` from the file name, else from ``mtime`` (a datetime, a date, an epoch number or ISO text)."""
    match = _DATE_RE.search(posixpath.basename(path))
    if match:
        try:
            return date(int(match.group(1)), int(match.group(2)), int(match.group(3))).isoformat()
        except ValueError:
            pass
    if isinstance(mtime, datetime):
        return mtime.date().isoformat()
    if isinstance(mtime, date):
        return mtime.isoformat()
    if isinstance(mtime, (int, float)):
        return datetime.fromtimestamp(mtime, tz=timezone.utc).date().isoformat()
    if isinstance(mtime, str) and _DATE_RE.match(mtime):
        return mtime[:10]
    return None


class FileIndex:
    """A searchable set of workspace paths (spec §4.4 ``jt.FileIndex``)."""

    def __init__(
        self,
        name: str,
        paths: Sequence[str] | Callable[[], Sequence[str]],
        *,
        attrs: Mapping[str, Mapping[str, Any]] | None = None,
        synonyms: Mapping[str, Sequence[str]] | None = None,
        hierarchy: Hierarchy = "dirname",
        k: int = 40,
        provides: Collection[str] = ("path", "file"),
        channel: Channel | str = Channel.REGISTRY,
        item: str = "file",
    ) -> None:
        self.name = name
        self._paths = paths
        self.extra_attrs = dict(attrs or {})
        self.synonyms = dict(synonyms or {})
        self.hierarchy = hierarchy
        self.k = k
        self.provides: frozenset[str] = frozenset(provides)
        self.channel = Channel(channel)
        self.item = item

    def __repr__(self) -> str:
        return f"FileIndex({self.name!r}, {len(self.paths)} paths)"

    @cached_property
    def paths(self) -> tuple[str, ...]:
        """The indexed paths (a provider is called once)."""
        raw = self._paths() if callable(self._paths) else self._paths
        return tuple(dict.fromkeys(posixpath.normpath(p) for p in raw))

    @cached_property
    def synonym_table(self) -> dict[str, str]:
        return synonym_table(self.synonyms)

    @cached_property
    def index(self) -> BM25:
        return BM25([canonical_terms(path_tokens(p), self.synonym_table) for p in self.paths])

    def __len__(self) -> int:
        return len(self.paths)

    def __contains__(self, path: object) -> bool:
        return isinstance(path, str) and posixpath.normpath(path) in set(self.paths)

    def lookup(self, path: str) -> dict[str, Any] | None:
        """Attributes of an indexed path (``None`` if it is not in the index; TOCTOU re-resolution)."""
        return self.attrs_of(posixpath.normpath(path)) if path in self else None

    def attribute_names(self) -> frozenset[str]:
        """Attributes candidates carry: ``path name dirname date group`` plus the host's extra attributes."""
        extra = {key for attrs in self.extra_attrs.values() for key in attrs}
        return frozenset({"path", "name", "dirname", "date", "group", *extra})

    def content_sha256(self) -> str:
        return sha256_of({"name": self.name, "paths": list(self.paths)})

    # -- attributes -------------------------------------------------------------------------------------------------

    def group(self, path: str) -> str:
        """Hierarchy group of a path: the first two directory levels for ``dirname``, else the function's value."""
        if callable(self.hierarchy):
            return str(self.hierarchy(path))
        parts = posixpath.dirname(path).split("/")
        return "/".join(parts[:2]) if parts != [""] else "."

    def attrs_of(self, path: str) -> dict[str, Any]:
        extra = dict(self.extra_attrs.get(path, {}))
        out: dict[str, Any] = {
            "path": path,
            "name": posixpath.basename(path),
            "dirname": posixpath.dirname(path) or ".",
            "date": date_attr(path, extra.get("mtime")),
        }
        if self.hierarchy is not None:
            out["group"] = self.group(path)
        out.update({k: v for k, v in extra.items() if k not in out or out[k] is None})
        return out

    # -- retrieval --------------------------------------------------------------------------------------------------

    def query_terms(self, text: str) -> list[str]:
        """Content words of the text (no stop words, command verbs, generic nouns or bare numbers), stemmed, with
        synonyms mapped onto their head term."""
        content = [w for w in words(text) if w not in _STOP and w not in _VERBS and not w.isdigit()]
        return [t for t in canonical_terms(content, self.synonym_table) if plural_stem(t) not in _GENERIC]

    def search(self, text: str) -> list[tuple[int, float, list[str]]]:
        """``(index, score, matched terms)`` of every path matching the text, best first."""
        terms = self.query_terms(text)
        return [(i, s, self.index.matched(terms, i)) for i, s in self.index.top(terms)]

    def candidate(self, index: int, score: float | None = None, terms: Sequence[str] = (), rank: int = 0) -> Candidate:
        path = self.paths[index]
        attrs = self.attrs_of(path)
        dated = f" dated {attrs['date']}" if attrs["date"] else ""
        if terms:
            quoted = ", ".join(f'"{t}"' for t in terms)
            text = f"Workspace {self.item}{dated}; its path matches {quoted}."
        else:
            text = f"Workspace {self.item}{dated}."
        prov: dict[str, Any] = {"source": self.name, "key": path}
        if score is not None:
            prov.update({"anchor": " ".join(terms), "score": round4(score), "rank": rank})
        return Candidate(value=path, text=text, channel=self.channel, prov=prov, attrs=attrs)

    def ranked(self, q: SourceQuery) -> list[Candidate]:
        """Every hit ranked by BM25; with ``q.widen`` the other paths follow in path order."""
        hits = self.search(q.retrieval_text)
        out = [self.candidate(i, s, terms, rank) for rank, (i, s, terms) in enumerate(hits, start=1)]
        if q.widen:
            seen = {i for i, _, _ in hits}
            out += [
                self.candidate(i) for i in sorted(range(len(self.paths)), key=lambda i: self.paths[i]) if i not in seen
            ]
        return out

    def candidates(self, q: SourceQuery) -> list[Candidate]:
        """The page ``[offset, offset + k)`` of the BM25 ranking (hits only: a path must match a query term)."""
        return self.ranked(q)[q.offset : q.offset + q.k]

    def group_of(self, candidate: Candidate) -> str | None:
        """Hierarchy group of a candidate path."""
        return self.group(str(candidate.value)) if self.hierarchy is not None else None


__all__ = ["FileIndex", "date_attr"]
