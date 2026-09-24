"""Candidate sources (spec §4.4): registries, file indexes and host callables, plus pure-Python retrieval."""

from jevtools.sources.base import RankedSource, Source, SourceQuery, item_noun
from jevtools.sources.files import FileIndex, date_attr
from jevtools.sources.provider import Provider
from jevtools.sources.registry import Registry, render_template
from jevtools.sources.retrieval import BM25, Match, fuzzy_matches, trigram

__all__ = [
    "BM25",
    "FileIndex",
    "Match",
    "Provider",
    "RankedSource",
    "Registry",
    "Source",
    "SourceQuery",
    "date_attr",
    "fuzzy_matches",
    "item_noun",
    "render_template",
    "trigram",
]
