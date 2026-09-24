"""Candidate sources (spec §4.4): registries, file indexes, host callables, catalog tools (``ToolSource``) and MCP
resources (``MCPResources``), plus pure-Python retrieval and source specs as data (``SourceConfig``,
``build_sources``: the format of ``jevtools.toml``, evaluation contexts and ``jevtools lint --sources``)."""

from jevtools.sources.base import RankedSource, Source, SourceQuery, item_noun
from jevtools.sources.files import FileIndex, date_attr
from jevtools.sources.mcp import MCPResources
from jevtools.sources.provider import Provider
from jevtools.sources.registry import Registry, render_template
from jevtools.sources.retrieval import BM25, Match, fuzzy_matches, trigram
from jevtools.sources.specs import SourceConfig, build_source, build_sources
from jevtools.sources.toolsource import ToolSource, jsonpath, tool_sources

__all__ = [
    "BM25",
    "FileIndex",
    "MCPResources",
    "Match",
    "Provider",
    "RankedSource",
    "Registry",
    "Source",
    "SourceConfig",
    "SourceQuery",
    "ToolSource",
    "build_source",
    "build_sources",
    "date_attr",
    "fuzzy_matches",
    "item_noun",
    "jsonpath",
    "render_template",
    "tool_sources",
    "trigram",
]
