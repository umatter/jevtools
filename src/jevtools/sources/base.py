"""The candidate-source protocol (spec §4.4).

A source turns a :class:`SourceQuery` (slot, round mentions, request, shortlist size) into candidates. Sources
declare the tags they ``provide`` (matched against slot tags by inference, §3.3.1 row 11) and the ``channel``
their candidates carry (``registry`` for app-owned data).
"""

from __future__ import annotations

from collections.abc import Collection, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from jevtools.candidates import Candidate, Channel
from jevtools.extract.base import Mention

if TYPE_CHECKING:
    from jevtools.context import Context
    from jevtools.spec.models import SlotSpec


@dataclass(frozen=True)
class SourceQuery:
    """What a source is asked (spec §4.4 ``SourceQuery = {slot, mentions, request, k, context}``).

    - ``mentions``: the round's mentions (anchors of registries are among them).
    - ``text``: retrieval text overriding ``request`` (superlative refs retrieve on the request without the cue).
    - ``offset``/``k``: the page of the ranking to return (widen rounds page beyond the first K).
    - ``widen``: also rank items that match nothing, after the matching ones (coverage beyond the matches).
    """

    slot: SlotSpec | None = None
    mentions: Sequence[Mention] = ()
    request: str = ""
    k: int = 40
    context: Context | None = None
    offset: int = 0
    text: str | None = None
    widen: bool = False
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def retrieval_text(self) -> str:
        """The text to retrieve on."""
        return self.text if self.text is not None else self.request


@runtime_checkable
class Source(Protocol):
    """A candidate source: ``name``, ``provides`` tags, ``channel``, and ``candidates(query)``."""

    name: str
    provides: Collection[str]
    channel: Channel

    def candidates(self, q: SourceQuery) -> list[Candidate]: ...


@runtime_checkable
class RankedSource(Source, Protocol):
    """A source with a full ranking (used by widen rounds to page beyond K and group by hierarchy)."""

    def ranked(self, q: SourceQuery) -> list[Candidate]: ...

    def group_of(self, candidate: Candidate) -> str | None: ...


def item_noun(name: str) -> str:
    """Default item noun of a source: its name singularized (``contacts`` → ``contact``)."""
    base = name.replace("_", " ").strip()
    if base.endswith("ies") and len(base) > 4:
        return base[:-3] + "y"
    if base.endswith("s") and not base.endswith("ss") and len(base) > 3:
        return base[:-1]
    return base


__all__ = ["RankedSource", "Source", "SourceQuery", "item_noun"]
