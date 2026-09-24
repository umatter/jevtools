"""Coreference candidates from the entity store (spec §4.2.1 "Coreference", §6.4).

Pronouns and anaphors (``her him them it this one the same the other again as last time``) add the entity
store's recent entities of the slot's type as ``history`` candidates whose trust is inherited from their origin
channel. ``the other X`` adds X-type entities *except* the previously bound one, with the description
"not the Anna Keller from your last email".

The entity store is duck-typed: ``None``, an iterable of entities, an object with an ``entities`` attribute or
a mapping with an ``"entities"`` key. An entity is an object or a mapping with ``id, type, value, label, channel,
origin, turn, step, pinned`` (and optionally ``tool``, ``attrs``).
"""

from __future__ import annotations

from collections.abc import Collection, Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

from jevtools.candidates import Candidate, Channel
from jevtools.extract.base import Mentions

MAX_COREF = 5


@dataclass(frozen=True)
class EntityView:
    """A normalized entity of the store."""

    id: str
    type: str
    value: Any
    label: str
    origin: Channel
    turn: int = 0
    step: int | None = None
    pinned: bool = False
    tool: str | None = None
    attrs: Mapping[str, Any] = field(default_factory=dict)


def _get(entity: Any, name: str, default: Any = None) -> Any:
    if isinstance(entity, Mapping):
        return entity.get(name, default)
    return getattr(entity, name, default)


def iter_entities(store: Any) -> list[EntityView]:
    """The store's entities, most recent first (by turn, then step)."""
    if store is None:
        return []
    raw: Any = _get(store, "entities", store) if not isinstance(store, (list, tuple)) else store
    if callable(raw):
        raw = raw()
    views: list[EntityView] = []
    for entity in raw if isinstance(raw, Iterable) else ():
        origin = _get(entity, "origin") or _get(entity, "channel") or Channel.HISTORY
        value = _get(entity, "value")
        views.append(
            EntityView(
                id=str(_get(entity, "id", value)),
                type=str(_get(entity, "type", "")),
                value=value,
                label=str(_get(entity, "label") or value),
                origin=Channel(origin),
                turn=int(_get(entity, "turn", 0)),
                step=_get(entity, "step"),
                pinned=bool(_get(entity, "pinned", False)),
                tool=_get(entity, "tool"),
                attrs=dict(_get(entity, "attrs") or {}),
            )
        )
    return sorted(views, key=lambda e: (-e.turn, -(e.step or 0)))


def anaphor(mentions: Mentions) -> str | None:
    """The request's anaphor cue (canonical form), if any."""
    for m in mentions.cues("anaphor"):
        if m.in_request:
            return str(m.attrs.get("canonical"))
    return None


def coref_candidates(mentions: Mentions, store: Any, types: Collection[str], *, k: int = MAX_COREF) -> list[Candidate]:
    """``history`` candidates for an anaphoric request: recent entities whose type is in ``types``."""
    cue = anaphor(mentions)
    if cue is None:
        return []
    entities = [e for e in iter_entities(store) if e.type in types]
    excluded: EntityView | None = None
    if cue == "the other":
        excluded = next((e for e in entities if e.pinned), None)
        entities = [e for e in entities if excluded is None or e.value != excluded.value]
    out: list[Candidate] = []
    seen: set[str] = set()
    for entity in entities:
        key = repr(entity.value)
        if key in seen:
            continue
        seen.add(key)
        where = f" from your last {entity.tool.replace('_', ' ')}" if entity.tool else ""
        text = f"Mentioned earlier{where}: {entity.label}."
        if excluded is not None:
            other = f" from your last {excluded.tool.replace('_', ' ')}" if excluded.tool else ""
            text = f"Mentioned earlier: {entity.label}; not the {excluded.label}{other}."
        out.append(
            Candidate(
                value=entity.value,
                label=entity.label,
                text=text,
                channel=Channel.HISTORY,
                origin=entity.origin,
                prov={"source": "entities", "entity": entity.id, "anaphor": cue},
                attrs=dict(entity.attrs),
            )
        )
        if len(out) >= k:
            break
    return out


__all__ = ["MAX_COREF", "EntityView", "anaphor", "coref_candidates", "iter_entities"]
