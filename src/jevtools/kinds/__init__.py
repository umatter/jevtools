"""Slot-kind resolvers (spec §4). Each kind module registers itself on import; :func:`get_resolver` imports lazily."""

from jevtools.kinds import enum as _enum  # noqa: F401  (registers the enum resolver)
from jevtools.kinds.base import (
    KIND_MODULES,
    RESOLVERS,
    Alternative,
    DefaultInfo,
    ResolveContext,
    Resolver,
    SlotResult,
    ValueEntry,
    Widenable,
    decode_choice,
    elect,
    get_resolver,
    probe_question,
    register_resolver,
    resolve_default,
    slot_question,
    slot_sentinels,
    unasked_result,
)

__all__ = [
    "KIND_MODULES",
    "RESOLVERS",
    "Alternative",
    "DefaultInfo",
    "ResolveContext",
    "Resolver",
    "SlotResult",
    "ValueEntry",
    "Widenable",
    "decode_choice",
    "elect",
    "get_resolver",
    "probe_question",
    "register_resolver",
    "resolve_default",
    "slot_question",
    "slot_sentinels",
    "unasked_result",
]
