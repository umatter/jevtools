"""Mention extraction (spec §4.2.1): code-only extractors over the request, history and observations.

``run_extractors(ctx, catalog)`` returns the round's :class:`Mentions` with span claiming and negation marks.
Each extractor module has a named R equivalent (extraction is *profiled*, not byte-identical, across ports).
"""

from jevtools.extract.base import (
    PRIORITY,
    ExtractProfile,
    Mention,
    Mentions,
    SourceText,
    claim,
    profile_for,
    run_extractors,
)
from jevtools.extract.coref import coref_candidates, iter_entities
from jevtools.extract.temporal import RangeReading, Reading, TemporalValue, localize
from jevtools.extract.text import perspective_variant

__all__ = [
    "PRIORITY",
    "ExtractProfile",
    "Mention",
    "Mentions",
    "RangeReading",
    "Reading",
    "SourceText",
    "TemporalValue",
    "claim",
    "coref_candidates",
    "iter_entities",
    "localize",
    "perspective_variant",
    "profile_for",
    "run_extractors",
]
