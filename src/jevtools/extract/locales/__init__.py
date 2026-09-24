"""Locale lexicons for the extractors (spec §4.2.1 "Locales").

``en`` ships complete. ``de`` and ``fr`` ship numbers, money markers, weekdays, relative days and clock times.
Every word is stored *folded* (casefolded, accents stripped; see :func:`jevtools.extract.tokens.fold`).
Unknown expressions produce no candidates, which is the safe failure mode (the slot resolves to ``NOT_STATED``
or ``NONE_OF_THESE`` and then to a clarify).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

Phrase = tuple[str, ...]


@dataclass(frozen=True)
class Locale:
    """One locale's word lists (all folded)."""

    code: str
    decimal: str = "."
    """Decimal separator for ambiguous single separators (``1.250`` is 1.25 in ``en``, 1250 in ``de``)."""
    stopwords: frozenset[str] = frozenset()
    articles: frozenset[str] = frozenset()
    determiners: frozenset[str] = frozenset()
    prepositions: frozenset[str] = frozenset()
    conjunctions: frozenset[str] = frozenset()
    object_pronouns: frozenset[str] = frozenset()
    subject_pronouns: frozenset[str] = frozenset()
    command_verbs: frozenset[str] = frozenset()
    politeness: tuple[Phrase, ...] = ()
    clause_markers: tuple[Phrase, ...] = ()
    tell_verbs: frozenset[str] = frozenset()
    number_words: Mapping[str, int] = field(default_factory=dict)
    scale_words: Mapping[str, int] = field(default_factory=dict)
    fraction_words: Mapping[str, str] = field(default_factory=dict)
    """Word → decimal string (``half`` → ``0.5``)."""
    weekdays: Mapping[str, int] = field(default_factory=dict)
    months: Mapping[str, int] = field(default_factory=dict)
    relative_days: Mapping[Phrase, int] = field(default_factory=dict)
    next_words: frozenset[str] = frozenset()
    """Before a weekday: ``next Tuesday`` (coming + following week)."""
    next_after: frozenset[str] = frozenset()
    """After a weekday: ``mardi prochain``."""
    this_words: frozenset[str] = frozenset()
    last_words: frozenset[str] = frozenset()
    on_words: frozenset[str] = frozenset()
    in_words: frozenset[str] = frozenset()
    """``in 3 days`` / ``dans 3 jours``."""
    from_now: tuple[Phrase, ...] = ()
    """``3 days from now``."""
    at_words: frozenset[str] = frozenset()
    duration_units: Mapping[str, str] = field(default_factory=dict)
    """Unit word → canonical unit (``min`` → ``minute``)."""
    am_words: frozenset[str] = frozenset()
    pm_words: frozenset[str] = frozenset()
    clock_words: frozenset[str] = frozenset()
    """``15 Uhr``, ``3 o'clock``."""
    noon: frozenset[str] = frozenset()
    midnight: frozenset[str] = frozenset()
    end_of_day: tuple[Phrase, ...] = ()
    day_parts: Mapping[str, tuple[int, int]] = field(default_factory=dict)
    """Vague parts of the day → hour range (``morning`` → 8–12)."""
    vague_phrases: Mapping[Phrase, tuple[int, int]] = field(default_factory=dict)
    """Vague cues with an hour range (``after lunch`` → 13–17)."""
    week_words: frozenset[str] = frozenset()
    weekend_words: frozenset[str] = frozenset()
    sometime_words: frozenset[str] = frozenset()
    between_words: frozenset[str] = frozenset()
    after_words: frozenset[str] = frozenset()
    before_words: frozenset[str] = frozenset()
    and_words: frozenset[str] = frozenset()
    currency_words: Mapping[str, str] = field(default_factory=dict)
    negations: frozenset[str] = frozenset()
    superlatives: Mapping[Phrase, str] = field(default_factory=dict)
    """Cue phrase → canonical cue (``most recent`` → ``latest``)."""
    hedges: frozenset[str] = frozenset()
    chitchat: tuple[Phrase, ...] = ()
    anaphors: tuple[Phrase, ...] = ()
    day_part_homographs: Mapping[str, tuple[int, int]] = field(default_factory=dict)
    """Relative-day words that are also a day-part noun (de ``Morgen``: tomorrow / morning): read as the day part
    (a RANGE) after one of :attr:`day_part_homograph_cues` (``heute Morgen``, ``jeden Morgen``)."""
    day_part_homograph_cues: frozenset[str] = frozenset()
    greetings: frozenset[str] = frozenset()
    """Words that turn a following homograph into a greeting with no temporal meaning (``Guten Morgen``)."""


def get_locale(code: str | None) -> Locale:
    """The lexicon for a locale tag (``en-CH`` → ``en``); unknown locales fall back to ``en``."""
    from jevtools.extract.locales import de, en, fr

    table = {"en": en.EN, "de": de.DE, "fr": fr.FR}
    lang = (code or "en").split("-")[0].split("_")[0].lower()
    return table.get(lang, en.EN)


def all_locales() -> tuple[Locale, ...]:
    """Every shipped locale, ``en`` first (temporal expressions are recognized in all of them)."""
    from jevtools.extract.locales import de, en, fr

    return (en.EN, de.DE, fr.FR)


__all__ = ["Locale", "Phrase", "all_locales", "get_locale"]
