"""Temporal parser (spec §4.2.5): emits **every reading** of an expression relative to ``now`` and the time zone.

- ``next Tuesday`` said on Thursday gives the coming Tuesday (+5 days) and the Tuesday of the following week
  (+12 days); said on any weekday it gives the next occurrence and the one a week later.
- ``at 3`` gives 03:00 and 15:00; ``03/04`` gives day/month and month/day; ``end of day`` gives 17:00, 18:00
  and 23:59.
- ``today``, ``tomorrow``, ``in N days/hours/minutes``, explicit and ISO dates, ``3pm``, ``15:00``, ``15 Uhr``,
  ``15h``, ``nächsten Dienstag``, ``mardi prochain``; explicit time zones are honoured.
- Ranges (``between 2 and 4pm``, ``after 3``) and vague cues (``after lunch``, ``sometime next week``,
  ``tomorrow morning``) become RANGE readings.

All calendar arithmetic is done in code with :mod:`zoneinfo`, DST-aware: a wall time that occurs twice (the
autumn fold) gives two readings, one that does not exist (the spring gap) is shifted forward with a note.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta, timezone
from decimal import Decimal
from zoneinfo import available_timezones

from jevtools.context import zone_of
from jevtools.extract.base import Mention, SourceText
from jevtools.extract.locales import Locale, all_locales
from jevtools.extract.tokens import fold

WEEKDAY_NAMES = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")
MONTH_NAMES = (
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
)
END_OF_DAY: tuple[tuple[time, str], ...] = (
    (time(17, 0), "the end of the working day"),
    (time(18, 0), "the end of the working day"),
    (time(23, 59), "the end of the calendar day"),
)
"""The ``end of day`` locale profile (§4.2.5)."""
TZ_ABBREVIATIONS: dict[str, str] = {
    "UTC": "UTC",
    "GMT": "UTC",
    "Z": "UTC",
    "CET": "Europe/Berlin",
    "CEST": "Europe/Berlin",
    "EST": "America/New_York",
    "EDT": "America/New_York",
    "ET": "America/New_York",
    "PST": "America/Los_Angeles",
    "PDT": "America/Los_Angeles",
    "PT": "America/Los_Angeles",
    "BST": "Europe/London",
    "MEZ": "Europe/Berlin",
    "MESZ": "Europe/Berlin",
}
_AMBIGUOUS_MONTHS = frozenset({"may", "mar"})
"""Month words that are also common English words: they need a capital letter before a day number."""
_CONNECTORS = frozenset({"at", "on", "um", "am", "a", "le", "the", "of", ",", "in", "for", "de", "du"})


# --------------------------------------------------------------------------------------------------------------------
# Readings
# --------------------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class DatePart:
    """One reading of a date expression."""

    value: date
    gloss: str
    """``the coming Tuesday, in 5 days``."""
    name: str
    """``next_weekday:coming``."""


@dataclass(frozen=True)
class TimePart:
    """One reading of a clock-time expression."""

    value: time
    gloss: str
    name: str


@dataclass(frozen=True)
class Reading:
    """A point reading: a complete ``dt`` (aware) or a date-only / time-only partial reading."""

    date: date | None
    time: time | None
    dt: datetime | None
    tz: str
    gloss: str
    """``read as the coming Tuesday, in 5 days`` (empty when the expression has a single reading)."""
    name: str
    """``next_weekday:coming`` (trace provenance)."""
    fold: str | None = None
    """``first``/``second`` occurrence of a wall time repeated by a DST fold."""

    @property
    def complete(self) -> bool:
        return self.dt is not None


@dataclass(frozen=True)
class RangeReading:
    """A RANGE reading (explicit range or vague cue)."""

    start: datetime | None
    end: datetime | None
    tz: str
    gloss: str
    name: str
    vague: bool


@dataclass(frozen=True)
class TemporalValue:
    """Everything one temporal mention can mean."""

    readings: tuple[Reading, ...] = ()
    ranges: tuple[RangeReading, ...] = ()
    tz: str = "UTC"

    @property
    def dates(self) -> list[date]:
        """Distinct dates, in reading order."""
        out: list[date] = []
        for r in self.readings:
            day = r.dt.date() if r.dt else r.date
            if day is not None and day not in out:
                out.append(day)
        return out

    @property
    def times(self) -> list[time]:
        """Distinct wall-clock times, in reading order."""
        out: list[time] = []
        for r in self.readings:
            clock = r.dt.time() if r.dt else r.time
            if clock is not None and clock not in out:
                out.append(clock)
        return out

    @property
    def vague(self) -> bool:
        """Only vague range readings (no point reading)."""
        return not self.readings and any(r.vague for r in self.ranges)


# --------------------------------------------------------------------------------------------------------------------
# Calendar helpers (DST-aware)
# --------------------------------------------------------------------------------------------------------------------


def localize(day: date, clock: time, zone_name: str) -> list[tuple[datetime, str | None, str]]:
    """Every aware datetime a wall-clock ``day clock`` denotes in ``zone_name``: ``(dt, fold, note)``.

    Normal times give one; a DST fold gives two (``first``/``second`` occurrence); a DST gap gives the time shifted
    forward by the gap, with a note.
    """
    zone = zone_of(zone_name)
    naive = datetime.combine(day, clock)
    first = naive.replace(tzinfo=zone, fold=0)
    second = naive.replace(tzinfo=zone, fold=1)
    if first.utcoffset() == second.utcoffset():
        return [(first, None, "")]
    roundtrip = first.astimezone(timezone.utc).astimezone(zone)
    if roundtrip.replace(tzinfo=None) != naive:  # gap: the wall time does not exist
        shifted = first.astimezone(timezone.utc).astimezone(zone)
        return [(shifted, None, f"the clock skips {clock:%H:%M} that night; read as {shifted:%H:%M}")]
    return [
        (first, "first", f"first occurrence, {utc_offset(first)}"),
        (second, "second", f"second occurrence, {utc_offset(second)}"),
    ]


def utc_offset(dt: datetime) -> str:
    """``UTC+02:00``."""
    delta = dt.utcoffset() or timedelta(0)
    sign = "+" if delta >= timedelta(0) else "-"
    minutes = abs(int(delta.total_seconds())) // 60
    return f"UTC{sign}{minutes // 60:02d}:{minutes % 60:02d}"


def in_days(n: int) -> str:
    """``today`` / ``tomorrow`` / ``in 5 days`` / ``2 days ago``."""
    if n == 0:
        return "today"
    if n == 1:
        return "tomorrow"
    if n == -1:
        return "yesterday"
    return f"in {n} days" if n > 0 else f"{-n} days ago"


def day_label(day: date) -> str:
    """``29 September 2026``."""
    return f"{day.day} {MONTH_NAMES[day.month - 1]} {day.year}"


def next_occurrence(month: int, day: int, today: date) -> date | None:
    """The next ``day.month`` on or after today (this year, else next year); ``None`` for impossible dates."""
    for year in (today.year, today.year + 1):
        try:
            candidate = date(year, month, day)
        except ValueError:
            continue
        if candidate >= today:
            return candidate
    return None


# --------------------------------------------------------------------------------------------------------------------
# Atoms
# --------------------------------------------------------------------------------------------------------------------


@dataclass
class _Atom:
    kind: str
    """``date time abs range tz``."""
    start: int
    end: int
    dates: list[DatePart] = field(default_factory=list)
    times: list[TimePart] = field(default_factory=list)
    absolute: list[tuple[datetime, str]] = field(default_factory=list)
    """``(instant, reading name)`` of relative offsets (``in 2 hours``)."""
    time_ranges: list[tuple[time | None, time | None, str, str, bool]] = field(default_factory=list)
    """``(start, end, gloss, name, vague)`` on the atom's date(s)."""
    date_ranges: list[tuple[date, date, str, str]] = field(default_factory=list)
    tz: str | None = None


class _Parser:
    """Finds temporal atoms in one text and merges adjacent ones into mentions."""

    def __init__(self, source: SourceText, now: datetime, tz: str, primary: Locale):
        self.source = source
        self.text = source.text
        self.tokens = source.tokens
        self.now = now
        self.tz = tz
        self.today = now.date()
        self.primary = primary
        self.locales = (primary,) + tuple(loc for loc in all_locales() if loc.code != primary.code)
        self.atoms: list[_Atom] = []

    # -- utilities ---------------------------------------------------------------------------------------------

    def _words(self, attr: str) -> frozenset[str]:
        return frozenset().union(*(getattr(loc, attr) for loc in self.locales))

    def _mapping(self, attr: str) -> dict[str, int]:
        merged: dict[str, int] = {}
        for loc in reversed(self.locales):
            merged.update(getattr(loc, attr))
        return merged

    def _token_index(self, offset: int) -> int:
        return next((i for i, t in enumerate(self.tokens) if t.start >= offset), len(self.tokens))

    def _free(self, start: int, end: int) -> bool:
        return all(end <= a.start or start >= a.end for a in self.atoms)

    def _add(self, atom: _Atom) -> None:
        if self._free(atom.start, atom.end):
            self.atoms.append(atom)

    # -- numeric dates and datetimes (regex over the raw text) ---------------------------------------------------

    def iso_datetimes(self) -> None:
        pattern = re.compile(r"\b(\d{4})-(\d{2})-(\d{2})[T ](\d{2}):(\d{2})(?::(\d{2}))?(Z|[+-]\d{2}:?\d{2})?(?!\d)")
        for m in pattern.finditer(self.text):
            try:
                day = date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
                clock = time(int(m.group(4)), int(m.group(5)), int(m.group(6) or 0))
            except ValueError:
                continue
            atom = _Atom("date", m.start(), m.end(), dates=[DatePart(day, day_label(day), "iso_date")])
            atom.times = [TimePart(clock, f"at {clock:%H:%M}", "iso_time")]
            if m.group(7):
                atom.tz = _offset_zone(m.group(7))
            self._add(atom)

    def iso_dates(self) -> None:
        for m in re.finditer(r"\b(\d{4})-(\d{1,2})-(\d{1,2})\b", self.text):
            try:
                day = date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
            except ValueError:
                continue
            self._add(_Atom("date", m.start(), m.end(), dates=[DatePart(day, day_label(day), "iso_date")]))

    def dotted_dates(self) -> None:
        for m in re.finditer(r"\b(\d{1,2})\.(\d{1,2})\.(\d{4}|\d{2})?(?![\d.]\d)", self.text):
            day = self._dmy(int(m.group(1)), int(m.group(2)), m.group(3))
            if day is not None:
                self._add(_Atom("date", m.start(), m.end(), dates=[DatePart(day, day_label(day), "date:d.m")]))

    def slash_dates(self) -> None:
        for m in re.finditer(r"\b(\d{1,2})/(\d{1,2})(?:/(\d{4}|\d{2}))?\b", self.text):
            a, b, year = int(m.group(1)), int(m.group(2)), m.group(3)
            parts: list[DatePart] = []
            dm = self._dmy(a, b, year)
            md = self._dmy(b, a, year)
            if dm is not None:
                parts.append(DatePart(dm, f"{day_label(dm)} (day/month)", "date:d/m"))
            if md is not None and md != dm:
                parts.append(DatePart(md, f"{day_label(md)} (month/day)", "date:m/d"))
            if parts:
                self._add(_Atom("date", m.start(), m.end(), dates=parts))

    def _dmy(self, day: int, month: int, year: str | None) -> date | None:
        if not (1 <= month <= 12 and 1 <= day <= 31):
            return None
        if year is None:
            return next_occurrence(month, day, self.today)
        full = int(year) + (2000 if len(year) == 2 else 0)
        try:
            return date(full, month, day)
        except ValueError:
            return None

    def month_name_dates(self) -> None:
        months = self._mapping("months")
        day_first = re.compile(
            r"\b(\d{1,2})(?:st|nd|rd|th|er|\.)?\s+(?:of\s+)?([^\W\d_]+)\.?(?:,?\s+(\d{4}))?", re.UNICODE
        )
        month_first = re.compile(r"\b([^\W\d_]+)\.?\s+(\d{1,2})(?:st|nd|rd|th)?\b(?:,?\s+(\d{4}))?", re.UNICODE)
        for m in day_first.finditer(self.text):
            month = months.get(fold(m.group(2)))
            if month is not None:
                self._month_atom(m.start(), m.end(), int(m.group(1)), month, m.group(3))
        for m in month_first.finditer(self.text):
            word = m.group(1)
            month = months.get(fold(word))
            if month is not None and (word[:1].isupper() or fold(word) not in _AMBIGUOUS_MONTHS):
                self._month_atom(m.start(), m.end(), int(m.group(2)), month, m.group(3))

    def _month_atom(self, start: int, end: int, day: int, month: int, year: str | None) -> None:
        value = self._dmy(day, month, year)
        if value is not None:
            self._add(_Atom("date", start, end, dates=[DatePart(value, day_label(value), "date:month_name")]))

    # -- clock times ---------------------------------------------------------------------------------------------

    def clock_times(self) -> None:
        at = r"(?:(?:at|um|à|by)\s+)?"
        ampm = re.compile(at + r"\b(\d{1,2})(?:[:.](\d{2}))?\s*([ap])\.?\s?m\b\.?", re.IGNORECASE)
        for m in ampm.finditer(self.text):
            hour, minute = int(m.group(1)), int(m.group(2) or 0)
            if not (1 <= hour <= 12 and minute < 60):
                continue
            hour = hour % 12 + (12 if m.group(3).lower() == "p" else 0)
            clock = time(hour, minute)
            self._add(_Atom("time", m.start(), m.end(), times=[TimePart(clock, f"at {clock:%H:%M}", "clock:ampm")]))
        uhr = re.compile(r"(?:(?:um|at)\s+)?\b(\d{1,2})\s*uhr(?:\s*(\d{2}))?\b", re.IGNORECASE)
        for m in uhr.finditer(self.text):
            self._fixed_time(m, int(m.group(1)), int(m.group(2) or 0), "clock:uhr")
        french = re.compile(r"(?:(?P<at>à|um|at)\s+)?\b(\d{1,2})\s?h(\d{2})?\b", re.IGNORECASE)
        for m in french.finditer(self.text):
            if m.group("at") or m.group(3) or self.primary.code in ("fr", "de"):
                self._fixed_time(m, int(m.group(2)), int(m.group(3) or 0), "clock:h")
        colon = re.compile(at + r"\b(\d{1,2}):(\d{2})\b(?!\s*[ap]\.?\s?m)", re.IGNORECASE)
        for m in colon.finditer(self.text):
            hour, minute = int(m.group(1)), int(m.group(2))
            raw = m.group(0).split()[-1]
            ambiguous = self.primary.code == "en" and 1 <= hour <= 11 and not raw.startswith("0")
            self._hour_atom(m.start(), m.end(), hour, minute, ambiguous, "clock:24h")
        bare = re.compile(r"\b(?:at|um|à|by)\s+(\d{1,2})\b(?:\s*(o'clock|oclock))?", re.IGNORECASE)
        for m in bare.finditer(self.text):
            if m.group(2) is None and not self._bare_hour_context(m.end()):
                continue
            self._hour_atom(m.start(), m.end(), int(m.group(1)), 0, True, "clock:bare")
        oclock = re.compile(r"\b(\d{1,2})\s*(?:o'clock|oclock)\b", re.IGNORECASE)
        for m in oclock.finditer(self.text):
            self._hour_atom(m.start(), m.end(), int(m.group(1)), 0, True, "clock:bare")

    def _bare_hour_context(self, end: int) -> bool:
        """``at 3`` is a time only when followed by nothing, punctuation or another temporal/connector word."""
        i = self._token_index(end)
        if i >= len(self.tokens):
            return True
        token = self.tokens[i]
        if token.kind == "punct":
            return token.text not in (":", ".", "%", "-", "/")
        word = token.folded
        temporal = (
            self._words("next_words")
            | self._words("this_words")
            | self._words("on_words")
            | frozenset(self._mapping("weekdays"))
            | {w for p in self._mapping_phrases() for w in p}
            | {"and", "or", "with", "for", "in", "tomorrow", "today", "tonight", "then", "so", "sharp", "please"}
        )
        return word in temporal

    def _mapping_phrases(self) -> set[tuple[str, ...]]:
        phrases: set[tuple[str, ...]] = set()
        for loc in self.locales:
            phrases |= set(loc.relative_days)
        return phrases

    def _fixed_time(self, m: re.Match[str], hour: int, minute: int, name: str) -> None:
        if hour <= 23 and minute < 60:
            clock = time(hour, minute)
            self._add(_Atom("time", m.start(), m.end(), times=[TimePart(clock, f"at {clock:%H:%M}", name)]))

    def _hour_atom(self, start: int, end: int, hour: int, minute: int, ambiguous: bool, name: str) -> None:
        if hour > 23 or minute > 59:
            return
        clocks = [time(hour, minute)]
        if ambiguous and 1 <= hour <= 11:
            clocks.append(time(hour + 12, minute))
        elif ambiguous and hour == 12:
            clocks.append(time(0, minute))
        parts = [TimePart(c, f"at {c:%H:%M}", name + (":am" if c.hour < 12 else ":pm") * ambiguous) for c in clocks]
        self._add(_Atom("time", start, end, times=parts))

    def word_times(self) -> None:
        noon, midnight = self._words("noon"), self._words("midnight")
        for token in self.tokens:
            if token.folded in noon:
                self._add(_Atom("time", token.start, token.end, times=[TimePart(time(12), "at noon", "clock:noon")]))
            elif token.folded in midnight:
                self._add(
                    _Atom("time", token.start, token.end, times=[TimePart(time(0), "at midnight", "clock:midnight")])
                )
        for loc in self.locales:
            for phrase in loc.end_of_day:
                for i in self._phrase_positions(phrase):
                    start, end = self.tokens[i].start, self.tokens[i + len(phrase) - 1].end
                    parts = [TimePart(c, f"at {c:%H:%M} ({what})", f"end_of_day:{c:%H%M}") for c, what in END_OF_DAY]
                    self._add(_Atom("time", start, end, times=parts))

    def _phrase_positions(self, phrase: Sequence[str]) -> list[int]:
        n = len(phrase)
        return [
            i for i in range(len(self.tokens) - n + 1) if all(self.tokens[i + j].folded == phrase[j] for j in range(n))
        ]

    # -- word dates ------------------------------------------------------------------------------------------------

    def relative_days(self) -> None:
        phrases = sorted(
            {(p, d) for loc in self.locales for p, d in loc.relative_days.items()}, key=lambda pd: -len(pd[0])
        )
        for phrase, offset in phrases:
            for i in self._phrase_positions(phrase):
                start, end = self.tokens[i].start, self.tokens[i + len(phrase) - 1].end
                day = self.today + timedelta(days=offset)
                gloss = {0: "today", 1: "tomorrow", 2: "the day after tomorrow", -1: "yesterday"}.get(offset, "")
                self._add(_Atom("date", start, end, dates=[DatePart(day, gloss, f"relative:{offset:+d}")]))

    def weekdays(self) -> None:
        weekdays = self._mapping("weekdays")
        next_before, next_after = self._words("next_words"), self._words("next_after")
        this_words, last_words, on_words = self._words("this_words"), self._words("last_words"), self._words("on_words")
        for i, token in enumerate(self.tokens):
            target = weekdays.get(token.folded)
            if target is None or (len(token.text) <= 3 and not token.text[:1].isupper()):
                continue
            start, end, mode = token.start, token.end, "bare"
            prev = self.tokens[i - 1] if i > 0 else None
            nxt = self.tokens[i + 1] if i + 1 < len(self.tokens) else None
            if prev is not None and prev.folded in next_before:
                start, mode = prev.start, "next"
            elif prev is not None and prev.folded in this_words:
                start, mode = prev.start, "this"
            elif prev is not None and prev.folded in last_words:
                start, mode = prev.start, "last"
            elif prev is not None and prev.folded in on_words:
                start = prev.start
            if nxt is not None and nxt.folded in next_after:
                end, mode = nxt.end, "next"
            self._add(_Atom("date", start, end, dates=self._weekday_parts(target, mode)))

    def _weekday_parts(self, target: int, mode: str) -> list[DatePart]:
        name = WEEKDAY_NAMES[target]
        delta = (target - self.today.weekday()) % 7
        if mode == "last":
            back = delta - 7 if delta else -7
            day = self.today + timedelta(days=back)
            return [DatePart(day, f"last {name}, {in_days(back)}", "last_weekday")]
        if mode == "this" and delta == 0:
            return [DatePart(self.today, "today", "this_weekday:today")]
        if mode in ("this", "bare") and delta != 0:
            day = self.today + timedelta(days=delta)
            return [DatePart(day, f"the coming {name}, {in_days(delta)}", "weekday:coming")]
        coming = delta or 7
        if mode == "bare":  # "Tuesday" said on a Tuesday: today or a week from today
            return [
                DatePart(self.today, "today", "weekday:today"),
                DatePart(self.today + timedelta(days=7), f"next week's {name}, {in_days(7)}", "weekday:next_week"),
            ]
        return [
            DatePart(
                self.today + timedelta(days=coming), f"the coming {name}, {in_days(coming)}", "next_weekday:coming"
            ),
            DatePart(
                self.today + timedelta(days=coming + 7),
                f"the {name} of the following week, {in_days(coming + 7)}",
                "next_weekday:following",
            ),
        ]

    # -- relative offsets ------------------------------------------------------------------------------------------

    def offsets(self) -> None:
        in_words = self._words("in_words")
        for i, token in enumerate(self.tokens):
            if token.folded not in in_words:
                continue
            found = self._amount_unit(i + 1)
            if found is not None:
                amount, unit, j = found
                self._offset_atom(token.start, self.tokens[j - 1].end, amount, unit)
        for loc in self.locales:
            for phrase in loc.from_now:
                for i in range(len(self.tokens)):
                    found = self._amount_unit(i)
                    if found is None:
                        continue
                    amount, unit, j = found
                    tail = tuple(t.folded for t in self.tokens[j : j + len(phrase)])
                    if tail == phrase:
                        end = self.tokens[j + len(phrase) - 1].end
                        self._offset_atom(self.tokens[i].start, end, amount, unit)

    def _amount_unit(self, i: int) -> tuple[Decimal, str, int] | None:
        """``3 days`` / ``an hour`` / ``half an hour`` / ``zwei Stunden`` starting at token ``i``."""
        from jevtools.extract.numbers import _phrase_at, _unit_at, parse_number

        if i >= len(self.tokens):
            return None
        phrase = _phrase_at(self.tokens, i, self.locales)
        if phrase is not None:
            return phrase
        token = self.tokens[i]
        amount: Decimal | None = None
        if token.kind == "number":
            amount = parse_number(token.text, self.primary.decimal)
        else:
            words = {k: v for loc in self.locales for k, v in loc.number_words.items()}
            if token.folded in words:
                amount = Decimal(words[token.folded])
        if amount is None:
            return None
        found = _unit_at(
            self.tokens, i + 1, i + 1 < len(self.tokens) and self.tokens[i + 1].start == token.end, self.locales
        )
        if found is None:
            return None
        return amount, found[0], found[1]

    def _offset_atom(self, start: int, end: int, amount: Decimal, unit: str) -> None:
        if unit in ("day", "week", "month", "year"):
            days = int(amount * {"day": 1, "week": 7, "month": 30, "year": 365}[unit])
            if unit == "month":
                day = _add_months(self.today, int(amount))
                days = (day - self.today).days
            day = self.today + timedelta(days=days)
            self._add(_Atom("date", start, end, dates=[DatePart(day, in_days(days), f"offset:{unit}")]))
            return
        seconds = {"second": 1, "minute": 60, "hour": 3600, "millisecond": 0.001}.get(unit)
        if seconds is None:
            return
        instant = (self.now.astimezone(timezone.utc) + timedelta(seconds=float(amount) * seconds)).astimezone(
            zone_of(self.tz)
        )
        self._add(_Atom("abs", start, end, absolute=[(instant, f"offset:{unit}")]))

    # -- ranges and vague cues ---------------------------------------------------------------------------------------

    def ranges(self) -> None:
        self._between()
        self._after_before()
        self._weeks()
        self._day_parts()

    def _between(self) -> None:
        between, joiners = self._words("between_words"), self._words("and_words")
        pattern = re.compile(
            r"\b(?P<kw>[^\W\d_]+)\s+(?P<h1>\d{1,2})(?::(?P<m1>\d{2}))?\s*(?P<ap1>[ap]\.?m\.?)?\s*"
            r"(?P<join>[^\W\d_]+|-|–)\s*(?P<h2>\d{1,2})(?::(?P<m2>\d{2}))?\s*(?P<ap2>[ap]\.?m\.?|uhr|h)?\b",
            re.IGNORECASE,
        )
        for m in pattern.finditer(self.text):
            if fold(m.group("kw")) not in between or fold(m.group("join")) not in joiners:
                continue
            h1, h2 = int(m.group("h1")), int(m.group("h2"))
            ap2 = (m.group("ap2") or "").lower()[:1]
            ap1 = (m.group("ap1") or "").lower()[:1] or (ap2 if ap2 in "ap" and ap2 else "")
            t1, t2 = _clock(h1, int(m.group("m1") or 0), ap1), _clock(h2, int(m.group("m2") or 0), ap2)
            if t1 is None or t2 is None:
                continue
            if not ap1 and not ap2 and not self._bare_hour_context(m.end()):
                continue
            spans: list[tuple[time | None, time | None]] = [(t1, t2)]
            if not ap1 and not ap2 and h2 <= 11:  # "between 2 and 4": afternoon too; "from 9 to 5": 09–17
                pm1, pm2 = _clock(h1, int(m.group("m1") or 0), "p"), _clock(h2, int(m.group("m2") or 0), "p")
                spans.append((pm1, pm2))
                if t2 <= t1:
                    spans.append((t1, pm2))
            valid = [(a, b) for a, b in dict.fromkeys(spans) if a is not None and b is not None and a < b]
            if not valid:
                continue
            atom = _Atom("range", m.start(), m.end())
            atom.time_ranges = [(a, b, f"between {a:%H:%M} and {b:%H:%M}", "range:between", False) for a, b in valid]
            self._add(atom)

    def _after_before(self) -> None:
        after, before = self._words("after_words"), self._words("before_words")
        pattern = re.compile(
            r"\b(?P<kw>[^\W\d_]+)\s+(?P<h>\d{1,2})(?::(?P<m>\d{2}))?\s*(?P<ap>[ap]\.?m\.?|uhr)?\b", re.IGNORECASE
        )
        for m in pattern.finditer(self.text):
            kw = fold(m.group("kw"))
            if kw not in after and kw not in before:
                continue
            ap = (m.group("ap") or "").lower()[:1]
            if not ap and not self._bare_hour_context(m.end()):
                continue
            hour, minute = int(m.group("h")), int(m.group("m") or 0)
            clocks = [_clock(hour, minute, ap)]
            if not ap and 1 <= hour <= 11:
                clocks.append(_clock(hour, minute, "p"))
            atom = _Atom("range", m.start(), m.end())
            for clock in clocks:
                if clock is None:
                    continue
                if kw in after:
                    atom.time_ranges.append((clock, None, f"after {clock:%H:%M}", "range:after", False))
                else:
                    atom.time_ranges.append((None, clock, f"before {clock:%H:%M}", "range:before", False))
            if atom.time_ranges:
                self._add(atom)

    def _weeks(self) -> None:
        week, weekend = self._words("week_words"), self._words("weekend_words")
        next_words, this_words = self._words("next_words") | self._words("next_after"), self._words("this_words")
        sometime = self._words("sometime_words")
        for i, token in enumerate(self.tokens):
            if token.folded not in week and token.folded not in weekend:
                continue
            prev = self.tokens[i - 1].folded if i > 0 else ""
            nxt = self.tokens[i + 1].folded if i + 1 < len(self.tokens) else ""
            is_next = prev in next_words or nxt in next_words
            if not (is_next or prev in this_words):
                continue
            start = self.tokens[i - 1].start if prev in next_words | this_words else token.start
            end = self.tokens[i + 1].end if nxt in next_words else token.end
            if i >= 2 and self.tokens[i - 2].folded in sometime:
                start = self.tokens[i - 2].start
            monday = self.today - timedelta(days=self.today.weekday()) + timedelta(days=7 if is_next else 0)
            if token.folded in weekend:
                first, last, what = monday + timedelta(days=5), monday + timedelta(days=6), "weekend"
            else:
                first, last, what = monday, monday + timedelta(days=6), "week"
            label = f"{'next' if is_next else 'this'} {what} ({day_label(first)} to {day_label(last)})"
            atom = _Atom("range", start, end, date_ranges=[(first, last, label, f"vague:{what}")])
            self._add(atom)

    def _day_parts(self) -> None:
        for loc in self.locales:
            for phrase, (a, b) in loc.vague_phrases.items():
                for i in self._phrase_positions(phrase):
                    start, end = self.tokens[i].start, self.tokens[i + len(phrase) - 1].end
                    gloss = f"{' '.join(phrase)} ({a:02d}:00 to {b:02d}:00)"
                    atom = _Atom("range", start, end)
                    atom.time_ranges = [(time(a), time(b), gloss, "vague:" + "_".join(phrase), True)]
                    self._add(atom)
            for word, (a, b) in loc.day_parts.items():
                for token in self.tokens:
                    if token.folded != word:
                        continue
                    atom = _Atom("range", token.start, token.end)
                    atom.time_ranges = [(time(a), time(b), f"{word} ({a:02d}:00 to {b:02d}:00)", f"vague:{word}", True)]
                    self._add(atom)

    def timezones(self) -> None:
        names = {n for n in available_timezones() if "/" in n}
        for m in re.finditer(r"\b([A-Z]{1,4}|[A-Z][A-Za-z]+/[A-Za-z_]+(?:/[A-Za-z_]+)?)\b", self.text):
            word = m.group(1)
            zone = TZ_ABBREVIATIONS.get(word) or (word if word in names else None)
            if zone is not None:
                self.atoms.append(_Atom("tz", m.start(), m.end(), tz=zone))

    # -- merging -----------------------------------------------------------------------------------------------------

    def parse(self) -> list[tuple[int, int, TemporalValue]]:
        self.iso_datetimes()
        self.iso_dates()
        self.dotted_dates()
        self.ranges()
        self.clock_times()
        self.word_times()
        self.relative_days()
        self.weekdays()
        self.offsets()
        self.slash_dates()
        self.month_name_dates()
        self.timezones()
        out: list[tuple[int, int, TemporalValue]] = []
        for group in self._groups():
            value = self._build(group)
            if value is not None and (value.readings or value.ranges):
                start, end = group[0].start, group[-1].end
                while end > start and self.text[end - 1] in ".,;":
                    end -= 1
                out.append((start, end, value))
        return out

    def _groups(self) -> list[list[_Atom]]:
        atoms = sorted(self.atoms, key=lambda a: (a.start, a.end))
        groups: list[list[_Atom]] = []
        for atom in atoms:
            if groups and self._joinable(groups[-1], atom):
                groups[-1].append(atom)
            else:
                if atom.kind == "tz":
                    continue
                groups.append([atom])
        return groups

    def _joinable(self, group: list[_Atom], atom: _Atom) -> bool:
        last = group[-1]
        gap = self.text[last.end : atom.start]
        words = [fold(w) for w in re.findall(r"[^\W\d_]+|,", gap)]
        if re.search(r"[.;!?]\s", gap) or len(words) > 2 or any(w not in _CONNECTORS for w in words):
            return False
        kinds = [a.kind for a in group]
        if atom.kind == "tz":
            return "time" in kinds or "abs" in kinds or "range" in kinds
        if atom.kind in ("date", "time") and atom.kind in kinds:
            return False
        return not ("abs" in kinds or atom.kind == "abs")

    def _build(self, group: list[_Atom]) -> TemporalValue | None:
        tz = next((a.tz for a in group if a.tz), None) or self.tz
        date_atoms = [a for a in group if a.dates]
        dates = min((a.dates for a in date_atoms), key=len) if date_atoms else []
        times = [t for a in group if a.kind in ("time",) or (a.kind == "date" and a.times) for t in a.times]
        readings: list[Reading] = []
        for a in group:
            for instant, name in a.absolute:
                local = instant.astimezone(zone_of(tz))
                readings.append(Reading(local.date(), local.time(), local, tz, "", name))
        has_range = any(a.kind == "range" for a in group)
        if not readings and times:
            readings = self._combine(dates, times, tz)
        elif not readings and dates and not has_range:
            ambiguous = len(dates) > 1
            readings = [
                Reading(d.value, None, None, tz, f"read as {d.gloss}" if ambiguous else "", d.name) for d in dates
            ]
        ranges = self._ranges(group, dates, tz)
        return TemporalValue(tuple(readings), tuple(ranges), tz)

    def _default_dates(self, clock: time, tz: str) -> list[DatePart]:
        today = self.today
        first = localize(today, clock, tz)[0][0]
        if first >= self.now:
            return [DatePart(today, "today", "default:today")]
        return [DatePart(today + timedelta(days=1), "tomorrow", "default:tomorrow")]

    def _combine(self, dates: list[DatePart], times: list[TimePart], tz: str) -> list[Reading]:
        date_ambiguous = len(dates) > 1
        time_ambiguous = len(times) > 1
        out: list[Reading] = []
        for t in times:
            for d in dates or self._default_dates(t.value, tz):
                for dt, fold_name, note in localize(d.value, t.value, tz):
                    gloss = _gloss(
                        d.gloss if date_ambiguous or time_ambiguous else "", t.gloss if time_ambiguous else ""
                    )
                    if note:
                        gloss = f"{gloss or f'read as {dt:%H:%M}'} ({note})"
                    name = d.name if not time_ambiguous else f"{d.name}+{t.name}"
                    out.append(Reading(dt.date(), dt.timetz().replace(tzinfo=None), dt, tz, gloss, name, fold_name))
        return out

    def _ranges(self, group: list[_Atom], dates: list[DatePart], tz: str) -> list[RangeReading]:
        out: list[RangeReading] = []
        for a in group:
            for first, last, gloss, name in a.date_ranges:
                start = localize(first, time(0), tz)[0][0]
                end = localize(last, time(23, 59), tz)[0][0]
                out.append(RangeReading(start, end, tz, gloss, name, True))
            for t1, t2, gloss, name, vague in a.time_ranges:
                for d in dates or [DatePart(self.today, "today", "default:today")]:
                    lower = localize(d.value, t1, tz)[0][0] if t1 else None
                    upper = localize(d.value, t2, tz)[0][0] if t2 else None
                    where = f" {d.gloss}" if d.gloss else ""
                    out.append(RangeReading(lower, upper, tz, f"{gloss}{where}", name, vague))
        return out


def _gloss(date_gloss: str, time_gloss: str) -> str:
    """``read as the coming Tuesday, in 5 days`` / ``read as tomorrow at 15:00`` / ``""`` (single reading)."""
    if not date_gloss and not time_gloss:
        return ""
    joiner = ", " if "," in date_gloss else " "
    return "read as " + joiner.join(p for p in (date_gloss, time_gloss) if p)


def _clock(hour: int, minute: int, ap: str) -> time | None:
    if minute > 59:
        return None
    if ap in ("a", "p"):
        if not 1 <= hour <= 12:
            return None
        hour = hour % 12 + (12 if ap == "p" else 0)
    if hour > 23:
        return None
    return time(hour, minute)


def _offset_zone(text: str) -> str:
    if text in ("Z", "z"):
        return "UTC"
    sign, digits = text[0], text[1:].replace(":", "")
    return f"UTC{sign}{digits[:2]}:{digits[2:4]}"


def _add_months(day: date, months: int) -> date:
    month = day.month - 1 + months
    year = day.year + month // 12
    month = month % 12 + 1
    for d in (day.day, 30, 29, 28):
        try:
            return date(year, month, d)
        except ValueError:
            continue
    return date(year, month, 28)


def parse(text: str, now: datetime, tz: str, locale: Locale | None = None) -> list[tuple[str, TemporalValue]]:
    """Every temporal expression in ``text`` with its readings (convenience wrapper for tests and tools)."""
    from jevtools.candidates import Channel
    from jevtools.extract.locales import get_locale
    from jevtools.extract.tokens import tokenize

    source = SourceText("request", text, Channel.USER, tuple(tokenize(text)))
    parser = _Parser(source, now, tz, locale or get_locale("en"))
    return [(text[s:e], value) for s, e, value in parser.parse()]


def extract(source: SourceText, now: datetime, tz: str, locale: Locale | None = None) -> list[Mention]:
    """Temporal mentions (kind ``temporal``, value :class:`TemporalValue`) in one text."""
    from jevtools.extract.locales import get_locale

    primary = locale or get_locale("en")
    parser = _Parser(source, now, tz, primary)
    return [
        Mention(
            "temporal", source.text[s:e], (s, e), value, "time", source.channel, source.ref, f"temporal/{primary.code}"
        )
        for s, e, value in parser.parse()
    ]


__all__ = [
    "END_OF_DAY",
    "DatePart",
    "RangeReading",
    "Reading",
    "TemporalValue",
    "TimePart",
    "day_label",
    "extract",
    "in_days",
    "localize",
    "parse",
    "utc_offset",
    "zone_of",
]
