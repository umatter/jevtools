"""The temporal parser emits every reading (spec §4.2.5), DST-aware (Europe/Zurich 2026-03-29 and 2026-10-25)."""

from __future__ import annotations

from datetime import date, datetime, time, timedelta
from zoneinfo import ZoneInfo

import pytest

from jevtools.extract.locales import get_locale
from jevtools.extract.temporal import TemporalValue, localize, parse
from tests.support import SCENARIO_NOW

ZURICH = "Europe/Zurich"


def at(y: int, m: int, d: int, hh: int = 12, mm: int = 0) -> datetime:
    return datetime(y, m, d, hh, mm, tzinfo=ZoneInfo(ZURICH))


def one(text: str, now: datetime = SCENARIO_NOW, locale: str = "en") -> TemporalValue:
    found = parse(text, now, ZURICH, get_locale(locale))
    assert len(found) == 1, found
    return found[0][1]


def isos(value: TemporalValue) -> list[str]:
    return [r.dt.isoformat() if r.dt else str(r.date) for r in value.readings]


# -- "next Tuesday" said on every weekday ------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("said", "coming", "following"),
    [
        (date(2026, 9, 21), 1, 8),  # Monday
        (date(2026, 9, 22), 7, 14),  # Tuesday
        (date(2026, 9, 23), 6, 13),
        (date(2026, 9, 24), 5, 12),  # Thursday (spec: +5 and +12)
        (date(2026, 9, 25), 4, 11),
        (date(2026, 9, 26), 3, 10),
        (date(2026, 9, 27), 2, 9),  # Sunday
    ],
)
def test_next_tuesday_on_every_weekday(said: date, coming: int, following: int) -> None:
    value = one("next Tuesday", at(said.year, said.month, said.day, 10))
    assert value.dates == [said + timedelta(days=coming), said + timedelta(days=following)]
    assert [r.name for r in value.readings] == ["next_weekday:coming", "next_weekday:following"]


def test_r5_readings_and_glosses() -> None:
    found = parse("Book a 45 min sync with Bob and Carol next Tuesday at 3pm", SCENARIO_NOW, ZURICH)
    assert [text for text, _ in found] == ["next Tuesday at 3pm"]
    value = found[0][1]
    assert isos(value) == ["2026-09-29T15:00:00+02:00", "2026-10-06T15:00:00+02:00"]
    assert [r.gloss for r in value.readings] == [
        "read as the coming Tuesday, in 5 days",
        "read as the Tuesday of the following week, in 12 days",
    ]


def test_at_3_gives_two_readings() -> None:
    value = one("call me at 3")
    assert sorted(r.dt.time() for r in value.readings if r.dt) == [time(3), time(15)]
    assert {r.gloss for r in value.readings} == {"read as tomorrow at 03:00", "read as today at 15:00"}


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("03/04", ["2027-04-03", "2027-03-04"]),
        ("05/10", ["2026-10-05", "2027-05-10"]),
        ("25/12", ["2026-12-25"]),
        ("12/25/2026", ["2026-12-25"]),
    ],
)
def test_slash_dates_day_month_and_month_day(text: str, expected: list[str]) -> None:
    assert isos(one(text)) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("today", ["2026-09-24"]),
        ("tomorrow", ["2026-09-25"]),
        ("the day after tomorrow", ["2026-09-26"]),
        ("in 3 days", ["2026-09-27"]),
        ("in 2 hours", ["2026-09-24T16:05:00+02:00"]),
        ("in 10 minutes", ["2026-09-24T14:15:00+02:00"]),
        ("in half an hour", ["2026-09-24T14:35:00+02:00"]),
        ("2 days from now", ["2026-09-26"]),
        ("2026-09-29", ["2026-09-29"]),
        ("29.09.2026", ["2026-09-29"]),
        ("29 September", ["2026-09-29"]),
        ("September 29th, 2027", ["2027-09-29"]),
        ("tomorrow at 3pm", ["2026-09-25T15:00:00+02:00"]),
        ("tomorrow 15:00", ["2026-09-25T15:00:00+02:00"]),
        ("at 09:30 tomorrow", ["2026-09-25T09:30:00+02:00"]),
        ("3:30pm", ["2026-09-24T15:30:00+02:00"]),
        ("noon tomorrow", ["2026-09-25T12:00:00+02:00"]),
        ("2026-09-29T15:00", ["2026-09-29T15:00:00+02:00"]),
        ("2026-09-29T15:00Z", ["2026-09-29T15:00:00+00:00"]),
        ("tomorrow at 3pm UTC", ["2026-09-25T15:00:00+00:00"]),
        ("Tuesday", ["2026-09-29"]),
        ("this Friday", ["2026-09-25"]),
        ("last Monday", ["2026-09-21"]),
    ],
)
def test_single_readings(text: str, expected: list[str]) -> None:
    value = one(text)
    assert isos(value) == expected
    assert all(r.gloss == "" for r in value.readings)  # a single reading needs no gloss


@pytest.mark.parametrize(
    ("text", "locale", "expected"),
    [
        ("nächsten Dienstag um 15 Uhr", "de", ["2026-09-29T15:00:00+02:00", "2026-10-06T15:00:00+02:00"]),
        ("morgen um 15 Uhr", "de", ["2026-09-25T15:00:00+02:00"]),
        ("übermorgen", "de", ["2026-09-26"]),
        ("in 3 Tagen", "de", ["2026-09-27"]),
        ("mardi prochain à 15h", "fr", ["2026-09-29T15:00:00+02:00", "2026-10-06T15:00:00+02:00"]),
        ("demain à 15h30", "fr", ["2026-09-25T15:30:00+02:00"]),
        ("dans 2 heures", "fr", ["2026-09-24T16:05:00+02:00"]),
    ],
)
def test_german_and_french(text: str, locale: str, expected: list[str]) -> None:
    assert isos(one(text, locale=locale)) == expected


def test_end_of_day_profile() -> None:
    value = one("by end of day")
    assert [r.dt.strftime("%H:%M") for r in value.readings if r.dt] == ["17:00", "18:00", "23:59"]


def test_bare_numbers_are_not_times() -> None:
    assert parse("at 3 locations", SCENARIO_NOW, ZURICH) == []
    assert parse("Email Anna that I'll be 10 minutes late", SCENARIO_NOW, ZURICH) == []
    assert parse("a 2h meeting", SCENARIO_NOW, ZURICH) == []


# -- DST ------------------------------------------------------------------------------------------------------------


def test_autumn_fold_gives_both_occurrences() -> None:
    value = one("2026-10-25 02:30")
    assert isos(value) == ["2026-10-25T02:30:00+02:00", "2026-10-25T02:30:00+01:00"]
    assert [r.fold for r in value.readings] == ["first", "second"]
    assert "first occurrence, UTC+02:00" in value.readings[0].gloss


def test_spring_gap_shifts_forward() -> None:
    value = one("2026-03-29 02:30")
    assert isos(value) == ["2026-03-29T03:30:00+02:00"]
    assert "the clock skips 02:30" in value.readings[0].gloss


def test_offsets_across_dst() -> None:
    before_autumn = at(2026, 10, 24, 12)
    assert isos(one("tomorrow at 15:00", before_autumn)) == ["2026-10-25T15:00:00+01:00"]
    assert isos(one("in 24 hours", before_autumn)) == ["2026-10-25T11:00:00+01:00"]
    before_spring = at(2026, 3, 28, 12)
    assert isos(one("tomorrow at 3pm", before_spring)) == ["2026-03-29T15:00:00+02:00"]
    assert isos(one("in 24 hours", before_spring)) == ["2026-03-29T13:00:00+02:00"]


def test_localize() -> None:
    assert [dt.isoformat() for dt, _, _ in localize(date(2026, 9, 29), time(15), ZURICH)] == [
        "2026-09-29T15:00:00+02:00"
    ]
    assert len(localize(date(2026, 10, 25), time(2, 30), ZURICH)) == 2
    assert localize(date(2026, 3, 29), time(2, 30), ZURICH)[0][0].hour == 3
    assert localize(date(2026, 9, 29), time(15), "UTC+05:30")[0][0].isoformat() == "2026-09-29T15:00:00+05:30"


# -- ranges and vague cues -------------------------------------------------------------------------------------------


def test_explicit_ranges() -> None:
    value = one("between 2 and 4pm tomorrow")
    assert not value.readings
    assert [(r.start.isoformat(), r.end.isoformat()) for r in value.ranges if r.start and r.end] == [
        ("2026-09-25T14:00:00+02:00", "2026-09-25T16:00:00+02:00")
    ]
    nine_to_five = one("from 9 to 5")
    assert [(r.start.strftime("%H:%M"), r.end.strftime("%H:%M")) for r in nine_to_five.ranges if r.start and r.end] == [
        ("09:00", "17:00")
    ]
    after = one("after 3")
    assert [r.start.strftime("%H:%M") for r in after.ranges if r.start] == ["03:00", "15:00"]
    assert all(r.end is None and not r.vague for r in after.ranges)


@pytest.mark.parametrize(
    ("text", "start", "end"),
    [
        ("after lunch", "2026-09-24T13:00:00+02:00", "2026-09-24T17:00:00+02:00"),
        ("tomorrow morning", "2026-09-25T08:00:00+02:00", "2026-09-25T12:00:00+02:00"),
        ("sometime next week", "2026-09-28T00:00:00+02:00", "2026-10-04T23:59:00+02:00"),
    ],
)
def test_vague_cues_are_range_readings(text: str, start: str, end: str) -> None:
    value = one(text)
    assert value.vague and not value.readings
    assert [(r.start.isoformat(), r.end.isoformat()) for r in value.ranges if r.start and r.end] == [(start, end)]
