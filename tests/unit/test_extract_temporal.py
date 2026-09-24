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


# -- DST ---------------------------------------------------------------------------------------------------------------


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


# -- ranges and vague cues ---------------------------------------------------------------------------------------------


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


# -- review regressions ------------------------------------------------------------------------------------------------

NOW_0900 = at(2026, 9, 24, 9, 0)  # Thursday


@pytest.mark.parametrize(
    "text", ["Set the thermostat to 21.5.", "Set the ratio to 1.5. Thanks!", "Upgrade to version 1.2.3 please"]
)
def test_decimals_and_versions_are_not_dotted_dates(text: str) -> None:
    assert parse(text, SCENARIO_NOW, ZURICH, get_locale("en")) == []


def test_dotted_runs_are_not_dates() -> None:
    found = parse("Ping 10.1.1.5 tomorrow", SCENARIO_NOW, ZURICH, get_locale("en"))
    assert [span for span, _ in found] == ["tomorrow"]


def test_german_dotted_dates_without_year() -> None:
    assert isos(one("am 21.5. um 10 Uhr", locale="de")) == ["2027-05-21T10:00:00+02:00"]


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("2026-09-24 12:00 UTC", ["2026-09-24T12:00:00+00:00"]),
        ("2026-09-24T12:00:00.000Z", ["2026-09-24T12:00:00+00:00"]),
        ("2026-09-24T12:00:00.123456+00:00", ["2026-09-24T12:00:00+00:00"]),
        ("Remind me at 15:00 UTC+2", ["2026-09-24T15:00:00+02:00"]),
        ("Remind me at 15:00 GMT-05:00", ["2026-09-24T15:00:00-05:00"]),
        ("at 3pm JST", ["2026-09-25T15:00:00+09:00"]),
        ("at 3pm EST", ["2026-09-24T15:00:00-04:00"]),
    ],
)
def test_explicit_zones_are_honoured(text: str, expected: list[str]) -> None:
    assert isos(one(text, NOW_0900)) == expected


def test_ambiguous_zone_abbreviations_give_one_reading_per_zone() -> None:
    value = one("at 3pm CST", NOW_0900)
    assert isos(value) == ["2026-09-24T15:00:00-05:00", "2026-09-24T15:00:00+08:00"]
    assert all("CST taken as" in r.gloss for r in value.readings)
    assert len(one("at 3pm IST", NOW_0900).readings) == 3


def test_word_like_zone_abbreviations_need_to_follow_the_time() -> None:
    [(span, value)] = parse("at 3pm for ICT team", NOW_0900, ZURICH, get_locale("en"))
    assert span == "at 3pm" and value.tz == ZURICH


MONDAY = at(2026, 9, 21, 9, 0)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Remind me Thursday next week at 3pm", ["2026-10-01T15:00:00+02:00"]),
        ("Book next week Tuesday at 10am", ["2026-09-29T10:00:00+02:00"]),
        ("Thursday of next week at 3pm", ["2026-10-01T15:00:00+02:00"]),
        ("Remind me Thursday next week", ["2026-10-01"]),
        ("Monday this week", ["2026-09-21"]),
    ],
)
def test_weekday_qualified_by_a_week(text: str, expected: list[str]) -> None:
    assert isos(one(text, MONDAY)) == expected


@pytest.mark.parametrize("said", [date(2026, 9, 21) + timedelta(days=k) for k in range(7)])
def test_thursday_next_week_on_every_weekday(said: date) -> None:
    now = at(said.year, said.month, said.day, 9, 0)
    monday = said - timedelta(days=said.weekday()) + timedelta(days=7)
    assert isos(one("Thursday next week", now)) == [str(monday + timedelta(days=3))]


@pytest.mark.parametrize(
    ("text", "start", "end"),
    [
        ("Book the room 4-6pm on Friday", "2026-09-25T16:00:00+02:00", "2026-09-25T18:00:00+02:00"),
        ("Book the room tomorrow 9-11am", "2026-09-25T09:00:00+02:00", "2026-09-25T11:00:00+02:00"),
        ("Schedule the call between 10 and 2pm", "2026-09-24T10:00:00+02:00", "2026-09-24T14:00:00+02:00"),
        ("Book the room 4pm-6pm on Friday", "2026-09-25T16:00:00+02:00", "2026-09-25T18:00:00+02:00"),
    ],
)
def test_dash_and_mixed_meridiem_ranges(text: str, start: str, end: str) -> None:
    value = one(text, NOW_0900)
    assert value.readings == ()  # never the end of the window as a point
    assert [(r.start.isoformat() if r.start else None, r.end.isoformat() if r.end else None) for r in value.ranges] == [
        (start, end)
    ]


def test_invalid_range_reserves_its_span() -> None:
    found = parse("between 11 and 9pm", NOW_0900, ZURICH, get_locale("en"))
    assert all(not v.readings for _, v in found)


@pytest.mark.parametrize(
    ("text", "locale"),
    [
        ("Remind me tomorrow at midnight", "en"),
        ("midnight tomorrow", "en"),
        ("morgen um Mitternacht", "de"),
        ("demain à minuit", "fr"),
    ],
)
def test_midnight_with_a_date_gives_start_and_end(text: str, locale: str) -> None:
    value = one(text, at(2026, 9, 24, 14, 5), locale)
    assert isos(value) == ["2026-09-25T00:00:00+02:00", "2026-09-26T00:00:00+02:00"]
    assert [r.name.split("+")[-1] for r in value.readings] == ["clock:midnight:start", "clock:midnight:end"]
    assert "the end of Fri" in value.readings[1].gloss


def test_friday_at_midnight_and_bare_midnight() -> None:
    assert len(one("due Friday at midnight", at(2026, 9, 24, 14, 5)).readings) == 2
    assert isos(one("at midnight", at(2026, 9, 24, 14, 5))) == ["2026-09-25T00:00:00+02:00"]


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Buche den Raum heute Morgen um 9 Uhr", ["2026-09-24T09:00:00+02:00"]),
        ("Guten Morgen, um 15 Uhr brauche ich den Raum", ["2026-09-24T15:00:00+02:00"]),
        ("Jeden Morgen um 8 Uhr Stand-up", ["2026-09-24T08:00:00+02:00"]),
        ("morgen um 9 Uhr", ["2026-09-25T09:00:00+02:00"]),
    ],
)
def test_german_morgen(text: str, expected: list[str]) -> None:
    found = parse(text, at(2026, 9, 24, 8, 0), ZURICH, get_locale("de"))
    assert [iso for _, v in found for iso in isos(v)] == expected


def test_guten_morgen_is_not_tomorrow() -> None:
    found = parse("Guten Morgen! Buche den Raum um 15 Uhr", at(2026, 9, 24, 8, 0), ZURICH, get_locale("de"))
    assert [span for span, _ in found] == ["um 15 Uhr"]


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Remind me in an hour and a half", "2026-09-24T15:35:00+02:00"),
        ("Remind me in 1 hour 30 minutes", "2026-09-24T15:35:00+02:00"),
        ("Remind me in 2 hours and 15 minutes", "2026-09-24T16:20:00+02:00"),
    ],
)
def test_compound_offsets(text: str, expected: str) -> None:
    assert isos(one(text, at(2026, 9, 24, 14, 5))) == [expected]
