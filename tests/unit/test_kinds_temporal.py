"""The temporal resolver (spec §4.2.5, ext.temporal): single Choice, factorized date × time, formats, ranges."""

from __future__ import annotations

from typing import Any

import pytest

from jevtools.candidates import NONE_OF_THESE, NOT_STATED, Bottom
from jevtools.context import Context
from jevtools.kinds import temporal as temporal_kind
from jevtools.spec.catalog import Catalog
from tests.kinds_support import choice, custom, decode, resolve, scenario
from tests.support import SCENARIO_NOW

R5 = "Book a 45 min sync with Bob and Carol next Tuesday at 3pm"
COMING, FOLLOWING = "Tue 2026-09-29 15:00 (Europe/Zurich)", "Tue 2026-10-06 15:00 (Europe/Zurich)"


def test_r5_start_single_choice() -> None:
    catalog, rc = scenario(R5)
    tool = catalog["create_event"]
    slot = tool.slot("start")
    pool, (question,) = resolve(tool, slot, rc)
    assert pool.meta["mode"] == "single"
    assert [(c.label, c.value, c.text) for c in pool.candidates] == [
        (COMING, "2026-09-29T15:00:00+02:00", '"next Tuesday at 3pm" read as the coming Tuesday, in 5 days.'),
        (
            FOLLOWING,
            "2026-10-06T15:00:00+02:00",
            '"next Tuesday at 3pm" read as the Tuesday of the following week, in 12 days.',
        ),
    ]
    assert pool.candidates[0].prov["reading"] == "next_weekday:coming"
    assert question.instructions.endswith(
        "Which option is the start time of the event? `now` is the current date and time."
    )
    answer = choice(question, {COMING: 0.80, FOLLOWING: 0.18, NOT_STATED: 0.01, NONE_OF_THESE: 0.01})
    result = decode(tool, slot, pool, {question.qid: answer}, rc)
    assert result.value == "2026-09-29T15:00:00+02:00" and result.factor == pytest.approx(0.80)
    assert result.normalizer == "temporal.iso8601@1" and result.label == COMING
    assert [(a.value, round(a.p, 2)) for a in result.alternatives] == [("2026-10-06T15:00:00+02:00", 0.18)]


def test_unary_constraint_drops_past_readings() -> None:
    props = {"start": {"type": "string", "format": "date-time"}}
    tool, rc = custom(
        "schedule_call", props, "call at 3", required=["start"], tool_xjev={"constraints": ["start > now"]}
    )
    pool, _ = resolve(tool, tool.slot("start"), rc)
    assert [c.value for c in pool.candidates] == ["2026-09-25T03:00:00+02:00", "2026-09-24T15:00:00+02:00"]
    tool, rc = custom(
        "schedule_call", props, "call yesterday at 3pm", required=["start"], tool_xjev={"constraints": ["start > now"]}
    )
    pool, _ = resolve(tool, tool.slot("start"), rc)
    assert pool.empty and any("start > now" in note for note in pool.notes)


def test_factorized_date_and_time(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(temporal_kind, "MAX_SINGLE", 1)
    catalog, rc = scenario(R5)
    tool = catalog["create_event"]
    slot = tool.slot("start")
    pool, questions = resolve(tool, slot, rc)
    assert pool.meta["mode"] == "factorized"
    date_q, time_q = questions
    assert (date_q.qid, date_q.family, time_q.qid, time_q.family) == (
        "create_event.start.date",
        "date",
        "create_event.start.time",
        "time",
    )
    assert [o.label for o in date_q.options] == ["Tue 2026-09-29", "Tue 2026-10-06"]
    assert [o.label for o in time_q.options] == ["15:00"]
    assert "Which option is the date of the start time of the event?" in str(date_q.instructions)
    answers = {
        date_q.qid: choice(date_q, {"Tue 2026-09-29": 0.8, "Tue 2026-10-06": 0.18, NOT_STATED: 0.02}),
        time_q.qid: choice(time_q, {"15:00": 0.9, NONE_OF_THESE: 0.1}),
    }
    result = decode(tool, slot, pool, answers, rc)
    assert result.value == "2026-09-29T15:00:00+02:00" and result.factor == pytest.approx(0.72)
    assert set(result.parts) == {"date", "time"} and result.shape == "ok"
    missing_time = {**answers, time_q.qid: choice(time_q, {NOT_STATED: 0.95, "15:00": 0.05})}
    assert decode(tool, slot, pool, missing_time, rc).value is Bottom.MISSING


def test_date_without_time_is_factorized() -> None:
    props = {"start": {"type": "string", "format": "date-time"}}
    tool, rc = custom("schedule_call", props, "call me next Tuesday", required=["start"])
    pool, questions = resolve(tool, tool.slot("start"), rc)
    assert pool.meta["mode"] == "factorized" and pool.evidence_backed
    assert [q.family for q in questions] == ["date", "time"] and questions[1].options == []


def test_date_time_and_duration_formats() -> None:
    props = {
        "day": {"type": "string", "format": "date"},
        "at": {"type": "string", "format": "time"},
        "length": {"type": "string", "format": "duration"},
    }
    tool, rc = custom("plan_it", props, "next Tuesday at 3pm for 45 min")
    pool, _ = resolve(tool, tool.slot("day"), rc)
    assert [(c.label, c.value) for c in pool.candidates] == [
        ("Tue 2026-09-29", "2026-09-29"),
        ("Tue 2026-10-06", "2026-10-06"),
    ]
    pool, _ = resolve(tool, tool.slot("at"), rc)
    assert [(c.label, c.value) for c in pool.candidates] == [("15:00", "15:00:00")]
    pool, _ = resolve(tool, tool.slot("length"), rc)
    assert [c.value for c in pool.candidates] == ["PT45M"]


def test_dst_fold_labels_stay_unique() -> None:
    props = {"start": {"type": "string", "format": "date-time"}}
    tool, rc = custom("schedule_call", props, "at 2026-10-25 02:30", required=["start"])
    pool, _ = resolve(tool, tool.slot("start"), rc)
    assert [c.label for c in pool.candidates] == [
        "Sun 2026-10-25 02:30 (Europe/Zurich, UTC+01:00)",
        "Sun 2026-10-25 02:30 (Europe/Zurich, UTC+02:00)",
    ]


def test_vague_cues_are_recorded_not_offered() -> None:
    catalog, rc = scenario("Book a sync with Bob sometime next week")
    pool, questions = resolve(catalog["create_event"], catalog["create_event"].slot("start"), rc)
    assert pool.empty and questions == []
    assert [r["vague"] for r in pool.meta["ranges"]] == [True]


def test_range_coupled_slots() -> None:
    xjev = {"x-jev": {"range": {"min": "start", "max": "end"}}}
    props = {
        "start": {"type": "string", "format": "date-time", **xjev},
        "end": {"type": "string", "format": "date-time", **xjev},
    }
    tool, rc = custom("block_time", props, "block between 2 and 4pm tomorrow", required=["start", "end"])
    start_pool, (question,) = resolve(tool, tool.slot("start"), rc)
    end_pool, end_questions = resolve(tool, tool.slot("end"), rc)
    assert end_questions == [] and end_pool.meta["mode"] == "range_max"
    (pair,) = start_pool.candidates
    assert pair.label == "Fri 2026-09-25 14:00 – 16:00 (Europe/Zurich)"
    answer = {question.qid: choice(question, {pair.label: 0.9, NOT_STATED: 0.1})}
    start = decode(tool, tool.slot("start"), start_pool, answer, rc)
    end = decode(tool, tool.slot("end"), end_pool, answer, rc)
    assert start.value == "2026-09-25T14:00:00+02:00" and end.value == "2026-09-25T16:00:00+02:00"
    assert start.factor == end.factor == pytest.approx(0.9)


# -- review regression: a factorized slot with a default ---------------------------------------------------------------

_DEFAULTED = {
    "start": {"type": "string", "format": "date-time", "default": "2026-10-01T09:00:00+02:00"},
    "title": {"type": "string", "enum": ["sync", "review"]},
}


def _defaulted_factorized() -> tuple[Any, Any, Any, Any, Any, Any]:
    tool, rc = custom("book_room", _DEFAULTED, "Book the room next Tuesday", required=["title"])
    slot = tool.slot("start")
    pool, (dq, tq) = resolve(tool, slot, rc)
    assert pool.meta["mode"] == "factorized" and tq.labels == [NOT_STATED, NONE_OF_THESE]
    return tool, rc, slot, pool, dq, tq


def test_factorized_default_is_never_a_date_or_time_part() -> None:
    tool, rc, slot, pool, dq, tq = _defaulted_factorized()
    for date_probs, time_probs in (
        ({"Tue 2026-09-29": 1.0}, {NONE_OF_THESE: 1.0}),
        ({"Tue 2026-09-29": 0.9, NOT_STATED: 0.1}, {NOT_STATED: 0.95, NONE_OF_THESE: 0.05}),
    ):
        result = decode(tool, slot, pool, {dq.qid: choice(dq, date_probs), tq.qid: choice(tq, time_probs)}, rc)
        assert result.shape in ("missing", "out_of_pool")  # a date without a time is incomplete: clarify


def test_factorized_default_applies_when_both_parts_are_not_stated() -> None:
    tool, rc, slot, pool, dq, tq = _defaulted_factorized()
    answers = {dq.qid: choice(dq, {NOT_STATED: 0.97}), tq.qid: choice(tq, {NOT_STATED: 0.97})}
    result = decode(tool, slot, pool, answers, rc)
    assert (result.value, result.shape) == ("2026-10-01T09:00:00+02:00", "ok")
    assert result.factor == pytest.approx(0.97 * 0.97)


def test_factorized_default_end_to_end_fails_closed() -> None:
    from jevtools.backends.scripted import ScriptedBackend
    from jevtools.router import Router

    catalog = Catalog.from_openai([{"type": "function", "function": {
        "name": "book_room", "description": "Book a meeting room.",
        "parameters": {"type": "object", "properties": _DEFAULTED, "required": ["title"]}}}])  # fmt: skip
    script = {"tool": {"book_room": 0.97, "NO_TOOL": 0.03}, "book_room.authorized": 0.97,
              "book_room.title": {"sync": 0.97, "NOT_STATED": 0.03},
              "book_room.start.date": {"Tue 2026-09-29": 0.9, "NOT_STATED": 0.1},
              "book_room.start.time": {"NOT_STATED": 0.95, "NONE_OF_THESE": 0.05}}  # fmt: skip
    ctx = Context(messages="Book the room for a sync next Tuesday", now=SCENARIO_NOW, locale="en-CH")
    decision = Router(catalog, backend=ScriptedBackend(script), context=ctx).decide(
        "Book the room for a sync next Tuesday"
    )
    assert decision.outcome != "execute"
