from __future__ import annotations

from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import pytest

from jevtools.context import Clock, Context, Observation, Turn, build_state, parse_messages, render_now
from tests.support import SCENARIO_NOW


def test_render_now() -> None:
    assert render_now(SCENARIO_NOW) == "Thursday 2026-09-24 14:05 Europe/Zurich (UTC+02:00)"
    utc = datetime(2026, 9, 24, 12, 5, tzinfo=timezone.utc)
    assert render_now(utc, "Europe/Zurich") == "Thursday 2026-09-24 14:05 Europe/Zurich (UTC+02:00)"
    assert (
        render_now(datetime(2026, 12, 1, 9, 0), "America/New_York")
        == "Tuesday 2026-12-01 09:00 America/New_York (UTC-05:00)"
    )
    assert render_now(utc) == "Thursday 2026-09-24 12:05 UTC (UTC+00:00)"
    with pytest.raises(ValueError):
        render_now(datetime(2026, 1, 1))


def test_clock() -> None:
    fixed = Clock("Europe/Zurich", fixed=datetime(2026, 10, 25, 1, 30, tzinfo=timezone.utc))
    assert fixed.now().isoformat() == "2026-10-25T02:30:00+01:00"  # after the DST switch
    assert Clock("Asia/Tokyo").now().tzinfo == ZoneInfo("Asia/Tokyo")


def test_message_parsing() -> None:
    assert parse_messages("hi") == [Turn(role="user", text="hi", content="hi")]
    turns = parse_messages(
        [
            {"role": "developer", "content": "Be brief."},
            {
                "role": "user",
                "content": [{"type": "text", "text": "a"}, {"type": "image_url"}, {"type": "text", "text": "b"}],
            },
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {"id": "c1", "type": "function", "function": {"name": "read_file", "arguments": '{"path":"a.txt"}'}}
                ],
            },  # fmt: skip
            {"role": "tool", "tool_call_id": "c1", "content": "hello world"},
        ]
    )
    assert [t.role for t in turns] == ["system", "user", "assistant", "tool"]
    assert turns[1].text == "a\nb"


def test_request_history_split(ctx_default: Context) -> None:
    assert ctx_default.request == "Email Anna that I'll be 10 minutes late"
    assert [t.to_state() for t in ctx_default.history] == [
        {"role": "user", "text": "What's next on my calendar?"},
        {"role": "assistant", "text": "14:30 ACME quarterly review with Anna Keller."},
    ]
    state = build_state(ctx_default)
    assert list(state) == ["request", "history", "now", "user"]
    assert state["now"] == "Thursday 2026-09-24 14:05 Europe/Zurich (UTC+02:00)"
    assert state["user"] == {"name": "Sam Muster", "home_city": "Zurich"}


def test_state_omits_empty_user_and_filters_shareable() -> None:
    ctx = Context(messages="hi", now=SCENARIO_NOW)
    assert "user" not in build_state(ctx)
    ctx = Context(messages="hi", now=SCENARIO_NOW, user={"name": "Sam", "iban": "CH00"}, shareable=("name",))
    assert build_state(ctx)["user"] == {"name": "Sam"}
    assert ctx.lookup("user.iban") == "CH00"
    with pytest.raises(KeyError):
        ctx.lookup("user.missing")
    with pytest.raises(KeyError):
        ctx.lookup("vault.secret")


def test_system_messages_only_when_included() -> None:
    messages = [{"role": "system", "content": "You are terse."}, {"role": "user", "content": "hi"}]
    assert "system" not in build_state(Context(messages=messages, now=SCENARIO_NOW))
    state = build_state(Context(messages=messages, now=SCENARIO_NOW, include_system=True))
    assert list(state) == ["request", "history", "now", "system"] and state["system"] == "You are terse."


def test_loop_sections_from_tool_messages() -> None:
    messages = [
        {"role": "user", "content": "Read a.txt"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {"id": "c1", "type": "function", "function": {"name": "read_file", "arguments": '{"path":"a.txt"}'}}
            ],
        },  # fmt: skip
        {"role": "tool", "tool_call_id": "c1", "content": "hello world"},
    ]
    ctx = Context(messages=messages, now=SCENARIO_NOW)
    state = build_state(ctx)
    assert list(state) == ["request", "history", "now", "progress", "observations"]
    assert state["history"] == []
    assert state["progress"] == ['Step 1: read_file(path="a.txt") → ok, 2 words']
    assert state["observations"] == [{"step": 1, "tool": "read_file", "status": "ok", "preview": "hello world"}]


def test_loop_mode_without_observations_and_previews() -> None:
    ctx = Context(messages="find it", now=SCENARIO_NOW)
    state = build_state(ctx, "loop")
    assert state["progress"] == [] and state["observations"] == []
    obs = Observation(step=2, tool="search", content={"hits": ["x" * 2000]}, summary="1 hit")
    assert obs.preview_text().endswith("…") and len(obs.preview_text()) == 1200  # §6.2 (review #14)
    assert obs.progress_line() == "Step 2: search() → ok, 1 hit"


def test_time_and_hash(ctx_default: Context) -> None:
    assert ctx_default.timezone_name == "Europe/Zurich"
    assert ctx_default.current_time() == SCENARIO_NOW
    assert Context(messages="x", tz="Asia/Tokyo", now=SCENARIO_NOW).current_time().hour == 21
    other = ctx_default.with_messages("Tell me a joke")
    assert other.request == "Tell me a joke" and other.sha256 != ctx_default.sha256
    assert ctx_default.sha256 == ctx_default.model_copy().sha256
    assert Context(
        sources=[type("S", (), {"name": "contacts", "content_sha256": lambda self: "sha256:x"})()]
    ).source_hashes() == {"contacts": "sha256:x"}


def test_a_fixed_utc_offset_without_tz_is_a_zone() -> None:
    """``now`` from an ISO timestamp with only an offset (no ``tz``): the zone name ``UTC+02:00`` is read as that
    fixed offset (``context.zone_of``) instead of failing in ``ZoneInfo``."""
    from datetime import timedelta

    from jevtools.context import zone_of

    now = datetime.fromisoformat("2026-09-24T14:05:00+02:00")
    ctx = Context(messages="Book it tomorrow at 3pm", now=now)
    assert ctx.timezone_name == "UTC+02:00" and ctx.current_time() == now
    assert build_state(ctx)["now"] == "Thursday 2026-09-24 14:05 UTC+02:00 (UTC+02:00)"
    assert zone_of("UTC-05:30").utcoffset(None) == -timedelta(hours=5, minutes=30)
    assert zone_of("Europe/Zurich") == ZoneInfo("Europe/Zurich")
