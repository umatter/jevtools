"""Template wording is normative: check it against the verbatim R2/R5 requests of spec §13.4–13.5."""

from __future__ import annotations

from typing import Any

import pytest

from jevtools import templates as t
from tests.support import load_fixture

R2: dict[str, Any] = load_fixture("spec_r2_request.json")["questions"]
R5: dict[str, Any] = load_fixture("spec_r5_request.json")["questions"]
SEND = "send an email from the user to one recipient"
EVENT = "create a calendar event and invite attendees"
WEATHER = "get the current weather for a city"


def test_tool_question() -> None:
    assert t.tool_instructions() == R2["tool"]["instructions"]
    assert t.tool_instructions(loop=True) == R2["tool"]["instructions"] + " Steps already taken are in `progress`."
    assert R2["tool"]["criteria"]["NO_TOOL"] == t.TOOL_SENTINEL_TEXT["NO_TOOL"]
    assert R2["tool"]["criteria"]["UNSUPPORTED"] == t.TOOL_SENTINEL_TEXT["UNSUPPORTED"]
    assert (
        t.tool_option_text("Get the current weather for a city. Uses a paid API.")
        == "Get the current weather for a city."
    )


def test_slot_probe_present_and_auth() -> None:
    unit = R2["get_weather.unit"]
    assert t.slot_instructions(WEATHER, t.slot_ask("the temperature unit")) == unit["instructions"]
    assert t.not_stated_text("celsius") == unit["criteria"]["NOT_STATED"]
    assert t.NONE_OF_THESE_TEXT == unit["criteria"]["NONE_OF_THESE"]
    to = R2["send_email.to"]
    assert t.not_stated_text() == to["criteria"]["NOT_STATED"]
    city = R2["get_weather.city"]
    assert t.probe_instructions(WEATHER, "the city name") == city["instructions"]
    gloss = t.default_display("Zurich", gloss=t.context_gloss("user.home_city"))
    assert t.probe_not_stated_text(gloss) == city["criteria"]["NOT_STATED"]
    assert t.PROBE_NONE_OF_THESE_TEXT == city["criteria"]["NONE_OF_THESE"]
    present = R2["send_email.to.present"]
    assert t.present_instructions(SEND, "the recipient's email address") == present["instructions"]
    assert t.PRESENT_CRITERIA == present["criteria"]
    auth = R2["send_email.authorized"]
    assert t.auth_instructions(SEND) == auth["instructions"]
    assert t.AUTH_CRITERIA == auth["criteria"]


def test_accept_nouls() -> None:
    body = R2["send_email.body.accept.1"]
    assert (
        t.accept_instructions(SEND, "the body text of the email", "I'll be 10 minutes late.", content=True)
        == body["instructions"]
    )
    assert t.ACCEPT_CONTENT_CRITERIA == body["criteria"]
    subject = R2["send_email.subject.accept.1"]
    assert t.accept_instructions(SEND, "the subject line", "Running late", content=False) == subject["instructions"]
    assert "criteria" not in subject


def test_temporal_mention_and_more() -> None:
    start = R5["create_event.start"]
    assert (
        t.slot_instructions(EVENT, t.slot_ask("the start time of the event", kind="temporal")) == start["instructions"]
    )
    m0 = R5["create_event.attendees.m0"]
    assert t.mention_instructions(EVENT, "Bob") == m0["instructions"]
    assert t.mention_exclude_text("Bob", "the invitees") == m0["criteria"]["EXCLUDE"]
    assert t.mention_none_text("Bob") == m0["criteria"]["NONE_OF_THESE"]
    assert t.mention_none_text("ACME", person=False) == '"ACME" is something not listed.'
    more = R5["create_event.attendees.more"]
    assert t.more_instructions(EVENT, ["Bob", "Carol"], "the invitees") == more["instructions"]
    assert t.quote_list(["A", "B", "C"]) == '"A", "B" and "C"'
    duration = R5["create_event.duration_minutes"]
    assert t.not_stated_text("30") == duration["criteria"]["NOT_STATED"]


def test_other_templates_render() -> None:
    assert t.done_after_instructions("read a file").startswith("Suppose the assistant now does this successfully: read")
    assert t.member_instructions("open a file", "latest", "a.pdf — 2026-09-15")["question"].endswith(
        "Ignore the word 'latest': the app picks the latest one among the matching items."
    )
    assert t.joint_instructions("move money").endswith("Which option is exactly what the user asks for?")
    assert t.branch_instructions("pay", "the payment method").endswith("Which option describes the payment method?")
    assert t.item_instructions("add", "Bob", "the members").endswith("Should Bob be included in the members?")
    assert t.reply_instructions() == t.T_REPLY
    assert t.default_ask("the reminder flag", kind="flag") == "Does the user want the reminder flag to be true?"
    with pytest.raises(KeyError):
        t.render(t.PREMISE)


def test_text_helpers() -> None:
    assert t.humanize("sendEmail") == "send email"
    assert t.humanize("get-weather_now") == "get weather now"
    assert t.first_sentence("One. Two.") == "One."
    assert t.first_sentence("x" * 300).endswith("…") and len(t.first_sentence("x" * 300)) == 200
    assert t.lower_first("Get it") == "get it" and t.upper_first("send it") == "Send it"
    assert t.context_gloss("locale") is None
