from __future__ import annotations

import pytest

from jevtools.qid import (
    is_valid_qid,
    make_qid,
    opaque_ids,
    sanitize_segment,
    sanitize_tool_name,
    tool_ids,
)


def test_tool_name_sanitizing() -> None:
    assert sanitize_tool_name("get_weather") == "get_weather"
    assert sanitize_tool_name("Get-Weather.v2") == "get_weather_v2"
    assert sanitize_tool_name("3d_print") == "t_3d_print"
    assert sanitize_tool_name("_private") == "t__private"
    assert tool_ids(["send-email", "send_email", "SEND EMAIL"]) == ["send_email", "send_email_2", "send_email_3"]


def test_segment_sanitizing_reserves_suffix_words() -> None:
    assert sanitize_segment("fromAccount") == "fromaccount"
    assert [sanitize_segment(w) for w in ("date", "time", "present", "authorized", "m0", "7")] == [
        "date_", "time_", "present_", "authorized_", "m0_", "7_",
    ]  # fmt: skip
    assert sanitize_segment("m") == "m" and sanitize_segment("members") == "members"


@pytest.mark.parametrize(
    "qid",
    ["tool", "reply", "segmentation", "send_email.to", "send_email.to.present", "send_email.body.accept.0",
     "create_event.attendees.m1", "transfer_funds.joint", "s1.get_weather.city", "x." + "a" * 126],
)  # fmt: skip
def test_valid_qids(qid: str) -> None:
    assert is_valid_qid(qid)


@pytest.mark.parametrize("qid", ["", "Tool", "send_email", "1x.y", "send_email..to", "send-email.to", "a.b@c",
                                 "x." + "a" * 127, "send_email.to."])  # fmt: skip
def test_invalid_qids(qid: str) -> None:
    assert not is_valid_qid(qid)


def test_make_qid_and_opaque_mode() -> None:
    assert make_qid("send_email", "body", "accept", 2) == "send_email.body.accept.2"
    assert make_qid("get_weather", "city", seg=0) == "s0.get_weather.city"
    with pytest.raises(ValueError):
        make_qid("Bad", "x")
    assert opaque_ids(["tool", "a.b", "a.c"]) == {"tool": "q0001", "a.b": "q0002", "a.c": "q0003"}
    assert list(opaque_ids([f"t.{i}" for i in range(10_000)]).values())[-1] == "q10000"
