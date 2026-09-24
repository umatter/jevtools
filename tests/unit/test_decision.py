"""Decision documents and emitted formats, using the R5 decision of spec §13.5."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest
from pydantic import ValidationError

from jevtools.canonical import canonical_json
from jevtools.decision import (
    AlternativeReport,
    Bottleneck,
    Confidence,
    Decision,
    DecisionIds,
    DecisionUsage,
    Pending,
    PendingAction,
    Prompt,
    PromptOption,
    SlotReport,
    ToolCall,
    call_hash,
)
from jevtools.policy import Outcome

TRACE = "tr_9b1f0c3e7a52d4e8"
ARGS = {
    "title": "Sync with Bob and Carol",
    "start": "2026-09-29T15:00:00+02:00",
    "duration_minutes": 45,
    "attendees": ["bob.meier@muster.ch", "carol.liu@muster.ch"],
}


def r5_decision() -> Decision:
    call = ToolCall.build("create_event", ARGS, trace_id=TRACE)
    pending = Pending.new(
        pending_id="pnd_9b1f0c3e7a52d4e8",
        decision_id="dec_9b1f0c3e7a52d4e8",
        ballot_sha256="sha256:b",
        call=call,
        options={"ok": PendingAction(action="confirm"), "cancel": PendingAction(action="cancel")},
    )
    return Decision(
        decision_id="dec_9b1f0c3e7a52d4e8",
        trace_id=TRACE,
        outcome=Outcome.CONFIRM,
        rule="P9.external.confirm_band",
        call=call,
        confidence=Confidence(
            call=0.550308,
            tier="external",
            composition="PI",
            W=0.785714,
            PI=0.550308,
            L=0.455686,
            execute_at=0.8,
            confirm_at=0.5,
        ),
        bottleneck=Bottleneck(slot="attendees", shape="ambiguous"),
        slots={
            "title": SlotReport(value="Sync with Bob and Carol", p=0.94, stakes="cosmetic", channel="user"),
            "start": SlotReport(
                value="2026-09-29T15:00:00+02:00",
                p=0.8,
                stakes="identity",
                channel="user",
                alternatives=[AlternativeReport(value="2026-10-06T15:00:00+02:00", p=0.18)],
            ),
            "duration_minutes": SlotReport(value=45, p=0.96, stakes="identity", channel="user"),
            "attendees": SlotReport(
                value=ARGS["attendees"],
                p=0.785714,
                stakes="identity",
                channel="registry",
                alternatives=[
                    AlternativeReport(part="m0", value="rbrown@partner.io", p=0.09),
                    AlternativeReport(part="m1", value="caroline.weber@muster.ch", p=0.05),
                ],
            ),
        },
        gates={"authorized": 0.96},
        prompt=Prompt(
            kind="confirm",
            text="Create “Sync with Bob and Carol” …?",
            options=[
                PromptOption(id="ok", text="Create"),
                PromptOption(id="alt:start:1", text="Tue 6 Oct 2026, 15:00 instead"),
                PromptOption(id="change", text="Change…"),
                PromptOption(id="cancel", text="Cancel"),
            ],
        ),
        pending=pending,
        rounds=1,
        usage=DecisionUsage(jev_calls=1, jev_input_tokens=1715, llm_calls=0, cost_usd=0.000072),
    )


def test_native_document_key_order_and_rounding() -> None:
    doc = r5_decision().to_doc()
    assert list(doc) == [
        "spec",
        "decision_id",
        "trace_id",
        "outcome",
        "rule",
        "call",
        "tool_calls",
        "confidence",
        "bottleneck",
        "slots",
        "gates",
        "flags",
        "prompt",
        "pending_id",
        "rounds",
        "usage",
    ]
    assert doc["call"] == {"name": "create_event", "arguments": ARGS}
    assert doc["confidence"] == {
        "call": 0.5503,
        "tier": "external",
        "composition": "PI",
        "W": 0.7857,
        "PI": 0.5503,
        "L": 0.4557,
        "J": None,
        "calibrated": False,
        "execute_at": 0.8,
        "confirm_at": 0.5,
    }
    assert doc["slots"]["attendees"]["alternatives"][0] == {"part": "m0", "value": "rbrown@partner.io", "p": 0.09}
    assert doc["slots"]["start"]["alternatives"] == [{"value": "2026-10-06T15:00:00+02:00", "p": 0.18}]
    assert doc["pending_id"] == "pnd_9b1f0c3e7a52d4e8"
    raw = r5_decision().to_json()
    assert raw == canonical_json(doc) and b'"cost_usd":0.000072' in raw and b'"p":0.7857' in raw


def test_ids_are_deterministic_and_shared() -> None:
    call = ToolCall.build("create_event", ARGS, trace_id=TRACE)
    digest = call_hash(TRACE, "create_event", ARGS)
    assert call.id == "call_jev_" + digest and call.idempotency_key == "idem_" + digest and len(digest) == 16
    assert ToolCall.build("create_event", dict(reversed(ARGS.items())), trace_id=TRACE).id != call.id
    assert call == ToolCall.build("create_event", json.loads(json.dumps(ARGS)), trace_id=TRACE)
    ids = DecisionIds.derive("sha256:ballot", "sha256:response")
    assert ids.decision_id[4:] == ids.trace_id[3:] == ids.pending_id[4:]
    assert ids == DecisionIds.derive("sha256:ballot", "sha256:response")


def test_execute_message_formats() -> None:
    call = ToolCall.build("get_weather", {"city": "Zurich", "unit": "fahrenheit"}, trace_id=TRACE)
    decision = Decision(
        decision_id="dec_1",
        trace_id=TRACE,
        outcome=Outcome.EXECUTE,
        rule="P9.read.execute",
        call=call,
        tool_calls=[call],
        confidence=Confidence(call=0.97, tier="read", composition="W", W=0.97, PI=0.92, L=0.92, execute_at=0.6),
    )
    message = decision.to_openai_message()
    assert message == {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": call.id,
                "type": "function",
                "function": {"name": "get_weather", "arguments": '{"city":"Zurich","unit":"fahrenheit"}'},
            }
        ],
        "x_jev": {"outcome": "execute", "confidence": 0.97, "trace_id": TRACE, "idempotency_key": call.idempotency_key},
    }
    assert decision.finish_reason == "tool_calls"
    assert decision.to_anthropic_content() == [
        {
            "type": "tool_use",
            "id": "toolu_jev_" + call.id[len("call_jev_") :],
            "name": "get_weather",
            "input": {"city": "Zurich", "unit": "fahrenheit"},
        },
    ]
    assert decision.to_doc()["tool_calls"][0]["id"] == call.id


def test_confirm_and_abstain_messages() -> None:
    confirm = r5_decision().to_openai_message()
    assert confirm["content"].startswith("Create") and "tool_calls" not in confirm
    assert confirm["x_jev"]["outcome"] == "confirm" and confirm["x_jev"]["pending_id"] == "pnd_9b1f0c3e7a52d4e8"
    assert [o["id"] for o in confirm["x_jev"]["options"]] == ["ok", "alt:start:1", "change", "cancel"]
    abstain = Decision(decision_id="dec_2", trace_id="tr_2", outcome=Outcome.ABSTAIN, rule="P1.tool.no_tool")
    assert abstain.to_openai_message() == {
        "role": "assistant",
        "content": "",
        "x_jev": {"outcome": "abstain", "trace_id": "tr_2"},
    }
    assert abstain.to_anthropic_content() == [] and abstain.finish_reason == "stop"
    joke = abstain.model_copy(update={"content": "Why did the …"})
    assert joke.to_anthropic_content() == [{"type": "text", "text": "Why did the …"}]


def test_only_execute_may_emit_tool_calls() -> None:
    call = ToolCall.build("x", {}, trace_id="tr_1")
    with pytest.raises(ValidationError):
        Decision(decision_id="d", trace_id="t", outcome=Outcome.CONFIRM, rule="r", tool_calls=[call])


def test_pending_expiry() -> None:
    created = datetime(2026, 9, 24, 12, 0, tzinfo=timezone.utc)
    pending = Pending.new(pending_id="p", decision_id="d", ballot_sha256="s", created_at=created)
    assert pending.expires_at == created + timedelta(minutes=15)
    assert not pending.expired(created + timedelta(minutes=14)) and pending.expired(created + timedelta(minutes=15))
    assert PendingAction(action="bind", slot="to", value="anna.rossi@gmail.com").slot == "to"
