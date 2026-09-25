"""The spec §13.3 walk-through end to end: real extractors, sources and resolvers, the ScriptedBackend answering with
the spec's [I] numbers, the Router's policy, prompts, resumes and traces (every trace must pass ``jt.verify``)."""

from __future__ import annotations

import json
from typing import Any

import pytest

from jevtools.policy import Outcome
from tests.scenario import scripts
from tests.scenario.fixtures import account_rows, accounts, contacts, files, scenario_context, scenario_messages
from tests.scenario.support import criteria, decide, ordered_json, qids, verified
from tests.support import load_fixture

R2_QIDS = [
    "tool", "get_weather.city", "get_weather.unit", "search_web.query.accept.0", "search_web.query.accept.1",
    "send_email.authorized", "send_email.to", "send_email.to.present", "send_email.to.verify.0",
    "send_email.to.verify.1", "send_email.to.verify.2", "send_email.subject.accept.0",
    "send_email.subject.accept.1", "send_email.subject.accept.2", "send_email.body.accept.0",
    "send_email.body.accept.1",
]  # fmt: skip
PROBES = ["get_weather.city", "get_weather.unit", "search_web.query.accept.0", "search_web.query.accept.1"]
R2_BODY = "Hi Anna,\n\nI'll be 10 minutes late.\n\nBest,\nSam"


class JokeLLM:
    """A text LLM that answers every abstain with the same joke."""

    joke = "Why did the scarecrow win an award? Because he was outstanding in his field."

    def complete(self, messages: Any) -> str:
        return self.joke

    async def acomplete(self, messages: Any) -> str:
        return self.joke


# --------------------------------------------------------------------------------------------------------------------
# R1 — read tier, execute
# --------------------------------------------------------------------------------------------------------------------


def test_r1_weather_executes() -> None:
    router, backend, d = decide(scripts.R1, scripts.R1_REQUEST)
    assert qids(backend) == ["tool", *PROBES]  # 5 questions; "Fahrenheit" is claimed by `unit`
    city = criteria(backend, "get_weather.city")
    assert list(city) == ["Zurich", "NOT_STATED", "NONE_OF_THESE"] and "the user's home city" in city["NOT_STATED"]
    assert (d.outcome, d.rule) == (Outcome.EXECUTE, "P9.read.execute")
    assert d.call is not None and d.call.arguments == {"city": "Zurich", "unit": "fahrenheit"}
    assert d.confidence is not None and d.confidence.W == pytest.approx(0.97) and d.confidence.composition == "W"
    assert d.slots["city"].p == pytest.approx(0.97)  # "Zurich" .95 pooled with NOT_STATED → home city Zurich .02
    assert d.rounds == 1 and d.tool_calls == [d.call]
    verified(d, router, scripts.R1_REQUEST)


# --------------------------------------------------------------------------------------------------------------------
# R2 — external tier: confirm with history, clarify menu without, a click executes
# --------------------------------------------------------------------------------------------------------------------


def test_r2_request_is_the_spec_request() -> None:
    router, backend, _ = decide(scripts.R2, scripts.R2_REQUEST, history=True)
    assert qids(backend) == R2_QIDS
    assert ordered_json(backend.requests[0].to_wire()) == ordered_json(load_fixture("spec_r2_request.json"))
    compact = json.dumps(backend.requests[0].to_wire(), ensure_ascii=False, separators=(",", ":"))
    assert len(compact) == 7460  # "16 questions, 7,460 characters" (§13.4)
    tools = {t.name: t for t in router.compile(scenario_messages(scripts.R2_REQUEST, history=True)).tools}
    assert not any(tools[name].speculated for name in ("create_event", "transfer_funds", "read_file"))


def test_r2_with_history_confirms_then_ok_executes() -> None:
    router, backend, d = decide(scripts.R2, scripts.R2_REQUEST, history=True)
    assert (d.outcome, d.rule) == (Outcome.CONFIRM, "P9.external.confirm_band") and d.tool_calls == []
    assert d.call is not None and d.call.arguments == {
        "to": "anna.keller@acme.com", "subject": "Running 10 minutes late", "body": R2_BODY}  # fmt: skip
    assert d.confidence is not None
    assert (round(d.confidence.PI, 3), d.confidence.W, round(d.confidence.L, 2)) == (0.714, 0.86, 0.68)
    assert d.bottleneck is not None and (d.bottleneck.slot, d.bottleneck.shape) == ("to", "ambiguous")
    assert d.gates["authorized"] == pytest.approx(0.95)
    assert d.prompt is not None and [o.id for o in d.prompt.options] == ["ok", "change", "cancel"]  # Rossi .07 < .10
    assert [(a.value, a.p) for a in d.slots["to"].alternatives] == [("anna.rossi@gmail.com", 0.07),
                                                                    ("annabel.frey@muster.ch", 0.03)]  # fmt: skip
    verified(d, router, scripts.R2_REQUEST, history=True)

    done = router.resume(d.pending_id or "", selection="ok")
    assert (done.outcome, done.rule) == (Outcome.EXECUTE, "P9.external.confirmed")
    assert len(backend.requests) == 1 and done.rounds == 0  # a click costs no Jev call
    assert done.tool_calls and done.tool_calls[0].arguments == d.call.arguments
    assert done.trace.resumed_from == d.pending_id and done.tool_calls[0].idempotency_key.startswith("idem_")
    verified(done, router, scripts.R2_REQUEST, history=True)


def test_r2_without_history_clarifies_then_a_click_executes() -> None:
    router, backend, d = decide(scripts.R2_NO_HISTORY, scripts.R2_REQUEST)
    assert qids(backend) == R2_QIDS
    assert (d.outcome, d.rule) == (Outcome.CLARIFY, "P9.external.ambiguous")
    assert d.confidence is not None and round(d.confidence.PI, 2) == 0.39
    assert d.bottleneck is not None and (d.bottleneck.slot, d.bottleneck.shape) == ("to", "ambiguous")
    assert d.prompt is not None and d.prompt.kind == "menu"
    assert d.prompt.text == "Which recipient did you mean?"
    assert [o.id for o in d.prompt.options] == ["pick:to:0", "pick:to:1", "pick:to:2", "other"]
    for option, label in zip(d.prompt.options, (scripts.KELLER, scripts.ROSSI, scripts.FREY), strict=False):
        assert label in option.text and option.text.startswith("Send an email")  # complete calls
    assert "Hi Annabel," in d.prompt.options[2].text  # each option re-binds the late-bound greeting
    assert d.prompt.options[-1].text == "Something else"
    verified(d, router, scripts.R2_REQUEST)

    done = router.resume(d.pending_id or "", selection="pick:to:0")
    assert (done.outcome, done.rule) == (Outcome.EXECUTE, "P9.external.confirmed") and len(backend.requests) == 1
    assert done.confidence is not None and round(done.confidence.PI, 2) == 0.83  # .96 · .95 · 1 · .91
    assert done.call is not None and done.call.arguments["to"] == "anna.keller@acme.com"
    assert done.slots["to"].channel == "user" and done.slots["to"].p == 1.0
    verified(done, router, scripts.R2_REQUEST)


def test_r2_click_on_another_recipient_rebinds_the_greeting() -> None:
    router, _, d = decide(scripts.R2_NO_HISTORY, scripts.R2_REQUEST)
    done = router.resume(d.pending_id or "", selection="pick:to:2")
    assert done.outcome is Outcome.EXECUTE and done.call is not None
    arguments = done.call.arguments
    assert arguments["to"] == "annabel.frey@muster.ch" and arguments["body"].startswith("Hi Annabel,")


# --------------------------------------------------------------------------------------------------------------------
# R3 — critical tier: confirm (never auto), TOCTOU before execution
# --------------------------------------------------------------------------------------------------------------------

R3_CALL = {"from_account": "acc_7731", "to_account": "acc_2210", "amount": "250.00", "currency": "CHF"}


def test_r3_transfer_confirms() -> None:
    router, backend, d = decide(scripts.R3, scripts.R3_REQUEST)
    assert qids(backend) == [
        "tool",
        *PROBES,
        "transfer_funds.authorized",
        "transfer_funds.joint",
        "transfer_funds.from_account",
        "transfer_funds.from_account.present",
        "transfer_funds.from_account.rev",
        "transfer_funds.from_account.verify.0",
        "transfer_funds.from_account.verify.1",
        "transfer_funds.from_account.verify.2",
        "transfer_funds.to_account",
        "transfer_funds.to_account.present",
        "transfer_funds.to_account.rev",
        "transfer_funds.to_account.verify.0",
        "transfer_funds.to_account.verify.1",
        "transfer_funds.to_account.verify.2",
        "transfer_funds.amount",
        "transfer_funds.currency",
    ]  # fmt: skip  (21 questions: 3 verify Nouls per account slot, §3.8.3)
    joint = criteria(backend, "transfer_funds.joint")  # 6 ordered pairs over the anchored accounts + NONE
    assert len(joint) == 6 + 1 and "250.00 CHF: Savings → Checking" in joint
    assert list(criteria(backend, "transfer_funds.currency"))[:2] == ["CHF", "EUR"]
    assert (d.outcome, d.rule) == (Outcome.CONFIRM, "P9.critical.confirm_band") and d.tool_calls == []
    assert d.call is not None and d.call.arguments == R3_CALL
    c = d.confidence
    assert c is not None and (c.composition, c.execute_at) == ("MIN_L_J", None)
    assert (round(c.L, 2), c.J, round(c.call, 2)) == (0.84, 0.92, 0.84)
    assert d.slots["currency"].p == pytest.approx(0.97)  # CHF .95 + NOT_STATED .02 (→ CHF via Savings)
    assert d.prompt is not None
    assert d.prompt.text == "Transfer 250.00 CHF from Savings · CHF · CH93…2957 to Checking · CHF · CH56…1180?"
    assert [(o.id, o.text) for o in d.prompt.options] == [
        ("ok", "Confirm"), ("alt:from_account:1", "Travel savings · EUR · CH08…4410 instead"),
        ("change", "Change…"), ("cancel", "Cancel"),
    ]  # fmt: skip
    verified(d, router, scripts.R3_REQUEST)

    done = router.resume(d.pending_id or "", selection="ok")
    assert (done.outcome, done.rule) == (Outcome.EXECUTE, "P9.critical.confirmed") and done.rounds == 0
    assert done.tool_calls[0].arguments == R3_CALL and done.trace.idempotency_key == done.tool_calls[0].idempotency_key
    verified(done, router, scripts.R3_REQUEST)


def test_r3_toctou_balance_drop_blocks_execution() -> None:
    router, backend, d = decide(scripts.R3, scripts.R3_REQUEST)
    drained = scenario_context(sources=[contacts(), accounts(account_rows(acc_7731=100.0)), files()])
    after = router.resume(d.pending_id or "", selection="ok", context=drained)
    assert after.outcome is not Outcome.EXECUTE and after.tool_calls == []
    assert "TOCTOU: constraint 'amount <= from_account.balance' no longer holds" in after.trace.notes
    assert len(backend.requests) == 2 and after.rounds == 1  # re-planned in a new round
    assert "250.00 CHF: Savings → Checking" not in criteria(backend, "transfer_funds.joint", 1)  # balance too low
    verified(after, router, scripts.R3_REQUEST, context=drained)


# --------------------------------------------------------------------------------------------------------------------
# R4 — read tier over 3,000 files; a miss widens (buckets + group, then hierarchy)
# --------------------------------------------------------------------------------------------------------------------


def test_r4_opens_the_payments_config() -> None:
    router, backend, d = decide(scripts.R4, scripts.R4_REQUEST)
    assert qids(backend) == ["tool", "get_weather.city", "get_weather.unit", "read_file.path",
                             "search_web.query.accept.0", "search_web.query.accept.1"]  # fmt: skip
    path = criteria(backend, "read_file.path")
    assert len(path) == 40 + 2 and scripts.APP_YAML in path and scripts.PROD_YAML in path
    assert (d.outcome, d.rule) == (Outcome.EXECUTE, "P9.read.execute")
    assert d.call is not None and d.call.arguments == {"path": scripts.APP_YAML}
    assert d.confidence is not None and d.confidence.W == pytest.approx(0.71)
    assert d.slots["path"].alternatives[0].value == scripts.PROD_YAML and d.slots["path"].alternatives[0].p == 0.16
    assert d.to_openai_message()["tool_calls"][0]["function"]["arguments"] == '{"path":"' + scripts.APP_YAML + '"}'
    verified(d, router, scripts.R4_REQUEST)


def test_r4_miss_widens_then_asks() -> None:
    router, backend, d = decide(scripts.R4_MISS, scripts.R4_REQUEST)
    assert len(backend.requests) == 3 and d.rounds == 3
    assert qids(backend, 1) == ["read_file.path.bucket.0", "read_file.path.bucket.1", "read_file.path.group"]
    buckets = [criteria(backend, f"read_file.path.bucket.{b}", 1) for b in (0, 1)]
    assert [len(b) for b in buckets] == [251, 251]  # results 41–540 in two buckets, each + NONE_OF_THESE
    assert not {label for label in buckets[0] if "/" in label} & set(criteria(backend, "read_file.path"))
    assert qids(backend, 2) == ["read_file.path"]  # hierarchy round: the items of docs/runbooks
    items = [label for label in criteria(backend, "read_file.path", 2) if "/" in label]
    assert items and all(label.startswith("docs/runbooks/") for label in items)
    assert (d.outcome, d.rule) == (Outcome.CLARIFY, "P7.slot.shape")
    assert d.bottleneck is not None and (d.bottleneck.slot, d.bottleneck.shape) == ("path", "out_of_pool")
    assert d.prompt is not None and (d.prompt.kind, d.prompt.text) == ("open", "What should the workspace-relative "
                                                                               "file path be?")  # fmt: skip
    verified(d, router, scripts.R4_REQUEST)


def test_r4_miss_found_in_a_bucket_executes() -> None:
    router, backend, d = decide(scripts.r4_found_in_bucket, scripts.R4_REQUEST)
    first = next(iter(criteria(backend, "read_file.path.bucket.0", 1)))
    assert len(backend.requests) == 2 and (d.outcome, d.rule) == (Outcome.EXECUTE, "P9.read.execute")
    assert d.call is not None and d.call.arguments == {"path": first}
    # D_b(v*) · D_b'(⊥uncovered): bucket 1 puts all of its mass on NONE_OF_THESE, so f = .9 · 1
    assert d.confidence is not None and d.confidence.W == pytest.approx(0.9) and d.slots["path"].p == pytest.approx(0.9)
    verified(d, router, scripts.R4_REQUEST)


# --------------------------------------------------------------------------------------------------------------------
# R5 — external tier: confirm with the Tue 6 Oct alternative; the click executes
# --------------------------------------------------------------------------------------------------------------------

R5_CALL = {"title": "Sync with Bob and Carol", "start": "2026-09-29T15:00:00+02:00", "duration_minutes": 45,
           "attendees": ["bob.meier@muster.ch", "carol.liu@muster.ch"]}  # fmt: skip


def test_r5_request_is_the_spec_request() -> None:
    _, backend, _ = decide(scripts.R5, scripts.R5_REQUEST)
    assert ordered_json(backend.requests[0].to_wire()) == ordered_json(load_fixture("spec_r5_request.json"))
    assert len(backend.requests[0].questions) == 14


def test_r5_event_confirms_with_the_start_alternative() -> None:
    router, backend, d = decide(scripts.R5, scripts.R5_REQUEST)
    assert (d.outcome, d.rule) == (Outcome.CONFIRM, "P9.external.confirm_band")
    assert d.call is not None and d.call.arguments == R5_CALL
    doc = json.loads(d.to_json())
    assert doc["confidence"] == {"call": 0.5503, "tier": "external", "composition": "PI", "W": 0.7857, "PI": 0.5503,
                                 "L": 0.4557, "J": None, "calibrated": False, "execute_at": 0.8,
                                 "confirm_at": 0.5}  # fmt: skip
    assert doc["bottleneck"] == {"slot": "attendees", "shape": "ambiguous"} and doc["gates"] == {"authorized": 0.96}
    assert doc["slots"] == {
        "title": {"value": "Sync with Bob and Carol", "p": 0.94, "stakes": "cosmetic", "channel": "user",
                  "alternatives": []},
        "start": {"value": "2026-09-29T15:00:00+02:00", "p": 0.8, "stakes": "identity", "channel": "user",
                  "alternatives": [{"value": "2026-10-06T15:00:00+02:00", "p": 0.18}]},
        "duration_minutes": {"value": 45, "p": 0.96, "stakes": "identity", "channel": "user", "alternatives": []},
        "attendees": {"value": ["bob.meier@muster.ch", "carol.liu@muster.ch"], "p": 0.7857, "stakes": "identity",
                      "channel": "registry", "alternatives": [
                          {"part": "m0", "value": "rbrown@partner.io", "p": 0.09},
                          {"part": "m1", "value": "caroline.weber@muster.ch", "p": 0.05}]},
    }  # fmt: skip
    assert doc["flags"] == [] and doc["rounds"] == 1 and doc["tool_calls"] == []
    assert [(o["id"], o["text"]) for o in doc["prompt"]["options"]] == [
        ("ok", "Create"), ("alt:start:1", f"{scripts.TUE_06} instead"), ("change", "Change…"), ("cancel", "Cancel"),
    ]  # fmt: skip
    verified(d, router, scripts.R5_REQUEST)

    alt = router.resume(d.pending_id or "", selection="alt:start:1")
    assert (alt.outcome, alt.rule) == (Outcome.EXECUTE, "P9.external.confirmed") and len(backend.requests) == 1
    assert alt.call is not None and alt.call.arguments == {**R5_CALL, "start": "2026-10-06T15:00:00+02:00"}
    assert alt.confidence is not None and round(alt.confidence.PI, 3) == 0.688  # .95 · .96 · 1 · .96 · .7857
    verified(alt, router, scripts.R5_REQUEST)


# --------------------------------------------------------------------------------------------------------------------
# R7 — no tool: abstain (empty stop, or the text LLM's answer)
# --------------------------------------------------------------------------------------------------------------------


def test_r7_joke_abstains() -> None:
    router, backend, d = decide(scripts.R7, scripts.R7_REQUEST)
    assert qids(backend) == ["tool", *PROBES]
    accepts = [scripts.accept_candidate(backend.requests[0], qid) for qid in qids(backend) if ".accept." in qid]
    assert accepts == ["a joke", "Tell me a joke"]
    assert (d.outcome, d.rule) == (Outcome.ABSTAIN, "P1.tool.no_tool") and d.call is None and d.tool_calls == []
    message = d.to_openai_message()
    assert message["content"] == "" and message["x_jev"]["outcome"] == "abstain" and "tool_calls" not in message
    verified(d, router, scripts.R7_REQUEST)


def test_r7_joke_goes_to_the_text_llm() -> None:
    _, _, d = decide(scripts.R7, scripts.R7_REQUEST, text_llm=JokeLLM())
    assert d.outcome is Outcome.ABSTAIN and d.content == JokeLLM.joke
    assert (d.usage.jev_calls, d.usage.llm_calls) == (1, 1)
