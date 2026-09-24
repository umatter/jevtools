"""Regression tests for the engine review findings (malformed answers, 422 isolation, secrets, budgets, previews)."""

from __future__ import annotations

import json
import math
from collections.abc import Callable
from datetime import datetime, timezone
from typing import Any

import httpx
import pytest

from jevtools.backends.http import HTTPBackend
from jevtools.backends.scripted import ScriptedBackend
from jevtools.context import Context
from jevtools.policy import Outcome, PolicyInput, SlotState, Tier, evaluate
from jevtools.router import Router
from jevtools.validate import Limits
from jevtools.wire import DecisionRequest, ScoreAnswer, ScoreQuestion

NOW = datetime(2026, 9, 24, tzinfo=timezone.utc)
PRIORITY = {"type": "function", "function": {"name": "set_priority",
            "description": "Set the priority of the current task.", "parameters": {"type": "object",
            "required": ["priority"],
            "properties": {"priority": {"type": "integer", "minimum": 1, "maximum": 5}}}}}  # fmt: skip


# --------------------------------------------------------------------------------------------------------------------
# #10 Score answers go through the answer-shape guards
# --------------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "answer",
    [
        ScoreAnswer(score=3.0, confidence=0.8, probabilities={3: 1.3}),
        ScoreAnswer(score=3.0, confidence=0.8, probabilities={3: -0.2}),
        ScoreAnswer(score=99.0, confidence=0.8, probabilities={3: 0.9}),
        ScoreAnswer(score=-1.0, confidence=0.8, probabilities={3: 0.9}),
    ],
)
def test_malformed_score_answer_fails_closed(answer: ScoreAnswer) -> None:
    def script(req: Any) -> dict[str, Any]:
        return {qid: answer for qid, q in req.questions.items() if isinstance(q, ScoreQuestion)}

    router = Router([PRIORITY], backend=ScriptedBackend(script), context=Context(now=NOW))
    d = router.decide("Set the priority to 4")
    assert (d.outcome, d.rule) == (Outcome.ABSTAIN, "P0.backend.fail_closed")


def _http_router(tools: list[dict[str, Any]], handler: Callable[[httpx.Request], httpx.Response], **kw: Any) -> Router:
    transport = httpx.MockTransport(handler)
    backend = HTTPBackend.openrouter_decisions("key", transport=transport, async_transport=transport, max_retries=0,
                                               sleep=lambda _: None)  # fmt: skip
    return Router(tools, backend=backend, context=Context(now=NOW), **kw)


def _score_body(request: httpx.Request, score: Any, p: Any) -> dict[str, Any]:
    answers: dict[str, Any] = {}
    for qid, q in json.loads(request.content)["questions"].items():
        if q["type"] == "score":
            answers[qid] = {"type": "score", "score": score, "confidence": 0.8, "probabilities": {"3": p}}
        elif q["type"] == "noul":
            answers[qid] = {"type": "noul", "noul": 0.9}
        else:
            labels = list(q["criteria"])
            answers[qid] = {"type": "choice", "choice": labels[0], "confidence": 0.9,
                            "probabilities": {k: (0.9 if k == labels[0] else 0.0) for k in labels}}  # fmt: skip
    return {"model": "jev-1.13.0", "answers": answers, "usage": {"input_tokens": 1}}


@pytest.mark.parametrize(("score", "p"), [("NaN", 0.9), (3.0, "NaN"), ("Infinity", 0.9)])
def test_non_finite_score_on_the_wire_fails_closed(score: Any, p: Any) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.dumps(_score_body(request, 0, 0.5))
        body = body.replace('"score": 0', f'"score": {score}').replace('"3": 0.5', f'"3": {p}')
        return httpx.Response(200, content=body.encode(), headers={"content-type": "application/json"})

    d = _http_router([PRIORITY], handler).decide("Set the priority to 4")
    assert (d.outcome, d.rule) == (Outcome.ABSTAIN, "P0.backend.fail_closed")


# --------------------------------------------------------------------------------------------------------------------
# #15 Malformed transport payloads are BackendErrors (P0), sync and async
# --------------------------------------------------------------------------------------------------------------------

WEATHER = {"type": "function", "function": {"name": "get_weather", "description": "Get the weather for a city",
           "parameters": {"type": "object", "properties": {"unit": {"type": "string", "enum": ["c", "f"]}},
                          "required": ["unit"]}}}  # fmt: skip
MALFORMED: dict[str, Callable[[httpx.Request], httpx.Response]] = {
    "gzip_garbage": lambda r: httpx.Response(200, headers={"content-encoding": "gzip"}, content=b"garbage"),
    "answers_list": lambda r: httpx.Response(200, json={"answers": [{"type": "choice"}]}),
    "answers_str": lambda r: httpx.Response(200, json={"answers": "oops"}),
    "answers_empty_list": lambda r: httpx.Response(200, json={"answers": []}),
}


@pytest.mark.parametrize("name", sorted(MALFORMED))
def test_malformed_transport_payloads_fail_closed(name: str) -> None:
    d = _http_router([WEATHER], MALFORMED[name]).decide("what's the weather in celsius?")
    assert (d.outcome, d.rule) == (Outcome.ABSTAIN, "P0.backend.fail_closed")


@pytest.mark.parametrize("name", sorted(MALFORMED))
async def test_malformed_transport_payloads_fail_closed_async(name: str) -> None:
    d = await _http_router([WEATHER], MALFORMED[name]).adecide("what's the weather in celsius?")
    assert (d.outcome, d.rule) == (Outcome.ABSTAIN, "P0.backend.fail_closed")


# --------------------------------------------------------------------------------------------------------------------
# #4 / #8 / #5 / #16: 422 isolation never removes a gate, never decodes a dropped slot as a default, works opaque
# --------------------------------------------------------------------------------------------------------------------


def _scripted_http(script: Any, reject: Callable[[str], bool], **kw: Any) -> tuple[Router, list[list[str]]]:
    """An HTTPBackend over a MockTransport answering from ``script``; the first request carrying a wire id for which
    ``reject(wire_id)`` holds gets a 422 naming it."""
    scripted = ScriptedBackend(script)
    sent: list[list[str]] = []
    rejected: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        sent.append(list(body["questions"]))
        bad = [w for w in body["questions"] if reject(w)]
        if bad and not rejected:
            rejected.append(bad[0])
            return httpx.Response(422, json={"detail": [{"loc": ["body", "questions", bad[0], "criteria"],
                                                         "msg": "bad", "type": "value_error"}]})  # fmt: skip
        resp = scripted.decide(DecisionRequest.model_validate(body))
        return httpx.Response(200, json={**resp.model_dump(mode="json", exclude_none=True),
                                         "usage": {"input_tokens": 1}})  # fmt: skip

    router = _http_router(kw.pop("tools"), handler, **kw)
    return router, sent


def _thermostat(risk: str) -> dict[str, Any]:
    return {"type": "function", "function": {"name": "set_thermostat_mode", "description": "Set the thermostat mode",
            "x-jev": {"risk": risk}, "parameters": {"type": "object", "required": ["mode"], "properties": {
                "mode": {"type": "string", "enum": ["heat", "cool", "off"], "description": "mode"}}}}}  # fmt: skip


THERMO_SCRIPT = {"tool": {"set_thermostat_mode": 0.99, "NO_TOOL": 0.005, "UNSUPPORTED": 0.005},
                 "set_thermostat_mode.authorized": 0.02, "set_thermostat_mode.mode": {"heat": 0.99}}  # fmt: skip


@pytest.mark.parametrize("risk", ["write", "external", "critical"])
def test_422_on_authorized_fails_closed(risk: str) -> None:
    text = "how would I set the thermostat to heat? don't do it yet"
    baseline = Router([_thermostat(risk)], backend=ScriptedBackend(THERMO_SCRIPT), context=Context(now=NOW))
    assert baseline.decide(text).outcome is Outcome.ABSTAIN
    router, sent = _scripted_http(THERMO_SCRIPT, lambda w: w.endswith(".authorized"), tools=[_thermostat(risk)])
    d = router.decide(text)
    assert (d.outcome, d.rule) == (Outcome.ABSTAIN, "P0.backend.fail_closed") and d.tool_calls == []
    assert len(sent) == 1  # not isolated, not re-sent without the gate


def test_missing_authorized_never_executes_in_the_policy() -> None:
    base = {"tools": {"t": 0.99, "NO_TOOL": 0.01}, "chosen": "t", "C": 0.99,
            "slots": [SlotState(name="mode", factor=0.99, top=[0.99], channel="enum")]}  # fmt: skip
    for tier in (Tier.WRITE, Tier.EXTERNAL):
        result = evaluate(PolicyInput(tier=tier, authorized=None, **base))
        assert (result.outcome, result.rule, result.reason) == (Outcome.ABSTAIN, "P4.safety.not_authorized",
                                                                "authorized_missing")  # fmt: skip
    assert evaluate(PolicyInput(tier=Tier.READ, authorized=None, **base)).outcome is Outcome.EXECUTE
    unspeculated = evaluate(PolicyInput(tier=Tier.WRITE, authorized=None, speculated=False, **base))
    assert unspeculated.rule == "P6.tool.not_speculated"


SET_THERMOSTAT = {"type": "function", "function": {"name": "set_thermostat",
                  "description": "Set the thermostat mode and target temperature", "x-jev": {"risk": "write"},
                  "parameters": {"type": "object", "required": ["mode"], "properties": {
                      "mode": {"type": "string", "enum": ["heat", "cool", "off"], "description": "thermostat mode"},
                      "temperature": {"type": "integer", "minimum": 10, "maximum": 30, "default": 20,
                                      "description": "target temperature in degrees Celsius"},
                      "fan": {"type": "string", "enum": ["auto", "high"], "description": "fan speed"}}}}}  # fmt: skip


def _thermostat_script(req: DecisionRequest) -> dict[str, Any]:
    out: dict[str, Any] = {"tool": {"set_thermostat": 0.99, "NO_TOOL": 0.005, "UNSUPPORTED": 0.005},
                           "set_thermostat.authorized": 0.98}  # fmt: skip
    for qid, q in req.questions.items():
        if isinstance(q, ScoreQuestion) or qid in out or q.type != "choice":
            continue
        labels = list(q.criteria)
        want = {"set_thermostat.mode": "heat", "set_thermostat.fan": "high"}.get(qid)
        want = want or next((label for label in labels if "25" in label), labels[0])
        out[qid] = {label: (0.98 if label == want else 0.02 / (len(labels) - 1)) for label in labels}
    return out


@pytest.mark.parametrize("slot", ["temperature", "fan"])
def test_422_dropped_slot_is_a_failed_answer_not_a_default(slot: str) -> None:
    text = "set the thermostat to heat at 25 degrees with the fan on high"
    ok = Router([SET_THERMOSTAT], backend=ScriptedBackend(_thermostat_script), context=Context(now=NOW)).decide(text)
    assert ok.outcome is Outcome.EXECUTE and ok.tool_calls[0].arguments == {"mode": "heat", "temperature": 25,
                                                                            "fan": "high"}  # fmt: skip
    router, _ = _scripted_http(_thermostat_script, lambda w: w.startswith(f"set_thermostat.{slot}"),
                               tools=[SET_THERMOSTAT])  # fmt: skip
    d = router.decide(text)
    assert d.outcome is Outcome.CLARIFY and d.tool_calls == []
    assert (d.rule, d.bottleneck.slot if d.bottleneck else None) == ("P7.slot.shape", slot)
    assert "invalid" in d.flags


@pytest.mark.parametrize("id_mode", ["dotted", "opaque"])
def test_422_isolation_in_a_split_round_with_opaque_ids(id_mode: str) -> None:
    tools = [{"type": "function", "function": {"name": "set_thermostat", "x-jev": {"risk": "write"},
              "description": "Set the thermostat mode and target temperature", "parameters": {"type": "object",
              "required": ["mode", "fan", "temperature"], "properties": {
                  "mode": {"type": "string", "enum": ["heat", "cool", "off"], "description": "thermostat mode"},
                  "fan": {"type": "string", "enum": ["auto", "on"], "description": "fan setting"},
                  "temperature": {"type": "integer", "minimum": 10, "maximum": 30}}}}}]  # fmt: skip
    limits = Limits(id_mode=id_mode, max_questions=3)
    text = "set the thermostat to heat, fan auto, 25 degrees"
    plain = Router(tools, backend=ScriptedBackend({}), context=Context(now=NOW), limits=limits)
    ballot = plain.compile(text)
    assert len(ballot.call_plan()) > 1
    mode_wire = ballot.wire_ids(id_mode)["set_thermostat.mode"]
    router, _ = _scripted_http(_thermostat_script, lambda w: w == mode_wire, tools=tools, limits=limits)
    d = router.decide(text)
    assert (d.outcome, d.rule) == (Outcome.CLARIFY, "P7.slot.shape")
    assert d.bottleneck is not None and d.bottleneck.slot == "mode"
    assert any("422 isolation" in note for note in d.trace.notes)


# --------------------------------------------------------------------------------------------------------------------
# #3 / #6 Secrets never reach Jev's state nor the Filler
# --------------------------------------------------------------------------------------------------------------------

SECRET = "sk-live-SUPERSECRET-123"


def _post_update(message: bool) -> dict[str, Any]:
    props: dict[str, Any] = {"api_key": {"type": "string", "description": "API key",
                                         "x-jev": {"default_from": "user.api_key"}}}  # fmt: skip
    if message:
        props = {"message": {"type": "string", "description": "The update text"}, **props}
    else:
        props = {"channel": {"type": "string", "enum": ["general", "ops"], "description": "Channel"}, **props}
    return {"type": "function", "function": {"name": "post_update", "description": "Post a status update.",
            "parameters": {"type": "object", "required": list(props), "properties": props}}}  # fmt: skip


@pytest.mark.parametrize("field", ["api_key", "vault"])
@pytest.mark.parametrize("shareable", [None, ("name", "api_key", "vault")])
def test_secret_profile_fields_are_never_sent_to_jev(field: str, shareable: tuple[str, ...] | None) -> None:
    tool = _post_update(message=False)
    xjev = {"kind": "secret", "default_from": f"user.{field}.key"}
    tool["function"]["parameters"]["properties"]["api_key"]["x-jev"] = xjev
    sent: list[str] = []

    def script(req: DecisionRequest) -> dict[str, Any]:
        sent.append(json.dumps(req.to_wire()))
        return {"tool": {"post_update": 0.97, "NO_TOOL": 0.02, "UNSUPPORTED": 0.01},
                "post_update.authorized": 0.97, "post_update.channel": {"ops": 0.97}}  # fmt: skip

    ctx = Context(now=NOW, user={"name": "Sam", field: {"key": SECRET}}, shareable=shareable)
    d = Router([tool], backend=ScriptedBackend(script), context=ctx).decide("Post in ops that the deploy is done")
    assert d.outcome is Outcome.EXECUTE and d.tool_calls[0].arguments["api_key"] == SECRET
    assert sent and not any(SECRET in body for body in sent)
    assert all(json.loads(body)["state"]["user"] == {"name": "Sam"} for body in sent)


def test_secret_named_profile_field_is_not_sent_even_without_a_secret_slot() -> None:
    from jevtools.context import build_state
    from jevtools.plan import secret_user_fields
    from jevtools.spec.catalog import Catalog

    ctx = Context(now=NOW, user={"name": "Sam", "password": "hunter2", "Token": "t"})
    fields = secret_user_fields(Catalog.from_openai([WEATHER]), ctx)
    assert fields == {"password", "Token"}
    assert build_state(ctx, secret_fields=fields)["user"] == {"name": "Sam"}


def test_secret_arguments_are_not_sent_to_the_filler() -> None:
    from jevtools.fallback import OpenAICompatibleFiller

    wire: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        wire.append(request.content.decode())
        content = json.dumps({"candidates": [{"message": "Deploy finished."}]})
        return httpx.Response(200, json={"choices": [{"message": {"role": "assistant", "content": content}}]})

    filler = OpenAICompatibleFiller("some/llm", api_key="k", transport=httpx.MockTransport(handler))
    script = {"tool": {"post_update": 0.97, "NO_TOOL": 0.02, "UNSUPPORTED": 0.01}, "post_update.authorized": 0.97,
              "post_update.message.accept.*": 0.1}  # fmt: skip
    ctx = Context(now=NOW, user={"name": "Sam", "api_key": SECRET}, shareable=("name",))
    router = Router([_post_update(message=True)], backend=ScriptedBackend(script), context=ctx, filler=filler)
    router.decide("Post an update that the deploy is done")
    assert wire and not any(SECRET in body for body in wire)


# --------------------------------------------------------------------------------------------------------------------
# #9 Router.pendings is bounded and released; #12 a pending whose tool left the tool list is recompiled
# --------------------------------------------------------------------------------------------------------------------

REPORT = {"type": "function", "function": {"name": "send_report",
          "description": "Send the weekly report to a recipient.", "x-jev": {"risk": "external", "confirm": "always"},
          "parameters": {"type": "object", "required": ["to"],
                         "properties": {"to": {"type": "string", "format": "email"}}}}}  # fmt: skip


def test_router_pendings_are_bounded_and_expired() -> None:
    from datetime import timedelta

    from jevtools.backends.simulator import LexicalSimulator
    from jevtools.router import LIVE_MAX

    router = Router([REPORT], backend=LexicalSimulator())
    first = router.decide("Send the weekly report to bob0@example.com")
    assert first.pending is not None and first.pending_id in router.pendings
    for i in range(1, LIVE_MAX + 20):
        router.decide(f"Send the weekly report to bob{i}@example.com")
    assert len(router.pendings) == LIVE_MAX and len(router._live) == LIVE_MAX
    assert first.pending_id not in router.pendings  # the oldest was evicted
    last = router.decide("Send the weekly report to carol@example.com")
    assert last.pending is not None and last.pending_id in router.pendings
    done = router.resume(last.pending_id or "", selection="ok")
    assert done.outcome is Outcome.EXECUTE and len(router.pendings) == LIVE_MAX
    # an evicted handle is still resumable from the Pending object (recompiled, never guessed)
    again = router.resume(first.pending, selection="ok")
    assert again.outcome is Outcome.CONFIRM and "pending expired or not in memory: recompiled" in again.trace.notes
    # expired handles are dropped on the next insert
    stale = next(iter(router.pendings.values()))
    router.pendings[stale.pending_id] = stale.model_copy(update={"expires_at": stale.created_at - timedelta(1)})
    router.decide("Send the weekly report to dave@example.com")
    assert stale.pending_id not in router.pendings and stale.pending_id not in router._live


NOTE = {"type": "function", "function": {"name": "post_note", "description": "Post a note to the team board.",
        "parameters": {"type": "object", "required": ["text"], "properties": {
            "text": {"type": "string", "description": "The note text"}}}}}  # fmt: skip


def test_open_clarify_resumed_without_its_tool_is_recompiled() -> None:
    script = {"tool": {"post_note": 0.9, "get_weather": 0.05, "NO_TOOL": 0.03, "UNSUPPORTED": 0.02},
              "post_note.text.accept.*": 0.05, "*authorized*": 0.9, "reply": {"OTHER": 0.9, "CANCEL": 0.1}}  # fmt: skip
    first = Router([NOTE, WEATHER], backend=ScriptedBackend(script), context=Context(now=NOW))
    d = first.decide("Post a note to the board")
    assert d.outcome is Outcome.CLARIFY and d.pending is not None and d.pending.state["tool"] == "post_note"
    other = Router([WEATHER], backend=ScriptedBackend(script), context=Context(now=NOW))
    r = other.resume(d.pending, reply="Lunch is at noon today")
    assert r.outcome is not Outcome.EXECUTE
    assert "pending tool 'post_note' is not in the tool list: recompiled" in r.trace.notes


# --------------------------------------------------------------------------------------------------------------------
# #13 The policy's [budget] is applied; #18 one token-estimate formula
# --------------------------------------------------------------------------------------------------------------------


def test_policy_budget_changes_the_split() -> None:
    from jevtools.policy import Policy
    from tests.scenario.fixtures import scenario_router
    from tests.scenario.scripts import R5, R5_REQUEST

    default_router, default_backend = scenario_router(R5, limits=Limits())
    default_router.decide(R5_REQUEST)
    assert len(default_backend.requests) == 1
    policy = Policy.from_toml_text("[budget]\nmax_questions_per_call = 3\n")
    router, backend = scenario_router(R5, policy=policy, limits=Limits())
    assert router.round_limits().max_questions == 3
    router.decide(R5_REQUEST)
    assert len(backend.requests) > 1 and all(len(r.questions) <= 3 for r in backend.requests)
    tight = Policy.from_toml_text("[budget]\nmax_tokens_per_call = 900\nmax_state_tokens = 500\n"
                                  "chars_per_token = 3.0\n")  # fmt: skip
    limits = Limits(max_questions=400).within_budget(tight.budget)
    assert (limits.max_tokens, limits.max_state_tokens, limits.chars_per_token, limits.max_questions) == \
        (900, 500, 3.0, 400)  # fmt: skip
    loose = Policy.from_toml_text("[budget]\nmax_questions_per_call = 1000\n")
    assert Limits(max_questions=8).within_budget(loose.budget).max_questions == 8
    assert Limits(max_questions=400).within_budget(Policy().budget).max_questions == 400  # probed limits kept


def test_planner_and_preflight_share_one_token_estimate() -> None:
    from jevtools.budget import TokenEstimator
    from jevtools.plan import _tokens, compile_round, state_tokens
    from jevtools.spec.catalog import Catalog

    limits = Limits(chars_per_token=3.3, token_ratio=1.375)
    assert _tokens(12468, limits) == limits.tokens_for_chars(12468) == math.ceil(12468 * 1.375 / 3.3)
    body = {"model": "m", "state": "x" * 12440, "questions": {}}
    assert limits.estimate_tokens(body) == limits.tokens_for_chars(len(json.dumps(body, separators=(",", ":"))))
    assert state_tokens("abc", limits) == limits.tokens_for_chars(5)
    estimator = TokenEstimator(3.3, ratios={("b", "m"): 1.375})
    assert estimator.est(body, backend="b", model="m") == limits.estimate_tokens(body)
    tools = [{"type": "function", "function": {"name": "ab", "description": "Set the mode", "parameters": {
        "type": "object", "x-jev": {"risk": "read"}, "properties": {"cd": {"type": "string", "enum": ["fast", "slow"]}},
        "required": ["cd"]}}}]  # fmt: skip
    ctx = Context(now=NOW).with_messages("set the mode to fast")
    edge = Limits(id_mode="opaque", chars_per_token=3.3, token_ratio=0.7025109170305677, max_tokens=195)
    plan = compile_round(Catalog.from_openai(tools), ctx, limits=edge)  # used to raise call.tokens (~196 tokens)
    assert all(edge.estimate_tokens(r) <= edge.max_tokens for r in plan.ballot.to_requests("", id_mode="opaque"))


# --------------------------------------------------------------------------------------------------------------------
# #14 Drop-in tool messages get the §6.2 BM25 preview, like ingested results
# --------------------------------------------------------------------------------------------------------------------


def test_drop_in_tool_message_preview_matches_ingest_observation() -> None:
    from jevtools import context as context_module
    from jevtools import loop
    from jevtools.context import build_state

    boiler = "\n".join(f"Line {i:02d}: Standard notice, see the terms and conditions." for i in range(60))
    content = boiler + "\nThe ACME invoice INV-2291 total is CHF 4,820.00.\nEnd of document."
    request = "What is the total of the ACME invoice INV-2291?"
    msgs = [
        {"role": "user", "content": request},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {"id": "c1", "type": "function", "function": {"name": "read_file", "arguments": '{"path": "inv.txt"}'}}
            ],
        },  # fmt: skip
        {"role": "tool", "tool_call_id": "c1", "name": "read_file", "content": content},
    ]
    drop_in = build_state(Context(messages=msgs, now=NOW, tz="UTC"), "loop")["observations"][0]["preview"]
    obs = loop.ingest_observation(content, "read_file", 1, arguments={"path": "inv.txt"}, request=request)
    ingested = build_state(Context(messages=[{"role": "user", "content": request}], observations=[obs], now=NOW,
                                   tz="UTC"), "loop")["observations"][0]["preview"]  # fmt: skip
    assert "INV-2291" in drop_in and drop_in == ingested
    assert context_module.PREVIEW_CHARS == loop.PREVIEW_CHARS == 1_200
    short = [*msgs[:2], {**msgs[2], "content": "line one\nline two"}]
    assert build_state(Context(messages=short, now=NOW), "loop")["observations"][0]["preview"] == "line one\nline two"
