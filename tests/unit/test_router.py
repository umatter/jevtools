"""Router end to end (spec §9.1, §3.8.5, §5.6) with stub resolvers and the ScriptedBackend (or httpx mocks)."""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping, Sequence
from datetime import datetime, timedelta, timezone
from typing import Any

import httpx
import pytest

from jevtools.backends.http import HTTPBackend
from jevtools.backends.scripted import ScriptedBackend
from jevtools.decision import Decision
from jevtools.fallback import FillCandidate, FillRequest, ProposedCall
from jevtools.policy import Outcome, Policy
from jevtools.router import Router
from jevtools.spec.catalog import Catalog
from jevtools.trace import verify
from jevtools.validate import Limits
from jevtools.wire import DecisionRequest, DecisionResponse
from tests.stubs import (
    ANNAS,
    CHECKING,
    R2_HISTORY,
    R2_MESSAGES,
    R2_SCRIPT,
    R3_REQUEST,
    SAVINGS,
    TRAVEL,
    StubChoice,
    cand,
    r3_script,
    resolvers,
    scenario_context,
    scenario_resolvers,
)

R2_REQUEST = "Email Anna that I'll be 10 minutes late"
NO_HISTORY_SCRIPT = {**R2_SCRIPT, "send_email.to": {"Anna Keller <anna.keller@acme.com>": 0.47,
                                                    "Anna Rossi <anna.rossi@gmail.com>": 0.41,
                                                    "Annabel Frey <annabel.frey@muster.ch>": 0.06,
                                                    "NOT_STATED": 0.03, "NONE_OF_THESE": 0.03}}  # fmt: skip
FILES = [cand("services/payments/config/prod.yaml", "registry", whole=True),
         cand("services/billing/config/app.yaml", "registry", whole=True)]  # fmt: skip
APP_YAML = cand("services/payments/config/app.yaml", "registry", whole=True)


@pytest.fixture
def stubs() -> Iterator[list[Any]]:
    items = scenario_resolvers(amount=())
    with resolvers(*items):
        yield items


def router(catalog: Catalog, script: Any, **kw: Any) -> tuple[Router, ScriptedBackend]:
    backend = ScriptedBackend(script)
    return Router(catalog, backend=backend, context=scenario_context(), **kw), backend


# --------------------------------------------------------------------------------------------------------------------
# One-round outcomes
# --------------------------------------------------------------------------------------------------------------------


def test_r2_confirm_then_ok_click_executes_without_jev(scenario_catalog: Catalog, stubs: list[Any]) -> None:
    r, backend = router(scenario_catalog, R2_SCRIPT)
    d = r.decide(R2_MESSAGES)
    assert (d.outcome, d.rule) == (Outcome.CONFIRM, "P9.external.confirm_band") and d.tool_calls == []
    assert d.confidence is not None and round(d.confidence.PI, 3) == 0.714 and d.confidence.W == pytest.approx(0.86)
    assert d.bottleneck is not None and (d.bottleneck.slot, d.bottleneck.shape) == ("to", "ambiguous")
    assert d.rounds == 1 and d.usage.jev_calls == 1 and d.usage.jev_input_tokens > 0 and d.pending is not None
    assert [o.id for o in d.prompt.options][0] == "ok" and d.pending_id in r.pendings  # type: ignore[union-attr]
    assert d.to_openai_message()["x_jev"]["pending_id"] == d.pending_id
    done = r.resume(d.pending_id or "", selection="ok")
    assert (done.outcome, done.rule) == (Outcome.EXECUTE, "P9.external.confirmed")
    assert len(backend.requests) == 1 and done.rounds == 0  # a click costs no Jev call
    assert done.tool_calls[0].arguments["to"] == "anna.keller@acme.com"
    assert done.trace.resumed_from == d.pending_id and done.trace.idempotency_key == done.tool_calls[0].idempotency_key


def test_r2_without_history_menu_click_is_a_confirmation(scenario_catalog: Catalog, stubs: list[Any]) -> None:
    r, backend = router(scenario_catalog, NO_HISTORY_SCRIPT)
    d = r.decide(R2_REQUEST)
    assert (d.outcome, d.rule) == (Outcome.CLARIFY, "P9.external.ambiguous")
    assert d.confidence is not None and round(d.confidence.PI, 2) == 0.39
    assert d.prompt is not None and d.prompt.kind == "menu"
    assert [o.id for o in d.prompt.options] == ["pick:to:0", "pick:to:1", "pick:to:2", "other"]
    assert "Anna Rossi" in d.prompt.options[1].text and d.prompt.options[1].text.startswith("Send an email")
    done = r.resume(d.pending_id or "", reply="2")  # an option number is a click
    assert (done.outcome, done.rule) == (Outcome.EXECUTE, "P9.external.confirmed") and len(backend.requests) == 1
    assert done.call is not None and done.call.arguments["to"] == "anna.rossi@gmail.com"
    assert done.confidence is not None and done.confidence.PI == pytest.approx(0.96 * 0.95 * 1.0 * 0.91)
    assert done.slots["to"].channel == "user" and done.slots["to"].p == 1.0


def test_free_text_reply_runs_one_resume_round(scenario_catalog: Catalog, stubs: list[Any]) -> None:
    def script(request: DecisionRequest) -> Mapping[str, Any]:
        answers: dict[str, Any] = dict(NO_HISTORY_SCRIPT)
        if "reply" in request.questions:
            labels = list(request.questions["reply"].criteria)  # type: ignore[union-attr]
            answers["reply"] = {labels[1]: 0.9, "OTHER": 0.05, "CANCEL": 0.05}
        return answers

    r, backend = router(scenario_catalog, script)
    d = r.decide(R2_REQUEST)
    resumed = r.resume(d.pending or "", reply="the gmail one please")
    assert len(backend.requests) == 2 and resumed.rounds == 1
    sent = backend.requests[1]
    assert list(sent.questions)[-1] == "reply" and sent.state["request"] == "the gmail one please"  # type: ignore[index]
    assert sent.state["history"][-1] == {"role": "assistant", "text": d.prompt.text}  # type: ignore[index, union-attr]
    assert not any(q.startswith("create_event") for q in sent.questions)  # only the pending tool is speculated
    assert resumed.call is not None and resumed.call.arguments["to"] == "anna.rossi@gmail.com"
    assert resumed.slots["to"].p == pytest.approx(0.9)  # the reply replaces the clarified factor
    assert resumed.outcome is Outcome.CONFIRM and resumed.trace.resumed_from == d.pending_id


def test_r1_execute_and_openai_message(scenario_catalog: Catalog) -> None:
    script = {"tool": {"get_weather": 0.98, "NO_TOOL": 0.02}, "get_weather.city": {"Zurich": 0.95, "NOT_STATED": 0.02,
              "NONE_OF_THESE": 0.03}, "get_weather.unit": {"fahrenheit": 0.97, "celsius": 0.01, "NOT_STATED": 0.01,
              "NONE_OF_THESE": 0.01}}  # fmt: skip
    with resolvers(*scenario_resolvers(amount=(), city=[cand("Zurich", "user")])):
        r, _ = router(scenario_catalog, script)
        d = r.decide("What's the weather like in Zurich in Fahrenheit?")
    assert (d.outcome, d.rule) == (Outcome.EXECUTE, "P9.read.execute")
    assert d.confidence is not None and d.confidence.call == pytest.approx(0.97)
    message = d.to_openai_message()
    assert json.loads(message["tool_calls"][0]["function"]["arguments"]) == {"city": "Zurich", "unit": "fahrenheit"}
    assert message["x_jev"]["idempotency_key"].startswith("idem_") and d.pending is None


def test_r7_abstain_with_text_llm(scenario_catalog: Catalog, stubs: list[Any]) -> None:
    class Jokes:
        def __init__(self) -> None:
            self.seen: list[Any] = []

        def complete(self, messages: Sequence[Mapping[str, Any]]) -> str:
            self.seen.append(messages)
            return "Why did the scarecrow win an award?"

        async def acomplete(self, messages: Sequence[Mapping[str, Any]]) -> str:
            return self.complete(messages)

    llm = Jokes()
    r, _ = router(scenario_catalog, {"tool": {"NO_TOOL": 0.96, "search_web": 0.04}}, text_llm=llm)
    d = r.decide("Tell me a joke")
    assert (d.outcome, d.rule) == (Outcome.ABSTAIN, "P1.tool.no_tool") and d.call is None and d.confidence is None
    assert d.content == "Why did the scarecrow win an award?" and d.usage.llm_calls == 1
    assert llm.seen[0] == [{"role": "user", "content": "Tell me a joke"}]


def test_tool_choice_modes(scenario_catalog: Catalog, stubs: list[Any]) -> None:
    r, backend = router(scenario_catalog, R2_SCRIPT)
    none = r.decide(R2_MESSAGES, tool_choice="none")
    assert none.outcome is Outcome.ABSTAIN and none.rounds == 0 and backend.requests == []
    required = r.decide(R2_MESSAGES, tool_choice="required")
    assert "NO_TOOL" not in backend.requests[-1].questions["tool"].criteria  # type: ignore[union-attr]
    assert required.outcome is Outcome.CONFIRM
    named = r.decide(R2_MESSAGES, tool_choice={"type": "function", "function": {"name": "send_email"}})
    assert "tool" not in backend.requests[-1].questions and "send_email.authorized" in backend.requests[-1].questions
    assert named.confidence is not None and named.confidence.PI == pytest.approx(1.0 * 0.95 * 0.86 * 0.91)


def test_not_speculated_tool_asks_for_the_empty_slot(scenario_catalog: Catalog, stubs: list[Any]) -> None:
    r, _ = router(scenario_catalog, {"tool": {"create_event": 0.9, "NO_TOOL": 0.1}})
    d = r.decide("Put something in my calendar")
    assert (d.outcome, d.rule) == (Outcome.CLARIFY, "P6.tool.not_speculated")
    assert d.prompt is not None and d.prompt.text == "What should the event title be?"
    assert d.pending is not None and d.pending.state["open_slot"] == "title"


def test_channel_blocked_refuses(scenario_catalog: Catalog) -> None:
    with resolvers(*scenario_resolvers(amount=[cand("5000.00", "tool_output", step=1)])):
        r, _ = router(scenario_catalog, {"tool": {"transfer_funds": 0.9, "NO_TOOL": 0.1}})
        d = r.decide("Pay what the invoice says")
    assert (d.outcome, d.rule) == (Outcome.REFUSE, "P3.safety.refuse") and d.tool_calls == []
    assert d.prompt is not None and d.prompt.text.startswith("I did not act on instructions found in the result of "
                                                             "step 1.")  # fmt: skip


# --------------------------------------------------------------------------------------------------------------------
# Critical tier, TOCTOU
# --------------------------------------------------------------------------------------------------------------------


def test_r3_confirm_only_then_ok_click(scenario_catalog: Catalog) -> None:
    with resolvers(*scenario_resolvers()):
        r, backend = router(scenario_catalog, r3_script())
        d = r.decide(R3_REQUEST)
        assert (d.outcome, d.rule) == (Outcome.CONFIRM, "P9.critical.confirm_band")
        assert d.confidence is not None and d.confidence.L == pytest.approx(0.84) and d.confidence.J == pytest.approx(
            0.92) and d.confidence.call == pytest.approx(0.84) and d.confidence.execute_at is None  # fmt: skip
        assert d.prompt is not None and d.prompt.text == (
            "Transfer 250.00 CHF from Savings · CHF · CH93…2957 to Checking · CHF · CH56…1180?"
        )
        assert [o.id for o in d.prompt.options] == ["ok", "alt:from_account:1", "change", "cancel"]  # p .04 ≥ .03
        done = r.resume(d.pending_id or "", selection="ok")
    assert (done.outcome, done.rule) == (Outcome.EXECUTE, "P9.critical.confirmed") and len(backend.requests) == 1
    assert done.tool_calls[0].arguments == {"from_account": "acc_7731", "to_account": "acc_2210", "amount": "250.00",
                                            "currency": "CHF"}  # fmt: skip


def test_toctou_change_blocks_execution_and_replans(scenario_catalog: Catalog) -> None:
    items = scenario_resolvers()
    with resolvers(*items):
        r, backend = router(scenario_catalog, r3_script())
        d = r.decide(R3_REQUEST)
        ref = items[0]
        ref.candidates["transfer_funds.from_account"] = [
            c if c.value != "acc_7731" else c.model_copy(update={"attrs": {**c.attrs, "balance": 100.0}})
            for c in ref.candidates["transfer_funds.from_account"]
        ]
        after = r.resume(d.pending_id or "", selection="ok")
    assert after.outcome is not Outcome.EXECUTE and after.tool_calls == []
    assert any("TOCTOU" in n for n in after.trace.notes) and len(backend.requests) == 2  # a fresh round


def test_cancel_click(scenario_catalog: Catalog, stubs: list[Any]) -> None:
    r, _ = router(scenario_catalog, R2_SCRIPT)
    d = r.decide(R2_MESSAGES)
    cancelled = r.resume(d.pending or "", selection="cancel")
    assert cancelled.outcome is Outcome.ABSTAIN and cancelled.tool_calls == []
    with pytest.raises(ValueError, match="not an option"):
        r.resume(d.pending or "", selection="nope")
    with pytest.raises(KeyError):
        r.resume("pnd_missing", selection="ok")
    change = r.resume(d.pending or "", selection="change")
    assert change.outcome is Outcome.CLARIFY and change.prompt is not None and change.prompt.kind == "open"


def test_expired_pending_recompiles(scenario_catalog: Catalog, stubs: list[Any]) -> None:
    r, backend = router(scenario_catalog, R2_SCRIPT)
    d = r.decide(R2_MESSAGES)
    assert d.pending is not None
    old = d.pending.model_copy(update={"expires_at": datetime.now(timezone.utc) - timedelta(seconds=1)})
    again = r.resume(old, selection="ok")
    assert (
        again.outcome is Outcome.CONFIRM and len(backend.requests) == 2 and again.trace.resumed_from == old.pending_id
    )


# --------------------------------------------------------------------------------------------------------------------
# Internal rounds: speculation miss, widen, fill, escalation
# --------------------------------------------------------------------------------------------------------------------


def test_speculation_miss_replans_the_chosen_tool(scenario_catalog: Catalog, stubs: list[Any]) -> None:
    script = {
        **R2_SCRIPT,
        "tool": {"get_weather": 0.9, "send_email": 0.05, "NO_TOOL": 0.05},
        "get_weather.city": {"NOT_STATED": 0.97, "NONE_OF_THESE": 0.03},
        "get_weather.unit": {"NOT_STATED": 0.96, "celsius": 0.02, "fahrenheit": 0.01, "NONE_OF_THESE": 0.01},
    }
    r, backend = router(scenario_catalog, script)
    full = r.compile(R2_MESSAGES)
    tokens = Limits().estimate_tokens(full.to_requests("")[0].to_wire())
    r.limits = Limits(max_tokens=tokens - 1)  # get_weather (probe-only) is cut by the budget
    d = r.decide(R2_MESSAGES)
    assert len(backend.requests) == 2 and d.rounds == 2
    assert not any(q.startswith("get_weather") for q in backend.requests[0].questions)
    assert set(backend.requests[1].questions) == {"tool", "get_weather.city", "get_weather.unit"}
    assert (d.outcome, d.call.arguments if d.call else None) == (Outcome.EXECUTE, {"city": "Zurich",
                                                                                   "unit": "celsius"})  # fmt: skip
    assert any("speculation miss" in n for n in d.trace.notes)


def widen_stubs(extra: Sequence[Any]) -> list[Any]:
    base = scenario_resolvers(amount=())
    ref = StubChoice("ref", {"send_email.to": ANNAS, "read_file.path": FILES},
                     widen_with={"read_file.path": list(extra)})  # fmt: skip
    return [ref, *base[1:]]


def test_widen_round_finds_the_value(scenario_catalog: Catalog) -> None:
    script = {
        "tool": {"read_file": 0.95, "search_web": 0.03, "NO_TOOL": 0.02},
        "read_file.path": {"services/payments/config/prod.yaml": 0.3, "NONE_OF_THESE": 0.65, "NOT_STATED": 0.05},
        "read_file.path.bucket.0": {"services/payments/config/app.yaml": 0.9, "NONE_OF_THESE": 0.1},
    }
    items = widen_stubs([APP_YAML])
    with resolvers(*items):
        r, backend = router(scenario_catalog, script)
        d = r.decide("Open the config file for the payments service")
        report = verify(d.trace, catalog=scenario_catalog)
    assert report.ok and next(c for c in report.checks if c.name == "redecode").ok
    assert items[0].widened == ["bucket"] and len(backend.requests) == 2
    assert list(backend.requests[1].questions) == ["read_file.path.bucket.0"]
    assert backend.requests[1].state == backend.requests[0].state  # same state: earlier answers stay valid
    assert (d.outcome, d.call.arguments if d.call else None) == (Outcome.EXECUTE, {
        "path": "services/payments/config/app.yaml"})  # fmt: skip
    assert d.rounds == 2 and d.trace.rounds[1].merged and d.trace.rounds[1].mode == "widen"


def test_widen_exhaustion_clarifies(scenario_catalog: Catalog) -> None:
    script = {
        "tool": {"read_file": 0.95, "NO_TOOL": 0.05},
        "read_file.path": {"services/payments/config/prod.yaml": 0.2, "NONE_OF_THESE": 0.8},
        "read_file.path.bucket.*": {"NONE_OF_THESE": 0.8, "NOT_STATED": 0.2},
    }
    items = widen_stubs([APP_YAML])
    with resolvers(*items):
        r, backend = router(scenario_catalog, script)
        d = r.decide("Open the config file for the payments service")
    assert items[0].widened == ["bucket", "hierarchy"] and len(backend.requests) == 3  # max_rounds = 2
    assert (d.outcome, d.rule, d.bottleneck.shape if d.bottleneck else None) == (
        Outcome.CLARIFY, "P7.slot.shape", "out_of_pool")  # fmt: skip
    assert d.prompt is not None and d.prompt.text == "What should the workspace-relative file path be?"


def test_fill_round_elects_generated_text(scenario_catalog: Catalog, stubs: list[Any]) -> None:
    class Filler:
        def __init__(self) -> None:
            self.requests: list[FillRequest] = []

        def fill(self, req: FillRequest) -> list[FillCandidate]:
            self.requests.append(req)
            return [FillCandidate(values={"body": "Sorry, I'm running 10 minutes late."})]

        async def afill(self, req: FillRequest) -> list[FillCandidate]:
            return self.fill(req)

    script = {
        **R2_SCRIPT,
        "send_email.body.accept.0": 0.3,
        "send_email.body.accept.1": 0.2,
        "send_email.body.accept.2": 0.9,
    }
    filler = Filler()
    r, backend = router(scenario_catalog, script, filler=filler)
    d = r.decide(R2_MESSAGES)
    assert len(filler.requests) == 1 and filler.requests[0].frozen["to"] == "anna.keller@acme.com"
    assert list(filler.requests[0].slots) == ["body"] and "x-jev" not in json.dumps(filler.requests[0].slots)
    assert list(backend.requests[1].questions) == ["send_email.body.accept.2"]  # only the new candidate is asked
    assert d.slots["body"].value == "Sorry, I'm running 10 minutes late." and d.slots["body"].channel == "generated"
    assert d.outcome is Outcome.CONFIRM and d.usage.llm_calls == 1 and d.rounds == 2
    assert "untrusted_channel" in d.trace.outcome.get("caps", []) or d.rule == "P9.external.confirm_band"


def test_escalation_gate_round(scenario_catalog: Catalog, stubs: list[Any]) -> None:
    class Escalator:
        def __init__(self, answer: Any) -> None:
            self.answer = answer
            self.seen: list[Decision] = []

        def escalate(self, messages: Any, tools: Any, decision: Decision) -> Any:
            assert all("x-jev" not in json.dumps(t) for t in tools)
            self.seen.append(decision)
            return self.answer

        async def aescalate(self, messages: Any, tools: Any, decision: Decision) -> Any:
            return self.escalate(messages, tools, decision)

    script = {**R2_SCRIPT, "tool": {"send_email": 0.45, "create_event": 0.3, "NO_TOOL": 0.25}}
    proposal = ProposedCall(
        name="send_email",
        arguments={"to": "anna.keller@acme.com", "subject": "Late", "body": "I'll be 10 minutes late."},
    )
    escalator = Escalator(proposal)
    r, backend = router(scenario_catalog, script, escalator=escalator)
    d = r.decide(R2_MESSAGES)
    assert escalator.seen[0].outcome is Outcome.ESCALATE and escalator.seen[0].rule == "P5.tool.ambiguous"
    assert len(backend.requests) == 2 and "tool" not in backend.requests[1].questions
    assert "send_email.subject.accept.3" in backend.requests[1].questions  # the LLM's subject, as generated
    assert d.outcome is Outcome.CONFIRM and d.usage.llm_calls == 1 and d.rounds == 2
    text = Escalator("I can't do that.")
    r2, _ = router(scenario_catalog, script, escalator=text)
    handed = r2.decide(R2_MESSAGES)
    assert handed.outcome is Outcome.ABSTAIN and handed.content == "I can't do that."


# --------------------------------------------------------------------------------------------------------------------
# Fail closed (P0), 422 isolation, splits
# --------------------------------------------------------------------------------------------------------------------


def http_router(catalog: Catalog, handler: Any, **kw: Any) -> Router:
    transport = httpx.MockTransport(handler)
    backend = HTTPBackend.openrouter_decisions("key", transport=transport, async_transport=transport,
                                               max_retries=0, sleep=lambda _: None)  # fmt: skip
    return Router(catalog, backend=backend, context=scenario_context(), **kw)


def answering(script: Mapping[str, Any]) -> Any:
    scripted = ScriptedBackend(script)

    def answer(body: Mapping[str, Any]) -> httpx.Response:
        response = scripted.decide(DecisionRequest.model_validate(body))
        return httpx.Response(200, json={**response.model_dump(mode="json", exclude_none=True),
                                         "usage": {"input_tokens": 1700, "cost": 0.00007}})  # fmt: skip

    return answer


def test_422_isolation_drops_the_family_and_resends(scenario_catalog: Catalog, stubs: list[Any]) -> None:
    bodies: list[dict[str, Any]] = []
    answer = answering(R2_SCRIPT)

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        bodies.append(body)
        if "get_weather.unit" in body["questions"]:
            loc = ["body", "questions", "get_weather.unit", "choice", "criteria"]
            return httpx.Response(422, json={"detail": [{"loc": loc, "msg": "bad", "type": "value_error"}]})
        return answer(body)  # type: ignore[no-any-return]

    d = http_router(scenario_catalog, handler).decide(R2_MESSAGES)
    assert len(bodies) == 2 and "get_weather.unit" not in bodies[1]["questions"]
    assert d.outcome is Outcome.CONFIRM and d.usage.jev_calls == 2 and d.usage.cost_usd == pytest.approx(0.00007)
    assert any("422 isolation" in n for n in d.trace.notes) and d.trace.rounds[0].calls[0].backend == (
        "openrouter_decisions")  # fmt: skip


def test_422_on_state_fails_closed(scenario_catalog: Catalog, stubs: list[Any]) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(422, json={"detail": [{"loc": ["body", "state"], "msg": "too long", "type": "x"}]})

    d = http_router(scenario_catalog, handler).decide(R2_MESSAGES)
    assert (d.outcome, d.rule) == (Outcome.ABSTAIN, "P0.backend.fail_closed") and d.tool_calls == []
    assert d.trace.rounds[0].calls[0].error is not None


def test_5xx_fails_closed_or_escalates(scenario_catalog: Catalog, stubs: list[Any]) -> None:
    class Handoff:
        def escalate(self, messages: Any, tools: Any, decision: Decision) -> str:
            return "A human will take over."

        async def aescalate(self, messages: Any, tools: Any, decision: Decision) -> str:
            return self.escalate(messages, tools, decision)

    down = http_router(scenario_catalog, lambda request: httpx.Response(503, json={}), escalator=Handoff())
    d = down.decide(R2_MESSAGES)
    assert (d.outcome, d.rule, d.content) == (Outcome.ESCALATE, "P0.backend.fail_closed", "A human will take over.")


def test_missing_answer_is_a_protocol_failure(scenario_catalog: Catalog, stubs: list[Any]) -> None:
    class Lossy(ScriptedBackend):
        def decide(self, request: DecisionRequest) -> DecisionResponse:
            response = super().decide(request)
            answers = {k: v for k, v in response.answers.items() if k != "send_email.to"}
            return response.model_copy(update={"answers": answers})

    r = Router(scenario_catalog, backend=Lossy(R2_SCRIPT), context=scenario_context())
    d = r.decide(R2_MESSAGES)
    assert (d.outcome, d.rule) == (Outcome.ABSTAIN, "P0.backend.fail_closed")


def test_split_round_is_one_round_with_identical_state(scenario_catalog: Catalog, stubs: list[Any]) -> None:
    r, backend = router(scenario_catalog, R2_SCRIPT, limits=Limits(max_questions=4))
    d = r.decide(R2_MESSAGES)
    assert len(backend.requests) > 1 and d.rounds == 1 and d.usage.jev_calls == len(backend.requests)
    assert all(req.state == backend.requests[0].state for req in backend.requests)
    assert "tool" in backend.requests[0].questions and d.outcome is Outcome.CONFIRM


async def test_async_decide_and_resume(scenario_catalog: Catalog, stubs: list[Any]) -> None:
    r, backend = router(scenario_catalog, NO_HISTORY_SCRIPT, limits=Limits(max_questions=4))
    d = await r.adecide(R2_REQUEST)
    assert d.outcome is Outcome.CLARIFY and len(backend.requests) > 1
    done = await r.aresume(d.pending_id or "", selection="pick:to:0")
    assert done.outcome is Outcome.EXECUTE and done.call is not None
    assert done.call.arguments["to"] == "anna.keller@acme.com"


# --------------------------------------------------------------------------------------------------------------------
# Trace and misc
# --------------------------------------------------------------------------------------------------------------------


def test_trace_store_and_verify(scenario_catalog: Catalog, stubs: list[Any]) -> None:
    class Store:
        def __init__(self) -> None:
            self.traces: list[Any] = []

        def put(self, trace: Any) -> None:
            self.traces.append(trace)

    store = Store()
    r, _ = router(scenario_catalog, R2_SCRIPT, trace_store=store)
    d = r.decide(R2_MESSAGES)
    assert store.traces == [d.trace]
    report = verify(d.trace, catalog=scenario_catalog, context=scenario_context(R2_MESSAGES))
    assert report.ok and {c.name for c in report.checks if c.ok} == {
        "hashes",
        "ballot_rebuild",
        "redecode",
        "values",
        "channels",
        "composition",
        "policy",
    }
    assert d.trace.composition["PI"] == pytest.approx(0.7137312)  # type: ignore[index]
    assert d.trace.bindings["to"]["label"] == "Anna Keller <anna.keller@acme.com>"


def test_decisions_are_deterministic(scenario_catalog: Catalog, stubs: list[Any]) -> None:
    r, _ = router(scenario_catalog, R2_SCRIPT)
    first, second = r.decide(R2_MESSAGES), r.decide(R2_MESSAGES)
    assert first.to_json() == second.to_json() and first.call == second.call


def test_history_context_is_used(scenario_catalog: Catalog, stubs: list[Any]) -> None:
    r, backend = router(scenario_catalog, R2_SCRIPT)
    r.decide(R2_MESSAGES)
    assert backend.requests[0].state["history"] == [  # type: ignore[index]
        {"role": "user", "text": R2_HISTORY[0]["content"]},
        {"role": "assistant", "text": R2_HISTORY[1]["content"]},
    ]
    assert Router(scenario_catalog, backend=ScriptedBackend(), policy=Policy()).backend_name == "scripted"


def test_critical_alternatives_use_the_critical_threshold(scenario_catalog: Catalog) -> None:
    script = r3_script(**{"transfer_funds.from_account": {SAVINGS: 0.95, TRAVEL: 0.02, CHECKING: 0.02,
                                                          "NONE_OF_THESE": 0.01}})  # fmt: skip
    with resolvers(*scenario_resolvers()):
        r, _ = router(scenario_catalog, script)
        d = r.decide(R3_REQUEST)
    assert d.prompt is not None and [o.id for o in d.prompt.options] == ["ok", "change", "cancel"]


def test_clicks_recompute_late_bound_values(scenario_catalog: Catalog, stubs: list[Any]) -> None:
    r, _ = router(scenario_catalog, NO_HISTORY_SCRIPT)
    d = r.decide(R2_REQUEST)
    assert d.prompt is not None and "Hi Annabel," in d.prompt.options[2].text  # each option is its own full call
    done = r.resume(d.pending_id or "", selection="pick:to:2")
    assert done.call is not None and done.call.arguments["to"] == "annabel.frey@muster.ch"
    assert done.call.arguments["body"].startswith("Hi Annabel,\n")


def test_empty_catalog_abstains_without_a_call() -> None:
    backend = ScriptedBackend()
    d = Router(Catalog.from_openai([]), backend=backend).decide("hello")
    assert (d.outcome, d.rule, d.rounds) == (Outcome.ABSTAIN, "P1.tool.no_tool", 0) and backend.requests == []


def test_loop_mode_reports_done_after(scenario_catalog: Catalog, stubs: list[Any]) -> None:
    r, backend = router(scenario_catalog, {**R2_SCRIPT, "send_email.done_after": 0.95})
    d = r.decide(R2_MESSAGES, mode="loop")
    assert "send_email.done_after" in backend.requests[0].questions
    assert d.gates["done_after"] == pytest.approx(0.95)
