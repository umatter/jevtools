"""Agent mechanics (spec §6.1, §6.5): guards, tool errors and retries, executor forms, idempotency, TOCTOU.

Scripted answers are illustrative [I]; they drive the policy branches, not Jev accuracy.
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from typing import Any

import pytest

from jevtools.decision import ToolCall
from jevtools.loop import (
    IDEMPOTENCY_META_KEY,
    RULE_MAX_COST,
    RULE_MAX_LLM_CALLS,
    RULE_MAX_ROUNDS,
    RULE_MAX_STEPS,
    RULE_NO_PROGRESS,
    RULE_RETRY_CONFIRM,
    Agent,
    LoopBudget,
    LoopObservation,
)
from jevtools.policy import Outcome, Policy
from jevtools.wire import DecisionRequest
from tests.loop.support import Workspace, observations_of
from tests.scenario import scripts
from tests.scenario.fixtures import (
    account_rows,
    accounts,
    contacts,
    default_sources,
    scenario_context,
    scenario_messages,
    scenario_router,
)

WEATHER = "What's the weather in Zurich, Basel and Geneva?"
CITIES = ["Zurich", "Basel", "Geneva"]


def weather_script(done_after: float = 0.1) -> Any:
    """Each step asks get_weather for the next city (by the number of observations so far)."""

    def script(request: DecisionRequest) -> Mapping[str, Any]:
        n = len(observations_of(request))
        return {
            "tool": {"get_weather": 0.95, "NO_TOOL": 0.05},
            "get_weather.city": {CITIES[min(n, 2)]: 0.95, "NONE_OF_THESE": 0.05},
            "get_weather.unit": {"NOT_STATED": 0.95, "NONE_OF_THESE": 0.05},
            "*.done_after": done_after,
            "search_web.query.accept.*": 0.1,
        }

    return script


def weather_agent(executor: Any, *, budget: LoopBudget | None = None, **kw: Any) -> tuple[Agent, Any, list[Any]]:
    router, backend = scenario_router(weather_script(), **kw)
    return Agent(router, {"get_weather": executor}, budget=budget), backend, []


def test_each_step_re_asks_with_the_new_state_and_three_cities_run() -> None:
    seen: list[str] = []

    def weather(city: str, unit: str = "celsius") -> dict[str, Any]:
        seen.append(city)
        return {"city": city, "temp": 10 + len(seen)}

    agent, backend, _ = weather_agent(weather, budget=LoopBudget(max_steps=3))
    result = agent.run(WEATHER)
    assert seen == CITIES and len(backend.requests) == 3  # I4: one fresh round per step
    assert [len(observations_of(r)) for r in backend.requests] == [0, 1, 2]
    assert (result.outcome, result.rule, result.reason) == (Outcome.ESCALATE, RULE_MAX_STEPS, "budget")
    assert result.usage.steps == 3 and result.usage.executions == 3
    assert [o.summary for o in result.observations] == ["2 fields"] * 3


def test_no_progress_escalates_after_two_steps_without_a_new_observation() -> None:
    agent, _, _ = weather_agent(lambda city, unit="celsius": "sunny")
    result = agent.run(WEATHER)
    assert (result.outcome, result.rule, result.reason) == (Outcome.ESCALATE, RULE_NO_PROGRESS, "no_progress")
    assert result.usage.executions == 3


@pytest.mark.parametrize(
    ("budget", "rule", "executions"),
    [
        (LoopBudget(max_rounds=1), RULE_MAX_ROUNDS, 1),
        (LoopBudget(max_cost_usd=1e-6), RULE_MAX_COST, 1),
        (LoopBudget(max_steps=0), RULE_MAX_STEPS, 0),
    ],
)
def test_budget_caps_stop_before_a_new_step(budget: LoopBudget, rule: str, executions: int) -> None:
    agent, _, _ = weather_agent(lambda city, unit="celsius": {"city": city}, budget=budget)
    result = agent.run(WEATHER)
    assert (result.outcome, result.rule, result.reason) == (Outcome.ESCALATE, rule, "budget")
    assert result.usage.executions == executions


def test_llm_call_cap_applies_when_the_router_has_an_llm() -> None:
    class Text:
        def complete(self, messages: Any) -> str:
            return "hi"

        async def acomplete(self, messages: Any) -> str:
            return "hi"

    agent, _, _ = weather_agent(lambda **_: {}, budget=LoopBudget(max_llm_calls=0), text_llm=Text())
    assert agent.run(WEATHER).rule == RULE_MAX_LLM_CALLS


def test_budget_defaults_follow_the_policy() -> None:
    router, _ = scenario_router({}, policy=Policy.from_dict({"loop": {"max_steps": 3}}))
    assert Agent(router, {}).budget.max_steps == 3
    assert LoopBudget() == LoopBudget(max_steps=6, max_rounds=12, max_cost_usd=0.01, max_llm_calls=2)


def test_done_after_ends_the_loop_after_a_success_only() -> None:
    router, _ = scenario_router(weather_script(done_after=0.9))
    ok = Agent(router, {"get_weather": lambda city, unit="celsius": {"t": 1}}).run(WEATHER)
    assert (ok.outcome, ok.reason, ok.usage.executions) == (Outcome.DONE, "done_after", 1)

    def broken(city: str, unit: str = "celsius") -> Any:
        raise TimeoutError("weather service down")

    router, _ = scenario_router(weather_script(done_after=0.9))
    failed = Agent(router, {"get_weather": broken}).run(WEATHER)
    assert failed.outcome is not Outcome.DONE and failed.observations[0].status == "error"


# -- tool errors and retries -----------------------------------------------------------------------------------------


def test_read_tool_errors_are_observations_and_retried_automatically() -> None:
    calls: list[str] = []

    def weather(city: str, unit: str = "celsius") -> Any:
        calls.append(city)
        raise ConnectionError(f"no route to weather for {city}")

    agent, backend, _ = weather_agent(weather)
    result = agent.run(WEATHER)
    first = result.observations[0]
    assert isinstance(first, LoopObservation) and first.status == "error"
    assert first.error == "ConnectionError: no route to weather for Zurich" and first.summary is not None
    assert result.steps[0].attempts == 2 and calls[:2] == ["Zurich", "Zurich"]  # read tier: one automatic retry
    state = backend.requests[1].state
    assert isinstance(state, dict) and state["observations"][0]["status"] == "error"
    assert state["progress"][0] == (
        'Step 1: get_weather(city="Zurich", unit="celsius") → error, ConnectionError: no route to weather for Zurich'
    )


R2_SURE: dict[str, Any] = {
    **{k: v for k, v in scripts.R2.items() if ".accept." not in k or k.startswith("search_web")},
    "tool": {"send_email": 0.99, "NO_TOOL": 0.01},
    "send_email.authorized": 0.99,
    "send_email.to": {scripts.KELLER: 0.99, "NONE_OF_THESE": 0.01},
    "send_email.to.present": 0.99,
    "send_email.subject.accept.*": 0.99,
    "send_email.body.accept.*": 0.99,
    "*.done_after": 0.1,
}
"""R2 at execute level (external tier: Π ≥ .80, authorized ≥ .90, content accept ≥ .80)."""


def test_external_tools_are_never_retried_automatically() -> None:
    attempts: list[str] = []

    def send_email(to: str, subject: str, body: str) -> Any:
        attempts.append(to)
        raise ConnectionError("SMTP relay unavailable")

    router, _ = scenario_router(R2_SURE, context=scenario_context(history=True))
    agent = Agent(router, {"send_email": send_email})
    result = agent.run(scenario_messages(scripts.R2_REQUEST, history=True))
    assert result.decisions[0].outcome is Outcome.EXECUTE and attempts == ["anna.keller@acme.com"]
    assert result.steps[0].attempts == 1 and result.observations[0].status == "error"
    # the next step proposes the same call again: a retry of an external tool needs a fresh confirmation
    assert (result.outcome, result.rule, result.reason) == (Outcome.ESCALATE, RULE_RETRY_CONFIRM, "retry_needs_confirm")
    assert attempts == ["anna.keller@acme.com"]


def test_missing_executor_is_an_error_observation() -> None:
    router, _ = scenario_router(weather_script(done_after=0.9))
    result = Agent(router, {}).run(WEATHER)
    assert result.observations[0].status == "error"
    assert "MissingExecutor: no executor for tool 'get_weather'" in str(result.observations[0].content)


def test_mcp_is_error_results_are_error_observations() -> None:
    error = {"content": [{"type": "text", "text": "rate limited"}], "isError": True}
    agent, _, _ = weather_agent(lambda city, unit="celsius": error)
    result = agent.run(WEATHER)
    assert result.observations[0].status == "error" and result.observations[0].content == "rate limited"


# -- executor forms and idempotency ----------------------------------------------------------------------------------


def test_a_dispatcher_receives_the_call_and_its_idempotency_key() -> None:
    received: list[tuple[ToolCall, str]] = []

    def dispatch(call: ToolCall, idempotency_key: str) -> Any:
        received.append((call, idempotency_key))
        return {"ok": True}

    router, _ = scenario_router(weather_script(done_after=0.9))
    result = Agent(router, dispatch).run(WEATHER)
    call, key = received[0]
    assert call.name == "get_weather" and key == call.idempotency_key == result.executed[0].idempotency_key


def test_an_mcp_session_executor_gets_the_key_in_meta() -> None:
    class Session:
        def __init__(self) -> None:
            self.calls: list[tuple[str, dict[str, Any], dict[str, Any] | None]] = []

        async def call_tool(self, name: str, arguments: dict[str, Any], meta: dict[str, Any] | None = None) -> Any:
            self.calls.append((name, arguments, meta))
            return {
                "content": [{"type": "text", "text": "12 degrees"}],
                "structuredContent": {"temp": 12},
                "isError": False,
            }

    session = Session()
    router, _ = scenario_router(weather_script(done_after=0.9))
    result = Agent(router, session).run(WEATHER)
    name, arguments, meta = session.calls[0]
    assert name == "get_weather" and arguments["city"] == "Zurich"
    assert meta == {IDEMPOTENCY_META_KEY: result.executed[0].idempotency_key}
    assert result.observations[0].content == {"temp": 12}  # structuredContent first


def test_async_executors_work_in_run_and_arun() -> None:
    async def weather(city: str, unit: str = "celsius") -> dict[str, Any]:
        await asyncio.sleep(0)
        return {"city": city}

    router, _ = scenario_router(weather_script(done_after=0.9))
    assert Agent(router, {"get_weather": weather}).run(WEATHER).outcome is Outcome.DONE
    router, _ = scenario_router(weather_script(done_after=0.9))
    result = asyncio.run(Agent(router, {"get_weather": weather}).arun(WEATHER))
    assert result.outcome is Outcome.DONE and result.observations[0].content == {"city": "Zurich"}


def test_a_key_is_never_executed_twice() -> None:
    ws = Workspace()
    router, _ = scenario_router(weather_script(done_after=0.9))
    agent = Agent(router, ws.executors())
    first = agent.run(WEATHER)
    call = first.executed[0]
    replayed = agent.execute(call)  # e.g. a host re-emitting the same decision: its key already ran
    assert replayed is first.observations[0] and len(ws.calls("get_weather")) == 1
    fresh = agent.execute(call.model_copy(update={"idempotency_key": "idem_other"}))
    assert fresh.status == "ok" and len(ws.calls("get_weather")) == 2


# -- confirm / clarify, TOCTOU ---------------------------------------------------------------------------------------


def test_clarify_pauses_and_a_click_continues_the_loop() -> None:
    ws = Workspace()
    script = {**scripts.R2_NO_HISTORY, "*.done_after": 0.9}
    router, _ = scenario_router(script)
    agent = Agent(router, ws.executors())
    first = agent.run(scripts.R2_REQUEST)
    assert first.outcome is Outcome.CLARIFY and first.pending is not None
    pick = next(i for i in first.pending.options if i.startswith("pick:"))
    done = agent.resume(first.pending, selection=pick)  # a complete-call menu click is a confirmation
    assert done.outcome is Outcome.DONE and [m["to"] for m in ws.sent] == ["anna.keller@acme.com"]


def test_toctou_balance_drop_blocks_execution_and_replans() -> None:
    ws = Workspace()
    router, backend = scenario_router(scripts.R3)
    agent = Agent(router, ws.executors())
    first = agent.run(scripts.R3_REQUEST)
    assert first.outcome is Outcome.CONFIRM and first.pending is not None
    drained = scenario_context(sources=[contacts(), accounts(account_rows(acc_7731=100.0)), default_sources()[2]])
    after = agent.resume(first.pending, selection="ok", context=drained)
    assert ws.transfers == [] and after.outcome is not Outcome.EXECUTE
    assert after.usage.rounds == 2 and len(backend.requests) == 2  # re-planned in a new round
    assert any("TOCTOU" in note for note in after.decisions[-1].trace.notes)


def test_the_host_revalidate_hook_cancels_a_delayed_call() -> None:
    ws = Workspace()
    checked: list[str] = []

    def revalidate(call: ToolCall, ctx: Any) -> list[str]:
        checked.append(call.name)
        return ["recipient left the company"] if len(checked) == 1 else []

    script = {**scripts.R2, "*.done_after": 0.9}
    router, backend = scenario_router(script, context=scenario_context(history=True))
    agent = Agent(router, ws.executors(), revalidate=revalidate)
    first = agent.run(scenario_messages(scripts.R2_REQUEST, history=True))
    assert first.outcome is Outcome.CONFIRM and first.pending is not None
    after = agent.resume(first.pending, selection="ok")
    assert checked == ["send_email"] and ws.sent == []
    note = after.steps[-2].note
    assert note is not None and note.startswith("TOCTOU: recipient left")
    assert after.outcome is Outcome.CONFIRM and len(backend.requests) == 2  # re-planned: a fresh confirm


def test_tool_sources_in_the_context_are_bound_to_the_executors() -> None:
    from jevtools.sources.toolsource import ToolSource

    listed: list[int] = []

    def list_cities() -> dict[str, Any]:
        listed.append(1)
        return {"cities": [{"name": "Zurich"}, {"name": "Basel"}]}

    source = ToolSource("list_cities", items="$.cities[*]", key="name")
    ctx = scenario_context(sources=[*default_sources(), source])
    router, _ = scenario_router(weather_script(), context=ctx)
    agent = Agent(
        router,
        {"get_weather": lambda city, unit="celsius": {"c": city}, "list_cities": list_cities},
        budget=LoopBudget(max_steps=2),
    )
    agent.run(WEATHER, context=ctx)
    assert source.bound and listed == [1] and [r["name"] for r in source.rows] == ["Zurich", "Basel"]


def test_a_free_text_reply_resumes_with_one_round_and_can_pause_again() -> None:
    ws = Workspace()

    def script(request: DecisionRequest) -> Mapping[str, Any]:
        answers: dict[str, Any] = {**scripts.R2_NO_HISTORY, "*.done_after": 0.9}
        if "reply" in request.questions:
            answers["reply"] = {"option_2": 0.9, "OTHER": 0.05, "CANCEL": 0.05}
        if observations_of(request):
            answers["tool"] = {"DONE": 0.95, "NO_TOOL": 0.05}
        return answers

    router, backend = scenario_router(script)
    agent = Agent(router, ws.executors())
    first = agent.run(scripts.R2_REQUEST)
    assert first.outcome is Outcome.CLARIFY and first.pending is not None
    second = agent.resume(first.pending, reply="the gmail one please")
    assert second.outcome is Outcome.CONFIRM and second.pending is not None and ws.sent == []
    assert second.usage.rounds == 2 and "reply" in backend.requests[1].questions
    # the resume round of a loop-mode prompt still asks done_after (the pending remembers the loop), so the
    # confirmed execution ends the run without one more round
    assert "send_email.done_after" in backend.requests[1].questions
    done = agent.resume(second.pending, selection="ok")
    assert [m["to"] for m in ws.sent] == ["anna.rossi@gmail.com"]
    assert (done.outcome, done.rule, done.usage.rounds) == (Outcome.DONE, "P10.loop.done", 2)


def test_a_pending_can_be_resumed_by_another_agent() -> None:
    ws = Workspace()
    router, _ = scenario_router({**scripts.R2, "*.done_after": 0.9}, context=scenario_context(history=True))
    first = Agent(router, ws.executors()).run(scenario_messages(scripts.R2_REQUEST, history=True))
    assert first.pending is not None
    later = Agent(router, ws.executors()).resume(first.pending, selection="ok")  # e.g. after a process restart
    assert later.outcome is Outcome.DONE and len(ws.sent) == 1
    assert later.notes == [f"resumed {first.pending.pending_id} without its paused run"]
