"""The Router (spec §9.1): compile a round, ask Jev, decode, compose, apply the policy, emit a Decision and a Trace.

The round loop is written once, as a generator that yields *effects* (ask Jev, call the Filler, the Escalator or
the text LLM) and receives their results; :meth:`Router.decide` drives it synchronously (split calls in a thread
pool) and :meth:`Router.adecide` asynchronously (split calls with ``asyncio.gather``).

Internal actions (§3.8.1): ``widen`` through a resolver's optional ``widen`` hook (≤ ``widen.max_rounds``), ``fill``
through the Filler (≤ 1 per decision), a re-plan after a speculation miss, and the Escalator's gate round. Backend
failures fail closed (P0) after 422 isolation: the offending family is dropped and the call re-sent once.

Resume (§3.8.5): a click binds the chosen value with ``p = 1`` (channel ``user``) without a Jev call — a click on a
complete-call option (external/critical) or on ``ok`` is a confirmation; a free-text reply compiles one resume round
(the pending tool's fan-out plus the ``reply`` Choice). Delayed calls are revalidated (TOCTOU) before execution.
The Router holds no kind-specific logic: everything about values goes through resolvers.
"""

from __future__ import annotations

import asyncio
import time
import warnings
from collections.abc import Callable, Generator, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from typing import Any

from jevtools.backends.base import Backend
from jevtools.backends.errors import BackendError, JevProtocolError, JevValidationError
from jevtools.ballot import Ballot, BallotQuestion
from jevtools.budget import TokenEstimator
from jevtools.candidates import CANCEL, Candidate, Channel, Pool, display_value, value_key
from jevtools.canonical import jsonable, sha256_of
from jevtools.confidence import Composition, IsotonicCalibrator, confidence
from jevtools.context import Context, Message, Mode, Turn
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
    SlotReport,
    ToolCall,
)
from jevtools.decode import (
    Decoded,
    ToolDecode,
    bind_value,
    choice_mass,
    collect_answers,
    decode_answers,
    decode_reply,
    policy_input,
    value_origin,
    with_decision,
    with_tool,
)
from jevtools.errors import PendingScopeError
from jevtools.fallback import Escalator, FillCandidate, Filler, FillRequest, ObservationPreview, ProposedCall, TextLLM
from jevtools.kinds.base import ResolveContext, get_resolver
from jevtools.plan import (
    PoolKey,
    RoundPlan,
    ToolChoice,
    compile_round,
    followup_ballot,
    inject_candidates,
    merge_ballots,
)
from jevtools.policy import (
    RULE_NO_TOOL,
    RULE_SLOT_SHAPE,
    Action,
    Outcome,
    Policy,
    PolicyInput,
    PolicyResult,
    Tier,
    bottleneck_shape,
    evaluate,
)
from jevtools.prompts import (
    Actions,
    AltChoice,
    Binding,
    clarify_menu,
    confirm_card,
    grid_menu,
    open_question,
    parse_short_reply,
    refuse_notice,
    tool_menu,
    yes_no_menu,
)
from jevtools.spec.catalog import Catalog, ToolLike, strip_xjev
from jevtools.spec.constraints import ConstraintContext
from jevtools.spec.models import SlotSpec, ToolSpec
from jevtools.spec.schema import validate
from jevtools.trace import RoundRecord, StoreBodies, Trace, build_trace, call_record
from jevtools.validate import Limits, cached_limits
from jevtools.wire import Answer, DecisionRequest, DecisionResponse

Revalidator = Callable[[ToolDecode, Context], list[str]]
"""TOCTOU hook: problems that forbid executing a delayed call (empty list = still valid)."""
LIVE_MAX = 256
"""Pending decisions (handles and click-resume state) kept in memory: expired ones are dropped on every insert, the
oldest beyond this bound are evicted (a handle no longer held is safely recompiled on resume)."""
REPLY_PREFIXES = ("say ", "tell her ", "tell him ", "tell them ")
WIDEN_STAGES = ("bucket", "hierarchy")
"""Coverage-round stages (§4.6): ranking pages plus a group Choice, then the items of the top groups."""


# --------------------------------------------------------------------------------------------------------------------
# Effects (what the flow asks its driver to do)
# --------------------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class _Ask:
    requests: list[DecisionRequest]


@dataclass(frozen=True)
class CallResult:
    """Outcome of one Jev call: a response or the backend error, and the latency."""

    response: DecisionResponse | None
    error: BackendError | None
    latency_ms: int


@dataclass(frozen=True)
class _Fill:
    request: FillRequest


@dataclass(frozen=True)
class _Escalate:
    messages: list[dict[str, Any]]
    tools: list[dict[str, Any]]
    decision: Decision


@dataclass(frozen=True)
class _Text:
    messages: list[dict[str, Any]]


Effect = _Ask | _Fill | _Escalate | _Text
Flow = Generator[Effect, Any, Decision]


# --------------------------------------------------------------------------------------------------------------------
# Per-decision state
# --------------------------------------------------------------------------------------------------------------------


@dataclass
class _Usage:
    jev_calls: int = 0
    tokens: int = 0
    cost: float | None = None
    llm_calls: int = 0

    def add(self, response: DecisionResponse) -> None:
        self.tokens += response.usage.input_tokens or 0
        if response.usage.cost is not None:
            self.cost = (self.cost or 0.0) + response.usage.cost

    def to_model(self) -> DecisionUsage:
        return DecisionUsage(jev_calls=self.jev_calls, jev_input_tokens=self.tokens, llm_calls=self.llm_calls,
                             cost_usd=self.cost)  # fmt: skip


@dataclass
class _Session:
    """One decision: its context, rounds, usage and the user bindings that survive re-decodes."""

    ctx: Context
    mode: Mode
    tool_choice: ToolChoice
    rounds: list[RoundRecord] = field(default_factory=list)
    usage: _Usage = field(default_factory=_Usage)
    notes: list[str] = field(default_factory=list)
    resumed_from: str | None = None
    widen_rounds: int = 0
    widen_exhausted: set[str] = field(default_factory=set)
    fills: int = 0
    respeculated: bool = False
    escalated: bool = False
    confirmed_call: tuple[str, str] | None = None
    """The call the user confirmed (``ok``, or a complete-call option): ``(tool, canonical arguments digest)``. It
    confirms only a decoded call equal to it (§3.8.5: the option showed the whole resulting call)."""
    bindings: dict[str, tuple[Any, float, str | None]] = field(default_factory=dict)
    """User bindings (clicks, reply picks, passthrough) applied after every decode: slot → (value, p, label)."""
    tool_pick: tuple[str, float] | None = None
    verified: set[tuple[str, tuple[str, ...], str]] = field(default_factory=set)
    """``(tool, slot path, value key)`` of every elected record a ``verify`` round has asked about."""
    replanned: bool = False
    """Set on the re-plan after a failed TOCTOU check (no second revalidation loop)."""
    loop: bool = False
    """A resume (or recompile) of a prompt raised in loop mode: the round still asks ``done_after`` (§6.1)."""

    @property
    def in_loop(self) -> bool:
        """Whether this decision is an agent-loop step (loop mode, or a resume of a loop-mode prompt)."""
        return self.mode == "loop" or self.loop


@dataclass
class _State:
    """The decode view of the current same-state rounds."""

    plan: RoundPlan
    ballot: Ballot
    answers: dict[str, Answer]
    pools: dict[PoolKey, Pool]
    dropped: set[PoolKey] = field(default_factory=set)
    decoded: Decoded | None = None
    failure: BaseException | None = None


@dataclass
class _Live:
    """What a click resume needs, kept in memory by pending id."""

    session: _Session
    state: _State
    tool: str | None


# --------------------------------------------------------------------------------------------------------------------
# Router
# --------------------------------------------------------------------------------------------------------------------


def _probed_limits(backend: Backend) -> Limits:
    """The limits ``jevtools probe`` measured for this backend and model (spec §8.7), else the defaults. An
    unreadable limits file is reported and ignored: the defaults are the conservative documented values."""
    try:
        return cached_limits(backend) or Limits()
    except (OSError, ValueError) as exc:
        warnings.warn(f"jevtools: ignoring the probed limits file of {backend.name}/{backend.model}: {exc}",
                      UserWarning, stacklevel=3)  # fmt: skip
        return Limits()


class Router:
    """Decide tool calls with Jev (spec §9.1).

    ``tools`` is a :class:`~jevtools.spec.catalog.Catalog` or anything :meth:`Catalog.from_any` accepts (compiled
    against ``context``'s sources). ``revalidate`` replaces the default TOCTOU check; ``calibrators`` map a tier to
    a fitted :class:`~jevtools.confidence.IsotonicCalibrator`. Without ``limits``, the limits ``jevtools probe``
    cached for this backend and model are used (:func:`~jevtools.validate.cached_limits`), else the defaults.
    """

    def __init__(
        self,
        tools: Catalog | Sequence[ToolLike],
        *,
        backend: Backend,
        policy: Policy | None = None,
        context: Context | None = None,
        filler: Filler | None = None,
        escalator: Escalator | None = None,
        text_llm: TextLLM | None = None,
        limits: Limits | None = None,
        trace_store: Any = None,
        estimator: TokenEstimator | None = None,
        calibrators: Mapping[str, IsotonicCalibrator] | None = None,
        revalidate: Revalidator | None = None,
        store_bodies: StoreBodies = "full",
    ) -> None:
        self.context = context or Context()
        sources = list(self.context.sources.values())
        self.catalog = tools if isinstance(tools, Catalog) else Catalog.from_any(tools, sources=sources)
        self.backend = backend
        self.policy = policy or Policy()
        self.filler = filler
        self.escalator = escalator
        self.text_llm = text_llm
        self.limits = (limits if limits is not None else _probed_limits(backend)).within_budget(self.policy.budget)
        self.trace_store = trace_store
        self.estimator = estimator or TokenEstimator(self.limits.chars_per_token)
        self.calibrators = dict(calibrators or {})
        self.revalidate = revalidate or self.default_revalidate
        self.store_bodies = store_bodies
        self.pendings: dict[str, Pending] = {}
        self._live: dict[str, _Live] = {}

    # -- identity -----------------------------------------------------------------------------------------------

    @property
    def backend_name(self) -> str:
        """The backend's ``name`` (class name as a fallback)."""
        return str(getattr(self.backend, "name", type(self.backend).__name__))

    @property
    def model(self) -> str:
        """The model id sent on the wire."""
        return str(getattr(self.backend, "model", ""))

    def round_limits(self) -> Limits:
        """Limits with the learned token ratio of this backend and model (§5.5)."""
        ratio = self.estimator.ratio(self.backend_name, self.model)
        return self.limits.model_copy(update={"token_ratio": ratio})

    # -- public API ---------------------------------------------------------------------------------------------

    def context_for(self, messages: str | Sequence[Message | Turn], context: Context | None = None) -> Context:
        """``context`` (or the router's) with the conversation replaced by ``messages``."""
        return (context or self.context).with_messages(messages)

    def compile(
        self,
        messages: str | Sequence[Message | Turn],
        *,
        context: Context | None = None,
        mode: Mode = "turn",
        tool_choice: ToolChoice = "auto",
    ) -> Ballot:
        """The round-1 Ballot, without any network call (tests, lint, explain)."""
        ctx = self.context_for(messages, context)
        return compile_round(self.catalog, ctx, self.policy, mode=mode, tool_choice=tool_choice,
                             limits=self.round_limits(), has_filler=self.filler is not None).ballot  # fmt: skip

    def decide(
        self,
        messages: str | Sequence[Message | Turn],
        *,
        context: Context | None = None,
        tool_choice: ToolChoice = "auto",
        parallel_tool_calls: bool = False,
        mode: Mode = "turn",
    ) -> Decision:
        """Decide the next action for ``messages`` (``parallel_tool_calls`` is accepted; multi-intent segmentation
        is not part of this version, so one call is decided)."""
        session = _Session(ctx=self.context_for(messages, context), mode=mode, tool_choice=tool_choice)
        return self._drive(self._flow(session))

    async def adecide(
        self,
        messages: str | Sequence[Message | Turn],
        *,
        context: Context | None = None,
        tool_choice: ToolChoice = "auto",
        parallel_tool_calls: bool = False,
        mode: Mode = "turn",
    ) -> Decision:
        """Async :meth:`decide` (split calls run concurrently with ``asyncio.gather``)."""
        session = _Session(ctx=self.context_for(messages, context), mode=mode, tool_choice=tool_choice)
        return await self._adrive(self._flow(session))

    def resume(
        self,
        pending: Pending | str,
        *,
        selection: str | None = None,
        reply: str | None = None,
        context: Context | None = None,
    ) -> Decision:
        """Resume a confirm/clarify: ``selection`` is a click on an option id; ``reply`` is free text (a reply that
        equals an option number, an option text, ``yes``/``ok``/``send`` or ``cancel`` counts as a click)."""
        return self._drive(self._resume_flow(pending, selection, reply, context))

    async def aresume(
        self,
        pending: Pending | str,
        *,
        selection: str | None = None,
        reply: str | None = None,
        context: Context | None = None,
    ) -> Decision:
        """Async :meth:`resume`."""
        return await self._adrive(self._resume_flow(pending, selection, reply, context))

    # -- drivers ------------------------------------------------------------------------------------------------

    def _drive(self, flow: Flow) -> Decision:
        value: Any = None
        while True:
            try:
                effect = flow.send(value)
            except StopIteration as stop:
                return self._stored(stop.value)
            value = self._perform(effect)

    async def _adrive(self, flow: Flow) -> Decision:
        value: Any = None
        while True:
            try:
                effect = flow.send(value)
            except StopIteration as stop:
                return self._stored(stop.value)
            value = await self._aperform(effect)

    def _perform(self, effect: Effect) -> Any:
        if isinstance(effect, _Ask):
            if len(effect.requests) == 1:
                return [self._call(effect.requests[0])]
            with ThreadPoolExecutor(max_workers=min(8, len(effect.requests))) as pool:
                return list(pool.map(self._call, effect.requests))
        if isinstance(effect, _Fill):
            assert self.filler is not None
            return self.filler.fill(effect.request)
        if isinstance(effect, _Escalate):
            assert self.escalator is not None
            return self.escalator.escalate(effect.messages, effect.tools, effect.decision)
        assert self.text_llm is not None
        return self.text_llm.complete(effect.messages)

    async def _aperform(self, effect: Effect) -> Any:
        if isinstance(effect, _Ask):
            return list(await asyncio.gather(*(self._acall(r) for r in effect.requests)))
        if isinstance(effect, _Fill):
            assert self.filler is not None
            return await self.filler.afill(effect.request)
        if isinstance(effect, _Escalate):
            assert self.escalator is not None
            return await self.escalator.aescalate(effect.messages, effect.tools, effect.decision)
        assert self.text_llm is not None
        return await self.text_llm.acomplete(effect.messages)

    def _call(self, request: DecisionRequest) -> CallResult:
        start = time.perf_counter()
        try:
            response = self.backend.decide(request)
        except BackendError as exc:
            return CallResult(None, exc, _ms(start))
        return CallResult(response, None, _ms(start))

    async def _acall(self, request: DecisionRequest) -> CallResult:
        start = time.perf_counter()
        try:
            response = await self.backend.adecide(request)
        except BackendError as exc:
            return CallResult(None, exc, _ms(start))
        return CallResult(response, None, _ms(start))

    def _stored(self, decision: Decision) -> Decision:
        if self.trace_store is not None and decision.trace is not None:
            self.trace_store.put(decision.trace)
        return decision

    # -- the round loop -----------------------------------------------------------------------------------------

    def _flow(
        self,
        s: _Session,
        *,
        pending: Pending | None = None,
        reply_options: Sequence[Mapping[str, str]] | None = None,
        extra: Mapping[PoolKey, Sequence[Candidate]] | None = None,
        tool_choice: ToolChoice | None = None,
    ) -> Flow:
        """Plan → ask → decode → policy loop → finish."""
        choice = s.tool_choice if tool_choice is None else tool_choice
        if choice == "none":
            result = PolicyResult(outcome=Outcome.ABSTAIN, rule=RULE_NO_TOOL, reason="tool_choice_none")
            return (yield from self._finish(s, None, result, None))
        plan = self._compile(
            s, tool_choice=choice, pending=pending, reply_options=reply_options, extra_candidates=extra
        )
        state = yield from self._round(s, plan)
        if pending is not None and state.decoded is not None:
            self._apply_reply(s, state, pending)
        state, result, inp = yield from self._policy_loop(s, state)
        if result.outcome is Outcome.ESCALATE and self.escalator is not None and not s.escalated:
            return (yield from self._escalate(s, state, result, inp))
        return (yield from self._finish(s, state, result, inp, delayed=s.resumed_from is not None))

    def _compile(self, s: _Session, **kw: Any) -> RoundPlan:
        return compile_round(self.catalog, s.ctx, self.policy, mode=s.mode, limits=self.round_limits(),
                             has_filler=self.filler is not None, round=len(s.rounds) + 1, **kw)  # fmt: skip

    def _round(self, s: _Session, plan: RoundPlan) -> Generator[Effect, Any, _State]:
        """Ask one full round and decode it (fail closed on backend or protocol errors)."""
        state = _State(plan=plan, ballot=plan.ballot, answers={}, pools=dict(plan.pools))
        if not plan.ballot.questions:  # nothing to ask (no tools): decodes to "no tool"
            self._decode(s, state)
            return state
        answered = yield from self._ask(s, plan.ballot, state, merged=False)
        if answered is not None:
            state.ballot = answered
            self._decode(s, state)
        return state

    def _ask(
        self, s: _Session, ballot: Ballot, state: _State, *, merged: bool
    ) -> Generator[Effect, Any, Ballot | None]:
        """Send a Ballot; on a 422 naming questions, drop their families and re-send the failed calls once.
        Returns the Ballot that was answered (``None`` and ``state.failure`` set when the round failed)."""
        limits = self.round_limits()
        requests = ballot.to_requests(self.model, id_mode=limits.id_mode,
                                      object_instructions=limits.instructions_as_object)  # fmt: skip
        results: list[CallResult] = yield _Ask(requests)
        s.usage.jev_calls += len(requests)
        record = RoundRecord(round=len(s.rounds) + 1, mode=ballot.mode, ballot_sha256=ballot.sha256, merged=merged)
        s.rounds.append(record)
        failed = [i for i, r in enumerate(results) if r.error is not None]
        if failed:
            retried = yield from self._isolate(s, ballot, requests, results, failed, state)
            if retried is None:
                self._record(s, record, ballot, requests, results)
                return None
            ballot, requests, results = retried
            record.ballot_sha256 = ballot.sha256
        self._record(s, record, ballot, requests, results)
        try:
            answers, notes = collect_answers(ballot, [r.response for r in results if r.response is not None],
                                             id_mode=limits.id_mode)  # fmt: skip
        except JevProtocolError as exc:
            state.failure = exc
            return None
        s.notes += notes
        state.answers.update(answers)
        return ballot

    def _isolate(
        self,
        s: _Session,
        ballot: Ballot,
        requests: list[DecisionRequest],
        results: list[CallResult],
        failed: list[int],
        state: _State,
    ) -> Generator[Effect, Any, tuple[Ballot, list[DecisionRequest], list[CallResult]] | None]:
        """422 isolation (§5.6): drop the offending qids' families and re-send the failed calls once."""
        drop = self._droppable(ballot, [results[i].error for i in failed])
        if drop is None:
            state.failure = next(r.error for r in results if r.error is not None)
            return None
        s.notes.append(f"422 isolation: dropped {sorted(drop)} (logged as a bug)")
        state.dropped |= {(q.tool, q.path) for q in ballot.questions if q.qid in drop and q.tool and q.path}
        calls = [[qid for qid in call if qid not in drop] for call in ballot.call_plan()]
        kept_calls = [i for i, call in enumerate(calls) if call]
        trimmed = Ballot.model_validate({**ballot.model_dump(), "questions": [q for q in ballot.questions
                                         if q.qid not in drop], "calls": [calls[i] for i in kept_calls]})  # fmt: skip
        limits = self.round_limits()
        new_requests = trimmed.to_requests(self.model, id_mode=limits.id_mode,
                                           object_instructions=limits.instructions_as_object)  # fmt: skip
        # A kept call is re-sent when it failed, or when its wire ids changed (opaque ids renumber over the trimmed
        # Ballot, so an old response would be keyed by ids the trimmed Ballot no longer sends).
        old_ids, new_ids = ballot.wire_ids(limits.id_mode), trimmed.wire_ids(limits.id_mode)
        resend = [j for j, i in enumerate(kept_calls)
                  if i in failed or any(old_ids[q] != new_ids[q] for q in calls[i])]  # fmt: skip
        again: list[CallResult] = (yield _Ask([new_requests[j] for j in resend])) if resend else []
        s.usage.jev_calls += len(resend)
        merged = [results[i] for i in kept_calls]
        for j, result in zip(resend, again, strict=True):
            merged[j] = result
        if any(r.error is not None for r in merged):
            state.failure = next(r.error for r in merged if r.error is not None)
            return None
        return trimmed, new_requests, merged

    @staticmethod
    def _droppable(ballot: Ballot, errors: Sequence[BaseException | None]) -> set[str] | None:
        """The qids to drop (whole slot families), or ``None`` when the failure cannot be isolated: only slot
        questions (a tool and a path) are isolatable. Tool-level questions — ``tool``, ``reply``, and the gates
        ``T.authorized``, ``T.joint[.G]``, ``T.done_after`` — fail closed (P0): dropping a gate would remove a check
        instead of failing it (I5)."""
        wire_to_qid = {w: q for q, w in ballot.wire_ids("dotted").items()}
        wire_to_qid.update({w: q for q, w in ballot.wire_ids("opaque").items()})
        named: set[str] = set()
        for error in errors:
            if not isinstance(error, JevValidationError) or error.request_level() or not error.qids():
                return None
            named |= {wire_to_qid.get(w, w) for w in error.qids()}
        by_qid = ballot.by_qid
        if not named <= set(by_qid) or any(not (by_qid[q].tool and by_qid[q].path) for q in named):
            return None
        drop = set(named)
        for qid in named:
            question = by_qid[qid]
            drop |= {q.qid for q in ballot.questions if q.tool == question.tool and q.path == question.path}
        return drop

    def _record(
        self,
        s: _Session,
        record: RoundRecord,
        ballot: Ballot,
        requests: list[DecisionRequest],
        results: list[CallResult],
    ) -> None:
        """Fill a round record (Ballot, bodies), count usage and feed the token estimator."""
        record.ballot = ballot.to_doc() if self.store_bodies == "full" else None
        record.calls = []
        for request, result in zip(requests, results, strict=False):
            body = request.to_wire()
            record.calls.append(call_record(
                backend=self.backend_name, model=self.model, request=body, response=result.response,
                latency_ms=result.latency_ms, error=str(result.error) if result.error else None,
                store_bodies=self.store_bodies,
            ))  # fmt: skip
            if result.response is not None:
                s.usage.add(result.response)
                self.estimator.observe(body, result.response.usage, backend=self.backend_name, model=self.model)

    def _decode(self, s: _Session, state: _State) -> None:
        """Decode the merged answers and re-apply the session's user bindings."""
        rc = replace(state.plan.rc, questions=state.ballot.by_qid)
        decoded = decode_answers(state.ballot, state.answers, rc, pools=state.pools, dropped=state.dropped)
        state.decoded = self._apply_bindings(s, decoded, rc)

    def _apply_bindings(self, s: _Session, decoded: Decoded, rc: ResolveContext) -> Decoded:
        if s.tool_pick is not None:
            decoded = with_tool(decoded, *s.tool_pick)
        td = decoded.decision
        if td is None or not s.bindings:
            return decoded
        for name, (value, p, label) in s.bindings.items():
            if name in td.slots:
                td = bind_value(td, name, value, rc, p=p, label=label, prov={"user": True})
        return with_decision(decoded, td)

    def _policy_loop(
        self, s: _Session, state: _State
    ) -> Generator[Effect, Any, tuple[_State, PolicyResult, PolicyInput | None]]:
        """Evaluate; run internal rounds (re-plan after a speculation miss, widen, fill) while the policy asks."""
        while True:
            if state.failure is not None or state.decoded is None:
                inp = PolicyInput(failed=True, escalator=self._can_escalate(s))
                return state, evaluate(inp, self.policy), inp
            decoded = state.decoded
            if self._speculation_miss(s, decoded):
                s.respeculated = True
                s.confirmed_call = None  # the re-plan elects another tool: nothing shown on the card applies
                s.notes.append(f"speculation miss: {decoded.chosen} ({decoded.viability(decoded.chosen or '')[0]})")
                plan = self._compile(s, tool_choice=s.tool_choice, speculate_only=[decoded.chosen])
                state = yield from self._round(s, plan)
                continue
            inp = self._policy_input(s, decoded)
            result = evaluate(inp, self.policy)
            if result.action is Action.WIDEN:
                if not (yield from self._widen(s, state, result.bottleneck or "")):
                    s.widen_exhausted.add(result.bottleneck or "")
                continue
            if result.action is Action.FILL:
                if not (yield from self._fill(s, state, result.bottleneck or "")):
                    s.fills += 1
                continue
            if result.outcome in (Outcome.EXECUTE, Outcome.CONFIRM) and not inp.confirmed:
                questions = self._verify_questions(s, state)
                if questions:
                    yield from self._followup(s, state, questions, mode="verify")
                    continue
            return state, result, inp

    def _verify_questions(self, s: _Session, state: _State) -> list[BallotQuestion]:
        """Follow-up ``verify`` Nouls for a call about to be shown (§3.8.3): one per identity REF slot whose elected
        record was not typed by its key, bound by the user, or verified already (the first round verifies the
        best-anchored record, so this round runs only when Jev elected another one); tiers ``policy.probes.verify``."""
        td = state.decoded.decision if state.decoded is not None else None
        if td is None or td.tool.tier not in self.policy.probes.verify:
            return []
        out: list[BallotQuestion] = []
        for slot in td.tool.slots:
            result = td.slots.get(slot.name)
            hook = getattr(get_resolver(slot.kind), "verify", None)
            if not callable(hook) or slot.stakes != "identity" or len(slot.path) != 1 or result is None \
                    or result.is_bottom \
                    or slot.name in s.bindings or slot.name in td.verify:  # fmt: skip
                continue
            key = (td.tool.name, slot.path, value_key(result.value))
            if key in s.verified:
                continue
            asked = [q for q in state.ballot.questions if q.family == "slot" and q.tool == td.tool.name
                     and q.path == slot.path]  # fmt: skip
            option = next((o for q in asked for o in q.options if value_key(o.value) == key[2]), None)
            index = sum(q.family == "verify" and q.tool == td.tool.name and q.path == slot.path
                        for q in state.ballot.questions)  # fmt: skip
            question = hook(td.tool, slot, option, index) if option is not None else None
            if question is None:
                continue
            s.verified.add(key)
            out.append(question)
        return out

    def _speculation_miss(self, s: _Session, decoded: Decoded) -> bool:
        chosen = decoded.chosen
        if s.respeculated or chosen is None or chosen not in self.catalog or chosen in decoded.tools:
            return False
        viable, _ = decoded.viability(chosen)
        return viable in ("ok", "budget") and decoded.tool_dist.get(chosen, 0.0) >= self.policy.tool.min_p

    def _can_escalate(self, s: _Session) -> bool:
        return self.escalator is not None and not s.escalated

    def _policy_input(self, s: _Session, decoded: Decoded) -> PolicyInput:
        td = decoded.decision
        comp = self._composition(td)
        widen_ok: list[str] = []
        fill_ok: list[str] = []
        if td is not None:
            for slot in td.tool.slots:
                resolver = get_resolver(slot.kind)
                if callable(getattr(resolver, "widen", None)) and s.widen_rounds < self.policy.widen.max_rounds \
                        and slot.name not in s.widen_exhausted:  # fmt: skip
                    widen_ok.append(slot.name)
                if self._fillable(s, slot):
                    fill_ok.append(slot.name)
        confirmed = td is not None and s.confirmed_call is not None and _decode_key(td) == s.confirmed_call
        return policy_input(decoded, comp, escalator=self._can_escalate(s), widen_ok=widen_ok, fill_ok=fill_ok,
                            confirmed=confirmed, loop=s.in_loop)  # fmt: skip

    def _fillable(self, s: _Session, slot: SlotSpec) -> bool:
        if self.filler is None or s.fills >= 1 or Channel.GENERATED not in slot.channels:
            return False
        return slot.fallback == "fill" or (slot.fallback is None and slot.stakes == "content")

    def _composition(self, td: ToolDecode | None) -> Composition | None:
        if td is None:
            return None
        return confidence(td.factors, td.tool.tier, J=td.joint, policy=self.policy, calibrators=self.calibrators)

    # -- internal rounds ----------------------------------------------------------------------------------------

    def _widen(self, s: _Session, state: _State, slot_name: str) -> Generator[Effect, Any, bool]:
        """One coverage round through the resolver's optional ``widen(tool, slot, pool, rc, stage)`` hook (stage
        ``bucket``, then ``hierarchy`` with the previous group distribution); ``False`` when no strategy remains."""
        td = state.decoded.decision if state.decoded is not None else None
        if td is None:
            return False
        slot = td.tool.slot(slot_name)
        key = (td.tool.name, slot.path)
        stage = WIDEN_STAGES[min(s.widen_rounds, len(WIDEN_STAGES) - 1)]
        slot_key = ResolveContext.slot_key(td.tool, slot)
        request = {"stage": stage, "round": s.widen_rounds + 1, "groups": _group_mass(state, td.tool.name, slot.path)}
        rc = replace(state.plan.rc, questions=state.ballot.by_qid, round=len(s.rounds) + 1, mode="widen",
                     widen={**state.plan.rc.widen, slot_key: request})  # fmt: skip
        pool = state.pools.get(key) or Pool(tool=td.tool.name, path=slot.path, kind=slot.kind)
        widened, questions = get_resolver(slot.kind).widen(td.tool, slot, pool, rc, stage)  # type: ignore[attr-defined]
        if not questions:
            return False
        s.widen_rounds += 1
        state.pools[key] = widened
        return (yield from self._followup(s, state, list(questions), mode="widen"))

    def _fill(self, s: _Session, state: _State, slot_name: str) -> Generator[Effect, Any, bool]:
        """One FILL: the Filler proposes, the candidates join the pool as ``generated``, Jev elects them. Questions
        identical to already-answered ones reuse their answers (FILL does not change the state)."""
        td = state.decoded.decision if state.decoded is not None else None
        if td is None:
            return False
        slot = td.tool.slot(slot_name)
        request = FillRequest(
            tool=td.tool.name, tool_description=td.tool.description, request=s.ctx.request, history=s.ctx.history,
            frozen=_frozen_arguments(td, slot_name),
            slots={slot_name: strip_xjev(slot.json_schema)},
            observations=[ObservationPreview.of(o) for o in s.ctx.all_observations()],
        )  # fmt: skip
        proposals: list[FillCandidate] = yield _Fill(request)
        s.usage.llm_calls += 1
        s.fills += 1
        extra = [Candidate(value=p.values[slot_name], text=str(p.values[slot_name]), channel=Channel.GENERATED,
                           prov={"source": "filler", "index": i})
                 for i, p in enumerate(proposals) if slot_name in p.values]  # fmt: skip
        resolver = get_resolver(slot.kind)
        rc = replace(state.plan.rc, questions={}, mode="fill",
                     injected={**state.plan.rc.injected, ResolveContext.slot_key(td.tool, slot): extra})  # fmt: skip
        pool = inject_candidates(resolver.pool(td.tool, slot, rc), extra, slot, self.limits.label_max)
        state.pools[(td.tool.name, slot.path)] = pool
        questions = resolver.questions(td.tool, slot, pool, rc)
        asked = {q.qid: q for q in state.ballot.questions if q.tool == td.tool.name and q.path == slot.path}
        fresh = [q for q in questions if not _same_question(asked.get(q.qid), q)]
        for question in fresh:
            state.answers.pop(question.qid, None)
        state.ballot = _with_slot_questions(state.ballot, td.tool.name, slot.path, questions)
        if not fresh:
            self._decode(s, state)
            return True
        return (yield from self._followup(s, state, fresh, mode="fill"))

    def _followup(
        self, s: _Session, state: _State, questions: list[BallotQuestion], *, mode: Mode
    ) -> Generator[Effect, Any, bool]:
        """Ask a same-state round and decode it together with the earlier answers (I4 holds: same state)."""
        ballot = followup_ballot(state.plan.ballot, questions, mode=mode, limits=self.round_limits())
        answered = yield from self._ask(s, ballot, state, merged=True)
        if answered is None:
            return True
        state.ballot = merge_ballots(state.ballot, answered)
        self._decode(s, state)
        return True

    # -- escalation ---------------------------------------------------------------------------------------------

    def _escalate(self, s: _Session, state: _State, result: PolicyResult, inp: PolicyInput | None) -> Flow:
        """Hand the turn to the Escalator; a proposed call is re-bound and gated by one Jev round (§4.7)."""
        s.escalated = True
        preliminary = self._build(s, state, result, inp)
        answer = yield _Escalate(_messages(s.ctx), self.catalog.to_openai(), preliminary)
        s.usage.llm_calls += 1
        proposed = ProposedCall.coerce(answer)
        if proposed is None or proposed.name not in self.catalog:
            text = answer if isinstance(answer, str) else ""
            if state.failure is not None:  # P0: the handoff itself is the outcome
                return (yield from self._finish(s, state, result, inp, content=text))
            outcome = result.model_copy(update={"outcome": Outcome.ABSTAIN, "reason": "escalated_text"})
            return (yield from self._finish(s, state, outcome, inp, content=text))
        if state.failure is not None:
            s.notes.append("escalator proposed a call while Jev is unavailable: not bound")
            return (yield from self._finish(s, state, result, inp, proposed=proposed))
        tool = self.catalog.get(proposed.name)
        extra = {(tool.name, slot.path): [Candidate(value=value, text="Proposed by the escalation assistant.",
                                                    channel=Channel.GENERATED, prov={"source": "escalator"})]
                 for name, value in proposed.arguments.items() if name in tool.slot_names
                 for slot in [tool.slot(name)]}  # fmt: skip
        plan = self._compile(
            s, tool_choice={"type": "function", "function": {"name": tool.name}}, extra_candidates=extra
        )
        gate = yield from self._round(s, plan)
        gate, gated, gate_inp = yield from self._policy_loop(s, gate)
        return (yield from self._finish(s, gate, gated, gate_inp))

    # -- resume -------------------------------------------------------------------------------------------------

    def _pending(self, pending: Pending | str) -> Pending:
        if isinstance(pending, Pending):
            return pending
        try:
            return self.pendings[pending]
        except KeyError:
            raise KeyError(f"unknown pending id {pending!r}") from None

    def _resume_flow(
        self, pending: Pending | str, selection: str | None, reply: str | None, context: Context | None
    ) -> Flow:
        handle = self._pending(pending)
        messages = [Turn.model_validate(m) for m in handle.state.get("messages", [])]
        live = self._live.get(handle.pending_id)
        if context is not None and live is not None and context.requester_sha256 != live.session.ctx.requester_sha256:
            raise PendingScopeError(f"{handle.pending_id} was raised for a different requester (user profile or "
                                    "sources); resume it with that requester's context")  # fmt: skip
        # Without an explicit context, a live handle resumes in the context of the decision that raised it (never
        # the router default, which may belong to someone else).
        base = context if context is not None else live.session.ctx if live is not None else self.context
        ctx = base.with_messages(messages)
        if selection is None and reply is not None:
            selection = parse_short_reply(reply, handle.state.get("options", []))
        loop = bool(handle.state.get("loop"))
        tool = handle.state.get("tool")
        unknown = bool(tool) and str(tool) not in self.catalog  # resumed on a router serving another tool list
        if handle.expired() or unknown or (selection is not None and live is None):
            s = _Session(ctx=ctx if reply is None else _with_reply(ctx, handle, reply), mode="loop" if loop else "turn",
                         tool_choice="auto", resumed_from=handle.pending_id)  # fmt: skip
            s.notes.append(f"pending tool {tool!r} is not in the tool list: recompiled" if unknown
                           else "pending expired or not in memory: recompiled")  # fmt: skip
            return (yield from self._flow(s))
        if selection is not None:
            assert live is not None
            return (yield from self._click(handle, live, selection, ctx))
        if reply is None:
            raise ValueError("resume() needs a selection or a reply")
        s = _Session(ctx=_with_reply(ctx, handle, reply), mode="resume", tool_choice="auto",
                     resumed_from=handle.pending_id, loop=loop)  # fmt: skip
        extra = self._reply_candidates(handle, reply)
        for key, carried in (_carried_candidates(live) if live is not None else {}).items():
            extra[key] = [*extra.get(key, []), *carried]
        return (yield from self._flow(s, pending=handle, extra=extra))

    def _click(self, handle: Pending, live: _Live, selection: str, ctx: Context) -> Flow:
        """A click: no Jev call; the clicked value is bound with ``p = 1`` (channel ``user``)."""
        try:
            action = handle.options[selection]
        except KeyError:
            raise ValueError(f"{selection!r} is not an option of {handle.pending_id}") from None
        s = replace(live.session, ctx=ctx, rounds=[], usage=_Usage(), notes=[], resumed_from=handle.pending_id,
                    bindings=dict(live.session.bindings), confirmed_call=None)  # fmt: skip
        state = replace(live.state, decoded=live.state.decoded)
        if action.action == "cancel":
            result = PolicyResult(outcome=Outcome.ABSTAIN, rule=RULE_NO_TOOL, reason="cancelled")
            return (yield from self._finish(s, None, result, None))
        if action.action == "tool":
            fresh = _Session(ctx=ctx, mode=s.mode, tool_choice={"type": "function", "function": {"name": action.tool}},
                             resumed_from=handle.pending_id, loop=s.loop)  # fmt: skip
            return (yield from self._flow(fresh))
        if action.action == "open":
            return (yield from self._open(s, state, action.slot))
        pending_tool = live.tool or handle.state.get("tool")
        if action.action == "bind" and pending_tool and not _decided(state, str(pending_tool), action.slot):
            return (yield from self._bind_unspeculated(handle, live, str(pending_tool), action, ctx))
        complete_call = False
        if action.action == "confirm":
            s.confirmed_call = _call_key(handle.call) if handle.call is not None else None
        else:
            s.bindings[str(action.slot)] = (action.value, 1.0, action.label)
            complete_call = selection.startswith(("pick:", "alt:")) and live.tool is not None \
                and self.catalog.get(live.tool).tier >= Tier.EXTERNAL  # fmt: skip
            s.confirmed_call = None
        rc = replace(state.plan.rc, questions=state.ballot.by_qid)
        assert state.decoded is not None
        state.decoded = self._apply_bindings(s, state.decoded, rc)
        td = state.decoded.decision
        if complete_call and td is not None and td.tool.name == live.tool:
            s.confirmed_call = _decode_key(td)  # the option showed this complete call
        inp = self._policy_input(s, state.decoded)
        result = evaluate(inp, self.policy)
        return (yield from self._finish(s, state, result, inp, delayed=True))

    def _bind_unspeculated(
        self, handle: Pending, live: _Live, tool_name: str, action: PendingAction, ctx: Context
    ) -> Flow:
        """A menu pick (a grid value, §4.2.4) for a tool that was not speculated (P6, §5.1): the clicked value is
        the user's (p = 1, channel ``user``), but no other slot was asked yet, so one round asks the rest of the
        call with the tool named (no tool question). The click is a binding, not a confirmation."""
        tool = self.catalog.get(tool_name)
        slot = tool.slot(str(action.slot))
        named: ToolChoice = {"type": "function", "function": {"name": tool.name}}
        fresh = _Session(ctx=ctx, mode=live.session.mode, tool_choice=named, resumed_from=handle.pending_id,
                         loop=live.session.loop)  # fmt: skip
        fresh.bindings[slot.name] = (action.value, 1.0, action.label)
        picked = Candidate(value=action.value, text=f"Chosen by the user from a menu: {display_value(action.value)}.",
                           channel=Channel.USER, prov={"source": "click"})  # fmt: skip
        return (yield from self._flow(fresh, extra={(tool.name, slot.path): [picked]}))

    def _open(self, s: _Session, state: _State, slot_name: str | None) -> Flow:
        """``change``/``Something else``: ask the open question of the slot (no Jev call)."""
        result = PolicyResult(outcome=Outcome.CLARIFY, rule=RULE_SLOT_SHAPE, bottleneck=slot_name, shape="missing",
                              ask="open", reason="change")  # fmt: skip
        inp = self._policy_input(s, state.decoded) if state.decoded is not None else None
        return (yield from self._finish(s, state, result, inp))

    def _apply_reply(self, s: _Session, state: _State, pending: Pending) -> None:
        """Resume round: the ``reply`` Choice replaces the clarified factor (or confirms / cancels / picks a tool);
        a passthrough slot binds the reply verbatim."""
        assert state.decoded is not None
        choice, p, _ = decode_reply(state.ballot, state.answers)
        open_slot = pending.state.get("open_slot")
        tool = pending.state.get("tool")
        if open_slot and tool and self._passthrough(tool, open_slot):
            s.bindings[open_slot] = (s.ctx.request, 1.0, None)
        elif choice == CANCEL and p >= 0.5:
            s.tool_pick = ("NO_TOOL", max(p, state.decoded.tool_dist.get("NO_TOOL", 0.0)))
        elif choice in pending.options and p >= 0.5:
            action = pending.options[choice]
            if action.action == "bind" and action.slot:
                s.bindings[action.slot] = (action.value, p, action.label)
            elif action.action == "confirm" and pending.call is not None:
                s.confirmed_call = _call_key(pending.call)  # confirms the card's call only (re-checked per decode)
            elif action.action == "tool" and action.tool:
                s.tool_pick = (action.tool, p)
        rc = replace(state.plan.rc, questions=state.ballot.by_qid)
        state.decoded = self._apply_bindings(s, state.decoded, rc)

    def _passthrough(self, tool: str, slot_name: str) -> bool:
        slot = self.catalog.get(tool).slot(slot_name)
        return slot.fallback == "passthrough" and slot.stakes in ("content", "cosmetic")

    def _reply_candidates(self, handle: Pending, reply: str) -> dict[PoolKey, list[Candidate]]:
        """``fallback: ask`` content/cosmetic slots: the reply (and the reply minus a leading "say"/"tell her")
        become ``user`` candidates elected in the resume round."""
        tool, open_slot = handle.state.get("tool"), handle.state.get("open_slot")
        if not tool or not open_slot:
            return {}
        slot = self.catalog.get(tool).slot(open_slot)
        if slot.stakes not in ("content", "cosmetic") or slot.fallback == "passthrough":
            return {}
        texts = [reply.strip()]
        lowered = texts[0].lower()
        texts += [texts[0][len(p) :].strip() for p in REPLY_PREFIXES if lowered.startswith(p)]
        return {(tool, slot.path): [Candidate(value=t, text=t, channel=Channel.USER, prov={"source": "reply"})
                                    for t in dict.fromkeys(texts) if t]}  # fmt: skip

    # -- TOCTOU -------------------------------------------------------------------------------------------------

    def default_revalidate(self, td: ToolDecode, ctx: Context) -> list[str]:
        """TOCTOU (§3.8.5): registry-bound values — including registry values the user clicked or picked (channel
        ``user``, origin ``registry``) — still exist with the same label, constraints and ``@checks`` hold with fresh
        attributes and the current time, and the arguments still validate."""
        rc = ResolveContext(ctx=ctx, catalog=self.catalog, policy=self.policy, limits=self.limits)
        problems: list[str] = []
        attrs: dict[str, Mapping[str, Any]] = {}
        for slot in td.tool.slots:
            result = td.slots[slot.name]
            attrs[slot.name] = result.attrs
            if result.is_bottom or value_origin(result) is not Channel.REGISTRY or result.prov.get("default"):
                continue
            pool = get_resolver(slot.kind).pool(td.tool, slot, rc)
            fresh = _membership(pool, result.value, result.label)
            if isinstance(fresh, str):
                problems.append(f"{slot.name}: {fresh}")
            elif fresh is not None:
                attrs[slot.name] = fresh
        env = ConstraintContext(attrs=attrs, now=ctx.current_time(), context=ctx)
        problems += [f"constraint {c.expr!r} no longer holds" for c in td.tool.constraints
                     if c.check(td.arguments, env) is not True]  # fmt: skip
        problems += [str(e) for e in validate(td.arguments, td.tool.parameters)]
        return problems

    # -- finishing ----------------------------------------------------------------------------------------------

    def _finish(
        self,
        s: _Session,
        state: _State | None,
        result: PolicyResult,
        inp: PolicyInput | None,
        *,
        content: str | None = None,
        proposed: ProposedCall | None = None,
        delayed: bool = False,
    ) -> Flow:
        """TOCTOU for delayed executes, the text LLM for abstains, then the Decision (and pending handle)."""
        td = state.decoded.decision if state is not None and state.decoded is not None else None
        if result.outcome is Outcome.EXECUTE and td is not None and delayed and not s.replanned:
            problems = self.revalidate(td, s.ctx)
            if problems:
                fresh = _Session(ctx=s.ctx, mode=s.mode, tool_choice=s.tool_choice, resumed_from=s.resumed_from,
                                 replanned=True, loop=s.loop)  # fmt: skip
                fresh.notes += [f"TOCTOU: {p}" for p in problems] + ["re-planned"]
                return (yield from self._flow(fresh))
        if result.outcome is Outcome.ABSTAIN and content is None and self.text_llm is not None:
            content = yield _Text(_messages(s.ctx))
            s.usage.llm_calls += 1
        return self._build(s, state, result, inp, content=content, proposed=proposed)

    def _build(
        self,
        s: _Session,
        state: _State | None,
        result: PolicyResult,
        inp: PolicyInput | None,
        *,
        content: str | None = None,
        proposed: ProposedCall | None = None,
    ) -> Decision:
        decoded = state.decoded if state is not None else None
        td = decoded.decision if decoded is not None else None
        ids = self._ids(s)
        call = None
        if td is not None:
            call = ToolCall.build(td.tool.name, jsonable(td.arguments), trace_id=ids.trace_id)
        elif proposed is not None:
            call = ToolCall.build(proposed.name, jsonable(proposed.arguments), trace_id=ids.trace_id)
        if result.outcome is Outcome.EXECUTE and (td is None or not td.complete):
            result = result.model_copy(update={"outcome": Outcome.CLARIFY, "ask": "open"})  # defensive: never guess
        comp = self._composition(td)
        prompt, actions = self._prompt(s, state, result, td, inp)
        pending = self._pending_handle(s, state, ids, result, call, prompt, actions, td)
        trace = self._trace(s, state, ids, result, inp, comp, call)
        decision = Decision(
            decision_id=ids.decision_id, trace_id=ids.trace_id, outcome=result.outcome, rule=result.rule, call=call,
            tool_calls=[call] if result.outcome is Outcome.EXECUTE and call is not None else [],
            confidence=self._confidence(td, comp), bottleneck=self._bottleneck(result, inp),
            slots=_slot_reports(td), gates=dict(td.gates) if td is not None else {},
            flags=_flags(td, inp), prompt=prompt, pending=pending, rounds=len(s.rounds), usage=s.usage.to_model(),
            content=content, trace=trace,
        )  # fmt: skip
        if pending is not None and state is not None:
            self._remember(pending, _Live(session=s, state=state, tool=td.tool.name if td else None))
        return decision

    def _ids(self, s: _Session) -> DecisionIds:
        parts = [self.policy.sha256, s.ctx.sha256, s.resumed_from or ""]
        parts += [c.request_sha256 + (c.response_sha256 or "") for r in s.rounds for c in r.calls]
        parts += [f"{k}={value_key(v[0])}" for k, v in sorted(s.bindings.items())]
        parts.append("confirmed" if s.confirmed_call is not None else "")
        return DecisionIds.derive(*parts)

    def _remember(self, pending: Pending, live: _Live) -> None:
        """Hold a pending handle and its click-resume state: expired handles are dropped, then the oldest beyond
        :data:`LIVE_MAX` (bounded memory for long-running routers)."""
        now = datetime.now(timezone.utc)
        for pending_id in [k for k, p in self.pendings.items() if p.expired(now)]:
            self._forget(pending_id)
        self._forget(pending.pending_id)
        self.pendings[pending.pending_id] = pending
        self._live[pending.pending_id] = live
        while len(self.pendings) > LIVE_MAX:
            self._forget(next(iter(self.pendings)))

    def _forget(self, pending_id: str) -> None:
        self.pendings.pop(pending_id, None)
        self._live.pop(pending_id, None)

    def _confidence(self, td: ToolDecode | None, comp: Composition | None) -> Confidence | None:
        if td is None or comp is None or comp.C is None:
            return None
        tier = td.tool.tier
        return Confidence(call=comp.C, tier=tier.value, composition=comp.rule or "", W=comp.W, PI=comp.PI, L=comp.L,
                          J=comp.J, calibrated=comp.calibrated, execute_at=self.policy.execute_at(tier),
                          confirm_at=self.policy.tier(tier).confirm)  # fmt: skip

    def _bottleneck(self, result: PolicyResult, inp: PolicyInput | None) -> Bottleneck | None:
        if result.bottleneck is not None and result.shape is not None:
            return Bottleneck(slot=result.bottleneck, shape=result.shape)
        if result.outcome is Outcome.CONFIRM and inp is not None:
            weakest = inp.bottleneck()
            if weakest is not None:
                return Bottleneck(slot=weakest.name, shape=bottleneck_shape(weakest, self.policy)[0])
        return None

    # -- prompts ------------------------------------------------------------------------------------------------

    def _prompt(
        self, s: _Session, state: _State | None, result: PolicyResult, td: ToolDecode | None, inp: PolicyInput | None
    ) -> tuple[Prompt | None, Actions]:
        ask = result.ask
        if result.outcome not in (Outcome.CONFIRM, Outcome.CLARIFY, Outcome.REFUSE) or ask is None:
            return None, {}
        if ask == "tool_menu" and len(result.tools) == 2:
            return tool_menu(self.catalog.get(result.tools[0]), self.catalog.get(result.tools[1]))
        if ask == "notice":
            tool = td.tool if td is not None else self._chosen_tool(state)
            return (refuse_notice(tool, _source(state, result)), {}) if tool is not None else (None, {})
        slot = self._slot_spec(state, td, result.bottleneck)
        if ask == "confirm" and td is not None:
            return confirm_card(td.tool, _bindings(td), self._alternatives(td), bottleneck=_weakest(inp))
        if ask == "yes_no" and td is not None and slot is not None:
            return yes_no_menu(td.tool, slot)
        if ask == "menu" and td is not None and slot is not None:
            choices = _menu_choices(td, slot.name, self.policy)
            if choices:
                rc = replace(state.plan.rc, questions=state.ballot.by_qid) if state is not None else None
                rebound = [_bindings(bind_value(td, slot.name, c.value, rc)) for c in choices] if rc else None
                return clarify_menu(td.tool, slot, choices, bindings=_bindings(td), choice_bindings=rebound,
                                    complete_call=td.tool.tier >= Tier.EXTERNAL)  # fmt: skip
        if ask == "open" and slot is not None and result.reason != "change":
            grid = self._grid_menu(state, td, slot)
            if grid is not None:
                return grid
        return open_question(slot)

    def _grid_menu(self, state: _State | None, td: ToolDecode | None, slot: SlotSpec) -> tuple[Prompt, Actions] | None:
        """An open clarify becomes a menu when the slot's resolver offers ``clarify_values`` (quantity grids,
        §4.2.4: a required slot without a default that nothing stated). With the rest of the call decoded, external
        and critical options show the complete call, as on a clarify menu."""
        tool = td.tool if td is not None else self._chosen_tool(state)
        hook = getattr(get_resolver(slot.kind), "clarify_values", None)
        if tool is None or state is None or not callable(hook):
            return None
        rc = replace(state.plan.rc, questions=state.ballot.by_qid)
        values: list[Candidate] = hook(tool, slot, state.pools.get((tool.name, slot.path)), rc)
        if not values:
            return None
        choices = [Binding.of(c) for c in values]
        if td is not None and slot.name in td.slots:
            rebound = [_bindings(bind_value(td, slot.name, c.value, rc)) for c in values]
            return grid_menu(tool, slot, choices, bindings=_bindings(td), choice_bindings=rebound,
                             complete_call=tool.tier >= Tier.EXTERNAL)  # fmt: skip
        return grid_menu(tool, slot, choices)

    def _chosen_tool(self, state: _State | None) -> ToolSpec | None:
        chosen = state.decoded.chosen if state is not None and state.decoded is not None else None
        return self.catalog.get(chosen) if chosen is not None and chosen in self.catalog else None

    def _slot_spec(self, state: _State | None, td: ToolDecode | None, name: str | None) -> SlotSpec | None:
        tool = td.tool if td is not None else self._chosen_tool(state)
        if tool is None or name is None or name not in tool.slot_names:
            return None
        return tool.slot(name)

    def _alternatives(self, td: ToolDecode) -> list[AltChoice]:
        """Runner-ups for the confirm card: per non-cosmetic slot, its first whole-value runner-up with
        ``p ≥ alternatives_min(tier)``, shown by its label; at most 3. Text slots offer none: their accept Nouls
        are independent judgments, not a distribution with a runner-up (the §13.3 R2 card shows no body)."""
        threshold = self.policy.alternatives_min(td.tool.tier)
        alts: list[AltChoice] = []
        for name, result in td.slots.items():
            if result.stakes == "cosmetic" or result.is_bottom or result.kind == "text":
                continue
            runner = next((a for a in result.alternatives if a.part is None), None)
            if runner is not None and runner.p >= threshold:
                alts.append(AltChoice(slot=name, index=1, value=runner.value, display=runner.label or runner.display,
                                      p=runner.p, label=runner.label))  # fmt: skip
        return alts[:3]

    def _pending_handle(
        self,
        s: _Session,
        state: _State | None,
        ids: DecisionIds,
        result: PolicyResult,
        call: ToolCall | None,
        prompt: Prompt | None,
        actions: Actions,
        td: ToolDecode | None,
    ) -> Pending | None:
        if prompt is None or result.outcome not in (Outcome.CONFIRM, Outcome.CLARIFY):
            return None
        ballot_sha = state.ballot.sha256 if state is not None else ""
        responses = [c.response_sha256 for r in s.rounds for c in r.calls if c.response_sha256]
        chosen = self._chosen_tool(state)
        opened = prompt.kind == "open" or result.ask == "open"  # an open clarify, possibly shown as a grid menu
        tool = td.tool.name if td is not None else chosen.name if chosen is not None and opened else None
        return Pending.new(
            pending_id=ids.pending_id, decision_id=ids.decision_id, ballot_sha256=ballot_sha,
            response_sha256s=responses, call=call, options=actions,
            state={"messages": [t.model_dump(mode="json") for t in s.ctx.messages], "tool": tool,
                   "prompt_text": prompt.text, "options": [o.model_dump() for o in prompt.options],
                   "open_slot": result.bottleneck if opened else None, "rule": result.rule,
                   **({"loop": True} if s.in_loop else {})},
        )  # fmt: skip

    # -- trace --------------------------------------------------------------------------------------------------

    def _trace(
        self,
        s: _Session,
        state: _State | None,
        ids: DecisionIds,
        result: PolicyResult,
        inp: PolicyInput | None,
        comp: Composition | None,
        call: ToolCall | None,
    ) -> Trace:
        return build_trace(
            ids=ids, policy=self.policy, catalog=self.catalog, context=s.ctx, rounds=s.rounds,
            decoded=state.decoded if state is not None else None, ballot=state.ballot if state is not None else None,
            comp=comp, result=result, inp=inp, call=call, flags=_flags(state.decoded.decision if state is not None and
            state.decoded is not None else None, inp), resumed_from=s.resumed_from, notes=s.notes,
        )  # fmt: skip


# --------------------------------------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------------------------------------


def _ms(start: float) -> int:
    return int((time.perf_counter() - start) * 1000)


def _messages(ctx: Context) -> list[dict[str, Any]]:
    """The conversation as OpenAI-style messages (for the Escalator and the text LLM)."""
    out: list[dict[str, Any]] = []
    for turn in ctx.messages:
        message: dict[str, Any] = {"role": turn.role, "content": turn.content if turn.content is not None
                                   else turn.text}  # fmt: skip
        if turn.tool_calls:
            message["tool_calls"] = list(turn.tool_calls)
        if turn.tool_call_id:
            message["tool_call_id"] = turn.tool_call_id
        out.append(message)
    return out


def _with_reply(ctx: Context, handle: Pending, reply: str) -> Context:
    """The conversation after a free-text reply: the card as an assistant turn, then the reply (§3.8.5)."""
    turns = [*ctx.messages, Turn(role="assistant", text=str(handle.state.get("prompt_text", ""))),
             Turn(role="user", text=reply, content=reply)]  # fmt: skip
    return ctx.model_copy(update={"messages": turns})


def _with_slot_questions(
    ballot: Ballot, tool: str, path: tuple[str, ...], questions: Sequence[BallotQuestion]
) -> Ballot:
    """The decode view with one slot's questions replaced (after its pool changed)."""
    kept = [q for q in ballot.questions if not (q.tool == tool and q.path == path)]
    merged = [*kept, *questions]
    return Ballot.model_validate({**ballot.model_dump(), "questions": merged, "calls": [[q.qid for q in merged]]})


def _carried_candidates(live: _Live) -> dict[PoolKey, list[Candidate]]:
    """Free-text resume: the pool candidate behind each bound value of the pending call (its channel, description
    and late recipe unchanged). The state's request is now the reply, so without them the resume round could not
    re-elect values that came from the original request (a body template, a start time)."""
    td = live.state.decoded.decision if live.state.decoded is not None else None
    if td is None:
        return {}
    carried: dict[PoolKey, list[Candidate]] = {}
    for slot in td.tool.slots:
        result = td.slots.get(slot.name)
        pool = live.state.pools.get((td.tool.name, slot.path))
        if result is None or result.is_bottom or pool is None:
            continue
        entry = result.entries.get(value_key(result.value))
        keys = {value_key(result.value)}
        if entry is not None and entry.late and "template" in entry.late:
            keys.add(value_key(entry.late["template"]))
        found = [c.model_copy(update={"label": ""}) for c in pool.candidates if value_key(c.value) in keys]
        if found:
            carried[(td.tool.name, slot.path)] = found
    return carried


def _group_mass(state: _State, tool: str, path: tuple[str, ...]) -> dict[str, float]:
    """The ``group`` Choice mass of an earlier widen stage of one slot (``{group value: p}``)."""
    mass: dict[str, float] = {}
    for question in state.ballot.questions:
        if question.family == "group" and question.tool == tool and question.path == path:
            labels = choice_mass(question, state.answers.get(question.qid))
            for option in question.options:
                group = option.value if isinstance(option.value, str) else value_key(option.value)
                mass[group] = labels[option.label]
    return mass


def _frozen_arguments(td: ToolDecode, slot_name: str) -> dict[str, Any]:
    """The decided arguments shown to the Filler, never a secret (§14: secrets are never sent; the Filler must not
    change frozen arguments, so it gains nothing from one)."""
    return {name: value for name, value in td.arguments.items()
            if name != slot_name and td.tool.slot(name).kind != "secret"
            and not (name in td.slots and td.slots[name].prov.get("secret"))}  # fmt: skip


def _same_question(old: BallotQuestion | None, new: BallotQuestion) -> bool:
    return old is not None and old.to_wire() == new.to_wire()


def _membership(pool: Pool, value: Any, label: str | None) -> Mapping[str, Any] | str | None:
    """Fresh attributes of a still-present registry value; a problem string when it vanished or was relabelled;
    ``None`` when the pool cannot tell (composite values over an empty pool)."""
    items = value if isinstance(value, list) else [value]
    candidates = [*pool.candidates, *pool.blocked]
    if isinstance(value, list) and not candidates:
        return None
    by_key = {value_key(c.value): c for c in candidates}
    missing = [item for item in items if value_key(item) not in by_key]
    if missing:
        return f"{missing[0]!r} no longer available"
    if not isinstance(value, list):
        match = by_key[value_key(value)]
        if label and match.label and match.label != label:
            return f"label changed to {match.label!r}"
        return dict(match.attrs)
    return None


def _call_key(call: ToolCall) -> tuple[str, str]:
    """``(tool, canonical arguments digest)`` of a call shown to the user."""
    return call.name, sha256_of(jsonable(call.arguments))


def _decode_key(td: ToolDecode) -> tuple[str, str]:
    """The same key for a decoded call (what :meth:`Router._build` would emit)."""
    return td.tool.name, sha256_of(jsonable(td.arguments))


def _bindings(td: ToolDecode) -> dict[str, Binding]:
    return {name: Binding.of_result(r) for name, r in td.slots.items() if not r.is_bottom}


def _decided(state: _State, tool: str, slot: str | None) -> bool:
    """Whether the state decoded ``tool`` (it was speculated) with ``slot`` among its results."""
    td = state.decoded.decision if state.decoded is not None else None
    return td is not None and td.tool.name == tool and slot is not None and slot in td.slots


def _weakest(inp: PolicyInput | None) -> str | None:
    weakest = inp.bottleneck() if inp is not None else None
    return weakest.name if weakest is not None else None


def _menu_choices(td: ToolDecode, slot_name: str, policy: Policy) -> list[Binding]:
    """The top real values of the bottleneck slot (whole-slot values), up to ``ambiguous_k``."""
    result = td.slots[slot_name]
    keys = sorted((k for k in result.values if result.dist.get(k, 0.0) > 0), key=lambda k: -result.dist[k])
    choices = []
    for key in keys[: policy.shapes.ambiguous_k]:
        entry = result.entries.get(key)
        shown = (entry.label or entry.display) if entry is not None else None
        choices.append(Binding.of_value(result.values[key], display=shown, label=entry.label if entry else None,
                                        attrs=entry.attrs if entry else None))  # fmt: skip
    return choices


def _source(state: _State | None, result: PolicyResult) -> str:
    """Where refused instructions came from (the observation step of a blocked candidate, if known)."""
    if state is not None and result.bottleneck and state.decoded is not None and state.decoded.chosen:
        tool = state.decoded.chosen
        for (name, path), pool in state.pools.items():
            if name == tool and path and path[0] == result.bottleneck:
                for candidate in pool.blocked:
                    step = candidate.prov.get("step")
                    if step is not None:
                        return f"the result of step {step}"
    return "tool results"


def _slot_reports(td: ToolDecode | None) -> dict[str, SlotReport]:
    if td is None:
        return {}
    reports: dict[str, SlotReport] = {}
    for name, result in td.slots.items():
        if result.is_bottom and result.value.value == "⊥omit":
            continue
        reports[name] = SlotReport(
            value=None if result.is_bottom else jsonable(result.value),
            p=result.factor if result.factor is not None else result.p, stakes=result.stakes,
            channel=result.channel.value if result.channel is not None and not result.is_bottom else None,
            alternatives=[AlternativeReport(value=jsonable(a.value), p=a.p, part=a.part) for a in result.alternatives],
        )  # fmt: skip
    return reports


def _flags(td: ToolDecode | None, inp: PolicyInput | None) -> list[str]:
    flags = list(inp.flags) if inp is not None else []
    if td is not None:
        flags += [f for r in td.slots.values() for f in r.flags]
    return list(dict.fromkeys(flags))


__all__ = ["CallResult", "Revalidator", "Router", "sha256_of"]
