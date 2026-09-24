"""The evaluation harness (spec §11.2): run labelled cases through a Router and score every decision.

``run(cases, router_factory, replays=1)`` builds one :class:`~jevtools.router.Router` per case (the factory decides
the backend, catalog, context and policy), decides the case ``replays`` times and turns each decision into an
:class:`~jevtools.eval.report.EvalRecord`:

- the outcome and rule, the proposed call and whether it is the gold call (per argument match mode);
- W, Π, L, J and the final C with the tier (inputs of risk–coverage curves and threshold tuning);
- **stage attribution** of a failure: Jev failure (``backend``), the gold tool not speculated (``plan``), a gold
  value missing from every pool (``extractor``: extractor, source or retrieval miss), gold offered but not elected
  (``model``), or the right call with a wrong outcome (``policy``);
- pool coverage of each gold argument (recall@K), clarify-menu usefulness, and per-question calibration records
  read off the trace (the Ballot tells which option is gold, the stored response gives the probability).

Nothing here depends on the backend being real: with :class:`~jevtools.backends.ScriptedBackend` or the simulator the
harness exercises the pipeline, and its numbers say nothing about Jev.
"""

from __future__ import annotations

import asyncio
import os
from collections.abc import Awaitable, Callable, Iterable, Mapping, Sequence
from typing import Any

from jevtools.ballot import Ballot, BallotOption, BallotQuestion
from jevtools.canonical import canonical_str, jsonable
from jevtools.confidence import IsotonicCalibrator
from jevtools.context import Context
from jevtools.decision import Decision
from jevtools.eval.dataset import CALL_OUTCOMES, EvalCase, Gold, load, same_value, template_matches
from jevtools.eval.report import EvalRecord, EvalReport, PoolHit, QuestionRecord, Stage, new_meta
from jevtools.policy import RULE_FAIL_CLOSED, Outcome, Policy
from jevtools.router import Router
from jevtools.spec.catalog import Catalog
from jevtools.trace import RoundRecord, Trace

RouterFactory = Callable[[EvalCase], Router]
"""Builds the Router a case runs on (backend, catalog, context, policy, calibrators)."""
Progress = Callable[[EvalRecord], None]

_TOOL_SENTINELS = frozenset({"NO_TOOL", "UNSUPPORTED", "DONE"})
_VALUE_FAMILIES = frozenset({"slot", "probe", "rev", "mention", "bucket"})
"""Choice families whose options are candidate values of one slot."""


# --------------------------------------------------------------------------------------------------------------------
# Running
# --------------------------------------------------------------------------------------------------------------------


def _as_factory(router_factory: RouterFactory | Router) -> RouterFactory:
    """A factory; a plain :class:`Router` serves every case (its own context and catalog)."""
    if isinstance(router_factory, Router):
        router = router_factory
        return lambda case: router
    return router_factory


def run(
    cases: Iterable[EvalCase],
    router_factory: RouterFactory | Router,
    *,
    replays: int = 1,
    keep_traces: bool = True,
    raise_errors: bool = False,
    progress: Progress | None = None,
    meta: Mapping[str, Any] | None = None,
) -> EvalReport:
    """Decide every case ``replays`` times and score the decisions.

    ``router_factory`` builds each case's router (a plain :class:`Router` serves every case). One router is built
    per case and reused across its replays (decisions do not share state). An exception while
    deciding is recorded as stage ``error`` unless ``raise_errors``. ``keep_traces=False`` drops traces and decision
    documents from the records (the per-question calibration records are extracted either way).
    """
    if replays < 1:
        raise ValueError("replays must be >= 1")
    factory = _as_factory(router_factory)
    records: list[EvalRecord] = []
    first_router: Router | None = None
    for case in cases:
        router = factory(case)
        first_router = first_router or router
        for replay in range(replays):
            record = evaluate_case(case, router, replay=replay, keep_traces=keep_traces, raise_errors=raise_errors)
            records.append(record)
            if progress is not None:
                progress(record)
    return EvalReport(meta=_meta(first_router, replays, meta), records=records)


async def arun(
    cases: Iterable[EvalCase],
    router_factory: RouterFactory | Router,
    *,
    replays: int = 1,
    concurrency: int = 8,
    keep_traces: bool = True,
    raise_errors: bool = False,
    meta: Mapping[str, Any] | None = None,
) -> EvalReport:
    """Async :func:`run`: up to ``concurrency`` decisions in flight (``Router.adecide``)."""
    if replays < 1:
        raise ValueError("replays must be >= 1")
    items = list(cases)
    factory = _as_factory(router_factory)
    routers = [factory(case) for case in items]
    gate = asyncio.Semaphore(max(1, concurrency))

    async def one(case: EvalCase, router: Router, replay: int) -> EvalRecord:
        async with gate:
            return await aevaluate_case(case, router, replay=replay, keep_traces=keep_traces,
                                        raise_errors=raise_errors)  # fmt: skip

    jobs: list[Awaitable[EvalRecord]] = [
        one(case, router, replay) for case, router in zip(items, routers, strict=True) for replay in range(replays)
    ]
    records = list(await asyncio.gather(*jobs))
    return EvalReport(meta=_meta(routers[0] if routers else None, replays, meta), records=records)


def _meta(router: Router | None, replays: int, extra: Mapping[str, Any] | None) -> dict[str, Any]:
    fields: dict[str, Any] = {"replays": replays}
    if router is not None:
        fields.update(policy_version=router.policy.version, policy_sha256=router.policy.sha256,
                      backend=router.backend_name, model=router.model)  # fmt: skip
    return new_meta(**fields, **dict(extra or {}))


def evaluate_case(
    case: EvalCase, router: Router, *, replay: int = 0, keep_traces: bool = True, raise_errors: bool = False
) -> EvalRecord:
    """Decide one case once and score the decision."""
    try:
        decision = router.decide(case.messages, tool_choice=case.tool_choice or "auto")
    except Exception as exc:  # noqa: BLE001 - recorded as stage "error"
        if raise_errors:
            raise
        return error_record(case, exc, replay=replay)
    return score_decision(case, decision, replay=replay, locale=router.context.locale, keep_traces=keep_traces)


async def aevaluate_case(
    case: EvalCase, router: Router, *, replay: int = 0, keep_traces: bool = True, raise_errors: bool = False
) -> EvalRecord:
    """Async :func:`evaluate_case`."""
    try:
        decision = await router.adecide(case.messages, tool_choice=case.tool_choice or "auto")
    except Exception as exc:  # noqa: BLE001 - recorded as stage "error"
        if raise_errors:
            raise
        return error_record(case, exc, replay=replay)
    return score_decision(case, decision, replay=replay, locale=router.context.locale, keep_traces=keep_traces)


def error_record(case: EvalCase, exc: BaseException, *, replay: int = 0) -> EvalRecord:
    """The record of a case whose decision raised."""
    return EvalRecord(
        case_id=case.id, replay=replay, tags=list(case.tags), outcome="error", rule="harness.error",
        outcomes_ok=[o.value for o in case.gold.outcomes_ok], gold_tool=case.gold.tool, stage="error",
        stage_detail=type(exc).__name__, error=f"{type(exc).__name__}: {exc}",
    )  # fmt: skip


# --------------------------------------------------------------------------------------------------------------------
# Router factories
# --------------------------------------------------------------------------------------------------------------------


def router_factory_for(
    backend: Any,
    *,
    policy: Policy | None = None,
    context: Context | None = None,
    calibrators: Mapping[str, IsotonicCalibrator] | None = None,
    tools: Sequence[Mapping[str, Any]] | None = None,
    **router_kw: Any,
) -> RouterFactory:
    """A factory that builds each case's Router from its own files: the case's catalog (or ``tools``) compiled
    against its context's sources, the given backend, policy and calibrators. ``context`` is the base context the
    case's context document updates."""

    def factory(case: EvalCase) -> Router:
        ctx = case.build_context(context)
        raw = case.catalog_tools()
        if raw is None:
            if tools is None:
                raise ValueError(f"case {case.id!r} names no catalog and no default tools were given")
            raw = [dict(t) for t in tools]
        catalog = Catalog.from_openai(raw, sources=list(ctx.sources.values()))
        return Router(catalog, backend=backend, policy=policy, context=ctx, calibrators=calibrators, **router_kw)

    return factory


def run_dataset(path: str | os.PathLike[str], backend: Any, *, replays: int = 1, **kw: Any) -> EvalReport:
    """:func:`run` over a JSONL dataset whose cases name their catalog and context files (what ``jevtools eval``
    runs). ``keep_traces``/``raise_errors``/``progress`` go to :func:`run`; the other keywords (``policy``,
    ``calibrators``, ``context``, ``tools``, Router options) to :func:`router_factory_for`."""
    run_kw = {k: kw.pop(k) for k in ("keep_traces", "raise_errors", "progress") if k in kw}
    return run(load(path), router_factory_for(backend, **kw), replays=replays, meta={"dataset": str(path)},
               **run_kw)  # fmt: skip


# --------------------------------------------------------------------------------------------------------------------
# Scoring one decision
# --------------------------------------------------------------------------------------------------------------------


def score_decision(
    case: EvalCase,
    decision: Decision,
    *,
    replay: int = 0,
    locale: str | None = None,
    keep_traces: bool = True,
) -> EvalRecord:
    """Score a decision against the case's gold label (pure: no Jev call)."""
    gold = case.gold
    outcome = Outcome(decision.outcome)
    call = decision.call
    name = call.name if call is not None else None
    arguments = dict(call.arguments) if call is not None else None
    call_match = gold.call_matches(name, arguments)
    outcome_ok = gold.allows(outcome)
    trace = decision.trace if isinstance(decision.trace, Trace) else None
    analysis = analyze_trace(gold, trace) if trace is not None else TraceAnalysis()
    conf = decision.confidence
    menu, has_gold = _menu(gold, decision)
    record = EvalRecord(
        case_id=case.id, replay=replay, tags=list(case.tags), locale=locale, outcome=outcome.value,
        rule=decision.rule, outcomes_ok=[o.value for o in gold.outcomes_ok], gold_tool=gold.tool, tool=name,
        arguments=arguments, tier=conf.tier if conf is not None else None,
        composition=conf.composition if conf is not None else None, C=conf.call if conf is not None else None,
        W=conf.W if conf is not None else None, PI=conf.PI if conf is not None else None,
        L=conf.L if conf is not None else None, J=conf.J if conf is not None else None,
        calibrated=conf.calibrated if conf is not None else False, outcome_ok=outcome_ok, call_match=call_match,
        correct=outcome_ok and (call_match or outcome not in CALL_OUTCOMES), executed=outcome is Outcome.EXECUTE,
        wrong_if_executed=not (call_match and gold.tool is not None and gold.allows(Outcome.EXECUTE)),
        tool_speculated=analysis.tool_speculated, tool_top=analysis.tool_top, pool=analysis.pool, menu=menu,
        menu_has_gold=has_gold, rounds=decision.rounds, jev_calls=decision.usage.jev_calls,
        llm_calls=decision.usage.llm_calls, input_tokens=decision.usage.jev_input_tokens,
        cost_usd=decision.usage.cost_usd, flags=list(decision.flags), planted_hit=_planted(case, decision),
        questions=analysis.questions, decision=decision.to_doc() if keep_traces else None,
        trace=trace.to_doc() if keep_traces and trace is not None else None,
    )  # fmt: skip
    stage, detail = attribute(gold, decision, record)
    record.stage = stage
    record.stage_detail = detail
    return record


def _menu(gold: Gold, decision: Decision) -> tuple[list[dict[str, Any]] | None, bool | None]:
    """The clarify menu (bind/tool options) and whether it offers the gold value or tool."""
    pending = decision.pending
    if decision.outcome is not Outcome.CLARIFY or pending is None:
        return None, None
    options = [(oid, a) for oid, a in pending.options.items() if a.action in ("bind", "tool")]
    if not options:
        return None, None
    menu = [{"id": oid, "action": a.action, "slot": a.slot, "value": jsonable(a.value), "tool": a.tool}
            for oid, a in options]  # fmt: skip
    tool = decision.call.name if decision.call is not None else None
    hit = False
    for _, action in options:
        if action.action == "tool":
            hit = hit or (gold.tool is not None and action.tool == gold.tool)
        elif tool == gold.tool and action.slot in gold.slots:
            hit = hit or gold.accepts(str(action.slot), action.value)
    return menu, hit


def _planted(case: EvalCase, decision: Decision) -> bool | None:
    """For injection cases (``meta.planted``: values an attacker planted), whether any reached a shown call."""
    planted = case.meta.get("planted")
    if not planted:
        return None
    if decision.outcome not in CALL_OUTCOMES or decision.call is None:
        return False
    text = canonical_str(decision.call.arguments)
    return any(str(value) in text for value in planted)


def attribute(gold: Gold, decision: Decision, record: EvalRecord) -> tuple[Stage, str | None]:
    """The stage a failed decision is attributed to (spec §11.2): ``backend`` → ``plan`` → ``extractor`` →
    ``model`` → ``policy``; ``ok`` for correct decisions."""
    if record.correct:
        return "ok", None
    if decision.rule == RULE_FAIL_CLOSED:
        return "backend", None
    if gold.tool is not None and record.tool_speculated is False:
        return "plan", f"{gold.tool} not speculated"
    missing = [slot for slot, hit in record.pool.items() if not hit.in_pool]
    if gold.tool is not None and missing:
        return "extractor", missing[0]
    if record.call_match:
        return "policy", decision.rule
    if gold.tool is None:
        if record.tool_top is not None and record.tool_top not in _TOOL_SENTINELS:
            return "model", "tool"
        return "policy", decision.rule
    if record.tool_top is not None and record.tool_top != gold.tool:
        return "model", "tool"
    if record.tool is not None and record.tool != gold.tool:
        return "model", "tool"
    args = record.arguments or {}
    for slot in gold.slots:
        value = args.get(slot)
        if value is None and slot in decision.slots:
            value = decision.slots[slot].value
        if not gold.accepts(slot, value):
            return "model", slot
    return "policy", decision.rule


# --------------------------------------------------------------------------------------------------------------------
# Trace analysis: pools, tool speculation, per-question calibration records
# --------------------------------------------------------------------------------------------------------------------


class TraceAnalysis:
    """What the harness reads off a trace."""

    def __init__(
        self,
        *,
        questions: list[QuestionRecord] | None = None,
        pool: dict[str, PoolHit] | None = None,
        tool_speculated: bool | None = None,
        tool_top: str | None = None,
    ) -> None:
        self.questions = questions or []
        self.pool = pool or {}
        self.tool_speculated = tool_speculated
        self.tool_top = tool_top


def round_answers(ballot: Ballot, rnd: RoundRecord) -> dict[str, dict[str, Any]]:
    """The wire answers of one round keyed by qid (dotted or opaque wire ids mapped back)."""
    by_qid = ballot.by_qid
    opaque = {wire: qid for qid, wire in ballot.wire_ids("opaque").items()}
    out: dict[str, dict[str, Any]] = {}
    for call in rnd.calls:
        answers = (call.response or {}).get("answers") or {}
        for wire, answer in answers.items():
            qid = wire if wire in by_qid else opaque.get(wire)
            if qid is not None and isinstance(answer, Mapping):
                out[qid] = dict(answer)
    return out


def trace_rounds(trace: Trace) -> list[tuple[RoundRecord, Ballot, dict[str, dict[str, Any]]]]:
    """Every round with a stored Ballot, with its answers by qid."""
    out = []
    for rnd in trace.rounds:
        if rnd.ballot is None:
            continue
        ballot = Ballot.from_doc(rnd.ballot)
        out.append((rnd, ballot, round_answers(ballot, rnd)))
    return out


def analyze_trace(gold: Gold, trace: Trace) -> TraceAnalysis:
    """Pools, tool speculation, the tool Choice's top label and the calibration records of a decision's trace."""
    rounds = trace_rounds(trace)
    questions: list[QuestionRecord] = []
    speculated: bool | None = None
    tool_top: str | None = None
    for rnd, ballot, answers in rounds:
        if gold.tool is not None:
            flags = [t.speculated for t in ballot.tools if t.name == gold.tool]
            if flags:
                speculated = bool(speculated) or flags[0]
        tool_answer = answers.get("tool")
        if tool_top is None and tool_answer is not None:
            tool_top = _top(tool_answer)
        for q in ballot.questions:
            answer = answers.get(q.qid)
            if answer is not None:
                questions += question_calibration(gold, q, answer, ballot, round=rnd.round)
    return TraceAnalysis(questions=questions, pool=pool_hits(gold, [b for _, b, _ in rounds]),
                         tool_speculated=speculated, tool_top=tool_top)  # fmt: skip


def _probabilities(answer: Mapping[str, Any]) -> dict[str, float]:
    probs = answer.get("probabilities") or {}
    return {str(k): float(v) for k, v in probs.items()}


def _top(answer: Mapping[str, Any]) -> str | None:
    probs = _probabilities(answer)
    if probs:
        return max(probs, key=lambda k: probs[k])
    choice = answer.get("choice")
    return str(choice) if choice is not None else None


def _slot(q: BallotQuestion) -> str | None:
    return q.path[0] if len(q.path) == 1 else None


def _candidate_value(q: BallotQuestion) -> tuple[bool, Any]:
    """The candidate a Noul judges (accept: ``meta.candidate.value``; member: ``meta.value``)."""
    candidate = q.meta.get("candidate")
    if isinstance(candidate, Mapping) and "value" in candidate:
        return True, candidate["value"]
    if "value" in q.meta:
        return True, q.meta["value"]
    return False, None


def _gold_list(gold: Gold, slot: str) -> list[Any] | None:
    values = gold.values(slot)
    if values and isinstance(values[0], list):
        return list(values[0])
    return None


def _matches(gold: Gold, slot: str, value: Any) -> bool:
    return any(template_matches(value, g) for g in gold.values(slot))


def _label_truth(gold: Gold, q: BallotQuestion, label: str) -> bool | None:
    """Whether choosing ``label`` in a slot-value Choice is correct (``None`` when gold cannot tell)."""
    slot = _slot(q)
    if slot is None or gold.mode(slot) == "ignore":
        return None
    in_options = any(_matches(gold, slot, o.value) for o in q.options)
    if label in q.sentinels:
        spec = q.sentinels[label]
        if spec.decodes_to in ("missing", "omit"):
            return gold.accepts(slot, None)
        if spec.decodes_to == "default":
            if spec.late is not None or (spec.value is None and spec.display is None):
                return None
            return gold.accepts(slot, spec.value)
        if spec.decodes_to == "uncovered":
            default = next((s for s in q.sentinels.values() if s.decodes_to == "default"), None)
            via_default = default is not None and default.late is None and gold.accepts(slot, default.value)
            return not in_options and not via_default and not gold.accepts(slot, None)
        return None
    option = q.decode_label(label)
    if not isinstance(option, BallotOption):
        return None
    return _matches(gold, slot, option.value)


def question_calibration(
    gold: Gold, q: BallotQuestion, answer: Mapping[str, Any], ballot: Ballot, *, round: int = 1
) -> list[QuestionRecord]:
    """Calibration records of one answered question (top label, plus sentinel masses) whose truth the gold call
    determines. Questions of other tools than the gold one have no gold (their premise is counterfactual) except
    ``authorized`` (the user did not ask for them) and the ``tool`` Choice."""
    on_gold = q.tool is not None and q.tool == gold.tool
    base: dict[str, Any] = {"qid": q.qid, "family": q.family, "tool": q.tool, "primitive": q.primitive,
                            "round": round, "on_gold_tool": on_gold}  # fmt: skip
    out: list[QuestionRecord] = []
    if q.primitive == "choice":
        probs = _probabilities(answer)
        top = _top(answer)
        if not probs or top is None:
            return []
        if q.family == "tool":
            gold_label = gold.tool if gold.tool is not None else "NO_TOOL"
            out.append(QuestionRecord(**base, p=probs[top], correct=top == gold_label, label=top))
            if "NO_TOOL" in probs:
                out.append(QuestionRecord(**base, p=probs["NO_TOOL"], correct=gold.tool is None, label="NO_TOOL",
                                          sentinel="NO_TOOL"))  # fmt: skip
            return out
        if not on_gold:
            return []
        if q.family in ("slot", "probe", "rev"):
            sub = q.kind if q.family == "slot" else None
            truth = _label_truth(gold, q, top)
            if truth is not None:
                out.append(QuestionRecord(**base, sub=sub, p=probs[top], correct=truth, label=top))
            for label in q.sentinels:
                s_truth = _label_truth(gold, q, label)
                if s_truth is not None and label in probs:
                    out.append(QuestionRecord(**base, p=probs[label], correct=s_truth, label=label, sentinel=label))
            return out
        if q.family == "mention":
            slot = _slot(q)
            members = _gold_list(gold, slot) if slot is not None and gold.mode(slot) != "ignore" else None
            option = next((o for o in q.options if o.label == top), None)
            if members is not None and option is not None:
                hit = any(same_value(option.value, m) for m in members)
                out.append(QuestionRecord(**base, p=probs[top], correct=hit, label=top))
            return out
        if q.family == "joint":
            option = next((o for o in q.options if o.label == top), None)
            if option is not None and isinstance(option.value, Mapping):
                checked = [k for k in option.value if k in gold.slots]
                if checked:
                    ok = all(gold.accepts(k, option.value[k]) for k in checked)
                    out.append(QuestionRecord(**base, p=probs[top], correct=ok, label=top))
            return out
        return []
    if q.primitive != "noul":
        return []
    p = float(answer.get("noul", 0.5))
    if q.family == "authorized":
        return [QuestionRecord(**base, p=p, correct=on_gold)]
    if not on_gold:
        return []
    slot = _slot(q)
    if slot is None or gold.mode(slot) == "ignore":
        return []
    if q.family == "accept":
        known, value = _candidate_value(q)
        if known:
            out.append(QuestionRecord(**base, sub=q.stakes, p=p, correct=_matches(gold, slot, value)))
    elif q.family == "present":
        out.append(QuestionRecord(**base, p=p, correct=not gold.accepts(slot, None)))
    elif q.family == "more":
        members = _gold_list(gold, slot)
        if members is not None:
            offered = [o.value for m in ballot.questions_for(str(q.tool), q.path) if m.family == "mention"
                       for o in m.options]  # fmt: skip
            more = any(not any(same_value(m, v) for v in offered) for m in members)
            out.append(QuestionRecord(**base, p=p, correct=more))
    return out


def pool_hits(gold: Gold, ballots: Sequence[Ballot]) -> dict[str, PoolHit]:
    """Coverage of each checked gold argument in the gold tool's candidate pools across rounds (only arguments that
    had at least one question)."""
    if gold.tool is None:
        return {}
    out: dict[str, PoolHit] = {}
    for slot in gold.slots:
        questions = [q for b in ballots for q in b.questions_for(gold.tool, (slot,))]
        if not questions:
            continue
        found: list[tuple[Any, Mapping[str, Any]]] = []
        defaults: list[Any] = []
        for q in questions:
            if q.family in _VALUE_FAMILIES and q.family != "rev":
                found += [(o.value, o.prov) for o in q.options]
            elif q.family in ("accept", "member"):
                known, value = _candidate_value(q)
                if known:
                    candidate = q.meta.get("candidate")
                    prov = candidate.get("prov") if isinstance(candidate, Mapping) else None
                    found.append((value, prov if isinstance(prov, Mapping) else {}))
            defaults += [s.value for s in q.sentinels.values() if s.decodes_to == "default" and s.late is None]
        unique: list[tuple[Any, Mapping[str, Any]]] = []
        for v, prov in found:
            if not any(same_value(v, u) for u, _ in unique):
                unique.append((v, prov))
        out[slot] = _hit(gold, slot, [v for v, _ in _retrieval_order(unique)], defaults, kind=questions[0].kind)
    return out


def _retrieval_order(items: list[tuple[Any, Mapping[str, Any]]]) -> list[tuple[Any, Mapping[str, Any]]]:
    """Candidates in retrieval order: options are sent in canonical (casefold) order, so the rank for recall@K
    comes from the provenance (``prov.rank``, else ``prov.score`` descending), ties keeping the sent order."""

    def key(item: tuple[int, tuple[Any, Mapping[str, Any]]]) -> tuple[float, float, int]:
        index, (_, prov) = item
        rank, score = prov.get("rank"), prov.get("score")
        return (float(rank) if isinstance(rank, (int, float)) else float("inf"),
                -float(score) if isinstance(score, (int, float)) else 0.0, index)  # fmt: skip

    return [item for _, item in sorted(enumerate(items), key=key)]


def _hit(gold: Gold, slot: str, values: list[Any], defaults: list[Any], *, kind: str | None) -> PoolHit:
    members = _gold_list(gold, slot)
    if members is not None:
        ranks = [next((i + 1 for i, v in enumerate(values) if same_value(v, m)), None) for m in members]
        found = all(r is not None for r in ranks)
        rank = max((r for r in ranks if r is not None), default=None) if found else None
        return PoolHit(in_pool=found, rank=rank, size=len(values), kind=kind)
    rank = next((i + 1 for i, v in enumerate(values) if _matches(gold, slot, v)), None)
    if rank is not None:
        return PoolHit(in_pool=True, rank=rank, size=len(values), kind=kind)
    via_default = any(gold.accepts(slot, d) for d in defaults) or gold.accepts(slot, None)
    return PoolHit(in_pool=via_default, rank=None, size=len(values), via_default=via_default, kind=kind)


__all__ = [
    "Progress",
    "RouterFactory",
    "TraceAnalysis",
    "aevaluate_case",
    "analyze_trace",
    "arun",
    "attribute",
    "error_record",
    "evaluate_case",
    "pool_hits",
    "question_calibration",
    "round_answers",
    "router_factory_for",
    "run",
    "run_dataset",
    "score_decision",
    "trace_rounds",
]
