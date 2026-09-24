"""The resolution trace (spec §3.9): the audit warrant of a decision, replayable with the model out of the loop.

A :class:`Trace` stores the policy/catalog/context hashes, every round (its Ballot and the verbatim request and
response bodies with their hashes), the tool distribution, the bindings, gates, factors, composition, flags, the
outcome with the rule that fired, the call and the exact :class:`~jevtools.policy.PolicyInput` the policy saw.

:func:`verify` re-checks a trace: body hashes, re-decoding the stored responses against the stored (or rebuilt)
Ballot, option values, channels, the composition and the policy rule.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from jevtools._version import SPEC_VERSION
from jevtools.ballot import Ballot
from jevtools.candidates import NO_TOOL, value_key
from jevtools.canonical import canonical_json, jsonable, round4, sha256_of
from jevtools.confidence import Composition, compose, prior_of
from jevtools.context import Context, Mode
from jevtools.decision import DecisionIds, ToolCall
from jevtools.kinds.base import ResolveContext, SlotResult
from jevtools.policy import Outcome, Policy, PolicyInput, PolicyResult, Tier, evaluate
from jevtools.spec.catalog import Catalog
from jevtools.wire import DecisionResponse

if TYPE_CHECKING:
    from jevtools.decode import Decoded

StoreBodies = Literal["full", "hash_only"]
TOLERANCE = 1e-3
"""Absolute tolerance when comparing recomputed numbers with the (4-decimal) stored ones."""


def _r(x: float | None) -> float | None:
    return None if x is None else round4(x)


class CallRecord(BaseModel):
    """One Jev call of a round."""

    model_config = ConfigDict(extra="forbid")

    backend: str
    model_requested: str
    model_answered: str | None = None
    request_sha256: str
    response_sha256: str | None = None
    request: dict[str, Any] | None = None
    response: dict[str, Any] | None = None
    usage: dict[str, Any] = Field(default_factory=dict)
    latency_ms: int | None = None
    retries: int | None = None
    error: str | None = None


class RoundRecord(BaseModel):
    """One round: its mode, Ballot and calls. ``merged`` marks same-state rounds decoded together with the
    previous ones (widen, fill, escalation gate)."""

    model_config = ConfigDict(extra="forbid")

    round: int
    mode: str
    ballot_sha256: str
    ballot: dict[str, Any] | None = None
    calls: list[CallRecord] = Field(default_factory=list)
    merged: bool = False


class Trace(BaseModel):
    """The resolution trace (§3.9). Numbers are rounded to 4 decimals in :meth:`to_doc`; bodies are verbatim."""

    model_config = ConfigDict(extra="forbid")

    spec: str = SPEC_VERSION
    trace_id: str
    decision_id: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc).replace(microsecond=0))
    policy: dict[str, str]
    catalog_sha256: str
    context: dict[str, Any]
    ballot_sha256: str | None = None
    rounds: list[RoundRecord] = Field(default_factory=list)
    tool: dict[str, Any] | None = None
    bindings: dict[str, dict[str, Any]] = Field(default_factory=dict)
    gates: dict[str, float] = Field(default_factory=dict)
    factors: dict[str, float] = Field(default_factory=dict)
    composition: dict[str, Any] | None = None
    flags: list[str] = Field(default_factory=list)
    outcome: dict[str, Any]
    call: dict[str, Any] | None = None
    idempotency_key: str | None = None
    resumed_from: str | None = None
    notes: list[str] = Field(default_factory=list)
    policy_input: dict[str, Any] | None = None

    def to_doc(self) -> dict[str, Any]:
        """The trace document in normative key order."""
        comp = self.composition
        tool = self.tool
        return {
            "spec": self.spec, "trace_id": self.trace_id, "decision_id": self.decision_id,
            "created_at": self.created_at.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "policy": dict(self.policy), "catalog_sha256": self.catalog_sha256, "context": jsonable(self.context),
            "ballot_sha256": self.ballot_sha256, "rounds": [jsonable(r) for r in self.rounds],
            "tool": None if tool is None else {**tool, "p": _r(tool.get("p")),
                                               "alternatives": {k: _r(v) for k, v in tool["alternatives"].items()}},
            "bindings": {name: _round_binding(b) for name, b in self.bindings.items()},
            "gates": {k: _r(v) for k, v in self.gates.items()},
            "factors": {k: _r(v) for k, v in self.factors.items()},
            "composition": None if comp is None else {k: _r(v) if isinstance(v, float) else v
                                                      for k, v in comp.items()},
            "flags": list(self.flags), "outcome": dict(self.outcome), "call": jsonable(self.call),
            "idempotency_key": self.idempotency_key, "resumed_from": self.resumed_from, "notes": list(self.notes),
            "policy_input": self.policy_input,
        }  # fmt: skip

    def to_json(self) -> bytes:
        """Canonical JSON of :meth:`to_doc`."""
        return canonical_json(self.to_doc())

    @classmethod
    def from_doc(cls, doc: Mapping[str, Any]) -> Trace:
        """Load a stored trace document."""
        return cls.model_validate(dict(doc))


def _round_binding(binding: Mapping[str, Any]) -> dict[str, Any]:
    out = dict(binding)
    out["p"] = _r(binding.get("p"))
    out["alternatives"] = [{**a, "p": _r(a.get("p"))} for a in binding.get("alternatives", [])]
    out["sentinels"] = {k: _r(v) for k, v in (binding.get("sentinels") or {}).items()}
    return jsonable(out)  # type: ignore[no-any-return]


# --------------------------------------------------------------------------------------------------------------------
# Building
# --------------------------------------------------------------------------------------------------------------------


def call_record(
    *,
    backend: str,
    model: str,
    request: Mapping[str, Any],
    response: DecisionResponse | None,
    latency_ms: int | None,
    error: str | None = None,
    store_bodies: StoreBodies = "full",
) -> CallRecord:
    """The record of one Jev call (bodies inline unless ``store_bodies == "hash_only"``)."""
    body = response.model_dump(mode="json", exclude_none=True) if response is not None else None
    usage = response.usage.model_dump(exclude_none=True) if response is not None else {}
    full = store_bodies == "full"
    return CallRecord(
        backend=backend, model_requested=model, model_answered=response.model if response is not None else None,
        request_sha256=sha256_of(request), response_sha256=sha256_of(body) if body is not None else None,
        request=dict(request) if full else None, response=body if full else None, usage=usage,
        latency_ms=latency_ms, error=error,
    )  # fmt: skip


def binding_doc(result: SlotResult, ballot: Ballot | None) -> dict[str, Any]:
    """The trace entry of one slot result."""
    qid = result.qids[0] if result.qids else None
    question = ballot.by_qid.get(qid) if ballot is not None and qid is not None else None
    return {
        "qid": qid, "family": question.family if question is not None else None,
        "value": None if result.is_bottom else jsonable(result.value),
        "bottom": result.value.value if result.is_bottom else None,
        "label": result.label, "p": result.factor if result.factor is not None else result.p,
        "alternatives": [{"label": a.label or a.display, "p": a.p, **({"part": a.part} if a.part else {})}
                         for a in result.alternatives],
        "sentinels": dict(result.sentinels), "channel": result.channel.value if result.channel else None,
        "prov": jsonable(result.prov), "normalizer": result.normalizer, "shape": result.shape,
        "flags": list(result.flags), "late": bool(result.late and {"placeholders", "derive"} & set(result.late)),
    }  # fmt: skip


def composition_doc(comp: Composition | None) -> dict[str, Any] | None:
    """``{W, PI, L, J, tier, rule, calibrator, C}``."""
    return comp.to_doc() if comp is not None else None


def outcome_doc(result: PolicyResult) -> dict[str, Any]:
    """``{value, rule, bottleneck, shape}`` (+ caps and reason when set)."""
    doc: dict[str, Any] = {"value": result.outcome.value, "rule": result.rule, "bottleneck": result.bottleneck,
                           "shape": result.shape}  # fmt: skip
    if result.caps:
        doc["caps"] = list(result.caps)
    if result.reason:
        doc["reason"] = result.reason
    return doc


def build_trace(
    *,
    ids: DecisionIds,
    policy: Policy,
    catalog: Catalog,
    context: Context,
    rounds: Sequence[RoundRecord],
    decoded: Decoded | None,
    ballot: Ballot | None,
    comp: Composition | None,
    result: PolicyResult,
    inp: PolicyInput | None,
    call: ToolCall | None,
    flags: Sequence[str] = (),
    resumed_from: str | None = None,
    notes: Sequence[str] = (),
) -> Trace:
    """Assemble the trace of one decision (``ballot`` is the decode view: the merged same-state rounds)."""
    td = decoded.decision if decoded is not None else None
    tool = None
    if decoded is not None and decoded.chosen is not None:
        alternatives = {k: v for k, v in decoded.tool_dist.items() if k != decoded.chosen and v >= 0.01}
        tool = {"chosen": decoded.chosen, "p": decoded.tool_dist.get(decoded.chosen, 0.0),
                "alternatives": alternatives, "call_map": decoded.call_map.call_map}  # fmt: skip
    executed = call is not None and result.outcome is Outcome.EXECUTE
    return Trace(
        trace_id=ids.trace_id, decision_id=ids.decision_id,
        policy={"version": policy.version, "sha256": policy.sha256}, catalog_sha256=catalog.sha256,
        context={"sha256": context.sha256, "sources": context.source_hashes(),
                 "now": context.current_time().isoformat(), "tz": context.timezone_name},
        ballot_sha256=rounds[0].ballot_sha256 if rounds else None, rounds=list(rounds), tool=tool,
        bindings={n: binding_doc(r, ballot) for n, r in td.slots.items()} if td is not None else {},
        gates=dict(td.gates) if td is not None else {}, factors=dict(td.factors.items) if td is not None else {},
        composition=composition_doc(comp), flags=list(flags), outcome=outcome_doc(result),
        call=call.to_doc(with_ids=False) if call is not None else None,
        idempotency_key=call.idempotency_key if executed and call is not None else None,
        resumed_from=resumed_from, notes=list(notes) + (list(decoded.notes) if decoded is not None else []),
        policy_input=inp.model_dump(mode="json") if inp is not None else None,
    )  # fmt: skip


# --------------------------------------------------------------------------------------------------------------------
# Verification
# --------------------------------------------------------------------------------------------------------------------


class Check(BaseModel):
    """One verification step: ``ok`` is ``True``, ``False``, or ``None`` when it could not run."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    ok: bool | None
    detail: str = ""


class VerifyReport(BaseModel):
    """Result of :func:`verify`: passes when no check failed."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    checks: list[Check] = Field(default_factory=list)

    @property
    def ok(self) -> bool:
        """No check failed (skipped checks do not fail)."""
        return all(check.ok is not False for check in self.checks)

    @property
    def failures(self) -> list[Check]:
        """The failed checks."""
        return [check for check in self.checks if check.ok is False]

    def __bool__(self) -> bool:
        return self.ok


def verify(
    trace: Trace | Mapping[str, Any],
    *,
    catalog: Catalog | None = None,
    context: Context | None = None,
    policy: Policy | None = None,
) -> VerifyReport:
    """Re-check a trace with the model out of the loop (§3.9): (1) body hashes, (2) re-decoding the stored
    responses reproduces the tool and every binding, (3) values equal their options' values, (4) channels were
    allowed, (5) the composition recomputes, (6) the policy (same version) yields the same rule and outcome.

    Re-decoding needs the ``catalog``; with a ``context`` the round-1 Ballot is also rebuilt and compared.
    """
    trace = trace if isinstance(trace, Trace) else Trace.from_doc(trace)
    checks = [*_check_hashes(trace)]
    ballots = _stored_ballots(trace)
    decoded_check, results = _check_redecode(trace, ballots, catalog, context, policy)
    checks += decoded_check
    checks.append(_check_values(trace, ballots))
    checks.append(_check_channels(trace, catalog))
    checks.append(_check_composition(trace, policy))
    checks.append(_check_policy(trace, policy))
    return VerifyReport(checks=checks)


def _check_hashes(trace: Trace) -> list[Check]:
    problems: list[str] = []
    skipped = 0
    for rnd in trace.rounds:
        if rnd.ballot is not None:
            try:
                ballot = Ballot.from_doc(rnd.ballot)
            except ValueError as exc:
                problems.append(f"round {rnd.round}: {exc}")
            else:
                if ballot.sha256 != rnd.ballot_sha256:
                    problems.append(f"round {rnd.round}: ballot hash mismatch")
        for i, call in enumerate(rnd.calls):
            if call.request is None:
                skipped += 1
                continue
            if sha256_of(call.request) != call.request_sha256:
                problems.append(f"round {rnd.round} call {i}: request hash mismatch")
            if call.response is not None and sha256_of(call.response) != call.response_sha256:
                problems.append(f"round {rnd.round} call {i}: response hash mismatch")
    detail = "; ".join(problems) or (f"{skipped} call(s) stored hash-only" if skipped else "")
    return [Check(name="hashes", ok=not problems, detail=detail)]


def _stored_ballots(trace: Trace) -> list[tuple[Ballot, list[DecisionResponse]] | None]:
    out: list[tuple[Ballot, list[DecisionResponse]] | None] = []
    for rnd in trace.rounds:
        if rnd.ballot is None or any(c.response is None for c in rnd.calls):
            out.append(None)
            continue
        try:
            ballot = Ballot.from_doc(rnd.ballot)
        except ValueError:
            out.append(None)
            continue
        out.append((ballot, [DecisionResponse.from_json(c.response or {}) for c in rnd.calls]))
    return out


def _check_redecode(
    trace: Trace,
    ballots: Sequence[tuple[Ballot, list[DecisionResponse]] | None],
    catalog: Catalog | None,
    context: Context | None,
    policy: Policy | None,
) -> tuple[list[Check], dict[str, SlotResult]]:
    from jevtools.decode import collect_answers, decode_answers
    from jevtools.plan import compile_round, merge_ballots

    if catalog is None or not trace.rounds:
        return [Check(name="redecode", ok=None, detail="needs a catalog and stored rounds")], {}
    tail = _decode_tail(trace)
    stored = [ballots[i] for i in tail]
    if any(s is None for s in stored):
        return [Check(name="redecode", ok=None, detail="bodies not stored")], {}
    checks: list[Check] = []
    policy = policy or Policy()
    pools = {}
    ctx = context or _minimal_context(trace, stored[0][0])  # type: ignore[index]
    first_mode = trace.rounds[tail[0]].mode
    if context is not None and tail[0] == 0 and first_mode in ("turn", "loop"):
        # only a decision's first round is a plain compile (later ones: re-plans, escalation gates, resumes)
        ballot = stored[0][0]  # type: ignore[index]
        mode: Mode = "loop" if first_mode == "loop" else "turn"
        rebuilt = compile_round(catalog, context, policy, mode=mode, tool_choice=_tool_choice_of(ballot))
        same = rebuilt.ballot.sha256 == trace.rounds[tail[0]].ballot_sha256
        checks.append(Check(name="ballot_rebuild", ok=same, detail="" if same else "rebuilt Ballot differs"))
        pools = rebuilt.pools
    merged = None
    answers: dict[str, Any] = {}
    for entry in stored:
        assert entry is not None
        ballot, responses = entry
        merged = ballot if merged is None else merge_ballots(merged, ballot)
        round_answers, _ = collect_answers(ballot, responses)
        answers.update(round_answers)
    assert merged is not None
    rc = ResolveContext(ctx=ctx, catalog=catalog, policy=policy, state=merged.state if isinstance(merged.state, dict)
                        else None, mode=merged.mode)  # fmt: skip
    decoded = decode_answers(merged, answers, rc, pools=pools)
    problems = _compare_decoded(trace, decoded, with_rows=context is not None)
    checks.append(Check(name="redecode", ok=not problems, detail="; ".join(problems)))
    td = decoded.decision
    return checks, (td.slots if td is not None else {})


def _tool_choice_of(ballot: Ballot) -> str | dict[str, Any]:
    """The ``tool_choice`` a stored Ballot was compiled with: no ``tool`` question and one speculated tool → that
    tool named; a ``tool`` Choice without ``NO_TOOL`` → ``required``; else ``auto``."""
    tool_question = ballot.by_qid.get("tool")
    if tool_question is None:
        speculated = [t.name for t in ballot.tools if t.speculated]
        if len(speculated) == 1:
            return {"type": "function", "function": {"name": speculated[0]}}
        return "auto"
    return "auto" if NO_TOOL in tool_question.sentinels else "required"


def _decode_tail(trace: Trace) -> list[int]:
    """Indices of the rounds decoded together for the final decision (the last round plus merged predecessors)."""
    tail = [len(trace.rounds) - 1]
    while tail[0] > 0 and trace.rounds[tail[0]].merged:
        tail.insert(0, tail[0] - 1)
    return tail


def _minimal_context(trace: Trace, ballot: Ballot) -> Context:
    state = ballot.state if isinstance(ballot.state, dict) else {}
    now = trace.context.get("now")
    moment = datetime.fromisoformat(now) if now else None
    tz = trace.context.get("tz")
    if moment is not None and not tz:
        moment, tz = moment.astimezone(timezone.utc), "UTC"
    return Context(messages=str(state.get("request", "")), now=moment, tz=tz)


def _compare_decoded(trace: Trace, decoded: Any, *, with_rows: bool) -> list[str]:
    """Differences between the stored and the re-decoded tool and bindings. Without a context (``with_rows``
    false) registry attributes are unknown, so late-bound values (placeholders, derivations) are not compared."""
    problems: list[str] = []
    if trace.tool is not None:
        chosen = decoded.chosen
        if chosen != trace.tool.get("chosen"):
            problems.append(f"tool: {chosen} != {trace.tool.get('chosen')}")
        elif not math.isclose(decoded.tool_dist.get(chosen, 0.0), trace.tool.get("p") or 0.0, abs_tol=TOLERANCE):
            problems.append("tool probability differs")
    td = decoded.decision
    if td is None:
        return problems
    for name, binding in trace.bindings.items():
        if binding.get("late") and not with_rows:
            continue
        result = td.slots.get(name)
        if result is None:
            problems.append(f"{name}: not decoded")
            continue
        value = None if result.is_bottom else jsonable(result.value)
        if value_key(value) != value_key(binding.get("value")):
            problems.append(f"{name}: value differs")
        p = result.factor if result.factor is not None else result.p
        if not math.isclose(p, binding.get("p") or 0.0, abs_tol=TOLERANCE):
            problems.append(f"{name}: p differs")
    return problems


def _check_values(trace: Trace, ballots: Sequence[tuple[Ballot, list[DecisionResponse]] | None]) -> Check:
    questions = {q.qid: q for entry in ballots if entry is not None for q in entry[0].questions}
    if not questions:
        return Check(name="values", ok=None, detail="no stored Ballot")
    problems = []
    for name, binding in trace.bindings.items():
        question = questions.get(binding.get("qid") or "")
        option = next((o for o in question.options if o.label == binding.get("label")), None) if question else None
        if option is None or option.late is not None or binding.get("value") is None:
            continue
        if value_key(option.value) != value_key(binding["value"]) and not binding.get("prov", {}).get("late"):
            problems.append(f"{name}: value is not its option's value")
    return Check(name="values", ok=not problems, detail="; ".join(problems))


def _check_channels(trace: Trace, catalog: Catalog | None) -> Check:
    if catalog is None or trace.call is None:
        return Check(name="channels", ok=None, detail="needs a catalog and a call")
    tool = catalog.get(trace.call["name"])
    problems = []
    for name, binding in trace.bindings.items():
        channel = binding.get("channel")
        if channel is None or binding.get("value") is None or (binding.get("prov") or {}).get("default"):
            continue
        if channel not in {c.value for c in tool.slot(name).channels}:
            problems.append(f"{name}: channel {channel} not allowed")
    return Check(name="channels", ok=not problems, detail="; ".join(problems))


def _check_composition(trace: Trace, policy: Policy | None) -> Check:
    comp = trace.composition
    if comp is None or not trace.factors:
        return Check(name="composition", ok=None, detail="no composition recorded")
    policy = policy or Policy()
    recomputed = compose(trace.factors, comp.get("J"))
    problems = [key for key in ("W", "PI", "L") if not math.isclose(getattr(recomputed, key), comp[key],
                                                                     abs_tol=TOLERANCE)]  # fmt: skip
    if comp.get("calibrator") is None and comp.get("tier"):
        prior = prior_of(recomputed, policy.tier(Tier(comp["tier"])).composition)
        c = min(prior, recomputed.W)
        if not math.isclose(c, comp["C"], abs_tol=TOLERANCE):
            problems.append("C")
    return Check(name="composition", ok=not problems, detail=", ".join(f"{p} differs" for p in problems))


def _check_policy(trace: Trace, policy: Policy | None) -> Check:
    policy = policy or Policy()
    if trace.policy_input is None:
        return Check(name="policy", ok=None, detail="no policy input recorded")
    if policy.version != trace.policy.get("version"):
        return Check(name="policy", ok=None, detail=f"policy version {policy.version} != {trace.policy['version']}")
    result = evaluate(PolicyInput.model_validate(trace.policy_input), policy)
    same = result.rule == trace.outcome.get("rule") and result.outcome.value == trace.outcome.get("value")
    detail = "" if same else f"{result.rule}/{result.outcome.value} != {trace.outcome.get('rule')}/" \
                             f"{trace.outcome.get('value')}"  # fmt: skip
    return Check(name="policy", ok=same, detail=detail)


__all__ = [
    "CallRecord",
    "Check",
    "RoundRecord",
    "StoreBodies",
    "Trace",
    "VerifyReport",
    "binding_doc",
    "build_trace",
    "call_record",
    "composition_doc",
    "outcome_doc",
    "verify",
]
