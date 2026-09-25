"""Run jevtools over BFCL cases and summarize the results.

Each case gets its own :class:`~jevtools.router.Router` over the case's functions, with a fixed clock. Two scores are
reported per case, because jevtools returns more than "a call or not":

- **strict**: only an ``execute`` decision emits a call (what an autonomous agent would run);
- **proposal**: the proposed call counts whatever the outcome (``confirm``/``clarify`` included), unless the outcome
  is ``abstain``/``refuse``; this is the call a user would see on a confirm card.

Backends: ``"oracle"`` (the ceiling, :mod:`jevtools.bench.oracle`), the offline ``LexicalSimulator`` (plumbing only,
never evidence about Jev), or any :class:`~jevtools.backends.base.Backend` such as ``jt.backends.auto()`` for live Jev.
Every case is also decided by the oracle first (no network), which gives the case's **ceiling** (could jevtools have
emitted the right call at all?) and its per-parameter coverage. A live run can then separate the two ways to fail:
the right value was never nominated (a coverage miss, below the ceiling) or it was on the ballot and not elected.
"""

from __future__ import annotations

import json
import time
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from jevtools.backends.base import Backend
from jevtools.bench.bfcl import IRRELEVANCE, BfclCase, check
from jevtools.bench.oracle import BfclGold, OracleBackend, ParamCoverage
from jevtools.context import Context
from jevtools.decision import Decision
from jevtools.plan import compile_round
from jevtools.router import Router

BENCH_NOW = datetime(2026, 9, 25, 10, 0, tzinfo=timezone.utc)
"""The fixed clock every case runs at, so that runs are reproducible."""
NO_CALL_OUTCOMES = ("abstain", "refuse")


@dataclass
class CaseRecord:
    """The result of one case."""

    id: str
    category: str
    outcome: str
    rule: str
    strict: bool
    proposal: bool
    ceiling: bool | None = None
    """Whether the oracle's proposal passes: the right call was reachable (``None`` when not computed)."""
    error_type: str | None = None
    """BFCL's error type for the proposal view (``None`` when it passed)."""
    call: dict[str, Any] | None = None
    jev_calls: int = 0
    input_tokens: int = 0
    cost_usd: float | None = None
    latency_ms: float = 0.0
    tool_speculated: bool | None = None
    coverage: list[ParamCoverage] = field(default_factory=list)
    error: str | None = None
    """An exception raised while deciding (the case counts as failed)."""

    @property
    def attribution(self) -> str:
        """Why the proposal view failed, most upstream cause first (``ok`` when it passed):

        ``error``; ``called_irrelevant``; ``tool_not_speculated`` (a required value had no candidate, so the tool
        was not asked about); ``value_not_nominated`` (a value BFCL expects was on no option); ``no_call`` (the
        decision proposed nothing); ``wrong_call`` (everything was on the ballot, but the call that came out is
        wrong: for the oracle, a decoding or normalization issue; for a live backend, mostly the model's choice).
        """
        if self.proposal:
            return "ok"
        if self.error is not None:
            return "error"
        if self.category in IRRELEVANCE:
            return "called_irrelevant"
        if self.tool_speculated is False:
            return "tool_not_speculated"
        if any(c.status in ("uncovered", "not_asked") for c in self.coverage):
            return "value_not_nominated"
        if self.outcome in NO_CALL_OUTCOMES or self.call is None:
            return "no_call"
        return "wrong_call"


@dataclass
class BenchReport:
    """All case records of a run plus how the run was configured."""

    mode: str
    records: list[CaseRecord]
    meta: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> dict[str, dict[str, Any]]:
        """Per category (and ``ALL``): case count, strict, proposal and ceiling accuracy, accuracy within the
        ceiling, outcomes, failure attribution, parameter coverage and Jev usage."""
        groups: dict[str, list[CaseRecord]] = {}
        for record in self.records:
            groups.setdefault(record.category, []).append(record)
        out = {category: _summarize(records) for category, records in groups.items()}
        if len(groups) > 1:
            out["ALL"] = _summarize(self.records)
        return out

    def to_json(self) -> str:
        """The report as JSON (records included)."""
        doc = {"mode": self.mode, "meta": self.meta, "summary": self.summary(),
               "records": [asdict(r) for r in self.records]}  # fmt: skip
        return json.dumps(doc, indent=1, default=str)

    def save(self, path: str | Path) -> None:
        """Write :meth:`to_json` to ``path``."""
        Path(path).write_text(self.to_json(), encoding="utf-8")

    def render(self) -> str:
        """A Markdown table of the summary."""
        rows = ["| category | n | ceiling | proposal | strict | within ceiling | execute | confirm | clarify | abstain "
                "| top failure causes |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|"]  # fmt: skip
        for category, s in self.summary().items():
            outcomes = s["outcomes"]
            causes = ", ".join(f"{k} {v}" for k, v in s["attribution"].items() if k != "ok")
            within = "–" if s["within_ceiling"] is None else f"{s['within_ceiling']:.1%}"
            ceiling = "–" if s["ceiling"] is None else f"{s['ceiling']:.1%}"
            rows.append(f"| {category} | {s['n']} | {ceiling} | {s['proposal']:.1%} | {s['strict']:.1%} | {within} | "
                        f"{outcomes.get('execute', 0)} | {outcomes.get('confirm', 0)} | {outcomes.get('clarify', 0)} | "
                        f"{outcomes.get('abstain', 0)} | {causes} |")  # fmt: skip
        return "\n".join(rows)


def _summarize(records: Sequence[CaseRecord]) -> dict[str, Any]:
    n = len(records)
    coverage = Counter(c.status for r in records for c in r.coverage)
    misses = [c for r in records for c in r.coverage if c.status in ("uncovered", "not_asked")]
    by_kind: dict[str, Counter[str]] = {}
    for r in records:
        for c in r.coverage:
            by_kind.setdefault(c.kind or "none", Counter())[c.status] += 1
    scored = [r for r in records if r.ceiling is not None]
    reachable = [r for r in scored if r.ceiling]
    return {
        "n": n,
        "strict": sum(r.strict for r in records) / n if n else 0.0,
        "proposal": sum(r.proposal for r in records) / n if n else 0.0,
        "ceiling": sum(bool(r.ceiling) for r in scored) / len(scored) if scored else None,
        "within_ceiling": sum(r.proposal for r in reachable) / len(reachable) if reachable else None,
        "outcomes": dict(Counter(r.outcome for r in records).most_common()),
        "attribution": dict(Counter(r.attribution for r in records).most_common()),
        "error_types": dict(Counter(r.error_type for r in records if r.error_type).most_common(8)),
        "coverage": dict(coverage.most_common()),
        "misses_in_text": sum(bool(c.in_text) for c in misses),
        "misses_not_in_text": sum(c.in_text is False for c in misses),
        "coverage_by_kind": {k: dict(v.most_common()) for k, v in sorted(by_kind.items())},
        "jev_calls": sum(r.jev_calls for r in records),
        "input_tokens": sum(r.input_tokens for r in records),
        "cost_usd": sum(r.cost_usd or 0.0 for r in records) or None,
    }


def bench_tools(case: BfclCase, *, risk: str | None = "read") -> list[dict[str, Any]]:
    """The case's functions as OpenAI tools, with ``x-jev.risk`` set to ``risk`` (``None`` keeps the inferred
    tier). BFCL functions are side-effect free test fixtures, so the bench treats them as read-tier tools by
    default; the inferred tier falls back to ``external`` for most of their names (``calculate``, ``find``…)."""
    tools = case.tools
    if risk is not None:
        for tool in tools:
            tool["function"]["x-jev"] = {"risk": risk}
    return tools


def _router(case: BfclCase, backend: Any, risk: str | None, now: datetime) -> Router:
    return Router(bench_tools(case, risk=risk), backend=backend, context=Context(now=now, tz="UTC", locale="en"))


def _calls(decision: Decision) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """``(executed, proposed)`` in BFCL's output shape ``[{name: arguments}]``."""
    executed = [{c.name: dict(c.arguments)} for c in decision.tool_calls]
    call = decision.call
    proposed = (
        [{call.name: dict(call.arguments)}] if call is not None and decision.outcome not in NO_CALL_OUTCOMES else []
    )
    return executed, proposed


def _oracle_run(case: BfclCase, risk: str | None, now: datetime) -> tuple[OracleBackend, Decision]:
    oracle = OracleBackend(BfclGold(case))
    router = _router(case, oracle, risk, now)
    ctx = router.context_for(case.messages)
    oracle.plan = compile_round(router.catalog, ctx, router.policy, mode="turn", limits=router.round_limits())
    return oracle, router.decide(case.messages)


def run_case(
    case: BfclCase,
    backend: Backend | str,
    *,
    risk: str | None = "read",
    now: datetime = BENCH_NOW,
    ceiling: bool = True,
) -> CaseRecord:
    """Decide one case with ``backend`` (``"oracle"`` for the oracle) and score it. With ``ceiling`` (the default),
    a non-oracle run first decides the case with the oracle to record its ceiling and coverage."""
    oracle: OracleBackend | None = None
    reachable: bool | None = None
    started = time.perf_counter()
    try:
        if backend == "oracle" or ceiling:
            oracle, oracle_decision = _oracle_run(case, risk, now)
            reachable = check(case, _calls(oracle_decision)[1]).valid
        if backend == "oracle":
            assert oracle is not None
            decision = oracle_decision
        else:
            started = time.perf_counter()
            decision = _router(case, backend, risk, now).decide(case.messages)
    except Exception as exc:  # noqa: BLE001 - a crash is a failed case, reported with its message
        return CaseRecord(id=case.id, category=case.category, outcome="error", rule="", strict=False,
                          proposal=False, ceiling=reachable, error=f"{type(exc).__name__}: {exc}"[:300])  # fmt: skip
    latency = (time.perf_counter() - started) * 1000
    executed, proposed = _calls(decision)
    proposal = check(case, proposed)
    speculated = None
    if oracle is not None and oracle.plan is not None and oracle.expected is not None:
        tools = {t["name"]: t for t in oracle.plan.ballot.to_doc()["tools"]}
        speculated = bool(tools.get(oracle.expected[0], {}).get("speculated", False))
    call = decision.call
    usage = decision.usage
    return CaseRecord(
        id=case.id,
        category=case.category,
        outcome=str(decision.outcome),
        rule=decision.rule,
        strict=check(case, executed).valid,
        proposal=proposal.valid,
        ceiling=reachable,
        error_type=proposal.error_type,
        call={"name": call.name, "arguments": dict(call.arguments)} if call is not None else None,
        jev_calls=usage.jev_calls,
        input_tokens=usage.jev_input_tokens,
        cost_usd=usage.cost_usd,
        latency_ms=round(latency, 1),
        tool_speculated=speculated,
        coverage=oracle.coverage() if oracle is not None else [],
    )


def run_bfcl(
    cases: Iterable[BfclCase],
    backend: Backend | str,
    *,
    risk: str | None = "read",
    now: datetime = BENCH_NOW,
    ceiling: bool = True,
    progress: Callable[[int, CaseRecord], None] | None = None,
    meta: Mapping[str, Any] | None = None,
) -> BenchReport:
    """Run every case and return the report (``backend="oracle"`` for the ceiling alone)."""
    records: list[CaseRecord] = []
    for index, case in enumerate(cases):
        record = run_case(case, backend, risk=risk, now=now, ceiling=ceiling)
        records.append(record)
        if progress is not None:
            progress(index, record)
    mode = backend if isinstance(backend, str) else str(getattr(backend, "name", type(backend).__name__))
    return BenchReport(mode=mode, records=records, meta={"risk": risk, "now": now.isoformat(), **dict(meta or {})})


__all__ = ["BENCH_NOW", "BenchReport", "CaseRecord", "bench_tools", "run_bfcl", "run_case"]
