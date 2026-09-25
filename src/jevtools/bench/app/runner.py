"""Run jevtools over the app-domain benchmark and summarize the results (``jevtools bench app``).

The benchmark is a set of §11.1 datasets (``domains/<name>/cases.jsonl``, each case naming its ``catalog.json`` and
``context.json``). Every case is scored by the eval harness (:func:`jevtools.eval.harness.score_decision`): whether
the outcome is one the gold label allows, whether a shown call is the gold call, whether a wrong call executed,
whether a clarify menu offered the right value, whether a planted (injected) value reached a call.

Every case is also decided by the oracle first (:mod:`jevtools.bench.oracle`, no network): its result is the case's
**ceiling**, the best decision jevtools can produce given the candidates code nominated. A live run then separates
coverage misses from the model's own choices ("within ceiling").
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path
from typing import Any

from jevtools.bench.oracle import EvalGold, OracleBackend, ParamCoverage
from jevtools.eval.dataset import EvalCase, load
from jevtools.eval.harness import evaluate_case, router_factory_for
from jevtools.eval.report import EvalRecord
from jevtools.plan import compile_round

DOMAINS = ("inbox", "crm", "banking", "workspace", "helpdesk", "research")
"""The bundled domains, in report order."""
CALL_EXPECTED = frozenset({"execute", "confirm"})


def domains_dir() -> Path:
    """The directory holding the bundled domains."""
    return Path(str(resources.files("jevtools.bench.app") / "domains"))


def load_domain(name: str, directory: str | Path | None = None) -> list[EvalCase]:
    """The cases of one domain (``directory`` defaults to the bundled domains)."""
    base = Path(directory) if directory is not None else domains_dir()
    path = base / name / "cases.jsonl"
    if not path.is_file():
        raise FileNotFoundError(f"no app-bench domain {name!r} at {path}")
    return load(path)


@dataclass
class AppRecord:
    """The result of one case: the harness record of the backend's decision plus the oracle's ceiling."""

    domain: str
    record: EvalRecord
    ceiling: bool | None = None
    coverage: list[ParamCoverage] = field(default_factory=list)
    gold_slots: tuple[str, ...] = ()
    """The arguments the gold label names values for (a clarify menu over one of them can offer the right one)."""

    @property
    def expects_call(self) -> bool:
        """Whether the gold label wants a call shown (execute or confirm)."""
        return bool(set(self.record.outcomes_ok) & CALL_EXPECTED)

    @property
    def expects_clarify(self) -> bool:
        return set(self.record.outcomes_ok) <= {"clarify"}

    @property
    def menu_scored(self) -> bool:
        """A clarify menu that can be judged: it offers tools, or values of an argument the gold label names."""
        menu = self.record.menu or []
        return any(o.get("action") == "tool" or o.get("slot") in self.gold_slots for o in menu)

    @property
    def wrong_execution(self) -> bool:
        """A call executed although executing it is wrong (the safety failure)."""
        return self.record.executed and self.record.wrong_if_executed


@dataclass
class AppReport:
    """All records of a run."""

    mode: str
    records: list[AppRecord]
    meta: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> dict[str, dict[str, Any]]:
        """Per domain (and ``ALL``) plus per tag (``tag:<name>``): see :func:`_summarize`."""
        out: dict[str, dict[str, Any]] = {}
        for domain in dict.fromkeys(r.domain for r in self.records):
            out[domain] = _summarize([r for r in self.records if r.domain == domain])
        if len(out) > 1:
            out["ALL"] = _summarize(self.records)
        for tag in sorted({t for r in self.records for t in r.record.tags}):
            out[f"tag:{tag}"] = _summarize([r for r in self.records if tag in r.record.tags])
        return out

    def render(self, *, tags: bool = False) -> str:
        """Markdown tables: per domain (and per tag with ``tags``)."""
        head = ["| {} | n | ceiling | correct | within ceiling | calls right | clarify useful | wrong executions "
                "| injections | failure stages |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|"]  # fmt: skip
        summary = self.summary()
        rows = [head[0].format("domain"), head[1]]
        for key, s in summary.items():
            if not key.startswith("tag:"):
                rows.append(_row(key, s))
        if tags:
            rows += ["", head[0].format("tag"), head[1]]
            rows += [_row(key[4:], s) for key, s in summary.items() if key.startswith("tag:")]
        return "\n".join(rows)

    def to_json(self) -> str:
        doc = {"mode": self.mode, "meta": self.meta, "summary": self.summary(),
               "records": [{"domain": r.domain, "ceiling": r.ceiling,
                            "coverage": [c.__dict__ for c in r.coverage],
                            "record": r.record.model_dump(mode="json", exclude={"trace", "decision", "questions"})}
                           for r in self.records]}  # fmt: skip
        return json.dumps(doc, indent=1, default=str)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(self.to_json(), encoding="utf-8")


def _pct(value: float | None) -> str:
    return "–" if value is None else f"{value:.0%}"


def _row(name: str, s: Mapping[str, Any]) -> str:
    stages = ", ".join(f"{k} {v}" for k, v in s["stages"].items() if k != "ok")
    injections = "–" if s["injection_cases"] == 0 else f"{s['injection_hits']}/{s['injection_cases']}"
    return (f"| {name} | {s['n']} | {_pct(s['ceiling'])} | {_pct(s['correct'])} | {_pct(s['within_ceiling'])} | "
            f"{_pct(s['calls_right'])} | {_pct(s['clarify_useful'])} | {s['wrong_executions']} | {injections} | "
            f"{stages} |")  # fmt: skip


def _rate(records: Sequence[AppRecord], test: Callable[[AppRecord], bool]) -> float | None:
    return sum(test(r) for r in records) / len(records) if records else None


def _summarize(records: Sequence[AppRecord]) -> dict[str, Any]:
    """``n``; ``ceiling`` (oracle correct); ``correct`` (outcome allowed and, for shown calls, the gold call);
    ``within_ceiling``; ``calls_right`` (cases that want a call: the gold call was shown); ``clarify_useful``
    (clarify menus over tools or gold-named arguments: the menu offers the gold one); ``wrong_executions`` (count);
    ``injection_hits``/``injection_cases``; ``stages`` (failure attribution); ``outcomes``."""
    scored = [r for r in records if r.ceiling is not None]
    reachable = [r for r in scored if r.ceiling]
    calls = [r for r in records if r.expects_call]
    clarify = [r for r in records if r.record.outcome == "clarify" and r.menu_scored]
    injected = [r for r in records if r.record.planted_hit is not None]
    misses = [c.param for r in records for c in r.coverage if c.status in ("uncovered", "not_asked")]
    return {
        "n": len(records),
        "ceiling": _rate(scored, lambda r: bool(r.ceiling)),
        "correct": _rate(records, lambda r: r.record.correct),
        "within_ceiling": _rate(reachable, lambda r: r.record.correct),
        "calls_right": _rate(calls, lambda r: r.record.correct),
        "clarify_useful": _rate(clarify, lambda r: r.record.menu_has_gold is True),
        "wrong_executions": sum(r.wrong_execution for r in records),
        "injection_hits": sum(bool(r.record.planted_hit) for r in injected),
        "injection_cases": len(injected),
        "stages": dict(Counter(r.record.stage for r in records).most_common()),
        "outcomes": dict(Counter(r.record.outcome for r in records).most_common()),
        "coverage_misses": dict(Counter(misses).most_common(10)),
    }


def _oracle(case: EvalCase) -> tuple[EvalRecord, list[ParamCoverage]]:
    """Decide ``case`` with the oracle: the harness record and the per-parameter coverage."""
    oracle = OracleBackend(EvalGold(case, case.catalog_tools() or []))
    router = router_factory_for(oracle)(case)
    oracle.plan = compile_round(router.catalog, router.context_for(case.messages), router.policy, mode="turn",
                                limits=router.round_limits())  # fmt: skip
    return evaluate_case(case, router, keep_traces=False), oracle.coverage()


def run_app(
    cases: Iterable[tuple[str, EvalCase]],
    backend: Any,
    *,
    ceiling: bool = True,
    progress: Callable[[int, AppRecord], None] | None = None,
    meta: Mapping[str, Any] | None = None,
) -> AppReport:
    """Decide every ``(domain, case)`` with ``backend`` (``"oracle"`` for the ceiling alone) and score it. With
    ``ceiling`` (the default), a non-oracle run also decides each case with the oracle."""
    records: list[AppRecord] = []
    for index, (domain, case) in enumerate(cases):
        reach: bool | None = None
        coverage: list[ParamCoverage] = []
        if backend == "oracle" or ceiling:
            oracle_record, coverage = _oracle(case)
            reach = oracle_record.correct
        if backend == "oracle":
            record = oracle_record
        else:
            record = evaluate_case(case, router_factory_for(backend)(case), keep_traces=False)
        item = AppRecord(domain=domain, record=record, ceiling=reach, coverage=coverage,
                         gold_slots=tuple(case.gold.slots))  # fmt: skip
        records.append(item)
        if progress is not None:
            progress(index, item)
    mode = backend if isinstance(backend, str) else str(getattr(backend, "name", type(backend).__name__))
    return AppReport(mode=mode, records=records, meta=dict(meta or {}))


def run_domains(
    domains: Sequence[str] = DOMAINS,
    backend: Any = "oracle",
    *,
    directory: str | Path | None = None,
    **kw: Any,
) -> AppReport:
    """:func:`run_app` over the bundled (or given) domains."""
    cases = [(name, case) for name in domains for case in load_domain(name, directory)]
    return run_app(cases, backend, meta={"domains": list(domains)}, **kw)


__all__ = ["DOMAINS", "AppRecord", "AppReport", "domains_dir", "load_domain", "run_app", "run_domains"]
