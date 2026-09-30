"""When2Call (NVIDIA, CC-BY-4.0): when to call a tool, ask for missing information, or decline.

The test set (``test/when2call_test_mcq.jsonl``, 3,652 cases) is built from BFCL's live categories: for each question
the right move is a tool call (``tool_call``), a follow-up question because a required value is missing
(``request_for_info``), or declining because no available tool fits (``cannot_answer``). The published evaluation
scores a language model's log-probabilities over four written answers; jevtools writes no answers, so here each
decision is mapped to the category it amounts to:

- ``execute`` or ``confirm`` with a call → ``tool_call`` (and whether the call names the target tool);
- ``clarify`` → ``request_for_info`` (a menu or an open question);
- ``abstain``, ``refuse``, ``escalate`` → ``cannot_answer`` (jevtools never answers directly, so the fourth option,
  ``direct``, is never predicted; it is never the correct answer in the test set either).

Reported: accuracy, per-category precision/recall/F1, macro F1, the confusion matrix, the share of predicted calls
that name the target tool, and the tool hallucination rate (a call proposed, or executed, on a ``cannot_answer``
case). A decision that fails closed at the backend is retried, then counted as an error, never as a decline.
"""

from __future__ import annotations

import json
import time
import urllib.request
from collections import Counter
from collections.abc import Callable, Iterable, Mapping
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from jevtools.bench.bfcl import to_openai_tool
from jevtools.context import Context
from jevtools.policy import RULE_FAIL_CLOSED
from jevtools.router import Router

URL = "https://huggingface.co/datasets/nvidia/When2Call/resolve/{ref}/test/when2call_test_mcq.jsonl"
FILE = "when2call_test_mcq.jsonl"
CATEGORIES = ("tool_call", "request_for_info", "cannot_answer")
NOW = datetime.fromisoformat("2026-09-25T10:00:00+00:00")
_CALL_OUTCOMES = frozenset({"execute", "confirm"})


@dataclass(frozen=True)
class W2CCase:
    """One test case: the question, the available tools (OpenAI format), the right category and target tool."""

    id: str
    source: str
    question: str
    answer: str
    tools: list[dict[str, Any]]
    target: str | None


def download(directory: str | Path, *, ref: str = "main") -> Path:
    """Fetch the MCQ test file from Hugging Face into ``directory``; returns its path."""
    path = Path(directory) / FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(URL.format(ref=ref), timeout=120) as response:  # noqa: S310 - fixed https URL
        path.write_bytes(response.read())
    return path


def load(path: str | Path, *, limit: int | None = None, per_category: int | None = None) -> list[W2CCase]:
    """The cases of the MCQ test file (``per_category``: at most N of each correct answer, in file order)."""
    out: list[W2CCase] = []
    counts: Counter[str] = Counter()
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        answer = str(row["correct_answer"])
        if per_category is not None and counts[answer] >= per_category:
            continue
        counts[answer] += 1
        tools = [to_openai_tool(json.loads(t) if isinstance(t, str) else t) for t in row.get("tools") or []]
        target = row.get("target_tool")
        target_doc = json.loads(target) if isinstance(target, str) else target
        out.append(W2CCase(id=str(row["uuid"]), source=str(row.get("source", "")), question=str(row["question"]),
                           answer=answer, tools=tools,
                           target=str(target_doc["name"]) if isinstance(target_doc, Mapping) else None))  # fmt: skip
        if limit is not None and len(out) >= limit:
            break
    return out


def category_of(outcome: str, has_call: bool) -> str:
    """The When2Call category a jevtools decision amounts to."""
    if outcome in _CALL_OUTCOMES and has_call:
        return "tool_call"
    if outcome == "clarify":
        return "request_for_info"
    return "cannot_answer"


@dataclass
class W2CRecord:
    """The result of one case."""

    id: str
    answer: str
    predicted: str
    outcome: str
    rule: str
    tool: str | None = None
    target: str | None = None
    executed: bool = False
    input_tokens: int = 0
    cost_usd: float | None = None
    error: str | None = None

    @property
    def correct(self) -> bool:
        return self.error is None and self.predicted == self.answer

    @property
    def right_tool(self) -> bool | None:
        """For a predicted call on a ``tool_call`` case: whether it names the target tool."""
        if self.predicted != "tool_call" or self.answer != "tool_call":
            return None
        return self.tool == self.target


class BackendFailure(Exception):
    """A live decision failed closed at the backend (P0): an outage, not a decline."""


def run_case(case: W2CCase, backend: Any, *, risk: str | None = "read") -> W2CRecord:
    """Decide one case and map the decision to a When2Call category."""
    tools = [dict(t, function=dict(t["function"])) for t in case.tools]
    if risk is not None:
        for tool in tools:
            tool["function"]["x-jev"] = {"risk": risk}
    try:
        router = Router(tools, backend=backend, context=Context(now=NOW, tz="UTC", locale="en"))
        decision = router.decide(case.question)
        if decision.rule == RULE_FAIL_CLOSED:
            raise BackendFailure(decision.trace.notes[-1] if decision.trace and decision.trace.notes else "")
    except Exception as exc:  # noqa: BLE001 - a crash is a failed case, reported with its message
        return W2CRecord(id=case.id, answer=case.answer, predicted="error", outcome="error", rule="",
                         target=case.target, error=f"{type(exc).__name__}: {exc}"[:300])  # fmt: skip
    call = decision.call
    outcome = str(decision.outcome)
    return W2CRecord(
        id=case.id, answer=case.answer, predicted=category_of(outcome, call is not None), outcome=outcome,
        rule=decision.rule, tool=call.name if call is not None else None, target=case.target,
        executed=outcome == "execute" and call is not None, input_tokens=decision.usage.jev_input_tokens,
        cost_usd=decision.usage.cost_usd,
    )  # fmt: skip


@dataclass
class W2CReport:
    mode: str
    records: list[W2CRecord]
    meta: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> dict[str, Any]:
        """Accuracy, per-category precision/recall/F1, macro F1, confusion, tool naming and hallucination."""
        recs = self.records
        n = len(recs)
        per: dict[str, dict[str, float | int | None]] = {}
        for cat in CATEGORIES:
            tp = sum(r.answer == cat and r.predicted == cat for r in recs)
            predicted = sum(r.predicted == cat for r in recs)
            actual = sum(r.answer == cat for r in recs)
            precision = tp / predicted if predicted else None
            recall = tp / actual if actual else None
            f1 = (2 * precision * recall / (precision + recall)) if precision and recall else 0.0
            per[cat] = {"n": actual, "precision": precision, "recall": recall, "f1": f1}
        cannot = [r for r in recs if r.answer == "cannot_answer"]
        named = [r.right_tool for r in recs if r.right_tool is not None]
        return {
            "n": n,
            "accuracy": sum(r.correct for r in recs) / n if n else None,
            "macro_f1": sum(float(per[c]["f1"] or 0.0) for c in CATEGORIES) / len(CATEGORIES),
            "per_category": per,
            "confusion": {a: dict(Counter(r.predicted for r in recs if r.answer == a)) for a in CATEGORIES},
            "right_tool": sum(named) / len(named) if named else None,
            "hallucination_proposed": sum(r.predicted == "tool_call" for r in cannot) / len(cannot) if cannot else None,
            "hallucination_executed": sum(r.executed for r in cannot) / len(cannot) if cannot else None,
            "errors": sum(r.error is not None for r in recs),
            "cost_usd": sum(r.cost_usd or 0.0 for r in recs),
            "input_tokens": sum(r.input_tokens for r in recs),
        }

    def render(self) -> str:
        s = self.summary()

        def pct(v: Any) -> str:
            return "–" if v is None else f"{v:.1%}"

        rows = [f"When2Call: {s['n']} case(s), accuracy {pct(s['accuracy'])}, macro F1 {pct(s['macro_f1'])}",
                "", "| answer | n | precision | recall | F1 | predicted tool_call | request_for_info | cannot_answer |",
                "|---|---:|---:|---:|---:|---:|---:|---:|"]  # fmt: skip
        for cat in CATEGORIES:
            p, conf = s["per_category"][cat], s["confusion"][cat]
            rows.append(f"| {cat} | {p['n']} | {pct(p['precision'])} | {pct(p['recall'])} | {pct(p['f1'])} | "
                        f"{conf.get('tool_call', 0)} | {conf.get('request_for_info', 0)} | "
                        f"{conf.get('cannot_answer', 0)} |")  # fmt: skip
        rows += ["", f"calls naming the target tool: {pct(s['right_tool'])}; tool hallucination on cannot_answer: "
                     f"{pct(s['hallucination_proposed'])} proposed, {pct(s['hallucination_executed'])} executed; "
                     f"errors {s['errors']}"]  # fmt: skip
        return "\n".join(rows)

    def to_json(self) -> str:
        return json.dumps({"mode": self.mode, "meta": self.meta, "summary": self.summary(),
                           "records": [asdict(r) for r in self.records]}, indent=1)  # fmt: skip


def run_when2call(
    cases: Iterable[W2CCase],
    backend: Any,
    *,
    risk: str | None = "read",
    retries: int = 0,
    progress: Callable[[int, W2CRecord], None] | None = None,
    meta: Mapping[str, Any] | None = None,
) -> W2CReport:
    """Decide every case (retrying, with backoff, a decision that failed closed at the backend)."""
    records: list[W2CRecord] = []
    for index, case in enumerate(cases):
        record = run_case(case, backend, risk=risk)
        for attempt in range(retries):
            if not (record.error or "").startswith("BackendFailure"):
                break
            time.sleep(5.0 * 4**attempt)
            record = run_case(case, backend, risk=risk)
        records.append(record)
        if progress is not None:
            progress(index, record)
    mode = str(getattr(backend, "name", type(backend).__name__))
    return W2CReport(mode=mode, records=records, meta={"risk": risk, **dict(meta or {})})


__all__ = ["CATEGORIES", "W2CCase", "W2CRecord", "W2CReport", "category_of", "download", "load", "run_when2call"]
