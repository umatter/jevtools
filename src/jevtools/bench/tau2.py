"""τ²-bench (Sierra, MIT), reduced to single decisions: given the conversation so far and the results of the tool
calls made so far, what is the next tool call?

τ²-bench runs full customer-service agents against an LLM-simulated customer, which measures what jevtools does
not do (writing conversation text, following a written policy). Its repository also publishes complete successful
conversations (``data/tau2/results/final/*.json``: a simulated customer talking to reference agents, with every
tool call, its exact arguments and its actual output). This module takes one successful conversation per task and
turns it into cases:

- a **call** case per tool call the reference agent made: the user and assistant turns so far as ``messages``, the
  earlier tool results as ``observations``, and the call as gold (tool plus every argument, id lists compared as
  sets); ``proposal`` counts a call shown on a card or menu, ``strict`` only an executed one;
- a **talk** case per assistant turn without a tool call (asking for details, asking for confirmation): right
  unless jevtools *executes* a call there.

jevtools sees the tool schemas and the conversation, not the domain policy text. ``trust`` chooses which tools'
results are first-party data (``Context.trusted_tools``): ``none`` (the default barrier: nothing from a tool result
reaches a write tool's identity slots) or ``reads`` (the lookup tools, ``get_*``, ``find_*``, ``list_*`` and
``search_*``, whose results are the app's own records).
"""

from __future__ import annotations

import json
import time
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from jevtools.context import Context, Observation
from jevtools.policy import RULE_FAIL_CLOSED
from jevtools.router import Router

DOMAINS = ("retail", "airline")
RESULTS = "data/tau2/results/final/claude-3-7-sonnet-20250219_{domain}_default_gpt-4.1-2025-04-14_4trials.json"
NOW = datetime.fromisoformat("2024-05-15T15:00:00-05:00")
"""The airline policy's "current time"; retail states none, so both domains use it."""
READ_PREFIXES = ("get_", "find_", "list_", "search_")
_CALL_OUTCOMES = frozenset({"execute", "confirm"})


@dataclass(frozen=True)
class Tau2Case:
    id: str
    domain: str
    kind: str
    """``call`` or ``talk``."""
    messages: list[dict[str, str]]
    observations: list[dict[str, Any]]
    gold: dict[str, Any] | None
    """``{"name", "arguments"}`` for a call case, ``None`` for a talk case."""

    @property
    def read(self) -> bool:
        return self.gold is not None and str(self.gold["name"]).startswith(READ_PREFIXES)


def inline_refs(schema: Any, defs: Mapping[str, Any] | None = None) -> Any:
    """A JSON Schema with its local ``$ref``s to ``$defs`` replaced by the definitions (τ²'s schemas come from
    pydantic)."""
    if isinstance(schema, dict):
        defs = {**(defs or {}), **schema.get("$defs", {})}
        ref = schema.get("$ref")
        if isinstance(ref, str) and ref.startswith("#/$defs/"):
            return inline_refs(deepcopy(defs[ref.rsplit("/", 1)[1]]), defs)
        return {k: inline_refs(v, defs) for k, v in schema.items() if k != "$defs"}
    if isinstance(schema, list):
        return [inline_refs(v, defs) for v in schema]
    return schema


EXPORT_TOOLS = (
    "from tau2.domains.{domain}.environment import get_environment; import json; "
    "json.dump([t.openai_schema for t in get_environment().get_tools()], open('{domain}_tools.json', 'w'))"
)
"""Run with τ²'s own Python (``python -c ...``) to write ``{domain}_tools.json``: the published results do not
include the tool schemas."""


def load_tools(data_dir: str | Path, domain: str) -> list[dict[str, Any]]:
    """The domain's tools (``{domain}_tools.json`` exported from a τ² install) with ``$ref``s inlined."""
    return [inline_refs(t) for t in json.loads((Path(data_dir) / f"{domain}_tools.json").read_text(encoding="utf-8"))]


def _content(text: Any) -> Any:
    if not isinstance(text, str):
        return text
    try:
        return json.loads(text)
    except ValueError:
        return text


def build_cases(tau2_dir: str | Path, domain: str) -> list[Tau2Case]:
    """Cases from the first successful published conversation of every task of ``domain``."""
    doc = json.loads((Path(tau2_dir) / RESULTS.format(domain=domain)).read_text(encoding="utf-8"))
    chosen: dict[str, dict[str, Any]] = {}
    for sim in sorted(doc["simulations"], key=lambda s: (str(s["task_id"]), s.get("trial", 0))):
        if (sim.get("reward_info") or {}).get("reward") == 1.0 and str(sim["task_id"]) not in chosen:
            chosen[str(sim["task_id"])] = sim
    out: list[Tau2Case] = []
    for task_id, sim in sorted(chosen.items(), key=lambda kv: (len(kv[0]), kv[0])):
        turns: list[dict[str, str]] = []
        observations: list[dict[str, Any]] = []
        pending: dict[str, dict[str, Any]] = {}
        n = 0
        for message in sim["messages"]:
            role = message.get("role")
            if role == "tool":
                call = pending.pop(str(message.get("id")), None) or {}
                observations.append({
                    "step": len(observations) + 1, "tool": call.get("name", "?"),
                    "arguments": call.get("arguments", {}),
                    "content": _content(message.get("content")), "status": "error" if message.get("error") else "ok",
                })  # fmt: skip
                continue
            calls = message.get("tool_calls") or []
            if role == "assistant" and calls:
                seen_obs = list(observations)
                for call in calls:
                    if turns and turns[-1]["role"] == "user":
                        n += 1
                        out.append(Tau2Case(id=f"{domain}-{task_id}-{n}", domain=domain, kind="call",
                                            messages=list(turns), observations=list(seen_obs),
                                            gold={"name": call["name"], "arguments": call["arguments"]}))  # fmt: skip
                    pending[str(call.get("id"))] = call
            elif role == "assistant" and message.get("content") and turns and turns[-1]["role"] == "user":
                n += 1
                out.append(Tau2Case(id=f"{domain}-{task_id}-{n}", domain=domain, kind="talk", messages=list(turns),
                                    observations=list(observations), gold=None))  # fmt: skip
            if role in ("user", "assistant") and message.get("content"):
                turns.append({"role": role, "content": str(message["content"])})
    return out


def _norm(value: Any) -> Any:
    if isinstance(value, list):
        items = [_norm(v) for v in value]
        return sorted(items, key=lambda v: json.dumps(v, sort_keys=True))
    if isinstance(value, dict):
        return {k: _norm(v) for k, v in value.items()}
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return value


def matches(gold: Mapping[str, Any], call: Mapping[str, Any] | None) -> bool:
    """The call is the gold call: same tool and every gold argument equal (lists compared as sets; extra arguments a
    schema default fills are ignored)."""
    if call is None or call.get("name") != gold["name"]:
        return False
    args = call.get("arguments") or {}
    return all(_norm(args.get(k)) == _norm(v) for k, v in (gold.get("arguments") or {}).items())


@dataclass
class Tau2Record:
    id: str
    domain: str
    kind: str
    read: bool
    outcome: str
    rule: str
    call: dict[str, Any] | None = None
    gold: dict[str, Any] | None = None
    proposal: bool = False
    strict: bool = False
    right_tool: bool | None = None
    input_tokens: int = 0
    cost_usd: float | None = None
    error: str | None = None


def run_case(case: Tau2Case, tools: Sequence[dict[str, Any]], backend: Any, *, trust: str = "none") -> Tau2Record:
    trusted = tuple(t["function"]["name"] for t in tools if t["function"]["name"].startswith(READ_PREFIXES)) \
        if trust == "reads" else ()  # fmt: skip
    try:
        ctx = Context(messages=case.messages, now=NOW, tz="America/New_York", locale="en", trusted_tools=trusted,
                      observations=[Observation(**o) for o in case.observations])  # fmt: skip
        router = Router([deepcopy(t) for t in tools], backend=backend, context=ctx)
        decision = router.decide(case.messages, mode="loop" if case.observations else "turn")
        if decision.rule == RULE_FAIL_CLOSED:
            raise BackendFailure("")
    except Exception as exc:  # noqa: BLE001 - a crash is a failed case
        return Tau2Record(id=case.id, domain=case.domain, kind=case.kind, read=case.read, outcome="error", rule="",
                          gold=case.gold, error=f"{type(exc).__name__}: {exc}"[:300])  # fmt: skip
    call = {"name": decision.call.name, "arguments": dict(decision.call.arguments)} if decision.call else None
    outcome = str(decision.outcome)
    shown = call if outcome in _CALL_OUTCOMES else None
    executed = call if outcome == "execute" else None
    record = Tau2Record(id=case.id, domain=case.domain, kind=case.kind, read=case.read, outcome=outcome,
                        rule=decision.rule, call=call, gold=case.gold, input_tokens=decision.usage.jev_input_tokens,
                        cost_usd=decision.usage.cost_usd)  # fmt: skip
    if case.gold is not None:
        record.proposal = matches(case.gold, shown)
        record.strict = matches(case.gold, executed)
        record.right_tool = call is not None and call["name"] == case.gold["name"]
    else:
        record.proposal = record.strict = executed is None  # talk: right unless a call was executed
    return record


class BackendFailure(Exception):
    """A live decision failed closed at the backend (P0)."""


@dataclass
class Tau2Report:
    mode: str
    records: list[Tau2Record]
    meta: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> dict[str, dict[str, Any]]:
        groups: dict[str, list[Tau2Record]] = {}
        for r in self.records:
            part = "talk" if r.kind == "talk" else ("read" if r.read else "write")
            for key in (f"{r.domain}:{part}", f"ALL:{part}"):
                groups.setdefault(key, []).append(r)
        out: dict[str, dict[str, Any]] = {}
        for key, recs in sorted(groups.items()):
            n = len(recs)
            out[key] = {"n": n, "proposal": sum(r.proposal for r in recs) / n,
                        "strict": sum(r.strict for r in recs) / n,
                        "right_tool": (sum(bool(r.right_tool) for r in recs) / n) if recs[0].kind == "call" else None,
                        "outcomes": dict(Counter(r.outcome for r in recs).most_common()),
                        "refused": sum(r.rule.startswith("P3") for r in recs),
                        "errors": sum(r.error is not None for r in recs)}  # fmt: skip
        return out

    def render(self) -> str:
        def pct(v: Any) -> str:
            return "–" if v is None else f"{v:.1%}"

        rows = [
            "| part | n | proposal | strict | right tool | refused (P3) | outcomes |",
            "|---|---:|---:|---:|---:|---:|---|",
        ]
        for key, s in self.summary().items():
            outcomes = ", ".join(f"{k} {v}" for k, v in s["outcomes"].items())
            rows.append(f"| {key} | {s['n']} | {pct(s['proposal'])} | {pct(s['strict'])} | {pct(s['right_tool'])} | "
                        f"{s['refused']} | {outcomes} |")  # fmt: skip
        return "\n".join(rows)

    def to_json(self) -> str:
        return json.dumps({"mode": self.mode, "meta": self.meta, "summary": self.summary(),
                           "records": [asdict(r) for r in self.records]}, indent=1, default=str)  # fmt: skip


def run_tau2(
    cases: Iterable[Tau2Case],
    tools: Mapping[str, Sequence[dict[str, Any]]],
    backend: Any,
    *,
    trust: str = "none",
    retries: int = 0,
    progress: Callable[[int, Tau2Record], None] | None = None,
    meta: Mapping[str, Any] | None = None,
) -> Tau2Report:
    records: list[Tau2Record] = []
    for index, case in enumerate(cases):
        record = run_case(case, tools[case.domain], backend, trust=trust)
        for attempt in range(retries):
            if not (record.error or "").startswith("BackendFailure"):
                break
            time.sleep(5.0 * 4**attempt)
            record = run_case(case, tools[case.domain], backend, trust=trust)
        records.append(record)
        if progress is not None:
            progress(index, record)
    return Tau2Report(mode=str(getattr(backend, "name", type(backend).__name__)), records=records,
                      meta={"trust": trust, **dict(meta or {})})  # fmt: skip


def save_cases(cases: Sequence[Tau2Case], path: str | Path) -> None:
    Path(path).write_text("".join(json.dumps(asdict(c)) + "\n" for c in cases), encoding="utf-8")


def load_cases(path: str | Path) -> list[Tau2Case]:
    return [Tau2Case(**json.loads(line)) for line in Path(path).read_text(encoding="utf-8").splitlines() if line]


__all__ = ["DOMAINS", "EXPORT_TOOLS", "Tau2Case", "Tau2Record", "Tau2Report", "build_cases", "inline_refs",
           "load_cases", "load_tools", "matches", "run_tau2", "save_cases"]  # fmt: skip
