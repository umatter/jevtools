"""Shared helpers of the runnable examples: backend selection (``--backend scripted|sim|live``) and compact printing
of decisions, prompts, traces and agent-loop runs.

- ``scripted`` (default) replays ``examples/fixtures/<name>.answers.json`` — the spec's illustrative [I] numbers;
- ``sim`` uses the offline ``LexicalSimulator`` (a lexical test double, not a model);
- ``live`` uses :func:`jevtools.backends.auto` and needs ``TYPESAFE_API_KEY`` or ``OPENROUTER_API_KEY``.

Scripted and simulated answers are never evidence about Jev's accuracy.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import textwrap
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jevtools as jt
from jevtools.backends import Backend, BackendConfigError, LexicalSimulator, ScriptedBackend
from jevtools.wire import DecisionRequest, DecisionResponse

EXAMPLES = Path(__file__).resolve().parent
FIXTURES = EXAMPLES / "fixtures"
BACKENDS = ("scripted", "sim", "live")
WIDTH = 120
KEY = 12
"""Width of the key column."""
NOT_EVIDENCE = "Scripted and simulated answers are never evidence about Jev's accuracy."


@dataclass(frozen=True)
class Args:
    """Parsed command line of an example."""

    backend: str


def parse_args(doc: str | None, argv: Sequence[str] | None = None) -> Args:
    """``--backend scripted|sim|live`` (default ``scripted``)."""
    first = (doc or "").strip().splitlines()[0] if doc else None
    parser = argparse.ArgumentParser(description=first)
    parser.add_argument(
        "--backend",
        choices=BACKENDS,
        default="scripted",
        help="scripted: replay examples/fixtures (default); sim: LexicalSimulator; live: jevtools.backends.auto()",
    )
    ns = parser.parse_args(list(argv) if argv is not None else None)
    if ns.backend == "live":
        backend("live", "")  # fail fast (exit 2) without a key; builds no connection
    return Args(backend=str(ns.backend))


def fixture(name: str) -> Path:
    """``examples/fixtures/<name>.answers.json``."""
    return FIXTURES / f"{name}.answers.json"


def backend(kind: str, name: str) -> Backend:
    """A fresh backend: the ``name`` fixture (scripted), the simulator (sim) or :func:`jevtools.backends.auto`
    (live). Without a Jev key, ``live`` exits with status 2 and a hint."""
    if kind == "scripted":
        return ScriptedBackend.from_fixture(fixture(name))
    if kind == "sim":
        return LexicalSimulator()
    if kind == "live":
        try:
            return jt.backends.auto()
        except BackendConfigError as exc:
            print(f"--backend live: {exc}", file=sys.stderr)
            raise SystemExit(2) from exc
    raise ValueError(f"unknown backend {kind!r} (expected one of {', '.join(BACKENDS)})")


class Recording:
    """Wraps a backend and keeps every request it answers (to list the questions of an adapter call, whose result is
    a plain document rather than a :class:`~jevtools.Decision`)."""

    def __init__(self, inner: Backend) -> None:
        self.inner = inner
        self.model = inner.model
        self.name = inner.name
        self.requests: list[DecisionRequest] = []

    def decide(self, request: DecisionRequest) -> DecisionResponse:
        self.requests.append(request)
        return self.inner.decide(request)

    async def adecide(self, request: DecisionRequest) -> DecisionResponse:
        self.requests.append(request)
        return await self.inner.adecide(request)


# --------------------------------------------------------------------------------------------------------------------
# printing
# --------------------------------------------------------------------------------------------------------------------


def out(text: str = "") -> None:
    """Print one line."""
    print(text)


def short(value: Any, limit: int = 60) -> str:
    """A one-line form of ``value`` (newlines escaped), cut to ``limit`` characters with "…"."""
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, default=str)
    text = text.replace("\n", "\\n")
    return text if len(text) <= limit else text[: max(1, limit - 1)] + "…"


def kv(key: str, value: str, *, indent: int = 2, wrap: bool = True) -> None:
    """``key`` in a fixed column and ``value`` wrapped under it (``wrap=False``: cut to one line)."""
    lead = " " * indent + key.ljust(KEY)
    if not wrap:
        out(lead + short(value, WIDTH - len(lead)))
        return
    lines = textwrap.wrap(value, width=WIDTH - len(lead), break_long_words=False, break_on_hyphens=False) or [""]
    out(lead + lines[0])
    for line in lines[1:]:
        out(" " * len(lead) + line)


def header(title: str, kind: str, fixtures: Iterable[str] = ()) -> None:
    """The example's title and which backend answers."""
    out("━" * WIDTH)
    out(title)
    out("━" * WIDTH)
    if kind == "scripted":
        names = list(fixtures)
        files = names[0] if len(names) == 1 else "{" + ",".join(names) + "}"
        kv("backend", f"scripted · replays examples/fixtures/{files}.answers.json (illustrative [I] numbers)",
           indent=0)  # fmt: skip
    elif kind == "sim":
        kv("backend", "sim · LexicalSimulator, an offline lexical test double (not a model)", indent=0)
    else:
        kv("backend", "live · jevtools.backends.auto() (TypeSafe or OpenRouter)", indent=0)
    if kind != "live":
        kv("", NOT_EVIDENCE, indent=0)


def step(title: str) -> None:
    """A section title."""
    out()
    out(f"▸ {title}")


def note(text: str) -> None:
    """A remark about what the output shows."""
    kv("note", text)


def call_text(call: jt.ToolCall | None, limit: int = 64) -> str:
    """``name(arg=value, …)`` with long values cut."""
    if call is None:
        return "–"
    args = ", ".join(f"{k}={short(json.dumps(v, ensure_ascii=False), limit)}" for k, v in call.arguments.items())
    return f"{call.name}({args})"


def _ranges(rests: Sequence[str]) -> list[str]:
    """Collapse runs such as ``accept.0``, ``accept.1``, ``accept.2`` into ``accept.0-2``."""
    out_: list[str] = []
    i = 0
    while i < len(rests):
        m = re.fullmatch(r"(.*\.)(\d+)", rests[i])
        if m is None:
            out_.append(rests[i])
            i += 1
            continue
        prefix, start = m.group(1), int(m.group(2))
        end = start
        while i + 1 < len(rests) and rests[i + 1] == f"{prefix}{end + 1}":
            end += 1
            i += 1
        out_.append(f"{prefix}{start}" if end == start else f"{prefix}{start}-{end}")
        i += 1
    return out_


def compact_qids(qids: Sequence[str]) -> str:
    """Question ids grouped by tool: ``tool · send_email: authorized, to, subject.accept.0-2``."""
    groups: dict[str, list[str]] = {}
    for qid in qids:
        head, _, rest = qid.partition(".")
        groups.setdefault(head, []).append(rest)
    parts = []
    for head, rests in groups.items():
        named = [r for r in rests if r]
        if len(named) < len(rests):
            parts.append(head)
        if named:
            parts.append(f"{head}: {', '.join(_ranges(named))}")
    return " · ".join(parts)


def rounds(decision: jt.Decision) -> list[tuple[int, str, list[list[str]]]]:
    """``(round, mode, [question ids per Jev call])`` from the decision's trace."""
    trace = decision.trace
    found: list[tuple[int, str, list[list[str]]]] = []
    for record in getattr(trace, "rounds", None) or []:
        calls = [list((c.request or {}).get("questions", {})) for c in record.calls]
        found.append((record.round, record.mode, calls))
    return found


def _num(x: float | None) -> str:
    return "–" if x is None else f"{x:.3f}"


def confidence_text(c: jt.Confidence) -> str:
    """``C 0.714 (PI, external) · W … · PI … · L … · J … · execute ≥ … · confirm ≥ …``."""
    bands = []
    if c.execute_at is not None:
        bands.append(f"execute ≥ {c.execute_at:.2f}")
    elif c.tier == "critical":
        bands.append("never auto-executes")
    if c.confirm_at is not None:
        bands.append(f"confirm ≥ {c.confirm_at:.2f}")
    calibrated = ", calibrated" if c.calibrated else ""
    head = f"C {c.call:.3f} ({c.composition}, {c.tier}{calibrated})"
    return " · ".join([head, f"W {_num(c.W)}", f"PI {_num(c.PI)}", f"L {_num(c.L)}", f"J {_num(c.J)}", *bands])


def decision(d: jt.Decision, *, slots: bool = True) -> None:
    """Questions asked, the decision (outcome, rule, call, confidence), the rendered prompt and a trace summary."""
    found = rounds(d)
    if not found:
        clicked = " (a click on a prompt option)" if getattr(d.trace, "resumed_from", None) else ""
        kv("questions", f"0 — no Jev call{clicked}")
    for number, mode, calls in found:
        qids = [q for qids in calls for q in qids]
        where = f"{len(calls)} Jev calls" if len(calls) != 1 else "1 Jev call"
        kv("questions", f"round {number} ({mode}): {len(qids)} in {where} — {compact_qids(qids)}")
    what = ""
    if d.tool_calls:
        what = " — emitted as tool_calls[0]"
    elif d.call is not None:
        what = " — not executed: the proposed call waits for the user"
    kv("outcome", f"{d.outcome.value.upper()}  ({d.rule}){what}")
    if d.call is not None:
        kv("call", call_text(d.call), wrap=False)
    if d.confidence is not None:
        kv("confidence", confidence_text(d.confidence))
    extras = []
    if d.bottleneck is not None:
        extras.append(f"bottleneck {d.bottleneck.slot} ({d.bottleneck.shape})")
    if d.gates:
        extras.append("gates " + ", ".join(f"{k} {v:.2f}" for k, v in d.gates.items()))
    if d.flags:
        extras.append("flags " + ", ".join(d.flags))
    if extras:
        kv("signals", " · ".join(extras))
    if slots:
        for name, report in d.slots.items():
            alts = ", ".join(f"{short(a.value, 34)} {a.p:.2f}" for a in report.alternatives)
            channel = f" [{report.channel}]" if report.channel else ""
            text = f"{name} = {short(report.value, 56)}  p {report.p:.2f}{channel}"
            kv("slot", text + (f"  alt: {alts}" if alts else ""), wrap=False)
    prompt(d)
    if d.content:
        kv("content", short(d.content, 200))
    trace(d)


def prompt(d: jt.Decision) -> None:
    """The rendered prompt and its options (one per line)."""
    p = d.prompt
    if p is None:
        return
    kv("prompt", f"{p.kind}: {short(p.text, 200)}")
    for i, option in enumerate(p.options, 1):
        kv("", f"{i}. [{option.id}] {option.text}", wrap=False)


def trace(d: jt.Decision) -> None:
    """``trace_id · rounds · Jev calls · input tokens`` plus ids and notes."""
    t = d.trace
    parts = [str(getattr(t, "trace_id", d.trace_id))]
    parts.append(f"{d.rounds} round" + ("s" if d.rounds != 1 else ""))
    parts.append(f"{d.usage.jev_calls} Jev call" + ("s" if d.usage.jev_calls != 1 else ""))
    parts.append(f"{d.usage.jev_input_tokens:,} input tokens")
    if d.usage.llm_calls:
        parts.append(f"{d.usage.llm_calls} LLM call" + ("s" if d.usage.llm_calls != 1 else ""))
    if d.usage.cost_usd is not None:
        parts.append(f"${d.usage.cost_usd:.6f}")
    if d.pending_id:
        parts.append(f"pending {d.pending_id}")
    resumed = getattr(t, "resumed_from", None)
    if resumed:
        parts.append(f"resumed from {resumed}")
    if d.tool_calls:
        parts.append(f"idempotency key {d.tool_calls[0].idempotency_key}")
    kv("trace", " · ".join(parts))
    for text in getattr(t, "notes", None) or []:
        kv("", f"note: {short(text, 160)}")


def message(doc: Mapping[str, Any], *, title: str = "openai", limit: int = 400) -> None:
    """An OpenAI message (or any JSON document): one entry per top-level key, values as JSON."""
    for i, (key, value) in enumerate(doc.items()):
        kv(title if i == 0 else "", f"{key}: {short(json.dumps(value, ensure_ascii=False), limit)}")


def pretty_json(value: Any, *, indent: int = 0, width: int = WIDTH, column: int | None = None) -> str:
    """JSON with key order kept, laid out like the spec's listings: a container stays on one line when it fits
    within ``width`` (from ``column``, where it starts), otherwise its items go one per line, indented."""
    flat = json.dumps(value, ensure_ascii=False)
    start = indent if column is None else column
    if start + len(flat) <= width or not isinstance(value, (dict, list)) or not value:
        return flat
    pad = " " * (indent + 2)
    if isinstance(value, dict):
        items = []
        for k, v in value.items():
            key = json.dumps(k, ensure_ascii=False) + ": "
            items.append(pad + key + pretty_json(v, indent=indent + 2, width=width, column=len(pad) + len(key)))
        return "{\n" + ",\n".join(items) + "\n" + " " * indent + "}"
    items = [pad + pretty_json(v, indent=indent + 2, width=width) for v in value]
    return "[\n" + ",\n".join(items) + "\n" + " " * indent + "]"


def loop(result: jt.LoopResult) -> None:
    """An agent-loop run: its steps, the final outcome and usage."""
    for s in result.steps:
        ran = f"ran: {s.status or 'ok'}" if s.executed else "not run"
        kv(
            "steps" if s is result.steps[0] else "",
            f"{s.step}. {s.outcome:<8} {ran:<9} {call_text(s.call, 56)}",
            wrap=False,
        )
        if s.note:
            kv("", f"   note: {short(s.note, 140)}")
    reason = f", reason {result.reason}" if result.reason else ""
    kv("outcome", f"{result.outcome.value.upper()}  ({result.rule}{reason})")
    u = result.usage
    kv("usage", f"{u.steps} steps · {u.rounds} Jev rounds · {u.jev_calls} Jev calls · {u.executions} executions · "
                f"{u.jev_input_tokens:,} input tokens · {u.llm_calls} LLM calls")  # fmt: skip
    for text in result.notes:
        kv("", f"note: {short(text, 160)}")


def verified(d: jt.Decision, router: jt.Router, context: jt.Context) -> bool:
    """``jt.verify`` of the decision's trace (model out of the loop), printed as one line."""
    report = jt.verify(d.trace, catalog=router.catalog, context=context)
    checks = ", ".join(c.name for c in report.checks if c.ok)
    failures = "; ".join(f"{c.name}: {c.detail}" for c in report.failures)
    kv("verify", "ok — " + checks if report.ok else "FAILED — " + failures)
    return bool(report.ok)


def first_option(d: jt.Decision, prefer: Sequence[str] = ("ok",)) -> str | None:
    """The option to click in a non-interactive run: the first of ``prefer`` offered, else the first option."""
    if d.prompt is None or not d.prompt.options:
        return None
    ids = [o.id for o in d.prompt.options]
    return next((p for p in prefer if p in ids), ids[0])


__all__ = [
    "BACKENDS",
    "EXAMPLES",
    "FIXTURES",
    "NOT_EVIDENCE",
    "Args",
    "Recording",
    "backend",
    "call_text",
    "compact_qids",
    "confidence_text",
    "decision",
    "first_option",
    "fixture",
    "header",
    "kv",
    "loop",
    "message",
    "note",
    "out",
    "parse_args",
    "pretty_json",
    "prompt",
    "rounds",
    "short",
    "step",
    "trace",
    "verified",
]
