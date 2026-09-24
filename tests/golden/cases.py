"""Golden conformance fixtures (spec §10.2): the case definitions, the generator behind ``jevtools fixtures`` and the
data-driven replay that ``test_golden.py`` runs (and that a port such as the planned R package mirrors).

Every case is a directory ``tests/golden/<case>/`` (see ``README.md``):

- ``case.json``: the manifest — model, limits, context files, the steps (``decide``, ``resume``, ``agent_run``,
  ``agent_resume``), every Jev exchange in call order, the recorded tool executions of agent cases, the output files;
- ``catalog.json`` (the §13.2 OpenAI tools) and ``context.json`` (``context_<k>.json`` for a changed context): the
  inputs, whose sources reference the shared rows in ``tests/golden/fixtures/``;
- ``ballot.json``: the round-1 Ballot of the first step;
- ``request.json``/``response.json`` (one exchange) or ``request_<i>.json``/``response_<i>.json``: the Jev exchanges;
  a failed call's response is ``{"error": {"type", "status", "detail"}}``;
- ``decision.json``/``trace.json``: the final decision and its trace; earlier decisions are ``decision_<k>.json`` and
  ``trace_<k>.json``; agent cases add ``loop.json`` (the run's outcome and steps).

Outputs are canonical JSON (spec §3.1: UTF-8, NFC, no whitespace, keys in normative order, shortest numbers); traces
have ``created_at`` set to :data:`CREATED_AT` and ``latency_ms`` to 0 (their only non-deterministic fields). Inputs
and manifests are indented JSON for reading.

The Jev answers are the scripted §13.3 walk-through numbers (illustrative, [I]): the fixtures pin bytes, plumbing and
policy branches, and are **never evidence about Jev's accuracy**.
"""

from __future__ import annotations

import json
import shutil
import tempfile
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from jevtools._version import SPEC_VERSION
from jevtools.backends import errors as backend_errors
from jevtools.backends.errors import BackendError, JevValidationError
from jevtools.backends.scripted import Script, ScriptedBackend
from jevtools.ballot import Ballot
from jevtools.canonical import canonical_json, jsonable
from jevtools.cli import load_catalog, load_context
from jevtools.context import Context, Observation
from jevtools.decision import Decision, Pending
from jevtools.loop import Agent, EntityStore, LoopResult
from jevtools.policy import Policy
from jevtools.router import Router
from jevtools.trace import Trace
from jevtools.validate import Limits
from jevtools.wire import DecisionRequest, DecisionResponse
from tests.scenario import scripts
from tests.scenario.fixtures import (
    FILE_SYNONYMS,
    SCENARIO_MODEL,
    Workspace,
    account_rows,
    accounts,
    contact_rows,
    contacts,
    default_sources,
    files,
    scenario_context,
    scenario_tools,
    workspace_paths,
)

GOLDEN_DIR = Path(__file__).resolve().parent
"""``tests/golden``."""
SHARED = "fixtures"
"""The shared input rows (contacts, workspace paths), referenced by every ``context.json``."""
MODEL = SCENARIO_MODEL
"""The wire model id of every request (the OpenRouter Decisions id of §13.4)."""
BACKEND = "scripted"
"""The backend name recorded in traces (generation replays scripted answers; the test replays the exchanges)."""
CREATED_AT = "2026-09-24T12:05:00Z"
"""The normalized ``created_at`` of every trace (the scenario's ``now`` in UTC)."""
EVIDENCE = (
    "Scripted answers: the spec §13.3 illustrative numbers [I]. The fixtures pin bytes, plumbing and policy branches; "
    "they are never evidence about Jev's accuracy."
)
FIXTURE_SUFFIX = ".json"

# ====================================================================================================================
# Inputs
# ====================================================================================================================


def source_specs(balances: Mapping[str, float] | None = None) -> list[dict[str, Any]]:
    """The §13.1 sources as source specs (:mod:`jevtools.sources.specs`): the ``contacts`` and ``files`` rows live in
    the shared fixtures, the four accounts inline (``balances`` overrides some, for TOCTOU)."""
    return [
        {"name": "contacts", "type": "registry", "rows_file": f"../{SHARED}/contacts.json", "key": "email",
         "label": "{name} <{email}>", "describe": "{notes}", "match": ["name", "aliases", "team"],
         "provides": ["email", "person"], "attrs": ["name", "team"], "recency": "last", "hierarchy": "team"},
        {"name": "accounts", "type": "registry", "rows": account_rows(**(balances or {})), "key": "id",
         "label": "{nickname} · {currency} · {iban_masked}", "describe": "{nickname} account in {currency}",
         "match": ["nickname"], "provides": ["account_id"], "attrs": ["nickname", "currency", "balance"]},
        {"name": "files", "type": "files", "paths_file": f"../{SHARED}/files.json", "synonyms": FILE_SYNONYMS},
    ]  # fmt: skip


def shared_files() -> dict[str, bytes]:
    """``fixtures/contacts.json`` (500 rows) and ``fixtures/files.json`` (3,000 paths)."""
    return {f"{SHARED}/contacts.json": pretty(contact_rows()), f"{SHARED}/files.json": pretty(workspace_paths())}


@dataclass(frozen=True)
class ContextSpec:
    """One context file: the §13.1 context with a request (after the R2 history turns when ``history``),
    observations, and account balance overrides."""

    request: str
    history: bool = False
    observations: tuple[Observation, ...] = ()
    balances: Mapping[str, float] = field(default_factory=dict)

    def context(self) -> Context:
        """The in-memory context this file must reproduce (same ``context_sha256``)."""
        sources = [contacts(), accounts(account_rows(**self.balances)), files()] if self.balances else None
        return scenario_context(self.request, history=self.history, observations=self.observations,
                                sources=sources if sources is not None else list(default_sources()))  # fmt: skip

    def doc(self) -> dict[str, Any]:
        """The ``context.json`` document: :meth:`Context.to_doc` with source specs instead of source hashes."""
        doc = self.context().to_doc()
        doc.pop("entities", None)
        doc["sources"] = source_specs(self.balances)
        return doc


@dataclass(frozen=True)
class Step:
    """One step of a case: ``decide`` (a context file, a mode), ``resume`` (a click ``selection`` or a ``reply`` on
    the previous decision's pending, optionally with a changed context), ``agent_run``/``agent_resume``."""

    action: str
    context: str | None = None
    mode: str = "turn"
    selection: str | None = None
    reply: str | None = None

    def doc(self) -> dict[str, Any]:
        doc: dict[str, Any] = {"action": self.action}
        if self.context is not None:
            doc["context"] = self.context
        if self.action in ("decide", "agent_run"):
            doc["mode"] = self.mode
        for key in ("selection", "reply"):
            if getattr(self, key) is not None:
                doc[key] = getattr(self, key)
        return doc


Fault = Callable[[DecisionRequest, int], BackendError | None]


@dataclass(frozen=True)
class Case:
    """A golden case: its answers (``script``), context files, steps, limit overrides and an optional backend fault
    (``fault(request, index)`` returns the error to raise instead of answering)."""

    name: str
    summary: str
    script: Script
    contexts: Mapping[str, ContextSpec]
    steps: tuple[Step, ...]
    limits: Mapping[str, Any] = field(default_factory=dict)
    fault: Fault | None = None
    rebuild: str | None = None
    """Why ``verify`` must not rebuild the round-1 Ballot from the context (it compiles with default limits and no
    isolation), or ``None``: the trace is then verified without the context (re-decode, values, channels,
    composition and policy still run)."""


def _r6_observation() -> Observation:
    """Observation 1 of §6.6: the synthetic invoice (with its embedded instruction) read in step 1."""
    path, text = next(iter(Workspace().files.items()))
    return Observation(step=1, tool="read_file", arguments={"path": path}, content=text)


def _unit_422(request: DecisionRequest, index: int) -> BackendError | None:
    """The first call is rejected with a 422 naming ``get_weather.unit`` (isolation drops that family)."""
    if index == 0 and "get_weather.unit" in request.questions:
        loc = ["body", "questions", "get_weather.unit", "choice", "criteria"]
        return JevValidationError([{"loc": loc, "msg": "bad", "type": "value_error"}])
    return None


def _one(request: str, script: Script, summary: str, name: str, *, history: bool = False, mode: str = "turn",
         observations: tuple[Observation, ...] = (), **kw: Any) -> Case:  # fmt: skip
    ctx = ContextSpec(request, history=history, observations=observations)
    return Case(name, summary, script, {"context.json": ctx}, (Step("decide", "context.json", mode),), **kw)


def cases() -> list[Case]:
    """Every golden case of §10.2."""
    r2_ctx = ContextSpec(scripts.R2_REQUEST)
    r3_ctx = ContextSpec(scripts.R3_REQUEST)
    step2 = (_r6_observation(),)
    return [
        _one(scripts.R1_REQUEST, scripts.R1, "Read tier: weather in Zurich in Fahrenheit executes.", "R1"),
        _one(scripts.R2_REQUEST, scripts.R2, "External tier with history: confirm the email to Anna Keller.", "R2",
             history=True),
        _one(scripts.R2_REQUEST, scripts.R2_NO_HISTORY, "No history: clarify menu of three complete calls.",
             "R2-no-history"),
        Case("R2-click", "No history: the click on Anna Rossi binds and confirms (no Jev call) → execute.",
             scripts.R2_NO_HISTORY, {"context.json": r2_ctx},
             (Step("decide", "context.json"), Step("resume", selection="pick:to:1"))),
        _one(scripts.R3_REQUEST, scripts.R3, "Critical tier: joint Choice, L/J, confirm (never auto-executes).", "R3"),
        Case("R3-TOCTOU-changed", "Confirmed, but the Savings balance dropped to 100.00: TOCTOU re-plans.",
             scripts.R3, {"context.json": r3_ctx,
                          "context_2.json": ContextSpec(scripts.R3_REQUEST, balances={"acc_7731": 100.0})},
             (Step("decide", "context.json"), Step("resume", "context_2.json", selection="ok"))),
        _one(scripts.R4_REQUEST, scripts.R4, "Read tier over 3,000 files: the payments config executes.", "R4"),
        _one(scripts.R4_REQUEST, scripts.R4_MISS, "A miss widens (buckets + group), then the hierarchy round, then "
             "clarify(open).", "R4-widen"),
        _one(scripts.R5_REQUEST, scripts.R5, "External tier: confirm with the Tue 6 Oct alternative.", "R5"),
        Case("R6", "The agent loop: read the latest ACME invoice, confirm the forward, the click executes, done.",
             scripts.r6_script(), {"context.json": ContextSpec(scripts.R6_REQUEST)},
             (Step("agent_run", "context.json", "loop"), Step("agent_resume", selection="ok"))),
        _one(scripts.R6_REQUEST, scripts.r6_script(), "Loop step 1: superlative member Nouls, read_file executes.",
             "R6-step1", mode="loop"),
        _one(scripts.R6_REQUEST, scripts.r6_script(), "Loop step 2: forward to finance (confirm); the injected "
             "address is never nominated.", "R6-step2", mode="loop", observations=step2),
        _one(scripts.R6_REQUEST, scripts.r6_script(scripts.R6_REFUSE_STEP2), "Loop step 2 picks transfer_funds: "
             "the amount exists only in the invoice → refuse.", "R6-injection", mode="loop", observations=step2),
        _one(scripts.R7_REQUEST, scripts.R7, "Small talk: NO_TOOL → abstain.", "R7"),
        _one(scripts.R2_REQUEST, scripts.R2, "A 422 names get_weather.unit: the family is dropped and the call "
             "re-sent once.", "422-isolation", history=True, fault=_unit_422,
             rebuild="the traced round-1 Ballot is the isolated one (its family dropped)"),
        _one(scripts.R5_REQUEST, scripts.R5, "At most 8 questions per call: probe-only tools are cut and the rest is "
             "split into parallel calls of one round.", "budget-split", limits={"max_questions": 8},
             rebuild="verify compiles with the default limits; this Ballot was cut and split by max_questions=8"),
    ]  # fmt: skip


def case_names() -> list[str]:
    """The case directory names."""
    return [case.name for case in cases()]


# ====================================================================================================================
# Backends and executors (record, replay)
# ====================================================================================================================


def error_doc(error: BaseException) -> dict[str, Any]:
    """``{"error": {"type", "status", "detail"?, "message"}}`` of a failed call."""
    doc: dict[str, Any] = {"type": type(error).__name__, "status": getattr(error, "status", None),
                           "message": str(error)}  # fmt: skip
    if isinstance(error, JevValidationError):
        doc["detail"] = error.detail
    return {"error": doc}


def error_from_doc(doc: Mapping[str, Any]) -> BackendError:
    """The backend error a stored ``{"error": …}`` response raises."""
    kind = str(doc.get("type", "BackendError"))
    status = doc.get("status")
    if kind == "JevValidationError":
        return JevValidationError(doc.get("detail") or [], status=int(status or 422))
    cls = getattr(backend_errors, kind, BackendError)
    if not (isinstance(cls, type) and issubclass(cls, BackendError)):
        cls = BackendError
    return cls(str(doc.get("message", kind)), status=status)


def response_doc(response: DecisionResponse) -> dict[str, Any]:
    """The response body as the trace stores it."""
    return response.model_dump(mode="json", exclude_none=True)


@dataclass
class Exchange:
    """One Jev call: the request and its response body (or ``{"error": …}``)."""

    request: dict[str, Any]
    response: dict[str, Any]


class RecordingBackend:
    """Answers from a script (``ScriptedBackend``) and records every exchange; ``fault`` injects backend errors."""

    name = BACKEND

    def __init__(self, script: Script, fault: Fault | None = None) -> None:
        self.inner = ScriptedBackend(script, model=MODEL, name=BACKEND)
        self.model = MODEL
        self.fault = fault
        self.exchanges: list[Exchange] = []

    def decide(self, request: DecisionRequest) -> DecisionResponse:
        error = self.fault(request, len(self.exchanges)) if self.fault is not None else None
        if error is not None:
            self.exchanges.append(Exchange(request.to_wire(), error_doc(error)))
            raise error
        response = self.inner.decide(request)
        self.exchanges.append(Exchange(request.to_wire(), response_doc(response)))
        return response

    async def adecide(self, request: DecisionRequest) -> DecisionResponse:
        return self.decide(request)


class GoldenMismatch(AssertionError):
    """A replayed request differs from the stored one (byte comparison of canonical JSON)."""


class ReplayBackend:
    """Replays stored exchanges in order: the ``i``-th request must equal ``request_i`` byte for byte (canonical
    JSON); the answer is the stored response (or its error)."""

    name = BACKEND

    def __init__(self, exchanges: Sequence[tuple[bytes, Mapping[str, Any]]], model: str = MODEL) -> None:
        self.exchanges = list(exchanges)
        self.model = model
        self.calls = 0

    def decide(self, request: DecisionRequest) -> DecisionResponse:
        index = self.calls
        self.calls += 1
        if index >= len(self.exchanges):
            raise GoldenMismatch(f"unexpected Jev call {index + 1}: only {len(self.exchanges)} exchange(s) stored")
        expected, response = self.exchanges[index]
        if canonical_json(request.to_wire()) != expected:
            raise GoldenMismatch(f"request {index + 1} differs from the stored request")
        if "error" in response:
            raise error_from_doc(response["error"])
        return DecisionResponse.from_json(dict(response))

    async def adecide(self, request: DecisionRequest) -> DecisionResponse:
        return self.decide(request)


def recording_executors(log: list[dict[str, Any]]) -> dict[str, Callable[..., Any]]:
    """The demo workspace's executors, logging ``{"tool", "arguments", "result"}`` per call."""
    workspace = Workspace()

    def wrap(name: str, fn: Callable[..., Any]) -> Callable[..., Any]:
        def run(**arguments: Any) -> Any:
            result = fn(**arguments)
            log.append({"tool": name, "arguments": jsonable(arguments), "result": jsonable(result)})
            return result

        return run

    return {name: wrap(name, fn) for name, fn in workspace.executors().items()}


def replay_executors(executions: Sequence[Mapping[str, Any]]) -> dict[str, Callable[..., Any]]:
    """Executors that return the recorded results in order (the arguments must match the recording)."""
    queue = list(executions)

    def make(name: str) -> Callable[..., Any]:
        def run(**arguments: Any) -> Any:
            if not queue:
                raise GoldenMismatch(f"unexpected execution of {name}")
            expected = queue.pop(0)
            if expected["tool"] != name or jsonable(arguments) != expected["arguments"]:
                raise GoldenMismatch(f"execution of {name}({arguments}) differs from {expected['tool']}"
                                     f"({expected['arguments']})")  # fmt: skip
            return expected["result"]

        return run

    return {name: make(name) for name in sorted({str(e["tool"]) for e in executions})}


# ====================================================================================================================
# Playing a case directory (shared by the generator and the test)
# ====================================================================================================================


@dataclass
class Played:
    """What playing a case produced."""

    catalog: Any
    contexts: dict[str, Context]
    ballot: Ballot
    requests: list[DecisionRequest]
    decisions: list[Decision]
    verify_contexts: list[Context | None]
    exchange_steps: list[int]
    loop: LoopResult | None = None


class ContextLog(Router):
    """A Router that records the context of every decision (the agent builds each step's context itself), so each
    trace can be verified against the context it was decided in."""

    def __init__(self, *args: Any, **kw: Any) -> None:
        super().__init__(*args, **kw)
        self.seen: list[Context] = []

    def decide(self, messages: Any, *, context: Context | None = None, **kw: Any) -> Decision:
        self.seen.append(_snapshot(self.context_for(messages, context)))
        return super().decide(messages, context=context, **kw)

    def resume(self, pending: Any, *, selection: str | None = None, reply: str | None = None,
               context: Context | None = None) -> Decision:  # fmt: skip
        handle = pending if isinstance(pending, Pending) else self.pendings.get(str(pending))
        messages = handle.state.get("messages", []) if handle is not None else []
        self.seen.append(_snapshot((context or self.context).with_messages(messages)))
        return super().resume(pending, selection=selection, reply=reply, context=context)


def _snapshot(ctx: Context) -> Context:
    """A copy whose entity store no longer changes (the agent keeps adding to the live one)."""
    if isinstance(ctx.entities, EntityStore):
        return ctx.model_copy(update={"entities": EntityStore.from_json(ctx.entities.to_json())})
    return ctx


def play(directory: Path, manifest: Mapping[str, Any], backend: Any,
         executors: Mapping[str, Callable[..., Any]] | None = None) -> Played:  # fmt: skip
    """Run a case from its input files: load the contexts and the catalog, compile the round-1 Ballot of the first
    step, then run every step through a Router over ``backend`` (and an Agent over ``executors``)."""
    contexts = {name: load_context(directory / name) for name in manifest["contexts"]}
    steps = list(manifest["steps"])
    first = contexts[steps[0]["context"]]
    catalog = load_catalog(directory / manifest["catalog"], sources=list(first.sources.values()))
    router = ContextLog(catalog, backend=backend, context=first, limits=Limits(**manifest.get("limits", {})))
    ballot = router.compile(first.messages, context=first, mode=steps[0].get("mode", "turn"))
    requests = ballot.to_requests(router.model)
    decisions: list[Decision] = []
    exchange_steps: list[int] = []
    agent: Agent | None = None
    loop: LoopResult | None = None
    for number, step in enumerate(steps, start=1):
        before = _calls(backend)
        ctx = contexts[step["context"]] if step.get("context") else None
        action = step["action"]
        if action == "decide":
            assert ctx is not None
            decisions.append(router.decide(ctx.messages, context=ctx, mode=step.get("mode", "turn")))
        elif action == "resume":
            pending = decisions[-1].pending_id or ""
            decisions.append(router.resume(pending, selection=step.get("selection"), reply=step.get("reply"),
                                           context=ctx))  # fmt: skip
        elif action in ("agent_run", "agent_resume"):
            if action == "agent_run":
                assert ctx is not None
                agent = Agent(router, dict(executors or {}))
                loop = agent.run(ctx.messages, context=ctx)
            else:
                assert agent is not None and loop is not None and loop.pending is not None
                loop = agent.resume(loop.pending, selection=step.get("selection"), reply=step.get("reply"))
            decisions += loop.decisions[len(decisions) :]
        else:
            raise ValueError(f"unknown step action {action!r}")
        exchange_steps += [number] * (_calls(backend) - before)
    with_context = bool(manifest.get("verify", {}).get("with_context", True))
    verify_contexts: list[Context | None] = list(router.seen) if with_context else [None] * len(router.seen)
    if len(verify_contexts) != len(decisions):
        raise AssertionError("every decision must come from one decide/resume call")
    return Played(catalog=catalog, contexts=contexts, ballot=ballot, requests=requests, decisions=decisions,
                  verify_contexts=verify_contexts, exchange_steps=exchange_steps, loop=loop)  # fmt: skip


def _calls(backend: Any) -> int:
    exchanges = getattr(backend, "exchanges", None)
    if isinstance(backend, RecordingBackend) and exchanges is not None:
        return len(exchanges)
    return int(getattr(backend, "calls", 0))


# ====================================================================================================================
# Documents
# ====================================================================================================================


def pretty(doc: Any) -> bytes:
    """Indented JSON (inputs and manifests): readable, stable."""
    return (json.dumps(doc, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


def trace_doc(trace: Trace) -> dict[str, Any]:
    """The trace document with ``created_at`` and every ``latency_ms`` normalized (the only non-deterministic
    fields)."""
    doc = trace.to_doc()
    doc["created_at"] = CREATED_AT
    for rnd in doc["rounds"]:
        for call in rnd.get("calls", []):
            call["latency_ms"] = 0
    return doc


def loop_doc(loop: LoopResult) -> dict[str, Any]:
    """The outcome of an agent run: outcome, rule, reason and its steps."""
    return {"outcome": loop.outcome.value, "rule": loop.rule, "reason": loop.reason,
            "steps": [step.model_dump(mode="json") for step in loop.steps]}  # fmt: skip


def numbered(stem: str, count: int) -> list[str]:
    """``stem.json`` for one file, else ``stem_1.json`` … ``stem_<count>.json``."""
    if count == 1:
        return [f"{stem}.json"]
    return [f"{stem}_{i}.json" for i in range(1, count + 1)]


def step_files(stem: str, count: int) -> list[str]:
    """``stem_1.json`` … ``stem_<count-1>.json`` then ``stem.json`` (the final one)."""
    return [f"{stem}_{i}.json" for i in range(1, count)] + [f"{stem}.json"]


def expectation(decision: Decision) -> dict[str, Any]:
    """A readable summary of the final decision (the bytes are in ``decision.json``)."""
    return {"outcome": decision.outcome.value, "rule": decision.rule,
            "call": decision.call.to_doc(with_ids=False) if decision.call is not None else None,
            "prompt": decision.prompt.text if decision.prompt is not None else None}  # fmt: skip


def base_manifest(case: Case) -> dict[str, Any]:
    """The manifest fields known before playing (inputs and steps)."""
    return {
        "case": case.name, "spec": SPEC_VERSION, "summary": case.summary, "evidence": EVIDENCE, "model": MODEL,
        "backend": BACKEND, "policy": Policy().version, "limits": dict(case.limits), "catalog": "catalog.json",
        "contexts": list(case.contexts), "steps": [step.doc() for step in case.steps],
        "verify": {"with_context": case.rebuild is None, **({"why_not": case.rebuild} if case.rebuild else {})},
    }  # fmt: skip


def build_case(case: Case, directory: Path) -> dict[str, bytes]:
    """Write the case's inputs into ``directory``, play it with recorded answers, write the outputs and
    ``case.json`` next to them and return every file of the case directory as ``{name: bytes}``."""
    directory.mkdir(parents=True, exist_ok=True)
    out: dict[str, bytes] = {"catalog.json": pretty(scenario_tools())}
    for name, spec in case.contexts.items():
        out[name] = pretty(spec.doc())
    for name, data in out.items():
        (directory / name).write_bytes(data)
    manifest = base_manifest(case)
    backend = RecordingBackend(case.script, case.fault)
    executions: list[dict[str, Any]] = []
    played = play(directory, manifest, backend, recording_executors(executions))
    for name, spec in case.contexts.items():
        if played.contexts[name].sha256 != spec.context().sha256:
            raise AssertionError(f"{case.name}/{name} does not reproduce the scenario context")
    decision_files = step_files("decision", len(played.decisions))
    trace_files = step_files("trace", len(played.decisions))
    request_files = numbered("request", len(backend.exchanges))
    response_files = numbered("response", len(backend.exchanges))
    out["ballot.json"] = played.ballot.to_json()
    for exchange, req_name, resp_name in zip(backend.exchanges, request_files, response_files, strict=True):
        out[req_name] = canonical_json(exchange.request)
        out[resp_name] = canonical_json(exchange.response)
    for decision, dec_name, tr_name in zip(played.decisions, decision_files, trace_files, strict=True):
        out[dec_name] = decision.to_json()
        out[tr_name] = canonical_json(trace_doc(decision.trace))
    ballot_requests = request_files[: len(played.requests)]
    for request, name in zip(played.requests, ballot_requests, strict=True):
        if canonical_json(request.to_wire()) != out[name]:
            raise AssertionError(f"{case.name}: the first sent request is not ballot.to_requests(model)")
    manifest.update({
        "ballot": "ballot.json", "ballot_requests": ballot_requests,
        "exchanges": [{"step": step, "request": req, "response": resp} for step, req, resp in
                      zip(played.exchange_steps, request_files, response_files, strict=True)],
        "decisions": decision_files, "traces": trace_files,
        "expect": expectation(played.decisions[-1]),
    })  # fmt: skip
    if executions:
        manifest["executions"] = executions
    if played.loop is not None:
        out["loop.json"] = canonical_json(loop_doc(played.loop))
        manifest["loop"] = "loop.json"
    out["case.json"] = pretty(manifest)
    for name, data in out.items():
        (directory / name).write_bytes(data)
    return out


def build_all(root: Path, names: Iterable[str] | None = None) -> dict[str, bytes]:
    """Generate the shared fixtures and the selected cases under ``root``; returns ``{relative path: bytes}``."""
    wanted = set(names) if names else None
    unknown = (wanted or set()) - set(case_names())
    if unknown:
        raise ValueError(f"unknown golden case(s): {', '.join(sorted(unknown))}")
    files: dict[str, bytes] = shared_files()
    for rel, data in files.items():
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_bytes(data)
    for case in cases():
        if wanted is None or case.name in wanted:
            for name, data in build_case(case, root / case.name).items():
                files[f"{case.name}/{name}"] = data
    return files


def _generated(names: Iterable[str] | None) -> dict[str, bytes]:
    with tempfile.TemporaryDirectory(prefix="jevtools-golden-") as tmp:
        return build_all(Path(tmp), names)


def _case_files(directory: Path, names: Iterable[str] | None) -> set[str]:
    wanted = set(names) if names else set(case_names())
    found: set[str] = set()
    for case in wanted:
        folder = directory / case
        if folder.is_dir():
            found |= {f"{case}/{p.name}" for p in folder.iterdir() if p.suffix == FIXTURE_SUFFIX}
    return found


def update_fixtures(directory: str | Path = GOLDEN_DIR, names: Iterable[str] | None = None) -> list[str]:
    """Regenerate the fixtures under ``directory`` (``jevtools fixtures --update``): stale JSON files of the
    regenerated cases are removed. Returns the written paths (relative)."""
    target = Path(directory)
    files = _generated(names)
    for stale in sorted(_case_files(target, names) - set(files)):
        (target / stale).unlink()
    for rel, data in sorted(files.items()):
        path = target / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists() or path.read_bytes() != data:
            path.write_bytes(data)
    return sorted(files)


def check_fixtures(directory: str | Path = GOLDEN_DIR, names: Iterable[str] | None = None) -> list[str]:
    """Problems between the stored fixtures and a fresh generation (``jevtools fixtures`` without ``--update``)."""
    target = Path(directory)
    files = _generated(names)
    problems = [f"missing {rel}" for rel in sorted(files) if not (target / rel).exists()]
    problems += [f"differs {rel}" for rel in sorted(files) if (target / rel).exists()
                 and (target / rel).read_bytes() != files[rel]]  # fmt: skip
    problems += [f"stale {rel}" for rel in sorted(_case_files(target, names) - set(files))]
    return problems


def copy_tree(source: Path, target: Path) -> None:  # pragma: no cover - a convenience for manual inspection
    """Copy a generated tree (``build_all``) to ``target``."""
    shutil.copytree(source, target, dirs_exist_ok=True)


__all__ = [
    "BACKEND",
    "CREATED_AT",
    "EVIDENCE",
    "GOLDEN_DIR",
    "MODEL",
    "SHARED",
    "Case",
    "ContextLog",
    "ContextSpec",
    "GoldenMismatch",
    "Played",
    "RecordingBackend",
    "ReplayBackend",
    "Step",
    "build_all",
    "build_case",
    "case_names",
    "cases",
    "check_fixtures",
    "error_doc",
    "error_from_doc",
    "loop_doc",
    "play",
    "replay_executors",
    "trace_doc",
    "update_fixtures",
]
