"""The ``jevtools`` command line (spec §7.5, §3.9, §8.7, §7.2.4).

.. code-block:: text

    jevtools lint <catalog.json> [--sidecar jevtools.json] [--sources sources.toml] [--filler] [--strict]
    jevtools explain <trace.json> [--policy policy.toml]
    jevtools verify <trace.json> [--catalog catalog.json] [--sidecar …] [--sources …] [--context ctx.json]
                                 [--policy policy.toml]
    jevtools probe [--backend auto] [--allow-offline] [--path limits.json] [--no-write] [--no-smoke]
    jevtools serve [--config jevtools.toml] [--host 127.0.0.1] [--port 8787]
    jevtools eval <dataset.jsonl> [--backend auto] [--allow-offline] [--policy …] [--calibrators …] [--replays N]
                                  [--out report.json] [--no-traces]
    jevtools tune <report.json> [--out DIR] [--policy base.toml] [--alpha write=0.01 …] [--method cp|crc]
                                [--calibrate | --held-out report.json]
    jevtools fixtures [--update] [--dir tests/golden] [--case NAME …]
    jevtools bench bfcl [--data DIR] [--download [--ref main]] [--categories c1,c2 | all] [--limit N]
                        [--backend oracle|sim|auto|typesafe|openrouter_decisions|…] [--risk read|infer] [--out F]
    jevtools bench app [--domains inbox,crm,…] [--dir DIR] [--backend oracle|sim|auto|…] [--tags] [--replays N]
                       [--controls] [--out F]

``lint`` prints one line per tool and per slot (``name  description  stakes  STATUS  note``) plus the description
overlap check, and exits 1 when any check is an ERROR (``--strict``: also WARN/WEAK). ``explain`` renders a trace
for humans (bindings, factors, composition, the rule that fired). ``verify`` re-checks a trace with the model out of
the loop (``jt.verify``). ``eval`` runs a labelled dataset (§11.2) and ``tune`` writes a tuned ``policy.toml``
(§11.3). ``fixtures`` regenerates (``--update``) or checks the golden conformance fixtures of §10.2; it runs from a
repository checkout (the case definitions live in ``tests/golden/cases.py``). ``probe``, ``serve``, ``eval``,
``tune`` and ``fixtures`` import their modules lazily.

Catalog files: an OpenAI tools list, an MCP ``tools/list`` result (``{"tools": [...]}``, optionally inside a
JSON-RPC ``result``) or a list of MCP tools. Source files (JSON, TOML or YAML): ``{"sources": [...]}``, a list, or
``[sources.<name>]`` tables of source specs (:mod:`jevtools.sources.specs`, the format of ``jevtools.toml``): each
entry has ``name``, ``provides`` and optionally ``type``/``kind`` (``registry`` | ``files`` | ``provider`` |
``source``), ``key``, ``label``, ``describe``, ``match``, ``attrs``, ``retriever``, ``k``, ``hierarchy``, ``channel``
and the data (``rows``, ``path``/``rows_file`` for registries, ``paths``/``paths_file`` for file indexes). Entries
without data are enough for ``lint`` (inference needs names and tags only).
"""

from __future__ import annotations

import argparse
import importlib
import json
import re
import sys
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, TextIO

from jevtools._compat import load_toml
from jevtools.errors import CatalogError, JevtoolsError
from jevtools.policy import Policy, Tier
from jevtools.sources.specs import build_sources as _build_sources
from jevtools.sources.specs import source_specs
from jevtools.spec.catalog import Catalog, collect_inline, to_raw
from jevtools.spec.infer import VERB_TIERS, tool_verb
from jevtools.spec.ingest import RawTool, mcp_tools
from jevtools.spec.models import SlotSpec, ToolSpec
from jevtools.spec.sidecar import Sidecar, load_sidecar
from jevtools.spec.xjev import PARAM_KEYS, TOOL_KEYS

OVERLAP_WARN = 0.5
"""Token Jaccard between two tool descriptions at or above which lint warns (§7.5)."""
K_DILUTION = 120
"""A REF shortlist larger than this dilutes the Choice (§7.5)."""

# --------------------------------------------------------------------------------------------------------------------
# File loading
# --------------------------------------------------------------------------------------------------------------------


def _load_data(path: str | Path) -> Any:
    """JSON, TOML (``.toml``) or YAML (``.yaml``/``.yml``; needs PyYAML)."""
    file = Path(path)
    text = file.read_text(encoding="utf-8")
    suffix = file.suffix.lower()
    if suffix == ".toml":
        return load_toml(text)
    if suffix in (".yaml", ".yml"):
        try:
            import yaml  # type: ignore[import-untyped]
        except ModuleNotFoundError as exc:
            raise JevtoolsError("YAML files need PyYAML: pip install 'jevtools[yaml]'") from exc
        return yaml.safe_load(text)
    return json.loads(text)


def load_tools(path: str | Path) -> list[Any]:
    """The tool definitions of a catalog file (OpenAI tools, an MCP ``tools/list`` result or a list of MCP tools)."""
    data = _load_data(path)
    if isinstance(data, Mapping) and isinstance(data.get("result"), Mapping):
        data = data["result"]
    if isinstance(data, Mapping) and "tools" in data:
        return list(mcp_tools(data))
    if isinstance(data, list):
        return data
    raise CatalogError(f"{path}: expected a list of tools or a tools/list result")


def load_source_specs(path: str | Path) -> list[dict[str, Any]]:
    """Source entries of a sources file (JSON, TOML or YAML; see :mod:`jevtools.sources.specs`), with file paths
    (``path``, ``rows_file``, ``paths_file``) resolved against the file's directory."""
    try:
        specs = source_specs(_load_data(path))
    except ValueError as exc:
        raise JevtoolsError(f"{path}: {exc}") from None
    return [_resolve_path(spec, Path(path).parent) for spec in specs]


def _resolve_path(spec: dict[str, Any], base: Path) -> dict[str, Any]:
    for key in ("path", "rows_file", "paths_file"):
        if isinstance(spec.get(key), str) and not Path(spec[key]).is_absolute():
            spec[key] = str(base / spec[key])
    return spec


def build_sources(specs: Iterable[Mapping[str, Any]]) -> list[Any]:
    """Source objects for the entries that carry data (:func:`jevtools.sources.specs.build_sources`); entries
    without data stay descriptors (``{"name", "provides"}``), enough for catalog inference."""
    try:
        return _build_sources(list(specs), descriptors=True)
    except ValueError as exc:  # pydantic's ValidationError is a ValueError too
        raise JevtoolsError(f"bad source entry: {exc}") from None


def load_catalog(path: str | Path, *, sidecar: str | None = None, sources: Sequence[Any] = ()) -> Catalog:
    """A compiled catalog from a catalog file, an optional sidecar and sources."""
    return Catalog.from_any(load_tools(path), sidecar=sidecar, sources=list(sources))


def load_context(path: str | Path) -> Any:
    """A :class:`~jevtools.context.Context` from JSON/TOML/YAML: ``messages``, ``now`` (ISO), ``tz``, ``locale``,
    ``user``, ``shareable``, ``include_system``, ``observations`` and ``sources`` (entries as in sources files)."""
    from jevtools.context import Context

    loaded = _load_data(path)
    if not isinstance(loaded, Mapping):
        raise JevtoolsError(f"{path}: expected a context object, got {type(loaded).__name__}")
    data = dict(loaded)
    raw = data.pop("sources", None)
    specs = [_resolve_path(spec, Path(path).parent) for spec in source_specs(raw)] if raw else []
    data.pop("entities", None)
    if isinstance(data.get("now"), str):
        data["now"] = datetime.fromisoformat(data["now"].replace("Z", "+00:00"))
    return Context.model_validate({**data, "sources": build_sources(specs)})


def load_trace(path: str | Path) -> Any:
    """A :class:`~jevtools.trace.Trace` from a trace document (or a document with a ``trace`` key)."""
    from jevtools.trace import Trace

    data = _load_data(path)
    if isinstance(data, Mapping) and "trace" in data and "trace_id" not in data:
        data = data["trace"]
    if not isinstance(data, Mapping):
        raise JevtoolsError(f"{path}: expected a trace object, got {type(data).__name__}")
    return Trace.from_doc(data)


# --------------------------------------------------------------------------------------------------------------------
# lint (§7.5)
# --------------------------------------------------------------------------------------------------------------------

COMMON_VERBS = frozenset(
    """accept add analyze answer append apply approve archive ask assign attach authorize backup block book build
    calculate call cancel change charge check classify clear close comment commit compare compose compute confirm
    connect convert copy count create crop decline decrypt delete deploy describe detach disable disconnect display
    download draft drop edit email enable encrypt estimate evaluate execute explain export extract fetch file fill
    filter find follow format forward generate get give grant hide import install invite join label launch leave list
    load lock log look lookup make mark measure merge message modify monitor move mute notify open order parse pause
    pay pick pin place plan play post predict print publish purchase put query rank rate read reboot record refund
    register reject reload remind remove rename render reorder reply report request reschedule reserve reset resize
    resolve restart restore resume retrieve return revert revoke rotate run save scan schedule score search select
    sell send set share show sign simulate snooze sort split start stop store submit subscribe suggest summarize
    switch sync synchronize tag take test toggle track transfer translate transcribe trigger turn undo unlock
    unsubscribe update upgrade upload validate verify view vote watch wire withdraw write""".split()
)
"""Imperative verbs a tool description may start with (plus the §3.3.2 tier verbs and the tool's own verb)."""
STOPWORDS = frozenset(
    "a an the of for to in on at by with from and or into onto as is are be this that it its their them user user's "
    "one two some any all".split()
)
_WORD = re.compile(r"[a-z0-9']+")


def _tokens(text: str) -> set[str]:
    return {w for w in _WORD.findall(text.lower()) if w not in STOPWORDS}


def jaccard(a: str, b: str) -> float:
    """Token Jaccard of two descriptions (stopwords removed)."""
    ta, tb = _tokens(a), _tokens(b)
    return len(ta & tb) / len(ta | tb) if ta | tb else 0.0


def verb_problem(tool: ToolSpec) -> str | None:
    """Why the description does not start with an imperative verb (``None`` when it does, or ``intent`` is
    declared)."""
    if tool.xjev.intent:
        return None
    words = _WORD.findall(tool.description.lower())
    if not words:
        return None
    first = words[0]
    known = COMMON_VERBS | set(VERB_TIERS) | {tool_verb(tool.name)}
    if first in known:
        return None
    for stem in (first[:-1], first[:-2]):
        if first.endswith("s") and stem in known:
            return f"description starts with {first!r}: use the imperative {stem!r} (the intent reads badly)"
    return f"description does not start with a verb ({first!r}): declare x-jev.intent"


@dataclass
class Line:
    """One lint line: ``name  what  stakes  STATUS  note``."""

    name: str
    what: str
    stakes: str
    status: str
    note: str = ""
    indent: int = 2

    def render(self, width: int, what_width: int) -> str:
        head = " " * self.indent + self.name
        text = f"{head:<{width}}{self.what:<{what_width}}"
        if self.stakes:
            text += f"{self.stakes:<10}"
        text += self.status
        return (text + ("  " + self.note if self.note else "")).rstrip()


def _unknown_keys(raw: RawTool, sidecar: Sidecar | None, names: Sequence[str]) -> list[tuple[str, list[str]]]:
    """``(where, unknown keys)`` over inline, MCP ``_meta`` and sidecar ``x-jev`` of one tool."""
    found: list[tuple[str, list[str]]] = []
    tool_layers = [raw.tool_xjev, {k: v for k, v in raw.meta_xjev.items() if k != "properties"}]
    root_x = raw.parameters.get("x-jev")
    root: Mapping[str, Any] = root_x if isinstance(root_x, Mapping) else {}
    side_tool, side_params = sidecar.for_tool(raw.name, all_names=names) if sidecar else ({}, {})
    for layer in [*tool_layers, root, side_tool]:
        bad = sorted(set(layer) - TOOL_KEYS)
        if bad:
            found.append((raw.name, bad))
    params: list[tuple[str, Mapping[str, Any]]] = list(collect_inline(raw.parameters).items())
    params += [(str(k), v) for k, v in (raw.meta_xjev.get("properties") or {}).items() if isinstance(v, Mapping)]
    params += list(side_params.items())
    for key, value in params:
        bad = sorted(set(value) - PARAM_KEYS)
        if bad:
            found.append((f"{raw.name}.{key}", bad))
    return found


def _source_info(specs: Sequence[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    """Source entries by name, with the defaults a built source has: ``kind`` or ``type`` (or ``paths``) names the
    kind, and a file index without declared tags provides ``path`` and ``file`` (as :class:`FileIndex` does)."""
    info: dict[str, Mapping[str, Any]] = {}
    for spec in specs:
        entry = dict(spec)
        files = "paths" in entry or "paths_file" in entry
        entry["kind"] = entry.get("kind") or entry.get("type") or ("files" if files else "registry")
        if entry["kind"] == "files" and not entry.get("provides"):
            entry["provides"] = ["path", "file"]
        info[str(entry["name"])] = entry
    return info


def describe_slot(slot: SlotSpec, sources: Mapping[str, Mapping[str, Any]], policy: Policy) -> str:
    """The lint description of a slot: its kind and where its candidates come from."""
    kind = slot.kind
    if kind == "ref":
        parts = []
        for name in slot.source_names or ("?",):
            info = sources.get(name, {})
            key = info.get("key")
            parts.append(f"ref({name}.{key})" if key else f"ref({name})")
        text = " | ".join(parts)
        if slot.format:
            text += f" | pattern({slot.format})"
        info = sources.get(slot.source_names[0], {}) if slot.source_names else {}
        retriever = info.get("retriever") or ("bm25" if info.get("kind") == "files" else None)
        extras = [f"k={slot.k or info.get('k') or policy.pools.ref_k}"]
        hierarchy = slot.hierarchy or info.get("hierarchy") or ("dirname" if info.get("kind") == "files" else None)
        if hierarchy:
            extras.append(f"hierarchy={hierarchy}")
        text = f"{text} {retriever + ' ' if retriever else ''}{', '.join(extras)}"
        if slot.kind_reason.startswith("row 11"):
            provides = set(info.get("provides") or ())
            matched = [t for t in slot.tags if t in provides] or list(slot.tags)
            text += f" [tag {', '.join(matched)}]" if matched else f" [name {slot.name}]"
        return text
    if kind == "text":
        packs = [f"{p} pack" for p in slot.packs] + (["templates"] if slot.templates else [])
        source = ", ".join(packs) or "no templates"
        return f"text/{slot.stakes} ({source}; extract {', '.join(slot.extract) or 'default'})"
    if kind == "enum":
        if slot.catalog:
            return f"enum(catalog {slot.catalog})"
        return f"enum({len(slot.values or ())} values)"
    if kind == "ordinal":
        return f"ordinal({len(slot.values or ()) or 'bounded'} levels)"
    if kind in ("quantity", "money"):
        unit = f"[{slot.unit}]" if slot.unit and kind == "quantity" else ""
        grid = f" grid({len(slot.values)})" if slot.values else ""
        return f"{kind}{unit}{grid}"
    if kind == "temporal":
        return f"temporal ({'duration' if 'duration' in slot.extract else 'readings'})"
    if kind == "span":
        return f"span/{slot.role or 'generic'} ({', '.join(slot.extract) or 'default'})"
    if kind == "list":
        item = slot.item
        return f"list of {item.kind if item is not None else '?'}" + (" (anchored)" if slot.anchored else "")
    if kind == "record":
        return f"record({len(slot.children)} fields)"
    if kind == "union":
        return f"union({len(slot.branches)} branches)"
    return kind


def lint_slot(
    tool: ToolSpec, slot: SlotSpec, *, sources: Mapping[str, Mapping[str, Any]], checked_sources: bool,
    filler: bool, policy: Policy, depth: int,
) -> Line:  # fmt: skip
    """One slot line with its status (OK, WEAK, WARN, ERROR) and note."""
    notes: list[tuple[str, str]] = []
    if slot.kind == "ref":
        if not slot.source_names:
            notes.append(("ERROR", "ref slot without a source: declare x-jev.source"))
        elif checked_sources:
            missing = [n for n in slot.source_names if n not in sources]
            if missing:
                notes.append(("ERROR", f"source {', '.join(repr(n) for n in missing)} is not registered"))
        k = slot.k or policy.pools.ref_k
        if k > K_DILUTION:
            notes.append(("WARN", f"k={k} > {K_DILUTION}: the shortlist dilutes the Choice"))
    if slot.weak:
        notes.append(("WEAK", f"{slot.kind_reason}: declare x-jev.kind, source or extract"))
    content_text = slot.kind == "text" and slot.stakes == "content" and slot.fallback in (None, "fill")
    if content_text and not filler:
        notes.append(("WEAK", "no Filler: uncovered bodies → clarify(open)"))
    if depth == 1 and not slot.description and slot.kind not in ("derived", "secret"):
        notes.append(("WARN", "no description"))
    order = {"ERROR": 0, "WARN": 1, "WEAK": 2}
    notes.sort(key=lambda n: order[n[0]])
    status = notes[0][0] if notes else "OK"
    note = "; ".join(text for _, text in notes)
    if tool.tier is Tier.CRITICAL or slot.xjev.channels is not None:
        note = (note + "  " if note else "") + "channels=" + ",".join(c.value for c in slot.channels)
    name = f"{tool.name}.{slot.key}"
    return Line(name, describe_slot(slot, sources, policy), slot.stakes, status, note, indent=2 * depth)


def lint_tool(tool: ToolSpec) -> Line:
    """The tool line: tier and why, plus description checks."""
    notes: list[tuple[str, str]] = []
    if tool.tier_reason.startswith("fail-safe default"):
        notes.append(("WARN", "tier by fail-safe default: declare x-jev.risk"))
    if not tool.description.strip():
        notes.append(("WARN", "no description"))
    else:
        problem = verb_problem(tool)
        if problem:
            notes.append(("WARN", problem))
    status = "WARN" if notes else "OK"
    return Line(f"TOOL {tool.name}", f"tier={tool.tier.value} ({tool.tier_reason})", "", status,
                "; ".join(n for _, n in notes), indent=0)  # fmt: skip


def _walk(slots: Iterable[SlotSpec], depth: int = 1) -> Iterable[tuple[SlotSpec, int]]:
    for slot in slots:
        yield slot, depth
        nested = [*([slot.item] if slot.item is not None else []), *slot.children, *slot.branches]
        yield from _walk(nested, depth + 1)


def _compile(raws: Sequence[RawTool], sidecar: Sidecar | None, sources: Sequence[Any]) -> tuple[
        list[ToolSpec], list[Line]]:  # fmt: skip
    try:
        return list(Catalog(raws, sidecar=sidecar, sources=sources)), []
    except CatalogError:
        pass
    tools: list[ToolSpec] = []
    errors: list[Line] = []
    names = [r.name for r in raws]
    for raw in raws:
        try:
            tools += list(Catalog([raw], sidecar=sidecar, sources=sources))
        except CatalogError as exc:
            errors.append(Line(f"TOOL {raw.name}", "does not compile", "", "ERROR", str(exc), indent=0))
    if len(set(names)) != len(names):
        errors.append(Line("CATALOG", "duplicate tool names", "", "ERROR", str(sorted(n for n in set(names)
                                                                                  if names.count(n) > 1))))  # fmt: skip
    return tools, errors


def lint(
    tools: Sequence[Any],
    *,
    sidecar: Sidecar | None = None,
    source_specs: Sequence[Mapping[str, Any]] | None = None,
    filler: bool = False,
    policy: Policy | None = None,
) -> tuple[list[str], dict[str, int]]:
    """Lint a tool list (§7.5): the report lines and the count of each status."""
    policy = policy or Policy()
    specs = list(source_specs or []) + [dict(s) for s in (sidecar.sources if sidecar else [])]
    sources = _source_info(specs)
    source_objs = [{"name": name, "provides": list(info.get("provides") or ())} for name, info in sources.items()]
    raws = [to_raw(t) for t in tools]
    names = [r.name for r in raws]
    lines: list[Line] = []
    for raw in raws:
        for where, keys in _unknown_keys(raw, sidecar, names):
            lines.append(Line(where, "unknown x-jev key(s) " + ", ".join(repr(k) for k in keys), "", "ERROR",
                              indent=0 if "." not in where else 2))  # fmt: skip
    compiled, errors = _compile(raws, sidecar, source_objs)
    lines += errors
    for tool in compiled:
        lines.append(lint_tool(tool))
        for slot, depth in _walk(tool.slots):
            lines.append(lint_slot(tool, slot, sources=sources, checked_sources=source_specs is not None or bool(specs),
                                   filler=filler, policy=policy, depth=depth))  # fmt: skip
    pairs = [(a, b, jaccard(a.description, b.description)) for i, a in enumerate(compiled) for b in compiled[i + 1 :]]
    flagged = [p for p in pairs if p[2] >= OVERLAP_WARN]
    shown = flagged or sorted(pairs, key=lambda p: -p[2])[:1]
    for a, b, overlap in shown:
        status = "WARN" if overlap >= OVERLAP_WARN else "OK"
        lines.append(Line(f"DESCRIPTIONS '{a.name}' vs '{b.name}':", f"token overlap {overlap:.2f}", "", status,
                          f"(warn ≥ {OVERLAP_WARN})", indent=0))  # fmt: skip
    width = max((len(line.name) + line.indent for line in lines), default=0) + 2
    what_width = max((len(line.what) for line in lines), default=0) + 2
    counts = {status: sum(1 for line in lines if line.status == status) for status in ("OK", "WEAK", "WARN", "ERROR")}
    return [line.render(width, what_width) for line in lines], counts


def _cmd_lint(args: argparse.Namespace, out: TextIO) -> int:
    sidecar = load_sidecar(args.sidecar) if args.sidecar else None
    specs = load_source_specs(args.sources) if args.sources else None
    policy = Policy.from_toml(args.policy) if args.policy else None
    lines, counts = lint(load_tools(args.catalog), sidecar=sidecar, source_specs=specs, filler=args.filler,
                         policy=policy)  # fmt: skip
    for line in lines:
        print(line, file=out)
    print(f"{counts['ERROR']} error(s), {counts['WARN']} warning(s), {counts['WEAK']} weak", file=out)
    if counts["ERROR"]:
        return 1
    return 1 if args.strict and (counts["WARN"] or counts["WEAK"]) else 0


# --------------------------------------------------------------------------------------------------------------------
# explain
# --------------------------------------------------------------------------------------------------------------------

RULE_TEXT: dict[str, str] = {
    "P0": "a Jev call failed after retries and isolation: fail closed",
    "P1": "the tool Choice says no tool is needed",
    "P2": "the tool Choice says no listed tool can do this",
    "P3": "suspected injection or a channel violation: refuse",
    "P4": "the request does not authorize the action",
    "P5": "the tool choice is ambiguous",
    "P6": "the chosen tool could not be speculated (a required value is missing)",
    "P7": "a slot's shape needs attention (missing, out of pool, uncovered text, uncertain flag)",
    "P8": "the bindings are inconsistent (presence, order, joint, constraints or schema)",
    "P9": "the call confidence was compared with the tier thresholds",
    "P10": "the loop is done",
}


def _p(x: Any) -> str:
    return "—" if x is None else f"{float(x):.4f}".rstrip("0").rstrip(".") if isinstance(x, (int, float)) else str(x)


def _value(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False) if not isinstance(value, str) else value


def explain(trace: Any, policy: Policy | None = None) -> str:
    """A human-readable explanation of a trace: outcome and rule, tool distribution, bindings with provenance,
    gates, factors, the composition and the thresholds it was compared with."""
    lines: list[str] = []
    out = lines.append
    outcome = trace.outcome
    out(f"Decision {trace.decision_id}  (trace {trace.trace_id}, {trace.spec})")
    created = trace.created_at.strftime("%Y-%m-%dT%H:%M:%SZ") if trace.created_at else "?"
    out(f"Created {created}  policy {trace.policy.get('version')}  catalog {trace.catalog_sha256[:19]}…")
    rule = str(outcome.get("rule", ""))
    prefix = rule.split(".", 1)[0]
    out("")
    out(f"Outcome: {outcome.get('value')}  —  rule {rule}")
    if prefix in RULE_TEXT:
        out(f"  {RULE_TEXT[prefix]}")
    details = [f"{k} {outcome[k]}" for k in ("bottleneck", "shape", "reason") if outcome.get(k)]
    if outcome.get("caps"):
        details.append("caps " + ", ".join(outcome["caps"]))
    if details:
        out("  " + ", ".join(details))
    if trace.resumed_from:
        out(f"  resumed from {trace.resumed_from}")
    tool = trace.tool
    if tool:
        alternatives = ", ".join(f"{k} {_p(v)}" for k, v in sorted(tool.get("alternatives", {}).items(),
                                                                     key=lambda kv: -kv[1]))  # fmt: skip
        out("")
        out(f"Tool: {tool['chosen']}  p={_p(tool.get('p'))}" + (f"  (alternatives: {alternatives})" if alternatives
                                                                 else ""))  # fmt: skip
        if tool.get("call_map") and tool["call_map"] != tool["chosen"]:
            out(f"  call MAP disagrees: {tool['call_map']}")
    if trace.call:
        args = ", ".join(
            f"{k}={json.dumps(v, ensure_ascii=False)}" for k, v in (trace.call.get("arguments") or {}).items()
        )
        out(f"Call: {trace.call.get('name')}({args})")
    if trace.idempotency_key:
        out(f"  idempotency key {trace.idempotency_key}")
    if trace.bindings:
        out("")
        out("Bindings:")
        for name, b in trace.bindings.items():
            shown = b.get("label") or (_value(b.get("value")) if b.get("value") is not None else b.get("bottom"))
            out(f"  {name} = {shown}  p={_p(b.get('p'))}  channel={b.get('channel')}  shape={b.get('shape')}")
            origin = [f"qid {b['qid']}" if b.get("qid") else "", f"family {b['family']}" if b.get("family") else "",
                      f"normalizer {b['normalizer']}" if b.get("normalizer") else ""]  # fmt: skip
            prov = b.get("prov") or {}
            mention = prov.get("mention") if isinstance(prov, Mapping) else None
            if isinstance(mention, Mapping) and mention.get("text"):
                origin.append(f'from "{mention["text"]}"')
            if isinstance(prov, Mapping) and prov.get("extractor"):
                origin.append(f"extractor {prov['extractor']}")
            if isinstance(prov, Mapping) and prov.get("source"):
                origin.append(f"source {prov['source']}")
            origin = [o for o in origin if o]
            if origin:
                out("      " + "; ".join(origin))
            if b.get("alternatives"):
                out("      alternatives: " + ", ".join(f"{a.get('label')} {_p(a.get('p'))}" for a in b["alternatives"]))
            sentinels = {k: v for k, v in (b.get("sentinels") or {}).items() if v}
            if sentinels:
                out("      sentinels: " + ", ".join(f"{k} {_p(v)}" for k, v in sentinels.items()))
            if b.get("flags"):
                out("      flags: " + ", ".join(b["flags"]))
    if trace.gates:
        out("")
        out("Gates: " + ", ".join(f"{k} {_p(v)}" for k, v in trace.gates.items()))
    if trace.factors:
        out("Factors: " + " · ".join(f"{k} {_p(v)}" for k, v in trace.factors.items()))
    comp = trace.composition
    if comp:
        out("")
        out(f"Composition: tier {comp.get('tier')} uses {comp.get('rule')} → C = {_p(comp.get('C'))}"
            + (f" (calibrated by {comp['calibrator']})" if comp.get("calibrator") else ""))  # fmt: skip
        out(f"  W {_p(comp.get('W'))}, PI {_p(comp.get('PI'))}, L {_p(comp.get('L'))}, J {_p(comp.get('J'))}")
        thresholds = _thresholds(trace, policy, comp.get("tier"))
        if thresholds:
            out("  " + thresholds)
    if trace.flags:
        out("Flags: " + ", ".join(trace.flags))
    if trace.rounds:
        calls = sum(len(r.calls) for r in trace.rounds)
        tokens = sum(int(c.usage.get("input_tokens") or 0) for r in trace.rounds for c in r.calls)
        costs = [c.usage.get("cost") for r in trace.rounds for c in r.calls if c.usage.get("cost") is not None]
        errors = [c.error for r in trace.rounds for c in r.calls if c.error]
        out("")
        out(f"Rounds: {len(trace.rounds)} ({calls} Jev call(s), {tokens} input tokens"
            + (f", cost ${sum(costs):.6f}" if costs else "") + ")")  # fmt: skip
        for r in trace.rounds:
            models = sorted({c.model_answered or c.model_requested for c in r.calls})
            out(f"  round {r.round} [{r.mode}{', merged' if r.merged else ''}]: {len(r.calls)} call(s) "
                f"{', '.join(models)}")  # fmt: skip
        for error in errors:
            out(f"  error: {error}")
    else:
        out("")
        out("Rounds: 0 (no Jev call)")
    if trace.notes:
        out("Notes:")
        for note in trace.notes:
            out(f"  - {note}")
    return "\n".join(lines)


def _thresholds(trace: Any, policy: Policy | None, tier: str | None) -> str:
    if tier is None:
        return ""
    policy = policy or Policy()
    if policy.version != trace.policy.get("version"):
        return f"thresholds: policy {trace.policy.get('version')} not given (pass --policy)"
    execute = policy.execute_at(tier)
    confirm = policy.tier(tier).confirm
    return (
        f"thresholds ({tier}): execute ≥ {_p(execute) if execute is not None else 'never'}, "
        f"confirm ≥ {_p(confirm) if confirm is not None else 'none'}, hysteresis {_p(policy.hysteresis)}"
    )


def _cmd_explain(args: argparse.Namespace, out: TextIO) -> int:
    policy = Policy.from_toml(args.policy) if args.policy else None
    print(explain(load_trace(args.trace), policy), file=out)
    return 0


# --------------------------------------------------------------------------------------------------------------------
# verify
# --------------------------------------------------------------------------------------------------------------------


def _cmd_verify(args: argparse.Namespace, out: TextIO) -> int:
    from jevtools.trace import verify

    trace = load_trace(args.trace)
    context = load_context(args.context) if args.context else None
    sources: list[Any] = list(context.sources.values()) if context is not None else []
    if args.sources:
        sources += build_sources(load_source_specs(args.sources))
    catalog = load_catalog(args.catalog, sidecar=args.sidecar, sources=sources) if args.catalog else None
    policy = Policy.from_toml(args.policy) if args.policy else None
    report = verify(trace, catalog=catalog, context=context, policy=policy)
    for check in report.checks:
        status = "SKIP" if check.ok is None else "OK" if check.ok else "FAIL"
        print(f"{status:<5} {check.name:<15} {check.detail}".rstrip(), file=out)
    failures = len(report.failures)
    print(f"VERIFIED {trace.trace_id}" if report.ok else f"FAILED {trace.trace_id}: {failures} check(s)", file=out)
    return 0 if report.ok else 1


# --------------------------------------------------------------------------------------------------------------------
# probe / serve (lazy)
# --------------------------------------------------------------------------------------------------------------------


def _lazy(module: str, what: str, err: TextIO) -> Any | None:
    try:
        return importlib.import_module(module)
    except ImportError as exc:
        print(f"jevtools {what}: {module} is not available ({exc}); install jevtools with the needed extra",
              file=err)  # fmt: skip
        return None


def _backend(name: str, allow_offline: bool) -> Any:
    import os

    from jevtools.backends.auto import auto

    if name == "auto":
        return auto(allow_offline=allow_offline)
    return auto(allow_offline=allow_offline, env={**os.environ, "JEVTOOLS_BACKEND": name})


def _cmd_probe(args: argparse.Namespace, out: TextIO, err: TextIO) -> int:
    probe = _lazy("jevtools.probe", "probe", err)
    if probe is None or not hasattr(probe, "run_probe"):
        if probe is not None:
            print("jevtools probe: jevtools.probe has no run_probe()", file=err)
        return 2
    from jevtools.backends.errors import BackendError

    try:
        backend = _backend(args.backend, args.allow_offline)
        report = probe.run_probe(backend, path=args.path, write=not args.no_write, smoke=not args.no_smoke)
    except (BackendError, JevtoolsError) as exc:
        print(f"jevtools probe: {type(exc).__name__}: {exc}", file=err)
        return 1
    print(f"backend {report.backend} model {report.model}: {report.calls} call(s)", file=out)
    for check in report.checks:
        print(f"  {'OK' if check.ok else 'FAIL':<5} {check.name:<22} {check.detail}".rstrip(), file=out)
    for key, value in report.limits.model_dump(mode="json").items():
        print(f"  limit {key} = {value}", file=out)
    if report.path:
        print(f"wrote {report.path}", file=out)
    if getattr(backend, "name", "") == "simulator":
        print("note: the simulator is a lexical test double; its answers are never evidence about Jev", file=out)
    return 0


def _cmd_serve(args: argparse.Namespace, out: TextIO, err: TextIO) -> int:
    serve = _lazy("jevtools.serve", "serve", err)
    if serve is None:
        return 2
    try:
        run = serve.run  # imports jevtools.serve.app: needs the `serve` extra (starlette, uvicorn)
    except ImportError as exc:
        print(f"jevtools serve: needs the serve extra ({exc}): pip install 'jevtools[serve]'", file=err)
        return 2
    print(f"jevtools serve on http://{args.host}:{args.port}/v1", file=out)
    run(config=args.config, host=args.host, port=args.port)
    return 0


# --------------------------------------------------------------------------------------------------------------------
# eval / tune (§11)
# --------------------------------------------------------------------------------------------------------------------

OFFLINE_NOTE = "note: {name} is an offline test double; its numbers test the harness and are never evidence about Jev"


def _rate(doc: Mapping[str, Any]) -> str:
    rate = doc.get("rate")
    shown = "n/a" if rate is None else f"{rate:.4f}"
    return f"{shown} ({doc.get('k', 0)}/{doc.get('n', 0)}, 95% upper {doc.get('upper95', 1.0):.4f})"


def _cmd_eval(args: argparse.Namespace, out: TextIO, err: TextIO) -> int:
    from jevtools.backends.errors import BackendError
    from jevtools.eval import run_dataset
    from jevtools.eval.tuning import load_calibrators

    try:
        backend = _backend(args.backend, args.allow_offline)
    except BackendError as exc:
        print(f"jevtools eval: {type(exc).__name__}: {exc}", file=err)
        return 1
    policy = Policy.from_toml(args.policy) if args.policy else None
    calibrators = load_calibrators(args.calibrators) if args.calibrators else None
    report = run_dataset(args.dataset, backend, replays=args.replays, policy=policy, calibrators=calibrators,
                         keep_traces=not args.no_traces)  # fmt: skip
    summary = report.summary()
    print(f"dataset {args.dataset}: {summary['cases']} case(s) x {summary['replays']} replay(s) on "
          f"{getattr(backend, 'name', '?')}/{getattr(backend, 'model', '?')}", file=out)  # fmt: skip
    print(f"  accuracy     {summary['accuracy']}", file=out)
    print(f"  exact_match  {summary['exact_match']}", file=out)
    print(f"  outcomes     {json.dumps(summary['outcomes'], sort_keys=True)}", file=out)
    print(f"  stages       {json.dumps(summary['stages'], sort_keys=True)}", file=out)
    for tier, doc in sorted(summary["wrong_execution_rate"].items()):
        print(f"  wrong execution ({tier}): {_rate(doc)}", file=out)
    print(f"  flip rate    {summary['flip_rate']}; hysteresis {summary['hysteresis']}", file=out)
    if args.out:
        report.save(args.out, traces=not args.no_traces)
        print(f"wrote {args.out}", file=out)
    if getattr(backend, "name", "") in ("simulator", "scripted"):
        print(OFFLINE_NOTE.format(name=backend.name), file=out)
    return 0


def _alphas(pairs: Sequence[str]) -> dict[str, float]:
    alphas: dict[str, float] = {}
    for pair in pairs:
        tier, sep, value = pair.partition("=")
        if not sep or tier not in {t.value for t in Tier}:
            raise JevtoolsError(f"--alpha expects <tier>=<float> with a tier of {', '.join(t.value for t in Tier)}")
        alphas[tier] = float(value)
    return alphas


def _cmd_tune(args: argparse.Namespace, out: TextIO) -> int:
    from jevtools.eval import EvalReport, tune

    report = EvalReport.load(args.report)
    base = Policy.from_toml(args.policy) if args.policy else None
    held_out: Any = EvalReport.load(args.held_out) if args.held_out else bool(args.calibrate)
    result = tune(report, _alphas(args.alpha or ()), base_policy=base, method=args.method, calibrate=held_out)
    for name, tier in result.tiers.items():
        print(f"  {name:<9} n={tier.n:<6} alpha={tier.alpha:<7} {tier.status:<10} {tier.composition:<3} "
              f"execute={tier.execute} confirm={tier.confirm}", file=out)  # fmt: skip
    cert = result.certification
    print(f"  critical auto-execution: {'certified' if cert.certified else 'not certified'} "
          f"({cert.cases} labelled case(s); {cert.reason})".rstrip(), file=out)  # fmt: skip
    toml_path, cal_path = result.save(args.out)
    print(f"wrote {toml_path}" + (f" and {cal_path}" if cal_path else ""), file=out)
    return 0


# --------------------------------------------------------------------------------------------------------------------
# fixtures (§10.2)
# --------------------------------------------------------------------------------------------------------------------

GOLDEN_MODULE = "tests.golden.cases"
"""The golden case definitions (a repository module: they replay the scenario scripts of the test suite)."""


def _golden(directory: Path, err: TextIO) -> Any | None:
    """Import :data:`GOLDEN_MODULE`, adding the checkout that holds ``directory`` (``<root>/tests/golden``) or the
    working directory to ``sys.path`` when needed."""
    try:
        return importlib.import_module(GOLDEN_MODULE)
    except ImportError:
        pass
    resolved = directory.resolve()
    roots = [resolved.parents[1]] if len(resolved.parents) > 1 and resolved.parent.name == "tests" else []
    for root in [*roots, Path.cwd()]:
        if (root / "tests" / "golden" / "cases.py").is_file() and str(root) not in sys.path:
            sys.path.insert(0, str(root))
    try:
        return importlib.import_module(GOLDEN_MODULE)
    except ImportError as exc:
        print(f"jevtools fixtures: {GOLDEN_MODULE} is not importable ({exc}); run from a jevtools checkout", file=err)
        return None


def _cmd_fixtures(args: argparse.Namespace, out: TextIO, err: TextIO) -> int:
    directory = Path(args.dir)
    golden = _golden(directory, err)
    if golden is None:
        return 2
    names = args.case or None
    if args.update:
        written = golden.update_fixtures(directory, names)
        print(f"wrote {len(written)} fixture file(s) under {directory}", file=out)
        print(f"note: {golden.EVIDENCE}", file=out)
        return 0
    problems = golden.check_fixtures(directory, names)
    for problem in problems:
        print(problem, file=out)
    if problems:
        print(f"{len(problems)} fixture file(s) out of date: run `jevtools fixtures --update --dir {directory}`",
              file=out)  # fmt: skip
        return 1
    print(f"golden fixtures under {directory} are up to date", file=out)
    return 0


# --------------------------------------------------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    """The ``jevtools`` argument parser."""
    parser = argparse.ArgumentParser(prog="jevtools", description="Tool calling for TypeSafe's Jev.")
    from jevtools._version import __version__

    parser.add_argument("--version", action="version", version=f"jevtools {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("lint", help="check a tool catalog (§7.5)")
    p.add_argument("catalog", help="OpenAI tools JSON, or an MCP tools/list result")
    p.add_argument("--sidecar", help="jevtools.json / jevtools.yaml annotations")
    p.add_argument("--sources", help="source registrations (JSON, TOML or YAML)")
    p.add_argument("--policy", help="policy.toml (for the REF shortlist size)")
    p.add_argument("--filler", action="store_true", help="a Filler will be configured (content text is covered)")
    p.add_argument("--strict", action="store_true", help="exit 1 on warnings and weak slots too")

    p = sub.add_parser("explain", help="explain a decision trace")
    p.add_argument("trace", help="trace JSON")
    p.add_argument("--policy", help="policy.toml of the trace (for thresholds)")

    p = sub.add_parser("verify", help="re-check a trace with the model out of the loop (§3.9)")
    p.add_argument("trace", help="trace JSON")
    p.add_argument("--catalog", help="the catalog the decision used (enables re-decoding)")
    p.add_argument("--sidecar", help="sidecar of the catalog")
    p.add_argument("--sources", help="source registrations with data (rows/paths)")
    p.add_argument("--context", help="the decision's context (enables the Ballot rebuild)")
    p.add_argument("--policy", help="policy.toml of the trace")

    p = sub.add_parser("probe", help="measure a backend's wire limits (§8.7)")
    p.add_argument(
        "--backend",
        default="auto",
        help="auto | typesafe | openrouter_systemone | openrouter_decisions | simulator | cassette:<path>",
    )
    p.add_argument("--allow-offline", action="store_true", help="let 'auto' fall back to the offline simulator")
    p.add_argument("--path", help="where to write the limits file (default: the cache)")
    p.add_argument("--no-write", action="store_true", help="do not write the limits file")
    p.add_argument("--no-smoke", action="store_true", help="skip the functional smoke checks")

    p = sub.add_parser("serve", help="run the OpenAI-compatible proxy (§7.2.4)")
    p.add_argument("--config", help="jevtools.toml")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8787)

    p = sub.add_parser("eval", help="run a labelled JSONL dataset and report the §11.2 metrics")
    p.add_argument("dataset", help="JSONL dataset (§11.1)")
    p.add_argument("--backend", default="auto", help="as for probe")
    p.add_argument("--allow-offline", action="store_true", help="let 'auto' fall back to the offline simulator")
    p.add_argument("--policy", help="policy.toml")
    p.add_argument("--calibrators", help="calibrators JSON written by `jevtools tune`")
    p.add_argument("--replays", type=int, default=1, help="decide every case N times (flip rate, hysteresis)")
    p.add_argument("--out", help="write the full report (JSON; the input of `jevtools tune`)")
    p.add_argument("--no-traces", action="store_true", help="do not keep traces in the report")

    p = sub.add_parser("fixtures", help="check or regenerate the golden conformance fixtures (§10.2)")
    p.add_argument("--update", action="store_true", help="regenerate the fixture files (else: check them)")
    p.add_argument("--dir", default="tests/golden", help="the golden fixture directory (default: tests/golden)")
    p.add_argument("--case", action="append", help="only this case (repeatable)")

    p = sub.add_parser("bench", help="run a tool-calling benchmark (BFCL, or the app-domain benchmark)")
    p.add_argument("suite", choices=("bfcl", "app"),
                   help="bfcl: Berkeley Function Calling Leaderboard; app: the bundled app-domain benchmark "
                        "(assistants over an app's own data)")  # fmt: skip
    p.add_argument("--domains", default=None, help="app: comma-separated domains (default: all bundled domains)")
    p.add_argument("--dir", default=None, help="app: a directory of domains (<name>/cases.jsonl) instead of the "
                                               "bundled ones")  # fmt: skip
    p.add_argument("--tags", action="store_true", help="app: also print the per-tag table")
    p.add_argument("--replays", type=int, default=1, help="app: decide every case N times (live Jev varies)")
    p.add_argument("--policy", help="app: policy.toml for the run and its ceiling (default: Appendix B)")
    p.add_argument("--controls", action="store_true",
                   help="app: also run the negative controls (gold rows removed; counts false bindings)")  # fmt: skip
    p.add_argument("--data", default=None, help="BFCL data directory (default: <cache>/bfcl)")
    p.add_argument("--download", action="store_true", help="fetch the BFCL data files from GitHub into --data")
    p.add_argument("--ref", default="main", help="BFCL repository branch, tag or commit to download (default: main)")
    p.add_argument("--categories", default=None, help="comma-separated categories, or 'all' (default: single-call, "
                                                      "irrelevance and relevance categories)")  # fmt: skip
    p.add_argument("--limit", type=int, default=None, help="at most N cases per category")
    p.add_argument("--backend", default="oracle",
                   help="oracle (the ceiling), sim (offline test double), or a Jev backend as for probe "
                        "(auto, typesafe, openrouter_decisions…)")  # fmt: skip
    p.add_argument("--allow-offline", action="store_true", help="let 'auto' fall back to the offline simulator")
    p.add_argument("--risk", default="read", help="bfcl: risk tier given to every BFCL tool; 'infer' keeps the "
                                                  "inferred tier")  # fmt: skip
    p.add_argument("--out", help="write the full report (JSON)")

    p = sub.add_parser("tune", help="tune policy thresholds on an evaluation report (§11.3, §11.4)")
    p.add_argument("report", help="report JSON written by `jevtools eval --out`")
    p.add_argument("--out", default=".", help="directory for policy.toml (and policy.calibrators.json)")
    p.add_argument("--policy", help="base policy.toml (default: Appendix B)")
    p.add_argument("--alpha", action="append", help="wrong-execution budget per tier, e.g. write=0.01 (repeatable)")
    p.add_argument("--method", choices=("cp", "crc"), default="cp", help="Clopper-Pearson or conformal risk control")
    p.add_argument("--calibrate", action="store_true", help="fit isotonic calibrators in-sample (optimistic)")
    p.add_argument("--held-out", help="fit calibrators on this held-out report instead")
    return parser


# --------------------------------------------------------------------------------------------------------------------
# bench (BFCL, app domains)
# --------------------------------------------------------------------------------------------------------------------


def _bench_backend(name: str, allow_offline: bool, err: TextIO) -> Any:
    """``"oracle"``, the offline simulator or a Jev backend; ``None`` after printing why it is unavailable."""
    from jevtools.backends.errors import BackendError

    if name == "oracle":
        return "oracle"
    if name in ("sim", "simulator"):
        from jevtools.backends.simulator import LexicalSimulator

        return LexicalSimulator()
    try:
        return _backend(name, allow_offline)
    except BackendError as exc:
        print(f"jevtools bench: {type(exc).__name__}: {exc}", file=err)
        return None


def _cmd_bench_app(args: argparse.Namespace, out: TextIO, err: TextIO) -> int:
    from jevtools.bench.app import DOMAINS, load_domain, run_app

    names = tuple(d.strip() for d in args.domains.split(",") if d.strip()) if args.domains else None
    if names is None:
        names = DOMAINS if args.dir is None else tuple(sorted(p.name for p in Path(args.dir).iterdir()
                                                              if (p / "cases.jsonl").is_file()))  # fmt: skip
    backend = _bench_backend(args.backend, args.allow_offline, err)
    if backend is None:
        return 1
    try:
        cases = [(name, case) for name in names for case in load_domain(name, args.dir)]
    except FileNotFoundError as exc:
        print(f"jevtools bench: {exc}", file=err)
        return 1
    if args.limit is not None:
        cases = [c for name in names for c in [x for x in cases if x[0] == name][: args.limit]]
    if args.replays < 1:
        print("jevtools bench: --replays must be >= 1", file=err)
        return 1
    policy = Policy.from_toml(args.policy) if args.policy else None
    report = run_app(cases, backend, replays=args.replays, controls=args.controls, policy=policy,
                     meta={"domains": list(names), "dir": args.dir, "replays": args.replays,
                           "policy": policy.version if policy is not None else None})  # fmt: skip
    print(f"app bench: {len(cases)} case(s) in {len(names)} domain(s) on {report.mode}", file=out)
    print(report.render(tags=args.tags), file=out)
    if report.mode == "oracle":
        print("note: the oracle answers every question perfectly from the gold label; its accuracy is the ceiling "
              "of what jevtools can emit (candidate coverage, decoding, policy), not a measurement of Jev",
              file=out)  # fmt: skip
    elif getattr(backend, "name", "") == "simulator":
        print(OFFLINE_NOTE.format(name="the simulator"), file=out)
    if args.out:
        report.save(args.out)
        print(f"wrote {args.out}", file=out)
    return 0


def _cmd_bench(args: argparse.Namespace, out: TextIO, err: TextIO) -> int:
    if args.suite == "app":
        return _cmd_bench_app(args, out, err)
    from jevtools.bench.bfcl import CATEGORIES, DEFAULT_CATEGORIES, download, load_category
    from jevtools.bench.run import run_bfcl
    from jevtools.validate import cache_dir

    data = Path(args.data) if args.data else cache_dir() / "bfcl"
    names = (
        CATEGORIES
        if args.categories == "all"
        else (
            tuple(c.strip() for c in args.categories.split(",") if c.strip()) if args.categories else DEFAULT_CATEGORIES
        )
    )
    if args.download:
        written = download(data, names, ref=args.ref)
        print(f"downloaded {len(written)} BFCL file(s) into {data} (ref {args.ref})", file=out)
    backend = _bench_backend(args.backend, args.allow_offline, err)
    if backend is None:
        return 1
    try:
        cases = [c for name in names for c in load_category(data, name, limit=args.limit)]
    except FileNotFoundError as exc:
        print(f"jevtools bench: {exc.filename} not found; run with --download (or --data DIR)", file=err)
        return 1

    def progress(index: int, record: Any) -> None:
        if (index + 1) % 100 == 0:
            print(f"  {index + 1}/{len(cases)}", file=err)

    risk = None if args.risk == "infer" else args.risk
    report = run_bfcl(cases, backend, risk=risk, progress=progress, meta={"data": str(data), "ref": args.ref})
    print(f"BFCL {len(cases)} case(s) on {report.mode} (risk {risk or 'inferred'})", file=out)
    print(report.render(), file=out)
    if report.mode == "oracle":
        print("note: the oracle answers every question perfectly from the BFCL answer; its accuracy is the ceiling "
              "of what jevtools can emit (candidate coverage, decoding, policy), not a measurement of Jev",
              file=out)  # fmt: skip
    elif getattr(backend, "name", "") == "simulator":
        print(OFFLINE_NOTE.format(name="the simulator"), file=out)
    if args.out:
        report.save(args.out)
        print(f"wrote {args.out}", file=out)
    return 0


def main(argv: Sequence[str] | None = None, *, out: TextIO | None = None, err: TextIO | None = None) -> int:
    """Run the command line; returns the exit status (0 ok, 1 failed check or error, 2 usage/unavailable)."""
    out = out or sys.stdout
    err = err or sys.stderr
    parser = build_parser()
    try:
        args = parser.parse_args(list(argv) if argv is not None else None)
    except SystemExit as exc:
        return int(exc.code or 0) if not isinstance(exc.code, str) else 2
    try:
        if args.command == "lint":
            return _cmd_lint(args, out)
        if args.command == "explain":
            return _cmd_explain(args, out)
        if args.command == "verify":
            return _cmd_verify(args, out)
        if args.command == "probe":
            return _cmd_probe(args, out, err)
        if args.command == "eval":
            return _cmd_eval(args, out, err)
        if args.command == "fixtures":
            return _cmd_fixtures(args, out, err)
        if args.command == "tune":
            return _cmd_tune(args, out)
        if args.command == "bench":
            return _cmd_bench(args, out, err)
        return _cmd_serve(args, out, err)
    except (OSError, ValueError, JevtoolsError) as exc:
        print(f"jevtools {args.command}: {type(exc).__name__}: {exc}", file=err)
        return 1


__all__ = [
    "build_parser",
    "build_sources",
    "describe_slot",
    "explain",
    "jaccard",
    "lint",
    "load_catalog",
    "load_context",
    "load_source_specs",
    "load_tools",
    "load_trace",
    "main",
    "verb_problem",
]

if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
