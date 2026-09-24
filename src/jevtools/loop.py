"""The agent loop (spec §6, profile ``ext.loop``): steps, observations as candidate pools, memory, execution safety.

An :class:`Agent` drives a :class:`~jevtools.router.Router` in loop mode. Each **step** compiles one round (the tool
Choice with ``DONE`` once something was done, ``done_after`` per speculated tool, ``progress`` and ``observations``
in the state), decides, and acts on the outcome:

- ``execute``: run the call through the host's executor with its idempotency key (never twice), ingest the result as
  an :class:`Observation` (:func:`ingest_observation`), update the :class:`EntityStore`, and stop with ``done`` when
  ``done_after(t*) ≥ 0.8`` after a success (P10); otherwise take the next step;
- ``confirm``/``clarify``: return control with the ``Pending``; :meth:`Agent.resume` continues the loop (a delayed
  call is revalidated by the router's TOCTOU check, and by the host's ``revalidate`` hook if given);
- ``escalate``/``abstain``/``refuse``/``done``: return.

Guards stop the loop with ``escalate``: the step, round, cost and LLM-call caps of :class:`LoopBudget`, **repeat
detection** (the same ``(tool, arguments)`` as an earlier successful step → ``escalate(loop)``) and **no progress**
(two steps without a new observation → ``escalate(no_progress)``). Every step re-asks every question, because the
state changed (I4); nothing is carried across steps but code-side memory.

Observations never reach Jev in full: the state shows a preview (BM25-selected chunks), the full content is
late-bound into arguments by code (the content handle ``⟨full text of the file read in step 1⟩``), and everything
parsed from a result carries the untrusted ``tool_output`` channel, so the slot allow-lists (I2) keep it out of
identity slots of external and critical tools.
"""

from __future__ import annotations

import asyncio
import inspect
import re
import threading
from collections.abc import Callable, Generator, Iterable, Iterator, Mapping, Sequence
from concurrent.futures import Future
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from functools import cache
from typing import Any, Literal, Protocol, TypeVar, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from jevtools._compat import run_sync
from jevtools.candidates import Channel, value_key
from jevtools.canonical import canonical_str, jsonable, sha256_hex, sha256_of
from jevtools.context import Context, Message, Observation, Turn, parse_messages
from jevtools.decision import Decision, Pending, ToolCall
from jevtools.extract.base import SourceText
from jevtools.extract.catalogs import currencies
from jevtools.extract.money import SYMBOLS as MONEY_SYMBOLS
from jevtools.extract.money import WORD_CODES
from jevtools.extract.patterns import extract as extract_patterns
from jevtools.extract.tokens import fold
from jevtools.kinds.text import handle_text
from jevtools.policy import Outcome, Policy, Tier, loop_done
from jevtools.preview import PREVIEW_CHARS, chunk_text, select_preview
from jevtools.sources.registry import render_template
from jevtools.sources.toolsource import (
    child_path,
    get_any,
    is_mcp_result,
    jsonpath_matches,
    mcp_text,
    result_data,
)
from jevtools.spec.models import ToolSpec

MAX_ITEMS = 200
"""Items kept per observation (typed items, flattened leaves and text entities together)."""
LABEL_MAX = 64
JEV_USD_PER_INPUT_TOKEN = 0.042e-6
"""Jev input price [V]; used only to *estimate* cost for the loop's cost cap when a backend reports none."""
IDEMPOTENCY_META_KEY = "jevtools/idempotency_key"
"""MCP ``_meta`` key carrying the idempotency key (§6.5)."""
_ASYNC_EXECUTOR_IN_LOOP = "an async executor cannot run from Agent.run() inside an event loop; use Agent.arun()"

RULE_REPEAT = "loop.repeat"
RULE_NO_PROGRESS = "loop.no_progress"
RULE_MAX_STEPS = "loop.max_steps"
RULE_MAX_ROUNDS = "loop.max_rounds"
RULE_MAX_COST = "loop.max_cost"
RULE_MAX_LLM_CALLS = "loop.max_llm_calls"
RULE_RETRY_CONFIRM = "loop.retry_needs_confirm"
RULE_REVALIDATE = "loop.revalidate"
RULE_DONE = "P10.loop.done"


# ====================================================================================================================
# Budget
# ====================================================================================================================


class LoopBudget(BaseModel):
    """Loop guards (spec §6.1). Every cap is checked before a new step starts.

    - ``max_steps``: loop steps (a step is one fresh round plus its resumes);
    - ``max_rounds``: Jev rounds over the whole run, resumes included;
    - ``max_cost_usd``: reported Jev cost (``usage.cost``), else an estimate from input tokens at the list price;
    - ``max_llm_calls``: Filler/Escalator/text-LLM calls (checked only when the router has an LLM configured);
    - ``max_retries``: automatic retries of a failed ``read`` or ``idempotent`` tool (others never auto-retry);
    - ``no_progress_steps``: consecutive executed steps without a new observation before ``escalate(no_progress)``.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    max_steps: int = 6
    max_rounds: int = 12
    max_cost_usd: float = 0.01
    max_llm_calls: int = 2
    max_retries: int = 1
    no_progress_steps: int = 2

    @classmethod
    def from_policy(cls, policy: Policy) -> LoopBudget:
        """The caps of ``policy.loop`` (Appendix B ``[loop]``)."""
        loop = policy.loop
        return cls(max_steps=loop.max_steps, max_rounds=loop.max_rounds, max_cost_usd=loop.max_cost_usd,
                   max_llm_calls=loop.max_llm_calls)  # fmt: skip


# ====================================================================================================================
# Observations (§6.3)
# ====================================================================================================================

_AMOUNT = r"\d{1,3}(?:[,' ’]\d{3})+(?:\.\d{1,2})?|\d+(?:\.\d{1,2})?"


@cache
def _money_re() -> re.Pattern[str]:
    """Money in observation text: an amount with an ISO 4217 code (from the currency catalog; codes that are also
    words, :data:`jevtools.extract.money.WORD_CODES`, are left out) or a currency symbol
    (:data:`jevtools.extract.money.SYMBOLS`) before or after it."""
    codes = sorted((c for c in currencies() if c not in WORD_CODES), key=lambda c: (-len(c), c))
    marker = "|".join([*codes, *(re.escape(sym) for sym in MONEY_SYMBOLS)])
    before, after = rf"(?P<c1>{marker})\s?(?P<a1>{_AMOUNT})", rf"(?P<a2>{_AMOUNT})\s?(?P<c2>{marker})"
    return re.compile(rf"(?<![\w.])(?:{before}|{after})(?![\w])")


DATE_RE = re.compile(r"(?<!\d)(?:\d{4}-\d{2}-\d{2}|\d{1,2}\.\d{1,2}\.\d{4})(?!\d)")
IBAN_RE = re.compile(r"\b[A-Z]{2}\d{2}(?: ?[A-Z0-9]{4}){2,7}(?: ?[A-Z0-9]{1,4})?\b")
DOC_ID_RE = re.compile(r"\b[A-Z][A-Z0-9]{0,9}-\d{2,}\b")
PATH_RE = re.compile(r"(?<![\w/@.:-])(?:[\w.-]+/)+[\w.-]+\.[A-Za-z0-9]{1,6}\b")
ENTITY_TYPES: tuple[str, ...] = ("email", "url", "uuid", "ipv4", "iban", "money", "date", "id", "path")
"""Entity types found in observation text by regex (``x-jev.emits.types`` narrows them)."""


class ObservationItem(BaseModel):
    """One thing parsed from a tool result: a typed item (``x-jev.emits``/``outputSchema``), a flattened JSON leaf
    (type ``leaf``) or a regex entity found in text (``email``, ``url``, ``money``, ``date``, ``id``, ``iban``,
    ``path``…). ``ref`` is the provenance ``obs:<step>:<JSONPath>``; the channel is always ``tool_output``."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    type: str
    value: Any
    label: str
    path: str
    """JSONPath inside the result (``$`` for the whole text)."""
    ref: str
    span: tuple[int, int] | None = None
    """Character span inside the text at ``path`` (regex entities)."""
    attrs: dict[str, Any] = Field(default_factory=dict)
    """Fields of a typed item (for ``render``/``order_by``; never sent to Jev)."""
    channel: Channel = Channel.TOOL_OUTPUT


class LoopObservation(Observation):
    """An :class:`~jevtools.context.Observation` with what code parsed from it (spec §6.3).

    ``content`` is the full result (late-bound by code, never sent), ``preview`` the BM25-selected chunks Jev sees,
    ``items`` the parsed items and entities (all ``tool_output``), ``handle`` the content handle's text
    (``⟨full text of the file read in step 1⟩``) and ``error`` the error text of a failed call (untrusted).
    """

    items: tuple[ObservationItem, ...] = ()
    handle: str = ""
    error: str | None = None

    def items_of(self, *types: str) -> list[ObservationItem]:
        """Items of the given types (all items without arguments)."""
        return [item for item in self.items if not types or item.type in types]


def _truncate(text: str, limit: int = LABEL_MAX) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def _money_value(amount: str, currency: str) -> str:
    digits = re.sub(r"[,' ’]", "", amount)
    try:
        number = Decimal(digits)
    except InvalidOperation:
        return f"{amount} {currency}"
    return f"{number.quantize(Decimal('0.01'))} {MONEY_SYMBOLS.get(currency, currency)}"


def text_entities(text: str, types: Iterable[str] | None = None) -> list[tuple[str, Any, str, tuple[int, int]]]:
    """Regex entities in ``text``: ``(type, normalized value, matched text, span)`` in text order.

    Types: ``email`` (domain lowercased), ``url``, ``uuid``, ``ipv4``, ``iban`` (spaces removed), ``money``
    (``"4820.00 CHF"``), ``date`` (as written), ``id`` (document ids such as ``INV-2291``) and ``path``. Emails,
    URLs, UUIDs and IPv4 addresses come from the request-side pattern extractor
    (:func:`jevtools.extract.patterns.extract`: valid IPv4 only, URL trailing punctuation cut from text and span).
    """
    wanted = set(types) if types is not None else set(ENTITY_TYPES)
    found: list[tuple[str, Any, str, tuple[int, int]]] = []
    taken: list[tuple[int, int]] = []

    def add(kind: str, value: Any, matched: str, span: tuple[int, int]) -> None:
        if kind not in wanted or any(s < span[1] and span[0] < e for s, e in taken):
            return
        taken.append(span)
        found.append((kind, value, matched, span))

    patterns = extract_patterns(SourceText("$", text, Channel.TOOL_OUTPUT, ()))
    for kind in ("email", "url", "uuid"):  # priority order; ipv4 comes last
        for m in patterns:
            if m.kind == kind:
                add(kind, str(m.value).lower() if kind == "uuid" else m.value, m.text, m.span)
    for match in IBAN_RE.finditer(text):
        add("iban", match.group().replace(" ", ""), match.group(), match.span())
    for match in _money_re().finditer(text):
        amount, currency = match.group("a1") or match.group("a2"), match.group("c1") or match.group("c2")
        add("money", _money_value(amount, currency), match.group(), match.span())
    for match in DATE_RE.finditer(text):
        add("date", match.group(), match.group(), match.span())
    for match in PATH_RE.finditer(text):
        add("path", match.group(), match.group(), match.span())
    for match in DOC_ID_RE.finditer(text):
        add("id", match.group(), match.group(), match.span())
    for m in patterns:
        if m.kind == "ipv4":
            add("ipv4", m.value, m.text, m.span)
    found.sort(key=lambda entry: entry[3])
    return found


def _flatten(value: Any, path: str = "$") -> Iterator[tuple[str, Any]]:
    """JSON leaves with their JSONPath (strings, numbers, booleans; ``null`` and empty containers skipped)."""
    if isinstance(value, Mapping):
        for key, item in value.items():
            yield from _flatten(item, child_path(path, str(key)))
    elif isinstance(value, list):
        for i, item in enumerate(value):
            yield from _flatten(item, f"{path}[{i}]")
    elif value is not None:
        yield path, value


def _schema_items(schema: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """An ``emits``-like spec from an MCP ``outputSchema``: the first array-of-objects property becomes the items;
    the key is a ``uri``/``email``/``uuid``-format property or one named ``id``/``url``/``uri``/``email``/``path``;
    the label a ``title``/``name``/``label``/``subject`` property."""
    if not isinstance(schema, Mapping):
        return None
    properties = schema.get("properties") or {}
    for name, sub in properties.items():
        items = sub.get("items") if isinstance(sub, Mapping) and sub.get("type") == "array" else None
        if not isinstance(items, Mapping) or not isinstance(items.get("properties"), Mapping):
            continue
        fields: Mapping[str, Any] = items["properties"]
        key = next((f for f, s in fields.items() if isinstance(s, Mapping)
                    and s.get("format") in ("uri", "email", "uuid")), None)  # fmt: skip
        key = key or next((f for f in ("id", "url", "uri", "email", "path", "key") if f in fields), None)
        label = next((f for f in ("title", "name", "label", "subject") if f in fields), key)
        return {"items": f"$.{name}[*]" if name.isidentifier() else f'$["{name}"][*]', "key": key,
                "label": f"{{{label}}}" if label else None}  # fmt: skip
    return None


def _normalize_result(result: Any, error: BaseException | None) -> tuple[str, Any, str | None]:
    """``(status, content, error text)`` of a raw result (an exception or an MCP ``isError`` result is an error)."""
    if error is None and isinstance(result, BaseException):
        error = result
    if error is not None:
        text = f"{type(error).__name__}: {error}"
        return "error", text, text
    if is_mcp_result(result) and mcp_is_error(result):
        text = mcp_text(result) or "tool error"
        return "error", text, text
    return "ok", result_data(result), None


def mcp_is_error(result: Any) -> bool:
    """``isError`` (or ``is_error``) of an MCP ``CallToolResult`` (dict or ``mcp`` object): the one error test of
    the Agent's observations and :func:`jevtools.adapters.mcp.is_error`."""
    return bool(get_any(result, "isError", "is_error"))


def ingest_observation(
    result: Any,
    tool: str | ToolSpec,
    step: int,
    *,
    arguments: Mapping[str, Any] | None = None,
    call_id: str | None = None,
    request: str = "",
    error: BaseException | None = None,
    emits: Mapping[str, Any] | None = None,
    output_schema: Mapping[str, Any] | None = None,
    preview_chars: int = PREVIEW_CHARS,
) -> LoopObservation:
    """Turn a tool result into an :class:`Observation` whose parts can become candidates (spec §6.3).

    1. **Receive**: a value, a JSON string, an MCP ``CallToolResult`` (``structuredContent`` first; ``isError`` →
       status ``error``) or an exception (``error``, or ``result`` itself) → status ``error`` with the error text.
    2. **Parse**: with ``x-jev.emits`` (or an MCP ``outputSchema``), the items it selects are typed items (value =
       the ``key`` field, label = the ``label`` template); unknown JSON is flattened to leaves with their JSONPath;
       text is chunked into sentences and scanned for regex entities (emails, URLs, dates, money, ids, paths…),
       restricted to ``emits.types`` when declared. JSON string leaves are scanned too.
    3. **Tag**: every item carries ``tool_output`` and the provenance ``obs:<step>:<path>``.
    4. **Content handle**: :attr:`LoopObservation.handle` is ``⟨full text of the … read in step k⟩``; the full
       content stays in ``content`` for code to late-bind, and only the preview reaches the state.

    ``tool`` may be a :class:`~jevtools.spec.models.ToolSpec` (its ``emits`` and ``output_schema`` are used) or a
    name. ``request`` steers the preview's chunk selection (BM25).
    """
    spec = tool if isinstance(tool, ToolSpec) else None
    name = spec.name if spec is not None else str(tool)
    emits = dict(emits if emits is not None else (spec.emits or {}) if spec is not None else {})
    if not emits.get("items"):
        emits = {**emits, **(_schema_items(output_schema or (spec.output_schema if spec else None)) or {})}
    status, content, error_text = _normalize_result(result, error)
    types = emits.get("types")
    items: list[ObservationItem] = []

    def ref(path: str) -> str:
        return f"obs:{step}:{path}"

    def scan(text: str, path: str) -> None:
        for kind, value, shown, span in text_entities(text, types):
            items.append(ObservationItem(type=kind, value=value, label=_truncate(shown), path=path, ref=ref(path),
                                         span=span))  # fmt: skip

    if isinstance(content, str):
        chunks = chunk_text(content)
        scan(content, "$")
        summary = f"{len(content.split()):,} words" if status == "ok" else _truncate(content, 80)
    else:
        chunks = []
        typed = 0
        if emits.get("items"):
            key, label = emits.get("key"), emits.get("label")
            for path, item in jsonpath_matches(content, str(emits["items"])):
                value = item.get(key) if key and isinstance(item, Mapping) else item
                if value is None:
                    continue
                shown = render_template(label, item) if label and isinstance(item, Mapping) else ""
                shown = shown or (value if isinstance(value, str) else canonical_str(jsonable(value)))
                attrs = dict(item) if isinstance(item, Mapping) else {}
                items.append(ObservationItem(type=str(key or "item"), value=jsonable(value), label=_truncate(shown),
                                             path=path, ref=ref(path), attrs=jsonable(attrs)))  # fmt: skip
                describe = emits.get("describe")
                extra = render_template(describe, item) if describe and isinstance(item, Mapping) else ""
                chunks.append(" — ".join(p for p in (_truncate(shown, 200), extra) if p))
                typed += 1
        leaves = list(_flatten(content))
        for path, leaf in leaves:
            if isinstance(leaf, str):
                scan(leaf, path)
            if not typed:
                shown = leaf if isinstance(leaf, str) else canonical_str(jsonable(leaf))
                items.append(ObservationItem(type="leaf", value=jsonable(leaf), label=_truncate(shown), path=path,
                                             ref=ref(path)))  # fmt: skip
                chunks.append(f"{path.removeprefix('$.')}: {_truncate(shown, 200)}")
        count = typed or len(leaves)
        summary = f"{count} item{'s' if count != 1 else ''}" if typed or isinstance(content, list) else \
            f"{count} field{'s' if count != 1 else ''}"  # fmt: skip
    rendered = content if isinstance(content, str) else canonical_str(jsonable(content))
    preview = rendered if len(rendered) <= preview_chars else select_preview(chunks or [rendered], request,
                                                                             preview_chars)  # fmt: skip
    base = LoopObservation(step=step, tool=name, status="ok" if status == "ok" else "error", content=content,
                           preview=preview, arguments=dict(arguments or {}), summary=summary, call_id=call_id,
                           items=tuple(items[:MAX_ITEMS]), error=error_text)  # fmt: skip
    return base.model_copy(update={"handle": handle_text(base)})


# ====================================================================================================================
# Memory: the entity store (§6.4)
# ====================================================================================================================

_TYPE_PRIORITY = ("email", "account_id", "path", "url", "uri", "phone", "iban", "person")


class Entity(BaseModel):
    """A remembered value (spec §6.4). ``channel`` is ``history`` (memory); ``origin`` is the channel the value
    came from, whose trust a coreference candidate inherits. Values bound in executed calls are ``pinned``."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str
    type: str
    value: Any
    label: str
    channel: Channel = Channel.HISTORY
    origin: Channel
    source_ref: str
    turn: int = 0
    step: int | None = None
    pinned: bool = False
    tool: str | None = None
    attrs: dict[str, Any] = Field(default_factory=dict)

    @staticmethod
    def make_id(type_: str, value: Any) -> str:
        """``ent_<hash>`` of the type and the canonical value."""
        return "ent_" + sha256_hex(f"{type_}\x1f{value_key(jsonable(value))}")[:12]


class EntityStore:
    """Entities bound in executed calls (pinned), parsed from observations (``origin = tool_output``) and named in
    assistant turns (spec §6.4). One entity per ``(type, value)``: a later sighting refreshes turn/step, keeps the
    most trusted origin seen and never unpins. Used for coreference candidates (``jevtools.extract.coref`` reads
    this store through ``Context.entities``) and ``same_as_last`` derivations."""

    def __init__(self, entities: Iterable[Entity | Mapping[str, Any]] = ()) -> None:
        self._entities: dict[str, Entity] = {}
        for entity in entities:
            self.add(entity if isinstance(entity, Entity) else Entity.model_validate(entity))

    # -- container ------------------------------------------------------------------------------------------------

    @property
    def entities(self) -> list[Entity]:
        """All entities, oldest first."""
        return list(self._entities.values())

    def __iter__(self) -> Iterator[Entity]:
        return iter(list(self._entities.values()))

    def __len__(self) -> int:
        return len(self._entities)

    def __contains__(self, value: object) -> bool:
        return any(e.value == value for e in self._entities.values())

    def __repr__(self) -> str:
        return f"EntityStore({len(self)} entities, {len(self.pinned())} pinned)"

    def get(self, entity_id: str) -> Entity | None:
        return self._entities.get(entity_id)

    def of_type(self, *types: str) -> list[Entity]:
        """Entities of the given types, most recent first."""
        found = [e for e in self._entities.values() if e.type in types]
        return sorted(found, key=lambda e: (-e.turn, -(e.step or 0)))

    def pinned(self) -> list[Entity]:
        """Entities bound in executed calls."""
        return [e for e in self._entities.values() if e.pinned]

    def last(self, type_: str) -> Entity | None:
        """The most recent pinned entity of a type (``same_as_last``), else the most recent one."""
        entities = self.of_type(type_)
        return next((e for e in entities if e.pinned), entities[0] if entities else None)

    def mentions_pinned(self, text: str) -> bool:
        """Whether ``text`` mentions a pinned entity by value or label (history trimming keeps such turns, §6.2)."""
        folded = fold(text)
        for entity in self.pinned():
            head = re.split(r" <| · | \(", entity.label, maxsplit=1)[0]
            probes = [entity.label, head, entity.attrs.get("name"), entity.value if isinstance(entity.value, str)
                      else None]  # fmt: skip
            if any(isinstance(p, str) and len(p) >= 3 and fold(p) in folded for p in probes):
                return True
        return False

    # -- adding ---------------------------------------------------------------------------------------------------

    def add(self, entity: Entity) -> Entity:
        """Add or merge an entity (merge: newest turn/step and label, most trusted origin, pinned sticks). A
        ``history`` sighting (an assistant mention, a coreference) is memory, not a source: it never raises the
        trust of a less trusted origin already seen (a ``tool_output`` value stays ``tool_output``, §3.4.2)."""
        old = self._entities.get(entity.id)
        if old is None:
            self._entities[entity.id] = entity
            return entity
        if entity.origin is Channel.HISTORY and old.origin.trust > Channel.HISTORY.trust:
            origin = old.origin
        else:
            origin = old.origin if old.origin.trust <= entity.origin.trust else entity.origin
        newer = (entity.turn, entity.step or 0) >= (old.turn, old.step or 0)
        merged = (entity if newer else old).model_copy(update={
            "origin": origin, "pinned": old.pinned or entity.pinned,
            "attrs": {**old.attrs, **entity.attrs},
        })  # fmt: skip
        self._entities[entity.id] = merged
        return merged

    def remember(
        self,
        type_: str,
        value: Any,
        *,
        label: str | None = None,
        origin: Channel | str,
        source_ref: str,
        turn: int = 0,
        step: int | None = None,
        pinned: bool = False,
        tool: str | None = None,
        attrs: Mapping[str, Any] | None = None,
    ) -> Entity:
        """Add one value (see :meth:`add`)."""
        value = jsonable(value)
        shown = label or (value if isinstance(value, str) else canonical_str(value))
        return self.add(Entity(id=Entity.make_id(type_, value), type=type_, value=value, label=_truncate(shown),
                               origin=Channel(origin), source_ref=source_ref, turn=turn, step=step, pinned=pinned,
                               tool=tool, attrs=dict(jsonable(dict(attrs or {})))))  # fmt: skip

    def pin_call(
        self, call: ToolCall, tool: ToolSpec | None, *, decision: Decision | None = None, turn: int = 0,
        step: int | None = None,
    ) -> list[Entity]:  # fmt: skip
        """Pin the identity values of an executed call (list items one by one); the origin is the bound value's
        channel from the decision's trace (``user`` when unknown), the label its elected label. A value bound
        through a ``history`` (coreference) candidate keeps its entity's origin: binding it never launders trust."""
        bindings: Mapping[str, Mapping[str, Any]] = {}
        trace = getattr(decision, "trace", None)
        if trace is not None:
            bindings = getattr(trace, "bindings", {}) or {}
        pinned: list[Entity] = []
        for name, value in call.arguments.items():
            slot = next((s for s in tool.slots if s.name == name), None) if tool is not None else None
            if slot is not None and (slot.stakes != "identity" or slot.kind in ("text", "secret", "derived")):
                continue
            binding = bindings.get(name, {})
            channel = self._bound_origin(binding)
            type_ = _slot_type(slot, name)
            values = value if isinstance(value, list) else [value]
            for item in values:
                if isinstance(item, (dict, list)) or item is None:
                    continue
                label = binding.get("label") if not isinstance(value, list) else None
                pinned.append(self.remember(type_, item, label=label, origin=channel, turn=turn, step=step,
                                            source_ref=f"call:{call.id}:{name}", pinned=True,
                                            tool=call.name))  # fmt: skip
        return pinned

    def _bound_origin(self, binding: Mapping[str, Any]) -> Channel:
        """The origin of a bound value: its trace channel, except that a ``history`` value inherits the origin of
        the entity it came from (``prov.entity``, spec §3.4.2 / §6.4)."""
        channel = Channel(binding.get("channel") or Channel.USER)
        prov = binding.get("prov")
        if channel is Channel.HISTORY and isinstance(prov, Mapping):
            entity = self._entities.get(str(prov.get("entity")))
            if entity is not None:
                return entity.origin
        return channel

    def add_observation(self, observation: Observation, *, turn: int = 0) -> list[Entity]:
        """Remember an observation's typed items and text entities (``origin = tool_output``; leaves are not
        remembered)."""
        added: list[Entity] = []
        for item in getattr(observation, "items", ()):
            if item.type == "leaf":
                continue
            added.append(self.remember(item.type, item.value, label=item.label, origin=Channel.TOOL_OUTPUT,
                                       source_ref=item.ref, turn=turn, step=observation.step, tool=observation.tool,
                                       attrs=item.attrs))  # fmt: skip
        return added

    def add_assistant_mentions(self, ctx: Context, *, turn: int = 0) -> list[Entity]:
        """Remember values named in assistant turns: regex entities (``origin = history``) and registry rows named by
        their full match value, e.g. "Anna Keller" (``origin = registry``: the value is the row's key)."""
        added: list[Entity] = []
        for index, message in enumerate(ctx.messages):
            if message.role != "assistant" or not message.text:
                continue
            where = f"assistant:{index}"
            for kind, value, shown, _ in text_entities(message.text, ("email", "url", "iban", "id", "path")):
                added.append(self.remember(kind, value, label=shown, origin=Channel.HISTORY, source_ref=where,
                                           turn=turn))  # fmt: skip
            for source in ctx.sources.values():
                added += self._registry_mentions(source, message.text, where, turn)
        return added

    def _registry_mentions(self, source: Any, text: str, where: str, turn: int) -> list[Entity]:
        find = getattr(source, "find_anchors", None)
        rows = getattr(source, "rows", None)
        match_fields = tuple(getattr(source, "match", ()) or ())
        if not callable(find) or rows is None or not match_fields:
            return []
        added: list[Entity] = []
        for mention in find(text, source_ref=where, channel=Channel.HISTORY):
            for match in mention.attrs.get("matches", ()):
                row = rows[match.index]
                named = row.get(match_fields[0])
                if not isinstance(named, str) or fold(named) != fold(mention.text):
                    continue
                key = getattr(source, "key", None)
                if key is None or row.get(key) in (None, ""):
                    continue
                label_of = getattr(source, "label_of", None)
                label = label_of(row) if callable(label_of) else named
                provides = set(getattr(source, "provides", ()) or ())
                type_ = next((t for t in _TYPE_PRIORITY if t in provides), sorted(provides)[0] if provides else
                             str(getattr(source, "name", "item")))  # fmt: skip
                attrs_of = getattr(source, "attrs_of", None)
                attrs = attrs_of(row) if callable(attrs_of) else {}
                added.append(self.remember(type_, row[key], label=label, origin=Channel.REGISTRY, source_ref=where,
                                           turn=turn, attrs=attrs))  # fmt: skip
        return added

    # -- serialization --------------------------------------------------------------------------------------------

    def to_doc(self) -> dict[str, Any]:
        """``{"entities": [...]}`` (JSON-ready)."""
        return {"entities": [jsonable(e) for e in self._entities.values()]}

    def to_json(self) -> str:
        """Canonical JSON of :meth:`to_doc` (stable across processes; round-trips through :meth:`from_json`)."""
        return canonical_str(self.to_doc())

    @classmethod
    def from_json(cls, data: str | bytes | Mapping[str, Any] | Sequence[Mapping[str, Any]]) -> EntityStore:
        """Rebuild a store from :meth:`to_json` output (or the parsed document)."""
        import json

        parsed: Any = json.loads(data) if isinstance(data, (str, bytes)) else data
        raw = parsed.get("entities", []) if isinstance(parsed, Mapping) else parsed
        return cls(Entity.model_validate(entry) for entry in raw)


def _slot_type(slot: Any, name: str) -> str:
    """Entity type of a slot's values: a specific tag (email, account_id, path…), else its format, else its kind."""
    if slot is None:
        return name
    item = getattr(slot, "item", None)
    if item is not None:
        return _slot_type(item, name)
    tags = tuple(getattr(slot, "tags", ()) or ())
    for preferred in _TYPE_PRIORITY:
        if preferred in tags:
            return preferred
    if tags:
        return str(tags[0])
    return str(getattr(slot, "format", None) or getattr(slot, "kind", None) or name)


# ====================================================================================================================
# Executors (§6.5)
# ====================================================================================================================


@runtime_checkable
class Executor(Protocol):
    """A tool implementation. Per-tool executors (``{"send_email": fn}``) are called with the call's arguments as
    keywords; a single dispatcher is called with the :class:`~jevtools.decision.ToolCall`. Either receives
    ``idempotency_key=`` when it declares that parameter. It may be sync or async (return an awaitable)."""

    def __call__(self, *args: Any, **kwargs: Any) -> Any: ...


Executors = Mapping[str, Callable[..., Any]] | Callable[[ToolCall], Any]


def accepts_keyword(fn: Callable[..., Any], name: str) -> bool:
    """Whether ``fn`` declares a keyword-capable parameter ``name`` (``**kwargs`` does not count)."""
    try:
        parameters = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return False
    parameter = parameters.get(name)
    return parameter is not None and parameter.kind in (parameter.POSITIONAL_OR_KEYWORD, parameter.KEYWORD_ONLY)


class MissingExecutor(LookupError):
    """No executor is registered for a tool (reported as an error observation, never a guess)."""


# ====================================================================================================================
# Steps and results
# ====================================================================================================================


class LoopStep(BaseModel):
    """One decision taken in the loop and what the agent did with it."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    step: int
    decision_id: str | None = None
    outcome: str
    rule: str
    call: ToolCall | None = None
    executed: bool = False
    status: Literal["ok", "error"] | None = None
    attempts: int = 0
    observation: int | None = None
    """Step number of the observation this execution produced."""
    rounds: int = 0
    resumed: bool = False
    note: str | None = None


class LoopUsage(BaseModel):
    """Resources used by a run (resumes included)."""

    model_config = ConfigDict(extra="forbid")

    steps: int = 0
    rounds: int = 0
    jev_calls: int = 0
    jev_input_tokens: int = 0
    llm_calls: int = 0
    cost_usd: float | None = None
    """Sum of reported costs (``None`` when no backend reported one); never estimated."""
    est_cost_usd: float = 0.0
    """What the cost cap counts: reported costs, else input tokens at the Jev list price."""
    executions: int = 0


class LoopResult(BaseModel):
    """What a run returns: the final outcome and rule, every step and decision, the observations, the pending handle
    of a confirm/clarify, usage and the entity store. ``reason`` names a guard (``loop``, ``no_progress``,
    ``budget``…) or the P10 path (``done_after``, ``done``)."""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    outcome: Outcome
    rule: str
    reason: str | None = None
    steps: list[LoopStep] = Field(default_factory=list)
    decisions: list[Decision] = Field(default_factory=list)
    observations: list[Observation] = Field(default_factory=list)
    pending: Pending | None = None
    usage: LoopUsage = Field(default_factory=LoopUsage)
    entities: Any = None
    """The :class:`EntityStore` after the run."""
    content: str | None = None
    """Handoff text of a final abstain/escalate decision (``text_llm`` or Escalator)."""
    notes: list[str] = Field(default_factory=list)

    @property
    def decision(self) -> Decision | None:
        """The last decision."""
        return self.decisions[-1] if self.decisions else None

    @property
    def pending_id(self) -> str | None:
        return self.pending.pending_id if self.pending is not None else None

    @property
    def executed(self) -> list[ToolCall]:
        """Calls executed by this run, in order."""
        return [s.call for s in self.steps if s.executed and s.call is not None]


# ====================================================================================================================
# Agent
# ====================================================================================================================


@dataclass(frozen=True)
class _Decide:
    messages: list[Turn]
    context: Context


@dataclass(frozen=True)
class _Resume:
    pending: Pending | str
    selection: str | None
    reply: str | None
    context: Context


@dataclass(frozen=True)
class _Invoke:
    call: ToolCall


@dataclass(frozen=True)
class _Refresh:
    context: Context


@dataclass(frozen=True)
class _Wait:
    """Wait until another run releases a claim (a pending being resumed, a key being executed)."""

    claim: Future[None]


_Effect = _Decide | _Resume | _Invoke | _Refresh | _Wait
_Flow = Generator[_Effect, Any, LoopResult]
_T = TypeVar("_T")


@dataclass
class _Run:
    """The state of one loop run (kept across confirm/clarify pauses)."""

    base: Context
    messages: list[Turn]
    observations: list[Observation]
    turn: int
    step: int = 0
    steps: list[LoopStep] = field(default_factory=list)
    decisions: list[Decision] = field(default_factory=list)
    outcomes: dict[str, str] = field(default_factory=dict)
    """Call hash → status of its last execution (repeat detection)."""
    contents: set[str] = field(default_factory=set)
    stale: int = 0
    usage: LoopUsage = field(default_factory=LoopUsage)
    notes: list[str] = field(default_factory=list)


def _call_hash(call: ToolCall) -> str:
    return sha256_of({"name": call.name, "arguments": jsonable(call.arguments)})


class Agent:
    """Drive a :class:`~jevtools.router.Router` through a multi-step task (spec §6.1).

    ``executors`` is ``{tool name: callable}`` (called with the arguments as keywords), one dispatcher
    ``callable(call: ToolCall)``, or an MCP ``ClientSession``-like object (``call_tool(name, arguments, meta=…)``,
    the idempotency key in ``_meta["jevtools/idempotency_key"]``). ``budget`` defaults to the router policy's loop
    caps. ``entities`` is the conversation's :class:`EntityStore` (kept across runs). ``revalidate(call, context)``
    is an optional host TOCTOU check run before executing a delayed call (the router's own check always runs);
    problems cancel the execution and re-plan in a new step.

    Idempotency is per agent (the "session"): a key that was executed is never executed again, and resuming the
    same pending twice replays the first result. This also holds for concurrent calls (``asyncio.gather`` of two
    ``aresume``, two threads calling ``resume``): a pending or a key is claimed before anything runs, and a second
    caller waits for the first and then replays its result.
    """

    def __init__(
        self,
        router: Any,
        executors: Executors | Any,
        *,
        budget: LoopBudget | None = None,
        entities: EntityStore | None = None,
        revalidate: Callable[[ToolCall, Context], Sequence[str]] | None = None,
        preview_chars: int = PREVIEW_CHARS,
    ) -> None:
        self.router = router
        self.executors = executors
        self.budget = budget or LoopBudget.from_policy(router.policy)
        self.entities = entities if entities is not None else EntityStore()
        self.revalidate = revalidate
        self.preview_chars = preview_chars
        self.executed: dict[str, LoopObservation] = {}
        """Observations by idempotency key of every call executed in this session."""
        self._paused: dict[str, _Run] = {}
        self._results: dict[str, LoopResult] = {}
        self._claims: dict[str, Future[None]] = {}
        """In-flight resumes (``resume:<pending id>``) and executions (``key:<idempotency key>``)."""
        self._claims_lock = threading.Lock()

    # -- public API -----------------------------------------------------------------------------------------------

    def run(self, messages: str | Sequence[Message | Turn], context: Context | None = None) -> LoopResult:
        """Run the loop on ``messages`` until done, a prompt, a stop outcome or a guard (sync)."""
        return self._drive(self._start(messages, context))

    async def arun(self, messages: str | Sequence[Message | Turn], context: Context | None = None) -> LoopResult:
        """Async :meth:`run` (async executors and sources are awaited)."""
        return await self._adrive(self._start(messages, context))

    def resume(
        self,
        pending: Pending | str,
        *,
        selection: str | None = None,
        reply: str | None = None,
        context: Context | None = None,
    ) -> LoopResult:
        """Continue a run paused on a confirm/clarify: ``selection`` is a click, ``reply`` free text. ``context``
        replaces the host context (fresh sources for the TOCTOU check). Resuming the same pending again replays
        the first result: nothing is executed twice."""
        return self._drive(self._resume(pending, selection, reply, context))

    async def aresume(
        self,
        pending: Pending | str,
        *,
        selection: str | None = None,
        reply: str | None = None,
        context: Context | None = None,
    ) -> LoopResult:
        """Async :meth:`resume`."""
        return await self._adrive(self._resume(pending, selection, reply, context))

    # -- drivers --------------------------------------------------------------------------------------------------

    def execute(self, call: ToolCall, *, step: int | None = None, request: str = "") -> LoopObservation:
        """Execute one call outside a run (drop-in hosts) with the loop's safety: an idempotency key that already
        ran returns its observation without calling the executor again; read/idempotent tools retry automatically."""
        return self._drive(self._execute(call, self._tool(call.name), step or len(self.executed) + 1, request))[0]

    async def aexecute(self, call: ToolCall, *, step: int | None = None, request: str = "") -> LoopObservation:
        """Async :meth:`execute`."""
        flow = self._execute(call, self._tool(call.name), step or len(self.executed) + 1, request)
        return (await self._adrive(flow))[0]

    def _drive(self, flow: Generator[_Effect, Any, _T]) -> _T:
        value: Any = None
        try:
            while True:
                try:
                    effect = flow.send(value)
                except StopIteration as stop:
                    result: _T = stop.value
                    return result
                value = self._perform(effect)
        finally:
            flow.close()  # an effect that raised: release the flow's claims now, not at garbage collection

    async def _adrive(self, flow: Generator[_Effect, Any, _T]) -> _T:
        value: Any = None
        try:
            while True:
                try:
                    effect = flow.send(value)
                except StopIteration as stop:
                    result: _T = stop.value
                    return result
                value = await self._aperform(effect)
        finally:
            flow.close()

    def _claim(self, name: str) -> tuple[Future[None], bool]:
        """Claim ``name`` for this flow: ``(new claim, True)``, or ``(the holder's claim, False)`` when another flow
        holds it (wait for it, then claim again)."""
        with self._claims_lock:
            held = self._claims.get(name)
            if held is not None:
                return held, False
            claim: Future[None] = Future()
            self._claims[name] = claim
            return claim, True

    def _release(self, name: str, claim: Future[None]) -> None:
        with self._claims_lock:
            if self._claims.get(name) is claim:
                del self._claims[name]
        claim.set_result(None)

    def _perform(self, effect: _Effect) -> Any:
        if isinstance(effect, _Wait):
            if not effect.claim.done():
                try:
                    asyncio.get_running_loop()
                except RuntimeError:
                    pass
                else:  # blocking here would stall the loop that holds the claim
                    raise RuntimeError("this pending or call is being resumed by a coroutine of the running event "
                                       "loop; use Agent.aresume()/aexecute() inside an event loop")  # fmt: skip
            effect.claim.result()
            return None
        if isinstance(effect, _Decide):
            return self.router.decide(effect.messages, context=effect.context, mode="loop")
        if isinstance(effect, _Resume):
            return self.router.resume(effect.pending, selection=effect.selection, reply=effect.reply,
                                      context=effect.context)  # fmt: skip
        if isinstance(effect, _Refresh):
            for source in effect.context.sources.values():
                refresh = getattr(source, "refresh", None)
                if callable(refresh) and getattr(source, "stale", False):
                    refresh()
            return None
        try:
            return run_sync(self._invoke(effect.call), _ASYNC_EXECUTOR_IN_LOOP), None
        except Exception as exc:  # noqa: BLE001 - a tool error becomes an observation
            return None, exc

    async def _aperform(self, effect: _Effect) -> Any:
        if isinstance(effect, _Wait):
            await asyncio.wrap_future(effect.claim)
            return None
        if isinstance(effect, _Decide):
            return await self.router.adecide(effect.messages, context=effect.context, mode="loop")
        if isinstance(effect, _Resume):
            return await self.router.aresume(effect.pending, selection=effect.selection, reply=effect.reply,
                                             context=effect.context)  # fmt: skip
        if isinstance(effect, _Refresh):
            for source in effect.context.sources.values():
                arefresh = getattr(source, "arefresh", None)
                if callable(arefresh) and getattr(source, "stale", False):
                    await arefresh()
            return None
        try:
            value = self._invoke(effect.call)
            if inspect.isawaitable(value):
                value = await value
            return value, None
        except Exception as exc:  # noqa: BLE001 - a tool error becomes an observation
            return None, exc

    # -- executors ------------------------------------------------------------------------------------------------

    def _invoke(self, call: ToolCall) -> Any:
        """Call the executor of ``call`` (the result may be awaitable)."""
        executors = self.executors
        arguments = dict(call.arguments)
        if isinstance(executors, Mapping):
            fn = executors.get(call.name)
            if fn is None:
                raise MissingExecutor(f"no executor for tool {call.name!r}")
            if accepts_keyword(fn, "idempotency_key"):
                return fn(**arguments, idempotency_key=call.idempotency_key)
            return fn(**arguments)
        call_tool = getattr(executors, "call_tool", None)
        if callable(call_tool) and not callable(executors):  # an MCP ClientSession-like object
            meta = {IDEMPOTENCY_META_KEY: call.idempotency_key}
            if accepts_keyword(call_tool, "meta"):
                return call_tool(call.name, arguments, meta=meta)
            return call_tool(call.name, arguments)
        if not callable(executors):
            raise MissingExecutor(f"no executor for tool {call.name!r}")
        dispatch: Callable[..., Any] = executors
        if accepts_keyword(dispatch, "idempotency_key"):
            return dispatch(call, idempotency_key=call.idempotency_key)
        return dispatch(call)

    def _source_call(self, tool: str, args: dict[str, Any]) -> Any:
        """The caller bound to :class:`~jevtools.sources.toolsource.ToolSource` objects: the tool's executor."""
        call = ToolCall.build(tool, jsonable(args), trace_id="source")
        return self._invoke(call)

    def _bind_sources(self, ctx: Context) -> None:
        for source in ctx.sources.values():
            bind = getattr(source, "bind", None)
            if callable(bind) and getattr(source, "bound", True) is False:
                bind(self._source_call)

    # -- runs -----------------------------------------------------------------------------------------------------

    def _context(self, run: _Run) -> Context:
        return run.base.model_copy(update={"messages": list(run.messages), "observations": list(run.observations),
                                           "entities": self.entities})  # fmt: skip

    def _start(self, messages: str | Sequence[Message | Turn], context: Context | None) -> _Flow:
        base = context or self.router.context
        turns = parse_messages(messages)
        run = _Run(base=base, messages=turns, observations=list(base.observations),
                   turn=sum(1 for t in turns if t.role == "user"))  # fmt: skip
        self._bind_sources(base)
        ctx = self._context(run)
        self.entities.add_assistant_mentions(ctx, turn=run.turn)
        return (yield from self._loop(run))

    def _resume(
        self, pending: Pending | str, selection: str | None, reply: str | None, context: Context | None
    ) -> _Flow:
        pending_id = pending if isinstance(pending, str) else pending.pending_id
        name = f"resume:{pending_id}"
        while True:  # claimed before the first effect, so a concurrent resume of the same pending waits and replays
            claim, mine = self._claim(name)
            if mine:
                break
            yield _Wait(claim)
        try:
            if pending_id in self._results:
                replay = self._results[pending_id]
                return replay.model_copy(update={"notes": [*replay.notes, f"replayed resume of {pending_id}"]})
            run = self._paused.pop(pending_id, None)
            if run is None:  # e.g. a restarted process: rebuild the run from the pending handle
                handle = pending if isinstance(pending, Pending) else self.router.pendings[pending_id]
                base = context or self.router.context
                turns = [Turn.model_validate(m) for m in handle.state.get("messages", [])]
                run = _Run(base=base, messages=turns, observations=list(base.observations),
                           turn=sum(1 for t in turns if t.role == "user"))  # fmt: skip
                run.notes.append(f"resumed {pending_id} without its paused run")
            elif context is not None:
                run.base = context
            self._bind_sources(run.base)
            ctx = self._context(run)
            yield _Refresh(ctx)
            decision: Decision = yield _Resume(pending, selection, reply, ctx)
            self._account(run, decision)
            result = yield from self._loop(run, first=decision)
            self._results[pending_id] = result
            return result
        finally:
            self._release(name, claim)

    def _account(self, run: _Run, decision: Decision) -> None:
        usage = decision.usage
        u = run.usage
        u.rounds += decision.rounds
        u.jev_calls += usage.jev_calls
        u.jev_input_tokens += usage.jev_input_tokens
        u.llm_calls += usage.llm_calls
        if usage.cost_usd is not None:
            u.cost_usd = (u.cost_usd or 0.0) + usage.cost_usd
            u.est_cost_usd += usage.cost_usd
        else:
            u.est_cost_usd += usage.jev_input_tokens * JEV_USD_PER_INPUT_TOKEN
        run.decisions.append(decision)

    def _guard(self, run: _Run) -> tuple[str, str] | None:
        """``(rule, reason)`` of the first exhausted cap, checked before a new step."""
        b, u = self.budget, run.usage
        if run.step >= b.max_steps:
            return RULE_MAX_STEPS, "budget"
        if u.rounds >= b.max_rounds:
            return RULE_MAX_ROUNDS, "budget"
        if u.est_cost_usd >= b.max_cost_usd:
            return RULE_MAX_COST, "budget"
        has_llm = any(getattr(self.router, a, None) is not None for a in ("filler", "escalator", "text_llm"))
        if has_llm and u.llm_calls >= b.max_llm_calls:
            return RULE_MAX_LLM_CALLS, "budget"
        return None

    def _tool(self, name: str) -> ToolSpec | None:
        catalog = self.router.catalog
        return catalog.get(name) if name in catalog else None

    def _loop(self, run: _Run, first: Decision | None = None) -> _Flow:
        """Steps until a stop: decide → act → (observe → remember → maybe done) → next step."""
        decision = first
        delayed = first is not None
        while True:
            if decision is None:
                stop = self._guard(run)
                if stop is not None:
                    return self._result(run, Outcome.ESCALATE, *stop)
                run.step += 1
                ctx = self._context(run)
                yield _Refresh(ctx)
                decision = yield _Decide(list(run.messages), ctx)
                self._account(run, decision)
            record = LoopStep(step=max(run.step, 1), decision_id=decision.decision_id, outcome=decision.outcome.value,
                              rule=decision.rule, call=decision.call, rounds=decision.rounds,
                              resumed=delayed)  # fmt: skip
            if decision.outcome is not Outcome.EXECUTE or not decision.tool_calls:
                run.steps.append(record)
                if decision.outcome in (Outcome.CONFIRM, Outcome.CLARIFY) and decision.pending is not None:
                    self._paused[decision.pending.pending_id] = run
                return self._result(run, decision.outcome, decision.rule, _reason(decision),
                                    pending=decision.pending, content=decision.content)  # fmt: skip
            call = decision.tool_calls[0]
            tool = self._tool(call.name)
            digest = _call_hash(call)
            previous = run.outcomes.get(digest)
            if previous == "ok" or call.idempotency_key in self.executed:
                run.steps.append(record.model_copy(update={"note": "repeat of an earlier step"}))
                return self._result(run, Outcome.ESCALATE, RULE_REPEAT, "loop")
            if previous == "error" and not delayed and not _auto_retry(tool):
                run.steps.append(record.model_copy(update={"note": "retry of a failed call needs a confirmation"}))
                return self._result(run, Outcome.ESCALATE, RULE_RETRY_CONFIRM, "retry_needs_confirm")
            if delayed and self.revalidate is not None:
                problems = list(self.revalidate(call, self._context(run)))
                if problems:
                    note = "; ".join(problems)
                    run.steps.append(record.model_copy(update={"note": f"TOCTOU: {note}; re-planned"}))
                    run.notes.append(f"{RULE_REVALIDATE}: {note}")
                    decision, delayed = None, False
                    continue
            step = max((o.step for o in self._context(run).all_observations()), default=0) + 1
            observation, attempts = yield from self._execute(call, tool, step, self._context(run).request)
            run.outcomes[digest] = observation.status
            run.observations.append(observation)
            run.usage.executions += 1
            executed = {"executed": True, "status": observation.status, "attempts": attempts,
                        "observation": observation.step}  # fmt: skip
            run.steps.append(record.model_copy(update=executed))
            if observation.status == "ok":
                self.entities.pin_call(call, tool, decision=decision, turn=run.turn, step=observation.step)
            self.entities.add_observation(observation, turn=run.turn)
            content_key = sha256_of({"tool": observation.tool, "status": observation.status,
                                     "content": jsonable(observation.content)})  # fmt: skip
            run.stale = run.stale + 1 if content_key in run.contents else 0
            run.contents.add(content_key)
            if observation.status == "ok" and loop_done(decision.gates.get("done_after"), self.router.policy):
                return self._result(run, Outcome.DONE, RULE_DONE, "done_after")
            if run.stale >= self.budget.no_progress_steps:
                return self._result(run, Outcome.ESCALATE, RULE_NO_PROGRESS, "no_progress")
            decision, delayed = None, False

    def _execute(
        self, call: ToolCall, tool: ToolSpec | None, step: int, request: str
    ) -> Generator[_Effect, Any, tuple[LoopObservation, int]]:
        """Run a call once (plus automatic retries for read/idempotent tools) and ingest its result; a key that
        already ran returns its observation (0 attempts)."""
        name = f"key:{call.idempotency_key}"
        while True:  # the key is claimed before the executor runs: a concurrent run waits for its observation
            claim, mine = self._claim(name)
            if mine:
                break
            yield _Wait(claim)
        try:
            if call.idempotency_key in self.executed:
                return self.executed[call.idempotency_key], 0
            attempts = 0
            retries = self.budget.max_retries if _auto_retry(tool) else 0
            while True:
                attempts += 1
                result, error = yield _Invoke(call)
                observation = ingest_observation(
                    result, tool if tool is not None else call.name, step, arguments=call.arguments,
                    call_id=call.id, request=request, error=error, preview_chars=self.preview_chars,
                )  # fmt: skip
                if observation.status == "ok" or attempts > retries:
                    break
            self.executed[call.idempotency_key] = observation
            return observation, attempts
        finally:
            self._release(name, claim)

    def _result(
        self,
        run: _Run,
        outcome: Outcome,
        rule: str,
        reason: str | None = None,
        *,
        pending: Pending | None = None,
        content: str | None = None,
    ) -> LoopResult:
        run.usage.steps = run.step
        return LoopResult(outcome=outcome, rule=rule, reason=reason, steps=list(run.steps),
                          decisions=list(run.decisions), observations=list(run.observations), pending=pending,
                          usage=run.usage.model_copy(), entities=self.entities, content=content,
                          notes=list(run.notes))  # fmt: skip


def _auto_retry(tool: ToolSpec | None) -> bool:
    """Automatic retry only for ``read``-tier or ``idempotent`` tools (§6.5)."""
    return tool is not None and (tool.tier is Tier.READ or tool.idempotent)


def _reason(decision: Decision) -> str | None:
    trace = decision.trace
    outcome = getattr(trace, "outcome", None)
    if isinstance(outcome, Mapping):
        reason = outcome.get("reason")
        return str(reason) if reason is not None else None
    return None


__all__ = [
    "ENTITY_TYPES",
    "IDEMPOTENCY_META_KEY",
    "PREVIEW_CHARS",
    "RULE_DONE",
    "RULE_MAX_COST",
    "RULE_MAX_LLM_CALLS",
    "RULE_MAX_ROUNDS",
    "RULE_MAX_STEPS",
    "RULE_NO_PROGRESS",
    "RULE_REPEAT",
    "RULE_RETRY_CONFIRM",
    "RULE_REVALIDATE",
    "Agent",
    "Entity",
    "EntityStore",
    "Executor",
    "Executors",
    "LoopBudget",
    "LoopObservation",
    "LoopResult",
    "LoopStep",
    "LoopUsage",
    "MissingExecutor",
    "ObservationItem",
    "accepts_keyword",
    "chunk_text",
    "ingest_observation",
    "mcp_is_error",
    "select_preview",
    "text_entities",
]
