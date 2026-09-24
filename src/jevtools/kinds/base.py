"""The resolver extension point (spec §4): one resolver per slot kind builds pools, questions and decodings.

A resolver is ``⟨pool → Candidates, questions(pool) → Ballot entries, decode(answers) → D, factor(D) → f⟩``.
Planning and decoding only orchestrate through :class:`Resolver`; everything kind-specific lives in the kind's
module under ``jevtools.kinds``, which registers itself with :func:`register_resolver` on import.

This module also provides the shared building blocks every Choice-based kind needs: default resolution
(:func:`resolve_default`), the ``NOT_STATED``/``NONE_OF_THESE`` sentinel entries (:func:`slot_sentinels`), the
slot/probe questions (:func:`slot_question`, :func:`probe_question`) and the general decoding rules 1–3 of §3.6
(:func:`decode_choice`, :func:`elect`).

Additive extensions (kinds agent; the original contract is unchanged):

- :attr:`ResolveContext.injected`: extra candidates per slot key (FILL output, Escalator values, passthrough
  replies). Resolvers add them to the pool before the allow-list is applied (so I2 still holds).
- :meth:`ResolveContext.get_mentions`: the round's mentions, running the extractors once when the planner did not
  supply ``mentions`` (cached in ``cache["mentions"]``).
- :attr:`SlotResult.probes`: non-factor probe results of a slot (``present`` Noul, ``rev`` mass on v*, ``more``),
  for the policy gates (critical ``require_present``) and the trace.
- :class:`Widenable`: the optional coverage-round protocol of §4.6, implemented by the ``ref`` and ``enum``
  resolvers: ``widen(tool, slot, pool, rc, stage) -> (Pool, questions)``. Stage ``"bucket"`` asks ``bucket.b``
  Choices over ranking pages K+1… plus a hierarchy ``group`` Choice; stage ``"hierarchy"`` asks one Choice over
  the items of the top groups (``rc.widen[slot_key]["groups"]`` carries the group distribution of the previous
  stage). Decoding a widen pool goes through the resolver's usual ``decode``.
- Every resolver also exposes ``normalizer`` (``name@version``, recorded in provenance); late-binding recipes and
  their evaluation live in :mod:`jevtools.kinds.late`.
"""

from __future__ import annotations

import importlib
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import datetime
from typing import TYPE_CHECKING, Any, Literal, Protocol, runtime_checkable

from jevtools import templates
from jevtools.ballot import BallotOption, BallotQuestion, SentinelSpec, slot_qid
from jevtools.candidates import (
    BOTTOM_OF,
    NONE_OF_THESE,
    NOT_STATED,
    Bottom,
    Candidate,
    Channel,
    Pool,
    display_value,
    value_key,
)
from jevtools.context import Context, build_state
from jevtools.policy import Policy
from jevtools.spec.models import SlotSpec, ToolSpec
from jevtools.validate import Limits
from jevtools.wire import Answer, ChoiceAnswer

if TYPE_CHECKING:
    from jevtools.spec.catalog import Catalog

Shape = Literal["ok", "missing", "out_of_pool", "uncovered_text", "flag_band"]
"""Slot shapes routed by policy rule P7 (§3.8.2)."""

LATE_DEFAULT = Bottom.LATE_DEFAULT.value
"""Distribution key holding ``NOT_STATED`` mass whose default is late-bound (``from_account.currency``) until the
orchestrator knows the source slot's value (see :meth:`SlotResult.bind_late_default`)."""

ALTERNATIVES_MAX = 3
ALTERNATIVE_MIN_P = 0.01
SUM_TOLERANCE = 0.05
"""A Choice's probabilities sum to ≈ 1 (§8.2 [V]; rounded or float32 values may exceed 1 slightly). Pooled masses
are clamped to 1 (never renormalized, I3); a total above ``1 + SUM_TOLERANCE`` is no distribution and fails closed."""


# --------------------------------------------------------------------------------------------------------------------
# ResolveContext
# --------------------------------------------------------------------------------------------------------------------


@dataclass
class ResolveContext:
    """Everything a resolver needs for one round.

    - ``ctx``: the host :class:`~jevtools.context.Context` (messages, time, user, sources, observations, entities).
    - ``catalog``: the compiled catalog (for cross-slot lookups such as ``default_from`` sources).
    - ``policy`` / ``limits``: thresholds, pool sizes (``policy.pools``) and wire limits (``limits.label_max``).
    - ``mode``: the round mode (``turn``, ``loop``, ``widen``, ``resume``, ``fill``).
    - ``mentions``: the round's claimed mentions (``jevtools.extract.Mentions``) or ``None`` before extraction.
    - ``state``: the round's Jev state; built from ``ctx`` on first access when not given.
    - ``questions``: at decode time, the Ballot's questions by qid (the decode map). Empty while planning.
    - ``round``: 1-based round number within the decision.
    - ``has_filler``: a Filler is configured (content text falls back to ``fill``).
    - ``preferred``: context-preferred values per slot key ``"<tool id>.<qpath>"`` (e.g. account currencies for a
      catalog shortlist), filled by the planner.
    - ``widen``: per slot key, the widen request of a coverage round (offsets, groups), filled by the planner.
    - ``cache``: per-round scratch space shared between resolvers (retrieval results, parsed readings).
    - ``injected``: extra candidates per slot key (FILL, Escalator, passthrough replies); subject to the allow-list.
    """

    ctx: Context
    catalog: Catalog | None = None
    policy: Policy = field(default_factory=Policy)
    limits: Limits = field(default_factory=Limits)
    mode: str = "turn"
    mentions: Any = None
    state: dict[str, Any] | None = None
    questions: Mapping[str, BallotQuestion] = field(default_factory=dict)
    round: int = 1
    has_filler: bool = False
    preferred: Mapping[str, Sequence[Any]] = field(default_factory=dict)
    widen: Mapping[str, Any] = field(default_factory=dict)
    cache: dict[str, Any] = field(default_factory=dict)
    injected: Mapping[str, Sequence[Candidate]] = field(default_factory=dict)

    @property
    def now(self) -> datetime:
        """The context time (aware)."""
        return self.ctx.current_time()

    @property
    def sources(self) -> Mapping[str, Any]:
        """Registered sources by name."""
        return self.ctx.sources

    def source(self, name: str) -> Any:
        """A registered source (``KeyError`` with a clear message if missing)."""
        try:
            return self.ctx.sources[name]
        except KeyError:
            raise KeyError(f"source {name!r} is not registered in the context") from None

    def get_mentions(self) -> Any:
        """The round's :class:`~jevtools.extract.Mentions`: ``mentions`` if the planner supplied them, else the
        extractors run once over ``ctx`` (with the catalog's enum terms) and the result is cached."""
        if self.mentions is not None:
            return self.mentions
        if "mentions" not in self.cache:
            from jevtools.extract import run_extractors

            self.cache["mentions"] = run_extractors(self.ctx, self.catalog)
        return self.cache["mentions"]

    def get_state(self) -> dict[str, Any]:
        """The round state (built once from ``ctx`` and ``mode`` when not supplied). An extension convenience for
        resolvers registered with :func:`register_resolver` (public API); the built-in resolvers do not read it."""
        if self.state is None:
            self.state = build_state(self.ctx, self.mode)
        return self.state

    @staticmethod
    def slot_key(tool: ToolSpec, slot: SlotSpec) -> str:
        """``"<tool id>.<qpath>"``: the key of ``preferred``/``widen`` and the prefix of the slot's qids."""
        return f"{tool.id}.{slot.qpath}"

    def slot_questions(self, tool: ToolSpec, slot: SlotSpec) -> list[BallotQuestion]:
        """At decode time: this slot's questions from the Ballot, in ballot order."""
        return [q for q in self.questions.values() if q.tool == tool.name and q.path == slot.path]

    def question(self, qid: str) -> BallotQuestion | None:
        """A Ballot question by qid, if present."""
        return self.questions.get(qid)


# --------------------------------------------------------------------------------------------------------------------
# SlotResult
# --------------------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Alternative:
    """A runner-up value of a slot (or of one part of a list slot)."""

    display: str
    p: float
    value: Any = None
    label: str | None = None
    part: str | None = None
    """For composite slots: which part (``m0``, ``item.3``, a record leaf path)."""


@dataclass(frozen=True)
class SlotResult:
    """The decoded outcome of one slot (spec §3.6).

    - ``path``: slot path. ``kind``/``stakes``: copied from the spec.
    - ``dist``: value distribution ``{value_key: p}`` over real values and bottoms (``⊥missing``, ``⊥uncovered``,
      ``⊥excluded``, ``⊥omit``; :data:`LATE_DEFAULT` while a late default is pending). Unnormalized: sentinel
      mass is never renormalized away (I3).
    - ``values``: ``{value_key: value}`` for every real key in ``dist``.
    - ``value``: the elected value (a real value or a :class:`~jevtools.candidates.Bottom`).
    - ``display``/``label``: display form and ballot label of the elected value.
    - ``factor``: the slot's confidence factor (§3.7.1); ``None`` for cosmetic slots and non-factors.
    - ``shape``: ``ok``/``missing``/``out_of_pool``/``uncovered_text``/``flag_band``.
    - ``flags``: consistency flags (``presence_conflict``, ``order_sensitive``, ``no_answer``…).
    - ``alternatives``: runner-ups, most probable first.
    - ``channel``/``prov``/``late``/``attrs``: provenance, late-binding recipe and row attributes of the elected
      value (attrs feed constraints and ``render``; never sent to Jev).
    - ``qids``: the questions this result was decoded from. ``sentinels``: sentinel masses for the trace.
    - ``parts``: sub-results of composite kinds (record leaves, list anchors, union branch).
    - ``normalizer``: ``name@version`` of the normalizer that produced ``value``.
    - ``notes``: decode notes (unknown labels ignored…).
    """

    path: tuple[str, ...]
    kind: str
    stakes: str
    dist: dict[str, float]
    values: dict[str, Any]
    value: Any
    shape: Shape
    factor: float | None
    display: str | None = None
    label: str | None = None
    flags: tuple[str, ...] = ()
    alternatives: tuple[Alternative, ...] = ()
    channel: Channel | None = None
    prov: dict[str, Any] = field(default_factory=dict)
    late: dict[str, Any] | None = None
    attrs: dict[str, Any] = field(default_factory=dict)
    qids: tuple[str, ...] = ()
    sentinels: dict[str, float] = field(default_factory=dict)
    parts: dict[str, SlotResult] = field(default_factory=dict)
    normalizer: str | None = None
    notes: tuple[str, ...] = ()
    entries: dict[str, ValueEntry] = field(default_factory=dict)
    """Per value key: the provenance of the option(s) that decode to it (used for re-election)."""
    probes: dict[str, float] = field(default_factory=dict)
    """Non-factor probe results: ``present`` (Noul), ``rev`` (reverse-order mass on v*), ``more`` (list Noul)."""

    @property
    def p(self) -> float:
        """Mass of the elected value."""
        return self.dist.get(value_key(self.value), 0.0)

    @property
    def is_bottom(self) -> bool:
        """The elected value is a bottom (missing/uncovered/excluded/omit)."""
        return isinstance(self.value, Bottom)

    def mass(self, key: Bottom | str) -> float:
        """Mass on a bottom or a value key."""
        return self.dist.get(key.value if isinstance(key, Bottom) else key, 0.0)

    def top(self, n: int = 3) -> list[tuple[Any, float]]:
        """The ``n`` most probable *real* values as ``(value, p)`` (constrained-MAP candidates, §3.6 rule 4)."""
        real = [(k, p) for k, p in self.dist.items() if k in self.values]
        real.sort(key=lambda kp: -kp[1])
        return [(self.values[k], p) for k, p in real[:n]]

    def bind_late_default(
        self, value: Any, *, out_of_pool: float, display: str | None = None, channel: Channel | None = None
    ) -> SlotResult:
        """Resolve the pending late default: move :data:`LATE_DEFAULT` mass onto ``value`` (pooling with an equal
        real candidate, §3.6 rule 2) and re-elect."""
        if LATE_DEFAULT not in self.dist:
            return self
        dist = dict(self.dist)
        mass = dist.pop(LATE_DEFAULT)
        key = value_key(value)
        dist[key] = dist.get(key, 0.0) + mass
        values = {**self.values, key: value}
        entries = dict(self.entries)
        entries.setdefault(
            key, ValueEntry(display=display or display_value(value), channel=channel, prov={"default": True})
        )
        return elect(
            path=self.path, kind=self.kind, stakes=self.stakes, dist=dist, values=values, entries=entries,
            out_of_pool=out_of_pool, qids=self.qids, sentinels=self.sentinels, notes=self.notes, flags=self.flags,
        )  # fmt: skip

    def with_(self, **changes: Any) -> SlotResult:
        """A copy with fields replaced."""
        return replace(self, **changes)


@dataclass(frozen=True)
class ValueEntry:
    """Provenance of the (first, most probable) option decoding to one value key."""

    display: str
    label: str | None = None
    channel: Channel | None = None
    prov: dict[str, Any] = field(default_factory=dict)
    late: dict[str, Any] | None = None
    attrs: dict[str, Any] = field(default_factory=dict)
    p: float = 0.0


def elect(
    *,
    path: tuple[str, ...],
    kind: str,
    stakes: str,
    dist: dict[str, float],
    values: dict[str, Any],
    entries: Mapping[str, ValueEntry],
    out_of_pool: float,
    qids: Sequence[str] = (),
    sentinels: Mapping[str, float] | None = None,
    notes: Sequence[str] = (),
    flags: Sequence[str] = (),
    order: Sequence[str] | None = None,
    offered: Collection[str] | None = None,
) -> SlotResult:
    """Rules 3 of §3.6 on a pooled distribution: ``v* = argmax D`` (ties: first key in ``order``/insertion order),
    shape (``⊥missing`` → missing, ``⊥uncovered`` elected or ``D(⊥uncovered) ≥ out_of_pool`` → out_of_pool),
    factor ``D(v*)`` (``None`` for cosmetic stakes) and up to 3 alternatives with p ≥ 0.01, taken from the
    ``offered`` keys (default: every real value; a Choice offers only its real options, so a default reached
    through ``NOT_STATED`` alone is no alternative — §13.5 ``duration_minutes``)."""
    keys = list(order) if order is not None else list(dist)
    keys += [k for k in dist if k not in keys]
    if not keys:
        return _no_answer(path, kind, stakes, qids, "empty distribution")
    best = max(keys, key=lambda k: (dist.get(k, 0.0), -keys.index(k)))
    value: Any = values[best] if best in values else Bottom(best)
    shape: Shape = "ok"
    if value is Bottom.MISSING:
        shape = "missing"
    elif value is Bottom.UNCOVERED or dist.get(Bottom.UNCOVERED.value, 0.0) >= out_of_pool:
        shape = "out_of_pool"
    entry = entries.get(best)
    alternatives = tuple(
        Alternative(
            display=entries[k].display if k in entries else display_value(values[k]),
            p=dist[k],
            value=values[k],
            label=entries[k].label if k in entries else None,
        )  # fmt: skip
        for k in sorted(
            (
                k
                for k in values
                if k != best and dist.get(k, 0.0) >= ALTERNATIVE_MIN_P and (offered is None or k in offered)
            ),
            key=lambda k: (-dist[k], keys.index(k)),
        )[:ALTERNATIVES_MAX]
    )
    return SlotResult(
        path=path,
        kind=kind,
        stakes=stakes,
        dist=dist,
        values=values,
        value=value,
        shape=shape,
        factor=None if stakes == "cosmetic" else min(1.0, dist.get(best, 0.0)),
        display=entry.display if entry else display_value(value),
        label=entry.label if entry else None,
        flags=tuple(flags),
        alternatives=alternatives,
        channel=entry.channel if entry else None,
        prov=dict(entry.prov) if entry else {},
        late=entry.late if entry else None,
        attrs=dict(entry.attrs) if entry else {},
        qids=tuple(qids),
        sentinels=dict(sentinels or {}),
        notes=tuple(notes),
        entries=dict(entries),
    )


def _no_answer(
    path: tuple[str, ...], kind: str, stakes: str, qids: Sequence[str], note: str, *, flag: str = "no_answer"
) -> SlotResult:
    """Fail closed (I5): a missing or malformed answer is a missing slot with factor 0."""
    return SlotResult(
        path=path, kind=kind, stakes=stakes, dist={Bottom.MISSING.value: 0.0}, values={}, value=Bottom.MISSING,
        shape="missing", factor=0.0, flags=(flag,), qids=tuple(qids), notes=(note,),
    )  # fmt: skip


# --------------------------------------------------------------------------------------------------------------------
# Defaults and sentinels
# --------------------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class DefaultInfo:
    """What ``NOT_STATED`` decodes to when the slot has a default (spec §3.3.1 "Other signals", §3.6 rule 2)."""

    display: str
    """The ``{default}`` substitution (value plus gloss for context defaults)."""
    value: Any = None
    late: dict[str, Any] | None = None
    """``{"default_from": "<slot>.<attr>"}`` when the value is late-bound to another slot."""
    channel: Channel | None = None
    omit: bool = False
    """A ``null`` default on an optional slot: omitting the argument is equivalent."""


def resolve_default(tool: ToolSpec, slot: SlotSpec, ctx: Context) -> DefaultInfo | None:
    """The slot's default: ``default_from`` (a sibling slot path → late-bound; else a context path), else the schema
    ``default``. A context path that does not resolve falls back to the schema default."""
    if slot.default_from:
        head, _, attr = slot.default_from.partition(".")
        if head in tool.slot_names:
            source = tool.slot(head)
            what = f"the {templates.humanize(attr)} of {source.noun}" if attr else source.noun
            return DefaultInfo(display=what, late={"default_from": slot.default_from}, channel=None)
        try:
            value = ctx.lookup(slot.default_from)
        except KeyError:
            value = None
        else:
            shown = templates.default_display(display_value(value), gloss=templates.context_gloss(slot.default_from))
            return DefaultInfo(display=shown, value=value, channel=Channel.REGISTRY)
    if slot.has_default:
        if slot.default is None and not slot.required:
            return DefaultInfo(display="none", omit=True)
        return DefaultInfo(display=display_value(slot.default), value=slot.default, channel=Channel.AUTHOR)
    return None


def not_stated_spec(slot: SlotSpec, default: DefaultInfo | None, *, probe: bool = False) -> SentinelSpec:
    """The ``NOT_STATED`` entry: → default value / late default, → ``omit`` (optional, no default), → ``missing``."""
    if default is not None and not default.omit:
        text = templates.probe_not_stated_text(default.display) if probe else templates.not_stated_text(default.display)
        return SentinelSpec(
            decodes_to="default", text=text, value=default.value, late=default.late, channel=default.channel,
            display=default.display,
        )  # fmt: skip
    text = templates.NOT_STATED_TEXT
    return SentinelSpec(decodes_to="omit" if not slot.required else "missing", text=text)


def slot_sentinels(slot: SlotSpec, default: DefaultInfo | None) -> dict[str, SentinelSpec]:
    """``NOT_STATED`` + ``NONE_OF_THESE`` of a slot Choice."""
    return {
        NOT_STATED: not_stated_spec(slot, default),
        NONE_OF_THESE: SentinelSpec(decodes_to="uncovered", text=templates.NONE_OF_THESE_TEXT),
    }


def slot_question(
    tool: ToolSpec,
    slot: SlotSpec,
    candidates: Sequence[Candidate],
    default: DefaultInfo | None,
    *,
    family: str = "slot",
    suffix: Sequence[str | int] = (),
    meta: Mapping[str, Any] | None = None,
) -> BallotQuestion:
    """The slot Choice ``T.P`` (``T_SLOT``): options in the given (canonical) order plus both sentinels."""
    ask = templates.slot_ask(slot.noun, slot.ask, kind=slot.kind)
    return BallotQuestion(
        qid=slot_qid(tool.id, slot.qpath, *suffix),
        family=family,
        tool=tool.name,
        path=slot.path,
        kind=slot.kind,
        stakes=slot.stakes,
        primitive="choice",
        instructions=templates.slot_instructions(tool.intent, ask),
        options=[BallotOption.from_candidate(c) for c in candidates],
        sentinels=slot_sentinels(slot, default),
        meta=dict(meta or {}),
    )


def probe_question(tool: ToolSpec, slot: SlotSpec, default: DefaultInfo) -> BallotQuestion:
    """The 2-option coverage probe ``T.P`` (``T_PROBE``) of an empty, defaulted slot (§3.5.3, §4.5)."""
    if default.omit:
        raise ValueError(f"{tool.name}.{slot.key}: a null default means omit; probe only real defaults")
    return BallotQuestion(
        qid=slot_qid(tool.id, slot.qpath),
        family="probe",
        tool=tool.name,
        path=slot.path,
        kind=slot.kind,
        stakes=slot.stakes,
        primitive="choice",
        instructions=templates.probe_instructions(tool.intent, slot.noun),
        sentinels={
            NOT_STATED: not_stated_spec(slot, default, probe=True),
            NONE_OF_THESE: SentinelSpec(decodes_to="uncovered", text=templates.PROBE_NONE_OF_THESE_TEXT),
        },
    )


def unasked_result(slot: SlotSpec, default: DefaultInfo | None, *, note: str = "not asked") -> SlotResult:
    """The result of a slot that got no question: its default (not a factor), omitted when optional, else missing
    (factor 0, fail closed)."""
    path, kind, stakes = slot.path, slot.kind, slot.stakes
    if default is not None and not default.omit and default.late is None:
        key = value_key(default.value)
        entry = ValueEntry(display=default.display, channel=default.channel, prov={"default": True}, p=1.0)
        return SlotResult(
            path=path, kind=kind, stakes=stakes, dist={key: 1.0}, values={key: default.value}, value=default.value,
            shape="ok", factor=None, display=default.display, channel=default.channel, prov={"default": True},
            notes=(note,), entries={key: entry},
        )  # fmt: skip
    if default is not None and default.late is not None:
        return SlotResult(
            path=path, kind=kind, stakes=stakes, dist={LATE_DEFAULT: 1.0}, values={}, value=Bottom.LATE_DEFAULT,
            shape="ok", factor=None, late=default.late, notes=(note,),
            entries={LATE_DEFAULT: ValueEntry(display=default.display, late=default.late, p=1.0)},
        )  # fmt: skip
    if not slot.required:
        omit = Bottom.OMIT.value
        return SlotResult(
            path=path, kind=kind, stakes=stakes, dist={omit: 1.0}, values={}, value=Bottom.OMIT, shape="ok",
            factor=None, notes=(note,),
        )  # fmt: skip
    return _no_answer(path, kind, stakes, (), note, flag="not_asked")


# --------------------------------------------------------------------------------------------------------------------
# Choice decoding (§3.6 rules 1–3)
# --------------------------------------------------------------------------------------------------------------------


def decode_choice(
    slot: SlotSpec,
    question: BallotQuestion,
    answer: Answer | None,
    *,
    out_of_pool: float,
    attrs: Mapping[str, Mapping[str, Any]] | None = None,
) -> SlotResult:
    """Decode a slot/probe Choice: labels → values via the decode map (unknown labels ignored, missing ones 0),
    value pooling, sentinel decodes, election, shape and factor.

    ``attrs`` optionally maps option labels to row attributes (from the pool) so the elected value carries them.
    A missing or non-Choice answer fails closed (shape ``missing``, factor 0, flag ``no_answer``).
    """
    path, kind, stakes, qids = slot.path, slot.kind, slot.stakes, (question.qid,)
    if not isinstance(answer, ChoiceAnswer):
        return _no_answer(path, kind, stakes, qids, "no choice answer")
    probabilities = answer.probabilities
    sent = set(question.labels)
    notes = [f"ignored unknown label {label!r}" for label in probabilities if label not in sent]
    total = sum(float(p) for label, p in probabilities.items() if label in sent)
    if total > 1.0 + SUM_TOLERANCE:
        return _no_answer(path, kind, stakes, qids, f"probabilities sum to {total:.4f}, not a distribution")
    dist: dict[str, float] = {}
    values: dict[str, Any] = {}
    entries: dict[str, ValueEntry] = {}
    order: list[str] = []
    sentinel_mass: dict[str, float] = {}

    def add(key: str, p: float, entry: ValueEntry | None) -> None:
        if key not in dist:
            order.append(key)
        # Labels decoding to one value pool (§3.6 rule 2); rounding may push the pooled mass a hair above 1
        # (``{medium: 0.6667, NOT_STATED: 0.3334}`` with default medium): clamp, never renormalize.
        dist[key] = min(1.0, dist.get(key, 0.0) + p)
        if entry is not None and (key not in entries or p > entries[key].p):
            entries[key] = entry

    for option in question.options:
        p = float(probabilities.get(option.label, 0.0))
        key = value_key(option.value)
        values[key] = option.value
        add(key, p, ValueEntry(
            display=display_value(option.value), label=option.label, channel=option.channel, prov=option.prov,
            late=option.late, attrs=dict((attrs or {}).get(option.label, {})), p=p,
        ))  # fmt: skip
    for label, spec in question.sentinels.items():
        p = float(probabilities.get(label, 0.0))
        sentinel_mass[label] = p
        key, value, entry = _sentinel_value(spec, label, p)
        if value is not None:
            values[key] = value
        add(key, p, entry)
    return elect(
        path=path, kind=kind, stakes=stakes, dist=dist, values=values, entries=entries, out_of_pool=out_of_pool,
        qids=qids, sentinels=sentinel_mass, notes=notes, order=order,
        offered={value_key(option.value) for option in question.options},
    )  # fmt: skip


def _sentinel_value(spec: SentinelSpec, label: str, p: float) -> tuple[str, Any, ValueEntry | None]:
    if spec.decodes_to == "default":
        if spec.late is not None:
            entry = ValueEntry(display=spec.display or "default", label=label, late=spec.late, p=p)
            return LATE_DEFAULT, None, entry
        entry = ValueEntry(
            display=spec.display or display_value(spec.value), label=label, channel=spec.channel,
            prov={"default": True}, p=p,
        )  # fmt: skip
        return value_key(spec.value), spec.value, entry
    bottom = BOTTOM_OF.get(spec.decodes_to)
    if bottom is None:
        raise ValueError(f"sentinel {label} decodes to {spec.decodes_to!r}, which is not a slot decode")
    return bottom.value, None, ValueEntry(display=bottom.value, label=label, p=p)


# --------------------------------------------------------------------------------------------------------------------
# The Resolver protocol and registry
# --------------------------------------------------------------------------------------------------------------------


@runtime_checkable
class Resolver(Protocol):
    """A slot-kind resolver. Implementations are stateless; per-round data lives in :class:`ResolveContext`.

    - ``pool``: enumerate candidates (code only), apply the slot's channel allow-list (I2, blocked ones go to
      ``Pool.blocked``), drop schema-invalid values, assign labels and canonical order.
    - ``questions``: the slot's question families (§3.5.3) in family order; qids from ``T.<qpath>…``.
    - ``decode``: turn the answers (keyed by qid; the questions are in ``rc.questions``) into a :class:`SlotResult`.
    """

    kind: str

    def pool(self, tool: ToolSpec, slot: SlotSpec, rc: ResolveContext) -> Pool: ...

    def questions(self, tool: ToolSpec, slot: SlotSpec, pool: Pool, rc: ResolveContext) -> list[BallotQuestion]: ...

    def decode(
        self, tool: ToolSpec, slot: SlotSpec, pool: Pool, answers: Mapping[str, Answer], rc: ResolveContext
    ) -> SlotResult: ...


@runtime_checkable
class Widenable(Protocol):
    """Optional coverage-round protocol (spec §4.6 steps 2–3), implemented by ``ref`` and catalog ``enum``.

    ``stage`` is ``"bucket"`` (``bucket.b`` Choices over ranking pages beyond K, plus a hierarchy ``group`` Choice
    when the source has a hierarchy) or ``"hierarchy"`` (one Choice over the items of the top groups; the previous
    group distribution comes in ``rc.widen[slot_key]["groups"]``). Returns the widened pool and its questions;
    an empty question list means the strategy is exhausted.
    """

    def widen(
        self, tool: ToolSpec, slot: SlotSpec, pool: Pool, rc: ResolveContext, stage: str
    ) -> tuple[Pool, list[BallotQuestion]]: ...


RESOLVERS: dict[str, Resolver] = {}
"""Registered resolvers by kind."""

KIND_MODULES: dict[str, str] = {
    "enum": "enum", "flag": "flag", "ordinal": "ordinal", "quantity": "quantity", "money": "money",
    "temporal": "temporal", "span": "span", "ref": "ref", "list": "listing", "record": "record",
    "union": "record", "text": "text", "derived": "derived", "secret": "derived",
}  # fmt: skip
"""Module under ``jevtools.kinds`` that registers each kind (imported lazily by :func:`get_resolver`)."""


def register_resolver(kind: str, resolver: Resolver) -> None:
    """Register ``resolver`` for ``kind`` (replacing any earlier one)."""
    RESOLVERS[kind] = resolver


def get_resolver(kind: str) -> Resolver:
    """The resolver of ``kind``, importing ``jevtools.kinds.<module>`` on first use."""
    if kind not in RESOLVERS and kind in KIND_MODULES:
        importlib.import_module(f"jevtools.kinds.{KIND_MODULES[kind]}")
    try:
        return RESOLVERS[kind]
    except KeyError:
        raise KeyError(f"no resolver registered for kind {kind!r}") from None


__all__ = [
    "ALTERNATIVES_MAX",
    "KIND_MODULES",
    "LATE_DEFAULT",
    "RESOLVERS",
    "SUM_TOLERANCE",
    "Alternative",
    "DefaultInfo",
    "ResolveContext",
    "Resolver",
    "Shape",
    "SlotResult",
    "ValueEntry",
    "Widenable",
    "decode_choice",
    "elect",
    "get_resolver",
    "not_stated_spec",
    "probe_question",
    "register_resolver",
    "resolve_default",
    "slot_question",
    "slot_sentinels",
    "unasked_result",
]
