"""Shared machinery of the Choice-based resolvers (quantity, money, span, temporal, ref, flag).

:func:`finalize_pool` is the pool pipeline every resolver runs, in this order:

1. add injected candidates (FILL / Escalator / passthrough, :attr:`ResolveContext.injected`);
   ``history`` candidates get the origin their trust is inherited from (:func:`trace_history`);
2. dedupe by normalized value (the most trusted channel wins, a ``history`` copy counting at its origin's trust);
3. drop schema-invalid values (late-bound candidates are validated after late binding);
4. drop values violating a unary constraint (``start > now``);
5. apply the slot's channel allow-list (I2) — blocked candidates are kept in ``Pool.blocked``;
6. assign WYSIWYG labels and canonical order.

:class:`ChoiceResolver` implements ``questions``/``decode`` for a single slot Choice (or the coverage probe of an
empty defaulted slot); subclasses only enumerate candidates.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from jevtools.ballot import BallotQuestion
from jevtools.candidates import (
    Candidate,
    Channel,
    Pool,
    apply_allow_list,
    assign_labels,
    canonical_order,
    display_value,
    least_trusted,
    truncate,
    value_key,
)
from jevtools.extract.base import Mention, Mentions
from jevtools.extract.coref import iter_entities
from jevtools.extract.tokens import fold
from jevtools.kinds.base import (
    ResolveContext,
    SlotResult,
    decode_choice,
    probe_question,
    resolve_default,
    slot_question,
    unasked_result,
)
from jevtools.spec.constraints import Constraint, ConstraintContext
from jevtools.spec.models import SlotSpec, ToolSpec
from jevtools.spec.schema import is_valid
from jevtools.templates import NEGATION_NOTE, OBSERVATION_TEXT
from jevtools.wire import Answer

UNTRUSTED_LABEL_MAX = 64


def mention_text(m: Mention, *, note: str | None = None) -> str:
    """Self-contained option description of a mention-derived candidate.

    ``From "45 min" in the request.``; earlier user turns and assistant turns say so; observation values use the
    normative ``Found in observation k: "…"``. Negated mentions add ``mentioned in a negation: '…'``.
    """
    if m.channel is Channel.TOOL_OUTPUT:
        head = OBSERVATION_TEXT.format(step=m.step, quote=truncate(m.text, UNTRUSTED_LABEL_MAX))
    elif m.in_request:
        head = f'From "{m.text}" in the request'
    elif m.channel is Channel.USER:
        head = f'From "{m.text}" in an earlier user turn'
    else:
        head = f'From "{m.text}" in an assistant turn in `history`'
    parts = [head]
    if note:
        parts.append(note)
    if m.negated:
        parts.append(NEGATION_NOTE.format(quote=m.attrs.get("negation", m.text)))
    text = "; ".join(parts)
    return text if m.channel is Channel.TOOL_OUTPUT and len(parts) == 1 else text + "."


POOL_CHANNELS: frozenset[Channel] = frozenset({Channel.USER, Channel.TOOL_OUTPUT})
"""Channels whose mentions enter pools directly. Assistant-turn (``history``) mentions enter only when the request
is anaphoric ("book it then"); otherwise history reaches pools through the entity store (§4.2.1, §6.4)."""


def pool_mentions(rc: ResolveContext, *kinds: str) -> list[Mention]:
    """Free mentions of the given kinds that may enter a pool: user and observation mentions, plus assistant-turn
    mentions when the request carries an anaphor cue."""
    mentions: Mentions = rc.get_mentions()
    channels = set(POOL_CHANNELS)
    if mentions.cues("anaphor") and any(m.in_request for m in mentions.cues("anaphor")):
        channels.add(Channel.HISTORY)
    return mentions.of(*kinds, channels=channels, free=True)


def mention_prov(m: Mention, **extra: Any) -> dict[str, Any]:
    """Provenance of a mention-derived candidate: ``{"extractor", "mention", …}``."""
    prov: dict[str, Any] = {"extractor": m.extractor, "mention": m.prov()}
    prov.update({k: v for k, v in extra.items() if v is not None})
    return prov


def mention_candidate(
    m: Mention, value: Any, *, display: str | None = None, note: str | None = None, **prov: Any
) -> Candidate:
    """A candidate whose value comes from a mention (channel = the mention's channel)."""
    shown = display
    if m.channel is Channel.TOOL_OUTPUT and shown is not None:
        shown = truncate(shown, UNTRUSTED_LABEL_MAX)
    return Candidate(
        value=value, display=shown, text=mention_text(m, note=note), channel=m.channel, prov=mention_prov(m, **prov)
    )


def unary_constraints(tool: ToolSpec, slot: SlotSpec) -> list[Constraint]:
    """Tool constraints that involve only this (top-level) slot and ``now``/literals: applied at pool time."""
    if len(slot.path) != 1:
        return []
    return [c for c in tool.constraints if c.unary and c.slots == frozenset({slot.name})]


def violates(constraints: Sequence[Constraint], slot: SlotSpec, value: Any, rc: ResolveContext) -> bool:
    """Whether a value definitely violates a unary constraint (unevaluable constraints do not drop values)."""
    context = ConstraintContext(now=rc.now, context=rc.ctx)
    return any(c.check({slot.name: value}, context) is False for c in constraints)


UNTRUSTED_ORIGINS: frozenset[Channel] = frozenset({Channel.TOOL_OUTPUT, Channel.GENERATED})


def history_origin(text: str, value: Any, rc: ResolveContext) -> Channel:
    """Where a value named in an assistant turn came from (§3.4.2: a ``history`` value inherits its origin's trust;
    §6.6, §11.2 E10: a value planted in tool output or history never reaches an external identity slot).

    An observation that contains it (or a ``tool_output`` entity) → ``tool_output``; else the user's own turns →
    ``user``; a registry row keyed by it or a trusted entity → that channel; untraceable assistant text is
    untrusted (``tool_output``): an assistant turn may repeat what a tool output planted.
    """
    needle = fold(text).strip()
    texts = list(rc.get_mentions().texts.values())
    if needle and any(needle in fold(t.text) for t in texts if t.channel is Channel.TOOL_OUTPUT):
        return Channel.TOOL_OUTPUT
    key = value_key(value)
    origins = [e.origin for e in iter_entities(rc.ctx.entities) if value_key(e.value) == key]
    if any(o in UNTRUSTED_ORIGINS for o in origins):
        return Channel.TOOL_OUTPUT
    if needle and any(needle in fold(t.text) for t in texts if t.channel is Channel.USER):
        return Channel.USER
    for source in rc.sources.values():
        rows, field = getattr(source, "rows", None), getattr(source, "key", None)
        if isinstance(field, str) and isinstance(rows, list):
            if any(isinstance(r, Mapping) and value_key(r.get(field)) == key for r in rows):
                return Channel.REGISTRY
    trusted = [o for o in origins if o in (Channel.USER, Channel.REGISTRY, Channel.AUTHOR)]
    return trusted[0] if trusted else Channel.TOOL_OUTPUT


def trace_history(candidates: Sequence[Candidate], rc: ResolveContext) -> list[Candidate]:
    """Give every ``history`` candidate the origin its trust is inherited from: an assistant-turn mention (no
    origin) is traced with :func:`history_origin`; a history copy of a value that is also in the pool from an
    untrusted channel (or an untrusted entity) takes that least-trusted provenance, so an assistant turn that
    echoes a tool output never launders it into a trusted ``history`` value."""
    # Seed with *known* untrusted provenance only: an untraced history copy (origin unknown) is what gets traced here.
    untrusted = {value_key(c.value) for c in candidates
                 if (c.origin if c.channel is Channel.HISTORY else c.channel) in UNTRUSTED_ORIGINS}  # fmt: skip
    untrusted |= {value_key(e.value) for e in iter_entities(rc.ctx.entities) if e.origin in UNTRUSTED_ORIGINS}
    out: list[Candidate] = []
    for c in candidates:
        if c.channel is Channel.HISTORY:
            origin = c.origin
            if origin is None:
                mention = c.prov.get("mention")
                text = str(mention.get("text")) if isinstance(mention, Mapping) else display_value(c.value)
                origin = history_origin(text, c.value, rc)
            if value_key(c.value) in untrusted:
                origin = least_trusted(origin, Channel.TOOL_OUTPUT)
            if origin != c.origin:
                c = c.model_copy(update={"origin": origin})
        out.append(c)
    return out


def dedupe(candidates: Iterable[Candidate]) -> list[Candidate]:
    """One candidate per normalized value; the most trusted (effective) channel wins, then the first seen. A
    ``history`` copy counts at its origin's trust (:func:`trace_history` runs first in :func:`finalize_pool`)."""
    best: dict[str, Candidate] = {}
    order: list[str] = []
    for candidate in candidates:
        key = value_key(candidate.value)
        if key not in best:
            best[key] = candidate
            order.append(key)
        elif candidate.effective_channel.trust < best[key].effective_channel.trust:
            best[key] = candidate
    return [best[k] for k in order]


def finalize_pool(
    tool: ToolSpec,
    slot: SlotSpec,
    rc: ResolveContext,
    candidates: Iterable[Candidate],
    kind: str,
    *,
    closed: bool = False,
    is_path: bool = False,
    validate: bool = True,
    order: bool = True,
    limit: int | None = None,
    notes: Iterable[str] = (),
    meta: Mapping[str, Any] | None = None,
) -> Pool:
    """Run the pool pipeline (module docstring) and build the :class:`Pool`."""
    notes = list(notes)
    pooled = dedupe(trace_history([*candidates, *rc.injected.get(rc.slot_key(tool, slot), ())], rc))
    if validate:
        valid = [c for c in pooled if c.late is not None or is_valid(c.value, slot.json_schema)]
        if len(valid) < len(pooled):
            notes.append(f"dropped {len(pooled) - len(valid)} schema-invalid value(s)")
        pooled = valid
    constraints = unary_constraints(tool, slot)
    if constraints:
        kept = [c for c in pooled if c.late is not None or not violates(constraints, slot, c.value, rc)]
        if len(kept) < len(pooled):
            notes.append(
                f"dropped {len(pooled) - len(kept)} value(s) violating {', '.join(c.expr for c in constraints)}"
            )
        pooled = kept
    admitted, blocked = apply_allow_list(pooled, slot.channels)
    cap = min(limit or rc.limits.max_real_options, rc.limits.max_real_options)
    if len(admitted) > cap:
        notes.append(f"cut {len(admitted)} candidates to {cap}")
        admitted = admitted[:cap]
    labelled = assign_labels(admitted, slot=slot.name, label_max=rc.limits.label_max, is_path=is_path)
    return Pool(
        tool=tool.name,
        path=slot.path,
        kind=kind,
        candidates=canonical_order(labelled) if order else labelled,
        closed=closed,
        evidence_backed=any(c.is_evidence for c in labelled),
        blocked=blocked,
        notes=notes,
        meta=dict(meta or {}),
    )


class ChoiceResolver(ABC):
    """A resolver whose slot is one Choice over its pool (plus ``NOT_STATED``/``NONE_OF_THESE``)."""

    kind: str = ""
    normalizer: str = "string@1"
    is_path: bool = False
    closed: bool = False

    @abstractmethod
    def candidates(self, tool: ToolSpec, slot: SlotSpec, rc: ResolveContext) -> list[Candidate]:
        """Enumerate raw candidates (normalized values, before the pool pipeline)."""

    def normalizer_for(self, slot: SlotSpec) -> str:
        """``name@version`` of the normalizer that produced this slot's values."""
        return self.normalizer

    def pool(self, tool: ToolSpec, slot: SlotSpec, rc: ResolveContext) -> Pool:
        return finalize_pool(
            tool, slot, rc, self.candidates(tool, slot, rc), self.kind, closed=self.closed, is_path=self.is_path
        )

    def questions(self, tool: ToolSpec, slot: SlotSpec, pool: Pool, rc: ResolveContext) -> list[BallotQuestion]:
        """The slot Choice, or the coverage probe when the pool is empty and the slot has a default."""
        default = resolve_default(tool, slot, rc.ctx)
        if pool.candidates:
            return [slot_question(tool, slot, pool.candidates, default)]
        if default is not None and not default.omit:
            return [probe_question(tool, slot, default)]
        return []

    def decode(
        self, tool: ToolSpec, slot: SlotSpec, pool: Pool, answers: Mapping[str, Answer], rc: ResolveContext
    ) -> SlotResult:
        """§3.6 rules 1–3 on the slot Choice (unasked slots fall back to their default / omit / missing)."""
        questions = rc.slot_questions(tool, slot) or self.questions(tool, slot, pool, rc)
        question = next((q for q in questions if q.family in ("slot", "probe")), None)
        normalizer = self.normalizer_for(slot)
        if question is None:
            return unasked_result(slot, resolve_default(tool, slot, rc.ctx)).with_(normalizer=normalizer)
        attrs = {c.label: c.attrs for c in pool.candidates}
        result = decode_choice(
            slot, question, answers.get(question.qid), out_of_pool=rc.policy.shapes.out_of_pool, attrs=attrs
        )
        return result.with_(normalizer=normalizer)


__all__ = [
    "POOL_CHANNELS",
    "ChoiceResolver",
    "dedupe",
    "finalize_pool",
    "mention_candidate",
    "mention_prov",
    "mention_text",
    "pool_mentions",
    "unary_constraints",
    "violates",
]
