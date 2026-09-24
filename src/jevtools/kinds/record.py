"""The ``record`` and ``union`` resolvers (spec §4.2.10, ext.records).

- **record**: a nested object flattened to dotted slot paths (depth ≤ 3). Each leaf is resolved by its own kind's
  resolver (qids ``T.<parent>.<leaf>``); decoding reassembles the object (omitted leaves are left out) and the
  factor is the product of the leaf factors.
- **union** (``oneOf``/``anyOf`` of objects): a ``branch`` Choice ``T.P.branch`` whose options are the branch
  titles (descriptions from the branch ``description``), plus the leaves of every *viable* branch, asked
  speculatively. Only the elected branch's leaves are decoded; the factor includes ``P(branch)``.

A composite pool keeps its leaf pools in ``meta["children"]`` (by leaf key) and reports
``meta["viability"]`` (``ok``, ``empty:<leaf>`` or ``channel_blocked:<leaf>``) for the planner.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from jevtools import templates
from jevtools.ballot import BallotOption, BallotQuestion, SentinelSpec, slot_qid
from jevtools.candidates import NONE_OF_THESE, Bottom, Candidate, Channel, Pool, assign_labels, least_trusted, value_key
from jevtools.kinds.base import (
    ResolveContext,
    SlotResult,
    ValueEntry,
    decode_choice,
    get_resolver,
    register_resolver,
    resolve_default,
)
from jevtools.kinds.listing import worst_shape
from jevtools.spec.models import SlotSpec, ToolSpec
from jevtools.wire import Answer


def leaf_viability(tool: ToolSpec, slot: SlotSpec, pool: Pool, rc: ResolveContext) -> str:
    """``ok`` / ``empty`` / ``channel_blocked`` for one slot (spec §5.1 filled-able rule)."""
    nested = pool.meta.get("viability")
    if isinstance(nested, str):
        return nested.split(":", 1)[0]
    if pool.evidence_backed or pool.closed or not slot.required:
        return "ok"
    if resolve_default(tool, slot, rc.ctx) is not None:
        return "ok"
    return "channel_blocked" if pool.channel_blocked else "empty"


def children_pools(tool: ToolSpec, children: Sequence[SlotSpec], rc: ResolveContext) -> dict[str, Pool]:
    """Each leaf's pool, by leaf key."""
    return {child.key: get_resolver(child.kind).pool(tool, child, rc) for child in children}


def viability(tool: ToolSpec, children: Sequence[SlotSpec], pools: Mapping[str, Pool], rc: ResolveContext) -> str:
    """The first non-``ok`` required leaf (``empty:<leaf>``), else ``ok``."""
    for child in children:
        state = leaf_viability(tool, child, pools[child.key], rc)
        if state != "ok":
            return f"{state}:{child.key}"
    return "ok"


def composite_pool(
    tool: ToolSpec,
    slot: SlotSpec,
    kind: str,
    pools: Mapping[str, Pool],
    state: str,
    meta: Mapping[str, Any] | None = None,
) -> Pool:
    """A pool summarizing leaf pools (candidates are the leaves' own; the leaves keep their labels)."""
    return Pool(
        tool=tool.name,
        path=slot.path,
        kind=kind,
        candidates=[c for p in pools.values() for c in p.candidates],
        closed=all(p.closed for p in pools.values()) if pools else False,
        evidence_backed=any(p.evidence_backed for p in pools.values()),
        blocked=[c for p in pools.values() for c in p.blocked],
        meta={"children": dict(pools), "viability": state, **dict(meta or {})},
    )


def decode_children(
    tool: ToolSpec,
    children: Sequence[SlotSpec],
    pools: Mapping[str, Pool],
    answers: Mapping[str, Answer],
    rc: ResolveContext,
) -> tuple[dict[str, Any], dict[str, SlotResult]]:
    """Decode every leaf; the object holds the non-bottom leaf values under their names."""
    obj: dict[str, Any] = {}
    parts: dict[str, SlotResult] = {}
    for child in children:
        result = get_resolver(child.kind).decode(tool, child, pools[child.key], answers, rc)
        parts[child.key] = result
        if not result.is_bottom:
            obj[child.name] = result.value
    return obj, parts


def assemble(
    slot: SlotSpec,
    obj: dict[str, Any],
    parts: Mapping[str, SlotResult],
    *,
    extra: float = 1.0,
    normalizer: str,
    qids: Sequence[str] = (),
) -> SlotResult:
    """The composite result: factor = extra × ∏ leaf factors, shape = the worst leaf shape."""
    factors = [r.factor for r in parts.values() if r.factor is not None]
    factor = extra * math.prod(factors)
    channels: list[Channel] = [r.channel for r in parts.values() if r.channel is not None]
    value: Any = obj if obj or slot.required else Bottom.OMIT
    key = value_key(value)
    entry = ValueEntry(display=str(obj), channel=least_trusted(*channels) if channels else None, p=factor)
    return SlotResult(
        path=slot.path,
        kind=slot.kind,
        stakes=slot.stakes,
        dist={key: factor},
        values={} if value is Bottom.OMIT else {key: value},
        value=value,
        shape=worst_shape([r.shape for r in parts.values()]),
        factor=factor,
        channel=entry.channel,
        flags=tuple(f for r in parts.values() for f in r.flags),
        parts=dict(parts),
        normalizer=normalizer,
        entries={key: entry},
        qids=tuple(qids) + tuple(q for r in parts.values() for q in r.qids),
    )


class RecordResolver:
    """Resolver for ``kind: record`` (flattened nested objects)."""

    kind = "record"
    normalizer = "record@1"

    def pool(self, tool: ToolSpec, slot: SlotSpec, rc: ResolveContext) -> Pool:
        pools = children_pools(tool, slot.children, rc)
        return composite_pool(tool, slot, self.kind, pools, viability(tool, slot.children, pools, rc))

    def questions(self, tool: ToolSpec, slot: SlotSpec, pool: Pool, rc: ResolveContext) -> list[BallotQuestion]:
        pools: Mapping[str, Pool] = pool.meta["children"]
        return [
            q for child in slot.children for q in get_resolver(child.kind).questions(tool, child, pools[child.key], rc)
        ]

    def decode(
        self, tool: ToolSpec, slot: SlotSpec, pool: Pool, answers: Mapping[str, Answer], rc: ResolveContext
    ) -> SlotResult:
        obj, parts = decode_children(tool, slot.children, pool.meta["children"], answers, rc)
        return assemble(slot, obj, parts, normalizer=self.normalizer)


class UnionResolver:
    """Resolver for ``kind: union`` (branch Choice + speculative leaves of viable branches)."""

    kind = "union"
    normalizer = "record@1"

    def pool(self, tool: ToolSpec, slot: SlotSpec, rc: ResolveContext) -> Pool:
        branches: dict[str, dict[str, Pool]] = {}
        states: dict[str, str] = {}
        for branch in slot.branches:
            pools = children_pools(tool, branch.children, rc)
            branches[branch.qpath] = pools
            states[branch.qpath] = viability(tool, branch.children, pools, rc)
        viable = [b for b in slot.branches if states[b.qpath] == "ok"]
        state = "ok" if viable else next(iter(states.values()), "empty:" + slot.key)
        flat = {f"{b.qpath}/{k}": p for b in slot.branches for k, p in branches[b.qpath].items()}
        return composite_pool(
            tool, slot, self.kind, flat, state, meta={"branches": branches, "branch_viability": states}
        )

    def branch_question(self, tool: ToolSpec, slot: SlotSpec) -> BallotQuestion:
        """``T.P.branch``: which branch describes the value (options = branch titles)."""
        branches = [
            Candidate(label=b.title or b.name, value=i, text=b.description, channel=Channel.AUTHOR, prov={"branch": i})
            for i, b in enumerate(slot.branches)
        ]
        options = [BallotOption.from_candidate(c) for c in assign_labels(branches, slot=slot.name)]
        return BallotQuestion(
            qid=slot_qid(tool.id, slot.qpath, "branch"),
            family="branch",
            tool=tool.name,
            path=slot.path,
            kind=slot.kind,
            stakes=slot.stakes,
            primitive="choice",
            instructions=templates.branch_instructions(tool.intent, slot.noun),
            options=options,
            sentinels={NONE_OF_THESE: SentinelSpec(decodes_to="uncovered", text=templates.NONE_OF_THESE_TEXT)},
        )

    def questions(self, tool: ToolSpec, slot: SlotSpec, pool: Pool, rc: ResolveContext) -> list[BallotQuestion]:
        branches: Mapping[str, Mapping[str, Pool]] = pool.meta["branches"]
        states: Mapping[str, str] = pool.meta["branch_viability"]
        out = [self.branch_question(tool, slot)]
        for branch in slot.branches:
            if states[branch.qpath] != "ok":
                continue
            out += [
                q
                for child in branch.children
                for q in get_resolver(child.kind).questions(tool, child, branches[branch.qpath][child.key], rc)
            ]
        return out

    def decode(
        self, tool: ToolSpec, slot: SlotSpec, pool: Pool, answers: Mapping[str, Answer], rc: ResolveContext
    ) -> SlotResult:
        question = rc.question(slot_qid(tool.id, slot.qpath, "branch")) or self.branch_question(tool, slot)
        choice = decode_choice(slot, question, answers.get(question.qid), out_of_pool=rc.policy.shapes.out_of_pool)
        if choice.is_bottom:
            return choice.with_(normalizer=self.normalizer)
        branch = slot.branches[int(choice.value)]
        if pool.meta["branch_viability"][branch.qpath] != "ok":
            state = pool.meta["branch_viability"][branch.qpath]
            return choice.with_(
                shape="missing", notes=(f"branch {branch.name!r} is not viable ({state})",), normalizer=self.normalizer
            )
        obj, parts = decode_children(tool, branch.children, pool.meta["branches"][branch.qpath], answers, rc)
        parts = {"branch": choice, **parts}
        result = assemble(
            slot,
            obj,
            {k: v for k, v in parts.items() if k != "branch"},
            extra=choice.p,
            normalizer=self.normalizer,
            qids=choice.qids,
        )
        return result.with_(parts=parts, notes=(f"branch {branch.title or branch.name}",))


register_resolver("record", RecordResolver())
register_resolver("union", UnionResolver())

__all__ = [
    "RecordResolver",
    "UnionResolver",
    "assemble",
    "children_pools",
    "composite_pool",
    "decode_children",
    "leaf_viability",
    "viability",
]
