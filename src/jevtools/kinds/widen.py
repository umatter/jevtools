"""Coverage rounds for ranked pools (spec §4.6 steps 2–3), shared by the ``ref`` and catalog ``enum`` resolvers.

- **Bucket stage.** ``T.P.bucket.b`` Choices over the ranking beyond the first K (B ≤ ``policy.widen.buckets``
  buckets of ≤ ``policy.widen.page`` items, each with ``NONE_OF_THESE``), plus a ``T.P.group`` Choice over the
  hierarchy groups of the whole ranking when the source has a hierarchy.
- **Hierarchy stage.** One Choice over the items inside the top groups of the previous group answer (cumulative
  mass ≥ 0.9, at most 3 groups, at most 252 items).

Bucket decoding is honest (I3): the elected value's factor is ``D_b(v*) · ∏_{b'≠b} D_b'(⊥uncovered)`` — the other
buckets must agree that the value is not among theirs — and nothing is renormalized.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from jevtools.ballot import BallotQuestion, SentinelSpec
from jevtools.candidates import NONE_OF_THESE, Bottom, Candidate, Channel, Pool, value_key
from jevtools.kinds.base import (
    ResolveContext,
    SlotResult,
    ValueEntry,
    decode_choice,
    elect,
    resolve_default,
    slot_question,
)
from jevtools.kinds.common import finalize_pool
from jevtools.spec.models import SlotSpec, ToolSpec
from jevtools.templates import NONE_OF_THESE_TEXT
from jevtools.wire import Answer

BUCKET = "bucket"
HIERARCHY = "hierarchy"
GROUP_MASS = 0.9
MAX_GROUPS = 3

GroupOf = Callable[[Candidate], "str | None"]


def _uncovered_only() -> dict[str, SentinelSpec]:
    return {NONE_OF_THESE: SentinelSpec(decodes_to="uncovered", text=NONE_OF_THESE_TEXT)}


def bucket_stage(
    tool: ToolSpec,
    slot: SlotSpec,
    rc: ResolveContext,
    ranked: Sequence[Candidate],
    *,
    skip: int,
    kind: str,
    group_of: GroupOf | None = None,
    is_path: bool = False,
) -> tuple[Pool, list[BallotQuestion]]:
    """Bucket Choices over ``ranked[skip : skip + page·B]`` plus a hierarchy group Choice."""
    page = min(rc.policy.widen.page, rc.limits.max_real_options)
    beyond = list(ranked[skip : skip + page * rc.policy.widen.buckets])
    questions: list[BallotQuestion] = []
    buckets: list[list[Candidate]] = []
    blocked: list[Candidate] = []
    for b in range(rc.policy.widen.buckets):
        chunk = beyond[b * page : (b + 1) * page]
        if not chunk:
            break
        bucket = finalize_pool(tool, slot, rc, chunk, kind, is_path=is_path)
        blocked += bucket.blocked
        if not bucket.candidates:
            continue
        question = slot_question(
            tool,
            slot,
            bucket.candidates,
            None,
            family="bucket",
            suffix=("bucket", len(buckets)),
            meta={"widen": BUCKET, "bucket": len(buckets)},
        )
        questions.append(question.model_copy(update={"sentinels": _uncovered_only()}))
        buckets.append(bucket.candidates)
    groups = group_candidates(ranked, group_of) if group_of is not None else []
    if len(groups) >= 1:
        group_slot = slot.model_copy(update={"noun": f"the group that contains {slot.noun}", "ask": None})
        group_pool = finalize_pool(tool, group_slot, rc, groups, kind, validate=False)
        question = slot_question(
            tool, group_slot, group_pool.candidates, None, family="group", suffix=("group",), meta={"widen": "group"}
        )
        questions.append(question.model_copy(update={"sentinels": _uncovered_only()}))
    pool = Pool(
        tool=tool.name,
        path=slot.path,
        kind=kind,
        candidates=[c for bucket in buckets for c in bucket],
        evidence_backed=True,
        blocked=blocked,
        notes=[f"widen: {len(buckets)} bucket(s) beyond {skip}"],
        meta={"mode": "widen_bucket", "buckets": len(buckets), "groups": [g.value for g in groups]},
    )
    return pool, questions


def group_candidates(ranked: Sequence[Candidate], group_of: GroupOf) -> list[Candidate]:
    """One candidate per hierarchy group (≤ 252), in first-seen ranking order, described by its size."""
    counts: dict[str, int] = {}
    for candidate in ranked:
        group = group_of(candidate)
        if group is not None:
            counts[group] = counts.get(group, 0) + 1
    return [
        Candidate(
            value=group,
            text=f"A group of {n} item{'s' if n != 1 else ''}.",
            channel=Channel.REGISTRY,
            prov={"group": group, "whole": True},
        )
        for group, n in list(counts.items())[:252]
    ]


def top_groups(distribution: Mapping[str, float]) -> list[str]:
    """The most probable groups until their cumulative mass reaches 0.9 (at most 3)."""
    top: list[str] = []
    mass = 0.0
    for group, p in sorted(distribution.items(), key=lambda gp: -gp[1]):
        if group.startswith("⊥"):
            continue
        top.append(group)
        mass += p
        if mass >= GROUP_MASS or len(top) == MAX_GROUPS:
            break
    return top


def hierarchy_stage(
    tool: ToolSpec,
    slot: SlotSpec,
    rc: ResolveContext,
    ranked: Sequence[Candidate],
    *,
    kind: str,
    group_of: GroupOf,
    is_path: bool = False,
) -> tuple[Pool, list[BallotQuestion]]:
    """One Choice over the items of the top groups of the previous group answer."""
    request = rc.widen.get(rc.slot_key(tool, slot), {})
    groups = top_groups(request.get("groups", {}))
    items = [c for c in ranked if group_of(c) in groups]
    pool = finalize_pool(
        tool, slot, rc, items, kind, is_path=is_path, meta={"mode": "widen_hierarchy", "groups": groups}
    )
    if not pool.candidates:
        return pool, []
    default = resolve_default(tool, slot, rc.ctx)
    question = slot_question(tool, slot, pool.candidates, default, meta={"widen": HIERARCHY, "groups": groups})
    return pool, [question]


def is_bucket_stage(questions: Sequence[BallotQuestion]) -> bool:
    """Whether a slot's Ballot questions are decoded as a bucket stage: bucket Choices were asked and no later
    hierarchy Choice replaced the slot question. Read from the Ballot alone, so a replay (``jt.verify``) decodes a
    widened slot without the widened pool."""
    hierarchy = any(q.family == "slot" and q.meta.get("widen") == HIERARCHY for q in questions)
    return not hierarchy and any(q.family == "bucket" for q in questions)


def decode_buckets(
    tool: ToolSpec, slot: SlotSpec, answers: Mapping[str, Answer], rc: ResolveContext, normalizer: str
) -> SlotResult:
    """Decode a bucket stage: the best bucket value times every other bucket's ``⊥uncovered`` mass."""
    oop = rc.policy.shapes.out_of_pool
    questions = rc.slot_questions(tool, slot)
    buckets = [decode_choice(slot, q, answers.get(q.qid), out_of_pool=oop) for q in questions if q.family == "bucket"]
    group_q = next((q for q in questions if q.family == "group"), None)
    uncovered = [r.mass(Bottom.UNCOVERED) for r in buckets]
    dist: dict[str, float] = {Bottom.UNCOVERED.value: math.prod(uncovered) if buckets else 0.0}
    values: dict[str, Any] = {}
    entries: dict[str, ValueEntry] = {}
    for i, result in enumerate(buckets):
        others = math.prod(u for j, u in enumerate(uncovered) if j != i)
        for value, p in result.top(3):
            key = value_key(value)
            dist[key] = max(dist.get(key, 0.0), p * others)
            values[key] = value
            if key in result.entries:
                entries[key] = result.entries[key]
    qids = tuple(q.qid for q in questions)
    elected = elect(
        path=slot.path,
        kind=slot.kind,
        stakes=slot.stakes,
        dist=dist,
        values=values,
        entries=entries,
        out_of_pool=oop,
        qids=qids,
    )
    parts: dict[str, SlotResult] = {f"bucket.{i}": r for i, r in enumerate(buckets)}
    if group_q is not None:
        parts["group"] = decode_choice(slot, group_q, answers.get(group_q.qid), out_of_pool=oop)
    return elected.with_(normalizer=normalizer, parts=parts)


__all__ = [
    "BUCKET",
    "HIERARCHY",
    "bucket_stage",
    "decode_buckets",
    "group_candidates",
    "hierarchy_stage",
    "is_bucket_stage",
    "top_groups",
]
