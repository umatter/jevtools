"""The ``ref`` resolver (spec §4.2.7, §4.2.8, §4.6): entity references into registered sources.

**Pool.** Each source's candidates for the round (registries: the anchored shortlist, cut to K = ``x-jev.k`` or
``policy.pools.ref_k``, or the whole registry when it has ≤ 12 rows; file indexes: the BM25 top K; providers:
whatever they return), plus literal format matches the user typed (an email address not in the contacts) and,
for anaphoric requests, recent entity-store values of the slot's type (``history`` channel, origin trust).
The value is the source key, verbatim; the label the source's WYSIWYG label.

**Questions.** The slot Choice; a ``present`` Noul in tiers ``policy.probes.present`` (external, critical); a ``rev``
Choice (real options in reverse canonical order) in tiers ``policy.probes.reverse`` (critical) when ≥ 2 real
candidates exist. ``x-jev.probe`` forces either probe on or off.

**Decode.** ``D_final = min(D_fwd, D_rev)`` (per value), flag ``order_sensitive`` when the two argmaxes differ;
flag ``presence_conflict`` when ``present < 0.5`` while v* is real with D ≥ 0.5, or ``present ≥ 0.5`` while v* is
``⊥missing``. ``SlotResult.probes`` records ``present`` and ``rev``.

**Superlative refs** (a ``latest/oldest/largest…`` cue and an orderable attribute): one ``member`` Noul per
retrieved item (≤ 40, retrieval on the request without the cue); code picks the extreme of ``M = {q > 0.5}`` and
the factor is ``q_chosen · ∏_{j beyond chosen} (1 − q_j)`` (§4.2.8). Jev never sorts.

**Widen** (:class:`~jevtools.kinds.base.Widenable`): bucket Choices beyond K plus a hierarchy group Choice, then a
Choice over the items of the top groups (:mod:`jevtools.kinds.widen`).
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from jevtools import templates
from jevtools.ballot import BallotOption, BallotQuestion, slot_qid
from jevtools.candidates import Bottom, Candidate, Pool, reverse_order, value_key
from jevtools.extract.base import Mention, Mentions
from jevtools.extract.coref import coref_candidates
from jevtools.kinds.base import (
    ALTERNATIVE_MIN_P,
    Alternative,
    ResolveContext,
    SlotResult,
    ValueEntry,
    decode_choice,
    register_resolver,
    resolve_default,
    unasked_result,
)
from jevtools.kinds.common import ChoiceResolver, finalize_pool, mention_candidate, pool_mentions
from jevtools.kinds.normalize import NormalizationError, normalize_email_value, normalize_path
from jevtools.kinds.widen import (
    BUCKET,
    HIERARCHY,
    bucket_stage,
    decode_buckets,
    hierarchy_stage,
    is_bucket_stage,
)
from jevtools.sources.base import SourceQuery
from jevtools.spec.models import SlotSpec, ToolSpec
from jevtools.wire import Answer, NoulAnswer

ORDER_DEFAULTS: dict[str, tuple[str, str]] = {
    "latest": ("date", "max"),
    "oldest": ("date", "min"),
    "largest": ("size", "max"),
    "smallest": ("size", "min"),
    "cheapest": ("price", "min"),
}
"""Canonical superlative cue → (default order attribute, direction)."""
FORMAT_MENTIONS: dict[str, str] = {"email": "email", "uri": "url", "uuid": "uuid", "ipv4": "ipv4"}
PATH_TAGS = frozenset({"path", "file"})


def slot_sources(slot: SlotSpec, rc: ResolveContext) -> list[Any]:
    """The registered sources named by the slot (missing ones are skipped)."""
    return [rc.ctx.sources[name] for name in slot.source_names if name in rc.ctx.sources]


def is_path_slot(slot: SlotSpec) -> bool:
    return bool(PATH_TAGS & set(slot.tags))


def superlative(slot: SlotSpec, mentions: Mentions, sources: Sequence[Any]) -> tuple[Mention, str, str] | None:
    """``(cue mention, order attribute, direction)`` when the request has a superlative cue and the slot's sources
    expose the attribute (``x-jev.order_by`` maps cue → attribute)."""
    for cue in mentions.cues("superlative"):
        if not cue.in_request:
            continue
        canonical = str(cue.attrs.get("canonical"))
        attr, direction = ORDER_DEFAULTS.get(canonical, ("date", "max"))
        attr = slot.order_by.get(cue.text.lower(), slot.order_by.get(canonical, attr))
        if any(_has_attr(source, attr) for source in sources):
            return cue, attr, direction
    return None


def _has_attr(source: Any, attr: str) -> bool:
    names = getattr(source, "attribute_names", None)
    return callable(names) and attr in names()


def superlative_text(mentions: Mentions, cue: Mention) -> str:
    """Retrieval text: the noun phrase containing the cue (else the request), without the cue words."""
    phrases = [m for m in mentions.of("noun_phrase", source_ref="request") if m.span[0] <= cue.span[0] < m.span[1]]
    phrases.sort(key=lambda m: (not m.attrs.get("main"), m.attrs.get("variant") != "full"))
    text, offset = (phrases[0].text, phrases[0].span[0]) if phrases else (mentions.request, 0)
    start, end = cue.span[0] - offset, cue.span[1] - offset
    return " ".join((text[:start] + text[end:]).split())


class RefResolver(ChoiceResolver):
    """Resolver for ``kind: ref`` (with superlative refs and widen rounds)."""

    kind = "ref"
    normalizer = "ref@1"

    # -- candidates ----------------------------------------------------------------------------------------------

    def query(self, slot: SlotSpec, rc: ResolveContext, **changes: Any) -> SourceQuery:
        k = slot.k or rc.policy.pools.ref_k
        fields: dict[str, Any] = {
            "slot": slot,
            "mentions": rc.get_mentions(),
            "request": rc.ctx.request,
            "k": k,
            "context": rc.ctx,
        }
        fields.update(changes)
        return SourceQuery(**fields)

    def candidates(self, tool: ToolSpec, slot: SlotSpec, rc: ResolveContext) -> list[Candidate]:
        q = self.query(slot, rc)
        out: list[Candidate] = []
        for source in slot_sources(slot, rc):
            out += source.candidates(q)
        out += self.literal(slot, rc)
        out += coref_candidates(rc.get_mentions(), rc.ctx.entities, (*slot.tags, *self._provided(slot, rc)))
        return out

    def _provided(self, slot: SlotSpec, rc: ResolveContext) -> list[str]:
        return [tag for source in slot_sources(slot, rc) for tag in getattr(source, "provides", ())]

    def literal(self, slot: SlotSpec, rc: ResolveContext) -> list[Candidate]:
        """Values of the slot's format typed literally (``anna@example.com``), from any pool channel."""
        kind = FORMAT_MENTIONS.get(slot.format or "")
        if kind is None:
            return []
        out: list[Candidate] = []
        for m in pool_mentions(rc, kind):
            value = normalize_email_value(m.value) if kind == "email" else m.value
            out.append(mention_candidate(m, value, display=value))
        return out

    def pool(self, tool: ToolSpec, slot: SlotSpec, rc: ResolveContext) -> Pool:
        sources = slot_sources(slot, rc)
        mentions = rc.get_mentions()
        found = superlative(slot, mentions, sources)
        if found is not None:
            return self._superlative_pool(tool, slot, rc, sources, *found)
        missing = [n for n in slot.source_names if n not in rc.ctx.sources]
        notes = [f"source {n!r} is not registered" for n in missing]
        candidates = [c for c in self.candidates(tool, slot, rc) if self._path_ok(slot, c)]
        return finalize_pool(
            tool,
            slot,
            rc,
            candidates,
            self.kind,
            is_path=is_path_slot(slot),
            notes=notes,
            meta={"mode": "choice", "k": slot.k or rc.policy.pools.ref_k},
        )

    def _path_ok(self, slot: SlotSpec, candidate: Candidate) -> bool:
        if not is_path_slot(slot) or not isinstance(candidate.value, str):
            return True
        try:
            normalize_path(candidate.value, known="source" in candidate.prov)
        except NormalizationError:
            return False
        return True

    # -- questions -----------------------------------------------------------------------------------------------

    def questions(self, tool: ToolSpec, slot: SlotSpec, pool: Pool, rc: ResolveContext) -> list[BallotQuestion]:
        mode = pool.meta.get("mode", "choice")
        if mode == "superlative":
            return self._member_questions(tool, slot, pool, rc)
        base = super().questions(tool, slot, pool, rc)
        if not base or base[0].family != "slot":
            return base
        out = list(base)
        if self._probe(slot.probe_present, tool, rc.policy.probes.present):
            out.append(self.present_question(tool, slot))
        real = len(pool.candidates)
        if real >= 2 and self._probe(slot.probe_reverse, tool, rc.policy.probes.reverse):
            out.append(self.rev_question(tool, base[0]))
        return out

    @staticmethod
    def _probe(forced: bool | None, tool: ToolSpec, tiers: Sequence[Any]) -> bool:
        return forced if forced is not None else tool.tier in tiers

    @staticmethod
    def present_question(tool: ToolSpec, slot: SlotSpec) -> BallotQuestion:
        """``T.P.present``: does the user say or clearly imply the value?"""
        return BallotQuestion(
            qid=slot_qid(tool.id, slot.qpath, "present"),
            family="present",
            tool=tool.name,
            path=slot.path,
            kind=slot.kind,
            stakes=slot.stakes,
            primitive="noul",
            instructions=templates.present_instructions(tool.intent, slot.noun),
            criteria=dict(templates.PRESENT_CRITERIA),
        )

    @staticmethod
    def rev_question(tool: ToolSpec, forward: BallotQuestion) -> BallotQuestion:
        """``T.P.rev``: the slot Choice with real options in reverse canonical order."""
        options: list[BallotOption] = reverse_order(forward.options)
        return forward.model_copy(update={"qid": forward.qid + ".rev", "family": "rev", "options": options})

    # -- decode --------------------------------------------------------------------------------------------------

    def decode(
        self, tool: ToolSpec, slot: SlotSpec, pool: Pool, answers: Mapping[str, Answer], rc: ResolveContext
    ) -> SlotResult:
        mode = pool.meta.get("mode", "choice")
        if mode == "superlative":
            return self._decode_members(tool, slot, pool, answers, rc)
        if is_bucket_stage(rc.slot_questions(tool, slot)):
            return decode_buckets(tool, slot, answers, rc, self.normalizer)
        result = super().decode(tool, slot, pool, answers, rc)
        questions = {q.family: q for q in rc.slot_questions(tool, slot)}
        flags = list(result.flags)
        probes: dict[str, float] = {}
        rev_q = questions.get("rev")
        if rev_q is not None and rev_q.qid in answers:
            attrs = {c.label: c.attrs for c in pool.candidates}
            rev = decode_choice(slot, rev_q, answers[rev_q.qid], out_of_pool=rc.policy.shapes.out_of_pool, attrs=attrs)
            probes["rev"] = rev.dist.get(value_key(result.value), 0.0)
            if value_key(rev.value) != value_key(result.value):
                flags.append("order_sensitive")
            result = self._min_merge(result, rev, rc)
        present_q = questions.get("present")
        answer = answers.get(present_q.qid) if present_q is not None else None
        if isinstance(answer, NoulAnswer):
            probes["present"] = answer.noul
            real = not result.is_bottom
            if (answer.noul < 0.5 and real and result.p >= 0.5) or (
                answer.noul >= 0.5 and result.value is Bottom.MISSING
            ):
                flags.append("presence_conflict")
        return result.with_(flags=tuple(dict.fromkeys(flags)), probes={**result.probes, **probes})

    def _min_merge(self, fwd: SlotResult, rev: SlotResult, rc: ResolveContext) -> SlotResult:
        """``D(v) = min(D_fwd(v), D_rev(v))`` for every value; v* stays the forward argmax (§3.6 ``rev``)."""
        dist = {k: min(p, rev.dist.get(k, 0.0)) for k, p in fwd.dist.items()}
        best = value_key(fwd.value)
        merged = (
            Alternative(display=a.display, p=dist.get(value_key(a.value), 0.0), value=a.value, label=a.label)
            for a in fwd.alternatives
        )
        alternatives = tuple(sorted((a for a in merged if a.p >= ALTERNATIVE_MIN_P), key=lambda a: -a.p))
        factor = None if fwd.factor is None else dist.get(best, 0.0)
        return fwd.with_(dist=dist, factor=factor, alternatives=alternatives, qids=fwd.qids + rev.qids)

    # -- superlative ---------------------------------------------------------------------------------------------

    def _superlative_pool(
        self,
        tool: ToolSpec,
        slot: SlotSpec,
        rc: ResolveContext,
        sources: Sequence[Any],
        cue: Mention,
        attr: str,
        direction: str,
    ) -> Pool:
        mentions = rc.get_mentions()
        text = superlative_text(mentions, cue)
        q = self.query(slot, rc, text=text, k=rc.policy.pools.members_max)
        candidates = [c for source in sources for c in source.candidates(q)]
        pool = finalize_pool(
            tool,
            slot,
            rc,
            candidates,
            self.kind,
            is_path=is_path_slot(slot),
            order=False,
            limit=rc.policy.pools.members_max,
        )
        meta = {
            "mode": "superlative",
            "cue": cue.text,
            "canonical": cue.attrs.get("canonical"),
            "attr": attr,
            "direction": direction,
            "query": text,
        }
        return pool.model_copy(update={"meta": meta})

    def _member_questions(self, tool: ToolSpec, slot: SlotSpec, pool: Pool, rc: ResolveContext) -> list[BallotQuestion]:
        attr = str(pool.meta["attr"])
        out: list[BallotQuestion] = []
        for i, candidate in enumerate(pool.candidates):
            item = f"{candidate.label} — {attr} {candidate.attrs.get(attr, 'unknown')}"
            out.append(
                BallotQuestion(
                    qid=slot_qid(tool.id, slot.qpath, "member", i),
                    family="member",
                    tool=tool.name,
                    path=slot.path,
                    kind=slot.kind,
                    stakes=slot.stakes,
                    primitive="noul",
                    instructions=templates.member_instructions(tool.intent, str(pool.meta["cue"]), item),
                    meta={"index": i, "value": candidate.value, "order": candidate.attrs.get(attr)},
                )
            )
        return out

    def _decode_members(
        self, tool: ToolSpec, slot: SlotSpec, pool: Pool, answers: Mapping[str, Answer], rc: ResolveContext
    ) -> SlotResult:
        questions = [q for q in rc.slot_questions(tool, slot) if q.family == "member"] or self._member_questions(
            tool, slot, pool, rc
        )
        if not questions:
            return unasked_result(slot, resolve_default(tool, slot, rc.ctx)).with_(normalizer=self.normalizer)
        items: list[tuple[Candidate, float, Any]] = []
        for question in questions:
            answer = answers.get(question.qid)
            q = answer.noul if isinstance(answer, NoulAnswer) else 0.0
            candidate = pool.candidates[int(question.meta["index"])]
            items.append((candidate, q, question.meta.get("order")))
        return member_result(slot, items, str(pool.meta["direction"]), tuple(q.qid for q in questions), self.normalizer)

    # -- widen ---------------------------------------------------------------------------------------------------

    def widen(
        self, tool: ToolSpec, slot: SlotSpec, pool: Pool, rc: ResolveContext, stage: str
    ) -> tuple[Pool, list[BallotQuestion]]:
        """Coverage rounds (§4.6): ``bucket`` then ``hierarchy`` over the sources' full rankings."""
        sources = [s for s in slot_sources(slot, rc) if callable(getattr(s, "ranked", None))]
        if not sources:
            return pool, []
        q = self.query(slot, rc, widen=True)
        ranked = [c for s in sources for c in s.ranked(q)]
        grouped = [s for s in sources if callable(getattr(s, "group_of", None))]

        def group_of(candidate: Candidate) -> str | None:
            return next((g for s in grouped if (g := s.group_of(candidate)) is not None), None)

        has_groups = any(group_of(c) is not None for c in ranked)
        is_path = is_path_slot(slot)
        if stage == BUCKET:
            skip = slot.k or rc.policy.pools.ref_k
            return bucket_stage(
                tool,
                slot,
                rc,
                ranked,
                skip=skip,
                kind=self.kind,
                group_of=group_of if has_groups else None,
                is_path=is_path,
            )
        if stage == HIERARCHY and has_groups:
            return hierarchy_stage(tool, slot, rc, ranked, kind=self.kind, group_of=group_of, is_path=is_path)
        return pool, []


def member_result(
    slot: SlotSpec,
    items: Sequence[tuple[Candidate, float, Any]],
    direction: str,
    qids: tuple[str, ...],
    normalizer: str,
) -> SlotResult:
    """Superlative decoding (§3.6 member row, §3.7.1): ``M = {q > 0.5}``, the extreme of M by the order attribute,
    factor ``q_chosen · ∏_{j beyond chosen} (1 − q_j) · ∏_{j tied with chosen} (1 − q_j)``; an empty M is
    ``out_of_pool``.

    Items tied with the chosen one on the order attribute (two invoices of the same day) are competitors too: "i
    is the extreme member" needs them out of M, so a tie at the extreme gives a low factor (flag ``tie``: the
    policy clarifies instead of picking by retrieval order) and the distribution stays ≤ 1. Among tied items the
    pick is deterministic (value order), never the pool order."""
    members = [(c, q, order) for c, q, order in items if q > 0.5 and order is not None]
    uncovered_mass = math.prod(1 - q for _, q, _ in items)
    if not members:
        return SlotResult(
            path=slot.path,
            kind=slot.kind,
            stakes=slot.stakes,
            dist={Bottom.UNCOVERED.value: uncovered_mass},
            values={},
            value=Bottom.UNCOVERED,
            shape="out_of_pool",
            factor=uncovered_mass,
            qids=qids,
            normalizer=normalizer,
            notes=("no item matched",),
        )
    reverse = direction == "max"

    def rivals(candidate: Candidate, order: Any) -> list[float]:
        """Items beyond ``order`` or tied with it (other than ``candidate`` itself)."""
        return [
            q
            for c, q, o in items
            if o is not None and c is not candidate and (o == order or (o > order if reverse else o < order))
        ]

    def factor_of(candidate: Candidate, q: float, order: Any) -> float:
        return q * math.prod(1 - other for other in rivals(candidate, order))

    by_value = sorted(members, key=lambda m: value_key(m[0].value))
    ranked = sorted(by_value, key=lambda m: m[2], reverse=reverse)  # stable: ties keep value order
    chosen, q_chosen, order_chosen = ranked[0]
    tied = [c for c, _, o in members if o == order_chosen]
    dist = {value_key(c.value): factor_of(c, q, o) for c, q, o in ranked}
    dist[Bottom.UNCOVERED.value] = uncovered_mass
    entries = {
        value_key(c.value): ValueEntry(
            display=c.shown, label=c.label, channel=c.channel, prov=c.prov, attrs=c.attrs, p=dist[value_key(c.value)]
        )
        for c, _, _ in ranked
    }
    factor = dist[value_key(chosen.value)]
    alternatives = tuple(
        Alternative(display=c.shown, p=dist[value_key(c.value)], value=c.value, label=c.label)
        for c, _, _ in ranked[1:4]
    )
    return SlotResult(
        path=slot.path,
        kind=slot.kind,
        stakes=slot.stakes,
        dist=dist,
        values={value_key(c.value): c.value for c, _, _ in ranked},
        value=chosen.value,
        shape="ok",
        factor=None if slot.stakes == "cosmetic" else factor,
        display=chosen.shown,
        label=chosen.label,
        alternatives=alternatives,
        channel=chosen.channel,
        prov=chosen.prov,
        attrs=chosen.attrs,
        qids=qids,
        normalizer=normalizer,
        entries=entries,
        flags=("tie",) if len(tied) > 1 else (),
        notes=(f"picked the {direction} of {len(members)} matching item(s) by order attribute ({order_chosen})",)
        + ((f"{len(tied)} matching items tie at {order_chosen}",) if len(tied) > 1 else ()),
    )


register_resolver("ref", RefResolver())

__all__ = ["ORDER_DEFAULTS", "RefResolver", "member_result", "superlative", "superlative_text"]
