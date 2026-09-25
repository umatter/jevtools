"""The ``list`` resolver (spec §4.2.9, §4.2.10 arrays of objects): anchored lists, multi-select, enumerative lists.

- **Anchored** (``ref`` items the user named, e.g. attendees): one ``mention`` Choice ``T.P.m<i>`` per user anchor
  (≤ ``policy.pools.mentions_max``), over that anchor's matches (≤ 12) plus ``EXCLUDE`` and ``NONE_OF_THESE``,
  and one ``more`` Noul. A group mention (an anchor matching only the source's group field, "the payments team")
  expands into item Nouls over the group's members. Mention Choices are preferred to per-candidate Nouls for
  people because Nouls do not compete ("Bob Meier" and "Bobby Tan" could both score high).
- **Multi-select** (array of enum): one ``item`` Noul per member.
- **Enumerative** (other item kinds, no anchors): item Nouls over the item resolver's pool (≤ 60).
- **Array of objects** (basic): one record per anchor of the first ``ref`` field; each field asked per anchor
  (``T.P.m<i>.<field>``), plus a ``more`` Noul.

Decoding (§3.6, §3.7.1): per anchor ``v = argmax`` (``EXCLUDE`` drops the mention, ``NONE_OF_THESE`` makes the list
``out_of_pool``); items are included at ``n ≥ 0.8``, excluded at ``n ≤ 0.2``, and the dead band gives shape
``flag_band``; ``more ≥ 0.5`` gives shape ``missing`` with flag ``more`` (clarify "Who else should I invite?").
The factor is ``∏ P(anchor value) · ∏ max(n, 1 − n) · (1 − n_more)``.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, cast

from jevtools import templates
from jevtools.ballot import BallotOption, BallotQuestion, SentinelSpec, slot_qid
from jevtools.candidates import EXCLUDE, NONE_OF_THESE, Candidate, Channel, Pool, least_trusted, value_key
from jevtools.extract.base import Mention
from jevtools.kinds.base import (
    Alternative,
    ResolveContext,
    Shape,
    SlotResult,
    ValueEntry,
    decode_choice,
    get_resolver,
    probe_question,
    register_resolver,
    resolve_default,
    unasked_result,
)
from jevtools.kinds.common import finalize_pool
from jevtools.kinds.normalize import normalize_list
from jevtools.sources.base import SourceQuery
from jevtools.spec.models import SlotSpec, ToolSpec
from jevtools.wire import Answer, NoulAnswer

PER_ANCHOR = 12
PEOPLE_TAGS = frozenset({"email", "person"})
SHAPE_RANK: dict[str, int] = {"ok": 0, "flag_band": 1, "missing": 2, "uncovered_text": 3, "out_of_pool": 4}


def worst_shape(shapes: Sequence[str]) -> Shape:
    """The most severe shape (out_of_pool > uncovered_text > missing > flag_band > ok)."""
    return cast(Shape, max(shapes, key=lambda s: SHAPE_RANK[s], default="ok"))


def item_band(n: float, band: tuple[float, float]) -> bool | None:
    """Include (``True``) at ``n ≥ 0.8``, exclude (``False``) at ``n ≤ 0.2``, uncertain (``None``) in between."""
    low, high = band
    if n >= high:
        return True
    if n <= low:
        return False
    return None


def _noul(answers: Mapping[str, Answer], qid: str) -> float | None:
    answer = answers.get(qid)
    return answer.noul if isinstance(answer, NoulAnswer) else None


@dataclass
class _Tally:
    """Accumulates the parts of a list decode (anchors, items, ``more``) into one :class:`SlotResult`."""

    values: list[Any] = field(default_factory=list)
    factors: list[float] = field(default_factory=list)
    shapes: list[str] = field(default_factory=list)
    flags: list[str] = field(default_factory=list)
    channels: list[Channel] = field(default_factory=list)
    alternatives: list[Alternative] = field(default_factory=list)
    parts: dict[str, SlotResult] = field(default_factory=dict)
    probes: dict[str, float] = field(default_factory=dict)
    included: list[tuple[float, int, Any]] = field(default_factory=list)
    """Included item values keyed by where they were mentioned: items are asked in canonical (neutral) option order,
    but the list keeps mention order (§4.3 "dedupe preserving mention order")."""

    def mention(self, part: str, r: SlotResult) -> None:
        """One anchor: ``argmax`` value (``EXCLUDE`` drops it), factor ``P(anchor value)``."""
        self.parts[part] = r
        self.factors.append(r.dist.get(value_key(r.value), 0.0))
        self.shapes.append(r.shape)
        self.flags += r.flags
        if not r.is_bottom:
            self.values.append(r.value)
            self.channels += [r.channel] if r.channel is not None else []
        self.alternatives += [Alternative(a.display, a.p, a.value, a.label, part=part) for a in r.alternatives]

    def item(self, candidate: Candidate, n: float | None, band: tuple[float, float]) -> None:
        """One item Noul (a missing answer is uncertain): include ≥ 0.8, exclude ≤ 0.2, else ``flag_band``."""
        n = 0.5 if n is None else n
        decision = item_band(n, band)
        self.factors.append(max(n, 1 - n))
        if decision is None:
            self.shapes.append("flag_band")
        elif decision:
            mention = candidate.prov.get("mention")
            span = mention.get("span") if isinstance(mention, Mapping) else None
            start = float(span[0]) if isinstance(span, (list, tuple)) and span else math.inf
            self.included.append((start, len(self.included), candidate.value))
            self.channels.append(candidate.channel)

    def more(self, n: float | None) -> None:
        """The ``more`` Noul (a missing answer fails closed): factor ``1 − n``; ``n ≥ 0.5`` asks "who else"."""
        n = 1.0 if n is None else n
        self.probes["more"] = n
        self.factors.append(1 - n)
        if n >= 0.5:
            self.shapes.append("missing")
            self.flags.append("more")

    def result(self, slot: SlotSpec, qids: tuple[str, ...], normalizer: str) -> SlotResult:
        items = [v for _, _, v in sorted(self.included, key=lambda entry: (entry[0], entry[1]))]
        value = normalize_list([*self.values, *items], slot.json_schema)
        factor = math.prod(self.factors) if self.factors else 0.0
        key = value_key(value)
        channel = least_trusted(*self.channels) if self.channels else None
        entry = ValueEntry(display=", ".join(str(v) for v in value), channel=channel, p=factor)
        return SlotResult(
            path=slot.path,
            kind=slot.kind,
            stakes=slot.stakes,
            dist={key: factor},
            values={key: value},
            value=value,
            shape=worst_shape(self.shapes),
            factor=None if slot.stakes == "cosmetic" else factor,
            display=entry.display,
            flags=tuple(dict.fromkeys(self.flags)),
            alternatives=tuple(self.alternatives),
            channel=channel,
            qids=qids,
            parts=self.parts,
            normalizer=normalizer,
            entries={key: entry},
            probes=self.probes,
        )


class ListResolver:
    """Resolver for ``kind: list``."""

    kind = "list"
    normalizer = "list@1"

    # -- pool ------------------------------------------------------------------------------------------------------

    def pool(self, tool: ToolSpec, slot: SlotSpec, rc: ResolveContext) -> Pool:
        item = slot.item
        if item is None:
            return Pool(tool=tool.name, path=slot.path, kind=self.kind, notes=["list without an item schema"])
        if item.kind == "enum":
            return self._items_pool(tool, slot, get_resolver("enum").pool(tool, item, rc), "multi", rc)
        if item.kind == "ref" and slot.anchored:
            return self._anchored_pool(tool, slot, item, rc)
        if item.kind == "record":
            return self._records_pool(tool, slot, item, rc)
        inner = get_resolver(item.kind).pool(tool, item, rc)
        return self._items_pool(tool, slot, inner, "enumerative", rc)

    def _items_pool(self, tool: ToolSpec, slot: SlotSpec, inner: Pool, mode: str, rc: ResolveContext) -> Pool:
        items = inner.candidates[: rc.policy.pools.items_max]
        return inner.model_copy(
            update={"kind": self.kind, "path": slot.path, "candidates": items, "meta": {"mode": mode}}
        )

    def _anchors(self, item: SlotSpec, rc: ResolveContext) -> list[Mention]:
        mentions = rc.get_mentions()
        found = [m for name in item.source_names for m in mentions.anchors(name)]
        found.sort(key=lambda m: (not m.in_request, m.source_ref, m.span))
        return found[: rc.policy.pools.mentions_max]

    def _sub_pool(self, tool: ToolSpec, item: SlotSpec, anchor: Mention, rc: ResolveContext) -> list[Candidate]:
        source = rc.ctx.sources.get(str(anchor.attrs.get("source")))
        if source is None:
            return []
        matches = list(anchor.attrs.get("matches", ()))[:PER_ANCHOR]
        if hasattr(source, "rows") and hasattr(source, "candidate"):
            candidates = [source.candidate(source.rows[m.index], m, anchor) for m in matches]
        else:
            candidates = source.candidates(SourceQuery(slot=item, request=anchor.text, k=PER_ANCHOR, context=rc.ctx))
        return finalize_pool(tool, item, rc, candidates, "ref").candidates

    def _group_members(self, tool: ToolSpec, item: SlotSpec, anchor: Mention, rc: ResolveContext) -> list[Candidate]:
        source = rc.ctx.sources.get(str(anchor.attrs.get("source")))
        matches = list(anchor.attrs.get("matches", ()))
        if source is None or not matches or any(m.how != "group" for m in matches):
            return []
        candidates = [source.candidate(source.rows[m.index], m, anchor) for m in matches]
        return finalize_pool(tool, item, rc, candidates, "ref").candidates[: rc.policy.pools.items_max]

    def _anchored_pool(self, tool: ToolSpec, slot: SlotSpec, item: SlotSpec, rc: ResolveContext) -> Pool:
        anchors: list[dict[str, Any]] = []
        groups: list[Candidate] = []
        blocked: list[Candidate] = []
        for anchor in self._anchors(item, rc):
            members = self._group_members(tool, item, anchor, rc)
            if members:
                groups += [c for c in members if value_key(c.value) not in {value_key(g.value) for g in groups}]
                continue
            candidates = self._sub_pool(tool, item, anchor, rc)
            if candidates:
                anchors.append(
                    {
                        "text": anchor.text,
                        "span": list(anchor.span),
                        "negated": anchor.negated,
                        "candidates": candidates,
                    }
                )
        everyone = [c for a in anchors for c in a["candidates"]] + groups
        return Pool(
            tool=tool.name,
            path=slot.path,
            kind=self.kind,
            candidates=everyone,
            evidence_backed=any(c.is_evidence for c in everyone),
            blocked=blocked,
            meta={"mode": "anchored", "anchors": anchors, "groups": groups, "person": self._people(item, rc)},
        )

    def _people(self, item: SlotSpec, rc: ResolveContext) -> bool:
        provides = {t for n in item.source_names if n in rc.ctx.sources for t in rc.ctx.sources[n].provides}
        return bool(provides & PEOPLE_TAGS) or item.format == "email"

    # -- questions -------------------------------------------------------------------------------------------------

    def questions(self, tool: ToolSpec, slot: SlotSpec, pool: Pool, rc: ResolveContext) -> list[BallotQuestion]:
        mode = pool.meta.get("mode")
        if mode == "anchored":
            out = self._anchored_questions(tool, slot, pool)
        elif mode == "records":
            out = self._record_questions(tool, slot, pool, rc)
        else:
            out = [self._item_question(tool, slot, c, i) for i, c in enumerate(pool.candidates)]
        if out:
            return out
        default = resolve_default(tool, slot, rc.ctx)
        return [probe_question(tool, slot, default)] if default is not None and not default.omit else []

    def _item_question(self, tool: ToolSpec, slot: SlotSpec, candidate: Candidate, i: int) -> BallotQuestion:
        item = f'"{candidate.label}"' + (f" ({candidate.text.rstrip('.')})" if candidate.text else "")
        return BallotQuestion(
            qid=slot_qid(tool.id, slot.qpath, "item", i),
            family="item",
            tool=tool.name,
            path=slot.path,
            kind=slot.kind,
            stakes=slot.stakes,
            primitive="noul",
            instructions=templates.item_instructions(tool.intent, item, slot.noun),
            meta={"index": i},
        )

    def _more_question(self, tool: ToolSpec, slot: SlotSpec, mentions: Sequence[str]) -> BallotQuestion:
        return BallotQuestion(
            qid=slot_qid(tool.id, slot.qpath, "more"),
            family="more",
            tool=tool.name,
            path=slot.path,
            kind=slot.kind,
            stakes=slot.stakes,
            primitive="noul",
            instructions=templates.more_instructions(tool.intent, mentions, slot.noun),
        )

    def _anchored_questions(self, tool: ToolSpec, slot: SlotSpec, pool: Pool) -> list[BallotQuestion]:
        person = bool(pool.meta.get("person"))
        item_noun = templates.ITEM_NOUN_DEFAULT if person else (slot.item.noun if slot.item else "item")
        out: list[BallotQuestion] = []
        anchors = pool.meta.get("anchors", [])
        for i, anchor in enumerate(anchors):
            text = str(anchor["text"])
            out.append(
                BallotQuestion(
                    qid=slot_qid(tool.id, slot.qpath, f"m{i}"),
                    family="mention",
                    tool=tool.name,
                    path=slot.path,
                    kind=slot.kind,
                    stakes=slot.stakes,
                    primitive="choice",
                    instructions=templates.mention_instructions(tool.intent, text, item_noun),
                    options=[BallotOption.from_candidate(c) for c in anchor["candidates"]],
                    sentinels={
                        EXCLUDE: SentinelSpec(
                            decodes_to="excluded", text=templates.mention_exclude_text(text, slot.noun)
                        ),
                        NONE_OF_THESE: SentinelSpec(
                            decodes_to="uncovered", text=templates.mention_none_text(text, person=person)
                        ),
                    },
                    meta={"anchor": i, "mention": text},
                )
            )
        groups: list[Candidate] = pool.meta.get("groups", [])
        if anchors or groups:
            out.append(self._more_question(tool, slot, [str(a["text"]) for a in anchors] or ["the group"]))
        out += [self._item_question(tool, slot, c, i) for i, c in enumerate(groups)]
        return out

    # -- decode ----------------------------------------------------------------------------------------------------

    def decode(
        self, tool: ToolSpec, slot: SlotSpec, pool: Pool, answers: Mapping[str, Answer], rc: ResolveContext
    ) -> SlotResult:
        questions = rc.slot_questions(tool, slot) or self.questions(tool, slot, pool, rc)
        if not questions:
            return unasked_result(slot, resolve_default(tool, slot, rc.ctx)).with_(normalizer=self.normalizer)
        if questions[0].family == "probe":
            return decode_choice(
                slot, questions[0], answers.get(questions[0].qid), out_of_pool=rc.policy.shapes.out_of_pool
            ).with_(normalizer=self.normalizer)
        if pool.meta.get("mode") == "records":
            return self._decode_records(tool, slot, pool, answers, rc)
        return self._decode_parts(slot, pool, questions, answers, rc)

    def _decode_parts(
        self,
        slot: SlotSpec,
        pool: Pool,
        questions: Sequence[BallotQuestion],
        answers: Mapping[str, Answer],
        rc: ResolveContext,
    ) -> SlotResult:
        assert slot.item is not None
        tally = _Tally()
        items = pool.meta.get("groups") if pool.meta.get("mode") == "anchored" else pool.candidates
        attrs = {c.label: c.attrs for c in pool.candidates}
        for q in questions:
            if q.family == "mention":
                oop = rc.policy.shapes.out_of_pool
                tally.mention(
                    f"m{q.meta['anchor']}",
                    decode_choice(slot.item, q, answers.get(q.qid), out_of_pool=oop, attrs=attrs),
                )
            elif q.family == "item":
                tally.item((items or [])[int(q.meta["index"])], _noul(answers, q.qid), rc.policy.shapes.flag_band)
            elif q.family == "more":
                tally.more(_noul(answers, q.qid))
        return tally.result(slot, tuple(q.qid for q in questions), self.normalizer)

    # -- arrays of objects (basic) ---------------------------------------------------------------------------------

    def _records_pool(self, tool: ToolSpec, slot: SlotSpec, item: SlotSpec, rc: ResolveContext) -> Pool:
        anchor_field = next((c for c in item.children if c.kind == "ref"), None)
        anchors = self._anchors(anchor_field, rc) if anchor_field is not None else []
        records: list[dict[str, Any]] = []
        everyone: list[Candidate] = []
        for i, anchor in enumerate(anchors):
            fields: dict[str, Any] = {}
            for child in item.children:
                spec = self._record_field(slot, child, i)
                if child is anchor_field:
                    candidates = self._sub_pool(tool, spec, anchor, rc)
                    fields[child.name] = Pool(
                        tool=tool.name,
                        path=spec.path,
                        kind="ref",
                        candidates=candidates,
                        evidence_backed=bool(candidates),
                    )
                else:
                    fields[child.name] = get_resolver(child.kind).pool(tool, spec, rc)
                everyone += fields[child.name].candidates
            records.append({"anchor": anchor.text, "fields": fields})
        return Pool(
            tool=tool.name,
            path=slot.path,
            kind=self.kind,
            candidates=everyone,
            evidence_backed=any(c.is_evidence for c in everyone),
            meta={"mode": "records", "records": records, "anchor_field": anchor_field.name if anchor_field else None},
        )

    @staticmethod
    def _record_field(slot: SlotSpec, child: SlotSpec, i: int) -> SlotSpec:
        """A per-anchor copy of a record field: path ``(…, "[i]", field)``, qpath ``P.m<i>.<field>``."""
        segment = child.qpath.rsplit(".", 1)[-1]
        return child.model_copy(
            update={"path": (*slot.path, f"[{i}]", child.name), "qpath": f"{slot.qpath}.m{i}.{segment}"}
        )

    def _record_questions(self, tool: ToolSpec, slot: SlotSpec, pool: Pool, rc: ResolveContext) -> list[BallotQuestion]:
        assert slot.item is not None
        out: list[BallotQuestion] = []
        for i, record in enumerate(pool.meta.get("records", [])):
            for child in slot.item.children:
                spec = self._record_field(slot, child, i)
                out += get_resolver(child.kind).questions(tool, spec, record["fields"][child.name], rc)
        if out:
            out.append(self._more_question(tool, slot, [r["anchor"] for r in pool.meta.get("records", [])]))
        return out

    def _decode_records(
        self, tool: ToolSpec, slot: SlotSpec, pool: Pool, answers: Mapping[str, Answer], rc: ResolveContext
    ) -> SlotResult:
        assert slot.item is not None
        records: list[dict[str, Any]] = []
        factors: list[float] = []
        shapes: list[str] = []
        parts: dict[str, SlotResult] = {}
        for i, record in enumerate(pool.meta.get("records", [])):
            obj: dict[str, Any] = {}
            for child in slot.item.children:
                spec = self._record_field(slot, child, i)
                r = get_resolver(child.kind).decode(tool, spec, record["fields"][child.name], answers, rc)
                parts[f"m{i}.{child.name}"] = r
                shapes.append(r.shape)
                if r.factor is not None:
                    factors.append(r.factor)
                if not r.is_bottom:
                    obj[child.name] = r.value
            records.append(obj)
        more = _noul(answers, slot_qid(tool.id, slot.qpath, "more"))
        more = 1.0 if more is None else more
        factors.append(1 - more)
        if more >= 0.5:
            shapes.append("missing")
        factor = math.prod(factors)
        key = value_key(records)
        # Like a nested record (record.assemble): the leaves' consistency flags (presence_conflict,
        # order_sensitive, no_answer) and least-trusted channel reach P8/P9 through the list result.
        flags = [f for r in parts.values() for f in r.flags] + (["more"] if more >= 0.5 else [])
        channels = [r.channel for r in parts.values() if r.channel is not None and not r.is_bottom]
        qids = [q.qid for q in rc.slot_questions(tool, slot)] + [q for r in parts.values() for q in r.qids]
        return SlotResult(
            path=slot.path,
            kind=slot.kind,
            stakes=slot.stakes,
            dist={key: factor},
            values={key: records},
            value=records,
            shape=worst_shape(shapes),
            factor=factor,
            parts=parts,
            normalizer=self.normalizer,
            flags=tuple(dict.fromkeys(flags)),
            channel=least_trusted(*channels) if channels else None,
            probes={"more": more},
            qids=tuple(dict.fromkeys(qids)),
        )


register_resolver("list", ListResolver())

__all__ = ["ListResolver", "item_band", "worst_shape"]
