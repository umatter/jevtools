"""The ``ordinal`` resolver (spec §4.2.3): a Score over author-described levels, for genuinely graded values.

Levels come from the enum members (with their descriptions) or from the integer bounds. Decoding (§3.6 score
row): exact — ``v = argmax level``, ``f = P(v)``; ``x-jev.tolerant`` — ``v = round(score)``, ``f = P(v)``. An expected
level is never used as an exact number (the ``jev`` package's ``ge/le → Score`` mapping is not adopted).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from jevtools import templates
from jevtools.ballot import BallotQuestion, slot_qid
from jevtools.candidates import Candidate, Channel, Pool, display_value, value_key
from jevtools.kinds.base import ResolveContext, SlotResult, ValueEntry, register_resolver
from jevtools.kinds.common import finalize_pool
from jevtools.spec.models import SlotSpec, ToolSpec
from jevtools.wire import Answer, ScoreAnswer

MAX_LEVELS = 11


def levels_of(slot: SlotSpec) -> list[tuple[Any, str | None]]:
    """``(value, description)`` per level, lowest first: enum members, else the integers between the bounds."""
    if slot.values:
        return [(m.value, m.text) for m in slot.values]
    schema = slot.json_schema
    low = schema.get("minimum", schema.get("exclusiveMinimum"))
    high = schema.get("maximum", schema.get("exclusiveMaximum"))
    if not isinstance(low, int) or not isinstance(high, int):
        return []
    low += 0 if "minimum" in schema else 1
    high -= 0 if "maximum" in schema else 1
    levels: list[tuple[Any, str | None]] = [(v, None) for v in range(low, high + 1)]
    return levels[:MAX_LEVELS]


class OrdinalResolver:
    """Resolver for ``kind: ordinal``."""

    kind = "ordinal"
    normalizer = "enum@1"

    def pool(self, tool: ToolSpec, slot: SlotSpec, rc: ResolveContext) -> Pool:
        levels = [
            Candidate(value=v, text=t, channel=Channel.AUTHOR, prov={"level": i})
            for i, (v, t) in enumerate(levels_of(slot))
        ]
        return finalize_pool(tool, slot, rc, levels, self.kind, closed=True, order=False)

    def questions(self, tool: ToolSpec, slot: SlotSpec, pool: Pool, rc: ResolveContext) -> list[BallotQuestion]:
        if not pool.candidates:
            return []
        levels: list[Any] = [f"{c.label}: {c.text}" if c.text else c.label for c in pool.candidates]
        ask = templates.slot_instructions(tool.intent, templates.slot_ask(slot.noun, slot.ask))
        return [
            BallotQuestion(
                qid=slot_qid(tool.id, slot.qpath),
                family="slot",
                tool=tool.name,
                path=slot.path,
                kind=slot.kind,
                stakes=slot.stakes,
                primitive="score",
                instructions=ask,
                levels=levels,
                meta={"values": [c.value for c in pool.candidates]},
            )
        ]

    def decode(
        self, tool: ToolSpec, slot: SlotSpec, pool: Pool, answers: Mapping[str, Answer], rc: ResolveContext
    ) -> SlotResult:
        qid = slot_qid(tool.id, slot.qpath)
        answer = answers.get(qid)
        values = [c.value for c in pool.candidates]
        if not isinstance(answer, ScoreAnswer) or not values:
            return SlotResult(
                path=slot.path,
                kind=slot.kind,
                stakes=slot.stakes,
                dist={},
                values={},
                value=None,
                shape="missing",
                factor=0.0,
                flags=("no_answer",),
                qids=(qid,),
                normalizer=self.normalizer,
            )
        probs = {int(k): float(p) for k, p in answer.probabilities.items() if 0 <= int(k) < len(values)}
        if slot.tolerant:
            index = min(max(round(answer.score), 0), len(values) - 1)
        else:
            index = max(range(len(values)), key=lambda i: (probs.get(i, 0.0), -i))
        dist = {value_key(v): probs.get(i, 0.0) for i, v in enumerate(values)}
        entries = {
            value_key(c.value): ValueEntry(
                display=display_value(c.value),
                label=c.label,
                channel=c.channel,
                prov=c.prov,
                p=dist[value_key(c.value)],
            )
            for c in pool.candidates
        }
        value = values[index]
        p = probs.get(index, 0.0)
        return SlotResult(
            path=slot.path,
            kind=slot.kind,
            stakes=slot.stakes,
            dist=dist,
            values={value_key(v): v for v in values},
            value=value,
            shape="ok",
            factor=None if slot.stakes == "cosmetic" else p,
            display=display_value(value),
            label=pool.candidates[index].label,
            channel=Channel.AUTHOR,
            qids=(qid,),
            normalizer=self.normalizer,
            entries=entries,
            notes=(f"score {answer.score:.4f}",),
        )


register_resolver("ordinal", OrdinalResolver())

__all__ = ["OrdinalResolver", "levels_of"]
