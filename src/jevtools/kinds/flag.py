"""The ``flag`` resolver (spec §4.2.3): booleans.

- Without a default: one Noul ``T.P`` phrased as a yes/no question (``Does the user want {noun} to be true?``).
  Decoding: true at ``n ≥ 0.8``, false at ``n ≤ 0.2``; the dead band (0.2, 0.8) gives shape ``flag_band``
  (clarify with a yes/no menu). The factor is ``max(n, 1 − n)``.
- With a default: a three-option Choice ``{true, false, NOT_STATED}`` so that ``NOT_STATED`` decodes to (and pools
  with) the default; the factor is ``D(v*)``.
"""

from __future__ import annotations

from collections.abc import Mapping

from jevtools import templates
from jevtools.ballot import BallotOption, BallotQuestion, slot_qid
from jevtools.candidates import NOT_STATED, Candidate, Channel, Pool, value_key
from jevtools.kinds.base import (
    ResolveContext,
    SlotResult,
    ValueEntry,
    decode_choice,
    not_stated_spec,
    register_resolver,
    resolve_default,
)
from jevtools.kinds.common import finalize_pool
from jevtools.kinds.listing import item_band
from jevtools.spec.models import SlotSpec, ToolSpec
from jevtools.wire import Answer, NoulAnswer


class FlagResolver:
    """Resolver for ``kind: flag``."""

    kind = "flag"
    normalizer = "flag@1"

    def pool(self, tool: ToolSpec, slot: SlotSpec, rc: ResolveContext) -> Pool:
        members = [Candidate(value=v, channel=Channel.AUTHOR, prov={"source": "flag"}) for v in (True, False)]
        return finalize_pool(tool, slot, rc, members, self.kind, closed=True, order=False)

    def questions(self, tool: ToolSpec, slot: SlotSpec, pool: Pool, rc: ResolveContext) -> list[BallotQuestion]:
        default = resolve_default(tool, slot, rc.ctx)
        ask = templates.slot_instructions(tool.intent, templates.slot_ask(slot.noun, slot.ask, kind="flag"))
        common = {"tool": tool.name, "path": slot.path, "kind": slot.kind, "stakes": slot.stakes}
        if default is not None and not default.omit:
            return [
                BallotQuestion(
                    qid=slot_qid(tool.id, slot.qpath),
                    family="slot",
                    primitive="choice",
                    instructions=ask,
                    options=[BallotOption.from_candidate(c) for c in pool.candidates],
                    sentinels={NOT_STATED: not_stated_spec(slot, default)},
                    **common,
                )
            ]
        return [
            BallotQuestion(
                qid=slot_qid(tool.id, slot.qpath), family="slot", primitive="noul", instructions=ask, **common
            )
        ]

    def decode(
        self, tool: ToolSpec, slot: SlotSpec, pool: Pool, answers: Mapping[str, Answer], rc: ResolveContext
    ) -> SlotResult:
        question = next(iter(rc.slot_questions(tool, slot)), None) or self.questions(tool, slot, pool, rc)[0]
        answer = answers.get(question.qid)
        if question.primitive == "choice":
            result = decode_choice(slot, question, answer, out_of_pool=rc.policy.shapes.out_of_pool)
            return result.with_(normalizer=self.normalizer)
        return noul_flag_result(slot, answer, question.qid, rc.policy.shapes.flag_band, self.normalizer)


def noul_flag_result(
    slot: SlotSpec, answer: Answer | None, qid: str, band: tuple[float, float], normalizer: str
) -> SlotResult:
    """A flag Noul: true ≥ 0.8, false ≤ 0.2, dead band → ``flag_band``; factor ``max(n, 1 − n)``."""
    if not isinstance(answer, NoulAnswer):
        return SlotResult(
            path=slot.path,
            kind=slot.kind,
            stakes=slot.stakes,
            dist={},
            values={},
            value=False,
            shape="missing",
            factor=0.0,
            flags=("no_answer",),
            qids=(qid,),
            normalizer=normalizer,
        )
    n = answer.noul
    decision = item_band(n, band)
    value = n >= 0.5 if decision is None else decision
    dist = {value_key(True): n, value_key(False): 1 - n}
    entries = {
        value_key(v): ValueEntry(
            display="true" if v else "false",
            label="true" if v else "false",
            channel=Channel.AUTHOR,
            p=dist[value_key(v)],
        )
        for v in (True, False)
    }
    return SlotResult(
        path=slot.path,
        kind=slot.kind,
        stakes=slot.stakes,
        dist=dist,
        values={value_key(True): True, value_key(False): False},
        value=value,
        shape="flag_band" if decision is None else "ok",
        factor=None if slot.stakes == "cosmetic" else max(n, 1 - n),
        display="true" if value else "false",
        channel=Channel.AUTHOR,
        qids=(qid,),
        normalizer=normalizer,
        entries=entries,
    )


register_resolver("flag", FlagResolver())

__all__ = ["FlagResolver", "noul_flag_result"]
