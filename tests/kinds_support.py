"""Helpers for resolver tests: scenario resolve contexts, custom one-tool catalogs and hand-built answers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from jevtools.backends.scripted import choice_answer
from jevtools.ballot import BallotQuestion
from jevtools.candidates import Pool
from jevtools.context import Context
from jevtools.kinds import ResolveContext, SlotResult, get_resolver
from jevtools.spec.catalog import Catalog
from jevtools.spec.models import SlotSpec, ToolSpec
from jevtools.wire import Answer, ChoiceAnswer, NoulAnswer
from tests.scenario_sources import scenario_context
from tests.support import SCENARIO_NOW, load_fixture


def scenario(request: str, *, history: bool = False, **kw: Any) -> tuple[Catalog, ResolveContext]:
    """The §13 catalog compiled against the real scenario sources, and a resolve context for ``request``."""
    ctx = scenario_context(request, history=history, **kw)
    catalog = Catalog.from_openai(load_fixture("scenario_catalog.json"), sources=list(ctx.sources.values()))
    return catalog, ResolveContext(ctx=ctx, catalog=catalog)


def custom(
    name: str,
    properties: Mapping[str, Any],
    request: str,
    *,
    required: Sequence[str] = (),
    description: str = "",
    tool_xjev: Mapping[str, Any] | None = None,
    sources: Sequence[Any] = (),
    **ctx_kw: Any,
) -> tuple[ToolSpec, ResolveContext]:
    """A one-tool catalog and its resolve context."""
    function: dict[str, Any] = {
        "name": name,
        "description": description or f"Do {name}.",
        "parameters": {"type": "object", "properties": dict(properties), "required": list(required)},
    }
    if tool_xjev:
        function["x-jev"] = dict(tool_xjev)
    catalog = Catalog.from_openai([{"type": "function", "function": function}], sources=list(sources))
    ctx = Context(messages=request, now=SCENARIO_NOW, locale="en-CH", sources=list(sources), **ctx_kw)
    return catalog[name], ResolveContext(ctx=ctx, catalog=catalog)


def resolve(tool: ToolSpec, slot: SlotSpec, rc: ResolveContext) -> tuple[Pool, list[BallotQuestion]]:
    """Pool and questions of one slot; the questions are registered as the decode map of ``rc``."""
    resolver = get_resolver(slot.kind)
    pool = resolver.pool(tool, slot, rc)
    questions = resolver.questions(tool, slot, pool, rc)
    rc.questions = {**rc.questions, **{q.qid: q for q in questions}}
    return pool, questions


def choice(question: BallotQuestion, probs: Mapping[str, float]) -> ChoiceAnswer:
    """A Choice answer over the question's labels (unlisted labels get 0)."""
    answer = choice_answer(question.labels, {**dict.fromkeys(question.labels, 0.0), **probs})
    return answer.model_copy(
        update={"probabilities": {label: float(probs.get(label, 0.0)) for label in question.labels}}
    )


def noul(p: float) -> NoulAnswer:
    return NoulAnswer(noul=p)


def decode(tool: ToolSpec, slot: SlotSpec, pool: Pool, answers: Mapping[str, Answer], rc: ResolveContext) -> SlotResult:
    return get_resolver(slot.kind).decode(tool, slot, pool, answers, rc)
