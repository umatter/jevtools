"""The ``text`` resolver (spec §4.2.11): free text elected by accept-Nouls, never written by Jev.

**Candidate ladder** (in this order; within an extracted rung, canonical order):

1. author templates (``x-jev.templates`` or the auto-attached pack: ``email.subject``, ``email.body``,
   ``email.forward``, ``event.title``) with early-bound placeholders filled from the request and the context
   (``{message} {duration} {object} {topic} {attendee_names} {observation_title} {user.<field>}``) and late-bound
   ones shown as ``⟨…⟩`` (``{recipient.first_name}`` → ``⟨recipient's first name⟩``, ``{observation}`` →
   ``⟨full text of the file read in step 1⟩``);
2. extracted text: message clauses and quotes; titles add the command's object phrase (full / core / head
   variants); queries use the request minus its command verb and the main noun chunk (plus the whole request
   when fewer than two remain);
3. rule-based perspective variants of the clauses (``tell her she should call me`` → ``You should call me.``),
   tagged ``rewrite:perspective``;
4. observation content handles (content slots);
5. schema ``examples``;
6. injected candidates (``generated`` via FILL, user replies) — :attr:`ResolveContext.injected`.

At most ``policy.pools.text_content_max`` (4) content or ``text_cosmetic_max`` (3) cosmetic candidates are asked.

**Questions.** One accept-Noul per candidate, ``T.P.accept.i``: content wording with criteria, cosmetic wording
without (§3.5.4). **Decode** (§3.6): ``elected = argmax n``, ties within 0.02 go to the lower index (templates
first). Content: uncovered (shape ``uncovered_text``) when ``max n < accept_min``; factor ``n(elected)``.
Cosmetic: floor ``cosmetic_floor``; below it the first author template, else omit (optional) or
``uncovered_text``; never a factor.

**Normalization** (§4.3, "never rewording"): content text gets a sentence-initial capital and final punctuation;
titles and subjects the capital only; queries neither.
"""

from __future__ import annotations

import posixpath
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from jevtools import templates
from jevtools.ballot import BallotQuestion, slot_qid
from jevtools.candidates import Bottom, Candidate, Channel, Pool, canonical_order, least_trusted, value_key
from jevtools.context import Observation
from jevtools.extract.base import Mention, Mentions
from jevtools.extract.catalogs import template_pack
from jevtools.extract.text import perspective_variant
from jevtools.extract.tokens import words
from jevtools.kinds.base import Alternative, ResolveContext, SlotResult, ValueEntry, register_resolver
from jevtools.kinds.common import finalize_pool, mention_prov
from jevtools.kinds.normalize import normalize_query, normalize_text, normalize_title
from jevtools.prompts import join_and
from jevtools.spec.models import SlotSpec, ToolSpec
from jevtools.wire import Answer, NoulAnswer

TIE = 0.02
PLACEHOLDER_RE = re.compile(r"\{([a-z_]+(?:\.[a-z_]+)?)\}")
PEOPLE_TAGS = frozenset({"email", "person"})
_PAST = {
    "read": "read",
    "open": "opened",
    "get": "retrieved",
    "fetch": "fetched",
    "download": "downloaded",
    "list": "listed",
    "load": "loaded",
}


def text_role(slot: SlotSpec) -> str:
    """``query``, ``title`` or ``body`` (content text without a role is a body)."""
    if slot.role in ("query", "title", "body"):
        return str(slot.role)
    return "body" if slot.stakes == "content" else "title"


def normalizer_name(slot: SlotSpec, *, template: bool = False) -> str:
    """The ``name@version`` of :func:`normalize_for` (an author template renders as written: ``text.template@1``)."""
    role = text_role(slot)
    if role in ("query", "title"):
        return f"text.{role}@1"
    return "text.template@1" if template else "text@1"


def normalize_for(slot: SlotSpec, raw: str, *, template: bool = False) -> str:
    """The role's text normalizer (content: capital + final punctuation; titles: capital; queries: as is).
    Author templates are rendered as written: no punctuation is added to them."""
    role = text_role(slot)
    if role == "query":
        return normalize_query(raw)
    if role == "title" or template:
        return normalize_title(raw)
    return normalize_text(raw)


# --------------------------------------------------------------------------------------------------------------------
# Template inputs
# --------------------------------------------------------------------------------------------------------------------


@dataclass
class TemplateInputs:
    """Early-bound values and late-bound handles available to templates for one tool."""

    early: dict[str, tuple[str, Channel]] = field(default_factory=dict)
    """Placeholder → (text, channel)."""
    late: dict[str, tuple[str, str, Channel]] = field(default_factory=dict)
    """Placeholder → (shown ``⟨…⟩`` text, late path, channel)."""
    cues: list[str] = field(default_factory=list)
    """Folded words of the request (for ``when`` cues)."""


def recipient_slot(tool: ToolSpec, rc: ResolveContext) -> SlotSpec | None:
    """The tool's recipient: the first identity ref/span slot over people (source ``provides`` email/person)."""
    for slot in tool.slots:
        if slot.kind not in ("ref", "span") or slot.stakes != "identity":
            continue
        provides = {t for n in slot.source_names if n in rc.ctx.sources for t in rc.ctx.sources[n].provides}
        if slot.format == "email" or provides & PEOPLE_TAGS:
            return slot
    return None


def people_list(tool: ToolSpec, rc: ResolveContext) -> SlotSpec | None:
    """The tool's anchored list of people (attendees), if any."""
    for slot in tool.slots:
        item = slot.item
        if slot.kind == "list" and item is not None and item.kind == "ref" and recipient_like(item, rc):
            return slot
    return None


def recipient_like(slot: SlotSpec, rc: ResolveContext) -> bool:
    provides = {t for n in slot.source_names if n in rc.ctx.sources for t in rc.ctx.sources[n].provides}
    return slot.format == "email" or bool(provides & PEOPLE_TAGS)


def handle_text(observation: Observation) -> str:
    """``⟨full text of the file read in step 1⟩`` (the late-bound content handle of an observation, §6.3)."""
    parts = observation.tool.replace("-", "_").split("_")
    verb, noun = parts[0].lower(), " ".join(parts[1:]).lower()
    if verb in _PAST and noun:
        what = f"the {noun} {_PAST[verb]} in step {observation.step}"
    else:
        what = f"the output of step {observation.step}"
    return templates.PLACEHOLDER.format(what=f"full text of {what}")


def observation_title(observation: Observation) -> str:
    """A short title of an observation: a path argument's file name (without date and extension), else the first
    line of the preview."""
    for value in observation.arguments.values():
        if isinstance(value, str) and ("/" in value or "." in value):
            stem = posixpath.splitext(posixpath.basename(value))[0]
            stem = re.sub(r"^\d{4}-\d{2}-\d{2}[_\- ]*", "", stem)
            return " ".join(stem.replace("_", " ").split())
    first = observation.preview_text().strip().split("\n", 1)[0]
    return first[:60].rstrip()


def template_inputs(tool: ToolSpec, rc: ResolveContext) -> TemplateInputs:
    """Everything a template may use, from the round's mentions and the context."""
    mentions: Mentions = rc.get_mentions()
    inputs = TemplateInputs(cues=[t.folded for t in mentions.tokens_of("request")])
    _request_inputs(inputs, tool, rc, mentions)
    _profile_inputs(inputs, rc)
    _late_inputs(inputs, tool, rc)
    return inputs


def _request_inputs(inputs: TemplateInputs, tool: ToolSpec, rc: ResolveContext, mentions: Mentions) -> None:
    """``{message} {duration} {object} {topic} {attendee_names}`` from the request's mentions."""
    request = [m for m in mentions if m.in_request]
    clause = next((m for m in request if m.kind == "clause"), None)
    if clause is not None:
        inputs.early["message"] = (normalize_text(clause.text), clause.channel)
    duration = next((m for m in request if m.kind == "quantity" and m.dim == "time" and m.free), None)
    if duration is not None:
        inputs.early["duration"] = (duration.text, duration.channel)
    chunks = {m.attrs.get("variant"): m for m in request if m.kind == "noun_phrase" and m.attrs.get("main")}
    obj = chunks.get("core") or chunks.get("full")
    if obj is not None:
        inputs.early["object"] = (obj.text, obj.channel)
    if "head" in chunks:
        inputs.early["topic"] = (templates.upper_first(chunks["head"].text), chunks["head"].channel)
    people = people_list(tool, rc)
    if people is not None and people.item is not None:
        names = [m.text for n in people.item.source_names for m in mentions.anchors(n) if m.in_request]
        if names:
            inputs.early["attendee_names"] = (join_and(names), Channel.USER)


def _profile_inputs(inputs: TemplateInputs, rc: ResolveContext) -> None:
    """``{user.<field>}`` from the shareable profile, plus ``{user.first_name}`` derived from ``name``."""
    profile = rc.ctx.user_state()
    for key, value in profile.items():
        if isinstance(value, str) and value:
            inputs.early[f"user.{key}"] = (value, Channel.REGISTRY)
    name = profile.get("name")
    if isinstance(name, str) and name.split() and "user.first_name" not in inputs.early:
        inputs.early["user.first_name"] = (name.split()[0], Channel.REGISTRY)


def _late_inputs(inputs: TemplateInputs, tool: ToolSpec, rc: ResolveContext) -> None:
    """Late-bound ``{recipient.first_name}`` and ``{observation}`` (plus the early ``{observation_title}``)."""
    recipient = recipient_slot(tool, rc)
    if recipient is not None:
        shown = "⟨recipient's first name⟩"
        inputs.late["recipient.first_name"] = (shown, f"{recipient.name}.first_name", Channel.REGISTRY)
    observations = rc.ctx.all_observations()
    if observations:
        latest = observations[-1]
        inputs.late["observation"] = (handle_text(latest), f"obs:{latest.step}", Channel.TOOL_OUTPUT)
        inputs.early["observation_title"] = (observation_title(latest), Channel.TOOL_OUTPUT)


def _cued(entry: Mapping[str, Any], inputs: TemplateInputs) -> bool:
    cues = entry.get("when")
    if not cues:
        return True
    return any(
        tuple(words(c)) == tuple(inputs.cues[i : i + len(words(c))]) for c in cues for i in range(len(inputs.cues))
    )


def fill_template(template: str, inputs: TemplateInputs) -> tuple[str, Channel, dict[str, Any] | None] | None:
    """Fill early placeholders and show late ones as ``⟨…⟩``; ``None`` when an early value is missing."""
    channels: list[Channel] = [Channel.AUTHOR]
    late_paths: list[str] = []
    fill: dict[str, str] = {}

    def substitute(match: re.Match[str]) -> str:
        name = match.group(1)
        if name in inputs.early:
            text, channel = inputs.early[name]
            channels.append(channel)
            return text
        if name in inputs.late:
            shown, path, channel = inputs.late[name]
            channels.append(channel)
            late_paths.append(path)
            fill[shown] = path
            return shown
        raise KeyError(name)

    try:
        text = PLACEHOLDER_RE.sub(substitute, template)
    except KeyError:
        return None
    late = {"placeholders": late_paths, "fill": fill} if late_paths else None
    return text, least_trusted(*channels), late


def template_candidates(slot: SlotSpec, inputs: TemplateInputs) -> list[Candidate]:
    """Rung 1: author templates (declared, else the attached packs) whose cues and inputs are present."""
    entries: list[tuple[str, Mapping[str, Any]]] = [("templates", {"templates": list(slot.templates)})]
    entries += [(pack, entry) for pack in slot.packs for entry in template_pack(pack)]
    out: list[Candidate] = []
    for source, entry in entries:
        if not _cued(entry, inputs) or any(
            r not in inputs.early and r not in inputs.late for r in entry.get("requires", ())
        ):
            continue
        for template in entry.get("templates", ()):
            filled = fill_template(template, inputs)
            if filled is None:
                continue
            text, channel, late = filled
            value = normalize_for(slot, text, template=True)
            out.append(
                Candidate(
                    value=value, text=value, channel=channel, late=late, prov={"source": source, "template": template}
                )
            )
    return out


def _mention_candidate(slot: SlotSpec, m: Mention, **prov: Any) -> Candidate:
    value = normalize_for(slot, m.text)
    return Candidate(value=value, text=value, channel=m.channel, prov=mention_prov(m, **prov))


def extracted_candidates(slot: SlotSpec, rc: ResolveContext) -> list[Candidate]:
    """Rung 2 (canonical order): clauses/quotes; title variants of the command's object; query variants."""
    mentions: Mentions = rc.get_mentions()
    request = [m for m in mentions if m.in_request]
    role = text_role(slot)
    out: list[Candidate] = []
    if role == "query":
        command = next((m for m in request if m.kind == "command"), None)
        chunks = [m for m in request if m.kind == "noun_phrase"]
        core = next((m for m in chunks if m.attrs.get("variant") == "core"), None)
        if core is None and command is None:  # a question: its first noun chunk is the main one
            core = next((m for m in chunks if not m.attrs.get("main")), None)
        out = [_mention_candidate(slot, m) for m in (command, core) if m is not None]
        if len({value_key(c.value) for c in out}) < 2 and rc.ctx.request.strip():
            whole = normalize_for(slot, rc.ctx.request)
            out.append(Candidate(value=whole, text=whole, channel=Channel.USER, prov={"extractor": "request"}))
    else:
        out = [_mention_candidate(slot, m) for m in request if m.kind in ("clause", "quote")]
        if role == "title":
            out += [
                _mention_candidate(slot, m, variant=m.attrs.get("variant"))
                for m in request
                if m.kind == "noun_phrase" and m.attrs.get("main")
            ]
    return canonical_order(out, key=lambda c: str(c.value))


_NOT_A_PERSON = frozenset({"place", "temporal", "enum", "money", "quantity"})


def _third_parties(mentions: Any, clause: Mention) -> bool:
    """Whether the clause names someone besides the recipient (a proper noun or a registry anchor that is not a
    place, date or enum value): its pronouns may then be theirs ("Tom is sick and he can't come")."""
    request = mentions.of(source_ref="request")
    other = [m.span for m in request if m.kind in _NOT_A_PERSON]
    return any(
        m.kind in ("proper_noun", "anchor")
        and clause.span[0] <= m.span[0]
        and m.span[1] <= clause.span[1]
        and not any(a < m.span[1] and m.span[0] < b for a, b in other)
        for m in request
    )


def perspective_candidates(slot: SlotSpec, rc: ResolveContext) -> list[Candidate]:
    """Rung 3: ``rewrite:perspective`` variants of the request's message clauses (content slots), only for clauses
    that name no one but the recipient: a third party's he/she/his is never rewritten to "you"."""
    out: list[Candidate] = []
    mentions = rc.get_mentions()
    for m in mentions.of("clause", source_ref="request"):
        if _third_parties(mentions, m):
            continue
        variant = perspective_variant(m.text)
        if variant is not None:
            value = normalize_for(slot, variant)
            out.append(
                Candidate(value=value, text=value, channel=m.channel, prov=mention_prov(m, rewrite="perspective"))
            )
    return out


def observation_candidates(rc: ResolveContext) -> list[Candidate]:
    """Rung 4: the latest observation's content handle (late-bound full text; ``tool_output``)."""
    observations = rc.ctx.all_observations()
    if not observations:
        return []
    latest = observations[-1]
    shown = handle_text(latest)
    late = {"placeholders": [f"obs:{latest.step}"], "fill": {shown: f"obs:{latest.step}"}}
    return [
        Candidate(
            value=shown,
            text=shown,
            channel=Channel.TOOL_OUTPUT,
            late=late,
            prov={"source": f"obs:{latest.step}", "handle": True},
        )
    ]


def example_candidates(slot: SlotSpec) -> list[Candidate]:
    """Rung 5: schema ``examples``."""
    examples = slot.json_schema.get("examples")
    return [
        Candidate(
            value=normalize_for(slot, e),
            text=normalize_for(slot, e),
            channel=Channel.AUTHOR,
            prov={"source": "examples"},
        )
        for e in (examples if isinstance(examples, list) else ())
        if isinstance(e, str) and e.strip()
    ]


# --------------------------------------------------------------------------------------------------------------------
# Resolver
# --------------------------------------------------------------------------------------------------------------------


class TextResolver:
    """Resolver for ``kind: text`` (accept-Nouls)."""

    kind = "text"
    normalizer = "text@1"

    def candidates(self, tool: ToolSpec, slot: SlotSpec, rc: ResolveContext) -> list[Candidate]:
        """The candidate ladder, rungs in order (duplicates are removed by the pool pipeline)."""
        content = text_role(slot) == "body"
        ladder = template_candidates(slot, template_inputs(tool, rc)) + extracted_candidates(slot, rc)
        if content:
            ladder += perspective_candidates(slot, rc) + observation_candidates(rc)
        return ladder + example_candidates(slot)

    def cap(self, slot: SlotSpec, rc: ResolveContext) -> int:
        pools = rc.policy.pools
        return pools.text_cosmetic_max if slot.stakes == "cosmetic" else pools.text_content_max

    def pool(self, tool: ToolSpec, slot: SlotSpec, rc: ResolveContext) -> Pool:
        return finalize_pool(
            tool,
            slot,
            rc,
            self.candidates(tool, slot, rc),
            self.kind,
            order=False,
            limit=self.cap(slot, rc),
            meta={"role": text_role(slot)},
        )

    def questions(self, tool: ToolSpec, slot: SlotSpec, pool: Pool, rc: ResolveContext) -> list[BallotQuestion]:
        content = slot.stakes != "cosmetic"
        return [
            BallotQuestion(
                qid=slot_qid(tool.id, slot.qpath, "accept", i),
                family="accept",
                tool=tool.name,
                path=slot.path,
                kind=slot.kind,
                stakes=slot.stakes,
                primitive="noul",
                instructions=templates.accept_instructions(tool.intent, slot.noun, str(c.value), content=content),
                criteria=dict(templates.ACCEPT_CONTENT_CRITERIA) if content else None,
                meta={"index": i, "candidate": candidate_doc(c)},
            )
            for i, c in enumerate(pool.candidates)
        ]

    def decode(
        self, tool: ToolSpec, slot: SlotSpec, pool: Pool, answers: Mapping[str, Answer], rc: ResolveContext
    ) -> SlotResult:
        questions = [q for q in rc.slot_questions(tool, slot) if q.family == "accept"] or self.questions(
            tool, slot, pool, rc
        )
        scored: list[tuple[Candidate, float]] = []
        for question in questions:
            answer = answers.get(question.qid)
            n = answer.noul if isinstance(answer, NoulAnswer) else 0.0
            scored.append((candidate_of(question), n))
        return accept_result(slot, scored, rc, tuple(q.qid for q in questions), normalizer_name(slot))


def candidate_doc(candidate: Candidate) -> dict[str, Any]:
    """The accept Noul's decode data: its candidate, so decoding needs the Ballot alone (a FILL round's generated
    candidates, a replay by ``jt.verify``)."""
    return {"value": candidate.value, "channel": candidate.channel.value, "prov": candidate.prov,
            "late": candidate.late}  # fmt: skip


def candidate_of(question: BallotQuestion) -> Candidate:
    """The candidate an accept Noul asked about (from :func:`candidate_doc`)."""
    doc = question.meta["candidate"]
    return Candidate(value=doc["value"], channel=Channel(doc["channel"]), prov=dict(doc.get("prov") or {}),
                     late=doc.get("late"))  # fmt: skip


def elect_accept(scores: Sequence[float]) -> int | None:
    """``argmax n`` with ties within 0.02 going to the lower index (§3.6 accept row)."""
    best: int | None = None
    for i, n in enumerate(scores):
        if best is None or n > scores[best] + TIE:
            best = i
    return best


def accept_result(
    slot: SlotSpec,
    scored: Sequence[tuple[Candidate, float]],
    rc: ResolveContext,
    qids: tuple[str, ...],
    normalizer: str,
) -> SlotResult:
    """Decode accept-Nouls into a :class:`SlotResult` (content: factor ``n(elected)``; cosmetic: no factor)."""
    content = slot.stakes != "cosmetic"
    shapes = rc.policy.shapes
    dist: dict[str, float] = {}
    values: dict[str, Any] = {}
    entries: dict[str, ValueEntry] = {}
    for c, n in scored:
        key = value_key(c.value)
        if n >= dist.get(key, -1.0):
            dist[key], values[key] = n, c.value
            entries[key] = ValueEntry(display=str(c.value), channel=c.channel, prov=c.prov, late=c.late, p=n)
    best = elect_accept([n for _, n in scored])
    top = scored[best][1] if best is not None else 0.0
    floor = shapes.accept_min if content else shapes.cosmetic_floor
    base = SlotResult(
        path=slot.path,
        kind=slot.kind,
        stakes=slot.stakes,
        dist=dist,
        values=values,
        value=Bottom.UNCOVERED,
        shape="uncovered_text",
        factor=None,
        qids=qids,
        normalizer=normalizer,
        entries=entries,
        probes={"accept": top},
    )
    if best is not None and top >= floor:
        chosen = scored[best][0]
        normalizer = normalizer_name(slot, template=True) if "template" in chosen.prov else normalizer
        alternatives = tuple(
            Alternative(display=str(c.value), p=n, value=c.value)
            for c, n in sorted(scored, key=lambda cn: -cn[1])
            if content and c is not chosen and n >= shapes.accept_min
        )[:3]
        return base.with_(
            value=chosen.value,
            shape="ok",
            factor=top if content else None,
            display=str(chosen.value),
            alternatives=alternatives,
            channel=chosen.channel,
            prov=chosen.prov,
            late=chosen.late,
            normalizer=normalizer,
        )
    template = next((c for c, _ in scored if "template" in c.prov), None)
    if not content and template is not None:
        return base.with_(
            value=template.value,
            shape="ok",
            display=str(template.value),
            channel=template.channel,
            prov={**template.prov, "below_floor": True},
            late=template.late,
            normalizer=normalizer_name(slot, template=True),
            notes=("no candidate reached the cosmetic floor; first author template used",),
        )
    if not content and not slot.required:
        omit = Bottom.OMIT.value
        return base.with_(
            dist={**dist, omit: 1.0 - top},
            value=Bottom.OMIT,
            shape="ok",
            notes=("no candidate reached the cosmetic floor; omitted",),
        )
    return base.with_(
        dist={**dist, Bottom.UNCOVERED.value: 1.0 - top},
        factor=top if content else None,
        notes=("no candidate was accepted",),
    )


register_resolver("text", TextResolver())

__all__ = [
    "TIE",
    "TemplateInputs",
    "TextResolver",
    "accept_result",
    "elect_accept",
    "fill_template",
    "handle_text",
    "normalize_for",
    "observation_title",
    "template_candidates",
    "template_inputs",
    "text_role",
]
