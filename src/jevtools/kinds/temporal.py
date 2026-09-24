"""The ``temporal`` resolver (spec §4.2.5, ext.temporal): every reading of every temporal mention.

- **Single Choice** over complete readings when there are at most 24 (``Tue 2026-09-29 15:00 (Europe/Zurich)``),
  each described by its reading ("next Tuesday at 3pm" read as the coming Tuesday, in 5 days).
- **Factorized** ``T.P.date`` × ``T.P.time`` Choices above 24 readings (or when a date-time slot has dates without
  times): decoded as ``D(date, time) = D_date · D_time`` and composed in code (zoneinfo, DST-aware).
- ``format: date`` slots get dates, ``format: time`` slots times, ``format: duration`` slots durations
  (``PT45M``).
- Readings violating a unary constraint (``start > now``) are dropped at pool time. Vague cues ("after lunch")
  are not point candidates: they are recorded in ``pool.meta["ranges"]`` for a clarify menu.
- **Ranges** (``x-jev.range``): the ``min`` slot asks one Choice over ``(min, max)`` pairs; the ``max`` slot asks
  nothing and decodes its component from the same answer.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import date, datetime, time
from decimal import Decimal
from typing import Any

from jevtools.ballot import BallotQuestion, slot_qid
from jevtools.candidates import (
    NOT_STATED,
    Bottom,
    Candidate,
    Channel,
    Pool,
    display_value,
    least_trusted,
    value_key,
)
from jevtools.extract.base import Mention
from jevtools.extract.numbers import TIME_UNITS
from jevtools.extract.temporal import RangeReading, Reading, TemporalValue, localize, utc_offset
from jevtools.kinds.base import (
    LATE_DEFAULT,
    ResolveContext,
    SlotResult,
    ValueEntry,
    decode_choice,
    elect,
    register_resolver,
    resolve_default,
    slot_question,
    unasked_result,
)
from jevtools.kinds.common import ChoiceResolver, finalize_pool, mention_prov, mention_text, pool_mentions
from jevtools.kinds.normalize import iso_duration, normalize_temporal
from jevtools.spec.models import SlotSpec, ToolSpec
from jevtools.wire import Answer

MAX_SINGLE = 24
"""At most this many complete readings go into one Choice; above it the slot is factorized (§3.5.3)."""
WEEKDAY_ABBR = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")
NORMALIZER = "temporal.iso8601@1"


def datetime_label(dt: datetime, tz: str, *, fold: str | None = None) -> str:
    """``Tue 2026-09-29 15:00 (Europe/Zurich)``; a DST-fold reading adds its offset to stay unique."""
    zone = tz + (f", {utc_offset(dt)}" if fold else "")
    return f"{WEEKDAY_ABBR[dt.weekday()]} {dt:%Y-%m-%d %H:%M} ({zone})"


def date_label(day: date) -> str:
    """``Tue 2026-09-29``."""
    return f"{WEEKDAY_ABBR[day.weekday()]} {day.isoformat()}"


def time_label(clock: time) -> str:
    """``15:00``."""
    return f"{clock:%H:%M}"


def reading_text(m: Mention, gloss: str) -> str:
    """``"next Tuesday at 3pm" read as the coming Tuesday, in 5 days.`` for request mentions with several
    readings; the plain mention description otherwise."""
    if not gloss:
        return mention_text(m)
    if m.in_request and not m.negated:
        return f'"{m.text}" {gloss}.'
    return mention_text(m, note=gloss)


def temporal_mentions(rc: ResolveContext) -> list[Mention]:
    """Free temporal pool mentions of the round (the allow-list filters channels later)."""
    return [m for m in pool_mentions(rc, "temporal") if isinstance(m.value, TemporalValue)]


def slot_format(slot: SlotSpec) -> str:
    """``date``, ``time``, ``duration`` or ``date-time`` (the default for temporal strings)."""
    return slot.format if slot.format in ("date", "time", "duration") else "date-time"


class TemporalResolver(ChoiceResolver):
    """Resolver for ``kind: temporal``."""

    kind = "temporal"
    normalizer = NORMALIZER

    # -- candidates ----------------------------------------------------------------------------------------------

    def candidates(self, tool: ToolSpec, slot: SlotSpec, rc: ResolveContext) -> list[Candidate]:
        fmt = slot_format(slot)
        if fmt == "duration":
            return self._durations(slot, rc)
        out: list[Candidate] = []
        for m in temporal_mentions(rc):
            for reading in m.value.readings:
                candidate = self._point(slot, m, reading, fmt)
                if candidate is not None:
                    out.append(candidate)
        return out

    def _point(self, slot: SlotSpec, m: Mention, reading: Reading, fmt: str) -> Candidate | None:
        prov = mention_prov(m, reading=reading.name)
        text = reading_text(m, reading.gloss)
        if fmt == "date":
            day = reading.dt.date() if reading.dt else reading.date
            if day is None:
                return None
            return Candidate(
                value=normalize_temporal(day), display=date_label(day), text=text, channel=m.channel, prov=prov
            )
        if fmt == "time":
            clock = reading.dt.time() if reading.dt else reading.time
            if clock is None:
                return None
            return Candidate(
                value=normalize_temporal(clock), display=time_label(clock), text=text, channel=m.channel, prov=prov
            )
        if reading.dt is None:
            return None
        return Candidate(
            value=normalize_temporal(reading.dt, slot.json_schema),
            display=datetime_label(reading.dt, reading.tz, fold=reading.fold),
            text=text,
            channel=m.channel,
            prov=prov,
        )

    def _durations(self, slot: SlotSpec, rc: ResolveContext) -> list[Candidate]:
        out: list[Candidate] = []
        for m in pool_mentions(rc, "quantity"):
            if m.dim != "time":
                continue
            seconds = Decimal(m.value) * TIME_UNITS[str(m.attrs["unit"])]
            value = iso_duration(seconds)
            out.append(Candidate(value=value, text=mention_text(m), channel=m.channel, prov=mention_prov(m)))
        return out

    # -- pool --------------------------------------------------------------------------------------------------------

    def pool(self, tool: ToolSpec, slot: SlotSpec, rc: ResolveContext) -> Pool:
        coupled = self._range_role(tool, slot)
        if coupled == "max":
            return Pool(
                tool=tool.name,
                path=slot.path,
                kind=self.kind,
                meta={"mode": "range_max"},
                notes=["decoded from the range question of the min slot"],
            )
        if coupled == "min":
            return self._range_pool(tool, slot, rc)
        candidates = self.candidates(tool, slot, rc)
        ranges = [
            {
                "start": r.start.isoformat() if r.start else None,
                "end": r.end.isoformat() if r.end else None,
                "gloss": r.gloss,
                "vague": r.vague,
                "mention": m.text,
            }
            for m in temporal_mentions(rc)
            for r in m.value.ranges
        ]
        meta: dict[str, Any] = {"mode": "single", "ranges": ranges}
        pool = finalize_pool(tool, slot, rc, candidates, self.kind, meta=meta)
        incomplete = slot_format(slot) == "date-time" and self._dates_without_time(rc)
        if len(pool.candidates) > MAX_SINGLE or (incomplete and not pool.candidates):
            return self._factorized(tool, slot, rc, pool)
        return pool

    def _dates_without_time(self, rc: ResolveContext) -> bool:
        return any(r.dt is None and r.date is not None for m in temporal_mentions(rc) for r in m.value.readings)

    def _factorized(self, tool: ToolSpec, slot: SlotSpec, rc: ResolveContext, pool: Pool) -> Pool:
        dates: list[Candidate] = []
        times: list[Candidate] = []
        for m in temporal_mentions(rc):
            for reading in m.value.readings:
                day = reading.dt.date() if reading.dt else reading.date
                clock = reading.dt.time().replace(tzinfo=None) if reading.dt else reading.time
                prov = mention_prov(m, reading=reading.name)
                text = reading_text(m, reading.gloss)
                if day is not None:
                    dates.append(
                        Candidate(
                            value=day.isoformat(), display=date_label(day), text=text, channel=m.channel, prov=prov
                        )
                    )
                if clock is not None:
                    times.append(
                        Candidate(
                            value=clock.strftime("%H:%M:%S"),
                            display=time_label(clock),
                            text=text,
                            channel=m.channel,
                            prov=prov,
                        )
                    )
        date_slot = slot.model_copy(update={"format": "date", "json_schema": {"type": "string", "format": "date"}})
        time_slot = slot.model_copy(update={"format": "time", "json_schema": {"type": "string", "format": "time"}})
        date_pool = finalize_pool(tool, date_slot, rc, dates, self.kind)
        time_pool = finalize_pool(tool, time_slot, rc, times, self.kind)
        meta = {
            **pool.meta,
            "mode": "factorized",
            "dates": date_pool.candidates,
            "times": time_pool.candidates,
            "tz": rc.ctx.timezone_name,
        }
        return pool.model_copy(
            update={
                "meta": meta,
                "evidence_backed": date_pool.evidence_backed,
                "blocked": pool.blocked + date_pool.blocked + time_pool.blocked,
            }
        )

    # -- questions ---------------------------------------------------------------------------------------------------

    def questions(self, tool: ToolSpec, slot: SlotSpec, pool: Pool, rc: ResolveContext) -> list[BallotQuestion]:
        mode = pool.meta.get("mode", "single")
        if mode == "range_max":
            return []
        if mode != "factorized":
            return super().questions(tool, slot, pool, rc)
        default = resolve_default(tool, slot, rc.ctx)
        out: list[BallotQuestion] = []
        for part, noun in (("date", "the date"), ("time", "the time of day")):
            part_slot = slot.model_copy(update={"noun": f"{noun} of {slot.noun}", "ask": None})
            out.append(
                slot_question(
                    tool, part_slot, pool.meta[f"{part}s"], default, family=part, suffix=(part,), meta={"part": part}
                )
            )
        return out

    # -- decode ------------------------------------------------------------------------------------------------------

    def decode(
        self, tool: ToolSpec, slot: SlotSpec, pool: Pool, answers: Mapping[str, Answer], rc: ResolveContext
    ) -> SlotResult:
        mode = pool.meta.get("mode", "single")
        if mode == "range_max":
            return self._decode_range_max(tool, slot, answers, rc)
        if mode == "range":
            result = super().decode(tool, slot, pool, answers, rc)
            return _component(result, "min", rc.policy.shapes.out_of_pool)
        if mode != "factorized":
            return super().decode(tool, slot, pool, answers, rc)
        return self._decode_factorized(tool, slot, pool, answers, rc)

    def _decode_factorized(
        self, tool: ToolSpec, slot: SlotSpec, pool: Pool, answers: Mapping[str, Answer], rc: ResolveContext
    ) -> SlotResult:
        questions = {q.meta.get("part"): q for q in rc.slot_questions(tool, slot) if q.family in ("date", "time")}
        if set(questions) != {"date", "time"}:
            questions = {q.meta["part"]: q for q in self.questions(tool, slot, pool, rc)}
        oop = rc.policy.shapes.out_of_pool
        dq, tq = questions["date"], questions["time"]
        rd = decode_choice(slot, dq, answers.get(dq.qid), out_of_pool=oop)
        rt = decode_choice(slot, tq, answers.get(tq.qid), out_of_pool=oop)
        tz = str(pool.meta.get("tz") or rc.ctx.timezone_name)
        dist: dict[str, float] = {}
        values: dict[str, Any] = {}
        entries: dict[str, ValueEntry] = {}
        # Only offered part values combine (never a sentinel-decoded whole default, never a zero-mass key).
        days, clocks = _part_values(rd, dq), _part_values(rt, tq)
        for day, pd in days:
            for clock, pt in clocks:
                dt = localize(date.fromisoformat(day), time.fromisoformat(clock), tz)[0][0]
                value = normalize_temporal(dt, slot.json_schema)
                key = value_key(value)
                dist[key] = dist.get(key, 0.0) + pd * pt
                values[key] = value
                entries.setdefault(
                    key,
                    ValueEntry(
                        display=datetime_label(dt, tz),
                        channel=_least_trusted(rd, rt),
                        prov={"date": day, "time": clock},
                    ),
                )
        for bottom in (Bottom.MISSING, Bottom.UNCOVERED, Bottom.OMIT):
            mass = rd.mass(bottom) + rt.mass(bottom)
            if mass:
                dist[bottom.value] = mass
        self._part_defaults(tool, slot, rc, (rd, dq), (rt, tq), dist, values, entries)
        result = elect(
            path=slot.path,
            kind=slot.kind,
            stakes=slot.stakes,
            dist=dist,
            values=values,
            entries=entries,
            out_of_pool=oop,
            qids=rd.qids + rt.qids,
            sentinels={**rd.sentinels, **{f"time.{k}": v for k, v in rt.sentinels.items()}},
            notes=rd.notes + rt.notes,
        )
        return result.with_(normalizer=self.normalizer, parts={"date": rd, "time": rt})

    def _part_defaults(
        self,
        tool: ToolSpec,
        slot: SlotSpec,
        rc: ResolveContext,
        date_part: tuple[SlotResult, BallotQuestion],
        time_part: tuple[SlotResult, BallotQuestion],
        dist: dict[str, float],
        values: dict[str, Any],
        entries: dict[str, ValueEntry],
    ) -> None:
        """A slot default applies only when *both* parts are not stated (joint ``NOT_STATED`` mass); a date stated
        without a time (or the reverse) is incomplete, never combined with a piece of the default (⊥missing)."""
        default = resolve_default(tool, slot, rc.ctx)
        if default is None or default.omit:
            return
        (rd, dq), (rt, tq) = date_part, time_part
        ns_date, ns_time = rd.sentinels.get(NOT_STATED, 0.0), rt.sentinels.get(NOT_STATED, 0.0)
        joint = ns_date * ns_time
        real_date = sum(p for _, p in _part_values(rd, dq, None))
        real_time = sum(p for _, p in _part_values(rt, tq, None))
        partial = ns_date * real_time + real_date * ns_time
        if partial:
            dist[Bottom.MISSING.value] = dist.get(Bottom.MISSING.value, 0.0) + partial
        if not joint:
            return
        if default.late is not None:
            dist[LATE_DEFAULT] = dist.get(LATE_DEFAULT, 0.0) + joint
            entries.setdefault(LATE_DEFAULT, ValueEntry(display=default.display, late=default.late, p=joint))
            return
        key = value_key(default.value)
        dist[key] = dist.get(key, 0.0) + joint
        values[key] = default.value
        entry = ValueEntry(display=default.display, channel=default.channel, prov={"default": True}, p=joint)
        entries.setdefault(key, entry)

    # -- ranges -----------------------------------------------------------------------------------------------------

    def _range_role(self, tool: ToolSpec, slot: SlotSpec) -> str | None:
        coupling = slot.range or {}
        if coupling.get("min") == slot.name and coupling.get("max") in tool.slot_names:
            return "min"
        if coupling.get("max") == slot.name and coupling.get("min") in tool.slot_names:
            return "max"
        return None

    def _range_pool(self, tool: ToolSpec, slot: SlotSpec, rc: ResolveContext) -> Pool:
        assert slot.range is not None
        other = tool.slot(slot.range["max"])
        out: list[Candidate] = []
        for m in temporal_mentions(rc):
            spans: Sequence[RangeReading] = m.value.ranges
            for r in spans:
                if r.start is None:
                    continue
                pair = {
                    "min": normalize_temporal(r.start, slot.json_schema),
                    "max": normalize_temporal(r.end, other.json_schema) if r.end else None,
                }
                end = (
                    f" – {r.end:%H:%M}"
                    if r.end and r.end.date() == r.start.date()
                    else (f" – {r.end:%Y-%m-%d %H:%M}" if r.end else " onwards")
                )
                display = f"{WEEKDAY_ABBR[r.start.weekday()]} {r.start:%Y-%m-%d %H:%M}{end} ({r.tz})"
                out.append(
                    Candidate(
                        value=pair,
                        display=display,
                        text=reading_text(m, r.gloss),
                        channel=m.channel,
                        prov=mention_prov(m, reading=r.name),
                    )
                )
        pool = finalize_pool(tool, slot, rc, out, self.kind, validate=False, meta={"mode": "range", "max": other.name})
        return pool

    def _decode_range_max(
        self, tool: ToolSpec, slot: SlotSpec, answers: Mapping[str, Answer], rc: ResolveContext
    ) -> SlotResult:
        assert slot.range is not None
        minimum = tool.slot(slot.range["min"])
        question = rc.question(slot_qid(tool.id, minimum.qpath))
        if question is None:
            return unasked_result(slot, resolve_default(tool, slot, rc.ctx)).with_(normalizer=self.normalizer)
        result = decode_choice(minimum, question, answers.get(question.qid), out_of_pool=rc.policy.shapes.out_of_pool)
        return _component(result, "max", rc.policy.shapes.out_of_pool).with_(path=slot.path, normalizer=self.normalizer)


def _part_values(result: SlotResult, question: BallotQuestion, n: int | None = 3) -> list[tuple[str, float]]:
    """The ``n`` most probable offered values of a factorized part with positive mass (option values only: a
    ``NOT_STATED`` that decodes to the slot default is not a date or a time)."""
    offered = {value_key(o.value) for o in question.options}
    real = [(k, p) for k, p in result.dist.items() if k in offered and k in result.values and p > 0]
    real.sort(key=lambda kp: -kp[1])
    return [(str(result.values[k]), p) for k, p in real[:n]]


def _least_trusted(*results: SlotResult) -> Channel | None:
    channels = [r.channel for r in results if r.channel is not None]
    return least_trusted(*channels) if channels else None


def _component(result: SlotResult, part: str, out_of_pool: float) -> SlotResult:
    """Project a range-pair result onto its ``min``/``max`` component (masses of equal components pool)."""
    dist: dict[str, float] = {}
    values: dict[str, Any] = {}
    entries: dict[str, ValueEntry] = {}
    for key, p in result.dist.items():
        if key in result.values and isinstance(result.values[key], Mapping):
            value = result.values[key].get(part)
            if value is None:
                bottom = Bottom.OMIT.value
                dist[bottom] = dist.get(bottom, 0.0) + p
                continue
            sub = value_key(value)
            dist[sub] = dist.get(sub, 0.0) + p
            values[sub] = value
            entry = result.entries.get(key)
            if entry is not None and sub not in entries:
                entries[sub] = ValueEntry(
                    display=display_value(value), label=entry.label, channel=entry.channel, prov=entry.prov, p=p
                )
        else:
            dist[key] = dist.get(key, 0.0) + p
    best = elect(
        path=result.path,
        kind=result.kind,
        stakes=result.stakes,
        dist=dist,
        values=values,
        entries=entries,
        out_of_pool=out_of_pool,
        qids=result.qids,
        sentinels=result.sentinels,
        notes=result.notes,
    )
    return best.with_(shape=result.shape if result.shape != "ok" else best.shape)


register_resolver("temporal", TemporalResolver())

__all__ = ["MAX_SINGLE", "TemporalResolver", "date_label", "datetime_label", "time_label"]
