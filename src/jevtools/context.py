"""The host context of a decision and the Jev ``state`` built from it (spec §3.5.1, §6).

``Context`` bundles the conversation (``messages``), the clock, the user profile, registered sources, loop
observations and the entity store. :func:`build_state` renders the *data-only* state that every question of a round
refers to, in the normative key order ``request, history, now, user, system, progress, observations``.
"""

from __future__ import annotations

import json
import re
from collections.abc import Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone, tzinfo
from typing import Any, Literal
from zoneinfo import ZoneInfo

from pydantic import BaseModel, ConfigDict, Field, field_validator

from jevtools.canonical import canonical_str, jsonable, sha256_of
from jevtools.preview import PREVIEW_CHARS, text_preview

Role = Literal["user", "assistant", "system", "tool"]
Mode = Literal["turn", "loop", "widen", "resume", "fill"]
"""Round modes recorded in the Ballot (§3.5.7)."""

SETTING_FIELDS: tuple[str, ...] = ("now", "tz", "locale", "user", "shareable", "include_system")
"""The context *settings* a host document may give (``jevtools.toml`` ``[context]``, an evaluation context file,
a per-request override); messages, sources, observations and entities travel separately."""

Message = Mapping[str, Any]
"""An OpenAI-style chat message: ``{"role", "content", "tool_calls"?, "tool_call_id"?, "name"?}``."""

WEEKDAYS = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")


class Turn(BaseModel):
    """One conversation message, normalized from OpenAI-style dicts."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    role: Role
    text: str = ""
    """Plain text content (text parts joined by newlines)."""
    name: str | None = None
    """For ``tool`` messages: the tool name, when the message carries it."""
    tool_call_id: str | None = None
    """For ``tool`` messages: the id of the assistant tool call this result answers."""
    tool_calls: tuple[dict[str, Any], ...] = ()
    """For ``assistant`` messages: OpenAI-shaped tool calls ``{"id", "type", "function": {"name", "arguments"}}``."""
    content: Any = None
    """The raw content (kept for tool results, which may be structured)."""

    @classmethod
    def from_message(cls, message: Message | Turn) -> Turn:
        """Normalize an OpenAI-style message (content may be a string, ``None`` or a list of parts)."""
        if isinstance(message, Turn):
            return message
        role = message.get("role", "user")
        if role == "developer":
            role = "system"
        content = message.get("content")
        return cls(
            role=role,
            text=_text_of(content),
            name=message.get("name"),
            tool_call_id=message.get("tool_call_id"),
            tool_calls=tuple(message.get("tool_calls") or ()),
            content=content,
        )

    def to_state(self) -> dict[str, str]:
        """The ``history`` entry: ``{"role", "text"}``."""
        return {"role": self.role, "text": self.text}


def _text_of(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [p.get("text", "") for p in content if isinstance(p, Mapping) and p.get("type", "text") == "text"]
        return "\n".join(part for part in parts if part)
    return canonical_str(content)


def parse_messages(messages: str | Turn | Iterable[Message | Turn]) -> list[Turn]:
    """Accept a bare string (one user turn), a :class:`Turn` or a sequence of OpenAI-style messages/turns."""
    if isinstance(messages, str):
        return [Turn(role="user", text=messages, content=messages)]
    if isinstance(messages, Turn):
        return [messages]
    return [Turn.from_message(m) for m in messages]


class Observation(BaseModel):
    """A tool result in loop mode (spec §6.3). Its full ``content`` never reaches Jev; only ``preview`` does."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    step: int
    tool: str
    status: Literal["ok", "error"] = "ok"
    content: Any = None
    """The full result (text or JSON). Late-bound into arguments by code; never sent to Jev."""
    preview: str | None = None
    """What ``state.observations`` shows (BM25-selected chunks, §6.2); ``None`` → the content's chunks, in order,
    up to :data:`~jevtools.preview.PREVIEW_CHARS` (no request to rank them against)."""
    arguments: dict[str, Any] = Field(default_factory=dict)
    """Arguments of the call that produced this result (rendered in ``progress``)."""
    summary: str | None = None
    """Short progress summary (``"1,412 words"``); defaults to a word count for text."""
    call_id: str | None = None

    def preview_text(self) -> str:
        """The preview shown to Jev."""
        if self.preview is not None:
            return self.preview
        text = self.content if isinstance(self.content, str) else canonical_str(jsonable(self.content))
        return text_preview(text, "", PREVIEW_CHARS)

    def to_state(self) -> dict[str, Any]:
        """The ``observations`` entry: ``{"step", "tool", "status", "preview"}``."""
        return {"step": self.step, "tool": self.tool, "status": self.status, "preview": self.preview_text()}

    def progress_line(self) -> str:
        """``Step k: tool(arg="…") → status, summary``."""
        args = ", ".join(f"{k}={json.dumps(v, ensure_ascii=False)}" for k, v in self.arguments.items())
        summary = self.summary
        if summary is None and isinstance(self.content, str):
            summary = f"{len(self.content.split()):,} words"
        tail = f", {summary}" if summary else ""
        return f"Step {self.step}: {self.tool}({args}) → {self.status}{tail}"


def zone_of(name: str) -> tzinfo:
    """A tzinfo for an IANA name or a ``UTC±HH:MM`` fixed offset (what :attr:`Context.timezone_name` gives for a
    ``now`` that carries only a UTC offset, e.g. an ISO timestamp ``…+02:00`` without ``tz``)."""
    match = re.fullmatch(r"UTC([+-])(\d{2}):(\d{2})", name)
    if match:
        sign = 1 if match.group(1) == "+" else -1
        return timezone(sign * timedelta(hours=int(match.group(2)), minutes=int(match.group(3))))
    return ZoneInfo(name)


@dataclass(frozen=True)
class Clock:
    """Source of ``now`` in an IANA time zone; ``fixed`` pins it (tests, replays)."""

    tz: str = "UTC"
    fixed: datetime | None = None

    def now(self) -> datetime:
        """The current time as an aware datetime in ``tz``."""
        zone = zone_of(self.tz)
        if self.fixed is None:
            return datetime.now(zone)
        fixed = self.fixed if self.fixed.tzinfo is not None else self.fixed.replace(tzinfo=zone)
        return fixed.astimezone(zone)


def _offset(dt: datetime) -> str:
    delta = dt.utcoffset() or timedelta(0)
    sign = "+" if delta >= timedelta(0) else "-"
    minutes = abs(int(delta.total_seconds())) // 60
    return f"UTC{sign}{minutes // 60:02d}:{minutes % 60:02d}"


def render_now(dt: datetime, tz: str | None = None) -> str:
    """``{Weekday} {YYYY-MM-DD} {HH:MM} {IANA tz} (UTC{±HH:MM})`` (spec §3.5.1).

    ``tz`` converts first (naive datetimes are taken to be in ``tz``); without it the datetime's own zone name is used.
    """
    if tz is not None:
        zone = zone_of(tz)
        dt = dt.replace(tzinfo=zone) if dt.tzinfo is None else dt.astimezone(zone)
    if dt.tzinfo is None:
        raise ValueError("render_now() needs an aware datetime or a tz")
    name = tz or _zone_name(dt.tzinfo, dt)
    return f"{WEEKDAYS[dt.weekday()]} {dt:%Y-%m-%d} {dt:%H:%M} {name} ({_offset(dt)})"


def _zone_name(zone: tzinfo, dt: datetime) -> str:
    key = getattr(zone, "key", None)
    if isinstance(key, str):
        return key
    if zone is timezone.utc:
        return "UTC"
    return dt.tzname() or "UTC"


class Context(BaseModel):
    """Everything the host knows about a decision (spec §3.1 ``Context``).

    - ``messages``: the conversation; a bare string is one user turn. ``role: tool`` messages become observations.
    - ``now``/``tz``/``clock``: the time reference. ``now`` wins over ``clock``; ``tz`` names the IANA zone.
    - ``user``: the user profile. ``shareable`` lists the fields that may be sent to Jev (``None`` = all but
      secrets: the planner never sends a field a ``secret`` slot reads or a field with a secret name).
      ``default_from: "user.*"`` may read every field, shareable or not.
    - ``sources``: registered candidate sources by name (``jevtools.sources.Source`` objects).
    - ``observations``: loop observations (§6.3), in step order.
    - ``entities``: the entity store (``jevtools.loop.EntityStore``) or ``None``.
    - ``include_system``: put system messages into ``state.system``.
    """

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    messages: list[Turn] = Field(default_factory=list)
    now: datetime | None = None
    tz: str | None = None
    locale: str = "en"
    user: dict[str, Any] = Field(default_factory=dict)
    shareable: tuple[str, ...] | None = None
    sources: dict[str, Any] = Field(default_factory=dict)
    observations: list[Observation] = Field(default_factory=list)
    entities: Any = None
    include_system: bool = False
    clock: Clock | None = None

    @field_validator("messages", mode="before")
    @classmethod
    def _parse_messages(cls, value: Any) -> list[Turn]:
        return parse_messages(value if value is not None else [])

    @field_validator("sources", mode="before")
    @classmethod
    def _index_sources(cls, value: Any) -> dict[str, Any]:
        if value is None:
            return {}
        if isinstance(value, Mapping):
            return dict(value)
        return {source.name: source for source in value}

    def with_messages(self, messages: str | Iterable[Message | Turn]) -> Context:
        """A copy with the conversation replaced."""
        return self.model_copy(update={"messages": parse_messages(messages)})

    # -- time ---------------------------------------------------------------------------------------------------

    @property
    def timezone_name(self) -> str:
        """The IANA zone: ``tz``, else the zone of ``now``, else the clock's, else ``UTC``."""
        if self.tz:
            return self.tz
        if self.now is not None and self.now.tzinfo is not None:
            return _zone_name(self.now.tzinfo, self.now)
        return self.clock.tz if self.clock is not None else "UTC"

    def current_time(self) -> datetime:
        """``now`` as an aware datetime in :attr:`timezone_name`."""
        zone_name = self.timezone_name
        if self.now is not None:
            return Clock(zone_name, self.now).now()
        clock = self.clock or Clock(zone_name)
        return clock.now().astimezone(zone_of(zone_name))

    # -- conversation split -------------------------------------------------------------------------------------

    @property
    def request_index(self) -> int | None:
        """Index of the latest user turn in :attr:`messages`."""
        for i in range(len(self.messages) - 1, -1, -1):
            if self.messages[i].role == "user":
                return i
        return None

    @property
    def request(self) -> str:
        """Text of the latest user message (``""`` if there is none)."""
        index = self.request_index
        return self.messages[index].text if index is not None else ""

    @property
    def history(self) -> list[Turn]:
        """User and assistant turns with text, oldest first, excluding the current request (§3.5.1)."""
        index = self.request_index
        return [
            turn
            for i, turn in enumerate(self.messages)
            if i != index and turn.role in ("user", "assistant") and turn.text
        ]

    @property
    def user_turns(self) -> list[Turn]:
        """The user's own turns including the request (the ``user`` channel's text)."""
        return [turn for turn in self.messages if turn.role == "user"]

    @property
    def system(self) -> str | None:
        """System messages joined, if any."""
        texts = [turn.text for turn in self.messages if turn.role == "system" and turn.text]
        return "\n\n".join(texts) if texts else None

    def all_observations(self) -> list[Observation]:
        """Declared observations followed by observations parsed from ``role: tool`` messages (drop-in loops),
        whose previews are the chunks BM25 ranks highest against the request (§6.2, as for ingested results)."""
        observations = list(self.observations)
        calls = _tool_calls_by_id(self.messages)
        step = max((o.step for o in observations), default=0)
        request = self.request
        for turn in self.messages:
            if turn.role != "tool":
                continue
            step += 1
            name, arguments = calls.get(turn.tool_call_id or "", (turn.name or "tool", {}))
            content = turn.content if turn.content is not None else turn.text
            text = content if isinstance(content, str) else canonical_str(jsonable(content))
            observations.append(
                Observation(
                    step=step,
                    tool=turn.name or name,
                    content=content,
                    preview=text_preview(text, request, PREVIEW_CHARS),
                    arguments=arguments,
                    call_id=turn.tool_call_id,
                )
            )
        return observations

    # -- user profile -------------------------------------------------------------------------------------------

    def user_state(self, exclude: Collection[str] = ()) -> dict[str, Any]:
        """The shareable profile fields, in profile order, without the ``exclude``d ones (secrets, shareable or not:
        §14 — secrets are never sent)."""
        return {k: v for k, v in self.user.items()
                if (self.shareable is None or k in self.shareable) and k not in exclude}  # fmt: skip

    def lookup(self, path: str) -> Any:
        """Resolve a context path such as ``user.home_city`` (roots: ``user``, ``locale``, ``tz``, ``now``).

        Raises ``KeyError`` if the path does not resolve.
        """
        root, _, rest = path.partition(".")
        value: Any
        if root == "user":
            value = self.user
        elif root == "locale":
            value = self.locale
        elif root == "tz":
            value = self.timezone_name
        elif root == "now":
            value = self.current_time()
        else:
            raise KeyError(path)
        for part in rest.split(".") if rest else ():
            if not isinstance(value, Mapping) or part not in value:
                raise KeyError(path)
            value = value[part]
        return value

    # -- documents ----------------------------------------------------------------------------------------------

    def source_hashes(self) -> dict[str, str | None]:
        """``{name: content hash}`` of the registered sources (a source may expose ``content_sha256()``)."""
        hashes: dict[str, str | None] = {}
        for name, source in self.sources.items():
            digest = getattr(source, "content_sha256", None)
            hashes[name] = digest() if callable(digest) else digest
        return hashes

    def to_doc(self) -> dict[str, Any]:
        """The Context document (spec §3.1); sources are referenced by name plus content hash."""
        entities = self.entities
        to_json = getattr(entities, "to_json", None)
        return {
            "messages": [turn.model_dump(mode="json") for turn in self.messages],
            "now": self.current_time().isoformat() if self.now is not None else None,
            "tz": self.timezone_name,
            "locale": self.locale,
            "user": jsonable(self.user),
            "shareable": list(self.shareable) if self.shareable is not None else None,
            "sources": self.source_hashes(),
            "observations": [jsonable(o) for o in self.observations],
            "entities": to_json() if callable(to_json) else None,
            "include_system": self.include_system,
        }

    @property
    def sha256(self) -> str:
        """``context_sha256``: digest of :meth:`to_doc`."""
        return sha256_of(self.to_doc())

    @property
    def requester_sha256(self) -> str:
        """Who a decision is for: the user profile, what of it is shareable, ``include_system`` and the names of the
        registered sources. The clock, the messages, time zone and source *contents* are left out, so a resume after
        a registry change (the TOCTOU case, §3.8.5) still belongs to the same requester."""
        return sha256_of({"user": jsonable(self.user), "include_system": self.include_system,
                          "shareable": list(self.shareable) if self.shareable is not None else None,
                          "sources": sorted(self.sources)})  # fmt: skip


def _tool_calls_by_id(turns: Sequence[Turn]) -> dict[str, tuple[str, dict[str, Any]]]:
    calls: dict[str, tuple[str, dict[str, Any]]] = {}
    for turn in turns:
        for call in turn.tool_calls:
            function = call.get("function") or {}
            raw = function.get("arguments") or "{}"
            try:
                arguments = json.loads(raw) if isinstance(raw, str) else dict(raw)
            except (ValueError, TypeError):
                arguments = {}
            calls[str(call.get("id", ""))] = (str(function.get("name", "tool")), arguments)
    return calls


def is_loop(ctx: Context, mode: str = "turn") -> bool:
    """Loop sections are present in loop mode and whenever the context carries observations."""
    return mode == "loop" or bool(ctx.all_observations())


def build_state(ctx: Context, mode: str = "turn", *, secret_fields: Collection[str] = ()) -> dict[str, Any]:
    """The data-only Jev state in normative key order (spec §3.5.1).

    ``request``, ``history`` (always present), ``now``, ``user`` (omitted when empty; never the ``secret_fields``),
    ``system`` (only with ``include_system``), then ``progress`` and ``observations`` in loop mode. Widen/fill/resume
    rounds of a loop step see the same state because loop sections follow the observations, not the mode alone.
    """
    state: dict[str, Any] = {
        "request": ctx.request,
        "history": [turn.to_state() for turn in ctx.history],
        "now": render_now(ctx.current_time(), ctx.timezone_name),
    }
    user = ctx.user_state(exclude=secret_fields)
    if user:
        state["user"] = jsonable(user)
    if ctx.include_system and ctx.system:
        state["system"] = ctx.system
    if is_loop(ctx, mode):
        observations = ctx.all_observations()
        state["progress"] = [o.progress_line() for o in observations]
        state["observations"] = [o.to_state() for o in observations]
    return state


__all__ = [
    "SETTING_FIELDS",
    "Clock",
    "Context",
    "Message",
    "Mode",
    "Observation",
    "Role",
    "Turn",
    "build_state",
    "is_loop",
    "parse_messages",
    "render_now",
]
