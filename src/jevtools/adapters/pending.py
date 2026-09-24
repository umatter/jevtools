"""Pending CONFIRM/CLARIFY handles for stateless clients (spec §7.2.1, §7.2.4).

A drop-in client (OpenAI SDK, LangChain, Pydantic AI, the ``jevtools serve`` proxy) sends the whole conversation on
every request and knows nothing about :class:`~jevtools.decision.Pending`. When a decision ends in ``confirm`` or
``clarify``, the adapter stores its pending handle in a :class:`PendingStore` under **two** keys:

1. the ``pending_id`` (``x_jev.pending_id`` on the assistant message, for clients that echo it);
2. the **prefix hash** ``sha256(canonical([scope, *messages[0..k]]))``: the conversation up to and including the
   assistant prompt message ``k``. Clients echo the assistant text verbatim, so the next request's prefix reproduces
   the key without any visible marker.

Both keys are bound to the **scope** of the request that stored the handle (:func:`pending_scope`: the requester's
user, zone, locale, shareable fields, system-message setting, source contents and tool names). A request from
another scope never resumes the handle, even with an identical conversation and card text or an echoed
``pending_id``: it is a miss, and the handle stays for its own requester.

On the next request, :func:`match_pending` looks for trailing ``user`` messages after an assistant message, and
resumes (:meth:`Router.resume <jevtools.router.Router.resume>` with the reply) when either key hits. A miss (expiry,
restart, another process) simply compiles a fresh turn: the history then holds the card and the reply as
``user``-channel evidence, so the result is still correct at the cost of one round.

Messages are hashed in a normalized form (``role``, text ``content``, tool calls with parsed arguments,
``tool_call_id``) so that extra fields a client adds or drops (``x_jev``, ``refusal: null``, content-part lists)
never change the key.
"""

from __future__ import annotations

import json
import threading
import weakref
from collections import OrderedDict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Literal, Protocol, runtime_checkable

from jevtools.canonical import jsonable, sha256_of
from jevtools.context import Context, Turn
from jevtools.decision import PENDING_TTL, Decision, Pending
from jevtools.plan import parse_tool_choice
from jevtools.router import Router

MessageLike = Mapping[str, Any] | Turn
"""An OpenAI-style message dict (``role``, ``content``, ``tool_calls``, ``tool_call_id``, ``x_jev``) or a Turn."""


# --------------------------------------------------------------------------------------------------------------------
# Store
# --------------------------------------------------------------------------------------------------------------------


@runtime_checkable
class PendingStore(Protocol):
    """Where pending CONFIRM/CLARIFY handles live between two stateless requests (spec §7.2.4).

    Implement ``get``/``put``/``delete`` over a shared store (Redis, a database) to run several proxy workers; a
    :class:`~jevtools.decision.Pending` round-trips through ``pending.model_dump_json()`` /
    ``Pending.model_validate_json(...)``. ``get`` returns ``None`` for unknown or expired keys.
    """

    def get(self, key: str) -> Pending | None: ...

    def put(self, key: str, pending: Pending) -> None: ...

    def delete(self, key: str) -> None: ...


class InMemoryPendingStore:
    """The default :class:`PendingStore`: a thread-safe in-process dict with a TTL (15 min) and a size bound.

    An entry expires at the earlier of the handle's own ``expires_at`` and ``put`` time + ``ttl``. When more than
    ``max_entries`` keys are held, the oldest are evicted first.
    """

    def __init__(
        self,
        *,
        ttl: timedelta = PENDING_TTL,
        max_entries: int = 10_000,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self.ttl = ttl
        self.max_entries = max_entries
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self._items: OrderedDict[str, tuple[Pending, datetime]] = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key: str) -> Pending | None:
        """The pending handle stored under ``key``, or ``None`` (unknown or expired; expired entries are dropped)."""
        with self._lock:
            item = self._items.get(key)
            if item is None:
                return None
            pending, deadline = item
            if self._clock() >= deadline:
                del self._items[key]
                return None
            return pending

    def put(self, key: str, pending: Pending) -> None:
        """Store ``pending`` under ``key`` (replacing any previous entry)."""
        deadline = min(pending.expires_at, self._clock() + self.ttl)
        with self._lock:
            self._items.pop(key, None)
            self._items[key] = (pending, deadline)
            while len(self._items) > self.max_entries:
                self._items.popitem(last=False)

    def delete(self, key: str) -> None:
        """Forget ``key`` (no error when absent)."""
        with self._lock:
            self._items.pop(key, None)

    def __len__(self) -> int:
        with self._lock:
            return len(self._items)

    def __contains__(self, key: object) -> bool:
        return isinstance(key, str) and self.get(key) is not None


_STORES: weakref.WeakKeyDictionary[Router, InMemoryPendingStore] = weakref.WeakKeyDictionary()
_STORES_LOCK = threading.Lock()


def default_store(router: Router) -> InMemoryPendingStore:
    """The in-memory store kept per router, used by the adapters when the host passes none."""
    with _STORES_LOCK:
        store = _STORES.get(router)
        if store is None:
            store = _STORES[router] = InMemoryPendingStore()
        return store


# --------------------------------------------------------------------------------------------------------------------
# Prefix key
# --------------------------------------------------------------------------------------------------------------------


def _role(message: MessageLike) -> str:
    if isinstance(message, Turn):
        return message.role
    role = str(message.get("role", "user"))
    return "system" if role == "developer" else role


def _text(message: MessageLike) -> str:
    if isinstance(message, Turn):
        return message.text
    return Turn.from_message({"role": "user", "content": message.get("content")}).text


def _arguments(raw: Any) -> Any:
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except ValueError:
            return raw
    return raw


def _call(call: Mapping[str, Any]) -> dict[str, Any]:
    function = call.get("function") or {}
    return {"id": call.get("id"), "name": function.get("name"), "arguments": _arguments(function.get("arguments"))}


def normalize_message(message: MessageLike) -> dict[str, Any]:
    """The form a message is hashed in: ``{"role", "content"}`` (text) plus ``tool_calls`` (parsed arguments)
    and ``tool_call_id`` when present. Client-side extras (``x_jev``, ``refusal``, ``name``…) are ignored."""
    doc: dict[str, Any] = {"role": _role(message), "content": _text(message)}
    if isinstance(message, Turn):
        calls: Sequence[Mapping[str, Any]] = message.tool_calls
        call_id = message.tool_call_id
    else:
        calls = message.get("tool_calls") or ()
        call_id = message.get("tool_call_id")
    if calls:
        doc["tool_calls"] = [_call(c) for c in calls]
    if call_id:
        doc["tool_call_id"] = str(call_id)
    return doc


def prefix_key(messages: Sequence[MessageLike], scope: str | None = None) -> str:
    """``sha256(canonical(messages[0..k]))`` over the normalized messages (spec §7.2.4), with the requester's
    :func:`pending_scope` mixed in front when given (``sha256(canonical([scope, *messages]))``)."""
    normalized = [normalize_message(m) for m in messages]
    return sha256_of(normalized if scope is None else [scope, *normalized])


SCOPE_STATE = "adapter_scope"
"""``Pending.state`` key of the scope a stored handle belongs to (written by :func:`remember`)."""


def _source_identity(source: Any) -> str | None:
    """The content hash of a source whose rows are given (registries, files); ``None`` for a source that fetches
    its rows lazily (``ToolSource``, MCP resources): hashing it would fetch them (a tool call, which an async
    caller cannot even make here) and its hash changes with every refresh."""
    if callable(getattr(source, "refresh", None)):
        return None
    digest = getattr(source, "content_sha256", None)
    value = digest() if callable(digest) else digest
    return value if isinstance(value, str) else None


def pending_scope(router: Router, context: Context | None = None) -> str:
    """The identity of a request a pending handle belongs to: the effective context's settings that identify the
    requester (``user``, ``tz``, ``locale``, ``shareable``, ``include_system``), its sources (names, and the content
    hashes of the sources whose rows are given) and the router's tool names. ``now`` and the messages are left out
    (a wall-clock ``now`` would make every lookup miss; the messages are the prefix key itself)."""
    ctx = context if context is not None else router.context
    return sha256_of({
        "user": jsonable(ctx.user), "tz": ctx.tz, "locale": ctx.locale, "include_system": ctx.include_system,
        "shareable": list(ctx.shareable) if ctx.shareable is not None else None,
        "sources": {name: _source_identity(source) for name, source in sorted(ctx.sources.items())},
        "tools": sorted(router.catalog.names),
    })  # fmt: skip


def _x_jev_pending_id(message: MessageLike) -> str | None:
    if isinstance(message, Turn):
        return None
    x = message.get("x_jev")
    if isinstance(x, Mapping) and isinstance(x.get("pending_id"), str):
        return str(x["pending_id"])
    return None


# --------------------------------------------------------------------------------------------------------------------
# Matching
# --------------------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class PendingMatch:
    """A request that answers a stored prompt: the handle, the user's reply and how it was found."""

    pending: Pending
    reply: str
    via: Literal["pending_id", "prefix"]
    prefix: str
    """The prefix key of ``messages[0..k]`` (deleted together with the pending id once resumed)."""


def split_reply(messages: Sequence[MessageLike]) -> tuple[int, str] | None:
    """``(k, reply)`` when the conversation ends with user message(s) right after an assistant message ``k``."""
    i = len(messages)
    while i > 0 and _role(messages[i - 1]) == "user":
        i -= 1
    if i == len(messages) or i == 0 or _role(messages[i - 1]) != "assistant":
        return None
    reply = "\n".join(t for t in (_text(m).strip() for m in messages[i:]) if t)
    return i - 1, reply


def match_pending(
    store: PendingStore,
    messages: Sequence[MessageLike],
    *,
    pending_id: str | None = None,
    now: datetime | None = None,
    scope: str | None = None,
) -> PendingMatch | None:
    """Find the pending prompt that ``messages`` answer: by an explicit ``pending_id``, by ``x_jev.pending_id`` on
    the assistant message, then by the prefix hash (spec §7.2.4). ``None`` on a miss, when the handle expired, or
    when it was stored for another ``scope`` (:func:`pending_scope`; the other requester's entry is kept)."""
    split = split_reply(messages)
    if split is None:
        return None
    k, reply = split
    if not reply:
        return None
    prefix = prefix_key(messages[: k + 1], scope)
    candidates: list[tuple[str, Literal["pending_id", "prefix"]]] = []
    for pid in (pending_id, _x_jev_pending_id(messages[k])):
        if pid:
            candidates.append((pid, "pending_id"))
    candidates.append((prefix, "prefix"))
    for key, via in candidates:
        pending = store.get(key)
        if pending is None or pending.expired(now) or pending.state.get(SCOPE_STATE) != scope:
            continue
        return PendingMatch(pending=pending, reply=reply, via=via, prefix=prefix)
    return None


def remember(
    store: PendingStore, messages: Sequence[MessageLike], decision: Decision, *, scope: str | None = None
) -> list[str]:
    """Store the decision's pending handle (if any) under its id and under the prefix key of ``messages`` plus the
    assistant message the client will echo, bound to ``scope`` (:func:`pending_scope`). Returns the keys written."""
    pending = decision.pending
    if pending is None:
        return []
    if scope is not None:
        pending = pending.model_copy(update={"state": {**pending.state, SCOPE_STATE: scope}})
    keys = [pending.pending_id, prefix_key([*messages, decision.to_openai_message()], scope)]
    for key in keys:
        store.put(key, pending)
    return keys


def forget(store: PendingStore, match: PendingMatch) -> None:
    """Delete a consumed handle (both keys)."""
    store.delete(match.pending.pending_id)
    store.delete(match.prefix)


# --------------------------------------------------------------------------------------------------------------------
# One conversation turn (decide, or resume a pending prompt)
# --------------------------------------------------------------------------------------------------------------------


def _resumable(router: Router, pending: Pending, tool_choice: Any) -> bool:
    """Whether the request's ``tool_choice`` lets a matched prompt resume (§7.2.2): never under ``none`` (no tool
    call may come back), and under a named choice only when it names the pending tool; ``auto``/``required``
    resume. An invalid ``tool_choice`` raises ``ValueError`` as on a fresh decide."""
    mode, name = parse_tool_choice(tool_choice, router.catalog)
    if mode == "none":
        return False
    if mode == "named":
        tool = pending.state.get("tool") or (pending.call.name if pending.call is not None else None)
        return tool == name
    return True


def _match(
    router: Router,
    messages: Sequence[MessageLike],
    tool_choice: Any,
    store: PendingStore | None,
    pending_id: str | None,
    scope: str | None,
) -> PendingMatch | None:
    if store is None:
        return None
    match = match_pending(store, messages, pending_id=pending_id, scope=scope)
    if match is None or not _resumable(router, match.pending, tool_choice):
        return None  # a declined resume leaves the handle for a later request that allows it
    return match


def decide_turn(
    router: Router,
    messages: Sequence[MessageLike],
    *,
    context: Context | None = None,
    tool_choice: Any = "auto",
    parallel_tool_calls: bool = False,
    store: PendingStore | None = None,
    pending_id: str | None = None,
    scope: str | None = None,
) -> Decision:
    """Decide the next action for a stateless client's conversation.

    If ``messages`` answer a stored CONFIRM/CLARIFY (:func:`match_pending`) of the same requester (``scope``,
    default :func:`pending_scope` of ``router`` and ``context``) and ``tool_choice`` allows it (not ``none``; a
    named choice must name the pending tool), the reply resumes it (a click when it equals an option number or
    text, ``yes``/``ok``/``send`` or ``cancel``; else a free-text resume round); otherwise :meth:`Router.decide`
    runs on the whole conversation with ``tool_choice``. A new pending handle is remembered under ``scope``.
    """
    if store is not None and scope is None:
        scope = pending_scope(router, context)
    match = _match(router, messages, tool_choice, store, pending_id, scope)
    if match is not None:
        assert store is not None
        decision = router.resume(match.pending, reply=match.reply, context=context)
        forget(store, match)
    else:
        decision = router.decide(list(messages), context=context, tool_choice=tool_choice,
                                  parallel_tool_calls=parallel_tool_calls)  # fmt: skip
    if store is not None:
        remember(store, messages, decision, scope=scope)
    return decision


async def adecide_turn(
    router: Router,
    messages: Sequence[MessageLike],
    *,
    context: Context | None = None,
    tool_choice: Any = "auto",
    parallel_tool_calls: bool = False,
    store: PendingStore | None = None,
    pending_id: str | None = None,
    scope: str | None = None,
) -> Decision:
    """Async :func:`decide_turn`."""
    if store is not None and scope is None:
        scope = pending_scope(router, context)
    match = _match(router, messages, tool_choice, store, pending_id, scope)
    if match is not None:
        assert store is not None
        decision = await router.aresume(match.pending, reply=match.reply, context=context)
        forget(store, match)
    else:
        decision = await router.adecide(list(messages), context=context, tool_choice=tool_choice,
                                        parallel_tool_calls=parallel_tool_calls)  # fmt: skip
    if store is not None:
        remember(store, messages, decision, scope=scope)
    return decision


__all__ = [
    "SCOPE_STATE",
    "InMemoryPendingStore",
    "MessageLike",
    "PendingMatch",
    "PendingStore",
    "adecide_turn",
    "decide_turn",
    "default_store",
    "forget",
    "match_pending",
    "normalize_message",
    "pending_scope",
    "prefix_key",
    "remember",
    "split_reply",
]
