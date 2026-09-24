"""Pending CONFIRM/CLARIFY handles for stateless clients (spec §7.2.1, §7.2.4).

A drop-in client (OpenAI SDK, LangChain, Pydantic AI, the ``jevtools serve`` proxy) sends the whole conversation on
every request and knows nothing about :class:`~jevtools.decision.Pending`. When a decision ends in ``confirm`` or
``clarify``, the adapter stores its pending handle in a :class:`PendingStore` under **two** keys:

1. the ``pending_id`` (``x_jev.pending_id`` on the assistant message, for clients that echo it);
2. the **prefix hash** ``sha256(canonical(messages[0..k]))``: the conversation up to and including the assistant
   prompt message ``k``. Clients echo the assistant text verbatim, so the next request's prefix reproduces the key
   without any visible marker.

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

from jevtools.canonical import sha256_of
from jevtools.context import Context, Turn
from jevtools.decision import PENDING_TTL, Decision, Pending
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


def prefix_key(messages: Sequence[MessageLike]) -> str:
    """``sha256(canonical(messages[0..k]))`` over the normalized messages (spec §7.2.4)."""
    return sha256_of([normalize_message(m) for m in messages])


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
) -> PendingMatch | None:
    """Find the pending prompt that ``messages`` answer: by an explicit ``pending_id``, by ``x_jev.pending_id`` on
    the assistant message, then by the prefix hash (spec §7.2.4). ``None`` on a miss or when the handle expired."""
    split = split_reply(messages)
    if split is None:
        return None
    k, reply = split
    if not reply:
        return None
    prefix = prefix_key(messages[: k + 1])
    candidates: list[tuple[str, Literal["pending_id", "prefix"]]] = []
    for pid in (pending_id, _x_jev_pending_id(messages[k])):
        if pid:
            candidates.append((pid, "pending_id"))
    candidates.append((prefix, "prefix"))
    for key, via in candidates:
        pending = store.get(key)
        if pending is not None and not pending.expired(now):
            return PendingMatch(pending=pending, reply=reply, via=via, prefix=prefix)
    return None


def remember(store: PendingStore, messages: Sequence[MessageLike], decision: Decision) -> list[str]:
    """Store the decision's pending handle (if any) under its id and under the prefix key of ``messages`` plus the
    assistant message the client will echo. Returns the keys written."""
    pending = decision.pending
    if pending is None:
        return []
    keys = [pending.pending_id, prefix_key([*messages, decision.to_openai_message()])]
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


def decide_turn(
    router: Router,
    messages: Sequence[MessageLike],
    *,
    context: Context | None = None,
    tool_choice: Any = "auto",
    parallel_tool_calls: bool = False,
    store: PendingStore | None = None,
    pending_id: str | None = None,
) -> Decision:
    """Decide the next action for a stateless client's conversation.

    If ``messages`` answer a stored CONFIRM/CLARIFY (:func:`match_pending`), the reply resumes it (a click when it
    equals an option number or text, ``yes``/``ok``/``send`` or ``cancel``; else a free-text resume round);
    otherwise :meth:`Router.decide` runs on the whole conversation. A new pending handle is remembered.
    """
    match = match_pending(store, messages, pending_id=pending_id) if store is not None else None
    if match is not None:
        assert store is not None
        decision = router.resume(match.pending, reply=match.reply, context=context)
        forget(store, match)
    else:
        decision = router.decide(list(messages), context=context, tool_choice=tool_choice,
                                  parallel_tool_calls=parallel_tool_calls)  # fmt: skip
    if store is not None:
        remember(store, messages, decision)
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
) -> Decision:
    """Async :func:`decide_turn`."""
    match = match_pending(store, messages, pending_id=pending_id) if store is not None else None
    if match is not None:
        assert store is not None
        decision = await router.aresume(match.pending, reply=match.reply, context=context)
        forget(store, match)
    else:
        decision = await router.adecide(list(messages), context=context, tool_choice=tool_choice,
                                        parallel_tool_calls=parallel_tool_calls)  # fmt: skip
    if store is not None:
        remember(store, messages, decision)
    return decision


__all__ = [
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
    "prefix_key",
    "remember",
    "split_reply",
]
