"""Ground an LLM agent's tool calls before they run.

An LLM agent plans; :class:`Guard` checks each call it proposes against the conversation before the call runs. Every
identity value of a write (an id, an address, an account, each field of an object, each number in a calculation) must
come from the user's turns or from a trusted tool's results; a call with a value that appears nowhere, such as one an
instruction planted in a tool result asks for or one the LLM made up, is blocked and answered with a tool error that
names the value, so the LLM can look it up or ask. Read tools pass unchecked. No model is called
(:meth:`jevtools.router.Router.check` with ``verify=False``).

On τ²-bench this guard kept gpt-4.1-mini's task success (31 of 60 against 31 and 34 alone) and stopped three payment
ids the LLM had made up (docs/BENCH.md).

OpenAI loop::

    guard = jt.Guard(tools)
    message = client.chat.completions.create(model=..., messages=messages, tools=tools).choices[0].message
    allowed, refusals = guard.screen(messages, message.tool_calls)
    # run `allowed`; append `refusals` (role "tool" messages) after the assistant message

LangGraph: :class:`jevtools.adapters.langchain.GuardedToolNode` in place of ``ToolNode``.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from jevtools.context import Context
from jevtools.decision import Decision
from jevtools.policy import Outcome, Tier
from jevtools.router import Router
from jevtools.spec.catalog import Catalog

BLOCKED_TEXT = (
    "Not executed: {detail}. Look the value up with a tool or ask the user, then call {tool} again with values from "
    "the conversation or a lookup."
)

__all__ = ["BLOCKED_TEXT", "Guard", "Screened", "Verdict"]


def _is_langchain_tool(tool: Any) -> bool:
    return hasattr(tool, "args_schema") and hasattr(tool, "invoke") and isinstance(getattr(tool, "name", None), str)


class _NoJev:
    """The backend of a guard that never verifies: a call to it is a bug."""

    name = "none"
    model = "none"

    def decide(self, request: Any) -> Any:
        raise RuntimeError("Guard(verify=False) does not call Jev")

    async def adecide(self, request: Any) -> Any:
        raise RuntimeError("Guard(verify=False) does not call Jev")


@dataclass(frozen=True)
class Verdict:
    """The guard's answer for one proposed call."""

    name: str
    arguments: dict[str, Any]
    allowed: bool
    reason: str | None = None
    """Why the call is blocked (the argument and value the conversation does not support)."""
    tool_call_id: str | None = None
    needs_confirmation: bool = False
    """A grounded call of a critical-tier tool (a refund, a payment): allowed, but it should run only after the user
    explicitly confirmed it, which the agent's own conversation shows."""
    decision: Decision | None = None
    """The check's decision (``None`` for a read tool, which passes unchecked)."""


@dataclass
class Screened:
    """:meth:`Guard.screen`'s result: the tool calls to run, the refusals to append, and every verdict."""

    allowed: list[Any] = field(default_factory=list)
    refusals: list[dict[str, Any]] = field(default_factory=list)
    """``{"role": "tool", "tool_call_id", "content"}`` messages answering the blocked calls."""
    verdicts: list[Verdict] = field(default_factory=list)

    def __iter__(self) -> Any:
        return iter((self.allowed, self.refusals))


def _get(obj: Any, key: str) -> Any:
    return obj.get(key) if isinstance(obj, Mapping) else getattr(obj, key, None)


def _parts(tool_call: Any) -> tuple[str | None, str, dict[str, Any]]:
    """``(id, name, arguments)`` of an OpenAI tool call (object or dict), a LangChain tool call dict or a
    ``{"name", "arguments"}`` mapping."""
    function = _get(tool_call, "function")
    if function is not None:
        name, raw = _get(function, "name"), _get(function, "arguments")
    else:
        raw = _get(tool_call, "arguments")
        name, raw = _get(tool_call, "name"), raw if raw is not None else _get(tool_call, "args")
    if isinstance(raw, str):
        try:
            raw = json.loads(raw) if raw.strip() else {}
        except ValueError:
            raw = {}
    return _get(tool_call, "id"), str(name), dict(raw or {})


class Guard:
    """Checks an LLM agent's proposed tool calls before they run (see the module docstring).

    - ``tools``: the agent's tools (OpenAI function tools, MCP tools, LangChain tools: anything a
      :class:`~jevtools.router.Router` accepts).
    - ``trusted_tools``: tools whose results are the app's own records, so values they return may be used in writes;
      default: the read-tier tools.
    - ``context``: a base :class:`~jevtools.context.Context` (user profile, sources, time).
    - ``backend`` / ``verify``: with a Jev backend and ``verify=True``, Jev also verifies each grounded value and the
      policy may ask for confirmation; on τ² that cost tasks, so the default only grounds.
    """

    def __init__(self, tools: Any, *, trusted_tools: Sequence[str] | None = None, context: Context | None = None,
                 backend: Any = None, verify: bool = False) -> None:  # fmt: skip
        if verify and backend is None:
            raise ValueError("Guard(verify=True) needs a Jev backend")
        if isinstance(tools, Sequence) and tools and all(_is_langchain_tool(t) for t in tools):
            tools = Catalog.from_langchain(tools, sources=list((context or Context()).sources.values()))
        self.router = Router(tools, backend=backend if backend is not None else _NoJev(), context=context)
        self.verify = verify
        self.reads = frozenset(t.name for t in self.router.catalog if t.tier is Tier.READ)
        self.trusted = tuple(trusted_tools) if trusted_tools is not None else tuple(sorted(self.reads))

    def _context(self, messages: Sequence[Mapping[str, Any]]) -> Context:
        ctx = self.router.context_for(list(messages), None)
        return ctx.model_copy(update={"trusted_tools": self.trusted})

    def check(self, messages: Sequence[Mapping[str, Any]], tool_call: Any) -> Verdict:
        """The verdict on one proposed call, given the conversation so far (OpenAI-format messages, tool results
        included)."""
        call_id, name, arguments = _parts(tool_call)
        if name in self.reads:
            return Verdict(name=name, arguments=arguments, allowed=True, tool_call_id=call_id)
        if name not in self.router.catalog:
            return Verdict(name=name, arguments=arguments, allowed=False, reason=f"{name} is not one of the tools",
                           tool_call_id=call_id)  # fmt: skip
        ctx = self._context(messages)
        decision = self.router.check(ctx.messages, {"name": name, "arguments": arguments}, context=ctx,
                                     mode="loop", verify=self.verify)  # fmt: skip
        confirm = not self.verify and decision.outcome is Outcome.CONFIRM and decision.rule.startswith("C2")
        allowed = decision.outcome is Outcome.EXECUTE or confirm
        reason = None
        if not allowed:
            slot = decision.bottleneck.slot if decision.bottleneck is not None else None
            if slot is not None and decision.rule.startswith("C1"):
                value = json.dumps(arguments.get(slot), default=str) if slot in arguments else "(missing)"
                reason = f"the value {value} for '{slot}' was neither given by the user nor returned by a lookup"
            elif decision.outcome is Outcome.CONFIRM:
                reason = "the call needs the user's confirmation first"
            else:
                reason = decision.prompt.text if decision.prompt and decision.prompt.text else decision.rule
        return Verdict(name=name, arguments=arguments, allowed=allowed, reason=reason, tool_call_id=call_id,
                       needs_confirmation=confirm, decision=decision)  # fmt: skip

    def screen(self, messages: Sequence[Mapping[str, Any]], tool_calls: Sequence[Any] | None) -> Screened:
        """Split proposed tool calls into those to run and refusals to append (one ``tool`` message per blocked call,
        answering its ``tool_call_id``). ``messages`` is the conversation before the assistant message that proposed
        them; iterating the result gives ``(allowed, refusals)``."""
        out = Screened()
        for tool_call in tool_calls or []:
            verdict = self.check(messages, tool_call)
            out.verdicts.append(verdict)
            if verdict.allowed:
                out.allowed.append(tool_call)
            else:
                out.refusals.append({"role": "tool", "tool_call_id": verdict.tool_call_id, "content":
                                     BLOCKED_TEXT.format(detail=verdict.reason, tool=verdict.name)})  # fmt: skip
        return out
