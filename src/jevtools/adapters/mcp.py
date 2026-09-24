"""MCP interop (spec §7.1): the MCP client still executes; jevtools decides.

Ingest lives in :meth:`Catalog.from_mcp <jevtools.spec.catalog.Catalog.from_mcp>` (a ``tools/list`` result as JSON
or ``mcp`` objects; only *explicitly present* annotation hints count, §3.3.2) and
:meth:`Catalog.afrom_mcp_session <jevtools.spec.catalog.Catalog.afrom_mcp_session>`. This module emits:

- :func:`call_decision`: ``await session.call_tool(name, arguments, meta={"jevtools/idempotency_key": …})`` for an
  executed decision (duck-typed over any object shaped like ``mcp.ClientSession``; no ``mcp`` dependency);
- :func:`result_content` / :func:`to_observation`: a ``CallToolResult`` as the content of a loop observation
  (§6.3, through :func:`jevtools.loop.ingest_observation`), so the next decision can bind values found in it
  (``tool_output`` channel). The Agent's MCP executor path shares the same helpers and ``_meta`` key.
"""

from __future__ import annotations

import inspect
from typing import Any

from jevtools.canonical import jsonable
from jevtools.decision import Decision, ToolCall
from jevtools.loop import IDEMPOTENCY_META_KEY, LoopObservation, ingest_observation
from jevtools.sources.toolsource import get_any


def _call(decision: Decision, index: int) -> ToolCall:
    if not decision.tool_calls:
        raise ValueError(
            f"decision {decision.decision_id} has no tool call to execute (outcome {decision.outcome.value}, rule "
            f"{decision.rule}); only an executed decision may reach the MCP server"
        )
    return decision.tool_calls[index]


def call_arguments(decision: Decision, index: int = 0) -> dict[str, Any]:
    """The keyword arguments of ``session.call_tool`` for ``decision.tool_calls[index]``:
    ``{"name", "arguments", "meta": {"jevtools/idempotency_key": …}}``."""
    call = _call(decision, index)
    return {"name": call.name, "arguments": jsonable(call.arguments),
            "meta": {IDEMPOTENCY_META_KEY: call.idempotency_key}}  # fmt: skip


async def call_decision(session: Any, decision: Decision, *, index: int = 0) -> Any:
    """Execute an executed decision on an MCP client session and return its ``CallToolResult``.

    Raises ``ValueError`` when the decision carries no tool call (confirm, clarify, abstain…): nothing but an
    ``execute`` outcome ever reaches a server. A synchronous session (``call_tool`` returning the result) works too.
    """
    result = session.call_tool(**call_arguments(decision, index))
    return await result if inspect.isawaitable(result) else result


def result_content(result: Any) -> Any:
    """The useful content of a ``CallToolResult`` (dict or ``mcp`` object): ``structuredContent`` when present,
    else the text blocks joined by newlines (other blocks as their JSON)."""
    structured = get_any(result, "structuredContent", "structured_content")
    if structured is not None:
        return jsonable(structured)
    parts: list[str] = []
    for block in get_any(result, "content") or ():
        kind = get_any(block, "type")
        if kind == "text":
            parts.append(str(get_any(block, "text") or ""))
        else:
            dump = getattr(block, "model_dump", None)
            parts.append(str(jsonable(dump(by_alias=True, exclude_none=True) if callable(dump) else block)))
    return "\n".join(parts)


def is_error(result: Any) -> bool:
    """``isError`` of a ``CallToolResult``."""
    return bool(get_any(result, "isError", "is_error"))


def to_observation(decision: Decision, result: Any, *, step: int, index: int = 0, **kw: Any) -> LoopObservation:
    """The loop observation of an executed call and its ``CallToolResult`` (:func:`jevtools.loop.ingest_observation`:
    ``structuredContent`` first, ``isError`` → status ``error``, typed items and entities for the next pools).
    Keywords (``request`` for the preview ranking, ``emits``, ``output_schema``, ``preview_chars``…) go to
    ``ingest_observation``."""
    call = _call(decision, index)
    return ingest_observation(result, call.name, step, arguments=jsonable(call.arguments), call_id=call.id, **kw)


__all__ = [
    "IDEMPOTENCY_META_KEY",
    "call_arguments",
    "call_decision",
    "is_error",
    "result_content",
    "to_observation",
]
