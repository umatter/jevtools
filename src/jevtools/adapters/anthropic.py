"""Anthropic Messages interop (spec §3.10; the proxy's ``POST /v1/messages`` is v0.2, the library emitter is v0.1).

- :func:`to_tool_use`: ``tool_use`` blocks of an executed decision
  (``{"type": "tool_use", "id": "toolu_jev_<hash>", "name", "input"}``; the id shares the digest of ``call_jev_…``).
- :func:`to_message`: a Messages API response document (``content`` blocks, ``stop_reason`` ``tool_use`` or
  ``end_turn``, ``usage`` with ``x_jev``).
- :func:`tools_to_openai` / :func:`catalog_from_anthropic`: Anthropic tool definitions
  (``{"name", "description", "input_schema"}``) → OpenAI function tools → :class:`~jevtools.spec.catalog.Catalog`.
- :func:`to_openai_messages`: Anthropic ``messages`` (+ ``system``) → OpenAI-style messages, so ``tool_result``
  blocks become ``role: tool`` observations and ``tool_use`` blocks assistant ``tool_calls``.
- :func:`tool_choice_to_openai`: ``{"type": "auto"|"any"|"tool"|"none"}`` → OpenAI ``tool_choice``.
- :func:`complete`: one decision over Anthropic-format input, answered as a Messages response.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from jevtools.adapters._router import merge_context, router_for
from jevtools.adapters.pending import PendingStore, decide_turn, default_store
from jevtools.canonical import canonical_str, jsonable
from jevtools.context import Context
from jevtools.decision import Decision
from jevtools.router import Router
from jevtools.spec.catalog import Catalog, SidecarLike

MODEL_NAME = "jevtools"


def to_tool_use(decision: Decision) -> list[dict[str, Any]]:
    """The ``tool_use`` blocks of ``decision`` (empty unless ``outcome == "execute"``)."""
    return [call.to_anthropic() for call in decision.tool_calls]


def to_content(decision: Decision) -> list[dict[str, Any]]:
    """Anthropic content blocks: ``tool_use`` blocks on execute, else one ``text`` block (the templated prompt or
    the handoff text; none when empty)."""
    return decision.to_anthropic_content()


def to_message(decision: Decision, *, model: str = MODEL_NAME) -> dict[str, Any]:
    """A Messages API response document for ``decision``; ``x_jev`` mirrors the OpenAI message's."""
    x_jev = decision.to_openai_message()["x_jev"]
    tokens = decision.usage.jev_input_tokens
    return {
        "id": "msg_jev_" + decision.decision_id.removeprefix("dec_"),
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": to_content(decision),
        "stop_reason": "tool_use" if decision.tool_calls else "end_turn",
        "stop_sequence": None,
        "usage": {
            "input_tokens": tokens,
            "output_tokens": 0,
            "x_jev": {**jsonable(decision.usage), "rounds": decision.rounds},
        },
        "x_jev": x_jev,
    }


# --------------------------------------------------------------------------------------------------------------------
# Tools
# --------------------------------------------------------------------------------------------------------------------


def tool_to_openai(tool: Mapping[str, Any]) -> dict[str, Any]:
    """One Anthropic tool (``name``, ``description``, ``input_schema``; tool-level ``x-jev`` kept) as an OpenAI
    function tool. Server tools without ``input_schema`` (``web_search_…``) are rejected."""
    if "input_schema" not in tool:
        raise ValueError(f"not a client tool with an input_schema: {tool.get('name')!r}")
    function: dict[str, Any] = {"name": tool["name"], "description": tool.get("description") or "",
                                "parameters": dict(tool["input_schema"])}  # fmt: skip
    if "x-jev" in tool:
        function["x-jev"] = tool["x-jev"]
    return {"type": "function", "function": function}


def tools_to_openai(tools: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Anthropic tool definitions as OpenAI function tools."""
    return [tool_to_openai(t) for t in tools]


def catalog_from_anthropic(
    tools: Iterable[Mapping[str, Any]], *, sidecar: SidecarLike = None, hints: SidecarLike = None,
    sources: Iterable[Any] = (),
) -> Catalog:  # fmt: skip
    """A :class:`Catalog` from Anthropic tool definitions."""
    return Catalog.from_openai(tools_to_openai(tools), sidecar=sidecar, hints=hints, sources=sources)


def tool_choice_to_openai(choice: Mapping[str, Any] | None) -> Any:
    """``auto`` → ``"auto"``, ``any`` → ``"required"``, ``tool`` → named, ``none`` → ``"none"`` (spec §7.2.2)."""
    if choice is None:
        return "auto"
    kind = choice.get("type", "auto")
    if kind == "any":
        return "required"
    if kind == "tool":
        return {"type": "function", "function": {"name": choice["name"]}}
    if kind in ("auto", "none"):
        return kind
    raise ValueError(f"unknown Anthropic tool_choice {choice!r}")


# --------------------------------------------------------------------------------------------------------------------
# Messages
# --------------------------------------------------------------------------------------------------------------------


def _blocks(content: Any) -> list[Mapping[str, Any]]:
    if content is None:
        return []
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    return [b for b in content if isinstance(b, Mapping)]


def _result_content(block: Mapping[str, Any]) -> Any:
    content = block.get("content")
    if isinstance(content, list):
        texts = [b.get("text", "") for b in content if isinstance(b, Mapping) and b.get("type") == "text"]
        return "\n".join(t for t in texts if t)
    return content


def to_openai_messages(messages: Sequence[Mapping[str, Any]], system: Any = None) -> list[dict[str, Any]]:
    """Anthropic ``messages`` (and ``system``) as OpenAI-style messages.

    ``user`` text blocks → a user message; ``tool_result`` blocks → ``role: tool`` messages (an ``is_error``
    result keeps its text); ``assistant`` text → assistant content, ``tool_use`` blocks → ``tool_calls`` with
    canonical-JSON arguments. An assistant message's ``x_jev`` (echoed by jevtools-aware clients) is kept.
    """
    out: list[dict[str, Any]] = []
    if system:
        text = system if isinstance(system, str) else "\n".join(b.get("text", "") for b in _blocks(system))
        out.append({"role": "system", "content": text})
    for message in messages:
        role = message.get("role", "user")
        blocks = _blocks(message.get("content"))
        texts = [str(b.get("text", "")) for b in blocks if b.get("type") == "text"]
        if role == "assistant":
            calls = [{"id": b.get("id"), "type": "function",
                      "function": {"name": b.get("name"), "arguments": canonical_str(b.get("input") or {})}}
                     for b in blocks if b.get("type") == "tool_use"]  # fmt: skip
            doc: dict[str, Any] = {"role": "assistant", "content": "\n".join(texts) if texts else None}
            if calls:
                doc["tool_calls"] = calls
            if isinstance(message.get("x_jev"), Mapping):
                doc["x_jev"] = dict(message["x_jev"])
            out.append(doc)
            continue
        for block in blocks:
            if block.get("type") == "tool_result":
                out.append({"role": "tool", "tool_call_id": block.get("tool_use_id"),
                            "content": _result_content(block)})  # fmt: skip
        if texts:
            out.append({"role": "user", "content": "\n".join(texts)})
    return out


def complete(
    messages: Sequence[Mapping[str, Any]],
    tools: Sequence[Mapping[str, Any]] | None = None,
    *,
    router: Router,
    system: Any = None,
    context: Context | Mapping[str, Any] | None = None,
    tool_choice: Mapping[str, Any] | None = None,
    store: PendingStore | None = None,
    model: str = MODEL_NAME,
) -> dict[str, Any]:
    """Decide one turn over Anthropic-format input and answer with a Messages response document (pending prompts
    are matched like in :func:`jevtools.adapters.openai.complete`)."""
    r = router_for(router, tools_to_openai(tools) if tools else None)
    decision = decide_turn(r, to_openai_messages(messages, system), context=merge_context(r.context, context),
                           tool_choice=tool_choice_to_openai(tool_choice),
                           store=store if store is not None else default_store(r))  # fmt: skip
    return to_message(decision, model=model)


__all__ = [
    "catalog_from_anthropic",
    "complete",
    "to_content",
    "to_message",
    "to_openai_messages",
    "to_tool_use",
    "tool_choice_to_openai",
    "tool_to_openai",
    "tools_to_openai",
]
