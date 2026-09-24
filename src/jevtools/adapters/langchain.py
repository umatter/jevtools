"""LangChain / LangGraph interop (spec §7.2.3); needs the ``langchain`` extra (``langchain-core``).

.. code-block:: python

    from jevtools.adapters.langchain import JevChatModel
    llm = JevChatModel(router=router, context=ctx)                 # a BaseChatModel
    agent = create_react_agent(llm.bind_tools(tools), tools)       # unchanged ToolNode loop

- :meth:`JevChatModel.bind_tools` binds ``tools=[convert_to_openai_tool(t) …]`` and ``tool_choice``.
- ``_generate``/``_agenerate`` convert the messages (Human → user, AI → assistant with its tool calls, Tool →
  ``role: tool`` observation, System → system, which reaches Jev only with ``Context.include_system``), decide with
  the router (or resume a pending prompt, matched by the ``pending_id`` in the previous ``AIMessage``'s
  ``response_metadata["jev"]`` or by the prefix hash) and return an ``AIMessage`` with ``tool_calls`` on execute,
  else the templated prompt or handoff text, and ``response_metadata={"jev": <native decision document>}``.
- A per-call context: ``config={"configurable": {"jev_context": Context | {field: value}}}``.
- ``text_llm``: another chat model that answers abstain handoffs (``NO_TOOL``) with text.
- :func:`confirm_node`: a LangGraph node mapping CONFIRM/CLARIFY to ``interrupt()`` (``langgraph`` imported lazily).
"""

from __future__ import annotations

import contextvars
import json
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from pydantic import ConfigDict, Field

try:
    from langchain_core.callbacks import AsyncCallbackManagerForLLMRun, CallbackManagerForLLMRun
    from langchain_core.language_models import LanguageModelInput
    from langchain_core.language_models.chat_models import BaseChatModel
    from langchain_core.messages import (
        AIMessage,
        BaseMessage,
        ChatMessage,
        HumanMessage,
        SystemMessage,
        ToolMessage,
    )
    from langchain_core.outputs import ChatGeneration, ChatResult
    from langchain_core.runnables import Runnable, RunnableConfig
    from langchain_core.runnables.config import ensure_config
    from langchain_core.tools import BaseTool
    from langchain_core.utils.function_calling import convert_to_openai_tool
except ModuleNotFoundError as exc:  # pragma: no cover - exercised only without the extra
    raise ImportError("jevtools.adapters.langchain needs langchain-core: pip install 'jevtools[langchain]'") from exc

from jevtools.adapters._router import as_output_tool, merge_context, router_for
from jevtools.adapters.pending import InMemoryPendingStore, PendingStore, adecide_turn, decide_turn
from jevtools.context import Context
from jevtools.decision import Decision
from jevtools.policy import Outcome
from jevtools.router import Router

CONFIG_KEY = "jev_context"
"""``config["configurable"][CONFIG_KEY]`` carries a per-call context."""

_CALL_CONTEXT: contextvars.ContextVar[Any] = contextvars.ContextVar("jevtools_langchain_context", default=None)


# --------------------------------------------------------------------------------------------------------------------
# Message conversion
# --------------------------------------------------------------------------------------------------------------------


def _text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [p if isinstance(p, str) else str(p.get("text", "")) for p in content
                 if isinstance(p, str) or (isinstance(p, Mapping) and p.get("type", "text") == "text")]  # fmt: skip
        return "\n".join(p for p in parts if p)
    return "" if content is None else str(content)


def to_openai_messages(messages: Sequence[BaseMessage]) -> list[dict[str, Any]]:
    """LangChain messages as OpenAI-style messages (Human → user, AI → assistant, Tool → tool, System → system).

    An ``AIMessage`` produced by :class:`JevChatModel` keeps its ``pending_id`` as ``x_jev`` so a reply to its
    prompt resumes it.
    """
    out: list[dict[str, Any]] = []
    for message in messages:
        if isinstance(message, HumanMessage):
            out.append({"role": "user", "content": _text(message.content)})
        elif isinstance(message, AIMessage):
            doc: dict[str, Any] = {"role": "assistant", "content": _text(message.content) or None}
            calls = [{"id": call.get("id"), "type": "function",
                      "function": {"name": call["name"], "arguments": json.dumps(call.get("args") or {},
                                                                                 ensure_ascii=False)}}
                     for call in message.tool_calls]  # fmt: skip
            if calls:
                doc["tool_calls"] = calls
            jev = message.response_metadata.get("jev") if message.response_metadata else None
            if isinstance(jev, Mapping) and jev.get("pending_id"):
                doc["x_jev"] = {"pending_id": jev["pending_id"]}
            out.append(doc)
        elif isinstance(message, ToolMessage):
            content = message.content if isinstance(message.content, str) else _text(message.content)
            doc = {"role": "tool", "tool_call_id": message.tool_call_id, "content": content}
            if message.name:
                doc["name"] = message.name
            out.append(doc)
        elif isinstance(message, SystemMessage):
            out.append({"role": "system", "content": _text(message.content)})
        elif isinstance(message, ChatMessage) and message.role in ("user", "assistant", "system", "developer"):
            out.append({"role": message.role, "content": _text(message.content)})
    return out


def tool_choice_to_openai(tool_choice: Any, tools: Sequence[Mapping[str, Any]] = ()) -> Any:
    """LangChain ``tool_choice`` → OpenAI: ``None``/``"auto"`` → ``auto``; ``"any"``/``True``/``"required"`` →
    ``required``; ``"none"``/``False`` → ``none``; a tool name → named; OpenAI dicts pass unchanged."""
    if tool_choice is None or tool_choice == "auto":
        return "auto"
    if tool_choice is True or tool_choice in ("any", "required"):
        return "required"
    if tool_choice is False or tool_choice == "none":
        return "none"
    if isinstance(tool_choice, str):
        names = {t.get("function", {}).get("name") for t in tools}
        if names and tool_choice not in names:
            raise ValueError(f"tool_choice {tool_choice!r} is not a bound tool")
        return {"type": "function", "function": {"name": tool_choice}}
    if isinstance(tool_choice, Mapping) and tool_choice.get("type") == "tool" and "name" in tool_choice:
        return {"type": "function", "function": {"name": tool_choice["name"]}}
    return tool_choice


def to_ai_message(decision: Decision, *, content: str | None = None) -> AIMessage:
    """The ``AIMessage`` of a decision: ``tool_calls`` on execute, else the prompt or handoff text;
    ``response_metadata["jev"]`` is the native decision document."""
    text = content if content is not None else (decision.prompt.text if decision.prompt else decision.content or "")
    tokens = decision.usage.jev_input_tokens
    return AIMessage(
        content="" if decision.tool_calls else text,
        tool_calls=[
            {"name": c.name, "args": dict(c.arguments), "id": c.id, "type": "tool_call"} for c in decision.tool_calls
        ],
        response_metadata={
            "jev": decision.to_doc(),
            "finish_reason": decision.finish_reason,
            "model_name": "jevtools",
        },
        usage_metadata={"input_tokens": tokens, "output_tokens": 0, "total_tokens": tokens},
    )


# --------------------------------------------------------------------------------------------------------------------
# The chat model
# --------------------------------------------------------------------------------------------------------------------


class JevChatModel(BaseChatModel):
    """A LangChain chat model whose tool calls are elected by Jev through a :class:`~jevtools.router.Router`."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    router: Router
    context: Any = None
    """A :class:`~jevtools.context.Context` or a mapping of context fields laid over the router's context."""
    text_llm: Any = None
    """A LangChain chat model (anything with ``invoke``/``ainvoke``) that answers abstain handoffs with text."""
    store: Any = Field(default_factory=InMemoryPendingStore)
    """The :class:`~jevtools.adapters.pending.PendingStore` of pending prompts."""

    @property
    def _llm_type(self) -> str:
        return "jevtools"

    @property
    def _identifying_params(self) -> dict[str, Any]:
        return {"model_name": "jevtools", "backend": self.router.backend_name, "model": self.router.model}

    def bind_tools(
        self,
        tools: Sequence[dict[str, Any] | type | Callable[..., Any] | BaseTool],
        *,
        tool_choice: str | None = None,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, AIMessage]:
        """``self.bind(tools=[convert_to_openai_tool(t) for t in tools], tool_choice=tool_choice, **kwargs)``.

        ``with_structured_output`` binds its schema here with ``ls_structured_output_format``: that schema is an
        output tool (returning the result has no side effect), declared ``risk: read`` like Pydantic AI's output
        tools unless it declares a risk itself; otherwise its name falls to the fail-safe external tier, whose
        ``authorized`` gate and confirm band turn a confident extraction into a prompt and the parser's ``None``.
        """
        converted = [convert_to_openai_tool(t) for t in tools]
        if kwargs.get("ls_structured_output_format") is not None:
            converted = [as_output_tool(t) for t in converted]
        return self.bind(tools=converted, tool_choice=tool_choice, **kwargs)

    # -- config → context ---------------------------------------------------------------------------------------

    def invoke(
        self, input: LanguageModelInput, config: RunnableConfig | None = None, *, stop: list[str] | None = None,
        **kwargs: Any,
    ) -> AIMessage:  # fmt: skip
        token = _CALL_CONTEXT.set(_configured_context(config))
        try:
            return super().invoke(input, config, stop=stop, **kwargs)
        finally:
            _CALL_CONTEXT.reset(token)

    async def ainvoke(
        self, input: LanguageModelInput, config: RunnableConfig | None = None, *, stop: list[str] | None = None,
        **kwargs: Any,
    ) -> AIMessage:  # fmt: skip
        token = _CALL_CONTEXT.set(_configured_context(config))
        try:
            return await super().ainvoke(input, config, stop=stop, **kwargs)
        finally:
            _CALL_CONTEXT.reset(token)

    def _context(self, kwargs: Mapping[str, Any]) -> Context:
        ctx = merge_context(self.router.context, self.context)
        per_call = kwargs.get(CONFIG_KEY)
        if per_call is None:
            per_call = _CALL_CONTEXT.get()
        if per_call is None:
            per_call = _configured_context(None)
        merged = merge_context(ctx, per_call)
        return merged if merged is not None else Context()

    # -- generation ---------------------------------------------------------------------------------------------

    def _prepare(self, messages: list[BaseMessage], kwargs: Mapping[str, Any]) -> dict[str, Any]:
        tools = list(kwargs.get("tools") or ())
        return {
            "router": router_for(self.router, tools),
            "messages": to_openai_messages(messages),
            "context": self._context(kwargs),
            "tool_choice": tool_choice_to_openai(kwargs.get("tool_choice"), tools),
            "parallel_tool_calls": bool(kwargs.get("parallel_tool_calls") or False),
            "store": self.store,
        }

    def _needs_text(self, decision: Decision) -> bool:
        return self.text_llm is not None and decision.outcome is Outcome.ABSTAIN and not decision.content

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        args = self._prepare(messages, kwargs)
        decision = decide_turn(args.pop("router"), args.pop("messages"), **args)
        content = _text(self.text_llm.invoke(messages).content) if self._needs_text(decision) else None
        return _result(decision, content)

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        args = self._prepare(messages, kwargs)
        decision = await adecide_turn(args.pop("router"), args.pop("messages"), **args)
        content = _text((await self.text_llm.ainvoke(messages)).content) if self._needs_text(decision) else None
        return _result(decision, content)


def _configured_context(config: RunnableConfig | None) -> Any:
    configurable = ensure_config(config).get("configurable") or {}
    return configurable.get(CONFIG_KEY)


def _result(decision: Decision, content: str | None) -> ChatResult:
    message = to_ai_message(decision, content=content)
    info = {"finish_reason": decision.finish_reason}
    return ChatResult(generations=[ChatGeneration(message=message, generation_info=info)],
                      llm_output={"model_name": "jevtools"})  # fmt: skip


# --------------------------------------------------------------------------------------------------------------------
# LangGraph human-in-the-loop
# --------------------------------------------------------------------------------------------------------------------


def pending_prompt(message: Any) -> dict[str, Any] | None:
    """The jevtools decision document of an ``AIMessage`` that asks the user (CONFIRM/CLARIFY), else ``None``."""
    metadata = getattr(message, "response_metadata", None) or {}
    jev = metadata.get("jev") if isinstance(metadata, Mapping) else None
    if not isinstance(jev, Mapping) or jev.get("outcome") not in ("confirm", "clarify") or not jev.get("prompt"):
        return None
    return dict(jev)


def needs_confirmation(state: Any) -> bool:
    """Whether the last message of a graph state is a jevtools CONFIRM/CLARIFY (for conditional edges)."""
    messages = _messages(state)
    return bool(messages) and pending_prompt(messages[-1]) is not None


def _messages(state: Any) -> list[Any]:
    if isinstance(state, Mapping):
        return list(state.get("messages") or [])
    if isinstance(state, Sequence):
        return list(state)
    return list(getattr(state, "messages", []) or [])


def _reply_text(answer: Any, options: Sequence[Mapping[str, Any]]) -> str:
    if isinstance(answer, Mapping):
        if "selection" in answer:
            selection = str(answer["selection"])
            return next((str(o["text"]) for o in options if o.get("id") == selection), selection)
        if "reply" in answer:
            return str(answer["reply"])
    return str(answer)


def confirm_node(state: Any, config: RunnableConfig | None = None, *, interrupt: Callable[[Any], Any] | None = None
                 ) -> dict[str, Any]:  # fmt: skip
    """LangGraph node: when the last ``AIMessage`` is a CONFIRM/CLARIFY, ``interrupt()`` with
    ``{"kind": "jevtools", "outcome", "prompt": {kind, text, options}, "pending_id", "decision_id"}``.

    Resume the graph with ``Command(resume=…)``: a string (the user's reply), ``{"selection": "<option id>"}`` (a
    click, sent as the option text so it matches exactly) or ``{"reply": "…"}``. The node returns the reply as a
    ``HumanMessage``; routing it back to the model resumes the pending prompt (no Jev call for a click).
    """
    messages = _messages(state)
    jev = pending_prompt(messages[-1]) if messages else None
    if jev is None:
        return {}
    ask = interrupt if interrupt is not None else _langgraph_interrupt()
    prompt = jev["prompt"]
    answer = ask({"kind": "jevtools", "outcome": jev["outcome"], "prompt": prompt, "pending_id": jev.get("pending_id"),
                  "decision_id": jev.get("decision_id")})  # fmt: skip
    return {"messages": [HumanMessage(content=_reply_text(answer, prompt.get("options") or []))]}


def _langgraph_interrupt() -> Callable[[Any], Any]:
    try:
        from langgraph.types import interrupt  # type: ignore[import-not-found, unused-ignore]
    except ModuleNotFoundError as exc:
        raise ImportError("confirm_node needs LangGraph: pip install langgraph") from exc
    return interrupt  # type: ignore[no-any-return, unused-ignore]


__all__ = [
    "CONFIG_KEY",
    "JevChatModel",
    "PendingStore",
    "confirm_node",
    "needs_confirmation",
    "pending_prompt",
    "to_ai_message",
    "to_openai_messages",
    "tool_choice_to_openai",
]
