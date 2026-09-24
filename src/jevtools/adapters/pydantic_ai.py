"""Pydantic AI interop (spec §7.2.5); needs the ``pydantic-ai`` extra (``pydantic-ai-slim>=1.0,<2``).

.. code-block:: python

    from jevtools.adapters.pydantic_ai import JevModel
    agent = Agent(JevModel(router, context=ctx), tools=[...])

:class:`JevModel` implements ``Model.request(messages, model_settings, model_request_parameters)``: it reads the
``function_tools`` and ``output_tools`` (``ToolDefinition``: ``name``, ``description``,
``parameters_json_schema``), converts the history (user prompts → user, tool returns → ``role: tool``
observations, model responses → assistant with tool calls), decides with the router and answers with
``ModelResponse(parts=[ToolCallPart(...)])`` on execute, else a ``TextPart`` (the templated prompt, a handoff, or
the loop's receipt). Output tools such as ``final_result`` are tools like any other, so structured output can be
*elected*. ``provider_details["jev"]`` carries the native decision document (its ``pending_id`` lets the next
request resume a prompt).

When the agent's output type allows no text (``output_type=SomeModel``), a decision that is not a call — a confirm
card, a clarify menu, an abstain or a finished loop without an elected output — cannot be answered as text
(pydantic-ai would reject it and re-ask, spending another Jev round on the same turn). ``request`` raises
:class:`JevPromptRequired` instead: it carries the decision, the prompt text, its ``pending_id`` and ``messages``
(the history plus the prompt), so the host shows the prompt and resumes with
``agent.run(reply, message_history=exc.messages)``. ``output_type=[SomeModel, str]`` answers prompts as text.

``request_stream`` (``agent.run_stream``, ``run_stream_events``) decides like ``request`` and replays the response
as one streamed message (a text handoff to ``text_model`` is requested non-streamed and replayed).

All knowledge of the pydantic-ai API lives in this module, so API drift is isolated here (verified against
pydantic-ai-slim 1.x: ``Model.prepare_request``, ``ModelRequestParameters.function_tools/output_tools/
allow_text_output``, ``ModelResponse.provider_details/finish_reason``).
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterable, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from typing import Any

try:
    from pydantic_ai.exceptions import AgentRunError
    from pydantic_ai.messages import (
        ModelMessage,
        ModelRequest,
        ModelResponse,
        ModelResponsePart,
        ModelResponseStreamEvent,
        RetryPromptPart,
        SystemPromptPart,
        TextPart,
        ToolCallPart,
        ToolReturnPart,
        UserPromptPart,
    )
    from pydantic_ai.models import Model, ModelRequestParameters, StreamedResponse
    from pydantic_ai.settings import ModelSettings
    from pydantic_ai.tools import ToolDefinition
    from pydantic_ai.usage import RequestUsage
except ModuleNotFoundError as exc:  # pragma: no cover - exercised only without the extra
    raise ImportError("jevtools.adapters.pydantic_ai needs pydantic-ai: pip install 'jevtools[pydantic-ai]'") from exc

from jevtools.adapters._router import OUTPUT_TOOL_XJEV, merge_context, router_for
from jevtools.adapters.pending import InMemoryPendingStore, PendingStore, adecide_turn
from jevtools.canonical import jsonable
from jevtools.context import Context
from jevtools.decision import Decision
from jevtools.policy import Outcome
from jevtools.router import Router

MODEL_NAME = "jevtools"


ABSTAIN_TEXT = "No tool applies to this request."
"""Default text of an abstain handoff without a ``text_model`` (a fixed template; pydantic-ai rejects an empty
response when text output is expected)."""


class JevPromptRequired(AgentRunError):
    """The decision needs the user (a confirm card, a clarify menu) or elected no output (abstain, done), but the
    agent's output type allows no text. ``text`` is the prompt; ``messages`` is the history plus the prompt as a
    model response: ``agent.run(reply, message_history=exc.messages)`` resumes it (a click costs no Jev call)."""

    def __init__(self, decision: Decision, text: str, messages: list[ModelMessage]) -> None:
        super().__init__(f"jevtools {decision.outcome.value} ({decision.rule}) needs the user, but the output type "
                         f"allows no text: {text}")  # fmt: skip
        self.decision = decision
        self.text = text
        self.messages = messages

    @property
    def pending_id(self) -> str | None:
        """The prompt's pending id (``None`` for an abstain or a finished loop)."""
        return self.decision.pending.pending_id if self.decision.pending is not None else None

    @property
    def doc(self) -> dict[str, Any]:
        """The native decision document."""
        return self.decision.to_doc()


def tool_to_openai(tool: ToolDefinition, *, output: bool = False) -> dict[str, Any]:
    """A pydantic-ai ``ToolDefinition`` as an OpenAI function tool; output tools are declared ``risk: read``."""
    function: dict[str, Any] = {"name": tool.name, "description": tool.description or "",
                                "parameters": dict(tool.parameters_json_schema)}  # fmt: skip
    if output:
        function["x-jev"] = dict(OUTPUT_TOOL_XJEV)
    return {"type": "function", "function": function}


def _user_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, Sequence):
        return "\n".join(item for item in content if isinstance(item, str))
    return ""


def _tool_content(part: ToolReturnPart) -> Any:
    content = part.content
    if content is None or isinstance(content, (str, int, float, bool, dict, list)):
        return jsonable(content)
    return part.model_response_str()


def to_openai_messages(messages: Sequence[ModelMessage]) -> list[dict[str, Any]]:
    """Pydantic AI messages as OpenAI-style messages.

    System prompts → system; user prompts → user (text items only); tool returns → ``role: tool``; a tool retry
    → ``role: tool`` with the error text; a non-tool retry prompt (output validation feedback) is dropped, because
    it is framework feedback, not the user's words. Responses → assistant (text, tool calls, and ``x_jev`` with
    the ``pending_id`` of a jevtools prompt).
    """
    out: list[dict[str, Any]] = []
    for message in messages:
        if isinstance(message, ModelRequest):
            for part in message.parts:
                if isinstance(part, SystemPromptPart):
                    out.append({"role": "system", "content": part.content})
                elif isinstance(part, UserPromptPart):
                    out.append({"role": "user", "content": _user_text(part.content)})
                elif isinstance(part, ToolReturnPart):
                    out.append({"role": "tool", "tool_call_id": part.tool_call_id, "name": part.tool_name,
                                "content": _tool_content(part)})  # fmt: skip
                elif isinstance(part, RetryPromptPart) and part.tool_name:
                    out.append({"role": "tool", "tool_call_id": part.tool_call_id, "name": part.tool_name,
                                "content": part.model_response()})  # fmt: skip
        elif isinstance(message, ModelResponse):
            texts = [p.content for p in message.parts if isinstance(p, TextPart) and p.content]
            calls = [{"id": p.tool_call_id, "type": "function",
                      "function": {"name": p.tool_name, "arguments": p.args_as_json_str()}}
                     for p in message.parts if isinstance(p, ToolCallPart)]  # fmt: skip
            doc: dict[str, Any] = {"role": "assistant", "content": "\n".join(texts) if texts else None}
            if calls:
                doc["tool_calls"] = calls
            jev = (message.provider_details or {}).get("jev")
            if isinstance(jev, dict) and jev.get("pending_id"):
                doc["x_jev"] = {"pending_id": jev["pending_id"]}
            out.append(doc)
    return out


def _last_tool_return(messages: Sequence[ModelMessage], *, latest_only: bool) -> str | None:
    """The last tool result (in the latest request only, or anywhere in the history)."""
    for message in reversed(messages[-1:] if latest_only else messages):
        if isinstance(message, ModelRequest):
            returns = [p for p in message.parts if isinstance(p, ToolReturnPart)]
            if returns:
                return returns[-1].model_response_str()
    return None


class JevModel(Model):
    """A pydantic-ai ``Model`` whose tool calls are elected by Jev through a :class:`~jevtools.router.Router`.

    ``text_model`` (another pydantic-ai ``Model``) answers abstain handoffs; it sees the request without function
    tools, so it can only answer with text (or an output tool). Without it, a finished loop (``done``, or ``NO_TOOL``
    right after a tool result) answers with the last tool result as its receipt, and other abstains with
    ``abstain_text``.
    """

    def __init__(
        self,
        router: Router,
        *,
        context: Context | dict[str, Any] | None = None,
        text_model: Model | None = None,
        store: PendingStore | None = None,
        model_name: str = MODEL_NAME,
        abstain_text: str = ABSTAIN_TEXT,
        settings: ModelSettings | None = None,
    ) -> None:
        super().__init__(settings=settings)
        self.router = router
        self.context = context
        self.text_model = text_model
        self.store: PendingStore = store if store is not None else InMemoryPendingStore()
        self._model_name = model_name
        self.abstain_text = abstain_text

    @property
    def model_name(self) -> str:
        """The model name reported in responses."""
        return self._model_name

    @property
    def system(self) -> str:
        """The provider name (``jevtools``)."""
        return "jevtools"

    @property
    def provider(self) -> None:
        """No HTTP provider: the router's backend talks to Jev."""
        return None

    def tool_choice(self, params: ModelRequestParameters) -> str:
        """``required`` when the agent disallows text output and offers output tools (``NO_TOOL`` is then removed
        from the tool Choice), else ``auto``."""
        return "required" if params.output_tools and not params.allow_text_output else "auto"

    async def request(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> ModelResponse:
        """Decide one step (spec §7.2.5)."""
        model_settings, params = self.prepare_request(model_settings, model_request_parameters)
        tools = [tool_to_openai(t) for t in params.function_tools]
        tools += [tool_to_openai(t, output=True) for t in params.output_tools]
        router = router_for(self.router, tools)
        ctx = merge_context(router.context, self.context)
        decision = await adecide_turn(router, to_openai_messages(messages), context=ctx,
                                      tool_choice=self.tool_choice(params), store=self.store)  # fmt: skip
        if decision.tool_calls:
            parts: list[ModelResponsePart] = [ToolCallPart(tool_name=c.name, args=jsonable(c.arguments),
                                                           tool_call_id=c.id) for c in decision.tool_calls]  # fmt: skip
            return self._response(decision, parts, "tool_call")
        if self.text_model is not None and decision.outcome is Outcome.ABSTAIN and not decision.content:
            handoff = await self.text_model.request(messages, model_settings, replace(params, function_tools=[]))
            return replace(handoff, provider_details={**(handoff.provider_details or {}), "jev": decision.to_doc()})
        text = self.text_of(decision, messages)
        response = self._response(decision, [TextPart(content=text)], "stop")
        if not params.allow_text_output:  # a text answer would be rejected and the same turn decided again
            raise JevPromptRequired(decision, text, [*messages, response])
        return response

    @asynccontextmanager
    async def request_stream(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
        run_context: Any = None,
    ) -> AsyncIterator[StreamedResponse]:
        """Decide one step like :meth:`request` and replay the response as one streamed message."""
        response = await self.request(messages, model_settings, model_request_parameters)
        _, params = self.prepare_request(model_settings, model_request_parameters)
        yield _ReplayedResponse(model_request_parameters=params, _response=response)

    def text_of(self, decision: Decision, messages: Sequence[ModelMessage]) -> str:
        """The text answer: the templated prompt, the handoff text, the last tool result when the loop is finished
        (the receipt), else :attr:`abstain_text` for an abstain."""
        if decision.prompt is not None:
            return decision.prompt.text
        if decision.content:
            return decision.content
        if decision.outcome is Outcome.DONE:
            return _last_tool_return(messages, latest_only=False) or ""
        if decision.outcome is Outcome.ABSTAIN:
            return _last_tool_return(messages, latest_only=True) or self.abstain_text
        return ""

    def _response(self, decision: Decision, parts: list[ModelResponsePart], finish: Any) -> ModelResponse:
        return ModelResponse(
            parts=parts,
            usage=RequestUsage(input_tokens=decision.usage.jev_input_tokens, output_tokens=0),
            model_name=self._model_name,
            provider_name="jevtools",
            provider_details={"jev": decision.to_doc()},
            provider_response_id=decision.decision_id,
            finish_reason=finish,
        )


def _events(result: Any) -> Iterable[ModelResponseStreamEvent]:
    """The events of a parts-manager call (one event, ``None``, or an iterator, depending on the pydantic-ai 1.x
    minor version)."""
    if result is None:
        return ()
    if hasattr(result, "event_kind"):
        return (result,)
    return list(result)


@dataclass
class _ReplayedResponse(StreamedResponse):
    """A decided :class:`ModelResponse` replayed as a stream: one text delta per text part, one event per call."""

    _response: ModelResponse = field(default_factory=lambda: ModelResponse(parts=[]))
    _timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc), init=False)

    def __post_init__(self) -> None:
        self._usage = self._response.usage
        self.provider_details = self._response.provider_details
        self.provider_response_id = self._response.provider_response_id
        self.finish_reason = self._response.finish_reason

    async def _get_event_iterator(self) -> AsyncIterator[ModelResponseStreamEvent]:
        for index, part in enumerate(self._response.parts):
            if isinstance(part, TextPart):
                for event in _events(self._parts_manager.handle_text_delta(vendor_part_id=index, content=part.content)):
                    yield event
            elif isinstance(part, ToolCallPart):
                for event in _events(self._parts_manager.handle_tool_call_part(
                        vendor_part_id=index, tool_name=part.tool_name, args=part.args,
                        tool_call_id=part.tool_call_id)):  # fmt: skip
                    yield event

    @property
    def model_name(self) -> str:
        return self._response.model_name or MODEL_NAME

    @property
    def provider_name(self) -> str | None:
        return self._response.provider_name

    @property
    def provider_url(self) -> str | None:
        return None

    @property
    def timestamp(self) -> datetime:
        return self._timestamp

    async def close_stream(self) -> None:
        """Nothing to close: the response is already complete."""


__all__ = [
    "ABSTAIN_TEXT",
    "MODEL_NAME",
    "OUTPUT_TOOL_XJEV",
    "JevModel",
    "JevPromptRequired",
    "to_openai_messages",
    "tool_to_openai",
]
