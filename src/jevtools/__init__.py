"""jevtools: tool calling for TypeSafe's Jev, where every argument value is elected from code-built candidate pools.

``import jevtools as jt`` (spec §9). The main entry points:

- declare tools: :func:`tool`, :class:`Catalog` (OpenAI, MCP, callables, pydantic), :func:`strip_xjev`, :data:`hints`
  and the ``Annotated`` markers (:class:`Ref`, :class:`Span`, :class:`Money`, …);
- give them evidence: :class:`Context` with sources (:class:`Registry`, :class:`FileIndex`, :class:`Provider`);
- decide: :class:`Router` (``decide``/``resume``) over a backend from :mod:`jevtools.backends` (``auto()``, the
  HTTP backends, ``ScriptedBackend``, ``LexicalSimulator``, ``Cassette``), governed by a :class:`Policy`; the result
  is a :class:`Decision` with its :class:`Trace`, replayable by :func:`verify`;
- run multi-step tasks: :class:`Agent` (§6) with observations as candidate pools and an :class:`EntityStore`;
- cover generative slots: :class:`OpenAICompatibleFiller`, :class:`OpenAICompatibleEscalator`,
  :class:`OpenAICompatibleTextLLM` (§4.7);
- interoperate: :mod:`jevtools.openai` (``complete``, ``wrap``), :mod:`jevtools.adapters` (Anthropic, MCP,
  LangChain, Pydantic AI), :mod:`jevtools.compat` (``cookbook_policy``, ``from_jev_fn``); the ``jevtools`` CLI,
  the proxy (:mod:`jevtools.serve`, extra ``serve``) and the evaluation harness (:mod:`jevtools.eval`) are imported
  explicitly;
- extend: register a :class:`Resolver` for a slot kind with :func:`register_resolver`.
"""

from jevtools import adapters, backends, compat
from jevtools._version import SPEC_VERSION, __version__
from jevtools.adapters import openai
from jevtools.ballot import Ballot, BallotOption, BallotQuestion, SentinelSpec, ToolViability
from jevtools.budget import TokenEstimator
from jevtools.candidates import Bottom, Candidate, Channel, Pool
from jevtools.canonical import canonical_json, round4, sha256_of
from jevtools.confidence import Composition, Factors, IsotonicCalibrator
from jevtools.context import Clock, Context, Observation, Turn, build_state, render_now
from jevtools.decision import (
    Bottleneck,
    Confidence,
    Decision,
    DecisionUsage,
    Pending,
    PendingAction,
    Prompt,
    PromptOption,
    SlotReport,
    ToolCall,
)
from jevtools.errors import BallotError, CatalogError, ConstraintError, JevtoolsError, PendingScopeError
from jevtools.extract import Mention, Mentions, run_extractors
from jevtools.fallback import (
    Escalator,
    FillCandidate,
    Filler,
    FillRequest,
    OpenAICompatibleEscalator,
    OpenAICompatibleFiller,
    OpenAICompatibleTextLLM,
    ProposedCall,
    TextLLM,
)
from jevtools.kinds import ResolveContext, Resolver, SlotResult, register_resolver
from jevtools.loop import (
    Agent,
    Entity,
    EntityStore,
    Executor,
    LoopBudget,
    LoopObservation,
    LoopResult,
    LoopStep,
    LoopUsage,
    ingest_observation,
)
from jevtools.plan import compile_round, plan_round
from jevtools.policy import Action, Outcome, Policy, PolicyInput, PolicyResult, Tier, evaluate
from jevtools.router import Router
from jevtools.sources import FileIndex, MCPResources, Provider, Registry, Source, SourceQuery, ToolSource
from jevtools.spec import (
    Ask,
    Catalog,
    Channels,
    CodeList,
    Default,
    Derive,
    ListOf,
    Money,
    Noun,
    Quantity,
    Ref,
    SlotSpec,
    Span,
    Stakes,
    Text,
    Tool,
    ToolSpec,
    When,
    hints,
    register_check,
    strip_xjev,
    tool,
)
from jevtools.trace import Trace, VerifyReport, verify
from jevtools.validate import Limits, preflight

__all__ = [
    "SPEC_VERSION",
    "Action",
    "Agent",
    "Ask",
    "Ballot",
    "BallotError",
    "BallotOption",
    "BallotQuestion",
    "Bottleneck",
    "Bottom",
    "Candidate",
    "Catalog",
    "CatalogError",
    "Channel",
    "Channels",
    "Clock",
    "CodeList",
    "Composition",
    "Confidence",
    "ConstraintError",
    "Context",
    "Decision",
    "DecisionUsage",
    "Default",
    "Derive",
    "Entity",
    "EntityStore",
    "Escalator",
    "Executor",
    "Factors",
    "FileIndex",
    "FillCandidate",
    "FillRequest",
    "Filler",
    "IsotonicCalibrator",
    "JevtoolsError",
    "Limits",
    "ListOf",
    "LoopBudget",
    "LoopObservation",
    "LoopResult",
    "LoopStep",
    "LoopUsage",
    "MCPResources",
    "Mention",
    "Mentions",
    "Money",
    "Noun",
    "Observation",
    "OpenAICompatibleEscalator",
    "OpenAICompatibleFiller",
    "OpenAICompatibleTextLLM",
    "Outcome",
    "Pending",
    "PendingScopeError",
    "PendingAction",
    "Policy",
    "PolicyInput",
    "PolicyResult",
    "Pool",
    "Prompt",
    "PromptOption",
    "ProposedCall",
    "Provider",
    "Quantity",
    "Ref",
    "Registry",
    "ResolveContext",
    "Resolver",
    "Router",
    "SentinelSpec",
    "SlotReport",
    "SlotResult",
    "SlotSpec",
    "Source",
    "SourceQuery",
    "Span",
    "Stakes",
    "Text",
    "TextLLM",
    "Tier",
    "TokenEstimator",
    "Tool",
    "ToolCall",
    "ToolSource",
    "ToolSpec",
    "ToolViability",
    "Trace",
    "Turn",
    "VerifyReport",
    "When",
    "__version__",
    "adapters",
    "backends",
    "build_state",
    "canonical_json",
    "compat",
    "compile_round",
    "evaluate",
    "hints",
    "ingest_observation",
    "openai",
    "plan_round",
    "preflight",
    "register_check",
    "register_resolver",
    "render_now",
    "round4",
    "run_extractors",
    "sha256_of",
    "strip_xjev",
    "tool",
    "verify",
]
