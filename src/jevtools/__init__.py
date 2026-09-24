"""jevtools: tool calling for TypeSafe's Jev, where every argument value is elected from code-built candidate pools.

``import jevtools as jt`` (spec §9). The main entry points:

- declare tools: :func:`tool`, :class:`Catalog` (OpenAI, MCP, callables, pydantic), :func:`strip_xjev`, :data:`hints`
  and the ``Annotated`` markers (:class:`Ref`, :class:`Span`, :class:`Money`, …);
- give them evidence: :class:`Context` with sources (:class:`Registry`, :class:`FileIndex`, :class:`Provider`);
- decide: :class:`Router` (``decide``/``resume``) over a backend from :mod:`jevtools.backends`, governed by a
  :class:`Policy`; the result is a :class:`Decision` with its :class:`Trace`, replayable by :func:`verify`;
- extend: register a :class:`Resolver` for a slot kind with :func:`register_resolver`.
"""

from jevtools import backends
from jevtools._version import SPEC_VERSION, __version__
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
from jevtools.errors import BallotError, CatalogError, ConstraintError, JevtoolsError
from jevtools.extract import Mention, Mentions, run_extractors
from jevtools.fallback import Escalator, FillCandidate, Filler, FillRequest, ProposedCall, TextLLM
from jevtools.kinds import ResolveContext, Resolver, SlotResult, register_resolver
from jevtools.plan import compile_round, plan_round
from jevtools.policy import Action, Outcome, Policy, PolicyInput, PolicyResult, Tier, evaluate
from jevtools.router import Router
from jevtools.sources import FileIndex, Provider, Registry, Source, SourceQuery
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
    "Escalator",
    "Factors",
    "FileIndex",
    "FillCandidate",
    "FillRequest",
    "Filler",
    "IsotonicCalibrator",
    "JevtoolsError",
    "Limits",
    "ListOf",
    "Mention",
    "Mentions",
    "Money",
    "Noun",
    "Observation",
    "Outcome",
    "Pending",
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
    "ToolSpec",
    "ToolViability",
    "Trace",
    "Turn",
    "VerifyReport",
    "When",
    "__version__",
    "backends",
    "build_state",
    "canonical_json",
    "compile_round",
    "evaluate",
    "hints",
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
