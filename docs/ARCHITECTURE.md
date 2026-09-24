# Architecture

This document is a map of the `jevtools` package. The protocol itself is specified in [SPEC.md](SPEC.md), and the
places where the implementation settles an ambiguity or departs from the spec are listed in
[DECISIONS.md](DECISIONS.md). Section numbers (§) refer to the spec.

## Data flow

One call to `Router.decide(messages)` runs the pipeline below (§2). Stages 1–4 are code only. Stage 5 is the only
one that talks to Jev. Everything from the Ballot onward is byte-specified, and the golden fixtures in
`tests/golden/` pin it (§10.2).

```
messages + tools + Context
  │
  ├─ ingest ─────── spec/ingest.py, spec/catalog.py, spec/infer.py, spec/sidecar.py      → Catalog (ToolSpec, SlotSpec)
  ├─ extract ────── extract/base.py run_extractors (claiming), extract/*                 → Mentions
  ├─ pool ───────── kinds/<kind>.py Resolver.pool, sources/*                             → Pool per (tool, slot)
  ├─ plan ───────── plan.py compile_round: viability → speculation → questions →
  │                 budget → split; validate.py preflight                                → Ballot
  ├─ ask ────────── Ballot.to_requests(model) → backends/* (parallel calls = one round)  → DecisionResponse[]
  ├─ decode ─────── decode.py collect_answers, decode_answers: Resolver.decode,
  │                 constrained MAP, late binding, schema check                          → Decoded (ToolDecode)
  ├─ compose ────── confidence.py: factors → W, Π, L, J → tier prior → calibrator → cap  → Confidence
  ├─ decide ─────── policy.py evaluate: rules P0–P10 (+ internal WIDEN / FILL rounds)    → PolicyResult
  ├─ emit ───────── prompts.py (cards, menus), decision.py (Decision, Pending,
  │                 to_openai_message, to_anthropic_content), adapters/*                 → Decision
  └─ record ─────── trace.py build_trace; verify replays with the model out of the loop  → Trace
```

| Stage | What happens | Main code | Spec |
|---|---|---|---|
| ingest | Any tool source (OpenAI dicts, MCP `tools/list`, callables, pydantic models, LangChain tools) is normalized to a `RawTool`. Its `x-jev` layers are merged (inline > MCP `_meta` > sidecar/`jt.hints` > `Annotated` markers) and compiled into a `Catalog`. Inference assigns each slot a kind, stakes, channels and wording, and each tool a risk tier. | `spec/ingest.py`, `spec/catalog.py`, `spec/infer.py`, `spec/xjev.py`, `spec/sidecar.py`, `spec/markers.py` | §3.2, §3.3, §7.1 |
| extract | Code-only extractors find mentions in the request, history and observations: numbers, money, temporal readings, emails, places, quotes, anchors into registries and cue words. Span claiming stops two slots from reading the same words. | `extract/base.py`, `extract/*.py`, `extract/locales/` | §4.2.1 |
| pool | Each slot's resolver builds a `Pool` of `Candidate`s (label, value, text, channel, provenance) from mentions and `sources/`. It adds the sentinels, filters by the channel allow-list (I2) and shortlists to K. | `kinds/<kind>.py`, `candidates.py`, `sources/*` | §3.4, §4, §4.4 |
| plan | Each tool is checked for viability, and viable tools are speculated. The planner emits questions in the plan order of §3.5.5 (`tool`; then per tool `authorized`, `joint`, each slot's families, `done_after`), applies the token and question budget, and splits the questions into parallel calls. The strict validator rejects any Ballot that breaks a limit. | `plan.py`, `ballot.py`, `templates.py`, `qid.py`, `budget.py`, `validate.py` | §3.5, §5 |
| ask | `Ballot.to_requests(model)` produces the wire requests, and the backend answers them. On a 422, the offending question family is dropped and the call re-sent once. Any other failure fails closed (P0). | `router.py` (`_ask`, `_isolate`), `backends/*`, `wire.py` | §5.6, §8 |
| decode | Answers become per-slot value distributions: sentinels, pooling of `NOT_STATED` into defaults, and family-specific rules in each resolver. The call is chosen by constrained MAP over cross-slot constraints. Late-bound placeholders (`⟨recipient's first name⟩`, `from_account.currency`) are filled, the result is validated against the schema, and the call MAP is computed across tools. | `decode.py`, `kinds/*` (`decode`), `kinds/late.py`, `kinds/normalize.py`, `spec/constraints.py`, `spec/schema.py` | §3.6, §3.7.4 |
| compose | Factors combine into W, Π, L and J. The tier's prior composition is applied, then the isotonic calibrator when one is fitted, then the coherence cap `C ≤ W`. | `confidence.py` | §3.7 |
| decide | Ordered rules P0–P10 run with tier thresholds, gates, hysteresis and caps. `widen` and `fill` are internal actions: the router runs another round and evaluates again. | `policy.py`, `router.py` (`_policy_loop`, `_widen`, `_fill`), `kinds/widen.py`, `fallback.py` | §3.8, §4.6, §4.7 |
| emit | Prompts are templates filled with candidate labels: confirm cards, clarify menus, grid menus, open questions and refuse notices. The `Decision` carries the call, confidence, bottleneck, slots, prompt and `Pending`, and adapters translate it to other ecosystems' formats. | `prompts.py`, `decision.py`, `adapters/*`, `serve/app.py` | §3.8.4, §3.10, §7.2 |
| record | The `Trace` holds hashes, rounds with request and response bodies, bindings, factors, the composition and the rule that fired. `jt.verify` rebuilds the Ballot, re-decodes and re-applies the policy offline. | `trace.py`, `canonical.py` | §3.1, §3.9 |

**Router mechanics.** The round loop is written once, as a generator that yields effects (ask Jev, fill, escalate,
text LLM). `decide` drives it synchronously, with split calls in a thread pool; `adecide` drives it with
`asyncio.gather`. `resume` either binds a click with no Jev call, or compiles one resume round for a free-text
reply. Before a delayed call executes, it is revalidated against fresh sources (TOCTOU, §3.8.5).

**Agent loop.** `loop.Agent` calls the router in loop mode once per step (§6). It executes calls through host
executors with idempotency keys and ingests each result as an `Observation`: a preview goes into the state and the
parsed values go into the pools with the `tool_output` channel. Entities are remembered in an `EntityStore`. The
loop stops on `done`/`done_after`, a budget cap, repeat detection or lack of progress.

**Extension point.** Kind-specific logic lives in resolvers (`kinds/base.py`: `Resolver.pool`, `questions` and
`decode`, plus the optional `widen` and `clarify_values` hooks). `register_resolver(kind, resolver)` replaces or
adds one. `plan.py`, `decode.py` and `router.py` go through this interface only.

## Module map

```
src/jevtools/
  __init__.py            public API (`import jevtools as jt`, §9)
  _version.py            __version__ = "0.1.0", SPEC_VERSION = "jevtools/0.1"
  _compat.py             Python 3.10 shims (StrEnum, TOML via tomli)
  errors.py              JevtoolsError, CatalogError, BallotError, ConstraintError
  wire.py                Jev wire models: DecisionRequest/Response, Choice/Noul/Score questions and answers, Usage
  canonical.py           canonical JSON, sha256_of, NFC, round4 (§3.1)
  context.py             Context, Turn, Observation, Clock; build_state: the Jev `state` (§3.5.1, §6.2)
  candidates.py          Channel, Candidate, Pool, sentinels, labels, canonical option order (§3.4)
  templates.py           normative question templates and sentinel texts (§3.5.4)
  qid.py                 question ids: grammar, sanitizing, opaque mode (§3.5.2)
  ballot.py              Ballot, BallotQuestion, BallotOption; Ballot.to_requests (§3.5.7)
  budget.py              TokenEstimator with a running correction factor (§5.5)
  validate.py            Limits, preflight validator, probed-limits cache (§3.5.6)
  plan.py                compile_round: viability, speculation, fan-out, joint questions, budget, split (§5)
  decode.py              collect_answers, decode_answers, constrained MAP, late binding, call MAP (§3.6)
  confidence.py          Factors, compose (W, Π, L, J), tier prior, IsotonicCalibrator, call_map (§3.7)
  policy.py              Tier, Outcome, Policy (Appendix B, from_toml), evaluate: rules P0–P10 (§3.8)
  prompts.py             confirm cards, clarify/grid/yes-no/tool menus, open questions, natural call rendering (§3.8.4)
  decision.py            Decision, ToolCall, Prompt, Pending, Confidence, SlotReport; emitted formats (§3.10)
  trace.py               Trace, build_trace, verify → VerifyReport (§3.9)
  router.py              Router: decide/adecide, resume/aresume, compile; the round loop and internal rounds (§9.1)
  loop.py                Agent, LoopBudget, LoopResult, EntityStore, ingest_observation (§6)
  fallback.py            Filler, Escalator, TextLLM protocols + OpenAI-compatible reference implementations (§4.7)
  compat.py              cookbook_policy, from_jev_fn: migration from existing Jev usage (§7.4)
  probe.py               conformance probe: measure undocumented wire limits (§8.7, E1)
  cli.py                 `jevtools lint | explain | verify | probe | serve | eval | tune | fixtures`
  spec/
    xjev.py              the x-jev vocabulary as strict models (§3.2)
    ingest.py            any tool source → RawTool (§7.1)
    catalog.py           Catalog (from_openai/from_mcp/from_callables/from_pydantic/from_langchain), strip_xjev
    models.py            ToolSpec, SlotSpec
    infer.py             slot kind (§3.3.1) and risk tier (§3.3.2) inference
    sidecar.py           jevtools.json/.yaml sidecars and jt.hints
    markers.py           Annotated markers: Ref, Span, Text, Quantity, Money, When, CodeList, ListOf, Default, …
    decorators.py        @jt.tool
    schema.py            dependency-free JSON Schema validator
    constraints.py       cross-slot constraints and registered @checks
  extract/
    base.py              Mention, Mentions, span claiming, run_extractors (§4.2.1)
    tokens.py            shared tokenization
    numbers.py money.py temporal.py patterns.py places.py text.py cues.py coref.py   one extractor family each
    catalogs.py          packaged data: ISO 4217/3166/639, compact city gazetteer, template packs (data/)
    locales/             en (complete), de, fr (basic) lexicons
  kinds/
    base.py              Resolver protocol, ResolveContext, SlotResult, register_resolver, shared Choice machinery
    common.py            shared code of the Choice-based resolvers
    enum.py flag.py ordinal.py quantity.py money.py temporal.py span.py ref.py listing.py record.py text.py derived.py
                         one resolver per kind (§4.2); ref.py also covers superlatives (§4.2.8)
    widen.py             coverage rounds: buckets, groups, hierarchy (§4.6)
    late.py              late-binding recipes (§5.4)
    normalize.py         versioned normalizers (§4.3)
  sources/
    base.py              Source protocol, SourceQuery (§4.4)
    registry.py          Registry: app-owned rows
    files.py             FileIndex: workspace paths, BM25, directory hierarchy
    provider.py          Provider: a host callable
    toolsource.py        ToolSource: a catalog tool called once per session and cached
    mcp.py               MCPResources: an MCP server's resources
    retrieval.py         fuzzy anchor matching, trigram similarity, BM25 (pure Python)
    specs.py             sources described as data (jevtools.toml, eval datasets, CLI)
  backends/
    base.py              Backend protocol (model, name, decide, adecide) (§8.1)
    http.py              HTTPBackend: TypeSafe(), OpenRouterSystemOne(), OpenRouterDecisions() (§8.2)
    auto.py              auto(allow_offline=False) (§8.3)
    errors.py            typed backend errors (§8.4)
    scripted.py          ScriptedBackend (+ from_fixture) (§8.5)
    simulator.py         LexicalSimulator: offline lexical test double, not a model (§8.6)
    cassette.py          Cassette record/replay (§8.7)
  adapters/
    openai.py            complete/acomplete, wrap, error mapping (§7.2.1, §7.2.2)
    anthropic.py         tool_use emitter, Anthropic tool ingest (§3.10)
    langchain.py         JevChatModel, confirm_node (§7.2.3)
    pydantic_ai.py       JevModel (§7.2.5)
    mcp.py               call_decision, result → observation (§7.1)
    pending.py           PendingStore protocol, InMemoryPendingStore, prefix-hash matching (§7.2.4)
    _router.py           per-request router derivation and context overrides
  serve/
    app.py               create_app: the OpenAI-compatible proxy (§7.2.4)
    config.py            ServeConfig from jevtools.toml
  eval/
    dataset.py           EvalCase JSONL (§11.1)
    harness.py report.py metrics.py stats.py   run cases, score decisions, ECE/Brier/risk-coverage, exact statistics
    tuning.py            threshold tuning (Clopper–Pearson, CRC), calibrators, certification (§11.3, §11.4)
    experiments.py       live experiments E1–E10 (§11.2)
  demo/
    scenario.py          the §13 world: synthetic contacts, accounts, 3,000 paths, six tools, fake Workspace
    scripts.py           the §13.3 answer scripts (illustrative numbers)
```

## Tests and fixtures

- `tests/unit`, `tests/backends`, `tests/policy`, `tests/scenario`, `tests/loop`, `tests/adapters`, `tests/serve`
  and `tests/eval` hold the offline tests. They never use the network. Tests marked `live` need a key.
- `tests/examples` runs every example in scripted and simulator mode. These tests are marked `fast`.
- `tests/golden/<case>/` holds the conformance fixtures of §10.2. `jevtools fixtures [--update]` checks or
  regenerates them, and a port must reproduce their bytes. See `tests/golden/README.md`.
- Scripted and simulated answers exercise the plumbing and the policy branches. They are never evidence about Jev's
  accuracy.
