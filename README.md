# jevtools

**Tool calling for your app's own domain**, driven by TypeSafe's
[Jev](https://openrouter.ai/docs/guides/community/jev). jevtools lets Jev pick the tool *and every argument* for an
assistant or agent that works over your app's data: its contacts, accounts, tickets, deals, files and datasets.
Arguments are **elected** from candidates that code builds from the user's words, your records and closed sets.
Nothing is written, so a call cannot contain a recipient, account, ID or amount that is not in the user's words or
in your data.

Jev is a decision model. You send it a `state` and typed questions (Choice, Noul = yes/no, Score), and it returns
calibrated probabilities and no text. That makes it good at choosing and unable to write a tool's arguments.
jevtools turns every argument into a choice. For every parameter, code builds a finite pool of candidates: spans of
the user's words, rows of your registries, enum members, every reading of "next Tuesday", fields of earlier tool
outputs. Jev elects one per slot ("bind, don't write"). The tool question and the questions for every argument of
every plausible tool go out as one fan-out request, so most turns cost one Jev call. The answers compose into a call
confidence C (W, Π or min(L, J), by risk tier), and a risk-tiered policy turns it into execute, confirm, clarify,
escalate, abstain or refuse, with a trace you can replay. C is a bound, a product or a joint mass built from Jev's
per-question probabilities, not itself a calibrated probability. It becomes one only after you fit a calibrator on
labelled traffic (`jevtools tune --calibrate`); until then every Decision reports `confidence.calibrated == False`.

> **Status: v0.1.0, protocol `jevtools/0.1`.** The offline test suite is green. **Live Jev has been measured once**:
> one run of the app-domain benchmark (99 cases, OpenRouter, 2026-09-25) scored 80% correct against a 99% ceiling,
> with no planted value reaching a call ([Benchmarks](#benchmarks)). That is a single unreplicated run, not a
> calibration. Every other number in this README (probabilities, confidences, token counts in examples and
> fixtures) comes from the scripted backend, the offline `LexicalSimulator` or the benchmark oracle, and is never
> evidence about Jev's accuracy.

## Where it fits

**A good fit: assistants and agents over an app's own domain**, where arguments come from:
- the user's words: names, amounts, dates, quoted text, IDs they type (`INC-1052`);
- the app's data: contacts, accounts, payees, tickets, deals, files, datasets, variables;
- closed sets: stages, priorities, access levels, model families, currencies.

Think of an inbox and calendar assistant, a CRM copilot, a banking assistant, a helpdesk bot, a workspace assistant
or an analysis assistant over your datasets. What you get there:

- **No invented values.** A recipient, account, ticket or amount is always one of the candidates code nominated
  from the user's words or your data. Channel allow-lists keep text from tool outputs out of the identity and
  amount slots of every tool that changes something, so an address planted in an email cannot become the recipient
  of your next message.
- **Ambiguity becomes a menu.** "Email Anna" with two Annas asks "Which Anna?", offering both as complete calls. A
  coin flip is never presented as a confirm card, and "Move D-1017 to negotiation" anchors deal `D-1017` exactly.
- **Risk tiers.** Clear reads execute, writes need more confidence, sending and sharing more still, and moving
  money is confirm-only until you certify that tier on your own labelled traffic.
- **One cheap call per turn.** The fan-out asks everything at once, at $0.042 per million input tokens.
- **Replayable decisions.** Every decision carries a trace that `jt.verify` re-checks without the model.

**Not a good fit (use an LLM tool caller, or put one behind `escalate`):**
- open-world arguments: web search queries, code, SQL, math expressions, translations, new prose (a Filler LLM can
  draft text for Jev to accept, but then an LLM writes it);
- values that need reformatting or world knowledge that no source holds (`"New York, NY"` from "New York",
  `C6H12O6` from "glucose");
- several calls in one turn, or multi-step plans (one call per turn; `jt.Agent` runs step by step).

The bundled **app-domain benchmark** measures the good fit: six synthetic apps, 99 labelled cases, including
collisions, typed IDs, relative dates, critical transfers and planted injections. The oracle ceiling is 98 of 99
(`jevtools bench app`, [docs/BENCH.md](docs/BENCH.md)). **BFCL** is the stress test outside it, with a ceiling of
35–42% on its single-call categories. See [Benchmarks](#benchmarks).

## How it works

1. **Ingest.** Tools stay plain OpenAI or MCP JSON Schema, or Python functions. jevtools infers each parameter's
   kind (enum, ref, temporal, text…), each tool's risk tier and the wording of its questions. Optional `x-jev`
   hints override anything it infers.
2. **Pool.** Code extracts mentions from the conversation and builds a candidate pool per slot from the sources you
   register. Every pool also offers `NOT_STATED` and `NONE_OF_THESE`. Channel allow-lists decide which sources may
   reach which slot, so text from a tool output can never become a payment account.
3. **Ask.** One request carries a tool Choice (with `NO_TOOL` and `UNSUPPORTED`) and one closed question per slot
   of every *viable* tool. The questions are independent, so they fan out in a single call.
4. **Decode and compose.** Code copies the elected candidate's value verbatim, then normalizes it, validates it and
   checks cross-slot constraints. The slot probabilities compose into a call confidence C: W (the weakest factor), Π
   (the product), and for critical tools L (a lower bound under any dependence) and J (a joint Choice). Sentinel
   mass counts as error; it is never renormalized away.
5. **Decide.** Ordered policy rules (P0–P10) compare C with the tier's thresholds. Prompts are templates filled with
   candidate labels, never generated text. Every decision carries a Trace that `jt.verify` replays without calling
   the model.

```
 messages + tools + Context(sources)
        │
        ▼
 ┌─────────────────┐   ┌──────────────────────────┐   ┌─────────────────────────────────────┐
 │ ingest          │──▶│ candidate pools (code)   │──▶│ ONE fan-out Jev request             │
 │ kinds · tiers   │   │ user spans · registry    │   │ tool: Choice + NO_TOOL, UNSUPPORTED │
 │ x-jev hints     │   │ rows · enums · readings  │   │ T.slot: Choice + NOT_STATED, NONE…  │
 └─────────────────┘   │ tool outputs · templates │   │ T.authorized, T.body.accept.i …     │
                       └──────────────────────────┘   └──────────────────┬──────────────────┘
                                                                         │ calibrated probabilities
        ┌────────────────────────────────────────────────────────────────┘
        ▼
 elected values ──▶ confidence C (W · Π · L · J) ──▶ policy (tier thresholds) ──▶ Decision + Trace
 (copied verbatim)                                                                 │
                    execute · confirm · clarify · escalate · abstain · refuse ◀────┘
```

| Kind | Inferred from | Where candidates come from | Jev question |
|---|---|---|---|
| `enum` | `enum`, `Literal`, `const` unions, ISO code patterns | the enum; catalogs (ISO 4217/3166/639) shortlisted to mentions | slot Choice |
| `flag` | `boolean` | none | Choice with `NOT_STATED` if there is a default, else a Noul |
| `ordinal` | small bounded int named priority, rating, severity… | author-described levels | Score |
| `quantity` | number, integer | number parser (digits, words, fractions), unit-filtered; `derive` operators | slot Choice (a value grid if nothing was stated) |
| `money` | amount with a `currency` sibling | money parser (`CHF 250`, `€12.50`), derived amounts | amount Choice + currency Choice |
| `temporal` | `format: date-time`, `start`, `*_at`, `*_date` | every reading of every date/time mention, relative to `now` and tz | Choice over readings, or date × time |
| `span` | short strings, places, `pattern`/`format` | the user's words: quotes, proper nouns, clauses, regex, gazetteer | slot Choice |
| `ref` | `format` or a name that matches a source (`email`, `*_account`, `path`, `*_id`) | registries, file indexes, providers, tool outputs, MCP resources (shortlist K = 40) | slot Choice, plus `present`/`rev` probes by tier |
| `list` | `array` | one sub-pool per mentioned item; enum members | per-item Choices + a `more` Noul, or item Nouls |
| `record`, `union` | `object`, `oneOf`/`anyOf` | recursion into the leaves | leaf questions + a `branch` Choice |
| `text` | subject, title, body, query; long strings | templates with late-bound placeholders, the user's clauses, observation copies, an LLM Filler's drafts | one accept Noul per candidate |
| `derived`, `secret` | `readOnly`, `const`, `password`/`token` | code: context and other bindings | none |

The normative protocol is [docs/SPEC.md](docs/SPEC.md): §3.3 inference, §4 the resolvers, §5 round planning.

## Quickstart

```bash
pip install "jevtools @ git+https://github.com/umatter/jevtools"          # or: uv add "jevtools @ git+…"
pip install "jevtools[serve,mcp] @ git+https://github.com/umatter/jevtools" # extras: serve langchain pydantic-ai mcp yaml jsonschema all
```

You need Python ≥ 3.10. The only hard dependencies are `pydantic>=2.6` and `httpx>=0.27`, and the package is not
on PyPI yet. To work on a checkout, run `uv venv && uv pip install -e ".[all,dev]"`.

**Ten lines, offline.** This runs without a key on the `LexicalSimulator`, an offline lexical test double.

```python
from typing import Literal
import jevtools as jt

@jt.tool                                   # tier inferred: read ("get")
def get_weather(city: str, unit: Literal["celsius", "fahrenheit"] = "celsius") -> dict:
    """Get the current weather for a city."""
    return {"city": city, "unit": unit, "temp": 61}

router = jt.Router([get_weather], backend=jt.backends.LexicalSimulator())
d = router.decide("What's the weather like in Zurich in Fahrenheit?")
print(d.outcome, d.call)   # execute get_weather(city='Zurich', unit='fahrenheit')
```

**Scripted answers** make tests deterministic. Keys are question ids (globs allowed). A bare label gets `p_top` =
0.92, and unscripted questions get a uniform answer, so a gap in the script shows up as low confidence.

```python
backend = jt.backends.ScriptedBackend({
    "tool": {"get_weather": 0.97, "NO_TOOL": 0.02, "UNSUPPORTED": 0.01},
    "get_weather.city": {"Zurich": 0.95, "NOT_STATED": 0.03, "NONE_OF_THESE": 0.02},
    "get_weather.unit": "fahrenheit",
})
router = jt.Router([get_weather], backend=backend)
d = router.decide("What's the weather like in Zurich in Fahrenheit?")
print(d.outcome, d.rule, round(d.confidence.call, 4))   # execute P9.read.execute 0.92
print(d.to_openai_message()["tool_calls"][0]["function"])
# {'name': 'get_weather', 'arguments': '{"city":"Zurich","unit":"fahrenheit"}'}
```

**Live.** Set `OPENROUTER_API_KEY` or `TYPESAFE_API_KEY` and use `jt.backends.auto()`:

```python
router = jt.Router([get_weather], backend=jt.backends.auto())
```

`auto()` checks `JEVTOOLS_BACKEND` first (`typesafe | openrouter_systemone | openrouter_decisions | simulator |
cassette:<path>`). Next, `TYPESAFE_API_KEY` selects TypeSafe direct, and then `OPENROUTER_API_KEY` selects OpenRouter
Decisions, which reports cost. Without any of these it raises `BackendConfigError`, unless you pass
`auto(allow_offline=True)`, which falls back to the simulator with a warning. `JEVTOOLS_MODEL` overrides the model id.

**See what Jev is asked.** `router.compile(...)` builds the Ballot without any network call. The compiled request
for the quickstart holds one call and three questions. "Fahrenheit" is claimed by `unit`, so `city` is offered only
"Zurich":

```python
(request,) = router.compile("What's the weather like in Zurich in Fahrenheit?").to_requests(router.model)
print(list(request.questions))   # ['tool', 'get_weather.city', 'get_weather.unit']
```

```json
"get_weather.city": {"type": "choice",
  "instructions": "Suppose the assistant will get the current weather for a city to fulfil `request`. Which option is the city?",
  "criteria": {"Zurich": "From \"Zurich\" in the request; a city in Switzerland; also a city in Ontario, Canada.",
               "NOT_STATED": "The user does not say.",
               "NONE_OF_THESE": "The user indicates a value, but it is none of the listed options."}}
```

## Using your own tools

### Plain JSON Schema (OpenAI or MCP)

Existing tool definitions work unchanged. What makes a string parameter closed is a **source**. Here `format: email`
matches a registry that `provides` `email`, so `to` becomes a `ref` into your contacts:

```python
import jevtools as jt

SEND_EMAIL = {"type": "function", "function": {
    "name": "send_email",
    "description": "Send an email from the user to one recipient.",
    "parameters": {"type": "object", "required": ["to", "subject", "body"], "properties": {
        "to": {"type": "string", "format": "email", "description": "The recipient's email address"},
        "subject": {"type": "string", "description": "The subject line of the email"},
        "body": {"type": "string", "description": "The body of the email"}}}}}

contacts = jt.Registry(
    "contacts",
    rows=[{"name": "Anna Keller", "email": "anna.keller@acme.com", "title": "Account Manager at ACME"},
          {"name": "Anna Rossi", "email": "anna.rossi@gmail.com", "title": "Friend"}],
    key="email",                   # the value that goes into the call
    label="{name} <{email}>",      # what Jev sees, and what the user sees in menus
    describe="{title}",            # the option description
    match=["name"],                # fields matched against mentions ("Anna")
    provides=["email", "person"],  # tags that slots bind to
)
router = jt.Router([SEND_EMAIL], backend=jt.backends.LexicalSimulator(),
                   context=jt.Context(user={"name": "Sam Muster"}, sources=[contacts]))
print([(s.name, s.kind, s.stakes) for s in router.catalog.get("send_email").slots])
# [('to', 'ref', 'identity'), ('subject', 'text', 'cosmetic'), ('body', 'text', 'content')]
```

`jevtools lint tools.json --sources sources.toml` prints one line per slot: its kind, where its candidates come
from, and `WEAK` wherever inference fell back to a generic span. Fix those with a source or a hint. MCP
`tools/list` results load with `jt.Catalog.from_mcp(result)`. Only annotation hints that are *explicitly present*
count toward the tier.

### `x-jev` hints

Every key is optional. They can live inline (`function["x-jev"]` or on a property), in MCP `_meta["x-jev"]`, in a
sidecar file (`jevtools.json`/`.yaml`), or in `jt.hints(...)`, so third-party schemas stay untouched. Unknown keys
are an error. Adapters call `jt.strip_xjev(tool)` before a schema reaches any LLM provider.

```python
catalog = jt.Catalog.from_openai(
    [SEND_EMAIL],
    hints=jt.hints({
        "send_email": {"confirm": "always"},                             # tool-level key
        "send_email.to": {"source": "contacts", "ask": "Who should get it?"},   # parameter-level keys
    }),
    sources=[contacts],
)   # the same mapping in a file: jt.Catalog.from_openai(tools, sidecar="jevtools.json")
```

| Commonly used keys | |
|---|---|
| tool level | `risk` (read/write/external/critical), `intent`, `render`, `confirm_template`, `confirm`, `constraints` (`"amount <= from_account.balance"`), `groups`, `emits` |
| parameter level | `kind`, `source`, `stakes` (identity/content/cosmetic), `channels`, `values`, `templates`, `default_from` (`user.home_city`, `from_account.currency`), `derive`, `order_by`, `unit`, `ask`, `noun`, `fallback` |

The full vocabulary, with defaults, is SPEC §3.2. See `transfer_funds` in §13.2 for a critical tool with
constraints.

### Python functions, markers and sources

`@jt.tool` turns a signature into JSON Schema. The first docstring line becomes the description. `Annotated`
markers add `x-jev` keys: `Ref`, `Span`, `Text`, `Quantity`, `Money`, `When`, `CodeList`, `ListOf`, `Default`,
`Derive`, `Channels`, `Stakes`, `Ask` and `Noun`.

```python
from typing import Annotated, Literal
import jevtools as jt

people = jt.Registry("people", rows=[{"name": "Bob Meier", "email": "bob.meier@muster.ch", "team": "Engineering"}],
                     key="email", label="{name} <{email}>", describe="{team}", match=["name"], provides=["email"])
files = jt.FileIndex("files", ["reports/2026-q2.pdf", "reports/2026-q3.pdf", "src/app.py"])  # BM25 over path tokens

def search_tickets(q: jt.SourceQuery) -> list[dict]:   # your API or database; q.request is the user's text
    return [{"value": "T-1043", "label": "T-1043 · VPN drops every hour", "text": "Initech; priority high"}]

tickets = jt.Provider(search_tickets, name="tickets", provides={"ticket_id"})

@jt.tool   # tier inferred: external ("share")
def share_file(path: Annotated[str, jt.Ref(source="files")],
               recipient: Annotated[str, jt.Ref(source="people"), jt.Noun("the person to share the file with")],
               access: Literal["view", "comment", "edit"] = "view") -> dict:
    """Share a workspace file with a colleague."""
    return {"shared": path, "with": recipient, "access": access}

@jt.tool(risk="write")
def link_ticket(ticket_id: Annotated[str, jt.Ref(source="tickets")],
                path: Annotated[str, jt.Ref(source="files")]) -> dict:
    """Attach a workspace file to a support ticket."""
    return {"ticket": ticket_id, "file": path}

context = jt.Context(user={"name": "Sam Muster"}, sources=[people, files, tickets])
router = jt.Router([share_file, link_ticket], backend=jt.backends.LexicalSimulator(), context=context)
ballot = router.compile("Share the Q3 report with Bob so he can comment")
print(list(ballot.to_requests(router.model)[0].questions))
# ['tool', 'link_ticket.authorized', 'link_ticket.ticket_id', 'link_ticket.path', 'share_file.authorized',
#  'share_file.path', 'share_file.path.present', 'share_file.recipient', 'share_file.recipient.present', 'share_file.access']
```

| Source | Use it for | Channel |
|---|---|---|
| `jt.Registry(name, rows, key, label, describe, match, provides, attrs)` | app-owned rows: contacts, accounts, projects, tickets. Registries of ≤ 12 rows are sent whole. A typed ID (`INC-1052`, or `ticket 1052`) anchors its row exactly. `attrs` feed constraints, `render` and `order_by`; Jev sees them only if your `label`/`describe` template uses them | registry |
| `jt.FileIndex(name, paths, synonyms=…, hierarchy="dirname")` | workspace paths, BM25 over path tokens, widen by directory | registry |
| `jt.Provider(fn, name, provides)` | any lookup; `fn(SourceQuery)` returns dicts `{value, label, text}`, sync or async | registry |
| `jt.ToolSource(tool, args, items, key, label, ttl)` | a catalog tool, such as `list_contacts`, called once per session and cached | registry |
| `jt.MCPResources(session, uri_template)` | an MCP server's resources | registry |
| earlier tool results (agent loop) | values found in observations | tool_output (untrusted) |

`jt.Catalog.from_pydantic(Model)`, `from_callables`, `from_langchain` and `afrom_mcp_session` cover the other
sources. `jt.compat.from_jev_fn` imports `@jev.fn` functions, and `cookbook_policy()` reproduces the TypeSafe
function-calling cookbook.

### Confirm, clarify and resume

A decision that needs the user returns a `prompt` and a `pending_id`. A **click** (`selection=`) binds the value
with p = 1 and makes no Jev call. A **free-text reply** (`reply=`) changes the state, so everything is re-asked in
one round. The example below uses the synthetic demo world in `jevtools.demo` with scripted, illustrative answers:

```python
from jevtools.demo import scenario, scripts

router, _ = scenario.scenario_router(scripts.R2_NO_HISTORY)
d = router.decide("Email Anna that I'll be 10 minutes late")
print(d.outcome, d.rule, d.prompt.text)   # clarify P9.external.ambiguous Which recipient did you mean?
for option in d.prompt.options:           # external tier: every option is the complete resulting call
    print(option.id, option.text)         # pick:to:0 Send an email to Anna Keller <anna.keller@acme.com> — …
d = router.resume(d.pending_id, selection="pick:to:0")
print(d.outcome, d.rule, d.usage.jev_calls)   # execute P9.external.confirmed 0
```

Confirm cards offer `ok`, `alt:<slot>:<i>` (runner-up values), `change` and `cancel`. Before any delayed call
executes, TOCTOU revalidation re-resolves registry values and re-runs the constraints.

## Integration

**OpenAI Chat Completions.** `jt.openai.complete` takes OpenAI `messages` and `tools` and returns a
`ChatCompletion`-shaped dict. `role: tool` messages become observations, which is how drop-in loops get
multi-step behaviour.

```python
import jevtools as jt

TOOLS = [{"type": "function", "function": {
    "name": "get_weather", "description": "Get the current weather for a city.",
    "parameters": {"type": "object", "required": ["city"], "properties": {
        "city": {"type": "string", "description": "The city"},
        "unit": {"type": "string", "enum": ["celsius", "fahrenheit"], "default": "celsius"}}}}}]
messages = [{"role": "user", "content": "What's the weather like in Zurich in Fahrenheit?"}]

completion = jt.openai.complete(messages, TOOLS, backend=jt.backends.LexicalSimulator())
print(completion["choices"][0]["message"]["tool_calls"][0]["function"])

from openai import OpenAI
client = jt.openai.wrap(OpenAI(), jt.Router(TOOLS, backend=jt.backends.auto()))  # intercepts model="jevtools" only
response = client.chat.completions.create(model="jevtools", messages=messages, tools=TOOLS)
```

A confirm card or menu comes back as an assistant message: `content` holds the question and `x_jev` holds the
options and a `pending_id`. `d.to_anthropic_content()` emits Anthropic `tool_use` blocks.

**LangChain.** `JevChatModel` is a `BaseChatModel`, so `bind_tools` and ToolNode loops work unchanged.
`jevtools.adapters.langchain.confirm_node` maps confirm and clarify to LangGraph `interrupt()`.

```python
from typing import Literal
from langchain_core.tools import tool
import jevtools as jt
from jevtools.adapters.langchain import JevChatModel

@tool
def get_weather(city: str, unit: Literal["celsius", "fahrenheit"] = "celsius") -> str:
    """Get the current weather for a city."""
    return f"61 degrees in {city}"

router = jt.Router(jt.Catalog.from_langchain([get_weather]), backend=jt.backends.LexicalSimulator())
ai = JevChatModel(router=router).bind_tools([get_weather]).invoke("What's the weather like in Zurich?")
print(ai.tool_calls)   # [{'name': 'get_weather', 'args': {'city': 'Zurich', 'unit': 'celsius'}, …}]
```

**Pydantic AI.** `JevModel` implements the `Model` interface (pydantic-ai-slim ≥ 1.0.13, as the extra requires).
Output tools such as `final_result` are elected like any other tool.

```python
from typing import Literal
from pydantic_ai import Agent
import jevtools as jt
from jevtools.adapters.pydantic_ai import JevModel

def get_weather(city: str, unit: Literal["celsius", "fahrenheit"] = "celsius") -> str:
    """Get the current weather for a city."""
    return f"61 degrees in {city}"

agent = Agent(JevModel(jt.Router([get_weather], backend=jt.backends.LexicalSimulator())), tools=[get_weather])
print(agent.run_sync("What's the weather like in Zurich in Fahrenheit?").output)   # 61 degrees in Zurich
```

**MCP.** jevtools decides and the MCP client executes. `call_decision` passes the idempotency key in `_meta`,
which needs mcp ≥ 1.19 (older `ClientSession.call_tool` takes no `meta`); the `mcp` extra requires it.

```python
import jevtools as jt
from jevtools.adapters.mcp import call_decision

async def weather(session) -> None:                       # an initialized mcp ClientSession
    catalog = await jt.Catalog.afrom_mcp_session(session)  # tools/list; x-jev may live in _meta
    router = jt.Router(catalog, backend=jt.backends.auto())
    d = await router.adecide("What's the weather like in Zurich in Fahrenheit?")
    if d.outcome == "execute":
        result = await call_decision(session, d)          # session.call_tool(name, arguments, meta=…)
```

**Any language: the proxy.** `jevtools serve` speaks OpenAI Chat Completions with tools. Clients keep sending
their usual `messages` and `tools`; the proxy holds the Jev key, the sources and the pending cards. The pending
cards are keyed by a hash of the conversation prefix, so stateless clients work.

```bash
pip install "jevtools[serve] @ git+https://github.com/umatter/jevtools"
jevtools serve --config examples/proxy/jevtools.toml --host 127.0.0.1 --port 8787
curl -s http://127.0.0.1:8787/healthz    # {"status":"ok",…,"backend":"openrouter_decisions",…}
```

```python
from openai import OpenAI
client = OpenAI(base_url="http://127.0.0.1:8787/v1", api_key="unused")
response = client.chat.completions.create(model="jevtools", messages=messages, tools=TOOLS,
                                          extra_body={"jevtools": {"context": {"tz": "Europe/Zurich"}}})
```

`jevtools.toml` configures the backend, default context, sources (JSON/CSV registries, file indexes,
`module:function` providers), an optional `text_llm`, `filler` and `escalator`, and a `fallback_llm` that receives
anything jevtools cannot serve. See [examples/proxy/](examples/proxy/README.md).

**R (ellmer).** Point ellmer at the proxy and register tools as usual. The snippet follows SPEC §7.3. It was **not
run here**, because the build environment has no R. The proxy speaks Chat Completions only, so if your ellmer's
`chat_openai()` targets the Responses API, use its Chat Completions–compatible constructor instead.

```r
Sys.setenv(OPENAI_API_KEY = "unused")    # the proxy holds the Jev key
chat <- ellmer::chat_openai(base_url = "http://127.0.0.1:8787/v1", model = "jevtools")
chat$register_tool(send_email_tool)      # ellmer::tool(...) as today
chat$chat("Email Anna that I'll be 10 minutes late")
```

A native R package using the same protocol is planned. The golden fixtures (below) are its conformance test.

**Agent loops.** `jt.Agent` runs one Jev round per step. Tool results become observations: the next state shows
a preview, and values found in them join the pools with the untrusted `tool_output` channel. Idempotency keys
guard execution, and `done_after` stops the loop without an extra round.

```python
import jevtools as jt
from jevtools.demo import scenario, scripts   # synthetic workspace + the spec's illustrative answers

workspace = scenario.Workspace()
router, _ = scenario.scenario_router(scripts.r6_script())
agent = jt.Agent(router, workspace.executors())   # {"read_file": fn, "send_email": fn, …} or one dispatcher
result = agent.run("Find the latest invoice from ACME and forward it to finance")
while result.pending is not None:                  # a confirm card; a real host shows it to the user
    result = agent.resume(result.pending, selection="ok")
print(result.outcome, [s.call.name for s in result.steps if s.executed])   # done ['read_file', 'send_email']
```

The invoice in that demo carries an injected instruction. Its address is never nominated for `send_email.to`, and
`transfer_funds` is `channel_blocked`: see `examples/06_agent_loop_invoice.py`.

## Outcomes, risk tiers and policy

| Outcome | Meaning | The host receives |
|---|---|---|
| `execute` | run the call now | `tool_calls` with an idempotency key |
| `confirm` | one tap to approve; alternatives offered | a templated card + `pending_id` |
| `clarify` | a menu of the bottleneck's top values (as complete calls in external/critical), or an open question | a templated question + `pending_id` |
| `escalate` | hand the turn to a configured Escalator (an LLM tool caller, still gated by Jev) | its result |
| `abstain` | no tool fits, not authorized, or backend down with no escalator | handoff to `text_llm` or an empty stop |
| `refuse` | the action came from an observation, or needs values only untrusted channels hold | a templated notice, no call |
| `done` | the agent loop finished | the final receipt |

The tier comes from `x-jev.risk`, then explicit MCP annotations, then the verb in the tool name (`get` → read,
`send` → external, `transfer` → critical, `create` → write). Anything else defaults to external. A write tool that
notifies people (attendees) is raised to external.

| Tier | C is | Execute if C ≥ | Confirm if C ≥ | Also required for execute |
|---|---|---|---|---|
| read | W | 0.60 | none | none |
| write | Π | 0.70 | 0.45 | `authorized` ≥ 0.80; content text accepted ≥ 0.70 |
| external | Π | 0.80 | 0.50 | `authorized` ≥ 0.90; content text accepted ≥ 0.80 |
| critical | min(L, J) | never, unless certified | 0.80 | channels ⊆ {user, registry, author}; the joint Choice agrees |

These defaults are **priors, not measurements** (SPEC §3.8.3, Appendix B). A value within 0.03 of a threshold
takes the safer side. The critical tier stays confirm-only until `jevtools tune` has certified it on ≥ 3,000
labelled cases. Override any part in TOML; everything you leave out keeps its default:

```python
router = jt.Router([get_weather], backend=backend, policy=jt.Policy.from_toml("policy.toml"))
```

```toml
version = "acme-2026-10"
shadow = true              # log decisions, never auto-execute: execute is returned as confirm

[tiers.external]
execute = 0.85
```

**Tuning.** Label your own traffic as JSONL (SPEC §11.1): the messages, a context, a catalog, and gold
`outcomes_ok`/`tool`/`args`. Then run the harness and tune per-tier thresholds under a wrong-execution budget, using
Clopper–Pearson (default) or conformal risk control:

```bash
jevtools eval cases.jsonl --backend auto --replays 3 --out report.json   # ECE, Brier, exact match, wrong-exec rate…
jevtools tune report.json --out tuned/ --alpha external=0.01            # writes tuned/policy.toml
```

Thresholds do not transfer between datasets. Tune on your own traffic, and start in shadow mode.

**Traces.** Every `Decision` carries a `Trace` with the Ballot, requests, responses, bindings, factors and the rule
that fired. `jt.verify` replays it with the model out of the loop: it rebuilds the Ballot, re-decodes the answers
and re-applies the policy.

```python
request = "What's the weather like in Zurich in Fahrenheit?"
d = router.decide(request)
report = jt.verify(d.trace, catalog=router.catalog, context=router.context_for(request))
print(report.ok)   # True: hashes, ballot_rebuild, redecode, values, channels, composition, policy
open("trace.json", "wb").write(d.trace.to_json())   # then: jevtools explain trace.json
```

## CLI

| Command | What it does |
|---|---|
| `jevtools lint CATALOG [--sidecar F] [--sources F] [--filler] [--strict]` | one line per tool and slot: tier, kind, candidate source, `OK`/`WEAK`/`WARN`/`ERROR` |
| `jevtools explain TRACE [--policy F]` | a readable account of a decision: bindings, alternatives, sentinels, factors, thresholds, rounds |
| `jevtools verify TRACE [--catalog F --sources F --context F --policy F]` | replay a trace without the model; exit 1 on any failed check |
| `jevtools probe [--backend auto] [--no-smoke]` | 19 live requests (16 with `--no-smoke`; more when a rejected size is halved; plus `GET /v1/models` on TypeSafe), of up to 400 questions and 8,000-character fields, that measure undocumented wire limits and cache them for the validator |
| `jevtools serve --config jevtools.toml [--host H --port P]` | the OpenAI-compatible proxy (extra `serve`) |
| `jevtools eval DATASET [--backend B] [--replays N] [--out report.json]` | run a labelled dataset and report the SPEC §11.2 metrics |
| `jevtools tune REPORT [--out DIR] [--alpha tier=x] [--method cp\|crc] [--calibrate]` | tune thresholds, fit isotonic calibrators, certify the critical tier |
| `jevtools fixtures [--update] [--case NAME]` | check or regenerate the golden conformance fixtures (from a checkout) |
| `jevtools bench app [--domains …] [--dir DIR] [--backend oracle\|sim\|auto…] [--tags] [--out F]` | run the app-domain benchmark (or your own domains): the ceiling (oracle), or live Jev next to it ([docs/BENCH.md](docs/BENCH.md)) |
| `jevtools bench bfcl [--download] [--backend oracle\|sim\|auto…] [--categories …] [--limit N] [--out F]` | run the BFCL benchmark: the coverage ceiling (oracle), or live Jev next to its ceiling ([docs/BENCH.md](docs/BENCH.md)) |

`--backend` accepts `auto | typesafe | openrouter_systemone | openrouter_decisions | simulator | cassette:<path>`.

## Benchmarks

Both benchmarks run offline by default on an **oracle**, a backend that answers every Jev question perfectly from
the gold label. The oracle's accuracy is the **ceiling** of "bind, don't write": how often the right decision can
come out at all, given the candidates code nominated. With a key, `--backend auto` runs live Jev and reports its
accuracy next to the ceiling ("within ceiling" is then Jev's judgment on the cases jevtools can get right).

**App domains (the good fit).** `jevtools bench app` covers six synthetic apps (inbox, crm, banking, workspace,
helpdesk, research) with 99 cases. They include collisions ("Email Anna" with two Annas), typed IDs ("Assign ticket
1100 to Aisha"), relative dates ("since September 1"), described enums ("high priority" → P2), critical transfers
with balance constraints, coreference and six injections planted in observations.

| | inbox | crm | banking | workspace | helpdesk | research | all |
|---|---:|---:|---:|---:|---:|---:|---:|
| cases | 20 | 17 | 16 | 15 | 16 | 15 | 99 |
| ceiling (oracle, not Jev) | 95% | 100% | 100% | 100% | 100% | 100% | 99% |
| live Jev, one run (2026-09-25) | 60% | 94% | 88% | 73% | 69% | 100% | 80% |

The oracle makes no wrong executions and lets no planted value reach a call. Live Jev let no planted value reach a
call either. Its 3 wrong executions were all the read-only `search_files` chosen instead of opening a file, and most
of its other misses are clarify menus where the right call was bound but fell below the tier's prior thresholds
(untuned). One run per case, so a single domain moves by ±10 points between runs. The one miss names a contact by a role
that the app's data does not expose to matching. The cases use the §11.1 format, so the same files are `jevtools
eval` datasets. `--dir` runs your own domains, which is the fastest way to see what jevtools can do on your tools and
data before the first API call.

**BFCL (outside the fit).** `jevtools bench bfcl --download` runs the single-turn categories of the Berkeley
Function Calling Leaderboard, scored by a port of BFCL's own checker. The oracle ceiling is 39% on `simple_python`,
42% on `multiple`, 36% on `live_simple`, 35% on `live_multiple` and 100% on the irrelevance categories. Almost every
miss is a value no extractor nominated. 57% of those values appear verbatim in the user's text, and the rest need
reformatting or world knowledge that selection cannot produce. Details, attribution and the full tables are in
[docs/BENCH.md](docs/BENCH.md).

## Backends

| Backend | Construct | Endpoint | Key | Model id |
|---|---|---|---|---|
| TypeSafe direct | `jt.backends.TypeSafe()` | `POST https://api.typesafe.ai/v1/systemone` (`TYPESAFE_BASE_URL` overrides the host) | `TYPESAFE_API_KEY` | `jev-latest` |
| OpenRouter System One | `jt.backends.OpenRouterSystemOne()` | `POST https://openrouter.ai/api/v1/systemone` | `OPENROUTER_API_KEY` | `~typesafe/jev-latest`, pinned `typesafe/jev-1.13` |
| OpenRouter Decisions | `jt.backends.OpenRouterDecisions()` | `POST https://openrouter.ai/api/alpha/decisions` (adds `usage.cost`) | `OPENROUTER_API_KEY` | `~typesafe/jev-latest`, pinned `typesafe/jev-1.13` |
| Simulator | `jt.backends.LexicalSimulator()` | offline lexical test double, **not a model** | none | `lexical-simulator` |
| Cassette | `jt.backends.Cassette(path, mode="replay", inner=None)` | JSONL keyed by the request's SHA-256; `record` wraps a live backend, and `replay` raises `CassetteMiss` on an unknown request | none | as recorded |
| Scripted | `jt.backends.ScriptedBackend(script)`, `.from_fixture(path)` | offline answers from a dict or a callable | none | `scripted` |

The three HTTP backends share one wire contract: only the URL and the model id differ. Pass `model="…"` to pin a
version. OpenRouter `chat/completions` rejects Jev, so jevtools always uses the endpoints above. Failures are typed
(`JevValidationError`, `JevRateLimited`, `JevUnavailable`, …) and **fail closed**: a missing or failed answer never
leads to execute. For cassettes, give the `Context` a fixed `now`, because the state includes the current time.

## Honest limitations

These are condensed from SPEC §14.

- **One live measurement.** A single run of the app-domain benchmark exists (see [Benchmarks](#benchmarks)).
  Every other probability, token count and cost in the docs, examples and fixtures is illustrative. The live
  experiments E1–E10 (SPEC §11.2) are defined and runnable but have not been run.
- **Thresholds are priors.** They mean something only after `jevtools eval` + `tune` on your own labelled traffic,
  and they do not transfer across datasets.
- **Coverage is the ceiling.** Jev cannot elect a value that code did not nominate. Over an app's own data the
  oracle ceiling is high (98 of 99 on the app-domain bench). Outside that setting it is low: 35–42% on BFCL's
  single-call categories (docs/BENCH.md), mostly because extractors miss values that are in the text. When a value
  is missed, the best case is `NONE_OF_THESE`, which leads to widen or clarify. The worst case is a confident wrong
  election among distractors. Whether the sentinels really absorb mass when the right value is missing is
  unverified (experiment E2).
- **Calibration of the new question forms is assumed.** This covers "Suppose…" premises, accept Nouls, joint
  sentences, membership Nouls and sentinels. The compositions are only as honest as their factors.
- **Text is extractive without a Filler.** Bodies are templates plus the user's clauses and can read stilted. New
  prose, summaries and translations need an LLM Filler, whose drafts Jev must still accept, or the user.
- **English first.** Only `en` extractors are complete. `de`/`fr` are basic, other locales produce no candidates
  (so they clarify), and all templates are English.
- **Arithmetic and world knowledge** exist only as declared `derive` operators, sources and catalogs. The bundled
  gazetteer is compact: a few hundred cities.
- **Single intent.** `parallel_tool_calls` is accepted but ignored: one call is decided per turn (`ext.parallel`
  is not implemented). Planning is next-step only.
- **Latency compounds.** A round costs about 0.4–2 s. Widen, fill, resume turns and loop steps each add one.
- **Privacy.** The request, recent history, observation previews and shortlisted rows (names, emails, paths) are
  sent to TypeSafe or OpenRouter. Registry `attrs` such as balances, and secrets, are not.
- **Proxy state** is in memory by default. Multi-instance deployments need a shared `PendingStore`, or they pay one
  extra round when a card is not found.
- **Undocumented API limits** (label length, id charset, question cap) are guessed conservatively. Run
  `jevtools probe` once per backend.

## Project layout

| Path | Contents |
|---|---|
| `src/jevtools/` | the package; module map in [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) |
| `src/jevtools/{spec,sources,extract,kinds}/` | ingest, candidate sources, extractors, one resolver per slot kind |
| `src/jevtools/{plan,decode,confidence,policy,router,trace}.py` | the decision pipeline |
| `src/jevtools/{adapters,serve,eval,backends,demo}/` | integrations, proxy, evaluation, Jev backends, synthetic demo world |
| `src/jevtools/bench/` | benchmarks: app domains (`bench/app`, 6 bundled domains) and BFCL; oracle, runners ([docs/BENCH.md](docs/BENCH.md)) |
| `examples/` | 01–08 runnable scripts (offline by default; `--backend scripted\|sim\|live`), `proxy/` |
| `tests/` | 1,000+ offline tests; `tests/golden/` holds the conformance fixtures, `tests/live/` the live smoke suite |
| [`docs/SPEC.md`](docs/SPEC.md) | the normative protocol (`jevtools/0.1`) |
| [`docs/DECISIONS.md`](docs/DECISIONS.md) | where the implementation resolved ambiguities in, or deviates from, the spec |
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | module map and data flow |

- **Examples.** `python examples/01_quickstart_weather.py` runs the spec's walk-through (R1–R7) and prints the
  questions, decision, prompt and trace summary. See [examples/README.md](examples/README.md).
- **Ports.** Everything from the Ballot onward is byte-specified. `tests/golden/<case>/` stores the Ballot, every
  Jev request and response, and the Decision and Trace for 16 cases. A port (R first) conforms when it reproduces
  those bytes. See [tests/golden/README.md](tests/golden/README.md).
- **Development.** `python -m pytest -q`, `ruff check .`, `ruff format --check .`, `mypy --strict src/jevtools`. The
  live smoke suite in `tests/live` (SPEC §10.7) is skipped without a key. `python -m pytest -m live` runs it against
  every HTTP backend whose key is set: the conformance probe, R1–R7 and a cassette round trip, asserting only the
  invariants (I1–I5, schema-valid calls, no unconfirmed critical call), never probabilities.
- **License.** None has been chosen yet: there is no `LICENSE` file, and the package metadata declares none.
