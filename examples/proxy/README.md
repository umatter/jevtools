# `jevtools serve`: Jev behind an OpenAI-compatible endpoint

`jevtools serve` is a small HTTP proxy (spec §7.2.4). It speaks OpenAI Chat Completions with tools. Any client
that can point an OpenAI SDK's `base_url` somewhere can use Jev for tool calling, in any language, with no port of
jevtools. The client keeps sending its usual `messages` and `tools`. The proxy builds candidate pools in code, asks
Jev typed questions and sends back one of these:

- a `tool_calls` message;
- a confirm card or a clarify menu: an assistant message whose `content` is the question and whose `x_jev` lists
  the options;
- an empty stop marked `x_jev.outcome = "abstain"` or `"done"`.

Everything in this folder is a demo. The contacts, accounts and files in `data/` are **synthetic**. They are
generated from `jevtools.demo` by `examples/fixtures/regenerate.py`.

## 1. Run the proxy

```bash
pip install 'jevtools[serve]'
export OPENROUTER_API_KEY=...        # or TYPESAFE_API_KEY=...
jevtools serve --config examples/proxy/jevtools.toml --host 127.0.0.1 --port 8787
```

`jevtools.toml` sets up four things:

- the backend: `auto` picks TypeSafe if `TYPESAFE_API_KEY` is set, else OpenRouter Decisions;
- default context: timezone, locale and user;
- three candidate sources read from JSON files: a `contacts` registry, an `accounts` registry and a `files` index;
- the pending-card TTL.

Relative paths resolve against the TOML file.

**Without a key**, the example config sets `allow_offline = true`, so the proxy starts on the offline
`LexicalSimulator` and prints a warning. The simulator is a lexical test double, not a model. Use it to try the
plumbing only: its answers are never evidence about Jev's accuracy. Set `allow_offline = false` in production, so a
missing key fails at startup.

Check the proxy is up:

```bash
curl -s http://127.0.0.1:8787/healthz     # {"status":"ok","backend":"openrouter_decisions",...}
curl -s http://127.0.0.1:8787/v1/models   # lists "jevtools"
```

## 2. Python: the OpenAI SDK

[`client.py`](client.py) does not import jevtools; it is what any app would write:

```bash
pip install openai
python examples/proxy/client.py --base-url http://127.0.0.1:8787/v1
```

```python
from openai import OpenAI

# the proxy holds the Jev key; TOOLS are plain OpenAI function tools
client = OpenAI(base_url="http://127.0.0.1:8787/v1", api_key="unused")
response = client.chat.completions.create(
    model="jevtools",
    messages=[{"role": "user", "content": "Email Anna that I'll be 10 minutes late"}],
    tools=TOOLS,
    extra_body={"jevtools": {"context": {"now": "2026-09-24T14:05:00+02:00"}}},
)
message = response.choices[0].message
message.tool_calls  # set when the call can run now
message.content  # otherwise: the card or menu text
message.model_dump()["x_jev"]  # outcome, options, pending_id, confidence, idempotency_key
```

- **Cards and menus.** Show `content` and the `x_jev.options` to the user. Echo the assistant message back
  unchanged and append the user's answer as a normal user message. The answer can be an option number, an option
  text, or `ok`, `yes` or `cancel`. The proxy keeps the pending card server-side for 15 minutes. It finds the card
  again by hashing the conversation prefix, so the client needs no extra state. A click costs no Jev call. Free
  text costs one Jev round.
- **Tool results.** Append a normal `{"role": "tool", "tool_call_id": ..., "content": ...}` message. The proxy
  treats it as an observation, so the next completion can take the next step or stop with `x_jev.outcome = "done"`.
- **Per-request context.** Use `extra_body={"jevtools": {...}}` for per-request settings:
  - `context` overrides `now`, `tz`, `locale` or `user`;
  - `sources` sends rows for this request only, for example `{"contacts": [...]}`;
  - `conversation_id` and `pending_id` identify the conversation and a pending card.

## 3. R: ellmer (spec §7.3)

ellmer can use the proxy as it is, with its existing tool registration:

```r
chat <- ellmer::chat_openai(base_url = "http://127.0.0.1:8787/v1", model = "jevtools")
chat$register_tool(send_email_tool)     # ellmer::tool(...) as today
chat$chat("Email Anna that I'll be 10 minutes late")
```

ellmer types map to jevtools slot kinds as follows:

| ellmer type | jevtools kind |
|---|---|
| `type_enum` | enum |
| `type_boolean` | flag |
| `type_integer`, `type_number` | quantity |
| `type_string` | inferred from name and format |
| `type_array` | list |
| `type_object` | record |

A native R package using the same protocol is planned (spec §7.3).

## 4. Other clients

Any OpenAI-compatible SDK works the same way: point its base URL at `http://127.0.0.1:8787/v1` and use the model
`jevtools`. With `stream: true`, the proxy sends two server-sent events: one with the whole message, then one with
the finish reason. Errors follow the OpenAI error format. A tool call is never produced on an error path.
