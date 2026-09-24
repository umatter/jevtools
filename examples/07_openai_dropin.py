"""07 · OpenAI drop-in: ``jt.openai.complete`` over a plain OpenAI ``tools`` list and ``role: tool`` messages.

The host keeps its OpenAI-style loop: it sends ``messages`` + ``tools`` and gets a ``ChatCompletion``-shaped dict
back. jevtools answers with Jev instead of a generative model:

- **a tool loop**: the first completion carries ``tool_calls`` (``get_weather``); the host runs the tool and appends
  a ``role: tool`` message; the next completion sees the result as an observation and stops with
  ``x_jev.outcome = "done"`` (``finish_reason: stop``);
- **a confirm card over a stateless client**: an external call comes back as an assistant message whose ``content``
  is the card and whose ``x_jev`` holds the options and ``pending_id``; the client echoes the text, the user says
  "ok", and the next completion carries the ``tool_calls`` without another Jev call (the pending prompt is found by
  the hash of the conversation prefix).

``jt.openai.wrap(client)`` does the same inside an existing OpenAI SDK client (``model="jevtools"``), and
``jevtools serve`` does it over HTTP (``examples/proxy/``).

Run: ``python examples/07_openai_dropin.py [--backend scripted|sim|live]``.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

import _show

import jevtools as jt
from jevtools.demo import scenario, scripts

TOOLS: list[dict[str, Any]] = scenario.scenario_tools()
"""The six plain OpenAI function tools of spec §13.2 (what a host already sends to a chat model)."""


def show_completion(doc: dict[str, Any], backend: _show.Recording, seen: int) -> dict[str, Any]:
    """The questions Jev was asked for this completion (requests after the first ``seen``), then the document."""
    qids = [qid for request in backend.requests[seen:] for qid in request.questions]
    asked = f"{len(qids)} in {len(backend.requests) - seen} Jev call(s) — {_show.compact_qids(qids)}" if qids \
        else "0 — no Jev call"  # fmt: skip
    _show.kv("questions", asked)
    choice = doc["choices"][0]
    usage = doc["usage"]["x_jev"]
    _show.kv("finish", f"{choice['finish_reason']} · x_jev.outcome {usage['outcome']} · {usage['jev_calls']} Jev "
                       f"call(s) · {usage['jev_input_tokens']:,} input tokens")  # fmt: skip
    _show.message(choice["message"], title="message")
    return dict(choice["message"])


def main(argv: Sequence[str] | None = None) -> dict[str, list[dict[str, Any]]]:
    args = _show.parse_args(__doc__, argv)
    _show.header("07 · OpenAI drop-in: complete() with plain tools and role:tool messages", args.backend,
                 ["R1-dropin", "R2"])  # fmt: skip
    context = scenario.scenario_context()  # clock, locale, user and the contacts/accounts/files sources
    _show.kv("tools", ", ".join(t["function"]["name"] for t in TOOLS) + " (plain OpenAI function tools)", indent=0)

    _show.step(f'a tool loop: "{scripts.R1_REQUEST}"')
    backend = _show.Recording(_show.backend(args.backend, "R1-dropin"))
    messages: list[dict[str, Any]] = [{"role": "user", "content": scripts.R1_REQUEST}]
    loop_docs = []
    for turn in range(1, 4):
        seen = len(backend.requests)
        doc = jt.openai.complete(messages, TOOLS, backend=backend, context=context)
        loop_docs.append(doc)
        _show.step(f"completion {turn}")
        message = show_completion(doc, backend, seen)
        messages.append(message)
        if not message.get("tool_calls"):
            break
        for call in message["tool_calls"]:  # the host executes the calls it receives
            arguments = json.loads(call["function"]["arguments"])
            result = scenario.Workspace().executors()[call["function"]["name"]](**arguments)
            messages.append({"role": "tool", "tool_call_id": call["id"], "content": json.dumps(result)})
            _show.kv("host ran", f"{call['function']['name']}({arguments}) → {result}", wrap=False)

    _show.step(f'a confirm card over a stateless client: "{scripts.R2_REQUEST}" (after two history turns)')
    backend = _show.Recording(_show.backend(args.backend, "R2"))
    router = scenario.demo_router(backend, context=context)  # holds the pending prompts between two requests
    messages = scenario.scenario_messages(scripts.R2_REQUEST, history=True)
    card_docs = [jt.openai.complete(messages, TOOLS, router=router)]
    card = show_completion(card_docs[0], backend, 0)
    if card.get("tool_calls") is None and card.get("x_jev", {}).get("pending_id"):
        reply = "ok"
        _show.step(f'the client echoes the card text and the user answers "{reply}"')
        messages = [*messages, {"role": "assistant", "content": card["content"]}, {"role": "user", "content": reply}]
        seen = len(backend.requests)
        card_docs.append(jt.openai.complete(messages, TOOLS, router=router))
        show_completion(card_docs[-1], backend, seen)
    return {"loop": loop_docs, "card": card_docs}


if __name__ == "__main__":
    main()
