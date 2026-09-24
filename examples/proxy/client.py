"""A client of ``jevtools serve`` written with the OpenAI Python SDK — no jevtools import, as any app would do.

    pip install openai
    jevtools serve --config examples/proxy/jevtools.toml        # in another terminal
    python examples/proxy/client.py [--base-url http://127.0.0.1:8787/v1]

Two conversations:

1. "Email Anna that I'll be 10 minutes late": the proxy answers with a confirm card or a clarify menu (an assistant
   message whose ``content`` is the question and whose ``x_jev`` lists the options). The client shows it, echoes it
   back and answers "1" (the first option); the next completion carries the ``tool_calls``. The proxy keeps the
   pending card server-side and finds it again by the conversation prefix.
2. "What's the weather like in Zurich in Fahrenheit?": ``tool_calls`` → the client runs the tool and appends a
   ``role: tool`` message → the proxy sees the observation and stops (``x_jev.outcome == "done"``).

Per-request context goes in ``extra_body={"jevtools": {...}}`` (here: a fixed clock). Without a Jev key the example
config falls back to the offline LexicalSimulator, whose answers are never evidence about Jev.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from typing import Any

from openai import OpenAI  # type: ignore[import-not-found, unused-ignore]

TOOLS: list[dict[str, Any]] = [
    {"type": "function", "function": {
        "name": "send_email", "description": "Send an email from the user to one recipient.",
        "parameters": {"type": "object", "required": ["to", "subject", "body"], "properties": {
            "to": {"type": "string", "format": "email", "description": "The recipient's email address"},
            "subject": {"type": "string", "description": "The subject line"},
            "body": {"type": "string", "description": "The body text of the email"}}}}},
    {"type": "function", "function": {
        "name": "get_weather", "description": "Get the current weather for a city.",
        "parameters": {"type": "object", "required": ["city"], "properties": {
            "city": {"type": "string", "description": "The city name"},
            "unit": {"type": "string", "enum": ["celsius", "fahrenheit"], "default": "celsius",
                     "description": "The temperature unit"}}}}},
]  # fmt: skip
"""Plain OpenAI function tools, exactly what the app would send to any chat model."""

EXTRA_BODY = {"jevtools": {"context": {"now": "2026-09-24T14:05:00+02:00"}}}
"""Per-request jevtools options: a fixed clock so relative dates ("next Tuesday") are reproducible."""


def get_weather(city: str, unit: str = "celsius") -> dict[str, Any]:
    """The app's own tool implementation (a stub)."""
    return {"city": city, "unit": unit, "temp": 63 if unit == "fahrenheit" else 17}


def complete(client: Any, messages: list[dict[str, Any]]) -> dict[str, Any]:
    """One chat completion through the proxy; returns the assistant message as a dict (``x_jev`` included)."""
    response = client.chat.completions.create(model="jevtools", messages=messages, tools=TOOLS, extra_body=EXTRA_BODY)
    message: dict[str, Any] = response.choices[0].message.model_dump(exclude_none=True)
    x_jev = message.get("x_jev") or {}
    print(f"  ← {response.choices[0].finish_reason} · x_jev.outcome = {x_jev.get('outcome')}")
    if message.get("content"):
        print(f"    {message['content']}")
    for i, option in enumerate(x_jev.get("options") or [], 1):
        print(f"    {i}. {option['text']}")
    for call in message.get("tool_calls") or []:
        print(f"    tool call: {call['function']['name']}({call['function']['arguments']})")
    return message


def main(argv: Sequence[str] | None = None, *, http_client: Any = None) -> list[list[dict[str, Any]]]:
    """Run both conversations; ``http_client`` (an ``httpx.Client``) replaces the network, e.g. in tests."""
    parser = argparse.ArgumentParser(description="An OpenAI SDK client of jevtools serve.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8787/v1")
    args = parser.parse_args(list(argv) if argv is not None else None)
    client = OpenAI(base_url=args.base_url, api_key="not-used-by-the-proxy", http_client=http_client)

    print('1) "Email Anna that I\'ll be 10 minutes late"')
    email: list[dict[str, Any]] = [
        {"role": "user", "content": "What's next on my calendar?"},
        {"role": "assistant", "content": "14:30 ACME quarterly review with Anna Keller."},
        {"role": "user", "content": "Email Anna that I'll be 10 minutes late"},
    ]
    message = complete(client, email)
    if not message.get("tool_calls") and (message.get("x_jev") or {}).get("options"):
        email += [{"role": "assistant", "content": message.get("content", "")}, {"role": "user", "content": "1"}]
        print('  → "1"')
        message = complete(client, email)
    email.append(message)

    print('\n2) "What\'s the weather like in Zurich in Fahrenheit?"')
    weather: list[dict[str, Any]] = [{"role": "user", "content": "What's the weather like in Zurich in Fahrenheit?"}]
    for _ in range(3):
        message = complete(client, weather)
        weather.append(message)
        if not message.get("tool_calls"):
            break
        for call in message["tool_calls"]:
            result = get_weather(**json.loads(call["function"]["arguments"]))
            print(f"  → ran {call['function']['name']}: {result}")
            weather.append({"role": "tool", "tool_call_id": call["id"], "content": json.dumps(result)})
    return [email, weather]


if __name__ == "__main__":
    main()
