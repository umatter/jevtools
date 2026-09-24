"""Interop adapters (spec §7.2): emit decisions in other ecosystems' formats and accept their conversations.

- :mod:`jevtools.adapters.openai`: ``complete``/``acomplete`` (a ``ChatCompletion``), ``wrap(client, router)`` and
  the proxy's error mapping;
- :mod:`jevtools.adapters.anthropic`: ``tool_use`` blocks, Messages responses, Anthropic tools → Catalog;
- :mod:`jevtools.adapters.mcp`: ``call_decision(session, decision)`` with the idempotency key in ``_meta``;
- :mod:`jevtools.adapters.pending`: ``PendingStore`` and prefix-hash matching of pending prompts;
- :mod:`jevtools.adapters.langchain` (extra ``langchain``) and :mod:`jevtools.adapters.pydantic_ai`
  (extra ``pydantic-ai``) are imported explicitly, because they need their optional dependency.
"""

from jevtools.adapters import anthropic, mcp, openai, pending
from jevtools.adapters.pending import InMemoryPendingStore, PendingStore

__all__ = ["InMemoryPendingStore", "PendingStore", "anthropic", "mcp", "openai", "pending"]
