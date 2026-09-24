"""Observation previews (spec §3.5.1, §6.2): what ``state.observations`` shows of a tool result.

The full content never reaches Jev; the preview is the whole text when it fits in :data:`PREVIEW_CHARS`, else the
sentence chunks BM25 ranks highest against the request, in document order. Selection is code only, never Jev. One
implementation serves every path — :func:`jevtools.loop.ingest_observation` (the Agent, MCP) and ``role: tool``
messages in drop-in mode (:meth:`jevtools.context.Context.all_observations`).
"""

from __future__ import annotations

import functools
import re
from collections.abc import Sequence

PREVIEW_CHARS = 1_200
"""Preview length (§6.2)."""
CHUNK_CHARS = 280
"""Target chunk size: sentences (and lines) are grouped up to about this many characters."""
_SENTENCE_RE = re.compile(r"[^.!?\n]+(?:[.!?]+|\n+|$)")


def chunk_text(text: str, size: int = CHUNK_CHARS) -> list[str]:
    """Sentences (and lines) grouped into chunks of about ``size`` characters, in document order."""
    chunks: list[str] = []
    current = ""
    for match in _SENTENCE_RE.finditer(text):
        sentence = " ".join(match.group().split())
        if not sentence:
            continue
        if current and len(current) + 1 + len(sentence) > size:
            chunks.append(current)
            current = sentence
        else:
            current = f"{current} {sentence}".strip()
    if current:
        chunks.append(current)
    return chunks


def select_preview(chunks: Sequence[str], request: str, limit: int = PREVIEW_CHARS) -> str:
    """The preview: the whole text when it fits, else the chunks BM25 ranks highest against ``request`` (then the
    first chunk, then the rest) while they fit, in document order, joined by `` … `` and cut to ``limit``
    characters. Selection is code only, never Jev (§6.2)."""
    whole = " ".join(chunks)
    if len(whole) <= limit:
        return whole
    from jevtools.extract.tokens import words  # local: context imports this module, sources import context
    from jevtools.sources.retrieval import BM25

    query = words(request)
    ranked = [i for i, _ in BM25([words(c) for c in chunks]).top(query)] if query else []
    chosen: list[int] = []
    used = 0
    for index in dict.fromkeys([*ranked, 0, *range(len(chunks))]):
        cost = len(chunks[index]) + (3 if chosen else 0)
        if used + cost <= limit:
            chosen.append(index)
            used += cost
    if not chosen:
        return chunks[ranked[0] if ranked else 0][: limit - 1].rstrip() + "…"
    return " … ".join(chunks[i] for i in sorted(chosen))


def text_preview(text: str, request: str = "", limit: int = PREVIEW_CHARS) -> str:
    """The preview of a text result: the text itself when it fits in ``limit``, else :func:`select_preview` over
    its chunks against ``request`` (memoized: a conversation's tool messages are previewed on every round)."""
    if len(text) <= limit:
        return text
    return _ranked_preview(text, request, limit)


@functools.lru_cache(maxsize=32)
def _ranked_preview(text: str, request: str, limit: int) -> str:
    return select_preview(chunk_text(text) or [text], request, limit)


__all__ = ["CHUNK_CHARS", "PREVIEW_CHARS", "chunk_text", "select_preview", "text_preview"]
