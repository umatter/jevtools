"""Question ids (spec §3.5.2): sanitizing names, building ids, validating the grammar, opaque mode.

Ids are addresses for code only; decoders use the Ballot's decode map and never parse ids::

    qid := "tool" | "reply" | "segmentation" | seg? T "." F
    seg := "s" DIGIT+ "."
    T   := sanitized tool name
    F   := "authorized" | "joint" ("." G)? | "done_after"
         | P ( "" | ".present" | ".rev" | ".date" | ".time" | ".branch" | ".more" | ".group"
                 | ".accept." N | ".m" N | ".item." N | ".member." N | ".bucket." N )
    P   := sanitized param path; a segment equal to a reserved suffix word gets "_" appended

Charset ``[a-z0-9_.]``, length ≤ 128, first character a letter.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence

TOOL_QID = "tool"
REPLY_QID = "reply"
SEGMENTATION_QID = "segmentation"
FIXED_QIDS: frozenset[str] = frozenset({TOOL_QID, REPLY_QID, SEGMENTATION_QID})

QID_MAX = 128
RESERVED_WORDS: frozenset[str] = frozenset(
    {
        "authorized", "joint", "done_after",
        "present", "rev", "date", "time", "branch", "more", "group",
        "accept", "item", "member", "bucket",
    }
)  # fmt: skip
"""Suffix words a param-path segment may not equal (they get ``_`` appended)."""

_UNSAFE = re.compile(r"[^a-z0-9_]")
_RESERVED_PATTERN = re.compile(r"m\d+|\d+")
_QID_RE = re.compile(r"(?:s\d+\.)?[a-z][a-z0-9_]*(?:\.[a-z0-9_]+)+")


def sanitize_tool_name(name: str) -> str:
    """Lowercase, ``[^a-z0-9_]`` → ``_``, and a ``t_`` prefix unless the result starts with a letter."""
    safe = _UNSAFE.sub("_", name.lower())
    if not safe or not safe[0].isalpha() or not safe[0].isascii():
        safe = "t_" + safe
    return safe


def sanitize_segment(segment: str) -> str:
    """Sanitize one param-path segment; reserved suffix words (and ``m<N>``/digit-only words) get ``_`` appended."""
    safe = _UNSAFE.sub("_", segment.lower()) or "_"
    if safe in RESERVED_WORDS or _RESERVED_PATTERN.fullmatch(safe):
        safe += "_"
    return safe


def dedupe(names: Iterable[str]) -> list[str]:
    """Resolve collisions in order: the second ``x`` becomes ``x_2``, the third ``x_3`` (skipping taken names)."""
    seen: set[str] = set()
    out: list[str] = []
    for name in names:
        candidate, i = name, 2
        while candidate in seen:
            candidate = f"{name}_{i}"
            i += 1
        seen.add(candidate)
        out.append(candidate)
    return out


def tool_ids(names: Sequence[str]) -> list[str]:
    """Sanitized, collision-free tool ids for ``names`` (in order)."""
    return dedupe(sanitize_tool_name(n) for n in names)


def make_qid(tool_id: str, *parts: str | int, seg: int | None = None) -> str:
    """Join ``[s<seg>.]<tool_id>.<part>.<part>…`` (parts are already-sanitized path strings, suffix words or ints).

    Raises ``ValueError`` if the result violates the grammar (a code bug).
    """
    pieces = [tool_id, *(str(p) for p in parts if p != "")]
    qid = ".".join(pieces)
    if seg is not None:
        qid = f"s{seg}.{qid}"
    if not is_valid_qid(qid):
        raise ValueError(f"invalid question id {qid!r}")
    return qid


def is_valid_qid(qid: str, *, max_len: int = QID_MAX) -> bool:
    """Grammar check: a fixed id, or ``[s<N>.]T.F`` with charset ``[a-z0-9_.]``, first char a letter, ≤ max_len."""
    if len(qid) > max_len:
        return False
    return qid in FIXED_QIDS or _QID_RE.fullmatch(qid) is not None


def opaque_ids(qids: Sequence[str]) -> dict[str, str]:
    """Opaque-mode mapping ``qid → q0001…`` in the given order (used when the probe finds dotted ids rejected)."""
    width = max(4, len(str(len(qids))))
    return {qid: f"q{i:0{width}d}" for i, qid in enumerate(qids, start=1)}


__all__ = [
    "FIXED_QIDS",
    "QID_MAX",
    "REPLY_QID",
    "RESERVED_WORDS",
    "SEGMENTATION_QID",
    "TOOL_QID",
    "dedupe",
    "is_valid_qid",
    "make_qid",
    "opaque_ids",
    "sanitize_segment",
    "sanitize_tool_name",
    "tool_ids",
]
