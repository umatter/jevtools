"""The strict pre-send validator (spec §3.5.6): a Ballot that breaks a rule is a code bug, never sent.

Rule ids raised in :class:`~jevtools.errors.BallotError`:

=====================  =====================================================================================
``choice.options``     2 ≤ options ≤ 255 per Choice, at most 252 real options
``label.grammar``      1..label_max printable characters, no newline/tab, no leading/trailing space
``label.reserved``     a real option uses a reserved sentinel label
``label.unique``       labels unique within a question after NFC and casefold
``description.length`` option descriptions ≤ desc_max (400)
``instructions.length`` instructions ≤ instr_max (2,000); accept candidates are checked separately
``accept.candidate``   an accept-Noul ``candidate`` ≤ accept_max (4,000)
``qid.grammar``        the §3.5.2 grammar (charset, first letter, ≤ 128)
``qid.unique``         unique qids (also enforced by :class:`~jevtools.ballot.Ballot`)
``call.questions``     ≤ max_questions per call (250)
``call.tokens``        estimated tokens per call ≤ max_tokens (24,000)
``value.schema``       every real option value passes its slot's JSON Schema (needs the catalog)
=====================  =====================================================================================
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, ConfigDict

from jevtools.ballot import Ballot, BallotQuestion, IdMode
from jevtools.budget import tokens_for_chars
from jevtools.candidates import is_reserved, is_valid_label, label_key
from jevtools.canonical import canonical_str
from jevtools.errors import BallotError
from jevtools.qid import is_valid_qid
from jevtools.spec.schema import validate as validate_value
from jevtools.wire import DecisionRequest

if TYPE_CHECKING:
    from jevtools.policy import BudgetPolicy
    from jevtools.spec.catalog import Catalog

_BUDGET_FIELDS = (("max_tokens", "max_tokens_per_call"), ("max_questions", "max_questions_per_call"),
                  ("max_state_tokens", "max_state_tokens"), ("chars_per_token", "chars_per_token"))  # fmt: skip


class Limits(BaseModel):
    """Wire limits used by the validator and the planner.

    Defaults come from §3.5.6; the conformance probe (§8.7) may relax or tighten them.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    label_max: int = 64
    desc_max: int = 400
    instr_max: int = 2_000
    accept_max: int = 4_000
    min_options: int = 2
    max_options: int = 255
    max_real_options: int = 252
    qid_max: int = 128
    max_questions: int = 250
    max_tokens: int = 24_000
    max_state_tokens: int = 16_000
    chars_per_token: float = 3.5
    token_ratio: float = 1.0
    """Running correction factor ``r`` from ``usage.input_tokens`` (§5.5)."""
    id_mode: IdMode = "dotted"
    instructions_as_object: bool = True
    """Whether the backend accepts JSON-object instructions (probe; else flattened to text)."""

    @classmethod
    def from_file(cls, path: str | os.PathLike[str]) -> Limits:
        """Load limits written by ``jevtools probe`` (unknown keys are ignored for forward compatibility)."""
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.model_validate({k: v for k, v in data.items() if k in cls.model_fields})

    def tokens_for_chars(self, chars: int) -> int:
        """The §5.5 estimate for ``chars`` characters: ``⌈chars × token_ratio / chars_per_token⌉`` — the one formula
        the planner (splits, state cuts) and the pre-send check share, so they never disagree by a rounding."""
        return tokens_for_chars(chars, chars_per_token=self.chars_per_token, ratio=self.token_ratio)

    def estimate_tokens(self, request: DecisionRequest | dict[str, Any]) -> int:
        """``chars(canonical JSON) / chars_per_token × token_ratio`` (§5.5), rounded up."""
        body = request.to_wire() if isinstance(request, DecisionRequest) else request
        return self.tokens_for_chars(len(canonical_str(body)))

    def within_budget(self, budget: BudgetPolicy) -> Limits:
        """These limits under a policy's ``[budget]`` (Appendix B, §5.5): every budget setting the policy changes
        from its default caps the matching limit (``max_tokens_per_call`` → ``max_tokens``,
        ``max_questions_per_call`` → ``max_questions``, ``max_state_tokens``, ``chars_per_token``), taking the
        smaller value — a tighter budget takes effect, a looser one never exceeds what the backend supports (probed
        or explicit limits). A default budget leaves the limits unchanged."""
        defaults = type(budget)()
        update: dict[str, Any] = {}
        for field, setting in _BUDGET_FIELDS:
            value = getattr(budget, setting)
            if value != getattr(defaults, setting) and value < getattr(self, field):
                update[field] = value
        return self.model_copy(update=update) if update else self


def cache_dir() -> Path:
    """Where ``jevtools probe`` writes limits: ``$JEVTOOLS_CACHE_DIR``, else ``$XDG_CACHE_HOME/jevtools``, else
    ``~/.cache/jevtools``."""
    explicit = os.environ.get("JEVTOOLS_CACHE_DIR")
    if explicit:
        return Path(explicit)
    xdg = os.environ.get("XDG_CACHE_HOME")
    return (Path(xdg) if xdg else Path.home() / ".cache") / "jevtools"


def limits_path(backend: str, model: str, directory: str | os.PathLike[str] | None = None) -> Path:
    """``<cache>/limits-<backend>-<model>.json`` (characters outside ``[A-Za-z0-9._-]`` become ``_``)."""

    def safe(part: str) -> str:
        return re.sub(r"[^A-Za-z0-9._-]+", "_", part).strip("_") or "default"

    base = Path(directory) if directory is not None else cache_dir()
    return base / f"limits-{safe(backend)}-{safe(model)}.json"


def cached_limits(backend: Any, directory: str | os.PathLike[str] | None = None) -> Limits | None:
    """The probed :class:`Limits` of ``backend`` (its ``name`` and ``model``) from the cache (spec §8.7: the file
    ``jevtools probe`` writes "which the validator uses"), or ``None`` if it was never probed."""
    path = limits_path(str(getattr(backend, "name", "")), str(getattr(backend, "model", "")), directory)
    return Limits.from_file(path) if path.is_file() else None


Rule = Literal[
    "choice.options", "label.grammar", "label.reserved", "label.unique", "description.length",
    "instructions.length", "accept.candidate", "qid.grammar", "qid.unique", "call.questions", "call.tokens",
    "value.schema",
]  # fmt: skip


def _text_length(value: Any) -> int:
    return len(value) if isinstance(value, str) else len(canonical_str(value))


def check_question(question: BallotQuestion, limits: Limits) -> None:
    """Question-level rules (options, labels, descriptions, instructions, accept candidate, qid grammar)."""
    qid = question.qid
    if not is_valid_qid(qid, max_len=limits.qid_max):
        raise BallotError(qid, "qid.grammar", f"charset [a-z0-9_.], first char a letter, ≤ {limits.qid_max}")
    _check_instructions(question, limits)
    if question.primitive != "choice":
        return
    total = len(question.options) + len(question.sentinels)
    if not limits.min_options <= total <= limits.max_options or len(question.options) > limits.max_real_options:
        raise BallotError(qid, "choice.options", f"{len(question.options)} real + {len(question.sentinels)} sentinels")
    seen: set[str] = set()
    for option in question.options:
        if not is_valid_label(option.label, limits.label_max):
            raise BallotError(qid, "label.grammar", repr(option.label))
        if is_reserved(option.label):
            raise BallotError(qid, "label.reserved", repr(option.label))
        if option.text is not None and _text_length(option.text) > limits.desc_max:
            raise BallotError(qid, "description.length", repr(option.label))
    for label in question.labels:
        key = label_key(label)
        if key in seen:
            raise BallotError(qid, "label.unique", repr(label))
        seen.add(key)
    for label, spec in question.sentinels.items():
        if spec.text is not None and _text_length(spec.text) > limits.desc_max:
            raise BallotError(qid, "description.length", label)


def _check_instructions(question: BallotQuestion, limits: Limits) -> None:
    instructions = question.instructions
    if isinstance(instructions, dict) and "candidate" in instructions:
        if _text_length(instructions["candidate"]) > limits.accept_max:
            raise BallotError(question.qid, "accept.candidate")
        instructions = {k: v for k, v in instructions.items() if k != "candidate"}
    if _text_length(instructions) > limits.instr_max:
        raise BallotError(question.qid, "instructions.length")


def check_values(question: BallotQuestion, catalog: Catalog) -> None:
    """``value.schema``: slot and ``rev`` options validate against the slot schema, mention options against the item
    schema. Late-bound options (placeholders, derived values) are checked after late binding instead."""
    if question.tool is None or question.family not in ("slot", "rev", "mention"):
        return
    try:
        spec = catalog.get(question.tool).slot(question.path)
    except KeyError:
        return
    schema = spec.item.json_schema if question.family == "mention" and spec.item is not None else spec.json_schema
    for option in question.options:
        if option.late is None and validate_value(option.value, schema):
            raise BallotError(question.qid, "value.schema", repr(option.label))


def preflight(ballot: Ballot, limits: Limits | None = None, *, catalog: Catalog | None = None) -> None:
    """Reject a Ballot before sending (spec §3.5.6). Raises :class:`~jevtools.errors.BallotError` on the first
    violated rule; returns ``None`` when every rule holds."""
    limits = limits or Limits()
    qids = [q.qid for q in ballot.questions]
    if len(set(qids)) != len(qids):
        duplicate = next(q for q in qids if qids.count(q) > 1)
        raise BallotError(duplicate, "qid.unique")
    for question in ballot.questions:
        check_question(question, limits)
        if catalog is not None:
            check_values(question, catalog)
    requests = ballot.to_requests("", id_mode=limits.id_mode, object_instructions=limits.instructions_as_object)
    for index, request in enumerate(requests):
        if len(request.questions) > limits.max_questions:
            raise BallotError(None, "call.questions", f"call {index}: {len(request.questions)} questions")
        tokens = limits.estimate_tokens(request)
        if tokens > limits.max_tokens:
            raise BallotError(None, "call.tokens", f"call {index}: ~{tokens} tokens")


__all__ = [
    "Limits",
    "Rule",
    "cache_dir",
    "cached_limits",
    "check_question",
    "check_values",
    "limits_path",
    "preflight",
]
