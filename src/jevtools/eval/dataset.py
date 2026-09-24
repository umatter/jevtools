"""The evaluation dataset (spec §11.1): one labelled case per JSONL line.

A case holds the conversation, the context and catalog it runs against (file paths relative to the dataset file,
or inline documents), and the **gold** label:

- ``outcomes_ok``: the outcomes a correct decision may take (``["execute", "confirm"]``);
- ``tool``: the gold tool (``null`` when no tool should be called);
- ``args``: gold argument values; ``match`` gives each argument's match mode (``exact`` | ``accepted_set`` |
  ``ignore``) and ``accepted`` the accepted values of ``accepted_set`` arguments.

Arguments a case does not mention are ignored (a gold label may be partial); an argument listed in ``args`` defaults
to ``exact``, one listed in ``accepted`` to ``accepted_set``. ``tags`` name the perturbation families
(``name_collision``, ``history``, ``dst``, ``injection``…) used to slice the metrics.

```json
{"id": "r2-anna-hist", "messages": [...], "context": "fixtures/ctx_default.json", "catalog": "fixtures/catalog.json",
 "gold": {"outcomes_ok": ["execute", "confirm"], "tool": "send_email", "args": {"to": "anna.keller@acme.com"},
          "match": {"to": "exact", "body": "accepted_set", "subject": "ignore"},
          "accepted": {"body": ["Hi Anna,\\n\\nI'll be 10 minutes late.\\n\\nBest,\\nSam"]}},
 "tags": ["name_collision", "history"]}
```
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, field_validator, model_validator

from jevtools.canonical import canonical_str, nfc
from jevtools.context import SETTING_FIELDS, Context, Turn
from jevtools.policy import Outcome
from jevtools.sources.specs import build_sources

MatchMode = Literal["exact", "accepted_set", "ignore"]
"""How a gold argument is compared with a decided one."""
CALL_OUTCOMES = frozenset({Outcome.EXECUTE, Outcome.CONFIRM})
"""Outcomes that put a concrete call in front of the host (executed, or shown on a confirm card)."""
_PLACEHOLDER = re.compile(r"⟨[^⟩]*⟩")


def same_value(a: Any, b: Any) -> bool:
    """Exact equality of two argument values on their canonical JSON (``45`` equals ``45.0``; strings NFC)."""
    try:
        return canonical_str(a) == canonical_str(b)
    except (TypeError, ValueError):
        return bool(a == b)


def template_matches(candidate: Any, gold: Any) -> bool:
    """Whether a candidate value equals ``gold``, reading late-bound ``⟨…⟩`` placeholders as wildcards.

    Text candidates may carry placeholders filled only after election (``"Hi ⟨recipient's first name⟩,…"``); the
    pool contains the gold body when the template can produce it. Used for pool coverage, never for call matching.
    """
    if same_value(candidate, gold):
        return True
    if not isinstance(candidate, str) or not isinstance(gold, str) or "⟨" not in candidate:
        return False
    parts = _PLACEHOLDER.split(nfc(candidate))
    pattern = ".+?".join(re.escape(p) for p in parts)
    return re.fullmatch(pattern, nfc(gold), flags=re.DOTALL) is not None


class Gold(BaseModel):
    """The gold label of a case (spec §11.1)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    outcomes_ok: list[Outcome] = Field(min_length=1)
    tool: str | None = None
    args: dict[str, Any] = Field(default_factory=dict)
    match: dict[str, MatchMode] = Field(default_factory=dict)
    accepted: dict[str, list[Any]] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _check(self) -> Gold:
        for slot, mode in self.match.items():
            if mode == "exact" and slot not in self.args:
                raise ValueError(f"gold.match[{slot!r}] is 'exact' but gold.args has no {slot!r}")
            if mode == "accepted_set" and slot not in self.accepted and slot not in self.args:
                raise ValueError(f"gold.match[{slot!r}] is 'accepted_set' but gold.accepted has no {slot!r}")
        if self.tool is None and (self.args or self.accepted):
            raise ValueError("gold.args/accepted need a gold.tool")
        return self

    def mode(self, slot: str) -> MatchMode:
        """The match mode of an argument: explicit, else ``accepted_set`` / ``exact`` / ``ignore`` by where it is
        listed."""
        if slot in self.match:
            return self.match[slot]
        if slot in self.accepted:
            return "accepted_set"
        if slot in self.args:
            return "exact"
        return "ignore"

    @property
    def slots(self) -> list[str]:
        """The checked arguments (mode ≠ ``ignore``), in label order."""
        names = list(dict.fromkeys([*self.args, *self.accepted, *self.match]))
        return [s for s in names if self.mode(s) != "ignore"]

    def values(self, slot: str) -> list[Any]:
        """Every value that counts as correct for ``slot`` (``args`` first, then ``accepted``)."""
        out: list[Any] = [self.args[slot]] if slot in self.args else []
        if self.mode(slot) == "accepted_set":
            out += [v for v in self.accepted.get(slot, []) if not any(same_value(v, o) for o in out)]
        return out

    def accepts(self, slot: str, value: Any) -> bool:
        """Whether ``value`` is a correct value of ``slot`` (always true for ignored arguments)."""
        if self.mode(slot) == "ignore":
            return True
        return any(same_value(value, v) for v in self.values(slot))

    def call_matches(self, name: str | None, arguments: Mapping[str, Any] | None) -> bool:
        """Whether a proposed call is the gold call: same tool, and every checked argument accepted (an argument
        the call omits fails unless it is ignored). ``name is None`` matches a gold label without a tool."""
        if self.tool is None or name is None:
            return self.tool is None and name is None
        if name != self.tool:
            return False
        args = dict(arguments or {})
        return all(slot in args and self.accepts(slot, args[slot]) for slot in self.slots)

    def allows(self, outcome: Outcome | str) -> bool:
        """Whether ``outcome`` is one of ``outcomes_ok``."""
        return Outcome(outcome) in self.outcomes_ok


class EvalCase(BaseModel):
    """One labelled case (spec §11.1). ``context``/``catalog`` are paths (relative to the dataset file) or inline
    documents; ``tool_choice`` optionally fixes the OpenAI ``tool_choice``; ``meta`` carries free-form data
    (perturbation provenance, experiment variants)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str
    messages: list[dict[str, Any]] = Field(min_length=1)
    context: str | dict[str, Any] | None = None
    catalog: str | list[dict[str, Any]] | None = None
    gold: Gold
    tags: list[str] = Field(default_factory=list)
    tool_choice: str | dict[str, Any] | None = None
    meta: dict[str, Any] = Field(default_factory=dict)
    _base_dir: Path | None = PrivateAttr(default=None)

    @field_validator("messages", mode="before")
    @classmethod
    def _messages(cls, value: Any) -> Any:
        if isinstance(value, str):
            return [{"role": "user", "content": value}]
        return value

    @property
    def base_dir(self) -> Path | None:
        """Directory relative paths resolve against (the dataset file's directory)."""
        return self._base_dir

    def with_base_dir(self, base_dir: str | os.PathLike[str] | None) -> EvalCase:
        """A copy whose relative paths resolve against ``base_dir``."""
        copy = self.model_copy()
        copy._base_dir = Path(base_dir) if base_dir is not None else None
        return copy

    def variant(self, suffix: str, **changes: Any) -> EvalCase:
        """A derived case ``<id>~<suffix>`` (experiment variants and perturbations keep the base directory)."""
        meta = {**self.meta, **changes.pop("meta", {}), "variant_of": self.id}
        copy = self.model_copy(update={"id": f"{self.id}~{suffix}", "meta": meta, **changes})
        copy._base_dir = self._base_dir
        return copy

    @property
    def request(self) -> str:
        """Text of the latest user message."""
        for message in reversed(self.messages):
            if message.get("role") == "user":
                return Turn.from_message(message).text
        return ""

    def resolve(self, path: str) -> Path:
        """``path`` resolved against the dataset directory."""
        p = Path(path)
        return p if p.is_absolute() or self._base_dir is None else self._base_dir / p

    def catalog_tools(self) -> list[dict[str, Any]] | None:
        """The case's OpenAI tool list (``None`` when the case names none)."""
        if self.catalog is None:
            return None
        if isinstance(self.catalog, str):
            data = json.loads(self.resolve(self.catalog).read_text(encoding="utf-8"))
            tools = data.get("tools", data) if isinstance(data, Mapping) else data
            return [dict(t) for t in tools]
        return [dict(t) for t in self.catalog]

    def context_doc(self) -> dict[str, Any] | None:
        """The case's context document (``None`` when the case names none)."""
        if self.context is None:
            return None
        if isinstance(self.context, str):
            data = json.loads(self.resolve(self.context).read_text(encoding="utf-8"))
            if not isinstance(data, Mapping):
                raise ValueError(f"{self.context}: a context document is a JSON object")
            return dict(data)
        return dict(self.context)

    def build_context(self, base: Context | None = None) -> Context:
        """A :class:`Context` for this case: ``base`` (or an empty context) updated with the context document's
        ``now``/``tz``/``locale``/``user``/``shareable``/``include_system`` and its ``sources`` (source specs, see
        :func:`jevtools.sources.specs.build_source`), with the case's messages."""
        doc = self.context_doc() or {}
        fields = {k: doc[k] for k in SETTING_FIELDS if k in doc}
        ctx = base or Context()
        if fields:
            parsed = Context.model_validate(fields)
            ctx = ctx.model_copy(update={k: getattr(parsed, k) for k in fields})
        specs = doc.get("sources")
        if specs:
            built = build_sources(specs, base_dir=self._base_dir)
            ctx = ctx.model_copy(update={"sources": {**ctx.sources, **{s.name: s for s in built}}})
        return ctx.with_messages(self.messages)

    def to_json(self) -> str:
        """One JSONL line (keys in schema order, non-ASCII kept)."""
        return json.dumps(self.model_dump(mode="json", exclude_defaults=True), ensure_ascii=False)


def parse_case(doc: Mapping[str, Any], *, base_dir: str | os.PathLike[str] | None = None) -> EvalCase:
    """Validate one case document."""
    return EvalCase.model_validate(dict(doc)).with_base_dir(base_dir)


def load(path: str | os.PathLike[str]) -> list[EvalCase]:
    """Read a JSONL dataset (blank lines skipped). Relative ``context``/``catalog`` paths resolve against the file's
    directory. Errors name the line."""
    file = Path(path)
    cases: list[EvalCase] = []
    seen: set[str] = set()
    for number, line in enumerate(file.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            case = parse_case(json.loads(line), base_dir=file.parent)
        except (ValueError, TypeError) as exc:
            raise ValueError(f"{file}:{number}: invalid case: {exc}") from exc
        if case.id in seen:
            raise ValueError(f"{file}:{number}: duplicate case id {case.id!r}")
        seen.add(case.id)
        cases.append(case)
    return cases


def loads(text: str, *, base_dir: str | os.PathLike[str] | None = None) -> list[EvalCase]:
    """Parse JSONL text."""
    return [parse_case(json.loads(line), base_dir=base_dir) for line in text.splitlines() if line.strip()]


def dump(cases: Iterable[EvalCase], path: str | os.PathLike[str]) -> None:
    """Write cases as JSONL."""
    Path(path).write_text("".join(case.to_json() + "\n" for case in cases), encoding="utf-8")


def select(cases: Sequence[EvalCase], *, tags: Iterable[str] = (), ids: Iterable[str] = ()) -> list[EvalCase]:
    """Cases carrying any of ``tags`` (all when empty) and, if given, whose id is in ``ids``."""
    wanted_tags, wanted_ids = set(tags), set(ids)
    return [
        c for c in cases
        if (not wanted_tags or wanted_tags & set(c.tags)) and (not wanted_ids or c.id in wanted_ids)
    ]  # fmt: skip


__all__ = [
    "CALL_OUTCOMES",
    "EvalCase",
    "Gold",
    "MatchMode",
    "dump",
    "load",
    "loads",
    "parse_case",
    "same_value",
    "select",
    "template_matches",
]
