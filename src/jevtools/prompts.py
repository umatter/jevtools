"""Prompts (spec §3.8.4): confirm cards, clarify menus, open questions and notices — templates filled with candidate
labels, never generated text. Every builder returns the :class:`~jevtools.decision.Prompt` together with the
:class:`~jevtools.decision.PendingAction` of each option id, so a click can be resumed without a Jev call.

Option ids: ``ok``, ``alt:<slot>:<i>``, ``change``, ``cancel`` on confirm cards; ``pick:<slot>:<i>`` and ``other``
on slot menus (and on grid menus, §4.2.4); ``tool:<name>`` on tool menus; ``yes``/``no`` on yes/no menus.

**Default call rendering** (a tool without ``x-jev.render``; see :func:`natural_call`, deviation recorded in
``docs/DECISIONS.md``, Core polish): a deterministic template instead of the spec's ``{intent}: p1=…, p2=…`` dump:

- the head is the intent with trailing phrases dropped when a slot restates them (:func:`short_intent`:
  ``send an email from the user to one recipient`` → ``send an email``);
- slots named like a preposition (``to``, ``from``, ``to_account``) attach to the head (``to Anna Keller <…>``);
  the others follow an em dash as ``<term> <value>`` pairs, where the term is the slot noun when it is short
  (≤ 2 words, ≤ 20 characters), else the humanized parameter name without its unit suffix (:func:`slot_term`);
- values: labels for ``ref``/``enum``/``temporal`` candidates, quoted single-line previews (≤ 60 characters, ``…``)
  for text, ``45 minutes`` for quantities with a unit, ``yes``/``no`` for flags, list items joined with commas and
  ``and``; empty values and secrets are omitted, and cosmetic values are dropped when the line exceeds 200
  characters (320 on a confirm card). The full call stays in ``Decision.call``.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from jevtools import templates
from jevtools.candidates import Candidate, display_value, value_key
from jevtools.canonical import nfc
from jevtools.decision import PendingAction, Prompt, PromptOption
from jevtools.extract.numbers import canonical_unit
from jevtools.policy import Tier
from jevtools.spec.models import SlotSpec, ToolSpec

CLARIFY_OPEN_TOOL = "What would you like me to do?"
"""Open clarify when no single slot is the bottleneck (tool undecided)."""
CLARIFY_YES_NO = "Do you want {noun} to be true?"
"""Yes/no menu of a flag in the Noul dead band."""
OPTION_YES = "Yes"
OPTION_NO = "No"
MISSING = "…"
"""Shown for a placeholder whose slot has no bound value."""
MENU_TEXT_MAX = 200
RENDER_BUDGET = 200
"""A default rendering (menu option, joint option) longer than this drops its cosmetic values (the bottleneck slot of
a menu is always kept)."""
CONFIRM_BUDGET = 320
"""The same limit for a confirm card, whose text is not cut like a menu option."""
PREVIEW_MAX = 60
"""Text values show a single-line preview of at most this many characters (``…`` marks a cut)."""
TERM_MAX_WORDS = 2
TERM_MAX_CHARS = 20
"""A slot noun this short names the value in a rendering; a longer one gives way to the parameter name."""
LEAD_WORDS: frozenset[str] = frozenset({"to", "from", "cc", "bcc", "into", "onto", "at", "on", "in", "for", "with",
                                         "via", "by"})  # fmt: skip
"""Parameter names that read as a preposition: ``to`` renders ``to <value>`` right after the head."""
LEAD_PREFIXES: tuple[str, ...] = ("to", "from")
"""``to_account``/``from_date`` render ``to <value>``/``from <value>`` too."""
INTENT_CUTS: frozenset[str] = frozenset({"from", "to", "between", "for", "in", "into", "on", "with", "and", "by",
                                          "via", "using", "at", "about"})  # fmt: skip
"""Words that open a trailing phrase of the intent (considered from the third word on)."""
GENERIC_WORDS: frozenset[str] = frozenset({"the", "a", "an", "one", "two", "some", "any", "each", "given",
                                           "specified", "provided", "specific", "user", "users", "own", "their",
                                           "his", "her", "its", "of", "that", "this", "s"})  # fmt: skip
"""Words that carry nothing a slot could restate (a trailing phrase of only these is dropped)."""
UNIT_WORDS: dict[str, tuple[str, str]] = {
    "millisecond": ("millisecond", "milliseconds"), "second": ("second", "seconds"), "minute": ("minute", "minutes"),
    "hour": ("hour", "hours"), "day": ("day", "days"), "week": ("week", "weeks"), "month": ("month", "months"),
    "year": ("year", "years"), "byte": ("byte", "bytes"), "kilobyte": ("kB", "kB"), "megabyte": ("MB", "MB"),
    "gigabyte": ("GB", "GB"), "terabyte": ("TB", "TB"),
}  # fmt: skip
"""Display words of canonical units (singular, plural); ``percent`` renders as ``%``."""
SHORT_CONFIRM = ("yes", "ok", "send")
SHORT_CANCEL = ("cancel",)

Actions = dict[str, PendingAction]
_PLACEHOLDER = re.compile(r"\{([A-Za-z_][A-Za-z0-9_]*)((?:\.[A-Za-z0-9_]+)*)\}")


@dataclass(frozen=True)
class Binding:
    """A bound slot value as prompts render it: ``{p}`` → ``display``, ``{p.label}`` → ``label``, ``{p.<attr>}``."""

    value: Any
    display: str
    label: str | None = None
    attrs: Mapping[str, Any] = field(default_factory=dict)
    items: tuple[str, ...] = ()
    """For list values: the label (else display form) of each item, in value order (default rendering only)."""

    @classmethod
    def of(cls, candidate: Candidate) -> Binding:
        """The binding of a pool candidate."""
        return cls(value=candidate.value, display=candidate.shown, label=candidate.label or None,
                   attrs=dict(candidate.attrs))  # fmt: skip

    @classmethod
    def of_value(cls, value: Any, *, display: str | None = None, label: str | None = None,
                 attrs: Mapping[str, Any] | None = None, items: Sequence[str] = ()) -> Binding:  # fmt: skip
        """A binding from a decoded value (display defaults to the value's display form)."""
        return cls(value=value, display=display if display is not None else display_value(value), label=label,
                   attrs=dict(attrs or {}), items=tuple(items))  # fmt: skip

    @classmethod
    def of_result(cls, result: Any) -> Binding:
        """The binding of a decoded :class:`~jevtools.kinds.base.SlotResult` (list items keep their part labels:
        ``Bob Meier <bob.meier@muster.ch>`` rather than the bare address)."""
        items: list[str] = []
        if isinstance(result.value, (list, tuple)):
            shown = {value_key(part.value): part.label or part.display for part in result.parts.values()
                     if part.label or part.display}  # fmt: skip
            items = [shown.get(value_key(v)) or display_value(v) for v in result.value]
        return cls.of_value(result.value, display=result.display, label=result.label, attrs=result.attrs, items=items)


def _lookup(binding: Binding, attrs: Sequence[str]) -> str:
    if not attrs:
        return binding.display
    if attrs == ["label"]:
        return binding.label or binding.display
    node: Any = binding.attrs
    for part in attrs:
        if isinstance(node, Mapping) and part in node:
            node = node[part]
        elif isinstance(binding.value, Mapping) and node is binding.attrs and part in binding.value:
            node = binding.value[part]
        else:
            return MISSING
    return display_value(node)


def fill_template(template: str, tool: ToolSpec, bindings: Mapping[str, Binding], *, render: str | None = None) -> str:
    """Substitute ``{intent}``, ``{Intent}``, ``{render}``/``{Render}`` and ``{slot}``, ``{slot.label}``,
    ``{slot.<attr>}`` placeholders; unbound slots show ``…``."""

    def sub(match: re.Match[str]) -> str:
        head, rest = match.group(1), match.group(2)
        attrs = rest[1:].split(".") if rest else []
        if head in bindings:
            return _lookup(bindings[head], attrs)
        if head in ("intent", "Intent") and not attrs:
            return tool.intent if head == "intent" else templates.upper_first(tool.intent)
        if head in ("render", "Render") and not attrs and render is not None:
            return render if head == "render" else templates.upper_first(render)
        return MISSING

    return _PLACEHOLDER.sub(sub, template)


def render_call(
    tool: ToolSpec,
    bindings: Mapping[str, Binding],
    *,
    only: Sequence[str] | None = None,
    focus: str | None = None,
    budget: int = RENDER_BUDGET,
) -> str:
    """One-line rendering of a call: ``x-jev.render`` when declared, else :func:`natural_call` over ``only`` (or every
    bound slot); ``focus`` (a menu's slot) is never dropped for length."""
    if tool.render is not None:
        return fill_template(tool.render, tool, bindings)
    return natural_call(tool, bindings, only=only, focus=focus, budget=budget)


# --------------------------------------------------------------------------------------------------------------------
# Default rendering (no x-jev.render): deterministic templates over labels, never generated text
# --------------------------------------------------------------------------------------------------------------------

_WORD = re.compile(r"[a-z0-9]+")


def _stem(word: str) -> str:
    return word[:-1] if len(word) > 3 and word.endswith("s") else word


def _content_words(text: str) -> set[str]:
    return {_stem(w) for w in _WORD.findall(text.lower())} - GENERIC_WORDS - INTENT_CUTS


def short_intent(tool: ToolSpec) -> str:
    """The intent without the trailing phrases a slot restates (or that carry only generic words).

    From the third word on, the intent is split before each word of :data:`INTENT_CUTS`; a phrase is dropped when
    its content words are empty or share a word with a slot's name or noun: ``send an email from the user to one
    recipient`` → ``send an email`` (``recipient`` is the ``to`` slot's noun), ``get the current weather for a
    city`` → ``get the current weather``; ``post a message to the #general channel`` keeps its phrase when no slot
    names a channel."""
    words = tool.intent.split()
    cuts = [i for i, word in enumerate(words) if i >= 2 and word.lower() in INTENT_CUTS]
    if not cuts:
        return tool.intent
    restated: set[str] = set()
    for slot in tool.slots:
        restated |= _content_words(f"{templates.humanize(slot.name)} {slot.noun}")
    kept = words[: cuts[0]]
    for start, end in zip(cuts, [*cuts[1:], len(words)], strict=True):
        phrase = words[start:end]
        content = _content_words(" ".join(phrase))
        if content and not content & restated:
            kept += phrase
    return " ".join(kept)


def slot_term(slot: SlotSpec) -> str:
    """What names a value in a rendering: the slot noun without its article when short (``subject line``,
    ``invitees``), else the humanized parameter name without a unit suffix (``duration_minutes`` → ``duration``)."""
    noun = noun_short(slot)
    if len(noun.split()) <= TERM_MAX_WORDS and len(noun) <= TERM_MAX_CHARS:
        return noun
    words = templates.humanize(slot.name).split()
    if len(words) > 1 and slot.unit is not None and canonical_unit(words[-1]) == canonical_unit(slot.unit):
        words = words[:-1]
    return " ".join(words) or slot.name


def _lead(slot: SlotSpec) -> str | None:
    name = slot.name.lower()
    if name in LEAD_WORDS:
        return name
    head, sep, rest = name.partition("_")
    return head if sep and rest and head in LEAD_PREFIXES else None


def preview(text: str, limit: int = PREVIEW_MAX) -> str:
    """A single-line preview: whitespace collapsed, cut at a word boundary to ``limit`` characters plus ``…``."""
    line = " ".join(text.split())
    if len(line) <= limit:
        return line
    cut = line[: limit - 1]
    space = cut.rfind(" ")
    if space >= limit // 2:
        cut = cut[:space]
    return cut.rstrip(" ,;:.-–—") + "…"


def _is_empty(value: Any) -> bool:
    return value is None or (isinstance(value, (str, list, tuple, dict)) and not value)


def _unit_text(amount: str, unit: str | None) -> str:
    if unit is None:
        return amount
    if unit == "percent":
        return f"{amount}%"
    singular, plural = UNIT_WORDS.get(unit, (unit, unit))
    return f"{amount} {singular if amount in ('1', '1.0') else plural}"


def _item_kind(slot: SlotSpec) -> str | None:
    return slot.item.kind if slot.item is not None else None


def value_text(slot: SlotSpec, binding: Binding) -> str | None:
    """The rendered value of one slot (``None``: omitted — empty values and secrets)."""
    value = binding.value
    if slot.kind == "secret" or _is_empty(value):
        return None
    if slot.kind == "list" and isinstance(value, (list, tuple)):
        items = list(binding.items) or [display_value(v) for v in value]
        quoted = _item_kind(slot) == "text"
        return join_and([f'"{preview(i)}"' if quoted else preview(i, TERM_MAX_CHARS * 4) for i in items])
    if slot.kind == "flag" and isinstance(value, bool):
        return "yes" if value else "no"
    if slot.kind == "text":
        return f'"{preview(binding.display)}"'
    if slot.kind == "quantity":
        return _unit_text(binding.display, canonical_unit(slot.unit))
    if slot.kind in ("ref", "enum", "temporal", "ordinal"):
        return binding.label or binding.display
    return preview(binding.label or binding.display)


def join_and(items: Sequence[str]) -> str:
    """``a``, ``a and b``, ``a, b and c``."""
    if len(items) <= 1:
        return "".join(items)
    return ", ".join(items[:-1]) + " and " + items[-1]


def natural_call(
    tool: ToolSpec,
    bindings: Mapping[str, Binding],
    *,
    only: Sequence[str] | None = None,
    focus: str | None = None,
    budget: int = RENDER_BUDGET,
) -> str:
    """The default one-line rendering (module docstring): ``<short intent> <lead phrases> — <term value>, …``.

    ``Send an email to Anna Keller <anna.keller@acme.com> — subject line "Running 10 minutes late", body "Hi
    Anna, I'll be 10 minutes late. Best, Sam"`` (the confirm card adds ``?``)."""
    names = list(only) if only is not None else [s.name for s in tool.slots if s.name in bindings]
    parts: list[tuple[SlotSpec, str | None, str]] = []
    for name in names:
        if name not in bindings or name not in tool.slot_names:
            continue
        slot = tool.slot(name)
        text = value_text(slot, bindings[name])
        if text is not None:
            parts.append((slot, _lead(slot), text))

    def line(selected: Sequence[tuple[SlotSpec, str | None, str]]) -> str:
        head = " ".join([short_intent(tool), *(f"{lead} {text}" for _, lead, text in selected if lead)])
        rest = [f"{slot_term(slot)} {text}" for slot, lead, text in selected if not lead]
        return f"{head} — {', '.join(rest)}" if rest else head

    rendered = line(parts)
    if len(rendered) > budget:
        rendered = line([p for p in parts if p[0].stakes != "cosmetic" or p[0].name == focus])
    return rendered


def confirm_text(tool: ToolSpec, bindings: Mapping[str, Binding]) -> str:
    """``confirm_template`` or ``{Render}?``."""
    rendered = render_call(tool, bindings, budget=CONFIRM_BUDGET)
    template = tool.confirm_template or templates.CONFIRM_DEFAULT
    return fill_template(template, tool, bindings, render=rendered)


def ok_text(tool: ToolSpec) -> str:
    """The confirm button: ``Confirm`` in the critical tier, else the intent's verb (``Send``, ``Create``)."""
    if tool.tier is Tier.CRITICAL:
        return "Confirm"
    verb = tool.intent.split()[0] if tool.intent.strip() else "OK"
    return templates.upper_first(verb)


@dataclass(frozen=True)
class AltChoice:
    """A runner-up offered on a confirm card: ``alt:<slot>:<i>``."""

    slot: str
    index: int
    value: Any
    display: str
    p: float
    label: str | None = None


def confirm_card(
    tool: ToolSpec, bindings: Mapping[str, Binding], alternatives: Sequence[AltChoice], *, bottleneck: str | None
) -> tuple[Prompt, Actions]:
    """Confirm card: ``ok``, one ``alt:<slot>:<i>`` per offered runner-up, ``change`` (opens the bottleneck menu),
    ``cancel``."""
    options = [PromptOption(id="ok", text=ok_text(tool))]
    actions: Actions = {"ok": PendingAction(action="confirm")}
    for alt in alternatives:
        oid = f"alt:{alt.slot}:{alt.index}"
        options.append(PromptOption(id=oid, text=f"{alt.display} instead"))
        actions[oid] = PendingAction(action="bind", slot=alt.slot, value=alt.value, label=alt.label)
    options += [PromptOption(id="change", text=templates.OPTION_CHANGE),
                PromptOption(id="cancel", text=templates.OPTION_CANCEL)]  # fmt: skip
    actions["change"] = PendingAction(action="open", slot=bottleneck)
    actions["cancel"] = PendingAction(action="cancel")
    return Prompt(kind="confirm", text=confirm_text(tool, bindings), options=options), actions


def noun_short(slot: SlotSpec) -> str:
    """The slot noun without a leading article (``the recipient's email address`` → ``recipient's email address``)."""
    return re.sub(r"^(the|a|an)\s+", "", slot.noun, flags=re.IGNORECASE)


def clarify_menu(
    tool: ToolSpec,
    slot: SlotSpec,
    choices: Sequence[Binding],
    *,
    bindings: Mapping[str, Binding],
    complete_call: bool,
    choice_bindings: Sequence[Mapping[str, Binding]] | None = None,
) -> tuple[Prompt, Actions]:
    """Slot menu ``Which {noun_short} did you mean?`` with ``pick:<slot>:<i>`` options plus ``Something else``.

    With ``complete_call`` (external/critical tiers) each option shows the whole resulting call, so a click is a
    binding and a confirmation (§3.8.5); ``choice_bindings[i]`` (the call re-bound with choice ``i``, late-bound
    values recomputed) replaces ``bindings`` for option ``i``.
    """
    text = templates.render(templates.CLARIFY_MENU, noun_short=noun_short(slot))
    return _slot_menu(text, tool, slot, choices, [c.display for c in choices], bindings=bindings,
                      complete_call=complete_call, choice_bindings=choice_bindings)  # fmt: skip


def grid_menu(
    tool: ToolSpec,
    slot: SlotSpec,
    choices: Sequence[Binding],
    *,
    bindings: Mapping[str, Binding] | None = None,
    complete_call: bool = False,
    choice_bindings: Sequence[Mapping[str, Binding]] | None = None,
) -> tuple[Prompt, Actions]:
    """Grid menu (§4.2.4) of a required slot without a default that nothing stated: the open question
    (``x-jev.ask`` or ``What should {noun} be?``) with ``pick:<slot>:<i>`` grid values (``45 minutes``) plus
    ``Something else``. When the rest of the call is known (``bindings``) and ``complete_call``, each option shows
    the whole resulting call, as on a clarify menu."""
    text = slot.ask or templates.render(templates.CLARIFY_OPEN, noun=slot.noun)
    shown = [value_text(slot, c) or c.display for c in choices]
    return _slot_menu(text, tool, slot, choices, shown, bindings=bindings or {},
                      complete_call=complete_call and bindings is not None,
                      choice_bindings=choice_bindings)  # fmt: skip


def _slot_menu(
    text: str,
    tool: ToolSpec,
    slot: SlotSpec,
    choices: Sequence[Binding],
    shown: Sequence[str],
    *,
    bindings: Mapping[str, Binding],
    complete_call: bool,
    choice_bindings: Sequence[Mapping[str, Binding]] | None,
) -> tuple[Prompt, Actions]:
    options: list[PromptOption] = []
    actions: Actions = {}
    for i, choice in enumerate(choices):
        oid = f"pick:{slot.name}:{i}"
        option = shown[i]
        if complete_call:
            base = choice_bindings[i] if choice_bindings is not None else bindings
            option = templates.upper_first(render_call(tool, {**base, slot.name: choice}, focus=slot.name))
        options.append(PromptOption(id=oid, text=_short(option)))
        actions[oid] = PendingAction(action="bind", slot=slot.name, value=choice.value, label=choice.label)
    options.append(PromptOption(id="other", text=templates.SOMETHING_ELSE))
    actions["other"] = PendingAction(action="open", slot=slot.name)
    return Prompt(kind="menu", text=text, options=options), actions


def yes_no_menu(tool: ToolSpec, slot: SlotSpec) -> tuple[Prompt, Actions]:
    """Yes/no menu for a flag in the Noul dead band."""
    options = [PromptOption(id="yes", text=OPTION_YES), PromptOption(id="no", text=OPTION_NO)]
    actions = {"yes": PendingAction(action="bind", slot=slot.name, value=True),
               "no": PendingAction(action="bind", slot=slot.name, value=False)}  # fmt: skip
    text = templates.render(CLARIFY_YES_NO, noun=slot.noun)
    return Prompt(kind="menu", text=text, options=options), actions


def tool_menu(first: ToolSpec, second: ToolSpec) -> tuple[Prompt, Actions]:
    """``Do you want me to {intent(t1)} or {intent(t2)}?`` with ``tool:<name>`` options and ``cancel``."""
    options = [PromptOption(id=f"tool:{t.name}", text=templates.upper_first(t.intent)) for t in (first, second)]
    actions: Actions = {f"tool:{t.name}": PendingAction(action="tool", tool=t.name) for t in (first, second)}
    options.append(PromptOption(id="cancel", text=templates.OPTION_CANCEL))
    actions["cancel"] = PendingAction(action="cancel")
    text = templates.render(templates.CLARIFY_TOOL, intent1=first.intent, intent2=second.intent)
    return Prompt(kind="menu", text=text, options=options), actions


def open_question(slot: SlotSpec | None) -> tuple[Prompt, Actions]:
    """``x-jev.ask`` or ``What should {noun} be?``; without a slot, ``What would you like me to do?``."""
    if slot is None:
        return Prompt(kind="open", text=CLARIFY_OPEN_TOOL), {}
    text = slot.ask or templates.render(templates.CLARIFY_OPEN, noun=slot.noun)
    return Prompt(kind="open", text=text), {}


def refuse_notice(tool: ToolSpec, source: str) -> Prompt:
    """``I did not act on instructions found in {source}. Tell me directly if you want me to {intent}.``"""
    return Prompt(kind="notice", text=templates.render(templates.REFUSE, source=source, intent=tool.intent))


def _short(text: str) -> str:
    return text if len(text) <= MENU_TEXT_MAX else text[: MENU_TEXT_MAX - 1].rstrip() + "…"


def _norm(text: str) -> str:
    return nfc(text).strip().rstrip(".!").strip().casefold()


def parse_short_reply(text: str, options: Sequence[PromptOption | Mapping[str, str]]) -> str | None:
    """Match a short free-text reply to an option id (§3.8.5 click): an option text or id, an option number
    (1-based), ``yes``/``ok``/``send`` (→ ``ok``) or ``cancel``. ``None`` when the reply is not a click.

    Texts win over numbers: in a menu whose option texts are themselves numbers (quantities, grids), ``"2"`` is the
    option that reads ``2``; there (any option text starting with a digit) a number matching no text is ambiguous —
    option number or value? — so it is not a click and goes through the free-text path."""
    entries = [(o.id, o.text) if isinstance(o, PromptOption) else (o["id"], o["text"]) for o in options]
    reply = _norm(text)
    for oid, label in entries:
        if reply in (_norm(label), oid.casefold()):
            return oid
    numeric_menu = any(_norm(label)[:1].isdigit() for _, label in entries)
    if reply.isdigit() and not numeric_menu and 1 <= int(reply) <= len(entries):
        return entries[int(reply) - 1][0]
    ids = {oid for oid, _ in entries}
    if reply in SHORT_CONFIRM and "ok" in ids:
        return "ok"
    if reply in SHORT_CANCEL and "cancel" in ids:
        return "cancel"
    return None


__all__ = [
    "CLARIFY_OPEN_TOOL",
    "CLARIFY_YES_NO",
    "Actions",
    "AltChoice",
    "Binding",
    "clarify_menu",
    "confirm_card",
    "confirm_text",
    "fill_template",
    "grid_menu",
    "join_and",
    "natural_call",
    "noun_short",
    "ok_text",
    "open_question",
    "parse_short_reply",
    "preview",
    "refuse_notice",
    "render_call",
    "short_intent",
    "slot_term",
    "tool_menu",
    "value_text",
    "yes_no_menu",
]
