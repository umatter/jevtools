"""Prompts (spec §3.8.4): confirm cards, clarify menus, open questions and notices — templates filled with candidate
labels, never generated text. Every builder returns the :class:`~jevtools.decision.Prompt` together with the
:class:`~jevtools.decision.PendingAction` of each option id, so a click can be resumed without a Jev call.

Option ids: ``ok``, ``alt:<slot>:<i>``, ``change``, ``cancel`` on confirm cards; ``pick:<slot>:<i>`` and ``other``
on slot menus; ``tool:<name>`` on tool menus; ``yes``/``no`` on yes/no menus.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from jevtools import templates
from jevtools.candidates import Candidate, display_value
from jevtools.canonical import nfc
from jevtools.decision import PendingAction, Prompt, PromptOption
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

    @classmethod
    def of(cls, candidate: Candidate) -> Binding:
        """The binding of a pool candidate."""
        return cls(value=candidate.value, display=candidate.shown, label=candidate.label or None,
                   attrs=dict(candidate.attrs))  # fmt: skip

    @classmethod
    def of_value(cls, value: Any, *, display: str | None = None, label: str | None = None,
                 attrs: Mapping[str, Any] | None = None) -> Binding:  # fmt: skip
        """A binding from a decoded value (display defaults to the value's display form)."""
        return cls(value=value, display=display if display is not None else display_value(value), label=label,
                   attrs=dict(attrs or {}))  # fmt: skip


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


def render_call(tool: ToolSpec, bindings: Mapping[str, Binding], *, only: Sequence[str] | None = None) -> str:
    """One-line rendering of a call (``x-jev.render``, else ``{intent}: p1=…, p2=…`` over ``only`` or all slots)."""
    if tool.render is not None:
        return fill_template(tool.render, tool, bindings)
    names = list(only) if only is not None else [s.name for s in tool.slots if s.name in bindings]
    params = ", ".join(f"{name}={{{name}}}" for name in names)
    return fill_template(f"{{intent}}: {params}" if params else "{intent}", tool, bindings)


def confirm_text(tool: ToolSpec, bindings: Mapping[str, Binding]) -> str:
    """``confirm_template`` or ``{Render}?``."""
    rendered = render_call(tool, bindings)
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
    options: list[PromptOption] = []
    actions: Actions = {}
    for i, choice in enumerate(choices):
        oid = f"pick:{slot.name}:{i}"
        text = choice.display
        if complete_call:
            base = choice_bindings[i] if choice_bindings is not None else bindings
            text = templates.upper_first(render_call(tool, {**base, slot.name: choice}))
        options.append(PromptOption(id=oid, text=_short(text)))
        actions[oid] = PendingAction(action="bind", slot=slot.name, value=choice.value, label=choice.label)
    options.append(PromptOption(id="other", text=templates.SOMETHING_ELSE))
    actions["other"] = PendingAction(action="open", slot=slot.name)
    text = templates.render(templates.CLARIFY_MENU, noun_short=noun_short(slot))
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
    """Match a short free-text reply to an option id (§3.8.5 click): an option number (1-based), an option text or
    id, ``yes``/``ok``/``send`` (→ ``ok``) or ``cancel``. ``None`` when the reply is not a click."""
    entries = [(o.id, o.text) if isinstance(o, PromptOption) else (o["id"], o["text"]) for o in options]
    reply = _norm(text)
    if reply.isdigit() and 1 <= int(reply) <= len(entries):
        return entries[int(reply) - 1][0]
    for oid, label in entries:
        if reply in (_norm(label), oid.casefold()):
            return oid
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
    "noun_short",
    "ok_text",
    "open_question",
    "parse_short_reply",
    "refuse_notice",
    "render_call",
    "tool_menu",
    "yes_no_menu",
]
