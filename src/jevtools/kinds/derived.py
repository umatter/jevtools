"""The ``derived``, ``const`` and ``secret`` resolvers (spec §4.1 last row): never asked.

- ``const`` (a ``derived`` slot with a schema ``const``/``default``) binds that value;
- ``readOnly`` (``derived`` without a value) is omitted — the server fills it;
- a ``default_from`` context path is read from the context (``derived`` or ``secret``);
- ``secret`` values are supplied from the context only; a required secret that the context lacks is ``missing``.

None of them is a factor; their pools are closed and empty (nothing reaches Jev).
"""

from __future__ import annotations

from collections.abc import Mapping

from jevtools.ballot import BallotQuestion
from jevtools.candidates import Bottom, Channel, Pool, value_key
from jevtools.kinds.base import ResolveContext, SlotResult, ValueEntry, register_resolver
from jevtools.spec.models import SlotSpec, ToolSpec
from jevtools.wire import Answer

SECRET_DISPLAY = "[secret]"


class DerivedResolver:
    """Resolver for ``kind: derived`` and ``kind: secret``."""

    def __init__(self, kind: str) -> None:
        self.kind = kind
        self.normalizer = "enum@1"

    def pool(self, tool: ToolSpec, slot: SlotSpec, rc: ResolveContext) -> Pool:
        return Pool(
            tool=tool.name,
            path=slot.path,
            kind=self.kind,
            closed=True,
            sentinels=(),
            notes=[f"{self.kind} slots are never asked"],
        )

    def questions(self, tool: ToolSpec, slot: SlotSpec, pool: Pool, rc: ResolveContext) -> list[BallotQuestion]:
        return []

    def decode(
        self, tool: ToolSpec, slot: SlotSpec, pool: Pool, answers: Mapping[str, Answer], rc: ResolveContext
    ) -> SlotResult:
        found, value, channel = self._value(slot, rc)
        if not found:
            bottom = Bottom.MISSING if slot.required and self.kind == "secret" else Bottom.OMIT
            return SlotResult(
                path=slot.path,
                kind=slot.kind,
                stakes=slot.stakes,
                dist={bottom.value: 1.0},
                values={},
                value=bottom,
                shape="missing" if bottom is Bottom.MISSING else "ok",
                factor=None,
                flags=("secret_unavailable",) if bottom is Bottom.MISSING else (),
                normalizer=self.normalizer,
                notes=(f"{self.kind} value not supplied",),
            )
        key = value_key(value)
        display = SECRET_DISPLAY if self.kind == "secret" else str(value)
        prov = {"secret": True} if self.kind == "secret" else {"derived": True}
        return SlotResult(
            path=slot.path,
            kind=slot.kind,
            stakes=slot.stakes,
            dist={key: 1.0},
            values={key: value},
            value=value,
            shape="ok",
            factor=None,
            display=display,
            channel=channel,
            prov=prov,
            normalizer=self.normalizer,
            entries={key: ValueEntry(display=display, channel=channel, prov=prov, p=1.0)},
        )

    def _value(self, slot: SlotSpec, rc: ResolveContext) -> tuple[bool, object, Channel | None]:
        if slot.default_from:
            try:
                return True, rc.ctx.lookup(slot.default_from), Channel.REGISTRY
            except KeyError:
                pass
        if slot.has_default and slot.default is not None:
            return True, slot.default, Channel.AUTHOR
        return False, None, None


register_resolver("derived", DerivedResolver("derived"))
register_resolver("secret", DerivedResolver("secret"))

__all__ = ["DerivedResolver"]
