"""``Provider``: a host callable as a candidate source (spec §4.4).

``fn(q: SourceQuery) -> Iterable[Candidate | dict]``, sync or async. Dicts are ``{"value", "label"?, "text"?,
"attrs"?, "prov"?}``. Returned candidates are taken as retrieved for this query, so they count as evidence
(``prov["anchor"]`` = ``provider:<name>``) and carry the declared channel.
"""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Callable, Collection, Iterable, Mapping
from typing import Any

from jevtools.candidates import Candidate, Channel
from jevtools.sources.base import SourceQuery

ProviderFn = Callable[[SourceQuery], Any]


class Provider:
    """A callable candidate source (``jt.Provider(fn, provides={"email"})``)."""

    def __init__(
        self,
        fn: ProviderFn,
        name: str | None = None,
        provides: Collection[str] = (),
        channel: Channel | str = Channel.REGISTRY,
        *,
        item: str = "item",
    ) -> None:
        self.fn = fn
        self.name = name or getattr(fn, "__name__", "provider")
        self.provides: frozenset[str] = frozenset(provides)
        self.channel = Channel(channel)
        self.item = item

    def __repr__(self) -> str:
        return f"Provider({self.name!r})"

    @property
    def is_async(self) -> bool:
        return inspect.iscoroutinefunction(self.fn)

    def candidates(self, q: SourceQuery) -> list[Candidate]:
        """Call the provider (an async provider runs in a fresh event loop; use :meth:`acandidates` inside one)."""
        result = self.fn(q)
        if not inspect.isawaitable(result):
            return self._convert(result)
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return self._convert(asyncio.run(_await(result)))
        if inspect.iscoroutine(result):
            result.close()
        raise RuntimeError(f"provider {self.name!r} is async; call acandidates() inside an event loop")

    async def acandidates(self, q: SourceQuery) -> list[Candidate]:
        """Async variant (works for sync and async providers)."""
        result = self.fn(q)
        if inspect.isawaitable(result):
            result = await result
        return self._convert(result)

    def _convert(self, items: Iterable[Candidate | Mapping[str, Any]] | None) -> list[Candidate]:
        out: list[Candidate] = []
        for item in items or ():
            if isinstance(item, Candidate):
                prov = {"source": self.name, "anchor": f"provider:{self.name}", **item.prov}
                out.append(item.model_copy(update={"prov": prov}))
                continue
            value = item["value"]
            prov = {"source": self.name, "key": value, "anchor": f"provider:{self.name}", **dict(item.get("prov", {}))}
            out.append(
                Candidate(
                    label=str(item.get("label", "")),
                    value=value,
                    text=item.get("text"),
                    channel=self.channel,
                    prov=prov,
                    attrs=dict(item.get("attrs", {})),
                )
            )
        return out


async def _await(result: Any) -> Any:
    return await result


__all__ = ["Provider", "ProviderFn"]
