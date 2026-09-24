"""The ``@jt.tool`` decorator (spec §2 quickstart, §7.1).

``@jt.tool`` or ``@jt.tool(risk="write", intent=...)`` turns a function into a :class:`Tool`: still callable, with
its JSON Schema (``Annotated`` markers included) and tool-level ``x-jev`` keys validated at decoration time.
"""

from __future__ import annotations

import functools
from collections.abc import Callable
from typing import Any, overload

from pydantic import ValidationError

from jevtools.errors import CatalogError
from jevtools.spec.ingest import RawTool, callable_schema
from jevtools.spec.xjev import ToolXJev


class Tool:
    """A Python callable declared as a jevtools tool. Calling it calls the function."""

    def __init__(
        self,
        fn: Callable[..., Any],
        *,
        name: str | None = None,
        description: str | None = None,
        xjev: dict[str, Any] | None = None,
    ) -> None:
        self.fn = fn
        self.name = name or fn.__name__
        self.parameters, summary = callable_schema(fn)
        self.description = description if description is not None else summary
        try:
            self.xjev = ToolXJev.model_validate(xjev or {}).declared()
        except ValidationError as exc:
            raise CatalogError(f"{self.name}: invalid x-jev: {exc}") from exc
        functools.update_wrapper(self, fn)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.fn(*args, **kwargs)

    def __repr__(self) -> str:
        return f"Tool({self.name!r})"

    def raw_tool(self) -> RawTool:
        """The ingest form used by :class:`~jevtools.spec.catalog.Catalog`."""
        return RawTool(
            name=self.name,
            description=self.description,
            parameters=self.parameters,
            tool_xjev=dict(self.xjev),
            markers_are_inline=True,
            origin="callable",
            definition=self,
        )

    def to_openai(self, *, strip: bool = False) -> dict[str, Any]:
        """OpenAI function-tool dict; ``strip=True`` removes every ``x-jev`` key (for forwarding to LLMs)."""
        from jevtools.spec.catalog import strip_xjev

        function: dict[str, Any] = {"name": self.name, "description": self.description, "parameters": self.parameters}
        if self.xjev:
            function["x-jev"] = self.xjev
        tool = {"type": "function", "function": function}
        return strip_xjev(tool) if strip else tool


@overload
def tool(fn: Callable[..., Any], /) -> Tool: ...


@overload
def tool(
    fn: None = None, /, *, name: str | None = None, description: str | None = None, **xjev: Any
) -> Callable[[Callable[..., Any]], Tool]: ...


def tool(
    fn: Callable[..., Any] | None = None,
    /,
    *,
    name: str | None = None,
    description: str | None = None,
    **xjev: Any,
) -> Tool | Callable[[Callable[..., Any]], Tool]:
    """Declare a function as a tool. Keyword arguments other than ``name``/``description`` are tool-level ``x-jev``
    keys (``risk``, ``intent``, ``constraints``, ``render``, …)."""

    def wrap(f: Callable[..., Any]) -> Tool:
        return Tool(f, name=name, description=description, xjev=xjev)

    return wrap(fn) if fn is not None else wrap


__all__ = ["Tool", "tool"]
