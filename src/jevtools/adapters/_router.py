"""Shared adapter plumbing: the router that serves a request's tool list, and context overrides.

Framework adapters receive the tool list with every request (OpenAI ``tools``, LangChain ``bind_tools``, Pydantic AI
``function_tools``). The host's :class:`~jevtools.router.Router` stays authoritative for every tool it already
knows (its catalog may carry sidecars, ``Annotated`` markers and sources the request cannot express); tools it does
not know are compiled from the request's definitions. A request naming exactly the router's tools uses the router
itself; any other tool set gets a derived router (same backend, policy, context, Filler, Escalator, text LLM,
limits, calibrators), cached per router and tool set so pending click-resumes find their in-memory state again.
"""

from __future__ import annotations

import threading
import weakref
from collections.abc import Mapping, Sequence
from typing import Any

from jevtools.canonical import jsonable, sha256_of
from jevtools.context import SETTING_FIELDS, Context
from jevtools.router import Router
from jevtools.spec.catalog import Catalog, to_raw
from jevtools.spec.ingest import RawTool

_DERIVED: weakref.WeakKeyDictionary[Router, dict[str, Router]] = weakref.WeakKeyDictionary()
_LOCK = threading.Lock()
MAX_DERIVED = 64
"""Derived routers kept per base router (oldest dropped first)."""

CONTEXT_FIELDS = (*SETTING_FIELDS, "observations")
"""Context fields a request may override (``extra_body.jevtools.context``, ``configurable.jev_context``…)."""


def _raw_doc(raw: RawTool) -> dict[str, Any]:
    return {"name": raw.name, "description": raw.description, "parameters": jsonable(raw.parameters),
            "x-jev": jsonable(raw.tool_xjev), "meta": jsonable(raw.meta_xjev)}  # fmt: skip


def router_for(router: Router, tools: Sequence[Any] | None) -> Router:
    """The router that serves ``tools`` (OpenAI/MCP tool dicts, callables, pydantic models, LangChain tools
    converted to OpenAI dicts): ``router`` itself when the names match its catalog, else a cached derived router
    whose catalog keeps the router's compiled definition of every known tool and compiles the others."""
    if not tools:
        return router
    raws = [to_raw(t) for t in tools]
    names = [raw.name for raw in raws]
    if set(names) == set(router.catalog.names) and len(names) == len(router.catalog):
        return router
    known = {raw.name: raw for raw in router.catalog.raw}
    merged = [known.get(raw.name, raw) for raw in raws]
    key = sha256_of([_raw_doc(raw) if raw.name not in known else raw.name for raw in merged])
    with _LOCK:
        cache = _DERIVED.setdefault(router, {})
        derived = cache.get(key)
        if derived is not None:
            return derived
    catalog = Catalog(merged, sidecar=router.catalog.sidecar, sources=router.catalog.sources)
    derived = derive_router(router, catalog)
    with _LOCK:
        cache = _DERIVED.setdefault(router, {})
        cache.setdefault(key, derived)
        while len(cache) > MAX_DERIVED:
            cache.pop(next(iter(cache)))
        return cache[key]


def derive_router(router: Router, catalog: Catalog) -> Router:
    """A router like ``router`` over another catalog (a custom ``revalidate`` hook is kept; the default one is
    rebound to the new router)."""
    revalidate = router.revalidate
    custom = getattr(revalidate, "__self__", None) is not router
    return type(router)(
        catalog,
        backend=router.backend,
        policy=router.policy,
        context=router.context,
        filler=router.filler,
        escalator=router.escalator,
        text_llm=router.text_llm,
        limits=router.limits,
        trace_store=router.trace_store,
        estimator=router.estimator,
        calibrators=router.calibrators,
        revalidate=revalidate if custom else None,
        store_bodies=router.store_bodies,
    )


def merge_context(base: Context | None, override: Context | Mapping[str, Any] | None) -> Context | None:
    """``override`` applied over ``base``: a :class:`Context` replaces it; a mapping updates the fields in
    :data:`CONTEXT_FIELDS` (validated through ``Context``). Messages and sources always stay the base's (the
    conversation comes from the request; sources are registered in host code)."""
    if override is None:
        return base
    if isinstance(override, Context):
        return override
    fields = {k: v for k, v in override.items() if k in CONTEXT_FIELDS}
    unknown = sorted(set(override) - set(CONTEXT_FIELDS))
    if unknown:
        raise ValueError(f"unsupported context field(s) {unknown}; allowed: {', '.join(CONTEXT_FIELDS)}")
    parsed = Context.model_validate(fields)
    update = {k: getattr(parsed, k) for k in fields}
    return (base or Context()).model_copy(update=update)


__all__ = ["CONTEXT_FIELDS", "MAX_DERIVED", "derive_router", "merge_context", "router_for"]
