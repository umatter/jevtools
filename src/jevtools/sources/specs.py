"""Source specs (spec §4.4): candidate sources described as data, shared by ``jevtools.toml`` (the proxy), evaluation
context documents (``jevtools.eval``) and the CLI's ``--sources`` files.

One spec per source::

    {"name": "contacts", "type": "registry", "path": "contacts.csv", "key": "email", "label": "{name} <{email}>",
     "match": ["name", "aliases"], "provides": ["email", "person"], "list_fields": ["aliases"]}

- ``type``: ``registry`` (:class:`~jevtools.sources.Registry`; rows inline or from JSON/JSONL/CSV/YAML),
  ``files`` (:class:`~jevtools.sources.FileIndex`; a JSON list or one path per line), ``provider``
  (:class:`~jevtools.sources.Provider` over a host ``module:function``) or ``source`` (any source object a
  ``module:function`` factory returns).
- The CLI spellings are accepted too: ``kind`` for ``type``, ``rows_file``/``paths_file`` for ``path``, ``paths``
  for the rows of a file index.
- A spec without data (no ``rows``, ``path`` or ``function``) is a *descriptor*: enough for catalog inference and
  ``jevtools lint`` (names and ``provides`` tags), built only with ``descriptors=True``.
"""

from __future__ import annotations

import csv
import importlib
import io
import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from jevtools.sources.files import FileIndex
from jevtools.sources.provider import Provider
from jevtools.sources.registry import Registry

SourceType = Literal["registry", "files", "provider", "source"]

KEY_GUESSES = ("id", "key", "email", "value", "name", "path")
"""Key fields tried, in order, for per-request rows of a source the configuration does not declare."""


def load_object(ref: str) -> Any:
    """Import ``package.module:attribute`` (the attribute may be dotted)."""
    module_name, sep, attr = ref.partition(":")
    if not sep or not module_name or not attr:
        raise ValueError(f"expected 'module:attribute', got {ref!r}")
    obj: Any = importlib.import_module(module_name)
    for part in attr.split("."):
        obj = getattr(obj, part)
    return obj


# --------------------------------------------------------------------------------------------------------------------
# Sources
# --------------------------------------------------------------------------------------------------------------------


class SourceConfig(BaseModel):
    """One candidate source (spec §4.4): a :class:`~jevtools.sources.Registry` (``registry``), a
    :class:`~jevtools.sources.FileIndex` (``files``), a :class:`~jevtools.sources.Provider` over a host callable
    (``provider``), or any source object a ``module:function`` factory returns (``source``)."""

    model_config = ConfigDict(extra="forbid")

    name: str
    type: SourceType = "registry"
    path: str | None = None
    rows: list[Any] | None = None
    function: str | None = None
    options: dict[str, Any] = Field(default_factory=dict)
    key: str | None = None
    label: str | None = None
    describe: str | None = None
    match: list[str] = Field(default_factory=list)
    provides: list[str] = Field(default_factory=list)
    attrs: list[str] = Field(default_factory=list)
    retriever: str = "fuzzy"
    send_whole_if_under: int = 12
    synonyms: dict[str, list[str]] | None = None
    channel: str = "registry"
    item: str | None = None
    recency: str | None = None
    hierarchy: str | None = None
    groups: str | None = None
    k: int = 40
    list_fields: list[str] = Field(default_factory=list)
    list_sep: str = ";"

    @model_validator(mode="before")
    @classmethod
    def _cli_spellings(cls, data: Any) -> Any:
        """Accept the CLI's spellings: ``kind``, ``rows_file``, ``paths_file`` and ``paths``."""
        if not isinstance(data, Mapping):
            return data
        spec = dict(data)
        if "kind" in spec:
            spec.setdefault("type", spec.pop("kind"))
        if "paths" in spec or "paths_file" in spec:
            spec.setdefault("type", "files")
        if "paths" in spec:
            spec.setdefault("rows", spec.pop("paths"))
        for key in ("rows_file", "paths_file"):
            if key in spec:
                spec.setdefault("path", spec.pop(key))
        return spec

    @property
    def has_data(self) -> bool:
        """Whether the spec can build a source (rows, a file or a function); otherwise it is a descriptor."""
        return self.rows is not None or self.path is not None or self.function is not None


def _read_rows(path: Path, spec: SourceConfig) -> list[Any]:
    text = path.read_text(encoding="utf-8")
    suffix = path.suffix.lower()
    if suffix == ".csv":
        rows: list[Any] = [dict(r) for r in csv.DictReader(io.StringIO(text))]
    elif suffix == ".jsonl":
        rows = [json.loads(line) for line in text.splitlines() if line.strip()]
    elif suffix in (".json", ".yaml", ".yml"):
        data = json.loads(text) if suffix == ".json" else _yaml(text)
        rows = list(data.get("rows", []) if isinstance(data, Mapping) else data)
    else:  # one item per line (paths)
        rows = [line.strip() for line in text.splitlines() if line.strip()]
    if spec.list_fields:
        for row in rows:
            if isinstance(row, dict):
                for name in spec.list_fields:
                    value = row.get(name)
                    if isinstance(value, str):
                        row[name] = [v.strip() for v in value.split(spec.list_sep) if v.strip()]
    return rows


def _yaml(text: str) -> Any:
    try:
        import yaml  # type: ignore[import-untyped]
    except ModuleNotFoundError as exc:  # pragma: no cover - depends on the environment
        raise ValueError("YAML files need PyYAML: pip install 'jevtools[yaml]'") from exc
    return yaml.safe_load(text)


def _rows(spec: SourceConfig, base_dir: Path | None) -> list[Any]:
    if spec.rows is not None:
        return list(spec.rows)
    if spec.path is None:
        raise ValueError(f"source {spec.name!r}: give `rows` or `path`")
    path = Path(spec.path)
    if not path.is_absolute() and base_dir is not None:
        path = base_dir / path
    return _read_rows(path, spec)


def guess_key(rows: Sequence[Mapping[str, Any]]) -> str:
    """The key field of rows without a declared key: the first of :data:`KEY_GUESSES` every row has."""
    for name in KEY_GUESSES:
        if rows and all(isinstance(r, Mapping) and r.get(name) not in (None, "") for r in rows):
            return name
    raise ValueError(f"cannot tell the key field of these rows; declare one of {', '.join(KEY_GUESSES)} or `key`")


def build_source(spec: SourceConfig | Mapping[str, Any], *, base_dir: str | os.PathLike[str] | None = None) -> Any:
    """Build one source from its spec (paths relative to ``base_dir``)."""
    cfg = spec if isinstance(spec, SourceConfig) else SourceConfig.model_validate(dict(spec))
    folder = Path(base_dir) if base_dir is not None else None
    if cfg.type == "registry":
        rows = _rows(cfg, folder)
        return Registry(
            cfg.name, rows, cfg.key or guess_key(rows), label=cfg.label, describe=cfg.describe, match=cfg.match,
            provides=cfg.provides, attrs=cfg.attrs, retriever=cfg.retriever,
            send_whole_if_under=cfg.send_whole_if_under, synonyms=cfg.synonyms, channel=cfg.channel,
            item=cfg.item, recency=cfg.recency, hierarchy=cfg.hierarchy, groups=cfg.groups,
        )  # fmt: skip
    if cfg.type == "files":
        paths = [str(p) for p in _rows(cfg, folder)]
        return FileIndex(cfg.name, paths, synonyms=cfg.synonyms, hierarchy=cfg.hierarchy or "dirname", k=cfg.k,
                         provides=cfg.provides or ("path", "file"), channel=cfg.channel,
                         item=cfg.item or "file")  # fmt: skip
    if cfg.function is None:
        raise ValueError(f'source {cfg.name!r} of type {cfg.type!r} needs `function = "module:function"`')
    target = load_object(cfg.function)
    if cfg.type == "provider":
        fn = target(**cfg.options) if cfg.options else target
        return Provider(fn, name=cfg.name, provides=cfg.provides, channel=cfg.channel, item=cfg.item or "item")
    source = target(**cfg.options)
    if not callable(getattr(source, "candidates", None)) or not getattr(source, "name", None):
        raise ValueError(f"source {cfg.name!r}: {cfg.function} did not return a source (name + candidates())")
    return source


def build_sources(
    specs: Sequence[SourceConfig | Mapping[str, Any]] | Mapping[str, Any],
    *,
    base_dir: str | os.PathLike[str] | None = None,
    descriptors: bool = False,
) -> list[Any]:
    """Build a list of sources; a mapping ``{name: spec}`` names each spec. With ``descriptors``, a spec without
    data becomes a ``{"name", "provides"}`` descriptor (catalog inference only) instead of an error."""
    items: list[SourceConfig | Mapping[str, Any]]
    if isinstance(specs, Mapping):
        items = [{**dict(spec), "name": name} if isinstance(spec, Mapping) else {"name": name, "rows": list(spec)}
                 for name, spec in specs.items()]  # fmt: skip
    else:
        items = list(specs)
    built: list[Any] = []
    for item in items:
        cfg = item if isinstance(item, SourceConfig) else SourceConfig.model_validate(dict(item))
        if descriptors and not cfg.has_data:
            built.append({"name": cfg.name, "provides": list(cfg.provides)})
        else:
            built.append(build_source(cfg, base_dir=base_dir))
    return built


def source_specs(data: Any) -> list[dict[str, Any]]:
    """Source entries of a loaded sources document: ``{"sources": [...]}``, a list of entries, or a mapping of
    ``{name: entry}`` tables (TOML ``[sources.<name>]``)."""
    if isinstance(data, Mapping) and "sources" in data:
        data = data["sources"]
    if isinstance(data, Mapping):
        data = [{"name": name, **dict(spec)} for name, spec in data.items()]
    if not isinstance(data, list) or not all(isinstance(s, Mapping) and "name" in s for s in data):
        raise ValueError("expected a list of source entries with a 'name'")
    return [dict(spec) for spec in data]


def request_sources(doc: Mapping[str, Any], templates: Sequence[SourceConfig] = ()) -> list[Any]:
    """Sources of one request (``extra_body.jevtools.sources``): ``{"contacts": [...rows]}`` reuses the configured
    source of that name with the request's rows; a full spec ``{"contacts": {"rows": [...], "key": …}}`` is built
    as given; rows of an unknown source get a registry keyed by the first of :data:`KEY_GUESSES`."""
    by_name = {t.name: t for t in templates}
    out = []
    for name, value in doc.items():
        if isinstance(value, Mapping):
            out.append(build_source({**dict(value), "name": name}))
            continue
        if not isinstance(value, list):
            raise ValueError(f"jevtools.sources.{name}: expected a list of rows or a source spec")
        template = by_name.get(name)
        if template is not None:
            spec = template.model_copy(update={"rows": list(value), "path": None})
        else:
            spec = SourceConfig(name=name, rows=list(value))
        out.append(build_source(spec))
    return out


__all__ = [
    "KEY_GUESSES",
    "SourceConfig",
    "SourceType",
    "build_source",
    "build_sources",
    "guess_key",
    "load_object",
    "request_sources",
    "source_specs",
]
