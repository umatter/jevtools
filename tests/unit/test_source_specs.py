"""Source specs as data (``jevtools.sources.specs``): one format for ``jevtools.toml``, evaluation context documents
and ``jevtools lint --sources`` files."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from jevtools.sources import FileIndex, Registry, SourceConfig, build_sources
from jevtools.sources.specs import source_specs

ROWS = [{"email": "anna@example.com", "name": "Anna Keller"}, {"email": "bob@example.com", "name": "Bob Meier"}]


def test_cli_spellings_normalize_to_the_proxy_format(tmp_path: Path) -> None:
    (tmp_path / "contacts.json").write_text(json.dumps(ROWS), encoding="utf-8")
    (tmp_path / "paths.txt").write_text("a/b.txt\nc/d.md\n", encoding="utf-8")
    cfg = SourceConfig.model_validate({"name": "files", "kind": "files", "paths_file": "paths.txt"})
    assert (cfg.type, cfg.path, cfg.has_data) == ("files", "paths.txt", True)
    contacts, files, inline = build_sources(
        [
            {"name": "contacts", "kind": "registry", "rows_file": "contacts.json", "key": "email"},
            {"name": "files", "kind": "files", "paths_file": "paths.txt"},
            {"name": "inline", "paths": ["x/y.txt"]},  # `paths` alone means a file index
        ],
        base_dir=tmp_path,
    )
    assert isinstance(contacts, Registry) and len(contacts.rows) == 2
    assert isinstance(files, FileIndex) and isinstance(inline, FileIndex)


def test_descriptors_only_when_asked() -> None:
    specs = [{"name": "accounts", "kind": "registry", "key": "id", "provides": ["account_id"]}]
    assert build_sources(specs, descriptors=True) == [{"name": "accounts", "provides": ["account_id"]}]
    with pytest.raises(ValueError, match="give `rows` or `path`"):
        build_sources(specs)


def test_source_documents() -> None:
    assert source_specs({"sources": [{"name": "a"}]}) == [{"name": "a"}]
    assert source_specs({"a": {"key": "id"}}) == [{"name": "a", "key": "id"}]
    with pytest.raises(ValueError, match="'name'"):
        source_specs([{"key": "id"}])
