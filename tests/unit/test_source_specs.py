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


# --------------------------------------------------------------------------------------------------------------------
# Per-request specs are untrusted data (review: proxy built code-loading and file-reading specs from request bodies)
# --------------------------------------------------------------------------------------------------------------------

MARKER_CALLS: list[dict[str, object]] = []


def record_call(**options: object) -> object:
    """A host factory a request must never be able to reach."""
    MARKER_CALLS.append(dict(options))
    return lambda query: []


@pytest.mark.parametrize(
    "spec",
    [
        {"type": "source", "function": "tests.unit.test_source_specs:record_call", "options": {"x": 1}},
        {"type": "provider", "function": "tests.unit.test_source_specs:record_call"},
        {"kind": "provider", "function": "tests.unit.test_source_specs:record_call", "options": {"x": 1}},
        {"function": "tests.unit.test_source_specs:record_call", "rows": ROWS},
        {"type": "provider", "rows": ROWS},
        {"type": "files", "path": "SECRET.env"},
        {"type": "registry", "path": "/etc/passwd"},
        {"rows_file": "contacts.json"},
        {"paths_file": "paths.txt"},
        {"rows": ROWS, "key": "email", "channel": "user"},
        {"key": "email"},  # no inline rows: nothing to build
    ],
)
def test_request_specs_are_data_only(spec: dict[str, object], tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from jevtools.sources.specs import request_sources

    monkeypatch.chdir(tmp_path)
    (tmp_path / "SECRET.env").write_text("API_KEY=sk-TOPSECRET\n", encoding="utf-8")
    (tmp_path / "contacts.json").write_text(json.dumps(ROWS), encoding="utf-8")
    (tmp_path / "paths.txt").write_text("a/b.txt\n", encoding="utf-8")
    MARKER_CALLS.clear()
    with pytest.raises(ValueError, match=r"jevtools\.sources\.x"):
        request_sources({"x": spec})
    assert MARKER_CALLS == []


def test_request_specs_keep_data_fields() -> None:
    from jevtools.sources.specs import request_sources

    people, files = request_sources({
        "people": {"rows": ROWS, "key": "email", "label": "{name} <{email}>", "match": ["name"],
                   "provides": ["person"], "describe": "{name}", "list_fields": [], "k": 10},
        "docs": {"type": "files", "rows": ["a/b.txt", "c/d.md"]},
    })  # fmt: skip
    assert isinstance(people, Registry) and people.provides == {"person"} and people.channel == "registry"
    assert isinstance(files, FileIndex)


def test_proxy_refuses_code_and_file_specs(tmp_path: Path) -> None:
    """End to end through the proxy: a ``function`` spec and a ``path`` spec answer 400 with no side effect."""
    pytest.importorskip("starlette")
    from starlette.testclient import TestClient

    from jevtools.backends.simulator import LexicalSimulator
    from jevtools.serve.app import create_app

    secret = tmp_path / "server_secret.env"
    secret.write_text("OPENROUTER_API_KEY=sk-or-v1-TOPSECRET\n", encoding="utf-8")
    c = TestClient(create_app({}, backend=LexicalSimulator()))
    MARKER_CALLS.clear()
    body = {"model": "jevtools", "messages": [{"role": "user", "content": "hi"}],
            "jevtools": {"sources": {"x": {"type": "source", "function": "tests.unit.test_source_specs:record_call",
                                           "options": {"args": ["touch"]}}}}}  # fmt: skip
    r = c.post("/v1/chat/completions", content=json.dumps(body), headers={"Content-Type": "text/plain"})
    assert r.status_code == 400 and r.json()["error"]["code"] == "jevtools_bad_sources"
    assert MARKER_CALLS == []
    tools = [{"type": "function", "function": {"name": "read_file", "description": "Read a file.", "parameters": {
        "type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}}]  # fmt: skip
    body = {"model": "jevtools", "messages": [{"role": "user", "content": "read the file OPENROUTER_API_KEY"}],
            "tools": tools, "jevtools": {"sources": {"files": {"type": "files", "path": str(secret)}}}}  # fmt: skip
    r = c.post("/v1/chat/completions", json=body)
    assert r.status_code == 400 and r.json()["error"]["code"] == "jevtools_bad_sources"
    assert "TOPSECRET" not in r.text
