"""``jt.backends.auto`` (spec §8.3, §10.4): environment precedence, ``cassette:<path>``, ``JEVTOOLS_MODEL``, the offline
simulator only on request (with a warning), and ``BackendConfigError`` otherwise."""

from __future__ import annotations

import warnings
from pathlib import Path

import pytest

from jevtools.backends.auto import auto
from jevtools.backends.cassette import Cassette
from jevtools.backends.errors import BackendConfigError
from jevtools.backends.http import OPENROUTER_DECISIONS_URL, OPENROUTER_SYSTEMONE_URL, HTTPBackend
from jevtools.backends.simulator import LexicalSimulator

VARIABLES = ("JEVTOOLS_BACKEND", "JEVTOOLS_MODEL", "JEVTOOLS_CASSETTE_MODE", "TYPESAFE_API_KEY", "OPENROUTER_API_KEY",
             "TYPESAFE_BASE_URL")  # fmt: skip


@pytest.fixture(autouse=True)
def clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in VARIABLES:
        monkeypatch.delenv(name, raising=False)


def test_typesafe_key_wins_over_openrouter(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TYPESAFE_API_KEY", "ts-key")
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-key")
    backend = auto()
    assert isinstance(backend, HTTPBackend) and backend.name == "typesafe" and backend.model == "jev-latest"
    assert backend.url == "https://api.typesafe.ai/v1/systemone"


def test_openrouter_key_gives_decisions(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-key")
    backend = auto()
    assert isinstance(backend, HTTPBackend) and backend.name == "openrouter_decisions"
    assert (backend.url, backend.model) == (OPENROUTER_DECISIONS_URL, "~typesafe/jev-latest")


def test_jevtools_backend_overrides_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TYPESAFE_API_KEY", "ts-key")
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-key")
    monkeypatch.setenv("JEVTOOLS_BACKEND", "openrouter_systemone")
    backend = auto()
    assert isinstance(backend, HTTPBackend) and backend.url == OPENROUTER_SYSTEMONE_URL
    monkeypatch.setenv("JEVTOOLS_BACKEND", "simulator")
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # an explicitly chosen simulator does not warn
        assert isinstance(auto(), LexicalSimulator)
    monkeypatch.setenv("JEVTOOLS_BACKEND", " TypeSafe ")
    assert auto().name == "typesafe"


def test_jevtools_model_overrides_the_model(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("JEVTOOLS_MODEL", "typesafe/jev-1.13")
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-key")
    assert auto().model == "typesafe/jev-1.13"
    monkeypatch.setenv("JEVTOOLS_BACKEND", "simulator")
    assert auto().model == "typesafe/jev-1.13"


def test_offline_only_when_allowed_and_with_a_warning() -> None:
    with pytest.raises(BackendConfigError, match="no Jev backend configured"):
        auto()
    with pytest.warns(UserWarning, match="LexicalSimulator"):
        backend = auto(allow_offline=True)
    assert isinstance(backend, LexicalSimulator) and backend.model == "lexical-simulator"


def test_named_backend_without_its_key_is_a_config_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("JEVTOOLS_BACKEND", "openrouter_decisions")
    with pytest.raises(BackendConfigError, match="OPENROUTER_API_KEY"):
        auto(allow_offline=True)
    monkeypatch.setenv("JEVTOOLS_BACKEND", "gpt-4")
    with pytest.raises(BackendConfigError, match="unknown JEVTOOLS_BACKEND"):
        auto(allow_offline=True)


def test_cassette_replays_by_default(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    path = tmp_path / "ci.jsonl"
    monkeypatch.setenv("JEVTOOLS_BACKEND", f"cassette:{path}")
    backend = auto()
    assert isinstance(backend, Cassette) and backend.mode == "replay" and backend.path == path
    monkeypatch.setenv("JEVTOOLS_BACKEND", "cassette:")
    with pytest.raises(BackendConfigError, match="needs a path"):
        auto()


def test_cassette_record_wraps_the_live_backend(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("JEVTOOLS_BACKEND", f"cassette:{tmp_path / 'rec.jsonl'}")
    monkeypatch.setenv("JEVTOOLS_CASSETTE_MODE", "record")
    with pytest.raises(BackendConfigError, match="needs TYPESAFE_API_KEY or OPENROUTER_API_KEY"):
        auto()
    monkeypatch.setenv("TYPESAFE_API_KEY", "ts-key")
    backend = auto()
    assert isinstance(backend, Cassette) and backend.mode == "record"
    assert isinstance(backend.inner, HTTPBackend) and backend.inner.name == "typesafe" and backend.name == "typesafe"
    monkeypatch.setenv("JEVTOOLS_CASSETTE_MODE", "sideways")
    with pytest.raises(BackendConfigError, match="JEVTOOLS_CASSETTE_MODE"):
        auto()


def test_explicit_env_mapping_and_http_kwargs() -> None:
    backend = auto(env={"OPENROUTER_API_KEY": "k", "JEVTOOLS_BACKEND": "openrouter_systemone"}, timeout=5.0)
    assert isinstance(backend, HTTPBackend) and backend.timeout == 5.0 and backend.url == OPENROUTER_SYSTEMONE_URL
