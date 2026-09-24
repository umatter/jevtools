"""Claims in the README and the other docs that code can check: install commands, the confidence headline, the probe's
request count and the live test suite. Scripted and simulated answers only; nothing here is evidence about Jev."""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Literal

import pytest

import jevtools as jt
from jevtools.backends.simulator import LexicalSimulator
from jevtools.confidence import IsotonicCalibrator
from jevtools.probe import run_probe

ROOT = Path(__file__).resolve().parents[2]
README = ROOT / "README.md"
MARKDOWN = sorted(
    [README, *(ROOT / "docs").rglob("*.md"), *(ROOT / "examples").rglob("*.md"), *(ROOT / "tests").rglob("*.md")]
)

INSTALL = re.compile(r"\b(?:pip3? install|uv pip install|uv add)\b(?P<args>[^#`\n]*)")
"""An install command and its arguments, up to a shell comment or the end of an inline code span."""
REQUIREMENT = re.compile(r"(?<![\w./-])jevtools(?:\[[^\]]*\])?(?P<url>\s*@\s*\S+)?")
"""A requirement on the ``jevtools`` distribution (not a path or a URL that ends in ``/jevtools``)."""


def readme() -> str:
    return README.read_text(encoding="utf-8")


def intro() -> str:
    """The README text before "How it works", whitespace-normalized."""
    return " ".join(readme().split("## How it works", 1)[0].split())


# --------------------------------------------------------------------------------------------------------------------
# install commands
# --------------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("path", MARKDOWN, ids=lambda p: str(p.relative_to(ROOT)))
def test_install_commands_name_the_git_source(path: Path) -> None:
    """jevtools is not on PyPI (README "Quickstart"), and the name is unclaimed there: a bare ``pip install
    'jevtools[serve]'`` fails today and could install someone else's package tomorrow."""
    bare = [
        f"{path.relative_to(ROOT)}:{number}: {line.strip()}"
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1)
        for command in INSTALL.finditer(line)
        for requirement in REQUIREMENT.finditer(command["args"])
        if requirement["url"] is None
    ]
    assert bare == [], 'install from git: pip install "jevtools[...] @ git+https://github.com/umatter/jevtools"'


def test_the_install_pattern_catches_a_bare_name() -> None:
    lines = ["pip install 'jevtools[serve]'", "uv add jevtools", 'pip install "jevtools @ git+https://x/jevtools"',
             'uv pip install -e ".[serve]"', "pip install openai"]  # fmt: skip
    flagged = [line for line in lines for c in INSTALL.finditer(line) for r in REQUIREMENT.finditer(c["args"])
               if r["url"] is None]  # fmt: skip
    assert flagged == lines[:2]


# --------------------------------------------------------------------------------------------------------------------
# the confidence headline
# --------------------------------------------------------------------------------------------------------------------


def _quickstart(calibrators: dict[str, IsotonicCalibrator] | None = None) -> jt.Decision:
    """The README quickstart (ScriptedBackend variant), optionally with fitted calibrators."""

    @jt.tool
    def get_weather(city: str, unit: Literal["celsius", "fahrenheit"] = "celsius") -> dict[str, object]:
        """Get the current weather for a city."""
        return {"city": city, "unit": unit, "temp": 61}

    backend = jt.backends.ScriptedBackend({
        "tool": {"get_weather": 0.97, "NO_TOOL": 0.02, "UNSUPPORTED": 0.01},
        "get_weather.city": {"Zurich": 0.95, "NOT_STATED": 0.03, "NONE_OF_THESE": 0.02},
        "get_weather.unit": "fahrenheit",
    })  # fmt: skip
    router = jt.Router([get_weather], backend=backend, calibrators=calibrators)
    return router.decide("What's the weather like in Zurich in Fahrenheit?")


def test_readme_calls_c_calibrated_only_after_a_calibrator_is_fitted() -> None:
    """C is W, Π or min(L, J) of the factors (§3.7.3) until a calibrator is fitted; the headline must not promise a
    calibrated call confidence by default."""
    default = _quickstart()
    assert default.confidence is not None and default.confidence.calibrated is False
    fitted = _quickstart(calibrators={"read": IsotonicCalibrator().fit([0.2, 0.9], [False, True])})
    assert fitted.confidence is not None and fitted.confidence.calibrated is True
    text = intro()
    assert "calibrated call confidence" not in text
    assert "jevtools tune --calibrate" in text and "confidence.calibrated == False" in text


# --------------------------------------------------------------------------------------------------------------------
# the probe's request count
# --------------------------------------------------------------------------------------------------------------------


def test_readme_probe_row_counts_the_requests_the_probe_sends() -> None:
    """``jevtools probe`` sends 19 requests to a permissive backend, 16 with ``--no-smoke`` (docs/DECISIONS.md,
    "Conformance probe"); the CLI table must say so, not the spec's "about 12"."""
    row = next(line for line in readme().splitlines() if line.startswith("| `jevtools probe"))
    total = re.search(r"(\d+) live requests", row)
    no_smoke = re.search(r"(\d+) with `--no-smoke`", row)
    assert total is not None and no_smoke is not None, row
    assert int(total[1]) == run_probe(LexicalSimulator(), write=False).calls
    assert int(no_smoke[1]) == run_probe(LexicalSimulator(), write=False, smoke=False).calls
    assert "400 questions" in row and "8,000-character" in row


# --------------------------------------------------------------------------------------------------------------------
# the live suite
# --------------------------------------------------------------------------------------------------------------------


def test_live_marker_selects_a_live_suite_per_backend() -> None:
    """README "Development" and SPEC §10.7 promise ``-m live`` tests; with a key, ``pytest -m live`` must select
    some (it used to deselect everything and exit 5). Collection only: nothing is sent."""
    env = {k: v for k, v in os.environ.items() if k not in ("TYPESAFE_API_KEY", "OPENROUTER_API_KEY")}
    env["OPENROUTER_API_KEY"] = "collect-only-never-sent"
    done = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q", "-m", "live", "-p", "no:cacheprovider", "tests"],
        cwd=ROOT, env=env, capture_output=True, text=True, timeout=300,
    )  # fmt: skip
    assert done.returncode == 0, done.stdout[-2000:] + done.stderr[-2000:]
    collected = [line for line in done.stdout.splitlines() if line.startswith("tests/")]
    assert collected and all(line.startswith("tests/live/") for line in collected)
    for backend in ("typesafe", "openrouter_systemone", "openrouter_decisions"):
        assert any(f"test_conformance_probe[{backend}]" in line for line in collected), backend
    assert "tests/live" in readme() and "tests/live" in (ROOT / "docs" / "ARCHITECTURE.md").read_text(encoding="utf-8")
