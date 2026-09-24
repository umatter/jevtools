"""The loop tests drive the demo world the examples ship (:mod:`jevtools.demo`), not a private copy of it.

``tests/loop/support.py`` used to redefine the R6 invoice, the injected address, the fake ``Workspace`` and the R6
scripts; the copies drifted (``get_weather`` in Fahrenheit, ``member_answers`` on non-object instructions), so a change
to the demo scenario would no longer have been covered by the loop and injection tests.
"""

from __future__ import annotations

import pytest

import tests.loop.support as support
from jevtools.demo import scenario, scripts

FROM_SCENARIO = ("FINANCE", "INJECTED_ADDRESS", "INJECTED_IBAN", "INJECTION", "INVOICE_TEXT", "INV_2291",
                 "R6_REQUEST", "Workspace")  # fmt: skip
FROM_SCRIPTS = ("R6_MEMBERS", "R6_STEP2", "member_answers", "observations_of", "r6_script", "r6_step1")


@pytest.mark.parametrize("name", FROM_SCENARIO)
def test_scenario_pieces_are_the_demo_ones(name: str) -> None:
    assert getattr(support, name) is getattr(scenario, name)


@pytest.mark.parametrize("name", FROM_SCRIPTS)
def test_script_pieces_are_the_demo_ones(name: str) -> None:
    assert getattr(support, name) is getattr(scripts, name)


def test_the_demo_modules_agree_on_r6() -> None:
    assert scripts.R6_REQUEST == scenario.R6_REQUEST  # scripts.py still defines its own copy of the string


def test_support_exports_what_it_defines() -> None:
    assert sorted(support.__all__) == sorted([*FROM_SCENARIO, *FROM_SCRIPTS, "r6_agent", "r6_messages"])
    assert all(hasattr(support, name) for name in support.__all__)
