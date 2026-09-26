from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from jevtools.policy import Action, Outcome, Policy, Tier

APPENDIX_B = """
version = "jevtools-default-0.1"
hysteresis = 0.03
shadow = false

[tool]
min_p = 0.50
min_margin = 0.20
pair_cover = 0.85
record_hints = 0

[shapes]
out_of_pool = 0.30
ambiguous_cover = 0.90
ambiguous_k = 4
flag_band = [0.20, 0.80]
accept_min = 0.50
cosmetic_floor = 0.50
verify_min = 0.50
alt_show_min = 0.10

[tiers.read]
composition = "W"
execute = 0.60

[tiers.write]
composition = "PI"
execute = 0.70
confirm = 0.45
authorized = 0.80
content_accept = 0.70

[tiers.external]
composition = "PI"
execute = 0.80
confirm = 0.50
authorized = 0.90
content_accept = 0.80

[tiers.critical]
composition = "MIN_L_J"
execute = "never"
confirm = 0.80
authorized = 0.90
show_alternatives_min = 0.03
require_present = 0.80

[probes]
present = ["external", "critical"]
reverse = ["critical"]
verify = ["write", "external", "critical"]

[widen]
max_rounds = 2
page = 250
buckets = 2

[pools]
ref_k = 40
text_content_max = 4
text_cosmetic_max = 3
mentions_max = 8
items_max = 60
members_max = 40
joint_max = 24
verify_k = 3

[loop]
max_steps = 6
max_rounds = 12
max_cost_usd = 0.01
max_llm_calls = 2
done_after = 0.80

[budget]
max_tokens_per_call = 24000
max_state_tokens = 16000
max_questions_per_call = 250
chars_per_token = 3.5
"""


def test_tier_ordering() -> None:
    assert Tier.READ < Tier.WRITE < Tier.EXTERNAL < Tier.CRITICAL
    assert max([Tier.EXTERNAL, Tier.CRITICAL, Tier.READ]) is Tier.CRITICAL
    assert Tier.WRITE >= Tier.WRITE and Tier("external").rank == 2
    assert Tier.READ == "read" and {Tier.READ: 1}["read"] == 1


def test_outcomes_and_actions() -> None:
    assert [o.value for o in Outcome] == ["execute", "confirm", "clarify", "escalate", "abstain", "refuse", "done"]
    assert {a.value for a in Action} == {"widen", "fill", "resume"}


def test_appendix_b_toml_equals_defaults(tmp_path: Path) -> None:
    file = tmp_path / "policy.toml"
    file.write_text(APPENDIX_B, encoding="utf-8")
    assert Policy.from_toml(file) == Policy.default()
    assert Policy.from_toml_text(APPENDIX_B).sha256 == Policy.default().sha256


def test_partial_override_and_unknown_keys() -> None:
    policy = Policy.from_dict({"version": "tuned-1", "tiers": {"external": {"execute": 0.85}}})
    assert policy.tiers.external.execute == 0.85 and policy.tiers.external.confirm == 0.50
    assert policy.sha256 != Policy.default().sha256
    with pytest.raises(ValidationError):
        Policy.from_dict({"tiers": {"external": {"exeucte": 0.85}}})
    with pytest.raises(ValidationError):
        Policy.from_dict({"shapes": {"flag_band": [0.9, 0.1]}})


def test_threshold_helpers() -> None:
    policy = Policy.default()
    assert policy.execute_at("read") == 0.60 and policy.execute_at(Tier.CRITICAL) is None
    assert policy.tier("write").authorized == 0.80
    assert policy.alternatives_min("critical") == 0.03 and policy.alternatives_min("external") == 0.10
    opted_in = Policy.from_dict({"tiers": {"critical": {"auto_execute": 0.97}}})
    assert opted_in.critical_auto_execute() is None  # uncertified: refused (§11.4)
    certified = Policy.from_dict({"tiers": {"critical": {"auto_execute": 0.97}}, "certified": {"critical_cases": 3000}})
    assert certified.execute_at("critical") == 0.97
