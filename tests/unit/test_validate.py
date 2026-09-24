"""Each pre-send rule of §3.5.6 raises ``BallotError`` with its rule id."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from jevtools.ballot import Ballot, BallotOption, BallotQuestion, SentinelSpec
from jevtools.candidates import NONE_OF_THESE, NOT_STATED
from jevtools.errors import BallotError
from jevtools.spec.catalog import Catalog
from jevtools.validate import Limits, preflight

SENTINELS = {
    NOT_STATED: SentinelSpec(decodes_to="missing", text="The user does not say."),
    NONE_OF_THESE: SentinelSpec(decodes_to="uncovered", text="none"),
}


def _choice(qid: str = "get_weather.unit", labels: list[str] | None = None, **kw: Any) -> BallotQuestion:
    options = [BallotOption(label=label, value=label) for label in (labels if labels is not None else ["celsius"])]
    fields: dict[str, Any] = {
        "family": "slot",
        "tool": "get_weather",
        "path": ("unit",),
        "primitive": "choice",
        "instructions": "Which option is the unit?",
        "options": options,
        "sentinels": SENTINELS,
    }
    fields.update(kw)
    return BallotQuestion(qid=qid, **fields)


def _ballot(*questions: BallotQuestion, **kw: Any) -> Ballot:
    return Ballot(catalog_sha256="sha256:c", context_sha256="sha256:x", policy_version="p", state={"request": "hi"},
                  questions=list(questions), **kw)  # fmt: skip


def _rule(ballot: Ballot, limits: Limits | None = None, **kw: Any) -> str:
    with pytest.raises(BallotError) as info:
        preflight(ballot, limits, **kw)
    return info.value.rule


def test_valid_ballot_passes() -> None:
    preflight(_ballot(_choice()))


def test_option_count() -> None:
    assert _rule(_ballot(_choice(sentinels={}))) == "choice.options"  # 1 option
    many = [f"v{i}" for i in range(253)]
    assert _rule(_ballot(_choice(labels=many))) == "choice.options"


def test_label_rules() -> None:
    assert _rule(_ballot(_choice(labels=["two\nlines"]))) == "label.grammar"
    assert _rule(_ballot(_choice(labels=["x" * 65]))) == "label.grammar"
    assert _rule(_ballot(_choice(labels=["not_stated"]))) == "label.reserved"
    assert _rule(_ballot(_choice(labels=["Zürich", "zürich"]))) == "label.unique"
    assert _rule(_ballot(_choice(labels=["x" * 40])), Limits(label_max=32)) == "label.grammar"


def test_length_rules() -> None:
    long_desc = [BallotOption(label="a", value="a", text="d" * 401)]
    assert _rule(_ballot(_choice(options=long_desc))) == "description.length"
    long_sentinel = {**SENTINELS, NONE_OF_THESE: SentinelSpec(decodes_to="uncovered", text="s" * 401)}
    assert _rule(_ballot(_choice(sentinels=long_sentinel))) == "description.length"
    assert _rule(_ballot(_choice(instructions="i" * 2001))) == "instructions.length"
    accept = BallotQuestion(qid="send_email.body.accept.0", family="accept", primitive="noul",
                            instructions={"question": "q", "candidate": "c" * 4001})  # fmt: skip
    assert _rule(_ballot(accept)) == "accept.candidate"
    preflight(_ballot(accept.model_copy(update={"instructions": {"question": "q", "candidate": "c" * 3000}})))


def test_qid_rules() -> None:
    assert _rule(_ballot(_choice(qid="Get_Weather.unit"))) == "qid.grammar"
    assert _rule(_ballot(_choice(qid="get_weather." + "u" * 120)), Limits(qid_max=128)) == "qid.grammar"
    duplicate = _ballot(_choice()).model_copy(update={"questions": [_choice(), _choice()]})
    assert _rule(duplicate) == "qid.unique"


def test_per_call_budget_rules() -> None:
    nouls = [BallotQuestion(qid=f"t.p.item.{i}", family="item", primitive="noul", instructions="Include?")
             for i in range(251)]  # fmt: skip
    assert _rule(_ballot(*nouls)) == "call.questions"
    preflight(_ballot(*nouls, calls=[[q.qid for q in nouls[:125]], [q.qid for q in nouls[125:]]]))
    big = [_choice(qid=f"t.p{i}", instructions="x" * 1900) for i in range(50)]
    assert _rule(_ballot(*big)) == "call.tokens"
    preflight(_ballot(*big), Limits(max_tokens=100_000))


def test_value_schema_rule(scenario_catalog: Catalog) -> None:
    ballot = _ballot(_choice(labels=["kelvin"]))
    preflight(ballot)
    assert _rule(ballot, catalog=scenario_catalog) == "value.schema"
    late = BallotOption(label="kelvin", value="kelvin", late={"derive": "x"})
    preflight(_ballot(_choice(options=[late])), catalog=scenario_catalog)


def test_limits_file_and_estimate(tmp_path: Path) -> None:
    file = tmp_path / "limits.json"
    file.write_text('{"label_max": 128, "id_mode": "opaque", "measured_at": "2026-09-24"}')
    limits = Limits.from_file(file)
    assert limits.label_max == 128 and limits.id_mode == "opaque"
    assert Limits().estimate_tokens({"model": "", "state": "x" * 34}) == 17  # 57 chars / 3.5
    assert Limits(token_ratio=1.6).estimate_tokens({"model": "", "state": "x" * 34}) == 27
