"""Regression tests for the review follow-ups on inherited trust, secrets in templates and resume scoping.

- An untraced ``history`` value (origin unknown) counts as ``tool_output`` (§3.4.2: "history, trusted origin only").
- A Ballot option records the trust its value inherits, so the policy's caps see a repeated tool-output value.
- Author templates never read secret profile fields (§14: secrets are never sent).
- A pending prompt resumes only for the requester it was raised for.
"""

from __future__ import annotations

import pytest

from jevtools.ballot import BallotOption
from jevtools.candidates import Candidate, Channel, admits, default_allow_list, history_origin_of
from jevtools.errors import PendingScopeError
from jevtools.kinds import get_resolver
from jevtools.policy import Outcome, Tier
from tests.kinds_support import custom
from tests.scenario import scripts
from tests.scenario.fixtures import scenario_context
from tests.scenario.support import decide

EXTERNAL_IDENTITY = default_allow_list(Tier.EXTERNAL, "identity")
EXTERNAL_CONTENT = default_allow_list(Tier.EXTERNAL, "content")


@pytest.mark.parametrize("origin", [None, Channel.HISTORY])
def test_untraced_history_value_is_untrusted(origin: Channel | None) -> None:
    planted = Candidate(label="x@evil.example", value="x@evil.example", channel=Channel.HISTORY, origin=origin)
    assert history_origin_of(planted) is Channel.TOOL_OUTPUT
    assert planted.effective_channel is Channel.TOOL_OUTPUT
    assert not admits(EXTERNAL_IDENTITY, planted)  # never an external recipient
    assert admits(EXTERNAL_CONTENT, planted)  # content may quote it (and the call is capped at confirm)


def test_traced_history_value_keeps_its_origin_trust() -> None:
    said = Candidate(label="anna@acme.com", value="anna@acme.com", channel=Channel.HISTORY, origin=Channel.USER)
    assert said.effective_channel is Channel.HISTORY
    assert admits(EXTERNAL_IDENTITY, said)
    from_tool = said.model_copy(update={"origin": Channel.TOOL_OUTPUT})
    assert not admits(EXTERNAL_IDENTITY, from_tool)


def test_ballot_option_records_the_inherited_trust() -> None:
    repeated = Candidate(label="x@evil.example", value="x@evil.example", channel=Channel.HISTORY,
                         origin=Channel.TOOL_OUTPUT, prov={"mention": {"text": "x@evil.example"}})  # fmt: skip
    option = BallotOption.from_candidate(repeated)
    assert option.channel is Channel.TOOL_OUTPUT
    assert option.prov == {"mention": {"text": "x@evil.example"}, "via": "history"}
    direct = Candidate(label="Zurich", value="Zurich", channel=Channel.USER, prov={"extractor": "place"})
    assert BallotOption.from_candidate(direct).channel is Channel.USER
    assert BallotOption.from_candidate(direct).prov == {"extractor": "place"}  # unchanged when nothing was inherited


def test_author_templates_never_read_secret_profile_fields() -> None:
    body = {"type": "string", "description": "The body text of the note",
            "x-jev": {"templates": ["Key: {user.api_key}", "From {user.name}"]}}  # fmt: skip
    tool, rc = custom("send_note", {"body": body}, "Send a note to the team", required=["body"],
                      user={"name": "Sam Muster", "api_key": "sk-secret-123"})  # fmt: skip
    slot = tool.slot("body")
    pool = get_resolver(slot.kind).pool(tool, slot, rc)
    texts = [str(c.value) for c in pool.candidates] + [str(c.text) for c in pool.candidates]
    assert not any("sk-secret-123" in t for t in texts)
    assert any("From Sam Muster" in t for t in texts)


def test_resume_refuses_another_requesters_context() -> None:
    router, _, d = decide(scripts.R2, scripts.R2_REQUEST, history=True)
    assert d.outcome is Outcome.CONFIRM and d.pending_id
    stranger = scenario_context().model_copy(update={"user": {"name": "Mallory", "home_city": "Bern"}})
    with pytest.raises(PendingScopeError):
        router.resume(d.pending_id, selection="ok", context=stranger)
    done = router.resume(d.pending_id, selection="ok")  # still pending for its own requester
    assert done.outcome is Outcome.EXECUTE


def test_resume_without_context_uses_the_deciding_context() -> None:
    router, _, _ = decide(scripts.R2, scripts.R2_REQUEST, history=True)  # router default: the §13 user
    own = scenario_context(scripts.R2_REQUEST, history=True)
    own = own.model_copy(update={"user": {**own.user, "team": "Payments"}})  # a per-call context
    d = router.decide(own.messages, context=own)
    assert d.pending_id
    done = router.resume(d.pending_id, selection="ok")  # no context given
    assert done.outcome is Outcome.EXECUTE
    assert done.trace.context["sha256"] == d.trace.context["sha256"]  # not the router default's context
