from __future__ import annotations

import unicodedata

import pytest

from jevtools.candidates import (
    NONE_OF_THESE,
    NOT_STATED,
    SENTINEL_ORDER,
    Bottom,
    Candidate,
    Channel,
    Pool,
    admits,
    apply_allow_list,
    assign_labels,
    canonical_order,
    default_allow_list,
    display_value,
    elide_path,
    is_reserved,
    is_valid_label,
    least_trusted,
    make_label,
    reverse_order,
    value_key,
)
from jevtools.policy import Tier

U, R, A, H, T, G = (Channel.USER, Channel.REGISTRY, Channel.AUTHOR, Channel.HISTORY, Channel.TOOL_OUTPUT,
                    Channel.GENERATED)  # fmt: skip


def test_trust_order_and_least_trusted() -> None:
    assert U.trust == R.trust == A.trust < H.trust < T.trust < G.trust
    assert least_trusted(U, T, H) is T
    assert least_trusted("registry", "user") is R  # first wins on ties
    with pytest.raises(ValueError):
        least_trusted()


@pytest.mark.parametrize(
    ("tier", "stakes", "quantity", "expected"),
    [
        (Tier.READ, "identity", False, {U, R, A, H, T}),
        (Tier.READ, "content", False, {U, R, A, H, T, G}),
        (Tier.WRITE, "identity", False, {U, R, A, H}),
        (Tier.WRITE, "content", False, {U, R, A, H, T, G}),
        (Tier.EXTERNAL, "identity", False, {U, R, A, H}),
        (Tier.EXTERNAL, "content", False, {U, R, A, H, T, G}),
        (Tier.EXTERNAL, "cosmetic", False, {U, R, A, H, T, G}),
        (Tier.CRITICAL, "identity", False, {U, R, A}),
        (Tier.CRITICAL, "identity", True, {U, R}),
        (Tier.CRITICAL, "content", False, {U, R, A}),
        (Tier.CRITICAL, "cosmetic", False, {U, R, A}),
    ],
)
def test_default_allow_lists(tier: Tier, stakes: str, quantity: bool, expected: set[Channel]) -> None:
    assert set(default_allow_list(tier, stakes, quantity=quantity)) == expected


def test_history_inherits_origin_trust() -> None:
    allow = default_allow_list(Tier.EXTERNAL, "identity")
    trusted = Candidate(value="a@x.io", channel=H, origin=U)
    injected = Candidate(value="evil@x.io", channel=H, origin=T)
    assert admits(allow, trusted)
    assert not admits(allow, injected)
    assert injected.effective_channel is T
    kept, blocked = apply_allow_list([trusted, injected, Candidate(value=1, channel=T)], allow)
    assert kept == [trusted] and len(blocked) == 2


def test_pool_channel_blocked_and_evidence() -> None:
    amount = Candidate(value="5000.00", channel=T)
    pool = Pool(tool="transfer_funds", path=("amount",), kind="money", blocked=[amount])
    assert pool.empty and pool.channel_blocked
    assert Candidate(value="x", channel=R, prov={"anchor": {"text": "Anna"}}).is_evidence
    assert not Candidate(value="x", channel=R).is_evidence
    assert not Candidate(value="celsius", channel=A).is_evidence


def test_value_key_and_display() -> None:
    assert value_key(Bottom.MISSING) == "⊥missing"
    assert value_key("⊥missing") == '"⊥missing"'
    assert value_key(45) == value_key(45.0)
    assert [display_value(v) for v in ("Zurich", 45, 2.5, True, None, [1])] == [
        "Zurich",
        "45",
        "2.5",
        "true",
        "null",
        "[1]",
    ]


def test_label_grammar() -> None:
    assert is_valid_label("Tue 2026-09-29 15:00 (Europe/Zurich)")
    assert not is_valid_label("")
    assert not is_valid_label(" padded")
    assert not is_valid_label("two\nlines")
    assert not is_valid_label("tab\there")
    assert not is_valid_label("x" * 65)
    assert is_valid_label("x" * 64)
    assert is_reserved("not_stated") and is_reserved("NONE_OF_THESE") and not is_reserved("NOT STATED")


def test_make_label_wysiwyg_and_collisions() -> None:
    assert make_label("Anna Keller <anna.keller@acme.com>", slot="to", n=1) == "Anna Keller <anna.keller@acme.com>"
    assert make_label("NOT_STATED", slot="x", n=1) == "NOT_STATED (value)"
    assert make_label("zürich", slot="city", n=2, taken=[unicodedata.normalize("NFD", "Zürich")]) == "zürich (2)"
    long_text = "a sentence that is much too long to be a label " * 3
    assert make_label(long_text, slot="body", n=3) == "body_3"
    assert make_label(long_text, slot="body", n=3, taken=["body_3"]) == "body_4"


def test_path_elision() -> None:
    path = "services/payments/very/deep/nested/directory/structure/config/settings.yaml"
    elided = elide_path(path, 40)
    assert elided == "services/…/config/settings.yaml"
    assert elided is not None and len(elided) <= 40
    assert make_label(path, slot="path", n=1, label_max=40, is_path=True) == elided
    assert make_label(path, slot="path", n=1, label_max=40, is_path=True, taken=[elided]) == "path_1"
    assert elide_path("a/b", 2) is None


def test_assign_labels_keeps_descriptions_self_contained() -> None:
    long_value = "A long template body that cannot possibly fit in a sixty-four character label at all."
    cands = [Candidate(value="x", label="Anna", channel=R), Candidate(value="y", label="anna", channel=R),
             Candidate(value=long_value, channel=A, text="Author template.")]  # fmt: skip
    labelled = assign_labels(cands, slot="body")
    assert [c.label for c in labelled] == ["Anna", "anna (2)", "body_3"]
    assert labelled[1].text is None
    assert labelled[2].text is not None and labelled[2].text.startswith('Author template. Full value: "A long')
    assert len(labelled[2].text) <= 400


def test_canonical_and_reverse_order() -> None:
    labels = ["NONE_OF_THESE", "b", "NOT_STATED", "B", "a", "Ä", "EXCLUDE"]
    ordered = canonical_order(labels)
    assert ordered == ["a", "B", "b", "Ä", "NOT_STATED", "NONE_OF_THESE", "EXCLUDE"]
    assert reverse_order(labels) == ["Ä", "b", "B", "a", "NOT_STATED", "NONE_OF_THESE", "EXCLUDE"]
    tools = canonical_order(["send_email", "UNSUPPORTED", "create_event", "NO_TOOL", "DONE"])
    assert tools == ["create_event", "send_email", "NO_TOOL", "UNSUPPORTED", "DONE"]
    assert SENTINEL_ORDER[:2] == (NOT_STATED, NONE_OF_THESE)
    objects = canonical_order([Candidate(label="z", value=1, channel=U), Candidate(label="y", value=2, channel=U)])
    assert [c.label for c in objects] == ["y", "z"]
