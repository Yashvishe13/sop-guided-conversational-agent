"""Security tests for the deterministic identity gate (VERIFY_ID)."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, date, datetime

import pytest

from insurance_claims.claims.fixtures import load_fixtures
from insurance_claims.claims.repository import PolicyholderDirectory
from insurance_claims.claims.verification import (
    VerificationOutcome,
    apply_identity_proposals,
)
from insurance_claims.config import APP_ROOT
from insurance_claims.domain.models import CallerRole, Policyholder
from insurance_claims.domain.state import VerificationState

FIXTURES_DIR = APP_ROOT / "fixtures"
NOW = datetime(2026, 10, 3, 12, 0, tzinfo=UTC)
MAX_FAILURES = 3

MARGARET = {
    "full_name": "Margaret Chen",
    "dob": "1985-03-15",
    "id_last4": "4472",
    "phone": "+16505212836",
    "email": "margaret@email.com",
    "policy_number": "POL-9921",
}
AVA = {
    "full_name": "Ava Lopez",
    "dob": "1990-08-21",
    "id_last4": "9180",
    "phone": "+16503882920",
    "email": "ava.lopez@email.com",
    "policy_number": "POL-1044",
}


@pytest.fixture(scope="module")
def directory() -> PolicyholderDirectory:
    return PolicyholderDirectory(load_fixtures(FIXTURES_DIR).policyholders)


def apply(
    directory: PolicyholderDirectory,
    vstate: VerificationState,
    proposals: Mapping[str, str],
    *,
    max_failures: int = MAX_FAILURES,
) -> tuple[VerificationState, VerificationOutcome]:
    return apply_identity_proposals(vstate, proposals, directory, max_failures=max_failures, now=NOW)


def pick(source: Mapping[str, str], *fields: str) -> dict[str, str]:
    return {field: source[field] for field in fields}


# ---------------------------------------------------------------------------
# Happy path and the demo utterance
# ---------------------------------------------------------------------------


def test_demo_name_dob_last4_verifies_p9(directory: PolicyholderDirectory) -> None:
    state, outcome = apply(directory, VerificationState(), pick(MARGARET, "full_name", "dob", "id_last4", "policy_number"))
    assert outcome.status == "verified"
    assert outcome.party_id == "P9"
    assert outcome.provided_pii_count == 3
    assert set(outcome.newly_provided) == {"full_name", "dob", "id_last4", "policy_number"}
    assert not outcome.attempt_counted
    assert state.status == "verified"
    assert state.party_id == "P9"
    assert state.verified_at == NOW
    assert state.failed_attempts == 0


def test_policy_number_never_counts_toward_three(directory: PolicyholderDirectory) -> None:
    state, outcome = apply(directory, VerificationState(), pick(MARGARET, "policy_number", "full_name", "dob"))
    assert outcome.status == "pending"
    assert outcome.party_id is None
    assert outcome.provided_pii_count == 2
    assert state.status == "unverified"
    assert state.party_id is None


def test_partial_identity_accumulates_across_calls(directory: PolicyholderDirectory) -> None:
    state = VerificationState()
    state, first = apply(directory, state, pick(MARGARET, "full_name"))
    assert (first.status, first.provided_pii_count) == ("pending", 1)
    state, second = apply(directory, state, {})
    assert (second.status, second.provided_pii_count, second.newly_provided) == ("pending", 1, ())
    state, third = apply(directory, state, {"dob": "March 15, 1985"})
    assert (third.status, third.provided_pii_count) == ("pending", 2)
    state, fourth = apply(directory, state, {"phone": "(650) 521 2836"})
    assert fourth.status == "verified" and fourth.party_id == "P9"
    assert fourth.newly_provided == ("phone",)


@pytest.mark.parametrize(
    "fields",
    [
        ("full_name", "dob", "phone"),
        ("full_name", "email", "id_last4"),
        ("dob", "phone", "email"),
        ("phone", "email", "id_last4"),
    ],
)
def test_any_three_distinct_pii_fields_verify(directory: PolicyholderDirectory, fields: tuple[str, ...]) -> None:
    _, outcome = apply(directory, VerificationState(), pick(MARGARET, *fields))
    assert outcome.status == "verified" and outcome.party_id == "P9"


@pytest.mark.parametrize("phone", ["+1 650-521-2836", "(650) 521 2836", "6505212836", "650.521.2836"])
def test_phone_formats_match(directory: PolicyholderDirectory, phone: str) -> None:
    _, outcome = apply(directory, VerificationState(), {**pick(MARGARET, "full_name", "dob"), "phone": phone})
    assert outcome.status == "verified" and outcome.party_id == "P9"


@pytest.mark.parametrize("dob", ["March 15, 1985", "03/15/1985", "15 March 1985", "15th of March, 1985", "Mar 15th 1985"])
def test_dob_formats_match(directory: PolicyholderDirectory, dob: str) -> None:
    _, outcome = apply(directory, VerificationState(), {**pick(MARGARET, "full_name", "id_last4"), "dob": dob})
    assert outcome.status == "verified" and outcome.party_id == "P9"


def test_name_order_and_case_do_not_matter(directory: PolicyholderDirectory) -> None:
    proposals = {**pick(MARGARET, "dob", "id_last4"), "full_name": "CHEN margaret"}
    _, outcome = apply(directory, VerificationState(), proposals)
    assert outcome.status == "verified"


# ---------------------------------------------------------------------------
# Aliases
# ---------------------------------------------------------------------------


def test_aliases_count_for_p13(directory: PolicyholderDirectory) -> None:
    proposals = {"full_name": "Yaven Li", "email": "yawen.li@example.com", "phone": "650-521-2830"}
    _, outcome = apply(directory, VerificationState(), proposals)
    assert outcome.status == "verified" and outcome.party_id == "P13"


@pytest.mark.parametrize(
    "proposals",
    [
        {"full_name": "Yaven Li", "dob": "1989-12-03", "id_last4": "5317"},
        {"full_name": "Ya Wen Li", "email": "yawen.li@example.com", "id_last4": "5317"},
        {"full_name": "Li Yaven", "email": "YAWEN.LI@example.com", "dob": "December 3, 1989"},
        {"phone": "+16505212830", "email": "yawen.li@gmail.com", "dob": "12/03/1989"},
    ],
)
def test_alias_combinations(directory: PolicyholderDirectory, proposals: dict[str, str]) -> None:
    _, outcome = apply(directory, VerificationState(), proposals)
    assert outcome.status == "verified" and outcome.party_id == "P13"


def test_distinct_phone_alias_counts() -> None:
    holder = Policyholder(
        party_id="PX", name="Kim Park", policy_number="POL-5555", dob=date(1970, 5, 5), id_type="ssn_last4",
        id_last4="1234", phone="+14155550100", phone_aliases=["+14155550199"], email="kim@example.com",
    )  # fmt: skip
    directory = PolicyholderDirectory([holder])
    _, outcome = apply(directory, VerificationState(), {"full_name": "Kim Park", "phone": "415 555 0199", "dob": "1970-05-05"})
    assert outcome.status == "verified" and outcome.party_id == "PX"


# ---------------------------------------------------------------------------
# Wrong and conflicting values
# ---------------------------------------------------------------------------


def test_wrong_last4_is_mismatch_then_correction_verifies(directory: PolicyholderDirectory) -> None:
    state, outcome = apply(directory, VerificationState(), {**pick(MARGARET, "full_name", "dob"), "id_last4": "4473"})
    assert outcome.status == "mismatch"
    assert outcome.party_id is None
    assert outcome.attempt_counted
    assert outcome.provided_pii_count == 3
    assert state.failed_attempts == 1
    # The outcome never reveals which field failed.
    assert "id_last4" not in outcome.reason and "last4" not in outcome.reason
    assert "name" not in outcome.reason and "dob" not in outcome.reason

    state, outcome = apply(directory, state, {"id_last4": "4472"})
    assert outcome.status == "verified" and outcome.party_id == "P9"
    assert state.failed_attempts == 1  # unchanged on success


def test_mismatch_outcomes_are_indistinguishable_across_failing_fields(directory: PolicyholderDirectory) -> None:
    wrong_variants = [
        {**pick(MARGARET, "full_name", "dob"), "id_last4": "0000"},
        {**pick(MARGARET, "full_name", "id_last4"), "dob": "1985-03-16"},
        {**pick(MARGARET, "dob", "id_last4"), "full_name": "Margaret Chan"},
        {"full_name": "Nobody Known", "dob": "2000-01-01", "id_last4": "0000"},
    ]
    shapes = set()
    for proposals in wrong_variants:
        _, outcome = apply(directory, VerificationState(), proposals)
        shapes.add((outcome.status, outcome.party_id, outcome.provided_pii_count, outcome.attempt_counted, outcome.reason))
    assert shapes == {("mismatch", None, 3, True, "failed_attempt_counted")}


def test_conflicting_fields_from_different_people_never_verify(directory: PolicyholderDirectory) -> None:
    proposals = {"full_name": MARGARET["full_name"], "dob": AVA["dob"], "id_last4": MARGARET["id_last4"]}
    state, outcome = apply(directory, VerificationState(), proposals)
    assert outcome.status == "mismatch" and outcome.party_id is None
    # Adding more correct Margaret fields does not override the conflicting DOB.
    state, outcome = apply(directory, state, pick(MARGARET, "phone", "email"))
    assert outcome.status == "mismatch" and outcome.party_id is None
    assert state.status == "unverified"
    # Ava's other fields do not help either: Margaret's name and last4 conflict with Ava.
    state, outcome = apply(directory, state, pick(AVA, "phone"))
    assert outcome.status != "verified"


def test_correcting_the_conflicting_field_then_verifies(directory: PolicyholderDirectory) -> None:
    state, _ = apply(directory, VerificationState(), {**pick(MARGARET, "full_name", "id_last4"), "dob": AVA["dob"]})
    state, outcome = apply(directory, state, pick(MARGARET, "dob"))
    assert outcome.status == "verified" and outcome.party_id == "P9"


@pytest.mark.parametrize("policy", [AVA["policy_number"], "POL-0000"])
def test_wrong_policy_number_blocks_otherwise_matching_party(directory: PolicyholderDirectory, policy: str) -> None:
    proposals = {**pick(MARGARET, "full_name", "dob", "id_last4", "phone"), "policy_number": policy}
    state, outcome = apply(directory, VerificationState(), proposals)
    assert outcome.status == "mismatch"
    assert state.party_id is None
    state, outcome = apply(directory, state, {"policy_number": "POL-9921"})
    assert outcome.status == "verified" and outcome.party_id == "P9"


def test_invalid_values_are_ignored_not_counted(directory: PolicyholderDirectory) -> None:
    proposals = {"full_name": "Margaret", "dob": "03/15/85", "phone": "12345", "email": "nope", "id_last4": "44"}
    state, outcome = apply(directory, VerificationState(), proposals)
    assert outcome.status == "pending"
    assert outcome.newly_provided == ()
    assert outcome.provided_pii_count == 0
    assert state.field_matches == {}


def test_latest_value_per_field_wins(directory: PolicyholderDirectory) -> None:
    state, _ = apply(directory, VerificationState(), {"dob": "1990-08-21"})
    assert state.field_matches["dob"] == ["P7"]
    state, _ = apply(directory, state, {"dob": "1985-03-15"})
    assert state.field_matches["dob"] == ["P9"]


# ---------------------------------------------------------------------------
# Failure counting and lockout
# ---------------------------------------------------------------------------


def test_same_failing_snapshot_counts_once(directory: PolicyholderDirectory) -> None:
    wrong = {**pick(MARGARET, "full_name", "dob"), "id_last4": "0000"}
    state, first = apply(directory, VerificationState(), wrong)
    state, second = apply(directory, state, wrong)
    state, third = apply(directory, state, {})
    assert first.attempt_counted and not second.attempt_counted and not third.attempt_counted
    assert (second.status, third.status) == ("mismatch", "mismatch")
    assert second.reason == "failed_attempt_repeated"
    assert state.failed_attempts == 1


def test_lock_at_max_failures_and_stays_locked(directory: PolicyholderDirectory) -> None:
    state = VerificationState()
    statuses = []
    for last4 in ("0001", "0002", "0003"):
        state, outcome = apply(directory, state, {**pick(MARGARET, "full_name", "dob"), "id_last4": last4})
        statuses.append(outcome.status)
    assert statuses == ["mismatch", "mismatch", "locked"]
    assert state.status == "locked" and state.failed_attempts == 3 and state.party_id is None

    locked_state, outcome = apply(directory, state, MARGARET)
    assert outcome.status == "locked"
    assert outcome.party_id is None
    assert outcome.newly_provided == ()
    assert locked_state.status == "locked"
    assert locked_state.field_matches == state.field_matches


def test_guessing_last4_value_by_value_locks(directory: PolicyholderDirectory) -> None:
    """Every wrong guess matches nobody ([]); each distinct guess must still count."""
    state, _ = apply(directory, VerificationState(), pick(MARGARET, "full_name", "dob"))
    outcomes = []
    for guess in ("1111", "2222", "3333", "4472"):
        state, outcome = apply(directory, state, {"id_last4": guess})
        outcomes.append(outcome.status)
    assert outcomes == ["mismatch", "mismatch", "locked", "locked"]
    assert state.status == "locked" and state.party_id is None


def test_guessing_other_fields_also_counts(directory: PolicyholderDirectory) -> None:
    state, _ = apply(directory, VerificationState(), pick(MARGARET, "full_name", "id_last4"))
    for dob in ("1985-03-16", "1985-03-17"):
        state, outcome = apply(directory, state, {"dob": dob})
        assert outcome.attempt_counted
    state, outcome = apply(directory, state, {"phone": "650 000 0000"})
    assert outcome.status == "locked"


def test_restating_values_after_failure_does_not_count_again(directory: PolicyholderDirectory) -> None:
    state, first = apply(directory, VerificationState(), {**pick(MARGARET, "full_name", "dob"), "id_last4": "0000"})
    assert first.attempt_counted
    state, outcome = apply(directory, state, {"id_last4": "0000"})  # same wrong value again
    assert (outcome.status, outcome.attempt_counted) == ("mismatch", False)
    state, outcome = apply(directory, state, {"full_name": "chen margaret"})  # same correct name again
    assert (outcome.status, outcome.attempt_counted) == ("mismatch", False)
    state, outcome = apply(directory, state, {"dob": "March 15, 1985"})  # same DOB, another spelling
    assert (outcome.status, outcome.attempt_counted) == ("mismatch", False)
    assert state.failed_attempts == 1


def test_switching_a_field_to_another_party_counts(directory: PolicyholderDirectory) -> None:
    state, _ = apply(directory, VerificationState(), {**pick(MARGARET, "full_name", "dob"), "id_last4": "0000"})
    state, outcome = apply(directory, state, {"id_last4": AVA["id_last4"]})
    assert outcome.attempt_counted and state.failed_attempts == 2


def test_lock_respects_custom_max_failures(directory: PolicyholderDirectory) -> None:
    _, outcome = apply(directory, VerificationState(), {**pick(MARGARET, "full_name", "dob"), "id_last4": "0000"}, max_failures=1)
    assert outcome.status == "locked"


def test_failures_already_at_limit_lock_even_without_new_count(directory: PolicyholderDirectory) -> None:
    wrong = {**pick(MARGARET, "full_name", "dob"), "id_last4": "0000"}
    state, _ = apply(directory, VerificationState(), wrong, max_failures=5)
    state, outcome = apply(directory, state, {}, max_failures=1)
    assert outcome.status == "locked" and not outcome.attempt_counted


def test_pending_does_not_count_failures(directory: PolicyholderDirectory) -> None:
    state, outcome = apply(directory, VerificationState(), {"full_name": "Margaret Chen", "id_last4": "0000"})
    assert outcome.status == "pending" and not outcome.attempt_counted
    assert state.failed_attempts == 0


# ---------------------------------------------------------------------------
# Verified sessions: restatement and caller change
# ---------------------------------------------------------------------------


@pytest.fixture
def verified_state(directory: PolicyholderDirectory) -> VerificationState:
    state, outcome = apply(directory, VerificationState(), pick(MARGARET, "full_name", "dob", "id_last4"))
    assert outcome.status == "verified"
    return state


def test_restating_own_details_is_already_verified(directory: PolicyholderDirectory, verified_state: VerificationState) -> None:
    state, outcome = apply(directory, verified_state, pick(MARGARET, "email", "policy_number"))
    assert outcome.status == "already_verified" and outcome.party_id == "P9"
    assert state.status == "verified" and state.party_id == "P9"
    assert state.verified_at == verified_state.verified_at
    assert {"email", "policy_number"} <= set(state.field_matches)


def test_no_new_fields_keeps_verification(directory: PolicyholderDirectory, verified_state: VerificationState) -> None:
    state, outcome = apply(directory, verified_state, {})
    assert outcome.status == "already_verified"
    assert state == verified_state


def test_caller_changed_on_different_name(directory: PolicyholderDirectory, verified_state: VerificationState) -> None:
    state, outcome = apply(directory, verified_state, {"full_name": "Ava Lopez"})
    assert outcome.status == "caller_changed"
    assert outcome.party_id is None
    assert state.status == "unverified"
    assert state.party_id is None and state.verified_at is None
    assert state.field_matches == {"full_name": ["P7"]}


@pytest.mark.parametrize(
    "proposals",
    [{"dob": "1990-08-21"}, {"policy_number": "POL-1044"}, {"phone": "650 000 0000"}, {"email": "x@y.com"}],
)
def test_caller_changed_on_any_non_matching_field(
    directory: PolicyholderDirectory, verified_state: VerificationState, proposals: dict[str, str]
) -> None:
    state, outcome = apply(directory, verified_state, proposals)
    assert outcome.status == "caller_changed"
    assert state.status != "verified"


def test_caller_change_keeps_failure_budget(directory: PolicyholderDirectory) -> None:
    state, _ = apply(directory, VerificationState(), {**pick(MARGARET, "full_name", "dob"), "id_last4": "0000"})
    state, outcome = apply(directory, state, {"id_last4": "4472"})
    assert outcome.status == "verified" and state.failed_attempts == 1
    state, outcome = apply(directory, state, {"full_name": "Ava Lopez"})
    assert outcome.status == "caller_changed"
    assert state.failed_attempts == 1


def test_after_caller_change_new_person_must_verify_fully(directory: PolicyholderDirectory, verified_state: VerificationState) -> None:
    state, _ = apply(directory, verified_state, {"full_name": "Ava Lopez"})
    state, outcome = apply(directory, state, pick(AVA, "dob"))
    assert outcome.status == "pending" and outcome.provided_pii_count == 2
    state, outcome = apply(directory, state, pick(AVA, "id_last4"))
    assert outcome.status == "verified" and outcome.party_id == "P7"


# ---------------------------------------------------------------------------
# Representatives
# ---------------------------------------------------------------------------


def test_representative_is_blocked_even_with_all_details(directory: PolicyholderDirectory) -> None:
    vstate = VerificationState(caller_role=CallerRole.REPRESENTATIVE, representative_declared=True)
    state, outcome = apply(directory, vstate, MARGARET)
    assert outcome.status == "representative_blocked"
    assert outcome.party_id is None
    assert state.status == "unverified" and state.party_id is None
    assert state.representative_declared


def test_representative_flag_is_sticky_across_turns(directory: PolicyholderDirectory) -> None:
    state = VerificationState(representative_declared=True)
    for fields in (("full_name",), ("dob",), ("id_last4", "phone")):
        state, outcome = apply(directory, state, pick(MARGARET, *fields))
        assert outcome.status == "representative_blocked"
    assert state.status == "unverified"


# ---------------------------------------------------------------------------
# Ambiguity and adversarial proposals
# ---------------------------------------------------------------------------


def test_two_parties_sharing_three_values_is_ambiguous() -> None:
    def holder(party_id: str, last4: str, email: str) -> Policyholder:
        return Policyholder(
            party_id=party_id, name="Sam Lee", policy_number=f"POL-{last4}", dob=date(1990, 1, 1),
            id_type="ssn_last4", id_last4=last4, phone="+14155550100", email=email,
        )  # fmt: skip

    directory = PolicyholderDirectory([holder("A", "1111", "a@x.com"), holder("B", "2222", "b@x.com")])
    state, outcome = apply(directory, VerificationState(), {"full_name": "Sam Lee", "dob": "1990-01-01", "phone": "4155550100"})
    assert outcome.status == "ambiguous" and outcome.party_id is None
    assert state.status == "unverified"
    state, outcome = apply(directory, state, {"id_last4": "2222"})
    assert outcome.status == "verified" and outcome.party_id == "B"


def test_model_cannot_inject_protected_keys(directory: PolicyholderDirectory) -> None:
    proposals = {"verified": "true", "party_id": "P9", "status": "verified", "full_name": "Margaret Chen"}
    state, outcome = apply(directory, VerificationState(), proposals)
    assert outcome.status == "pending"
    assert outcome.newly_provided == ("full_name",)
    assert state.status == "unverified" and state.party_id is None
    assert set(state.field_matches) == {"full_name"}


def test_non_string_values_are_ignored(directory: PolicyholderDirectory) -> None:
    proposals = {"full_name": None, "dob": 19850315, "id_last4": 4472, "phone": ["6505212836"]}
    _, outcome = apply(directory, VerificationState(), proposals)  # type: ignore[arg-type]
    assert outcome.status == "pending" and outcome.newly_provided == ()


def test_corrupt_verified_state_without_party_is_not_trusted(directory: PolicyholderDirectory) -> None:
    corrupt = VerificationState(status="verified", party_id=None, field_matches={"full_name": ["P9"]})
    state, outcome = apply(directory, corrupt, {})
    assert outcome.status == "pending"
    assert state.status == "unverified"


# ---------------------------------------------------------------------------
# Purity and state hygiene
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "proposals",
    [
        pick(MARGARET, "full_name", "dob", "id_last4"),
        {**pick(MARGARET, "full_name", "dob"), "id_last4": "0000"},
        {"full_name": "Ava Lopez"},
        {},
    ],
)
def test_input_state_is_never_mutated(directory: PolicyholderDirectory, proposals: dict[str, str]) -> None:
    for start in (
        VerificationState(field_matches={"email": ["P9"]}, failed_attempts=1, last_failed_signature="abc"),
        VerificationState(status="verified", party_id="P9", verified_at=NOW, field_matches={"full_name": ["P9"]}),
        VerificationState(status="locked", failed_attempts=3),
    ):
        before = start.model_dump()
        new_state, _ = apply(directory, start, proposals)
        assert start.model_dump() == before
        assert new_state is not start
        new_state.field_matches.setdefault("phone", []).append("ZZ")
        assert start.model_dump() == before


def test_state_stores_no_raw_identity_values(directory: PolicyholderDirectory) -> None:
    state, _ = apply(directory, VerificationState(), MARGARET)
    dumped = state.model_dump_json()
    for raw in ("Margaret", "1985", "4472", "6505212836", "margaret@email.com", "9921"):
        assert raw not in dumped


def test_outcome_exposes_only_counts_and_codes(directory: PolicyholderDirectory) -> None:
    _, outcome = apply(directory, VerificationState(), {**pick(MARGARET, "full_name", "dob"), "id_last4": "0000"})
    text = repr(outcome)
    for raw in ("Margaret", "1985", "0000", "P9"):
        assert raw not in text


# ---------------------------------------------------------------------------
# Outdated detail after a counted failure (C15)
# ---------------------------------------------------------------------------

OLD_PHONE = "650-555-0000"


def test_outdated_detail_gives_way_to_a_new_correct_detail(directory: PolicyholderDirectory) -> None:
    state, outcome = apply(directory, VerificationState(), {**pick(MARGARET, "full_name", "dob"), "phone": OLD_PHONE})
    assert outcome.status == "mismatch" and state.failed_attempts == 1

    state, outcome = apply(directory, state, pick(MARGARET, "id_last4"))
    assert outcome.status == "verified" and outcome.party_id == "P9"
    assert state.failed_attempts == 1


def test_outdated_detail_in_the_same_turn_as_correct_ones_then_email_verifies(directory: PolicyholderDirectory) -> None:
    proposals = {**pick(MARGARET, "full_name", "dob", "id_last4"), "phone": OLD_PHONE}
    state, outcome = apply(directory, VerificationState(), proposals)
    assert outcome.status == "mismatch" and state.failed_attempts == 1

    state, outcome = apply(directory, state, pick(MARGARET, "email"))
    assert outcome.status == "verified" and state.failed_attempts == 1


def test_uncharged_unmatched_value_still_blocks(directory: PolicyholderDirectory) -> None:
    # Before any counted failure nothing is dropped: the pending value is part of the next snapshot.
    state, outcome = apply(directory, VerificationState(), {"full_name": MARGARET["full_name"], "phone": OLD_PHONE})
    assert outcome.status == "pending"
    state, outcome = apply(directory, state, pick(MARGARET, "dob"))
    assert outcome.status == "mismatch" and outcome.attempt_counted


def test_value_matching_another_party_is_never_dropped(directory: PolicyholderDirectory) -> None:
    state, _ = apply(directory, VerificationState(), {**pick(MARGARET, "full_name", "id_last4"), "dob": AVA["dob"]})
    state, outcome = apply(directory, state, pick(MARGARET, "email"))
    assert outcome.status == "mismatch" and outcome.party_id is None


def test_every_turn_after_a_failure_is_counted_or_verifies(directory: PolicyholderDirectory) -> None:
    """Dropping stale values must never make a non-verifying turn come back as uncounted ``pending``."""
    state, outcome = apply(directory, VerificationState(), {"full_name": MARGARET["full_name"], "dob": "1970-01-01", "phone": OLD_PHONE})
    assert state.failed_attempts == 1
    turns = [{"id_last4": "0001"}, {"email": "nobody@example.org"}, {"id_last4": "0002"}]
    for proposals in turns:
        before = state.failed_attempts
        state, outcome = apply(directory, state, proposals, max_failures=10)
        assert outcome.status in ("mismatch", "locked")
        assert outcome.attempt_counted and state.failed_attempts == before + 1


def test_caller_change_snapshot_is_not_treated_as_charged(directory: PolicyholderDirectory) -> None:
    state, _ = apply(directory, VerificationState(), {**pick(MARGARET, "full_name", "dob"), "id_last4": "0000"})
    state, _ = apply(directory, state, {"id_last4": "4472"})
    state, outcome = apply(directory, state, {"full_name": "Ava Lopez", "phone": OLD_PHONE})
    assert outcome.status == "caller_changed"
    assert state.last_failed_signature is None
    # The unmatched phone from the caller-change turn still blocks Ava until it is charged.
    state, outcome = apply(directory, state, pick(AVA, "dob", "id_last4"))
    assert outcome.status == "mismatch" and outcome.attempt_counted


# ---------------------------------------------------------------------------
# Name variants through the gate (C22)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["Margaret A. Chen", "Margaret Anne Chen", "Mrs. Margaret Chen", "Dr Margaret Chen"])
def test_stated_name_variant_with_three_correct_fields_verifies(directory: PolicyholderDirectory, name: str) -> None:
    state, outcome = apply(directory, VerificationState(), {"full_name": name, **pick(MARGARET, "dob", "id_last4")})
    assert outcome.status == "verified" and outcome.party_id == "P9"
    assert state.failed_attempts == 0


def test_given_name_alone_is_not_enough_for_the_name_field(directory: PolicyholderDirectory) -> None:
    _, outcome = apply(directory, VerificationState(), {"full_name": "Dr Margaret", **pick(MARGARET, "dob", "id_last4")})
    assert outcome.status == "mismatch" and outcome.party_id is None  # a title plus one name matches nobody
    _, outcome = apply(directory, VerificationState(), {"full_name": "Margaret Chen Smith", **pick(MARGARET, "dob", "id_last4")})
    assert outcome.status == "mismatch"
