"""Tests for party-scoped repositories and guideline lookup."""

from __future__ import annotations

from datetime import date

import pytest

from insurance_claims.claims.fixtures import FixtureBundle, load_fixtures
from insurance_claims.claims.repository import (
    AuthorizationError,
    ClaimRepository,
    GuidelineRepository,
    LookupResult,
    PolicyholderDirectory,
    RepresentativeDirectory,
    natural_join,
)
from insurance_claims.config import APP_ROOT
from insurance_claims.domain.models import (
    Claim,
    DocumentGuidelines,
    FollowupRule,
    Policyholder,
)

FIXTURES_DIR = APP_ROOT / "fixtures"


@pytest.fixture(scope="module")
def bundle() -> FixtureBundle:
    return load_fixtures(FIXTURES_DIR)


@pytest.fixture(scope="module")
def directory(bundle: FixtureBundle) -> PolicyholderDirectory:
    return PolicyholderDirectory(bundle.policyholders)


@pytest.fixture(scope="module")
def claims(bundle: FixtureBundle) -> ClaimRepository:
    return ClaimRepository(bundle.claims)


@pytest.fixture(scope="module")
def guidelines(bundle: FixtureBundle) -> GuidelineRepository:
    return GuidelineRepository(bundle.guidelines)


def _claim(case_id: str, **overrides: object) -> Claim:
    base: dict[str, object] = {
        "case_id": case_id,
        "party_id": "P1",
        "case_type": "healthcare",
        "created_at": "2026-01-12",
        "status": "denied",
        "expected_reimbursement_amount": "0.00",
        "allowed_max_amount": "10.00",
        "net_pay": "0.00",
        "net_fee": "10.00",
    }
    base.update(overrides)
    return Claim.model_validate(base)


# ---------------------------------------------------------------------------
# PolicyholderDirectory
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("field", "value", "expected"),
    [
        ("full_name", "Margaret Chen", {"P9"}),
        ("full_name", "chen MARGARET", {"P9"}),
        ("full_name", "Yaven Li", {"P13"}),
        ("full_name", "Ya Wen Li", {"P13"}),
        ("full_name", "Margaret", set()),
        ("full_name", "Margaret Chan", set()),
        ("dob", "1985-03-15", {"P9"}),
        ("dob", "March 15, 1985", {"P9"}),
        ("dob", "03/15/1985", {"P9"}),
        ("dob", "15 March 1985", {"P9"}),
        ("dob", "1985-03-16", set()),
        ("phone", "+1 650-521-2836", {"P9"}),
        ("phone", "(650) 521 2836", {"P9"}),
        ("phone", "6505212836", {"P9"}),
        ("phone", "650-521-2830", {"P13"}),
        ("email", "MARGARET@email.com", {"P9"}),
        ("email", "yawen.li@example.com", {"P13"}),
        ("email", "yawen.li@gmail.com", {"P13"}),
        ("email", "someone@else.com", set()),
        ("id_last4", "4472", {"P9"}),
        ("id_last4", "6688", {"P12"}),  # national_id_last4 compares the same way
        ("id_last4", "123-45-4472", {"P9"}),
        ("policy_number", "POL-9921", {"P9"}),
        ("policy_number", "pol 9921", {"P9"}),
        ("policy_number", "9921", {"P9"}),
        ("policy_number", "POL-0000", set()),
    ],
)
def test_parties_matching(directory: PolicyholderDirectory, field: str, value: str, expected: set[str]) -> None:
    assert directory.parties_matching(field, value) == frozenset(expected)


@pytest.mark.parametrize(
    ("field", "value"),
    [("phone", "123"), ("dob", "03/15/85"), ("email", "not-an-email"), ("id_last4", "12"), ("policy_number", "x")],
)
def test_parties_matching_invalid_values_match_nobody(directory: PolicyholderDirectory, field: str, value: str) -> None:
    assert directory.parties_matching(field, value) == frozenset()


def test_parties_matching_non_string_value(directory: PolicyholderDirectory) -> None:
    assert directory.parties_matching("dob", None) == frozenset()  # type: ignore[arg-type]


def test_parties_matching_rejects_unknown_field(directory: PolicyholderDirectory) -> None:
    with pytest.raises(ValueError):
        directory.parties_matching("party_id", "P9")


def test_shared_values_match_every_owner() -> None:
    holders = [
        Policyholder(party_id="A", name="Sam Lee", policy_number="POL-1", dob=date(1990, 1, 1), id_type="ssn_last4",
                     id_last4="1111", phone="6505550000", email="sam@a.com"),
        Policyholder(party_id="B", name="Lee Sam", policy_number="POL-2", dob=date(1990, 1, 1), id_type="ssn_last4",
                     id_last4="2222", phone="6505550001", email="sam@b.com", phone_aliases=["6505550000"]),
    ]  # fmt: skip
    directory = PolicyholderDirectory(holders)
    assert directory.parties_matching("full_name", "Sam Lee") == {"A", "B"}
    assert directory.parties_matching("dob", "1990-01-01") == {"A", "B"}
    assert directory.parties_matching("phone", "650 555 0000") == {"A", "B"}


def test_contact_email_and_first_name(directory: PolicyholderDirectory) -> None:
    assert directory.contact_email("P13") == "yawen.li@gmail.com"  # canonical, never an alias
    assert directory.first_name("P9") == "Margaret"


@pytest.mark.parametrize("party_id", ["", None, "P404"])
def test_contact_lookups_require_known_party(directory: PolicyholderDirectory, party_id: str | None) -> None:
    with pytest.raises(AuthorizationError):
        directory.contact_email(party_id)  # type: ignore[arg-type]
    with pytest.raises(AuthorizationError):
        directory.first_name(party_id)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# ClaimRepository
# ---------------------------------------------------------------------------


def test_list_for_party_newest_first_and_scoped(claims: ClaimRepository) -> None:
    listed = claims.list_for_party("P9")
    assert [c.case_id for c in listed] == ["CL-2102", "CL-2048", "CL-1899", "CL-2011"]
    assert all(c.party_id == "P9" for c in listed)
    assert [c.case_id for c in claims.list_for_party("P12")] == ["CL-3001"]
    assert claims.list_for_party("P7") == []


@pytest.mark.parametrize("party_id", ["", None, "   "])
def test_list_for_party_requires_party(claims: ClaimRepository, party_id: str | None) -> None:
    with pytest.raises(AuthorizationError):
        claims.list_for_party(party_id)  # type: ignore[arg-type]


def test_authorization_error_is_permission_error() -> None:
    assert issubclass(AuthorizationError, PermissionError)


@pytest.mark.parametrize("case_id", ["CL-2048", "cl 2048", "CL2048"])
def test_get_for_party_found(claims: ClaimRepository, case_id: str) -> None:
    result = claims.get_for_party("P9", case_id)
    assert result.status == "found"
    assert [c.case_id for c in result.claims] == ["CL-2048"]


def test_get_for_party_other_party_is_indistinguishable_from_missing(claims: ClaimRepository) -> None:
    foreign = claims.get_for_party("P9", "CL-3001")
    missing = claims.get_for_party("P9", "CL-9999")
    garbage = claims.get_for_party("P9", "drop table")
    assert foreign == missing == garbage == LookupResult(status="not_found", claims=[])


def test_get_for_party_requires_party(claims: ClaimRepository) -> None:
    with pytest.raises(AuthorizationError):
        claims.get_for_party("", "CL-2048")


def test_all_claims_unscoped_returns_everything(claims: ClaimRepository) -> None:
    assert {c.case_id for c in claims.all_claims_unscoped()} == {"CL-2048", "CL-2011", "CL-1899", "CL-2102", "CL-3001"}


def test_duplicate_case_ids_keep_first() -> None:
    repo = ClaimRepository([_claim("CL-100", summary="first"), _claim("CL-100", summary="second")])
    assert [c.summary for c in repo.list_for_party("P1")] == ["first"]


def test_list_tie_break_is_deterministic() -> None:
    repo = ClaimRepository([_claim("CL-101"), _claim("CL-103"), _claim("CL-102")])
    assert [c.case_id for c in repo.list_for_party("P1")] == ["CL-103", "CL-102", "CL-101"]


def test_returned_list_does_not_alias_internal_state(claims: ClaimRepository) -> None:
    first = claims.list_for_party("P9")
    first.clear()
    assert len(claims.list_for_party("P9")) == 4


# ---------------------------------------------------------------------------
# RepresentativeDirectory
# ---------------------------------------------------------------------------


def test_representative_find(bundle: FixtureBundle) -> None:
    reps = RepresentativeDirectory(bundle.representatives)
    found = reps.find("David Chen", "Margaret Chen")
    assert found is not None and found.relationship == "son" and found.buyer_party_id == "P9"
    assert reps.find("chen david", None) == found
    assert reps.find(None, "Margaret Chen") == found
    assert reps.find("David Chen", "Ava Lopez") is None
    assert reps.find("Eve Stone", "Margaret Chen") is None
    assert reps.find(None, None) is None
    assert reps.find("", "") is None
    assert reps.find("David", None) is None  # a single token never matches


# ---------------------------------------------------------------------------
# GuidelineRepository
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("requested", "expected"),
    [
        ("pathology report", "original pathology report"),
        ("Original Pathology Report", "original pathology report"),
        ("office note", "treating provider office note"),
        ("  OFFICE   note ", "treating provider office note"),
        ("repair estimate", "repair estimate"),
        ("accident photos", "supplemental accident scene photos"),
        ("report", "original pathology report"),
        ("diagnosis report", None),
        ("note report", None),
        ("default", None),
        ("", None),
        ("!!!", None),
    ],
)
def test_resolve_document_name(guidelines: GuidelineRepository, requested: str, expected: str | None) -> None:
    assert guidelines.resolve_document_name(requested) == expected


def test_resolve_document_name_ambiguous_is_none() -> None:
    docs = DocumentGuidelines.model_validate(
        {
            "default_guidance": {"en": "d"},
            "document_guidance": {"blue report": {"en": "b"}, "red report": {"en": "r"}},
            "claim_followup_fallback": {"en": "f"},
        }
    )
    repo = GuidelineRepository(docs)
    assert repo.resolve_document_name("report") is None
    assert repo.resolve_document_name("blue report") == "blue report"
    assert repo.default_alternative() is None


def test_document_guidance(guidelines: GuidelineRepository) -> None:
    entry = guidelines.document_guidance("pathology report")
    assert entry is not None
    assert entry["document"] == "pathology report"
    assert entry["guideline_name"] == "original pathology report"
    assert entry["guidance"].startswith("The pathology report should include")
    assert entry["alternative"].startswith("If the original pathology report is missing")
    assert guidelines.document_guidance("diagnosis report") is None


def test_case_type_and_defaults(guidelines: GuidelineRepository) -> None:
    assert guidelines.case_type_guidance("healthcare").startswith("For medical claims")
    assert guidelines.case_type_guidance("HEALTHCARE ").startswith("For medical claims")
    assert guidelines.case_type_guidance("dental") is None
    assert guidelines.default_guidance().startswith("Use the member portal")
    assert guidelines.default_alternative().startswith("If the exact item is not available yet")
    assert guidelines.setting("average_processing_time_after_submission") == "usually less than a week"
    assert guidelines.setting("nope") is None
    assert guidelines.fallback_followup().startswith("I do not see a separate claim-specific rule")


def test_render_followup(bundle, guidelines: GuidelineRepository, claims: ClaimRepository) -> None:
    claim = claims.get_for_party("P9", "CL-2048").claims[0]
    rule = next(r for r in bundle.guidelines.claim_followup_guidance if r.topic == "processing_time_after_submission")
    text = guidelines.render_followup(rule, claim)
    assert "CL-2048" in text
    assert "the pathology report and the office note" in text
    assert "usually less than a week" in text
    assert "{" not in text and "}" not in text


def test_render_followup_without_documents_and_unknown_placeholders(guidelines: GuidelineRepository) -> None:
    claim = _claim("CL-500", documents_needed=[])
    rule = FollowupRule(topic="t", en="For {case_id} send {documents}{unknown} by {case_id.__class__} now.")
    text = guidelines.render_followup(rule, claim)
    assert text == "For CL-500 send the requested documents by {case_id.__class__} now."
    assert "class '" not in text  # never evaluated as a format field


@pytest.mark.parametrize(
    ("items", "expected"),
    [
        ([], "the requested documents"),
        (["pathology report"], "the pathology report"),
        (["pathology report", "office note"], "the pathology report and the office note"),
        (["a", "b", "c"], "the a, the b, and the c"),
        (["the repair estimate", "  "], "the repair estimate"),
    ],
)
def test_natural_join(items: list[str], expected: str) -> None:
    assert natural_join(items) == expected


# ---------------------------------------------------------------------------
# Name variants (C22): titles, suffixes, and a middle name or initial
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name",
    [
        "Margaret A. Chen",
        "Margaret Anne Chen",
        "Mrs. Margaret Chen",
        "Dr Margaret Chen",
        "dr. margaret anne chen",
        "Margaret Chen Jr",
    ],
)
def test_name_with_title_suffix_or_middle_name_matches_the_record(directory: PolicyholderDirectory, name: str) -> None:
    assert directory.parties_matching("full_name", name) == frozenset({"P9"})


@pytest.mark.parametrize(
    "name",
    [
        "Margaret",
        "Mrs Margaret",
        "Dr Chen",
        "Margaret Chen Smith",
        "Smith Margaret Chen",
        "Margaret Chan",
        "Margaret Anne Marie Chen",
    ],
)
def test_partial_or_different_names_still_match_nobody(directory: PolicyholderDirectory, name: str) -> None:
    assert directory.parties_matching("full_name", name) == frozenset()


def test_name_fallback_only_applies_to_full_name(directory: PolicyholderDirectory) -> None:
    assert directory.parties_matching("email", "dr margaret@email.com") == frozenset()
