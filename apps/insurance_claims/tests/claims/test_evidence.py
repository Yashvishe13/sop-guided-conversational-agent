"""Tests for authorized evidence packets, grounding tokens, and leak tokens."""

from __future__ import annotations

import json
from dataclasses import FrozenInstanceError
from datetime import date
from decimal import Decimal

import pytest

from insurance_claims.claims.evidence import (
    EvidencePacket,
    GroundingTokens,
    build_case_evidence,
    find_amounts,
    find_dates,
    grounding_tokens,
    secret_claim_tokens,
)
from insurance_claims.claims.fixtures import FixtureBundle, load_fixtures
from insurance_claims.claims.repository import ClaimRepository, GuidelineRepository
from insurance_claims.config import APP_ROOT
from insurance_claims.domain.models import Claim

FIXTURES_DIR = APP_ROOT / "fixtures"
TODAY = date(2026, 10, 3)


@pytest.fixture(scope="module")
def bundle() -> FixtureBundle:
    return load_fixtures(FIXTURES_DIR)


@pytest.fixture(scope="module")
def guidelines(bundle: FixtureBundle) -> GuidelineRepository:
    return GuidelineRepository(bundle.guidelines)


@pytest.fixture(scope="module")
def repo(bundle: FixtureBundle) -> ClaimRepository:
    return ClaimRepository(bundle.claims)


def _get(repo: ClaimRepository, party_id: str, case_id: str) -> Claim:
    return repo.get_for_party(party_id, case_id).claims[0]


def _packet(bundle: FixtureBundle, guidelines: GuidelineRepository, claim: Claim, *, today: date = TODAY) -> EvidencePacket:
    return build_case_evidence(claim, guidelines, bundle.claim_schema, today=today)


@pytest.fixture(scope="module")
def cl2048_packet(bundle: FixtureBundle, guidelines: GuidelineRepository, repo: ClaimRepository) -> EvidencePacket:
    return _packet(bundle, guidelines, _get(repo, "P9", "CL-2048"))


# ---------------------------------------------------------------------------
# Evidence packet for CL-2048
# ---------------------------------------------------------------------------


def test_case_facts(cl2048_packet: EvidencePacket) -> None:
    case = cl2048_packet.case
    assert case["case_id"] == "CL-2048"
    assert case["case_type"] == "healthcare"
    assert case["status"] == "denied"
    assert case["created_at"] == "2026-01-12"
    assert case["appeal_deadline"] == "2026-03-18"
    assert case["documents_needed"] == ["pathology report", "office note"]
    assert case["denial_reason"].startswith("the review file did not include")
    assert case["amounts"] == {
        "expected_reimbursement_amount": "0.00",
        "allowed_max_amount": "1450.00",
        "net_pay": "0.00",
        "net_fee": "1450.00",
    }
    assert cl2048_packet.today == "2026-10-03"


def test_deadline_passed_with_frozen_clock(cl2048_packet: EvidencePacket) -> None:
    assert cl2048_packet.deadline == {"date": "2026-03-18", "status": "passed", "days_from_today": -199}


@pytest.mark.parametrize(
    ("today", "status", "days"),
    [(date(2026, 3, 18), "today", 0), (date(2026, 3, 1), "upcoming", 17), (date(2026, 3, 19), "passed", -1)],
)
def test_deadline_status_boundaries(bundle: FixtureBundle, guidelines: GuidelineRepository, repo: ClaimRepository,
                                    today: date, status: str, days: int) -> None:  # fmt: skip
    packet = _packet(bundle, guidelines, _get(repo, "P9", "CL-2048"), today=today)
    assert packet.deadline["status"] == status
    assert packet.deadline["days_from_today"] == days


def test_no_deadline(bundle: FixtureBundle, guidelines: GuidelineRepository, repo: ClaimRepository) -> None:
    packet = _packet(bundle, guidelines, _get(repo, "P9", "CL-2102"))
    assert packet.deadline == {"date": None, "status": "none", "days_from_today": None}
    assert packet.case["appeal_deadline"] is None
    assert packet.case["denial_reason"] is None


def test_documents_map_to_guidelines(cl2048_packet: EvidencePacket) -> None:
    docs = {doc["requested_name"]: doc for doc in cl2048_packet.documents}
    assert list(docs) == ["pathology report", "office note"]
    assert docs["pathology report"]["guideline_name"] == "original pathology report"
    assert docs["office note"]["guideline_name"] == "treating provider office note"
    assert docs["pathology report"]["guidance"].startswith("The pathology report should include")
    assert docs["pathology report"]["alternative"].startswith("If the original pathology report is missing")
    assert docs["office note"]["guidance"].startswith("The office note should show")
    assert docs["office note"]["alternative"].startswith("If you cannot get the full office note")


def test_unknown_document_falls_back_to_default_guidance(
    bundle: FixtureBundle, guidelines: GuidelineRepository, repo: ClaimRepository
) -> None:
    packet = _packet(bundle, guidelines, _get(repo, "P12", "CL-3001"))
    (doc,) = packet.documents
    assert doc["requested_name"] == "diagnosis report"
    assert doc["guideline_name"] is None
    assert doc["guidance"] == guidelines.default_guidance()
    assert doc["alternative"] == guidelines.default_alternative()


def test_guidance_and_settings(cl2048_packet: EvidencePacket) -> None:
    assert cl2048_packet.case_type_guidance.startswith("For medical claims")
    assert cl2048_packet.default_guidance.startswith("Use the member portal")
    assert cl2048_packet.default_alternative.startswith("If the exact item is not available yet")
    assert cl2048_packet.processing_time == "usually less than a week"
    assert cl2048_packet.human_review_rule.startswith("If the caller still cannot provide")
    assert cl2048_packet.fallback_followup.startswith("I do not see a separate claim-specific rule")
    assert set(cl2048_packet.field_definitions) == {
        "expected_reimbursement_amount",
        "allowed_max_amount",
        "net_pay",
        "net_fee",
    }


def test_auto_claim_gets_case_type_guidance(bundle: FixtureBundle, guidelines: GuidelineRepository, repo: ClaimRepository) -> None:
    packet = _packet(bundle, guidelines, _get(repo, "P9", "CL-2102"))
    assert packet.case_type_guidance.startswith("For auto claims")


def test_packet_contains_only_the_selected_claim(cl2048_packet: EvidencePacket) -> None:
    dumped = json.dumps(cl2048_packet.model_dump())
    for other in ("CL-2011", "CL-1899", "CL-2102", "CL-3001", "diagnosis report", "780.00", "3200.00"):
        assert other not in dumped


def test_packet_serializes_to_json(cl2048_packet: EvidencePacket) -> None:
    restored = EvidencePacket.model_validate_json(cl2048_packet.model_dump_json())
    assert restored == cl2048_packet


# ---------------------------------------------------------------------------
# Grounding tokens
# ---------------------------------------------------------------------------


def test_grounding_tokens_for_cl2048(cl2048_packet: EvidencePacket) -> None:
    tokens = grounding_tokens(cl2048_packet)
    assert isinstance(tokens, GroundingTokens)
    assert tokens.case_ids == frozenset({"CL-2048"})
    assert Decimal("1450.00") in tokens.amounts
    assert Decimal("0.00") in tokens.amounts
    assert Decimal("1450") in tokens.amounts  # numeric comparison
    assert date(2026, 1, 12) in tokens.dates
    assert date(2026, 3, 18) in tokens.dates
    assert TODAY not in tokens.dates
    assert {"pathology report", "office note", "original pathology report", "treating provider office note"} <= tokens.documents
    assert Decimal("780.00") not in tokens.amounts


def test_grounding_tokens_include_extra_texts(cl2048_packet: EvidencePacket) -> None:
    tokens = grounding_tokens(cl2048_packet, ["We received $2,500.75 on April 2, 2026 and 99.10 dollars on 2026-05-01."])
    assert {Decimal("2500.75"), Decimal("99.10")} <= tokens.amounts
    assert {date(2026, 4, 2), date(2026, 5, 1)} <= tokens.dates


def test_grounding_tokens_are_frozen(cl2048_packet: EvidencePacket) -> None:
    tokens = grounding_tokens(cl2048_packet)
    with pytest.raises(FrozenInstanceError):
        tokens.case_ids = frozenset()  # type: ignore[misc]


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("$1,450.00", {Decimal("1450.00")}),
        ("$1450", {Decimal("1450.00")}),
        ("1450.00 dollars", {Decimal("1450.00")}),
        ("1,450.00", {Decimal("1450.00")}),
        ("paid $1,450.50, then", {Decimal("1450.50")}),
        ("3200 USD", {Decimal("3200.00")}),
        ("call 2026-03-18 or 650 521 2836", set()),
        ("a week or 3 days", set()),
    ],
)
def test_find_amounts(text: str, expected: set[Decimal]) -> None:
    assert find_amounts(text) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("due 2026-03-18.", {date(2026, 3, 18)}),
        ("by March 18, 2026", {date(2026, 3, 18)}),
        ("by March 18th 2026", {date(2026, 3, 18)}),
        ("on 3/18/2026", {date(2026, 3, 18)}),
        ("18 March 2026", {date(2026, 3, 18)}),
        ("March 18", set()),  # no year: not a full date
        ("2026-02-30", set()),
    ],
)
def test_find_dates(text: str, expected: set[date]) -> None:
    assert find_dates(text) == expected


# ---------------------------------------------------------------------------
# Secret claim tokens (pre-verification leak detection)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def secrets(repo: ClaimRepository) -> list[str]:
    return secret_claim_tokens(repo.all_claims_unscoped())


@pytest.mark.parametrize(
    "token",
    [
        "cl-2048",
        "cl-3001",
        "1450.00",
        "1,450.00",
        "1450",
        "780.00",
        "pathology report",
        "office note",
        "diagnosis report",
        "march 18, 2026",
        "march 18",
        "2026-03-18",
        "january 12, 2026",
        "2026-01-12",
        "the review file did not include the pathology report and the treating provider office note",
    ],
)
def test_secret_tokens_include_distinctive_strings(secrets: list[str], token: str) -> None:
    assert token in secrets


@pytest.mark.parametrize(
    "generic",
    ["denied", "healthcare", "dental", "auto", "closed", "open", "claim", "0.00", "0", "january", "2026", "report"],
)
def test_secret_tokens_exclude_generic_words(secrets: list[str], generic: str) -> None:
    assert generic not in secrets


def test_secret_tokens_are_lowercase_sorted_unique(secrets: list[str]) -> None:
    assert secrets == sorted(set(secrets))
    assert all(token == token.lower() for token in secrets)


def test_secret_tokens_avoid_single_digit_day_substring_traps(secrets: list[str]) -> None:
    # CL-3001 was created 2026-03-01: "march 1" would match inside "march 15, 1985" (a DOB).
    assert "march 1" not in secrets
    assert "march 1st" in secrets and "march 1, 2026" in secrets
    dob_sentence = "thanks, i have your date of birth as march 15, 1985"
    assert not [token for token in secrets if token in dob_sentence]


def test_secret_tokens_never_flag_the_demo_verification_prompt(secrets: list[str]) -> None:
    reply = (
        "thanks margaret. i can help with your denied healthcare claim from january once i verify your identity. "
        "could you share your phone number or email on file?"
    )
    assert not [token for token in secrets if token in reply]


def test_secret_tokens_empty_input() -> None:
    assert secret_claim_tokens([]) == []
