"""Authorized evidence packets and grounding token sets.

``build_case_evidence`` assembles everything the agent may state about one
authorized claim (claim facts, deadline status from an injected clock, document
guidance, rendered follow-up rules). ``grounding_tokens`` lists the case IDs,
amounts, dates, and documents a reply may cite. ``secret_claim_tokens`` lists
distinctive claim strings that must never appear in a reply before
verification. Fixture text is data, never instructions.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from datetime import date
from decimal import Decimal, InvalidOperation

from pydantic import BaseModel

from insurance_claims.claims.normalize import normalize_case_id, normalize_dob
from insurance_claims.claims.repository import GuidelineRepository
from insurance_claims.domain.models import Claim, ClaimSchemaDoc

__all__ = [
    "AMOUNT_FIELDS",
    "EvidencePacket",
    "GroundingTokens",
    "build_case_evidence",
    "find_amounts",
    "find_dates",
    "grounding_tokens",
    "secret_claim_tokens",
]

AMOUNT_FIELDS: tuple[str, ...] = ("expected_reimbursement_amount", "allowed_max_amount", "net_pay", "net_fee")
PROCESSING_TIME_KEY = "average_processing_time_after_submission"
HUMAN_REVIEW_KEY = "human_review_after_document_alternatives_exhausted"

_CENTS = Decimal("0.01")
_MONTH_NAMES = (
    "january", "february", "march", "april", "may", "june",
    "july", "august", "september", "october", "november", "december",
)  # fmt: skip
_MONTH_WORD = r"(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|june?|july?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)"
_AMOUNT_RES = (
    re.compile(r"\$\s?(\d{1,3}(?:,\d{3})+|\d+)(\.\d{1,2})?(?!\d|,\d)"),
    re.compile(r"(?<![\d.,$])(\d{1,3}(?:,\d{3})+)(\.\d{2})?(?!\d|,\d)"),
    re.compile(r"(?<![\d.,$])(\d+)(\.\d{2})(?![\d])"),
    re.compile(r"(?<![\d.,$])(\d{1,3}(?:,\d{3})+|\d+)(\.\d{1,2})?\s*(?:dollars|usd)\b", re.IGNORECASE),
)
_DATE_RES = (
    re.compile(r"(?<!\d)\d{4}-\d{1,2}-\d{1,2}(?!\d)"),
    re.compile(r"(?<!\d)\d{1,2}/\d{1,2}/\d{4}(?!\d)"),
    re.compile(rf"\b{_MONTH_WORD}\.?\s+\d{{1,2}}(?:st|nd|rd|th)?,?\s+\d{{4}}\b", re.IGNORECASE),
    re.compile(rf"\b\d{{1,2}}(?:st|nd|rd|th)?\s+(?:of\s+)?{_MONTH_WORD}\.?,?\s+\d{{4}}\b", re.IGNORECASE),
)
_MIN_SECRET_LEN = 3


class EvidencePacket(BaseModel):
    case: dict
    deadline: dict
    today: str
    field_definitions: dict[str, str]
    documents: list[dict]
    case_type_guidance: str | None
    default_guidance: str
    default_alternative: str | None
    processing_time: str | None
    human_review_rule: str | None
    fallback_followup: str


@dataclass(frozen=True)
class GroundingTokens:
    case_ids: frozenset[str]
    amounts: frozenset[Decimal]
    dates: frozenset[date]
    documents: frozenset[str]


# ---------------------------------------------------------------------------
# Text scanning helpers (shared with validators)
# ---------------------------------------------------------------------------


def _to_amount(whole: str, cents: str | None) -> Decimal | None:
    try:
        return Decimal(whole.replace(",", "") + (cents or "")).quantize(_CENTS)
    except (InvalidOperation, ValueError):
        return None


def find_amounts(text: str) -> set[Decimal]:
    """Money amounts written as ``$1,450.00``, ``$1450``, ``1450.00``, ``1,450.00`` or ``1450 dollars``."""
    found: set[Decimal] = set()
    if not isinstance(text, str):
        return found
    for pattern in _AMOUNT_RES:
        for match in pattern.finditer(text):
            amount = _to_amount(match.group(1), match.group(2))
            if amount is not None:
                found.add(amount)
    return found


def find_dates(text: str) -> set[date]:
    """Full dates (with a year) in ISO, US numeric, or month-name spellings."""
    found: set[date] = set()
    if not isinstance(text, str):
        return found
    for pattern in _DATE_RES:
        for match in pattern.finditer(text):
            parsed = normalize_dob(match.group(0))
            if parsed is not None:
                found.add(parsed)
    return found


# ---------------------------------------------------------------------------
# Evidence packet
# ---------------------------------------------------------------------------


def _deadline(claim: Claim, today: date) -> dict:
    deadline = claim.appeal_deadline
    if deadline is None:
        return {"date": None, "status": "none", "days_from_today": None}
    days = (deadline - today).days
    status = "passed" if days < 0 else "today" if days == 0 else "upcoming"
    return {"date": deadline.isoformat(), "status": status, "days_from_today": days}


def _case_facts(claim: Claim) -> dict:
    return {
        "case_id": claim.case_id,
        "case_type": claim.case_type,
        "created_at": claim.created_at.isoformat(),
        "status": claim.status,
        "summary": claim.summary,
        "denial_reason": claim.denial_reason,
        "documents_needed": list(claim.documents_needed),
        "appeal_deadline": claim.appeal_deadline.isoformat() if claim.appeal_deadline else None,
        "amounts": {name: str(claim.amount(name).quantize(_CENTS)) for name in AMOUNT_FIELDS},
    }


def _documents(claim: Claim, guidelines: GuidelineRepository) -> list[dict]:
    entries: list[dict] = []
    for requested in claim.documents_needed:
        guidance = guidelines.document_guidance(requested)
        entries.append(
            {
                "requested_name": requested,
                "guideline_name": guidance["guideline_name"] if guidance else None,
                "guidance": guidance["guidance"] if guidance else guidelines.default_guidance(),
                "alternative": guidance["alternative"] if guidance else guidelines.default_alternative(),
            }
        )
    return entries


def build_case_evidence(
    claim: Claim,
    guidelines: GuidelineRepository,
    schema_doc: ClaimSchemaDoc,
    *,
    today: date,
) -> EvidencePacket:
    """Everything the agent may state about one authorized claim, and nothing else."""
    return EvidencePacket(
        case=_case_facts(claim),
        deadline=_deadline(claim, today),
        today=today.isoformat(),
        field_definitions={name: spec.description for name, spec in schema_doc.field_descriptions.items()},
        documents=_documents(claim, guidelines),
        case_type_guidance=guidelines.case_type_guidance(claim.case_type),
        default_guidance=guidelines.default_guidance(),
        default_alternative=guidelines.default_alternative(),
        processing_time=guidelines.setting(PROCESSING_TIME_KEY),
        human_review_rule=guidelines.setting(HUMAN_REVIEW_KEY),
        fallback_followup=guidelines.fallback_followup(),
    )


# ---------------------------------------------------------------------------
# Grounding tokens
# ---------------------------------------------------------------------------


def _packet_texts(packet: EvidencePacket) -> Iterator[str]:
    yield packet.case.get("summary") or ""
    yield packet.case.get("denial_reason") or ""
    for document in packet.documents:
        yield document.get("guidance") or ""
        yield document.get("alternative") or ""
    yield from packet.field_definitions.values()
    for text in (
        packet.case_type_guidance,
        packet.default_guidance,
        packet.default_alternative,
        packet.processing_time,
        packet.human_review_rule,
        packet.fallback_followup,
    ):
        yield text or ""


def _iso_date(raw: object) -> date | None:
    if not isinstance(raw, str):
        return None
    try:
        return date.fromisoformat(raw)
    except ValueError:
        return None


def grounding_tokens(packet: EvidencePacket, extra_texts: Sequence[str] = ()) -> GroundingTokens:
    """Case IDs, amounts, dates, and document names the reply may cite for this packet."""
    case = packet.case
    case_ids = {cid for cid in (normalize_case_id(str(case.get("case_id", ""))),) if cid}
    amounts: set[Decimal] = set()
    for raw in (case.get("amounts") or {}).values():
        amount = _to_amount(str(raw), None)
        if amount is not None:
            amounts.add(amount)
    dates = {d for d in (_iso_date(case.get("created_at")), _iso_date(case.get("appeal_deadline"))) if d}
    documents: set[str] = set()
    for document in packet.documents:
        for key in ("requested_name", "guideline_name"):
            if isinstance(document.get(key), str) and document[key].strip():
                documents.add(document[key].strip().lower())
    for text in (*_packet_texts(packet), *(t for t in extra_texts if isinstance(t, str))):
        amounts |= find_amounts(text)
        dates |= find_dates(text)
    return GroundingTokens(
        case_ids=frozenset(case_ids),
        amounts=frozenset(amounts),
        dates=frozenset(dates),
        documents=frozenset(documents),
    )


# ---------------------------------------------------------------------------
# Pre-verification leak tokens
# ---------------------------------------------------------------------------


def _amount_forms(raw: str) -> list[str]:
    amount = _to_amount(raw, None)
    if amount is None or amount == 0:
        return []
    whole = int(amount)
    forms = [f"{amount:.2f}", f"{amount:,.2f}"]
    if amount == whole:
        forms.extend([str(whole), f"{whole:,}"])
    return forms


def _date_forms(value: date) -> list[str]:
    """ISO, month-name, and US numeric spellings of one date.

    The year-less ``March 18`` form is only emitted for two-digit days: for
    single-digit days it would be a substring of unrelated text (``march 1`` is
    inside ``march 15, 1985``), so ``march 1st`` is emitted instead.
    """
    month = _MONTH_NAMES[value.month - 1]
    forms = [
        value.isoformat(),
        f"{month} {value.day}, {value.year}",
        f"{month} {value.day} {value.year}",
        f"{value.month}/{value.day}/{value.year}",
        f"{value.month:02d}/{value.day:02d}/{value.year}",
    ]
    forms.append(f"{month} {value.day}" if value.day >= 10 else f"{month} {value.day}{_ordinal(value.day)}")
    return forms


def _ordinal(day: int) -> str:
    return {1: "st", 2: "nd", 3: "rd"}.get(day, "th")


def _case_id_forms(case_id: str) -> list[str]:
    digits = case_id.removeprefix("CL-")
    return [case_id, f"CL{digits}", f"CL {digits}"]


def _claim_secret_forms(claim: Claim) -> Iterator[str]:
    yield from _case_id_forms(claim.case_id)
    for name in AMOUNT_FIELDS:
        yield from _amount_forms(getattr(claim, name))
    if claim.denial_reason:
        yield claim.denial_reason
    yield from claim.documents_needed
    for value in (claim.created_at, claim.appeal_deadline):
        if value is not None:
            yield from _date_forms(value)


def secret_claim_tokens(claims: Iterable[Claim]) -> list[str]:
    """Lowercased, de-duplicated strings that must never appear in a reply before verification."""
    tokens: set[str] = set()
    for claim in claims:
        for form in _claim_secret_forms(claim):
            cleaned = " ".join(form.lower().split())
            if len(cleaned) >= _MIN_SECRET_LEN:
                tokens.add(cleaned)
    return sorted(tokens)
