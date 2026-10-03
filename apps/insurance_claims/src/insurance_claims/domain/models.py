"""Domain vocabulary and fixture record models shared by every package.

* Workflow vocabulary: the four SOP phases and the identity fields that count
  toward verification.
* Fixture records: ``Policyholder``, ``Claim``, ``Representative``, and the
  document guidelines. ``claims.fixtures`` validates the JSON fixtures into these
  models at startup, so the rest of the code works with typed, trusted data.
"""

from __future__ import annotations

from datetime import date
from decimal import Decimal
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

# ---------------------------------------------------------------------------
# Workflow vocabulary
# ---------------------------------------------------------------------------


class Phase(StrEnum):
    VERIFY_ID = "VERIFY_ID"
    RESOLVE_INTENT = "RESOLVE_INTENT"
    PROCESS_CASE = "PROCESS_CASE"
    POST_PROCESS = "POST_PROCESS"


PHASE_ORDER: tuple[Phase, ...] = (
    Phase.VERIFY_ID,
    Phase.RESOLVE_INTENT,
    Phase.PROCESS_CASE,
    Phase.POST_PROCESS,
)


class IdentityField(StrEnum):
    """The five PII fields that count toward verification."""

    FULL_NAME = "full_name"
    DOB = "dob"
    PHONE = "phone"
    EMAIL = "email"
    ID_LAST4 = "id_last4"


POLICY_NUMBER = "policy_number"
"""Locator field: helps find a candidate, never counts toward the three PII matches."""

PII_FIELDS: tuple[str, ...] = tuple(f.value for f in IdentityField)
ALL_IDENTITY_KEYS: tuple[str, ...] = (*PII_FIELDS, POLICY_NUMBER)
REQUIRED_MATCHES = 3


class CallerRole(StrEnum):
    POLICYHOLDER = "policyholder"
    REPRESENTATIVE = "representative"
    UNKNOWN = "unknown"


CASE_TYPES: tuple[str, ...] = ("healthcare", "dental", "auto", "other")

# ---------------------------------------------------------------------------
# Fixture records
# ---------------------------------------------------------------------------

_AMOUNT_PATTERN = r"^\d{1,9}\.\d{2}$"


class Policyholder(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)

    party_id: str = Field(min_length=1, max_length=32)
    name: str = Field(min_length=1, max_length=120)
    name_aliases: list[str] = Field(default_factory=list)
    policy_number: str = Field(min_length=1, max_length=32)
    dob: date
    id_type: Literal["ssn_last4", "national_id_last4"]
    id_last4: str = Field(pattern=r"^\d{4}$")
    phone: str = Field(min_length=7, max_length=24)
    phone_aliases: list[str] = Field(default_factory=list)
    email: str = Field(min_length=3, max_length=254)
    email_aliases: list[str] = Field(default_factory=list)

    @field_validator("email", "email_aliases")
    @classmethod
    def _email_shape(cls, value: str | list[str]) -> str | list[str]:
        values = value if isinstance(value, list) else [value]
        for item in values:
            if "@" not in item or item.startswith("@") or item.endswith("@"):
                raise ValueError("invalid email in fixture")
        return value


class Claim(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)

    case_id: str = Field(pattern=r"^CL-\d{3,8}$")
    party_id: str = Field(min_length=1, max_length=32)
    case_type: str = Field(min_length=1, max_length=32)
    created_at: date
    status: str = Field(min_length=1, max_length=32)
    summary: str = Field(default="", max_length=2000)
    denial_reason: str | None = Field(default=None, max_length=2000)
    documents_needed: list[str] = Field(default_factory=list)
    appeal_deadline: date | None = None
    expected_reimbursement_amount: str = Field(pattern=_AMOUNT_PATTERN)
    allowed_max_amount: str = Field(pattern=_AMOUNT_PATTERN)
    net_pay: str = Field(pattern=_AMOUNT_PATTERN)
    net_fee: str = Field(pattern=_AMOUNT_PATTERN)

    @field_validator("case_type", "status")
    @classmethod
    def _lower(cls, value: str) -> str:
        return value.strip().lower()

    def amount(self, field_name: str) -> Decimal:
        """Parse a fixture amount string with Decimal (never float)."""
        return Decimal(getattr(self, field_name))


class Representative(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)

    rep_name: str
    relationship: str
    buyer_name: str
    buyer_party_id: str


class LocalizedText(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)
    en: str


class FollowupRule(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)

    """Approved follow-up guidance for one topic (looked up by ``get_followup_guidance``)."""

    topic: str
    requires_documents: bool = False
    en: str


class DocumentGuidelines(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)

    default_guidance: LocalizedText
    case_type_guidance: dict[str, LocalizedText] = Field(default_factory=dict)
    document_guidance: dict[str, LocalizedText] = Field(default_factory=dict)
    document_alternative_guidance: dict[str, LocalizedText] = Field(default_factory=dict)
    claim_followup_settings: dict[str, LocalizedText] = Field(default_factory=dict)
    claim_followup_guidance: list[FollowupRule] = Field(default_factory=list)
    claim_followup_fallback: LocalizedText


class FieldDescription(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)
    type: str = "string"
    example: str | None = None
    description: str


class ClaimSchemaDoc(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)
    notes: list[str] = Field(default_factory=list)
    field_descriptions: dict[str, FieldDescription] = Field(default_factory=dict)


class FixtureIssue(BaseModel):
    """A fixture row that failed validation and was quarantined (not loaded)."""

    file: str
    index: int
    error: str
