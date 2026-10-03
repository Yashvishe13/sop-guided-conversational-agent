"""Durable per-session state (the checkpoint).

``SessionState`` is the single object persisted after every accepted turn. It is
serialized to JSON, encrypted, and stored in SQLite by
:mod:`insurance_claims.persistence.store`, together with an optimistic ``version``
counter kept in the database row.

Who may change what:

* The agent's tools (:mod:`insurance_claims.agent.tools`) are the only writers of
  the protected fields: ``phase``, ``verification``, ``case``, ``email``. Model
  output never maps onto them directly; a tool call is a proposal that code checks.
* ``hints`` is untrusted conversational memory (what the caller said they are
  calling about). It can be captured before verification and is only used after.

Raw PII policy: identity values the caller typed are not stored as fields. The
verification engine keeps only field names and the party IDs each latest value
matched (``field_matches``). The chat ``history`` necessarily contains what the
caller typed; it lives in the encrypted blob and is only returned to the cookie
holder of that session.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from insurance_claims.domain.models import PHASE_ORDER, CallerRole, Phase

STATE_SCHEMA_VERSION = 3
"""Bump when a field is added, renamed, or removed, and add a migration in persistence.store."""


class _Model(BaseModel):
    # extra="forbid": a typo or a stale field fails loudly instead of being silently ignored.
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class ChatMessage(_Model):
    role: Literal["user", "assistant"]
    text: str
    turn_index: int
    at: datetime
    phase: Phase
    kind: Literal["chat", "email_offer", "notice"] = "chat"
    """``email_offer`` renders with Send/Skip buttons; ``notice`` is a system message (e.g. delivery result)."""


class PhaseTransition(_Model):
    """Audit record of one phase change. ``reason`` is a machine code, e.g. ``identity_verified``."""

    turn_index: int
    from_phase: Phase
    to_phase: Phase
    reason: str
    at: datetime


class VerificationState(_Model):
    status: Literal["unverified", "verified", "locked"] = "unverified"
    party_id: str | None = None
    verified_at: datetime | None = None
    field_matches: dict[str, list[str]] = Field(default_factory=dict)
    """Field name -> sorted party IDs that the caller's latest value for that field matched.
    An empty list means the latest value matched nobody."""
    failed_attempts: int = 0
    last_failed_signature: str | None = None
    """Hash of the snapshot that last counted as a failure, so the same failure is not counted twice."""
    caller_role: CallerRole = CallerRole.UNKNOWN
    representative_declared: bool = False
    """Sticky: once a caller says they act for someone else, policyholder access is never granted."""


class CaseHints(_Model):
    """Untrusted memory of why the caller is calling, captured in any phase and used after verification."""

    case_type: str | None = None
    status: str | None = None
    month: int | None = None
    year: int | None = None
    first_turn: int | None = None
    captured_in_phase: Phase | None = None
    question_summaries: list[str] = Field(default_factory=list)
    """Short, PII-free paraphrases of the caller's reason for calling."""


class CaseSelection(_Model):
    """The claim currently being discussed (always one of the verified caller's own claims)."""

    case_id: str | None = None
    selected_turn: int | None = None


class Counters(_Model):
    off_topic_total: int = 0
    refusals: int = 0
    """Verification refusals; the agent may only stop persuading after ``Settings.max_refusals``."""


class Handoff(_Model):
    offered: bool = False
    requested: bool = False
    reason: str | None = None
    at: datetime | None = None


class EmailSummary(_Model):
    """The summary the caller was shown. ``body_text`` is exactly what is emailed if they choose Send."""

    case_id: str | None = None
    case_type: str | None = None
    status: str | None = None
    body_text: str = ""


EmailStatus = Literal[
    "none",  # nothing offered yet
    "offered",  # summary shown, waiting for an explicit choice
    "consented",  # explicit yes recorded, operation persisted, not yet dispatched
    "dispatching",  # transport call in flight (ambiguous if we crash here)
    "sent",  # real transport accepted the message
    "queued",  # written to the local demo outbox only (not delivered)
    "skipped",  # caller declined
    "failed",  # transport definitely rejected; no message left the system
    "delivery_unknown",  # cannot prove either way; routed to manual review
    "needs_human",  # recipient change or representative request; not sent
]


class EmailOffer(_Model):
    status: EmailStatus = "none"
    summary: EmailSummary | None = None
    summary_hash: str | None = None
    offered_turn: int | None = None
    op_key: str | None = None
    """Idempotency key of the consented send (persisted before dispatch)."""
    decided_turn: int | None = None
    receipt_id: str | None = None
    detail: str | None = None
    send_count: int = 0
    """Number of distinct consented sends in this session (each needs its own consent)."""


class SessionState(_Model):
    schema_version: int = STATE_SCHEMA_VERSION
    session_id: str
    phase: Phase = Phase.VERIFY_ID
    turn_index: int = 0
    created_at: datetime
    last_activity_at: datetime

    verification: VerificationState = Field(default_factory=VerificationState)
    hints: CaseHints = Field(default_factory=CaseHints)
    case: CaseSelection = Field(default_factory=CaseSelection)
    counters: Counters = Field(default_factory=Counters)
    handoff: Handoff = Field(default_factory=Handoff)
    email: EmailOffer = Field(default_factory=EmailOffer)
    documents: dict[str, dict[str, str]] = Field(default_factory=dict)
    """Per claim: what the caller said about each required document (has_it, can_request, ...)."""

    history: list[ChatMessage] = Field(default_factory=list)
    phase_log: list[PhaseTransition] = Field(default_factory=list)
    closed: bool = False

    @property
    def verified(self) -> bool:
        return self.verification.status == "verified" and self.verification.party_id is not None

    def public_view(self) -> dict[str, Any]:
        """Browser-safe projection. Never includes party IDs, match state, or claim data before verification."""
        current = PHASE_ORDER.index(self.phase)
        steps = []
        for idx, phase in enumerate(PHASE_ORDER):
            if idx < current:
                status = "done"
            elif idx == current:
                # The wrap-up step counts as done once the caller has made their email choice.
                status = "done" if (phase is Phase.POST_PROCESS and self.email.status in _EMAIL_FINAL) else "active"
            else:
                status = "pending"
            steps.append({"phase": phase.value, "status": status})
        view: dict[str, Any] = {
            "phase": self.phase.value,
            "steps": steps,
            "verified": self.verified,
            "verification_locked": self.verification.status == "locked",
            "handoff": {"offered": self.handoff.offered, "requested": self.handoff.requested},
            "email": {"status": self.email.status, "offer_active": self.email.status == "offered"},
            "turn_index": self.turn_index,
            "closed": self.closed,
        }
        if self.verified and self.case.case_id:
            view["case"] = {"case_id": self.case.case_id}
        return view


_EMAIL_FINAL = frozenset({"sent", "queued", "skipped", "failed", "delivery_unknown", "needs_human"})
