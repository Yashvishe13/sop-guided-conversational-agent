"""Output guardrails: decide whether a drafted reply may be sent to the caller.

The agent loop calls :meth:`ReplyGuard.check` on every draft. An empty list means the
reply is safe to send; otherwise each string names one problem, and the loop hands the
list back to the model to fix (see ``loop.MAX_GUARDRAIL_RETRIES``).

What is checked, and why:

* Every reply: no colons or em dashes (a product requirement), and no internal vocabulary
  (phase names, party IDs, "system prompt").
* Before verification: no claim facts at all (IDs, amounts, dates, denial text, documents).
* After verification, each reply must be grounded in the caller's own claims:
  - claim IDs, amounts, and dates must come from tool results (``guardrails.check_grounding``);
  - a description like "closed healthcare claim from January 2026" must match a real claim;
  - a passed appeal deadline must not be described as open, and the stated status must match;
  - the reply must not say verification is pending, or that an email was sent when it was not;
  - a claim conversation may not end without offering the email summary (POST_PROCESS).
* Required content: when code decided this reply must offer a human representative (repeated
  off-topic requests or refusals, someone acting for the policyholder, a verification lockout),
  the guard checks the reply makes that offer.
* The guard's review, for every reply that passes the checks above, in every phase: the guard
  model (``agent/guard.py``) checks each claim fact against the record, the claim tool results
  of this turn, and the caller's words (for example "the pathology report is on file" when it is
  still required), and also checks for disclosure before verification, promises or invented
  actions, out-of-scope help, and internal details. No verdict counts as a problem, so a reply is
  never sent unreviewed.

The code checks come first because they are exact and cannot be argued with; the guard covers
what only language understanding can catch.

The low-level checks live in :mod:`insurance_claims.agent.guardrails`; this module applies
them with the session's context.
"""

from __future__ import annotations

import re
from datetime import date
from typing import Any, Callable

from insurance_claims.agent.guard import Guard
from insurance_claims.agent.guardrails import (
    GroundingContext,
    check_email_status,
    check_grounding,
    check_internal_reference,
    check_pre_verification_leak,
    check_style,
)
from insurance_claims.agent.tools import TurnEffects
from insurance_claims.claims.evidence import GroundingTokens, build_case_evidence, grounding_tokens
from insurance_claims.claims.repository import ClaimRepository, GuidelineRepository
from insurance_claims.domain.models import ClaimSchemaDoc, Phase
from insurance_claims.domain.state import SessionState

_MONTHS = ("january", "february", "march", "april", "may", "june", "july", "august", "september", "october", "november", "december")
_MONTH_ALT = "|".join(_MONTHS)

# "...said verification is still needed" after the caller is already verified.
_PENDING_VERIFICATION = re.compile(
    r"\b(once|after|until|before) (?:your )?(?:identity )?verification|\bneed to verify you\b|\bbefore (?:i can )?discuss",
    re.IGNORECASE,
)
# "I've sent the summary", "it's in your inbox", "the email was delivered".
_EMAIL_CLAIMED = re.compile(
    r"\b(sent|emailed|delivered)\b[^.]{0,40}\b(summary|email|inbox)\b"
    r"|\b(summary|email)\b[^.]{0,30}\b(has been|was|is) (sent|delivered)\b",
    re.IGNORECASE,
)
# A goodbye; used to require the email offer before a claim conversation ends.
_CLOSING = re.compile(
    r"\b(take care|goodbye|have a (?:great|good|nice) (?:day|evening)|glad i could help|thanks for (?:calling|contacting))\b",
    re.IGNORECASE,
)
# "January 2026" without a day (a full date is checked by check_grounding).
_MONTH_YEAR = re.compile(rf"\b({_MONTH_ALT})(?!\s+\d{{1,2}}(?:st|nd|rd|th)?\b)(?:\s+of)?,?\s+(20\d\d)\b", re.IGNORECASE)
# "closed healthcare claim from January 28, 2025": status, type, month, and year must match one claim.
_CLAIM_DESC = re.compile(
    rf"\b(open|closed|denied)?\s*(auto|healthcare|health|medical|dental)?\s*claim\s+from\s+({_MONTH_ALT})"
    r"(?:\s+\d{1,2}(?:st|nd|rd|th)?)?,?\s+(20\d\d)\b",
    re.IGNORECASE,
)
_TYPE_ALIASES = {"health": "healthcare", "medical": "healthcare"}


# What the application requires a reply to contain when a human offer is due (see TurnEffects.human_offer_due).
_HUMAN_OFFER_REQUIREMENT = (
    "Offer to connect the caller with a human claims representative, or tell them one has been asked to follow up "
    "if application_state shows a human follow-up was requested."
)


class ReplyGuard:
    def __init__(
        self,
        *,
        claims: ClaimRepository,
        guidelines: GuidelineRepository,
        schema_doc: ClaimSchemaDoc,
        secret_tokens: list[str],
        today: Callable[[], date],
        guard: Guard,
        followup_rules: tuple[Any, ...] = (),
    ) -> None:
        self.claims = claims
        self.guidelines = guidelines
        self.schema_doc = schema_doc
        self.secret_tokens = secret_tokens
        self.today = today
        self.guard = guard
        self.followup_rules = followup_rules

    def check(self, text: str, state: SessionState, effects: TurnEffects) -> list[str]:
        """Return the problems with ``text`` (empty when the reply may be sent)."""
        if not text.strip():
            return ["the reply was empty"]
        problems = self._code_checks(text, state, effects)
        return problems or self._guard_review(text, state, effects)

    def _guard_review(self, text: str, state: SessionState, effects: TurnEffects) -> list[str]:
        transcript = [
            {"speaker": "caller" if m.role == "user" else "representative", "text": m.text} for m in state.messages_since_reset()[-12:]
        ]
        verdict = self.guard.judge_reply(
            transcript=transcript,
            reply=text,
            verified=state.verified,
            record=self._record(state),
            tool_results=effects.tool_results,
            application_state=self._application_state(state, effects),
            required_in_reply=[_HUMAN_OFFER_REQUIREMENT] if effects.human_offer_due else [],
        )
        if verdict is None:
            return ["reply_unreviewed: the reply could not be reviewed; write it again"]
        if verdict.allowed:
            return []
        if not verdict.problems:
            return ["guard_blocked: the reply was not allowed; keep to the claim record and the caller's words"]
        return [f"{p.category}: {p.detail}" for p in verdict.problems]

    @staticmethod
    def _application_state(state: SessionState, effects: TurnEffects) -> dict[str, Any]:
        """Actions the application has really taken, so the guard can tell a true statement from an invented one."""
        email = state.email
        return {
            "human_follow_up_requested": state.handoff.requested,
            "human_offered": state.handoff.offered,
            "email_summary": {
                "status": email.status,
                "prepared_and_shown_with_send_and_skip_buttons": email.status == "offered",
                "goes_to": "the email address on file",
                "decision_recorded_this_turn": effects.email_decision,
            },
        }

    def _guidance(self, packet: Any, claim: Any) -> list[str]:
        """Every approved guidance text for the selected claim; statements based on them are supported."""
        texts = [d.get(k) or "" for d in packet.documents for k in ("guidance", "alternative")]
        texts += [packet.case_type_guidance or "", packet.default_guidance, packet.default_alternative or ""]
        texts += [packet.processing_time or "", packet.human_review_rule or "", packet.fallback_followup]
        texts += [self.guidelines.render_followup(rule, claim) for rule in self.followup_rules]
        return [t for t in texts if t]

    def _record(self, state: SessionState) -> dict[str, Any] | None:
        """What the guard may treat as true: nothing before verification; afterwards the caller's claims."""
        if not state.verified:
            return None
        claims = self.claims.list_for_party(state.verification.party_id or "")
        record: dict[str, Any] = {
            "today": self.today().isoformat(),
            "claims_on_account": [
                {"case_id": c.case_id, "case_type": c.case_type, "created_at": c.created_at.isoformat(), "status": c.status} for c in claims
            ],
            "selected_claim": None,
        }
        selected = next((c for c in claims if c.case_id == state.case.case_id), None)
        if selected is not None:
            packet = build_case_evidence(selected, self.guidelines, self.schema_doc, today=self.today())
            recorded = state.documents.get(selected.case_id, {})
            record["selected_claim"] = {
                **packet.case,
                "appeal_deadline_status": packet.deadline,
                "document_status": {doc: recorded.get(doc, "unknown") for doc in selected.documents_needed},
                "approved_guidance": self._guidance(packet, selected),
            }
        return record

    def _code_checks(self, text: str, state: SessionState, effects: TurnEffects) -> list[str]:
        problems = [f"{v.code}: {v.detail}" for v in check_style(text) + check_internal_reference(text)]
        if self._skipped_verification(state, effects):
            problems.append(
                "missed_verification: the caller has already given three identity details; "
                "call verify_identity with every detail they gave before replying, instead of asking for more"
            )
        if not state.verified:
            if check_pre_verification_leak(text, self.secret_tokens):
                problems.append("it contains claim information but the caller is not verified")
            return problems

        if state.phase != Phase.VERIFY_ID and _PENDING_VERIFICATION.search(text):
            problems.append("it says verification is still pending, but the caller is already verified")
        if self._ends_without_email_offer(text, state, effects):
            problems.append("the conversation is ending but the email summary was never offered; call offer_email_summary first, then ask")
        if state.email.status != "sent" and _EMAIL_CLAIMED.search(text):
            problems.append("it says the email was sent, but it has not been sent")

        ctx = self._grounding_context(state, effects)
        problems += self._claim_description_problems(text, state)
        problems += self._month_year_problems(text, ctx)
        problems += [f"{v.code}: {v.detail}" for v in check_grounding(text, ctx)]
        problems += [
            f"{v.code}: {v.detail}" for v in check_email_status(text, state.email.status if state.email.status != "none" else None)
        ]
        return problems

    # ------------------------------------------------------------------ individual rules
    @staticmethod
    def _skipped_verification(state: SessionState, effects: TurnEffects) -> bool:
        """The guard heard three identity details from the account holder, but the agent never tried to verify."""
        review = effects.caller_review
        if review is None or state.verified or state.phase != Phase.VERIFY_ID or "verify_identity" in effects.tools_called:
            return False
        if state.verification.status == "locked" or state.verification.representative_declared or review.speaker != "account_holder":
            return False
        return len(set(review.identity_fields_given) - {"policy_number"}) >= 3

    @staticmethod
    def _ends_without_email_offer(text: str, state: SessionState, effects: TurnEffects) -> bool:
        return (
            state.phase == Phase.PROCESS_CASE
            and state.case.case_id is not None
            and state.email.status == "none"
            and not effects.email_offered
            and bool(_CLOSING.search(text))
        )

    def _claim_description_problems(self, text: str, state: SessionState) -> list[str]:
        party_claims = self.claims.list_for_party(state.verification.party_id or "")
        problems = []
        for m in _CLAIM_DESC.finditer(text):
            status = (m.group(1) or "").lower()
            ctype = _TYPE_ALIASES.get((m.group(2) or "").lower(), (m.group(2) or "").lower())
            month, year = _MONTHS.index(m.group(3).lower()) + 1, int(m.group(4))
            matches = any(
                c.created_at.month == month
                and c.created_at.year == year
                and (not status or c.status == status)
                and (not ctype or c.case_type == ctype)
                for c in party_claims
            )
            if not matches:
                problems.append(f"unsupported_claim_description: no {m.group(0).strip()} on this account")
        return problems

    def _month_year_problems(self, text: str, ctx: GroundingContext) -> list[str]:
        known = {(d.month, d.year) for d in ctx.tokens.dates} | {(self.today().month, self.today().year)}
        return [
            f"unsupported_date: {m.group(0)} does not match any claim date"
            for m in _MONTH_YEAR.finditer(text)
            if (_MONTHS.index(m.group(1).lower()) + 1, int(m.group(2))) not in known
        ]

    def _grounding_context(self, state: SessionState, effects: TurnEffects) -> GroundingContext:
        """Facts the reply may use: the caller's claims, plus everything tools returned this turn."""
        claims = self.claims.list_for_party(state.verification.party_id or "")
        selected = next((c for c in claims if c.case_id == state.case.case_id), None)
        packets = list(effects.evidence)
        if selected is not None:
            packets.append(build_case_evidence(selected, self.guidelines, self.schema_doc, today=self.today()))
        tokens = GroundingTokens(
            case_ids=frozenset(), amounts=frozenset(), dates=frozenset(c.created_at for c in claims), documents=frozenset()
        )
        for packet in packets:
            t = grounding_tokens(packet)
            tokens = GroundingTokens(
                tokens.case_ids | t.case_ids, tokens.amounts | t.amounts, tokens.dates | t.dates, tokens.documents | t.documents
            )
        return GroundingContext(
            tokens=tokens,
            allowed_case_ids=frozenset(c.case_id for c in claims),
            today=self.today(),
            deadline=selected.appeal_deadline if selected else None,
            deadline_passed=bool(selected and selected.appeal_deadline and selected.appeal_deadline < self.today()),
            case_status=selected.status if selected else None,
        )
