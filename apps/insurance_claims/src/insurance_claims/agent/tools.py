"""Tools for the claims ReAct agent, with guardrails enforced at execution.

The model decides *when* to call a tool and with what arguments. Every call is a
proposal: the executor checks it against trusted session state and fixture data
before anything changes. Phase transitions happen only here, as the side effect
of a tool succeeding:

* ``verify_identity`` succeeding        -> VERIFY_ID      -> RESOLVE_INTENT
* ``select_claim`` succeeding           -> RESOLVE_INTENT -> PROCESS_CASE
* ``offer_email_summary`` accepted      -> PROCESS_CASE   -> POST_PROCESS
* any claim tool after the offer        -> POST_PROCESS   -> PROCESS_CASE

Which tools the model can see depends on the phase (``tool_schemas``), and the
executor re-checks the phase anyway, so a hallucinated or out-of-phase call is
refused with an explanation the model can act on.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field, replace
from datetime import date, datetime, timedelta
from typing import Any, Callable

from insurance_claims.agent.guardrails import check_style
from insurance_claims.claims.evidence import EvidencePacket, build_case_evidence
from insurance_claims.claims.normalize import (
    MONTHS,
    identity_value_grounded,
    normalize_case_id,
    normalize_dob,
    normalize_email,
    normalize_last4,
    normalize_name,
    normalize_phone,
    normalize_policy_number,
)
from insurance_claims.claims.repository import (
    ClaimRepository,
    GuidelineRepository,
    PolicyholderDirectory,
)
from insurance_claims.claims.verification import apply_identity_proposals
from insurance_claims.domain.models import PII_FIELDS, POLICY_NUMBER, Claim, Phase
from insurance_claims.domain.state import (
    CaseSelection,
    EmailOffer,
    EmailSummary,
    PhaseTransition,
    SessionState,
)
from insurance_claims.observability import tracing

MAX_ARG_CHARS = 2000
REQUIRED_PII = 3
"""Distinct identity fields (policy number excluded) that must match before claim tools unlock."""
DOC_STATUSES = ("has_it", "can_request", "cannot_obtain", "already_sent", "unknown")
FOLLOWUP_TOPICS = (
    "missing_required_material_alternatives",
    "submission_timing",
    "processing_time_after_submission",
    "submission_method",
    "file_format_requirements",
    "receipt_confirmation",
)
CASE_TYPES = ("healthcare", "dental", "auto", "other")
STATUSES = ("denied", "open", "closed")


def _dob_from_phrase(raw: str) -> str | None:
    """Accept a bare date or a date inside a phrase ("Born March 15, 1985", "15 de marzo de 1985")."""
    if (d := normalize_dob(raw)) is not None:
        return d.isoformat()
    text = raw.strip()
    for start in range(len(text)):
        for end in range(len(text), start, -1):
            piece = text[start:end].strip(" ,.;")
            if len(piece) >= 6 and piece[:1].isalnum() and (d := normalize_dob(piece)) is not None:
                return d.isoformat()
    spanish = re.search(r"(\d{1,2})\s+de\s+([a-z]+)\s+de\s+(\d{4})", text.lower())
    if spanish and spanish.group(2) in MONTHS:
        try:
            return date(int(spanish.group(3)), MONTHS[spanish.group(2)], int(spanish.group(1))).isoformat()
        except ValueError:
            return None
    return None


# How the model should re-ask when a value it submitted was not grounded or not parseable.
_HOW_TO_FIX = {
    "dob": "ask the caller to restate the full date of birth with a four digit year, for example March 15, 1985",
    "phone": "ask for the full 10 digit phone number",
    "id_last4": "ask for exactly four digits",
    "full_name": "ask for first and last name exactly as on the policy",
    "email": "ask the caller to type the email address",
    "policy_number": "policy number is optional",
}

# Turn a value the model submitted into the canonical form used for matching (None = unusable).
_NORMALIZERS: dict[str, Callable[[str], Any]] = {
    "full_name": normalize_name,
    "dob": _dob_from_phrase,
    "phone": normalize_phone,
    "email": normalize_email,
    "id_last4": normalize_last4,
    "policy_number": normalize_policy_number,
}


# ---------------------------------------------------------------------- tool schemas (strict JSON schema)
def _nullable(kind: str, desc: str, enum: tuple[str, ...] | None = None) -> dict[str, Any]:
    out: dict[str, Any] = {"type": [kind, "null"], "description": desc}
    if enum:
        out["enum"] = [*enum, None]
    return out


def _fn(name: str, desc: str, props: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "function",
        "name": name,
        "description": desc,
        "parameters": {"type": "object", "properties": props, "required": list(props), "additionalProperties": False},
        "strict": True,
    }


SCHEMAS: dict[str, dict[str, Any]] = {
    "note_caller_context": _fn(
        "note_caller_context",
        "Remember why the caller is calling (any phase, even before verification), e.g. 'denied healthcare claim from January'. "
        "Call it as soon as the caller mentions it. Hints are used after verification to find the claim.",
        {
            "reason": _nullable("string", "Short paraphrase of the caller's reason for calling, no personal data."),
            "case_type": _nullable("string", "Claim type if mentioned.", CASE_TYPES),
            "status": _nullable("string", "Claim status if mentioned.", STATUSES),
            "month": _nullable("integer", "Month 1-12 the claim is from, if mentioned."),
            "year": _nullable("integer", "Year the claim is from, if mentioned."),
        },
    ),
    "verify_identity": _fn(
        "verify_identity",
        "Check the caller's identity. Submit every identity detail the caller has stated anywhere in this conversation (values only, normalized as described per field). "
        "At least three of full name, date of birth, phone, email, and last four digits of SSN or national ID are required; "
        "a policy number helps but does not count. Returns verified, need_more_information, not_matched, locked, or not_allowed.",
        {
            "full_name": _nullable("string", "First and last name as stated."),
            "dob": _nullable(
                "string", "Date of birth normalized to YYYY-MM-DD (for a two digit year, use the past century, e.g. 85 -> 1985)."
            ),
            "phone": _nullable("string", "Phone number as stated."),
            "email": _nullable("string", "Email as stated."),
            "id_last4": _nullable("string", "Exactly the four digits, e.g. 4472."),
            "policy_number": _nullable("string", "Policy number such as POL-1234."),
            "caller_is_policyholder": {
                "type": "boolean",
                "description": "False if the caller is acting for someone else (a relative, friend, or agent).",
            },
        },
    ),
    "list_my_claims": _fn("list_my_claims", "List the verified caller's claims (type, date, status, claim ID).", {}),
    "select_claim": _fn(
        "select_claim",
        "Select the claim the caller is asking about (must be one of their claims). Required before discussing claim details.",
        {"case_id": {"type": "string", "description": "Claim ID from list_my_claims, e.g. CL-2048."}},
    ),
    "get_claim_details": _fn(
        "get_claim_details",
        "Full facts of the selected claim: status, denial reason, documents still needed, amounts with definitions, appeal deadline and whether it has passed, today's date.",
        {},
    ),
    "get_document_guidance": _fn(
        "get_document_guidance",
        "How to prepare and submit one required document, and what to do if the caller cannot get it.",
        {"document": {"type": "string", "description": "A document listed on the selected claim."}},
    ),
    "get_followup_guidance": _fn(
        "get_followup_guidance",
        "Approved guidance for the selected claim on one follow-up topic.",
        {"topic": {"type": "string", "enum": list(FOLLOWUP_TOPICS)}},
    ),
    "record_document_status": _fn(
        "record_document_status",
        "Record what the caller says about a required document (has_it, can_request, cannot_obtain, already_sent, unknown).",
        {
            "document": {"type": "string", "description": "A document listed on the selected claim."},
            "status": {"type": "string", "enum": list(DOC_STATUSES)},
        },
    ),
    "offer_email_summary": _fn(
        "offer_email_summary",
        "When the caller's claim has been fully covered, offer to email them a summary. Write the summary body: what was discussed, "
        "the claim status or outcome, and the follow-up items. Plain text, no colons, no em dashes. The app shows it with Send and Skip buttons.",
        {"summary": {"type": "string", "description": "The email body, 3 to 7 sentences, signed Claims Support Team."}},
    ),
    "record_email_decision": _fn(
        "record_email_decision",
        "Record the caller's explicit answer to the email offer. Only call it when the caller clearly said to send or to skip.",
        {"decision": {"type": "string", "enum": ["send", "skip"]}},
    ),
    "report_caller_change": _fn(
        "report_caller_change",
        "Call when, after verification, the person speaking appears to be someone else (a different name, or acting for someone else). "
        "Access is withdrawn and the new caller must verify. Do not share anything more about the previous caller's claims.",
        {"acting_for_someone_else": {"type": "boolean", "description": "True if the new speaker says they act for the policyholder."}},
    ),
    "flag_off_topic": _fn("flag_off_topic", "Record that the caller asked something unrelated to insurance claims.", {}),
    "note_refusal": _fn(
        "note_refusal",
        "Call whenever the caller refuses or resists verification (e.g. 'I already told you who I am', 'I'm not giving you that'). "
        "The result tells you whether to keep persuading (explain why and offer alternative details) or to stop and offer a human.",
        {},
    ),
    "request_human": _fn(
        "request_human",
        "Hand the conversation to a human representative.",
        {
            "reason": {
                "type": "string",
                "enum": [
                    "caller_asked",
                    "repeated_refusal",
                    "verification_locked",
                    "representative_authorization",
                    "manual_review",
                    "new_claim",
                ],
            },
        },
    ),
}

# ---------------------------------------------------------------------- which tools each phase exposes
ALWAYS = ("note_caller_context", "flag_off_topic", "request_human")
VERIFY_ONLY = ("note_refusal",)
BY_PHASE: dict[Phase, tuple[str, ...]] = {
    Phase.VERIFY_ID: ("verify_identity",),
    Phase.RESOLVE_INTENT: ("list_my_claims", "select_claim", "report_caller_change"),
    Phase.PROCESS_CASE: (
        "list_my_claims",
        "select_claim",
        "get_claim_details",
        "get_document_guidance",
        "get_followup_guidance",
        "record_document_status",
        "offer_email_summary",
        "report_caller_change",
    ),
    Phase.POST_PROCESS: (
        "list_my_claims",
        "select_claim",
        "get_claim_details",
        "get_document_guidance",
        "get_followup_guidance",
        "record_email_decision",
        "offer_email_summary",
        "report_caller_change",
    ),
}


def tool_names(state: SessionState) -> tuple[str, ...]:
    """The tool menu for the current state. The executor re-checks it on every call."""
    names = list(BY_PHASE[state.phase]) + list(ALWAYS) + (list(VERIFY_ONLY) if state.phase == Phase.VERIFY_ID else [])
    if state.phase == Phase.POST_PROCESS and state.email.status != "offered":
        names.remove("record_email_decision")
    if state.phase == Phase.POST_PROCESS and state.email.status in ("offered", "consented", "dispatching", "sent", "queued"):
        names.remove("offer_email_summary")  # re-offer only after a skip or a failed/blocked send
    return tuple(names)


def tool_schemas(state: SessionState) -> list[dict[str, Any]]:
    return [SCHEMAS[n] for n in tool_names(state)]


# ---------------------------------------------------------------------- execution
@dataclass
class ToolResult:
    """What the model sees as the function_call_output. ``ok=False`` results explain how to proceed."""

    name: str
    ok: bool
    output: dict[str, Any]

    def to_json(self) -> str:
        return json.dumps(self.output, ensure_ascii=False, default=str)


@dataclass
class TurnEffects:
    """What the tools did this turn (read by the agent after the loop)."""

    email_offered: bool = False
    email_decision: str | None = None
    evidence: list[EvidencePacket] = field(default_factory=list)
    fetched_case_ids: set[str] = field(default_factory=set)
    claims_listed: bool = False


class ToolExecutor:
    """Runs the model's tool calls for one turn against the session state.

    One executor is created per turn. Each ``_<tool_name>`` method validates its arguments,
    changes ``state`` only if the guardrails allow it, and returns a ToolResult.
    """

    def __init__(
        self,
        *,
        state: SessionState,
        now: datetime,
        today: date,
        caller_text: str,
        directory: PolicyholderDirectory,
        claims: ClaimRepository,
        guidelines: GuidelineRepository,
        schema_doc: Any,
        followup_rules: tuple[Any, ...],
        settings: Any,
        failure_ledger: Any | None,
    ) -> None:
        self.state = state
        self.now = now
        self.today = today
        self.caller_text = caller_text
        self.directory = directory
        self.claims = claims
        self.guidelines = guidelines
        self.schema_doc = schema_doc
        self.followup_rules = followup_rules
        self.settings = settings
        self.failure_ledger = failure_ledger
        self.effects = TurnEffects()
        self.calls: list[dict[str, Any]] = []
        self._seen: dict[str, int] = {}

    # ------------------------------------------------------------------ dispatch
    def execute(self, name: str, arguments: str) -> ToolResult:
        with tracing.span(f"tool.{name}" if name in SCHEMAS else "tool.blocked", type="tool") as node:
            result = self._dispatch(name, arguments)
            node["output"] = {"ok": result.ok, "status": result.output.get("status") or result.output.get("error")}
            node["metadata"] = {"phase": self.state.phase.value}
            self.calls.append({"name": name, "ok": result.ok, "status": node["output"]["status"]})
            return result

    def _dispatch(self, name: str, arguments: str) -> ToolResult:
        if name not in SCHEMAS:
            tracing.event("guardrail", rule="unknown_tool", tool=name[:60])
            return ToolResult(name, False, {"error": "unknown_tool", "message": "That tool does not exist."})
        if name not in tool_names(self.state):
            tracing.event("guardrail", rule="tool_not_allowed_in_phase", tool=name, phase=self.state.phase.value)
            return ToolResult(name, False, {"error": "not_allowed_now", "message": self._not_allowed_message(name)})
        if not isinstance(arguments, str) or len(arguments) > MAX_ARG_CHARS:
            return ToolResult(name, False, {"error": "invalid_arguments"})
        try:
            args = json.loads(arguments or "{}")
        except json.JSONDecodeError:
            return ToolResult(name, False, {"error": "invalid_arguments", "message": "Arguments must be JSON."})
        if not isinstance(args, dict):
            return ToolResult(name, False, {"error": "invalid_arguments"})
        # The same call twice in one turn is a loop; verify_identity is exempt (the caller may add details).
        key = f"{name}:{json.dumps(args, sort_keys=True)}"
        self._seen[key] = self._seen.get(key, 0) + 1
        if self._seen[key] > 1 and name not in ("verify_identity",):
            return ToolResult(name, False, {"error": "repeated_call", "message": "You already have this result. Reply to the caller now."})
        try:
            return getattr(self, f"_{name}")(args)
        except Exception as exc:  # a tool failure is a result, never a crash
            tracing.event("tool_error", tool=name, error=exc.__class__.__name__)
            return ToolResult(
                name, False, {"error": "tool_error", "message": "That lookup failed. Apologise and offer a human representative if needed."}
            )

    def _not_allowed_message(self, name: str) -> str:
        if self.state.phase == Phase.VERIFY_ID:
            return "The caller is not verified. Claim tools are unavailable until verify_identity returns verified. Do not share or hint at any claim information."
        if name == "offer_email_summary":
            return "Select the claim and cover it before offering the email summary."
        if name == "record_email_decision":
            return "There is no active email offer."
        return "That tool is not available in this step."

    def _transition(self, to: Phase, reason: str) -> None:
        if self.state.phase == to:
            return
        tracing.event("phase_transition", from_phase=self.state.phase.value, to_phase=to.value, reason=reason)
        self.state.phase_log.append(
            PhaseTransition(turn_index=self.state.turn_index, from_phase=self.state.phase, to_phase=to, reason=reason, at=self.now)
        )
        self.state.phase_log = self.state.phase_log[-50:]
        self.state.phase = to

    # ------------------------------------------------------------------ memory, scope, handoff
    def _note_caller_context(self, a: dict[str, Any]) -> ToolResult:
        h = self.state.hints
        if a.get("reason"):
            h.question_summaries = [*h.question_summaries, str(a["reason"])[:200]][-5:]
        if a.get("case_type") in CASE_TYPES:
            h.case_type = a["case_type"]
        if a.get("status") in STATUSES:
            h.status = a["status"]
        if isinstance(a.get("month"), int) and 1 <= a["month"] <= 12:
            h.month = a["month"]
        if isinstance(a.get("year"), int) and 2000 <= a["year"] <= self.today.year + 1:
            h.year = a["year"]
        if h.first_turn is None:
            h.first_turn, h.captured_in_phase = self.state.turn_index, self.state.phase
        tracing.event("memory", phase=self.state.phase.value, hints=h.model_dump(exclude={"question_summaries"}, mode="json"))
        note = (
            "Noted."
            if self.state.verified
            else "Noted. Keep it for after verification; do not discuss the claim until verify_identity returns verified."
        )
        return ToolResult("note_caller_context", True, {"status": "noted", "message": note})

    def _flag_off_topic(self, a: dict[str, Any]) -> ToolResult:
        c = self.state.counters
        c.off_topic_total += 1
        offer = c.off_topic_total >= self.settings.max_off_topic
        if offer:
            self.state.handoff.offered = True
        tracing.event("guardrail", rule="off_topic", count=c.off_topic_total, offer_human=offer)
        msg = "Politely decline in one sentence and steer back to the claim."
        if offer:
            msg += " The caller has gone off topic repeatedly: also offer to connect them with a human representative."
        return ToolResult(
            "flag_off_topic", True, {"status": "recorded", "off_topic_count": c.off_topic_total, "offer_human": offer, "message": msg}
        )

    def _report_caller_change(self, a: dict[str, Any]) -> ToolResult:
        old = self.state.verification
        self.state.verification = type(old)(
            failed_attempts=old.failed_attempts,
            representative_declared=bool(a.get("acting_for_someone_else")) or old.representative_declared,
        )
        self.state.case = CaseSelection()
        self.state.hints = type(self.state.hints)()
        self.state.documents = {}
        if self.state.email.status == "offered":
            self.state.email = EmailOffer(send_count=self.state.email.send_count)
        self._transition(Phase.VERIFY_ID, "caller_changed")
        tracing.event("guardrail", rule="caller_changed")
        return ToolResult(
            "report_caller_change",
            True,
            {
                "status": "access_withdrawn",
                "message": "Access withdrawn. Say nothing more about the previous claims. Explain that you need to verify the person you are now speaking with, or offer a human representative if they act for someone else.",
            },
        )

    def _note_refusal(self, a: dict[str, Any]) -> ToolResult:
        c = self.state.counters
        c.refusals += 1
        stop = c.refusals >= self.settings.max_refusals
        tracing.event("guardrail", rule="refusal", count=c.refusals, stop_persuading=stop)
        if stop:
            self.state.handoff.offered = True
            return ToolResult(
                "note_refusal",
                True,
                {
                    "status": "stop_persuading",
                    "refusals": c.refusals,
                    "message": "The caller has refused again. Acknowledge their feelings, stop asking for details, and offer a human representative (request_human with reason repeated_refusal if they agree or if they keep refusing).",
                },
            )
        return ToolResult(
            "note_refusal",
            True,
            {
                "status": "keep_persuading",
                "refusals": c.refusals,
                "message": "First refusal. Acknowledge the feeling, explain briefly that verification protects their private claim information, and offer the alternative details they can use (full name, date of birth, phone or email on file, last four of SSN or national ID; any three). Do not hand off yet unless they ask for a person.",
            },
        )

    def _request_human(self, a: dict[str, Any]) -> ToolResult:
        if a.get("reason") == "repeated_refusal" and self.state.counters.refusals < self.settings.max_refusals:
            tracing.event("guardrail", rule="premature_handoff", refusals=self.state.counters.refusals)
            return ToolResult(
                "request_human",
                False,
                {
                    "error": "too_early",
                    "message": "The caller has not refused repeatedly yet. Keep helping: explain why verification matters and offer the alternative identity details.",
                },
            )
        h = self.state.handoff
        h.requested = h.offered = True
        h.reason = str(a.get("reason") or "requested")[:120]
        h.at = self.now
        tracing.event("handoff", reason=h.reason)
        return ToolResult(
            "request_human",
            True,
            {
                "status": "handoff_requested",
                "message": "A human claims representative has been flagged to follow up. Tell the caller; do not promise a time.",
            },
        )

    # ------------------------------------------------------------------ VERIFY_ID
    def _verify_identity(self, a: dict[str, Any]) -> ToolResult:
        v = self.state.verification
        if a.get("caller_is_policyholder") is False:
            v.representative_declared = True
        if v.representative_declared:
            self.state.handoff.offered = True
            tracing.event("guardrail", rule="representative_not_authorized")
            return ToolResult(
                "verify_identity",
                False,
                {
                    "status": "not_allowed",
                    "message": "Claim details can only be discussed with the verified policyholder. The caller is acting for someone else: explain kindly and offer a human representative who can review authorization.",
                },
            )
        if v.status == "locked":
            return ToolResult(
                "verify_identity",
                False,
                {"status": "locked", "message": "Verification is locked for this conversation. Offer a human representative."},
            )

        # Guardrail 1: only values the caller actually said count.
        accepted: dict[str, str] = {}
        rejected: list[str] = []
        for fld in (*PII_FIELDS, POLICY_NUMBER):
            raw = a.get(fld)
            if not isinstance(raw, str) or not raw.strip():
                continue
            norm = _NORMALIZERS[fld](raw)
            if norm is None or not identity_value_grounded(fld, norm, self.caller_text):
                rejected.append(fld)
                continue
            accepted[fld] = norm if fld == "dob" else raw
        pii = [f for f in accepted if f in PII_FIELDS]
        tracing.event("guardrail", rule="identity_fields", accepted=sorted(accepted), rejected_ungrounded=rejected, pii_count=len(pii))

        # Guardrail 2: three distinct PII fields before any matching happens.
        fixes = {f: _HOW_TO_FIX[f] for f in rejected}
        if len(pii) < REQUIRED_PII:
            return ToolResult(
                "verify_identity",
                False,
                {
                    "how_to_fix_ignored_fields": fixes,
                    "status": "need_more_information",
                    "fields_received": pii,
                    "fields_still_needed": REQUIRED_PII - len(pii),
                    "ignored_fields": rejected,
                    "message": "Not enough identity details yet, so the caller is NOT verified and you must not move on. "
                    "Ask for the remaining details; accepted ones are full name, date of birth, phone or email on file, last four of SSN or national ID. "
                    "A policy number does not count. Values the caller did not actually say are ignored.",
                },
            )

        # Guardrail 3: deterministic matching against records, with lockout.
        vstate, outcome = apply_identity_proposals(
            v, accepted, self.directory, max_failures=self.settings.max_verification_failures, now=self.now
        )
        if outcome.status == "verified" and vstate.party_id and self._party_locked(vstate.party_id):
            vstate = vstate.model_copy(update={"status": "locked", "party_id": None, "verified_at": None})
            outcome = replace(outcome, status="locked")
        if outcome.attempt_counted:
            self._record_failure(vstate)
        self.state.verification = vstate
        tracing.event("verification_gate", status=outcome.status, failed_attempts=vstate.failed_attempts, reason=outcome.reason)

        if self.state.verified:
            self._transition(Phase.RESOLVE_INTENT, "identity_verified")
            party = vstate.party_id or ""
            return ToolResult(
                "verify_identity",
                True,
                {
                    "status": "verified",
                    "first_name": self.directory.first_name(party),
                    "remembered_context": self._hints(),
                    "message": "Verified. Tell the caller they are verified, then use the remembered context to find their claim with list_my_claims and select_claim instead of asking from scratch. If it clearly matches one claim, select it and continue straight into explaining it.",
                },
            )
        if vstate.status == "locked":
            self.state.handoff.offered = True
            return ToolResult(
                "verify_identity",
                False,
                {"status": "locked", "message": "Too many unsuccessful attempts. Verification is locked; offer a human representative."},
            )
        return ToolResult(
            "verify_identity",
            False,
            {
                "status": "not_matched",
                "message": "The details do not match our records. Do not say which detail failed or whether the person exists. Ask the caller to double check or offer a different detail.",
            },
        )

    def _party_locked(self, party: str) -> bool:
        """Cross-session lockout: too many recent failures against this policyholder. Fails closed."""
        counter = getattr(self.failure_ledger, "count_verification_failures", None)
        if counter is None:
            return False
        try:
            return (
                counter(f"party:{party}", self.now - timedelta(hours=self.settings.party_failure_window_hours))
                >= self.settings.max_party_failures
            )
        except Exception:
            return True

    def _record_failure(self, vstate: Any) -> None:
        """Count a failed attempt against every policyholder the submitted values partly matched."""
        recorder = getattr(self.failure_ledger, "record_verification_failure", None)
        parties = sorted({p for ps in vstate.field_matches.values() for p in ps})
        if recorder and parties:
            try:
                recorder([f"party:{p}" for p in parties], self.now)
            except Exception:
                pass

    def _hints(self) -> dict[str, Any]:
        h = self.state.hints
        return {
            k: v
            for k, v in {
                "reason": h.question_summaries[-1] if h.question_summaries else None,
                "case_type": h.case_type,
                "status": h.status,
                "month": h.month,
                "year": h.year,
            }.items()
            if v
        }

    # ------------------------------------------------------------------ claims (party-scoped)
    def _party(self) -> str:
        if not self.state.verified:
            raise PermissionError("not verified")
        return self.state.verification.party_id or ""

    def _list_my_claims(self, a: dict[str, Any]) -> ToolResult:
        claims = self.claims.list_for_party(self._party())
        self.effects.claims_listed = True
        self.effects.fetched_case_ids |= {c.case_id for c in claims}
        rows = [
            {"case_id": c.case_id, "case_type": c.case_type, "created_at": c.created_at.isoformat(), "status": c.status} for c in claims
        ]
        return ToolResult("list_my_claims", True, {"status": "ok", "claims": rows, "remembered_context": self._hints()})

    def _selected(self) -> Claim:
        if not self.state.case.case_id:
            raise PermissionError("no claim selected")
        found = self.claims.get_for_party(self._party(), self.state.case.case_id)
        if found.status != "found":
            raise PermissionError("selected claim not owned")
        return found.claims[0]

    def _select_claim(self, a: dict[str, Any]) -> ToolResult:
        case_id = normalize_case_id(str(a.get("case_id", "")))
        found = self.claims.get_for_party(self._party(), case_id or "")
        if found.status != "found":
            tracing.event("guardrail", rule="claim_not_owned")
            return ToolResult(
                "select_claim",
                False,
                {"error": "not_found", "message": "No claim with that ID on this caller's account. Use list_my_claims."},
            )
        claim = found.claims[0]
        self._back_to_case()  # a new claim question withdraws any open email offer
        if self.state.case.case_id != claim.case_id:
            self.state.case = CaseSelection(case_id=claim.case_id, selected_turn=self.state.turn_index)
        self._transition(Phase.PROCESS_CASE, "claim_selected")
        self.effects.fetched_case_ids.add(claim.case_id)
        return ToolResult(
            "select_claim",
            True,
            {
                "status": "selected",
                "case_id": claim.case_id,
                "message": "Now call get_claim_details and walk the caller through the claim.",
            },
        )

    def _packet(self, claim: Claim) -> EvidencePacket:
        packet = build_case_evidence(claim, self.guidelines, self.schema_doc, today=self.today)
        self.effects.evidence.append(packet)
        self.effects.fetched_case_ids.add(claim.case_id)
        return packet

    def _get_claim_details(self, a: dict[str, Any]) -> ToolResult:
        claim = self._selected()
        packet = self._packet(claim)
        self._back_to_case()
        docs = self.state.documents.setdefault(claim.case_id, {})
        return ToolResult(
            "get_claim_details",
            True,
            {
                "status": "ok",
                "case": packet.case,
                "deadline": packet.deadline,
                "today": packet.today,
                "field_definitions": packet.field_definitions,
                "document_status_so_far": {d: docs.get(d, "unknown") for d in claim.documents_needed},
                "processing_time_after_submission": packet.processing_time,
            },
        )

    def _back_to_case(self) -> None:
        """A claim question after the wrap-up reopens PROCESS_CASE and withdraws an unanswered email offer."""
        if self.state.phase == Phase.POST_PROCESS:
            if self.state.email.status == "offered":
                self.state.email = EmailOffer(send_count=self.state.email.send_count)
            self._transition(Phase.PROCESS_CASE, "case_question_after_wrap_up")

    def _doc(self, claim: Claim, name: str) -> str | None:
        """Match the model's document name to one listed on the claim (loose, case-insensitive)."""
        low = name.strip().lower()
        for d in claim.documents_needed:
            if d.lower() == low or low in d.lower() or d.lower() in low:
                return d
        return None

    def _get_document_guidance(self, a: dict[str, Any]) -> ToolResult:
        claim = self._selected()
        doc = self._doc(claim, str(a.get("document", "")))
        if doc is None:
            return ToolResult("get_document_guidance", False, {"error": "not_on_claim", "documents_needed": claim.documents_needed})
        self._back_to_case()
        info = self.guidelines.document_guidance(doc) or {}
        out = {k: v for k, v in info.items() if k != "document"}
        out.update(
            status="ok",
            document=doc,
            case_type_guidance=self.guidelines.case_type_guidance(claim.case_type),
            default_guidance=self.guidelines.default_guidance(),
        )
        self._packet(claim)
        return ToolResult("get_document_guidance", True, out)

    def _get_followup_guidance(self, a: dict[str, Any]) -> ToolResult:
        claim = self._selected()
        topic = a.get("topic")
        rules = [r for r in self.followup_rules if r.topic == topic]
        self._back_to_case()
        self._packet(claim)
        if not rules or (rules[0].requires_documents and not claim.documents_needed):
            return ToolResult("get_followup_guidance", True, {"status": "ok", "topic": topic, "text": self.guidelines.fallback_followup()})
        return ToolResult(
            "get_followup_guidance", True, {"status": "ok", "topic": topic, "text": self.guidelines.render_followup(rules[0], claim)}
        )

    def _record_document_status(self, a: dict[str, Any]) -> ToolResult:
        claim = self._selected()
        doc = self._doc(claim, str(a.get("document", "")))
        status = a.get("status")
        if doc is None or status not in DOC_STATUSES:
            return ToolResult("record_document_status", False, {"error": "invalid", "documents_needed": claim.documents_needed})
        self.state.documents.setdefault(claim.case_id, {})[doc] = status
        tracing.event("document_status", case_id=claim.case_id, document=doc, status=status)
        return ToolResult("record_document_status", True, {"status": "recorded", "documents": self.state.documents[claim.case_id]})

    # ------------------------------------------------------------------ POST_PROCESS
    def _offer_email_summary(self, a: dict[str, Any]) -> ToolResult:
        """The model writes the summary; code checks it covers the claim ID, status, and every open document."""
        claim = self._selected()
        summary = str(a.get("summary", "")).strip()
        problems = []
        if claim.case_id not in summary:
            problems.append(f"mention claim {claim.case_id}")
        if claim.status not in summary.lower():
            problems.append(f"state the status ({claim.status})")
        for doc in claim.documents_needed:
            if doc.split()[-1].lower() not in summary.lower():
                problems.append(f"include the follow-up item '{doc}'")
        if check_style(summary):
            problems.append("remove every colon and em dash")
        if len(summary) < 80 or len(summary) > 1800:
            problems.append("write 3 to 7 sentences")
        if problems:
            tracing.event("guardrail", rule="email_summary_incomplete", problems=problems)
            return ToolResult("offer_email_summary", False, {"error": "summary_rejected", "fix": problems})
        self.state.email = EmailOffer(
            status="offered",
            summary=EmailSummary(case_id=claim.case_id, case_type=claim.case_type, status=claim.status, body_text=summary),
            summary_hash=None,
            offered_turn=self.state.turn_index,
            send_count=self.state.email.send_count,
        )
        self._transition(Phase.POST_PROCESS, "email_summary_offered")
        self.effects.email_offered = True
        return ToolResult(
            "offer_email_summary",
            True,
            {
                "status": "offered",
                "message": "The app shows the summary with Send and Skip buttons. In your reply, briefly say you've prepared a summary and ask if they'd like it emailed to the address on file. Do not repeat the summary.",
            },
        )

    def _record_email_decision(self, a: dict[str, Any]) -> ToolResult:
        """Consent must come in a later turn than the offer; the actual send happens in the agent after the loop."""
        if self.state.email.status != "offered":
            return ToolResult("record_email_decision", False, {"error": "no_active_offer"})
        if self.state.email.offered_turn == self.state.turn_index:
            tracing.event("guardrail", rule="consent_same_turn_as_offer")
            return ToolResult("record_email_decision", False, {"error": "not_yet", "message": "The caller has not answered the offer yet."})
        decision = a.get("decision")
        if decision not in ("send", "skip"):
            return ToolResult("record_email_decision", False, {"error": "invalid"})
        self.effects.email_decision = decision
        return ToolResult(
            "record_email_decision",
            True,
            {"status": decision, "message": "Recorded. If skipped, acknowledge briefly. Never say the email was sent."},
        )
