"""The claims support agent: a ReAct tool loop with guardrails.

Each caller turn runs one loop against the Responses API::

    observe (conversation + session context)
      -> model reasons, then either calls tools or drafts a reply
           tools: run by ToolExecutor (agent/tools.py), which enforces the phase gates in code
           reply: checked by ReplyGuard (agent/reply_guard.py); a blocked draft goes back
                  to the model with the reasons, at most MAX_GUARDRAIL_RETRIES times

The model owns the conversation: what to ask, which tool to call, when the claim is
covered. The guardrails own what must never bend: verification needs three distinct
matching PII fields the caller actually said; claim tools exist only after verification
and only for the caller's own claims; no claim data before verification; replies are
grounded in tool results; email goes out only after an explicit choice on an active offer.

Public interface (used by web/service.py): ``new_state``, ``handle_turn``,
``complete_email``, ``rebuild_pending_email``.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Any, Callable, Literal

from insurance_claims.agent.budget import BudgetExceeded, TurnBudget
from insurance_claims.agent.guardrails import mechanical_style_fix, sanitize_markup
from insurance_claims.agent.prompts import PromptSet
from insurance_claims.agent.reply_guard import ReplyGuard
from insurance_claims.agent.tools import ToolExecutor, tool_names, tool_schemas
from insurance_claims.claims.evidence import secret_claim_tokens
from insurance_claims.claims.fixtures import FixtureBundle
from insurance_claims.claims.repository import ClaimRepository, GuidelineRepository, PolicyholderDirectory
from insurance_claims.config import Settings
from insurance_claims.domain.models import Phase
from insurance_claims.domain.state import (
    CaseSelection,
    ChatMessage,
    EmailOffer,
    PhaseTransition,
    SessionState,
    VerificationState,
)
from insurance_claims.llm.base import LLMError, ModelTransport
from insurance_claims.mail.sender import DispatchOutcome, make_op_key, summary_hash
from insurance_claims.observability import tracing

MAX_STEPS = 8
"""Model calls per turn that may use tools. The last step forces a plain reply (tool_choice="none")."""
MAX_GUARDRAIL_RETRIES = 2
"""How many times a blocked reply is sent back to the model before the safe fallback is used."""
HISTORY_MESSAGES = 24
"""Most recent chat messages replayed to the model each turn."""

_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
# Phase changes after which earlier messages are hidden from the model and from identity
# grounding (the old verification no longer applies, so neither does what was said for it).
_RESET_REASONS = ("verification_expired", "caller_changed")


# ---------------------------------------------------------------------- fixed texts
# Everything the application says by itself (not written by the model) lives here.
GREETING = (
    "Hi, thanks for contacting claims support. I can help with questions about your insurance claims. "
    "To get started, could you share your full name and two more details, such as your date of birth, "
    "the phone number or email on file, or the last four digits of your SSN or national ID?"
)
_SAFE_FALLBACK = {
    # Used when the model fails or no draft passes the guardrails. Safe in every situation of that phase.
    Phase.VERIFY_ID: "I'm sorry, I had trouble with that. To help with your claim, could you share your full name and two more details, such as your date of birth, the phone number or email on file, or the last four digits of your SSN or national ID?",
    Phase.RESOLVE_INTENT: "I'm sorry, I had trouble with that. Which claim would you like to talk about?",
    Phase.PROCESS_CASE: "I'm sorry, I had trouble pulling that together. Could you ask that again, or would you like me to connect you with a claims representative?",
    Phase.POST_PROCESS: "I'm sorry, I had trouble with that. Would you like the summary emailed to the address on file, or would you prefer to skip it?",
}
_EXPIRED_NOTICE = "For your security, your verification expired after a period of inactivity, so I'll need to confirm your identity again."
_EXPIRED_BUTTON = "For your security, your verification expired, so I'll need to confirm your identity again before sending anything."
_BUTTON_UNAVAILABLE = "That option isn't available right now."
_SKIPPED = "No problem, I won't send an email. Is there anything else I can help with?"
_TURN_LIMIT = "This conversation has gotten long, so I've flagged it for a human representative to follow up with you."
_EXPIRED_NOTE_FOR_MODEL = (
    "Note from the application: the caller's verification expired after inactivity. They must verify again before any claim details."
)
_DELIVERY_NOTICE = {
    "sent": "Done, I've sent the summary to {masked}. Is there anything else I can help with?",
    "queued": "I've saved the summary to the local demo outbox. This demo doesn't deliver real email, so nothing was sent to your inbox. Is there anything else I can help with?",
    "failed": "I'm sorry, the email couldn't be sent and nothing was delivered. A claims representative can help if you still need a copy.",
    "delivery_unknown": "I'm sorry, I can't confirm whether the email went through, so I've flagged it for a team member to check instead of risking a duplicate.",
}


# ---------------------------------------------------------------------- data passed to/from the service
@dataclass
class TurnInput:
    """One caller turn: either free text or a Send/Skip button press. ``client_turn_id`` makes retries idempotent."""

    client_turn_id: str
    text: str | None = None
    action: Literal["email_send", "email_skip"] | None = None


@dataclass
class PendingEmail:
    """A consented email the service must dispatch after the state is committed."""

    op_key: str
    recipient_ref: str
    to_address: str = field(repr=False)
    subject: str
    body_text: str = field(repr=False)
    draft_hash: str


@dataclass
class TurnOutcome:
    state: SessionState
    messages: list[ChatMessage]
    """New messages to show the caller for this turn."""
    pending_email: PendingEmail | None = None
    stop_reason: str = "completed"
    trace: dict[str, Any] = field(default_factory=dict)


def fingerprint(value: str) -> str:
    """Short, non-reversible ID for logs and idempotency keys."""
    return hashlib.sha256(value.encode()).hexdigest()[:12]


def mask_email(address: str) -> str:
    """``jane.doe@example.com`` -> ``j********@example.com``."""
    local, _, domain = address.partition("@")
    return f"{local[:1]}{'*' * max(3, min(len(local) - 1, 8))}@{domain}" if local and domain else "the email address on file"


class ClaimsAgent:
    def __init__(
        self,
        *,
        settings: Settings,
        fixtures: FixtureBundle,
        prompts: PromptSet,
        model: ModelTransport,
        clock: Callable[[], datetime],
        failure_ledger: Any | None = None,
    ) -> None:
        self.settings = settings
        self.prompts = prompts
        self.model = model
        self.clock = clock
        self.failure_ledger = failure_ledger
        self.directory = PolicyholderDirectory(fixtures.policyholders)
        self.claims = ClaimRepository(fixtures.claims)
        self.guidelines = GuidelineRepository(fixtures.guidelines)
        self.schema_doc = fixtures.claim_schema
        self.followup_rules = tuple(fixtures.guidelines.claim_followup_guidance)
        self.reply_guard = ReplyGuard(
            claims=self.claims,
            guidelines=self.guidelines,
            schema_doc=self.schema_doc,
            # Every claim fact in the dataset; none of them may appear before verification.
            secret_tokens=secret_claim_tokens(self.claims.all_claims_unscoped()),
            today=self.today,
        )

    # ------------------------------------------------------------------ public interface
    def today(self) -> date:
        return self.settings.frozen_today or self.clock().date()

    def new_state(self, session_id: str) -> SessionState:
        now = self.clock()
        state = SessionState(session_id=session_id, created_at=now, last_activity_at=now)
        state.history.append(ChatMessage(role="assistant", text=GREETING, turn_index=0, at=now, phase=Phase.VERIFY_ID))
        return state

    def handle_turn(self, state: SessionState, turn: TurnInput) -> TurnOutcome:
        """Run one caller turn on a copy of ``state``; the service persists ``outcome.state``."""
        state = state.model_copy(deep=True)
        now = self.clock()
        state.turn_index += 1
        trace_input = {"turn_index": state.turn_index, "phase_before": state.phase.value, "kind": "action" if turn.action else "text"}
        with tracing.span("agent", type="task", input=trace_input) as node:
            expired = self._expire_if_needed(state, now)
            if turn.action:
                outcome = self._handle_button(state, turn.action, now, expired)
            else:
                text = _CONTROL_CHARS.sub("", turn.text or "").strip()[: self.settings.max_message_chars]
                outcome = self._handle_message(state, text, now, expired)
            outcome.state.last_activity_at = now
            node["output"] = {
                "phase_after": outcome.state.phase.value,
                "verified": outcome.state.verified,
                "case_id": outcome.state.case.case_id,
                "email": outcome.state.email.status,
                "stop_reason": outcome.stop_reason,
            }
            node["metadata"] = {"prompt_version": self.prompts.version, "prompt_sha256": self.prompts.sha256[:16]}
            return outcome

    def rebuild_pending_email(self, state: SessionState) -> PendingEmail | None:
        """Recreate a consented-but-undelivered email after a crash (the op_key keeps it idempotent)."""
        email = state.email
        if not (state.verified and email.op_key and email.summary and email.summary_hash):
            return None
        party = state.verification.party_id or ""
        return PendingEmail(
            email.op_key, party, self.directory.contact_email(party), _subject(email), email.summary.body_text, email.summary_hash
        )

    def complete_email(self, state: SessionState, result: DispatchOutcome) -> TurnOutcome:
        """Record the delivery result and tell the caller exactly what happened."""
        state = state.model_copy(deep=True)
        now = self.clock()
        state.email.status = result.status
        state.email.receipt_id = result.receipt_id
        state.email.detail = (result.detail or "")[:200] or None
        if result.status in ("sent", "queued"):
            state.email.send_count += 1
        if result.status in ("failed", "delivery_unknown"):
            state.handoff.offered = True
        masked = mask_email(self.directory.contact_email(state.verification.party_id or "")) if state.verified else "the address on file"
        tracing.event("email_result", status=result.status)
        msg = self._append(state, "assistant", _DELIVERY_NOTICE[result.status].format(masked=masked), now, kind="notice")
        return TurnOutcome(state=state, messages=[msg], stop_reason=f"email_{result.status}")

    # ------------------------------------------------------------------ expiry and email buttons
    def _expire_if_needed(self, state: SessionState, now: datetime) -> bool:
        """Drop verification (and the selected claim) after the idle TTL. Returns True if it expired."""
        if not state.verified or now - state.last_activity_at <= timedelta(minutes=self.settings.verification_idle_ttl_minutes):
            return False
        tracing.event("guardrail", rule="verification_expired")
        old = state.verification
        state.verification = VerificationState(
            caller_role=old.caller_role, representative_declared=old.representative_declared, failed_attempts=old.failed_attempts
        )
        state.case = CaseSelection()
        if state.email.status == "offered":
            state.email = EmailOffer(send_count=state.email.send_count)
        self._transition(state, Phase.VERIFY_ID, "verification_expired", now)
        return True

    def _handle_button(self, state: SessionState, action: str, now: datetime, expired: bool) -> TurnOutcome:
        """Send/Skip buttons bypass the model; they only work on an active offer to a verified caller."""
        self._append(state, "user", "Send the email summary" if action == "email_send" else "Skip the email", now)
        allowed = not expired and state.verified and state.phase == Phase.POST_PROCESS and state.email.status == "offered"
        tracing.event("guardrail", rule="email_button", action=action, allowed=allowed)
        if not allowed:
            msg = self._append(state, "assistant", _EXPIRED_BUTTON if expired else _BUTTON_UNAVAILABLE, now, kind="notice")
            return TurnOutcome(state=state, messages=[msg], stop_reason="action_rejected")
        return self._apply_email_decision(state, "send" if action == "email_send" else "skip", now, messages=[])

    def _apply_email_decision(self, state: SessionState, decision: str, now: datetime, messages: list[ChatMessage]) -> TurnOutcome:
        """Skip: record it. Send: persist consent + idempotency key and hand a PendingEmail to the service."""
        email = state.email
        email.decided_turn = state.turn_index
        if decision == "skip":
            email.status = "skipped"
            if not messages:
                messages = [self._append(state, "assistant", _SKIPPED, now, kind="notice")]
            return TurnOutcome(state=state, messages=messages, stop_reason="email_skipped")
        assert email.summary is not None
        party = state.verification.party_id or ""
        body = email.summary.body_text
        email.summary_hash = summary_hash(body)
        email.op_key = make_op_key(state.session_id, email.summary_hash, state.turn_index)
        email.status = "consented"
        tracing.event("email_consent", op_key=email.op_key[:12])
        pending = PendingEmail(email.op_key, party, self.directory.contact_email(party), _subject(email), body, email.summary_hash)
        return TurnOutcome(state=state, messages=messages, pending_email=pending, stop_reason="email_consented")

    # ------------------------------------------------------------------ a typed message: the ReAct loop
    def _handle_message(self, state: SessionState, text: str, now: datetime, expired: bool) -> TurnOutcome:
        self._append(state, "user", text, now)
        if state.turn_index > self.settings.max_turns_per_session:
            state.handoff.requested = state.handoff.offered = True
            msg = self._append(state, "assistant", _TURN_LIMIT, now)
            return TurnOutcome(state=state, messages=[msg], stop_reason="turn_limit")

        executor = ToolExecutor(
            state=state, now=now, today=self.today(), caller_text=self._caller_text(state),
            directory=self.directory, claims=self.claims, guidelines=self.guidelines, schema_doc=self.schema_doc,
            followup_rules=self.followup_rules, settings=self.settings, failure_ledger=self.failure_ledger,
        )  # fmt: skip
        items = self._history_items(state)
        if expired:
            items.append(_developer(_EXPIRED_NOTE_FOR_MODEL))
        budget = TurnBudget(
            max_model_calls=MAX_STEPS + MAX_GUARDRAIL_RETRIES,
            max_tool_calls=self.settings.max_tool_calls_per_turn + 4,
            deadline_s=self.settings.turn_deadline_s,
            model_timeout_s=self.settings.model_timeout_s,
        )
        with budget:
            reply, stop = self._react(state, items, executor, budget)
        tracing.event("turn_summary", tools=[c["name"] for c in executor.calls], stop_reason=stop, budget=budget.snapshot())
        return self._finish_turn(state, reply or _SAFE_FALLBACK[state.phase], stop, now, expired, executor)

    def _react(
        self, state: SessionState, items: list[dict[str, Any]], executor: ToolExecutor, budget: TurnBudget
    ) -> tuple[str | None, str]:
        """Loop model -> tools -> model until a reply passes the guardrails. Returns (reply or None, stop reason).

        Tools may change ``state`` (via the executor), so the instructions and tool menu are rebuilt
        every step: once ``verify_identity`` succeeds, the next step already sees the claim tools.
        """
        retries = 0
        for step in range(MAX_STEPS):
            try:
                resp = budget.call(
                    self.model,
                    task="agent",
                    instructions=self._instructions(state),
                    input=items,
                    tools=tool_schemas(state),
                    tool_choice="none" if step == MAX_STEPS - 1 else "auto",
                    parallel_tool_calls=True,
                    max_output_tokens=self.settings.max_output_tokens_reply,
                    reasoning_effort=self.settings.reasoning_effort_reply,
                )
            except (LLMError, BudgetExceeded) as exc:
                return None, f"model_error:{getattr(exc, 'kind', getattr(exc, 'reason', 'unknown'))}"

            # Act: run the requested tools and feed their results back.
            if resp.function_calls:
                if not self._run_tools(resp, items, executor, budget):
                    return None, "call_id_mismatch"
                continue

            # Reply: check the draft; if blocked, show the model why and let it try again.
            draft = sanitize_markup(resp.text or "")
            problems = self.reply_guard.check(draft, state, executor.effects)
            if not problems:
                return draft, "completed"
            tracing.event("guardrail", rule="reply_blocked", problems=problems, retry=retries + 1)
            if retries >= MAX_GUARDRAIL_RETRIES:
                # Last chance: a purely stylistic problem (e.g. a colon) can be fixed without the model.
                fixed = mechanical_style_fix(draft) if draft else ""
                if fixed and not self.reply_guard.check(fixed, state, executor.effects):
                    return fixed, "completed_after_fix"
                return None, "guardrail_exhausted"
            retries += 1
            items.extend(resp.raw_output_items)
            items.append(
                _developer(
                    f"GUARDRAIL (from the application): your reply was not sent because: {'; '.join(problems)}. Write a corrected reply to the caller."
                )
            )
        return None, "max_steps"

    @staticmethod
    def _run_tools(resp: Any, items: list[dict[str, Any]], executor: ToolExecutor, budget: TurnBudget) -> bool:
        """Execute each function call once and append its output. False if the response is malformed."""
        raw_ids = {i.get("call_id") for i in resp.raw_output_items if i.get("type") == "function_call"}
        if raw_ids != {fc.call_id for fc in resp.function_calls}:
            return False  # every function_call needs exactly one matching output, or the next request fails
        items.extend(resp.raw_output_items)  # echo calls (and reasoning items) back, as the API requires
        answered: set[str] = set()
        for fc in resp.function_calls:
            if fc.call_id in answered:
                continue
            answered.add(fc.call_id)
            result = executor.execute(fc.name, fc.arguments) if budget.charge_tool() else None
            output = result.to_json() if result else json.dumps({"error": "tool_budget_exhausted", "message": "Reply to the caller now."})
            items.append({"type": "function_call_output", "call_id": fc.call_id, "output": output})
        return True

    def _finish_turn(self, state: SessionState, reply: str, stop: str, now: datetime, expired: bool, executor: ToolExecutor) -> TurnOutcome:
        """Turn the loop's result and the tools' side effects into the messages the caller sees."""
        effects = executor.effects
        if effects.email_decision == "send":
            # The delivery notice from complete_email is the truthful reply; no model text about sending.
            return self._apply_email_decision(state, "send", now, [])
        messages = []
        if expired:
            messages.append(self._append(state, "assistant", _EXPIRED_NOTICE, now, kind="notice"))
        messages.append(self._append(state, "assistant", reply, now))
        if effects.email_offered and state.email.summary is not None:
            # The summary card is written by code from claim data, not by the model.
            masked = mask_email(self.directory.contact_email(state.verification.party_id or ""))
            offer = f"{state.email.summary.body_text}\n\nWould you like me to email this summary to {masked}?"
            messages.append(self._append(state, "assistant", offer, now, kind="email_offer"))
        if effects.email_decision:
            return self._apply_email_decision(state, effects.email_decision, now, messages)
        return TurnOutcome(state=state, messages=messages, stop_reason=stop)

    # ------------------------------------------------------------------ what the model sees
    def _instructions(self, state: SessionState) -> str:
        base = self.prompts.instructions("agent", state.phase.value)
        return f"{base}\n\n# Session context (from the application, trusted)\n{json.dumps(self._context(state), indent=1)}"

    def _context(self, state: SessionState) -> dict[str, Any]:
        """Trusted facts about the session. Claim facts are only included after verification."""
        h = state.hints
        remembered = {"reasons": h.question_summaries[-3:], "case_type": h.case_type, "status": h.status, "month": h.month, "year": h.year}
        ctx: dict[str, Any] = {
            "phase": state.phase.value,
            "verified": state.verified,
            "today": self.today().isoformat(),
            "tools_available": list(tool_names(state)),
            "remembered_context": {k: v for k, v in remembered.items() if v},
            "off_topic_count": state.counters.off_topic_total,
            "handoff": {"offered": state.handoff.offered, "requested": state.handoff.requested},
            "failed_verification_attempts": state.verification.failed_attempts,
        }
        if state.verified:
            ctx["caller_first_name"] = self.directory.first_name(state.verification.party_id or "")
            ctx["selected_claim"] = state.case.case_id
            if state.case.case_id:
                ctx["document_status"] = state.documents.get(state.case.case_id, {})
            ctx["email_offer"] = state.email.status
        return ctx

    def _history_floor(self, state: SessionState) -> int:
        """Turn index of the last verification reset; earlier messages are not used."""
        for t in reversed(state.phase_log):
            if t.reason in _RESET_REASONS:
                return t.turn_index
        return -1

    def _history_items(self, state: SessionState) -> list[dict[str, Any]]:
        """Recent chat as Responses API input. Caller text is wrapped in tags to mark it as untrusted."""
        floor = self._history_floor(state)
        recent = [m for m in state.history if m.turn_index >= floor][-HISTORY_MESSAGES:]
        return [
            {"role": "user", "content": f"<caller_message>\n{m.text}\n</caller_message>"}
            if m.role == "user"
            else {"role": "assistant", "content": m.text}
            for m in recent
        ]

    def _caller_text(self, state: SessionState) -> str:
        """Everything the caller typed since the last reset; identity values must appear here."""
        floor = self._history_floor(state)
        return "\n".join(m.text for m in state.history if m.role == "user" and m.turn_index >= floor)

    # ------------------------------------------------------------------ state helpers
    def _append(self, state: SessionState, role: Literal["user", "assistant"], text: str, now: datetime, kind: str = "chat") -> ChatMessage:
        msg = ChatMessage(role=role, text=text, turn_index=state.turn_index, at=now, phase=state.phase, kind=kind)  # type: ignore[arg-type]
        state.history.append(msg)
        state.history = state.history[-self.settings.history_limit :]
        return msg

    def _transition(self, state: SessionState, to: Phase, reason: str, now: datetime) -> None:
        if state.phase == to:
            return
        tracing.event("phase_transition", from_phase=state.phase.value, to_phase=to.value, reason=reason)
        state.phase_log.append(PhaseTransition(turn_index=state.turn_index, from_phase=state.phase, to_phase=to, reason=reason, at=now))
        state.phase = to


def _developer(text: str) -> dict[str, str]:
    """An application note to the model (developer role outranks caller text)."""
    return {"role": "developer", "content": text}


def _subject(email: EmailOffer) -> str:
    cid = email.summary.case_id if email.summary else None
    return f"Summary of your claims support conversation ({cid})" if cid else "Summary of your claims support conversation"
