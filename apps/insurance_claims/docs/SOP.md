# Claims support SOP

Generated from `sop.toml` (version `2026-10-04.sop-1`); do not edit by hand.
Regenerate with `python -m insurance_claims.agent.sop > docs/SOP.md`.

Workflow: VERIFY_ID → RESOLVE_INTENT → PROCESS_CASE → POST_PROCESS.

## In every phase

Tools always available: `request_human`.

| Strict (enforced by code or the guard) | Enforced by |
| --- | --- |
| Every caller message is reviewed by the guard before the agent runs (who is speaking, refusals, off-topic requests, what claim they describe). | agent/loop.py ClaimsAgent._apply_caller_review, verdict from agent/guard.py review_caller |
| Every reply the agent drafts is checked by code (style, internal details, grounding of IDs, amounts, dates, status) and then reviewed by the guard (facts against the record, disclosure, promises, scope, required content) before the caller sees it. | agent/reply_guard.py ReplyGuard.check (code checks, then guard judge_reply) |
| When the off-topic or refusal limit is reached, the caller acts for someone else, or verification locks, the reply must offer a human representative. | agent/loop.py human_offer_due + agent/reply_guard.py required_in_reply + fallback offer |
| A tool outside the current phase's menu is refused, even if the model calls it. | agent/tools.py ToolExecutor._dispatch re-checks Sop.tool_names on every call |
| Someone acting for the policyholder, or a new person after verification, loses access and starts over at VERIFY_ID. | agent/loop.py ClaimsAgent._withdraw_access |
| After the idle time limit, verification expires and the caller must verify again. | agent/loop.py ClaimsAgent._expire_if_needed |

Left to the model:

* Tone, empathy, and wording of every reply.
* Acknowledging emotions first, explaining why a step matters, and keeping the conversation moving.
* Declining unrelated requests politely and steering back.
* Asking for a human at any time; the agent hands over with request_human.

## VERIFY_ID

Goal: Confirm the caller is the policyholder before anything about any claim is shared.

Tools: `verify_identity`.

| Strict (enforced by code or the guard) | Enforced by |
| --- | --- |
| No claim information at all is disclosed before verification. | agent/guardrails.py check_pre_verification_leak + guard disclosed_before_verification |
| Only the account holder can be verified; someone acting for another person is refused for the rest of the conversation. | agent/tools.py ToolExecutor._verify_identity (speaker must be account_holder) |
| Identity values count only if the caller actually said them. | agent/tools.py ToolExecutor._verify_identity + claims/normalize.py identity_value_grounded |
| At least three distinct identity fields are required (full name, date of birth, phone, email, SSN or national ID last four); a policy number never counts. | agent/tools.py ToolExecutor._verify_identity (REQUIRED_PII) |
| All fields must match exactly one policyholder with no conflicting field, and the reply never says which field failed. | claims/verification.py apply_identity_proposals |
| Repeated failures lock verification, per conversation and per policyholder across conversations. | claims/verification.py (per session) + agent/tools.py _party_locked (across sessions) |
| Refusals are counted by code; the agent may hand over for repeated refusal only after the limit. | agent/loop.py _apply_caller_review + agent/tools.py _request_human gate |
| When three identity details were given, the agent must try verification instead of asking for more. | agent/reply_guard.py ReplyGuard._skipped_verification |

Left to the model:

* Which details to ask for, in what order, and how to handle partial answers, corrections, and clarifying questions.
* Offering alternative identity fields when the caller cannot or will not give one.
* How to persuade gently after a refusal, and when the conversation needs a human.

Leaves to: RESOLVE_INTENT when verify_identity returns verified (`identity_verified`).

## RESOLVE_INTENT

Goal: Find which of the verified caller's claims they mean, without making them start over.

Tools: `list_my_claims`, `select_claim`.

| Strict (enforced by code or the guard) | Enforced by |
| --- | --- |
| Only the verified caller's own claims can be listed or selected; another party's claim ID is reported as not found. | claims/repository.py ClaimRepository.get_for_party / list_for_party |

Left to the model:

* Matching the remembered description to a claim, and asking one short question when several fit.

Leaves to: PROCESS_CASE when select_claim succeeds (in POST_PROCESS only when switching to another claim, which withdraws an open offer) (`claim_selected`); VERIFY_ID when the guard finds a different person, or someone acting for the policyholder, after verification (`caller_changed`); VERIFY_ID when the caller was idle longer than the verification time limit (`verification_expired`).

## PROCESS_CASE

Goal: Explain the selected claim from its record and move it toward resolution.

Tools: `list_my_claims`, `select_claim`, `get_claim_details`, `get_document_guidance`, `get_followup_guidance`, `record_document_status`, `offer_email_summary`.

| Strict (enforced by code or the guard) | Enforced by |
| --- | --- |
| Every statement about the claim must be supported by the record, the tool results, or the caller's own words. | agent/guardrails.py check_grounding + guard unsupported_fact / promise_or_invented_action |
| No guaranteed outcomes, no appeal rights beyond the record, and no claims that the agent submitted or received anything. | agent/guardrails.py check_grounding + guard unsupported_fact / promise_or_invented_action |
| A document status is stored only when the caller's words say the same. | agent/tools.py ToolExecutor._record_document_status + guard judge_document |
| The email summary must name the claim, its status, and every outstanding document, and every statement must be supported. | agent/tools.py ToolExecutor._offer_email_summary + guard judge_summary |
| A claim conversation may not end without offering the email summary. | agent/reply_guard.py ReplyGuard._ends_without_email_offer |

Left to the model:

* What to explain first, which follow-up questions to ask, and when the claim is covered.
* Answering side questions from the guidance and returning to the open point.

Leaves to: PROCESS_CASE when select_claim succeeds (in POST_PROCESS only when switching to another claim, which withdraws an open offer) (`claim_selected`); POST_PROCESS when offer_email_summary is accepted (`email_summary_offered`); VERIFY_ID when the guard finds a different person, or someone acting for the policyholder, after verification (`caller_changed`); VERIFY_ID when the caller was idle longer than the verification time limit (`verification_expired`).

## POST_PROCESS

Goal: Offer the email summary, record the caller's explicit choice, and close or return to the claim.

Tools: `list_my_claims`, `select_claim`, `get_claim_details`, `get_document_guidance`, `get_followup_guidance`, `record_email_decision` (only when email offer open), `offer_email_summary` (only when no open offer or send).

| Strict (enforced by code or the guard) | Enforced by |
| --- | --- |
| A typed choice is recorded only when the guard reads the same choice in the caller's own words; Send and Skip buttons work only on an open offer. | agent/tools.py ToolExecutor._record_email_decision + guard judge_consent; agent/loop.py _handle_button |
| Consent can never come in the same turn as the offer, and the email goes only to the address on file. | agent/tools.py _record_email_decision (not in the offer's turn); agent/loop.py PendingEmail to the on-file address |
| The reply never says the email was sent; the application reports the real delivery result. | agent/reply_guard.py (no 'sent' claims) + agent/loop.py complete_email delivery notices |

Left to the model:

* Answering more questions about the same claim while the offer stays open.
* Re-offering the summary after a skip or a failed send.

Leaves to: PROCESS_CASE when select_claim succeeds (in POST_PROCESS only when switching to another claim, which withdraws an open offer) (`claim_selected`); POST_PROCESS when offer_email_summary is accepted (`email_summary_offered`); PROCESS_CASE when a claim tool is used after the wrap-up when no offer is open (`case_question_after_wrap_up`); VERIFY_ID when the guard finds a different person, or someone acting for the policyholder, after verification (`caller_changed`); VERIFY_ID when the caller was idle longer than the verification time limit (`verification_expired`).

## Memory across phases

| What | Captured | Used | Cleared |
| --- | --- | --- | --- |
| Why the caller is calling and which claim they describe (type, status, month, year) | In any phase, from the guard's review of every caller message, even before verification | After verification, to select the claim without asking again | When access is withdrawn (a new person or someone acting for the policyholder) |
| What the caller said about each required document | PROCESS_CASE, through record_document_status (confirmed by the guard) | Later replies, the email summary, and its fact check | When access is withdrawn |
| The conversation itself | Every turn, stored encrypted | Recent messages since the last identity reset are given to the agent and the guard | Messages before an identity reset (expiry or a new person) are never shown to a model again |
