# Design

This document explains how the SOP-guided insurance claims agent is built. Product requirements
are in `instructions.md` and `plan.md` at the repository root; run instructions are in the README.

Package: `apps/insurance_claims/src/insurance_claims/` (import name `insurance_claims`), Python 3.12,
Pydantic v2, FastAPI, OpenAI Responses API (`gpt-5.6-luna`).

## 1. The idea in one paragraph

The model runs the conversation; code runs the rules. Each caller turn is one ReAct loop: the
model reads the conversation and a trusted session-context block, reasons, calls tools, reads
their results, and writes a reply. Tools are the only way anything changes (verification, the
selected claim, the phase, the email offer), and every tool call is a *proposal* that code checks
before it takes effect. Every drafted reply passes output guardrails before the caller sees it.
The model therefore has freedom where the SOP gives freedom (tone, empathy, what to ask next,
when the claim is covered) and none where it does not (identity, data access, consent, facts).

## 2. Request flow

```
browser (web/static: HTML/CSS/JS, textContent only, client-side send queue)
   │  same-origin JSON, HttpOnly session cookie + X-CSRF-Token, client_turn_id per message
   ▼
web/app.py         FastAPI: validation, cookie/CSRF, body limit, security headers, error envelope
   ▼
web/service.py     ConversationService: per-session lock, turn dedupe, load -> agent -> commit, email dispatch
   ▼
agent/loop.py      ClaimsAgent.handle_turn: the ReAct loop
   ├─ agent/tools.py        tool schemas, per-phase tool menu, ToolExecutor (tool guardrails)
   │    └─ claims/          fixtures, party-scoped repositories, identity matching, evidence packets
   ├─ agent/reply_guard.py  ReplyGuard: checks each draft (built on agent/guardrails.py)
   ├─ agent/prompts.py      prompts.toml: global, style, per-phase, and task instructions
   └─ llm/                  Responses API transport with retries (fake and chaos transports for tests)
persistence/store.py   encrypted SQLite: sessions (versioned), turns, email ledger, verification failures
mail/sender.py         consent-gated, idempotent email dispatch (outbox or SMTP)
observability/         redaction, nested JSON traces per turn, trace reader
```

## 3. The ReAct loop (`agent/loop.py`)

`ClaimsAgent.handle_turn(state, TurnInput)` works on a deep copy of the state and returns a
`TurnOutcome` (new state, messages to show, optional `PendingEmail`, stop reason).

1. Expire verification if the caller was idle longer than `VERIFICATION_IDLE_TTL_MINUTES`
   (back to `VERIFY_ID`; the model is told; the caller sees a notice).
2. A Send/Skip button press skips the model entirely: it is accepted only for a verified caller
   with an active offer in `POST_PROCESS`.
3. For a typed message, build the input: the last 24 messages since the latest identity reset,
   with caller text wrapped in `<caller_message>` tags (untrusted), and instructions =
   `prompts.toml` text for the current phase + a JSON session-context block (phase, verified,
   today, available tools, remembered context, counters, and, only after verification, first name,
   selected claim, document status, email status).
4. Loop up to `MAX_STEPS = 8` model calls (the last one with `tool_choice="none"`):
   * **function calls**: check that every call ID is answered exactly once, echo the output items
     (including reasoning items, since `store=False`), run each tool through `ToolExecutor`, append
     `function_call_output` items, and loop. Instructions and the tool menu are rebuilt on every
     step, so a successful `verify_identity` unlocks claim tools for the very next step.
   * **text**: strip markup and run `ReplyGuard.check`. Clean: done. Blocked: append a developer
     message naming the problems and loop again (at most `MAX_GUARDRAIL_RETRIES = 2`; then try a
     purely mechanical style fix; then the phase's safe fallback reply).
5. Turn tool side effects into messages: an `email_offer` card (summary text + masked address,
   shown with Send/Skip), or an email decision (skip: acknowledge; send: persist consent and hand a
   `PendingEmail` to the service, whose delivery result becomes the reply).

`TurnBudget` (`agent/budget.py`) caps model calls (10), tool calls (`MAX_TOOL_CALLS_PER_TURN` + 4),
and wall-clock time (`TURN_DEADLINE_S`); per-call timeouts are clipped to the remaining time.

## 4. Tools and their guardrails (`agent/tools.py`)

Tool menu per phase (`BY_PHASE` + `ALWAYS`); the executor re-checks the menu on every call, so a
hallucinated or out-of-phase tool returns an explanation instead of acting.

| Phase | Tools |
| --- | --- |
| all | `note_caller_context`, `flag_off_topic`, `request_human` |
| VERIFY_ID | `verify_identity`, `note_refusal` |
| RESOLVE_INTENT | `list_my_claims`, `select_claim`, `report_caller_change` |
| PROCESS_CASE | + `get_claim_details`, `get_document_guidance`, `get_followup_guidance`, `record_document_status`, `offer_email_summary` |
| POST_PROCESS | claim tools + `record_email_decision` (only while an offer is active); `offer_email_summary` again only after a skip or a failed send |

Phase changes happen only as side effects of successful tools:
`verify_identity` -> RESOLVE_INTENT, `select_claim` -> PROCESS_CASE, `offer_email_summary` ->
POST_PROCESS, any claim tool after the offer -> PROCESS_CASE (withdrawing an unanswered offer),
`report_caller_change` -> VERIFY_ID.

| Freedom | Enforced by |
| --- | --- |
| Verification | `verify_identity`: each value must normalize and appear in the caller's own messages since the last reset (`identity_value_grounded`); at least 3 distinct PII fields (policy number never counts) before any matching; deterministic match to exactly one policyholder with no conflicting field (`claims/verification.py`); per-session lockout after `MAX_VERIFICATION_FAILURES`, cross-session lockout per policyholder (fails closed); someone acting for the policyholder is never verified |
| Data access | Claim tools exist only after verification and are scoped to the verified party; another party's claim ID returns "not found" |
| Persuasion | `note_refusal` counts refusals; `request_human(reason=repeated_refusal)` is refused before `MAX_REFUSALS` |
| Scope | `flag_off_topic` counts off-topic requests and tells the model to offer a human after `MAX_OFF_TOPIC` |
| Email summary | `offer_email_summary` rejects a summary that misses the claim ID, the status, any outstanding document, or breaks the style rules |
| Consent | `record_email_decision` only on an active offer and never in the same turn as the offer; recipient is always the address on file |
| Repetition | The same tool with the same arguments twice in one turn is refused (except `verify_identity`) |
| Failures | Any tool exception becomes a `tool_error` result; the model apologises and can offer a human |

Memory: `note_caller_context` stores why the caller is calling (type, status, month, year, a
short PII-free reason) in any phase; it is returned by `verify_identity` and `list_my_claims`, so
the model can select the claim without asking again.

## 5. Reply guardrails (`agent/reply_guard.py`, `agent/guardrails.py`)

`ReplyGuard.check(text, state, effects)` returns a list of problems (empty = send it).

* Every reply: not empty; no colon and no em dash (`check_style`); no internal vocabulary such as
  phase names, party IDs, or "system prompt" (`check_internal_reference`).
* Before verification: no claim facts from any claim in the dataset (IDs, amounts, dates, denial
  text, documents; `check_pre_verification_leak`).
* After verification:
  * grounding (`check_grounding`): claim IDs must belong to the caller; amounts and dates must
    appear in claim records or tool results; no promises of outcomes or invented actions; a
    passed appeal deadline must not read as open; the stated status must match the claim;
  * a description like "closed healthcare claim from January 2026" must match one of the caller's claims;
  * no "once you're verified" wording, no claim that an email was sent unless it was;
  * a claim conversation may not end with a goodbye before the email summary was offered.

## 6. State and persistence

`domain/state.py` `SessionState` is the checkpoint (`STATE_SCHEMA_VERSION = 3`): phase, turn
index, verification (status, party ID, per-field match sets, failed attempts, caller role,
sticky representative flag), hints, selected case, counters, handoff, email offer, per-claim
document status, chat history, phase log. Raw identity values are never stored as fields.
`public_view()` is the only projection sent to the browser.

`persistence/store.py`: SQLite (WAL) with tables `sessions` (Fernet-encrypted state blob,
optimistic `version`), `turns` (encrypted response per `client_turn_id`, HMAC request hash),
`email_operations` (the email ledger), `verification_failures` (subjects like `party:P9`), and
`meta`. One transaction per accepted turn; a version mismatch raises `VersionConflict`.
A checkpoint of another schema version is rejected as `unsupported_schema` (the caller starts a
new conversation) unless an upgrade is registered in `_MIGRATIONS`; none exist today. `STATE_ENCRYPTION_KEY` is optional
(a key file is generated in the data directory) except with `APP_ENV=production`.

## 7. Service (`web/service.py`)

* Session lifecycle: opaque session ID, per-session secret in an HttpOnly cookie (only its hash
  is stored), HMAC-derived CSRF token, expiry, hourly retention purge (sessions, turns, ledgers,
  traces, outbox files).
* Idempotency: a `client_turn_id` is processed once; a retry replays the stored response; the same
  ID with a different body is rejected.
* Concurrency: a per-session lock plus optimistic versioning.
* Email: the consent (status `consented` + `op_key`) is committed *before* dispatch; after a crash
  the pending operation is reconciled from the ledger before anything else happens. One consent
  produces at most one send.
* Views: reads apply the idle TTL and hide messages from before the latest identity reset.

## 8. Email (`mail/sender.py`)

`EmailDispatcher.dispatch` drives the ledger row `pending -> dispatching -> sent | queued | failed |
delivery_unknown` with compare-and-set updates. `OutboxTransport` writes an `.eml` file (local
demo, status `queued`, never delivered); `SmtpTransport` sends for real (`sent`). An ambiguous
transport error is reconciled when the transport supports lookup, otherwise recorded as
`delivery_unknown` and routed to a human; nothing is ever re-sent automatically. Addresses and
bodies are never logged.

## 9. Model transport (`llm/`)

`OpenAITransport` calls `client.responses.create` with `store=False`, strict function tools, and
`include=["reasoning.encrypted_content"]`, and converts the result into `LLMResponse` (text,
function calls, raw output items, usage). `ResilientTransport` retries only transient errors with
backoff and never sleeps past the turn deadline. `OfflineFakeModel` (`MODEL_PROVIDER=fake`) is a
small deterministic tool-using agent for demos and tests. `ChaosTransport` injects faults into
tool calls (extra, missing, mistyped, or malformed arguments, hallucinated tools, mismatched or
duplicate call IDs) and into replies (injected text, style violations, refusals, truncation,
timeouts, rate limits). `tests/agent/test_agent_chaos.py` runs the agent over it for 30 seeds and
checks the guardrails hold on every turn.

## 10. Observability (`observability/`)

Each turn writes one nested JSON trace: the `agent` span with phase before/after and stop reason,
`llm` spans (model, token use, tool names; never prompt or output text), `tool.<name>` spans,
and events for guardrail decisions, phase transitions, memory, email consent and results.
`Redactor` replaces names, dates of birth, contact details, ID digits, claim text, email bodies,
and API keys before serialization, and once more over the serialized JSON.

```bash
.venv/bin/python -m insurance_claims.observability.trace_reader traces/ --last 5
```

## 11. HTTP contract (`web/app.py`)

```
POST   /api/sessions                 -> 201 {session_id, csrf_token, state, messages}   sets the session cookie
GET    /api/sessions/{id}            -> 200 {session_id, csrf_token, state, messages}   cookie required
POST   /api/sessions/{id}/messages   -> 200 {turn_index, duplicate, messages, state}   cookie + X-CSRF-Token
       body {"client_turn_id": "...", "text": "..."} or {"client_turn_id": "...", "action": "email_send" | "email_skip"}
DELETE /api/sessions/{id}            -> 204                                              cookie + X-CSRF-Token
GET    /health                       -> 200 {status, checks, version}
errors -> {"error": {"code", "message", "request_id"}}
messages[] -> {"role", "text", "kind": "chat" | "email_offer" | "notice", "turn_index"}
```

Bodies over 16 KB get 413 (streamed bodies are counted); cross-site state-changing requests get
403; every response carries security headers (strict CSP, no inline script), `X-Request-ID`, and
`Cache-Control: no-store` on API paths.

## 12. Frontend (`web/static/`)

Vanilla HTML/CSS/JS on a light theme. Server text is rendered with `textContent` only. A send
queue gives every message a `client_turn_id` and sends them in order, so Enter pressed while a
session is starting or reconnecting is never lost and retries are idempotent. The four-step
progress bar follows `state.steps`; Send/Skip buttons appear under the email offer while it is
active. The session ID is kept in `localStorage` so a refresh resumes the conversation.

## 13. Tests

`tests/` mirrors the package (`agent/`, `claims/`, `llm/`, `mail/`, `persistence/`,
`observability/`, `web/`) plus `browser/` (Playwright, opt-in with `-m browser`). The agent tests
script a misbehaving model (verifying with two fields, inventing identity values, calling claim
tools early, leaking claim data, selecting another party's claim, inventing amounts, consenting
in the same turn as the offer) and assert that code stops every attempt. No unit test calls the
network; `evals/live_eval.py` runs scripted conversations against the real model on request.
