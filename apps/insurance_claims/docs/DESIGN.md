# Design

This document explains how the SOP-guided insurance claims agent is built. Product requirements
are in `instructions.md` and `plan.md` at the repository root; run instructions are in the README.

Package: `apps/insurance_claims/src/insurance_claims/` (import name `insurance_claims`), Python 3.12,
Pydantic v2, FastAPI, OpenAI Responses API (`gpt-5.6-luna`).

## 1. The idea in one paragraph

The model runs the conversation; code runs the rules. Each caller turn is one ReAct loop: the
agent model reads the conversation and a trusted session-context block, reasons, calls tools, reads
their results, and writes a reply. Tools are the only way anything changes (verification, the
selected claim, the phase, the email offer), and every tool call is a *proposal* that code checks
before it takes effect. Every drafted reply passes output guardrails before the caller sees it.
Where a rule depends on understanding language (who is speaking, what the caller agreed to,
whether a summary is true, whether a reply is on topic), code asks a second, independent model,
the **guard**, and refuses the action unless the guard clearly allows it. The agent therefore has
freedom where the SOP gives freedom (tone, empathy, what to ask next, when the claim is covered)
and none where it does not (identity, data access, consent, facts, scope).

## 2. Request flow

```
browser (web/static: HTML/CSS/JS, textContent only, client-side send queue)
   │  same-origin JSON, HttpOnly session cookie + X-CSRF-Token, client_turn_id per message
   ▼
web/app.py         FastAPI: validation, cookie/CSRF, body limit, security headers, error envelope
   ▼
web/service.py     ConversationService: per-session lock, turn dedupe, load -> agent -> commit, email dispatch
   ▼
agent/loop.py      ClaimsAgent.handle_turn: guard reviews the caller, then the ReAct loop
   ├─ agent/guard.py        Guard: independent model reviewer with its own prompts and transport
   ├─ agent/tools.py        tool schemas, per-phase tool menu, ToolExecutor (tool guardrails)
   │    └─ claims/          fixtures, party-scoped repositories, identity matching, evidence packets
   ├─ agent/reply_guard.py  ReplyGuard: checks each draft (agent/guardrails.py, then the guard's full review)
   ├─ agent/prompts.py      prompts.toml: agent prompts (global, style, per phase) and guard prompts
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
2. A Send/Skip button press skips both models: it is accepted only for a verified caller with an
   active offer in `POST_PROCESS`.
3. For a typed message, the guard first reviews the caller (section 5) and code applies the
   result to the state.
4. Build the agent's input: the last 24 messages since the latest identity reset, with caller text
   wrapped in `<caller_message>` tags (untrusted), and instructions = `prompts.toml` text for the
   current phase + a JSON session-context block (phase, verified, today, available tools, the
   guard's caller review, remembered context, refusal and off-topic counters, and, only after
   verification, first name, selected claim, document status, email status).
5. Loop up to `MAX_STEPS = 8` agent calls (the last one with `tool_choice="none"`):
   * **function calls**: check that every call ID is answered exactly once, echo the output items
     (including reasoning items, since `store=False`), run each tool through `ToolExecutor`, append
     `function_call_output` items, and loop. Instructions and the tool menu are rebuilt on every
     step, so a successful `verify_identity` unlocks claim tools for the very next step.
   * **text**: strip markup and run `ReplyGuard.check`. Clean: done. Blocked: append a developer
     message naming the problems and loop again (at most `MAX_GUARDRAIL_RETRIES = 2`; then try a
     purely mechanical style fix; then the phase's safe fallback reply).
6. Turn tool side effects into messages: an `email_offer` card (summary text + masked address,
   shown with Send/Skip), or an email decision (skip: acknowledge; send: persist consent and hand a
   `PendingEmail` to the service, whose delivery result becomes the reply).

`TurnBudget` (`agent/budget.py`) caps agent calls (10), tool calls (`MAX_TOOL_CALLS_PER_TURN` + 4),
and wall-clock time (`TURN_DEADLINE_S`, 150 s by default); per-call timeouts are clipped to the
remaining time. Guard calls are not counted against the agent's call budget.

## 4. Tools and their guardrails (`agent/tools.py`)

Tool menu per phase (`BY_PHASE` + `ALWAYS`); the executor re-checks the menu on every call, so a
hallucinated or out-of-phase tool returns an explanation instead of acting.

| Phase | Tools |
| --- | --- |
| all | `request_human` |
| VERIFY_ID | `verify_identity` |
| RESOLVE_INTENT | `list_my_claims`, `select_claim` |
| PROCESS_CASE | + `get_claim_details`, `get_document_guidance`, `get_followup_guidance`, `record_document_status`, `offer_email_summary` |
| POST_PROCESS | claim tools + `record_email_decision` (only while an offer is active); `offer_email_summary` again only after a skip or a failed send |

Phase changes happen only as side effects of successful tools (`verify_identity` ->
RESOLVE_INTENT, `select_claim` -> PROCESS_CASE, `offer_email_summary` -> POST_PROCESS, any claim
tool after the offer -> PROCESS_CASE, withdrawing an unanswered offer) or of the guard's caller
review (a different person or someone acting for the policyholder -> VERIFY_ID).

| Freedom | Enforced by |
| --- | --- |
| Verification | `verify_identity` runs only when this turn's guard review says the person typing is the account holder. Then each value must normalize and appear in the caller's own messages since the last reset (`identity_value_grounded`); at least 3 distinct PII fields (policy number never counts) before any matching; deterministic match to exactly one policyholder with no conflicting field (`claims/verification.py`); per-session lockout after `MAX_VERIFICATION_FAILURES`, cross-session lockout per policyholder (fails closed) |
| Data access | Claim tools exist only after verification and are scoped to the verified party; another party's claim ID returns "not found" |
| Persuasion | Refusals are counted by code from the guard's review; `request_human(reason=repeated_refusal)` is refused before `MAX_REFUSALS` |
| Document status | `record_document_status` stores a status only when the guard reads the same status in the caller's words |
| Email summary | `offer_email_summary`: code requires the claim ID, the status, every outstanding document, the style rules, and grounded amounts and dates; then the guard fact checks every statement against the claim record, recorded document status, approved guidance, and the caller's words |
| Consent | `record_email_decision` only on an active offer, never in the same turn as the offer, and only when the guard reads the same choice in the caller's own words; the recipient is always the address on file |
| Repetition | The same tool with the same arguments twice in one turn is refused (except `verify_identity`) |
| Failures | Any tool exception becomes a `tool_error` result; the model apologises and can offer a human |

## 5. The guard (`agent/guard.py`, `[tasks.guard_*]` in `prompts.toml`)

The guard follows the pattern of a permission classifier: a separate model call, made by code
(never by the agent), that returns a verdict before an action is allowed. It runs every time, not
at selected moments: on every caller message (in), on every reply the agent drafts in every phase
(out), and on every action that changes what the application stores or sends. The only texts it
does not review are the application's own fixed messages (greeting, safe fallbacks, delivery
notices), which contain no model output.

| Checkpoint | When | Verdict | Code's response |
| --- | --- | --- | --- |
| `review_caller` | every typed message, before the agent | `speaker` (account_holder / acting_for_someone_else / unclear), `different_person`, `refused_verification`, `off_topic_request`, `claim_mentioned`, `identity_fields_given` (names only), `reason` | acting for someone else: blocked from verification for the session and offered a human, and access is withdrawn if already verified; a different person after verification: access withdrawn; refusals and off-topic requests counted (a human is offered at the limits); the described claim remembered; a reply that asks for more identity details after three were given, without trying `verify_identity`, is blocked |
| `judge_consent` | `record_email_decision` | `decision` (send / skip / unclear) | the agent's decision is recorded only if it matches |
| `judge_summary` | `offer_email_summary` | `supported`, `problems` | an unsupported or incomplete summary is rejected with the problems |
| `judge_document` | `record_document_status` | `status` the caller's words indicate | the status is stored only if it matches the agent's |
| `judge_reply` | every draft reply, every phase, after the code checks | `allowed`, `problems` with a category: `unsupported_fact`, `disclosed_before_verification`, `promise_or_invented_action`, `out_of_scope`, `internal_details`, `missing_required` | any problem blocks the reply and is sent back to the agent to fix |

The reply review gets the recent transcript, whether the caller is verified, the record (nothing
before verification; afterwards the claims on the account, the selected claim's full record with
what the caller said about each document), and the results of the claim tools the agent called
this turn. It checks every claim statement against them, so "the pathology report is complete and
on file" is blocked when the record lists it as required.

**Required content.** Some turns oblige the reply to say something. Code decides when a reply must
offer a human representative: the turn the off-topic or refusal limit is reached, when the caller
turns out to act for someone else, and when verification locks (`TurnEffects.human_offer_due`).
The agent is told in its context, the guard checks the reply makes the offer (`missing_required`),
and if no compliant reply is produced, the fallback reply makes the offer itself. Once a human
follow-up has been requested, no further offer is required.

* **Independent.** The guard has its own transport (`Runtime.guard_model`) and its own prompts, used
  alone (without the agent's prompts). It sees only what it judges, as one JSON document: the
  transcript, the offer and the caller's replies, the summary with the record, the caller's words
  about one document, or the reply with the record and this turn's claim tool results. It never
  sees the agent's reasoning or the arguments of its tool calls.
* **Untrusted text stays data.** Caller text is a JSON string value, so it cannot close a tag or
  pose as instructions; the prompts tell the guard to ignore instructions inside the data.
* **Strict output.** Each checkpoint uses a strict JSON schema (structured output) and the result
  is validated again with Pydantic (`extra="forbid"`).
* **Fails closed.** An error, timeout, refusal, truncated or invalid verdict becomes `None`, and
  every call site treats `None` as not allowed (no verification, no consent, no summary, the
  reply is rewritten or replaced by the safe fallback).
* **Settings.** `OPENAI_REASONING_EFFORT_GUARD` (default `medium`); traces keep only verdict flags,
  never the guard's free-text rationale.

## 6. Reply guardrails (`agent/reply_guard.py`, `agent/guardrails.py`)

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
* Every phase: a reply that asks for more identity details after the guard heard three, while
  `verify_identity` was not tried this turn, is sent back (`missed_verification`).
* Then the guard's full review (`judge_reply`) on every draft that passed the checks above.

## 7. State and persistence

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

## 8. Service (`web/service.py`)

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

## 9. Email (`mail/sender.py`)

`EmailDispatcher.dispatch` drives the ledger row `pending -> dispatching -> sent | queued | failed |
delivery_unknown` with compare-and-set updates. `OutboxTransport` writes an `.eml` file (local
demo, status `queued`, never delivered); `SmtpTransport` sends for real (`sent`). An ambiguous
transport error is reconciled when the transport supports lookup, otherwise recorded as
`delivery_unknown` and routed to a human; nothing is ever re-sent automatically. Addresses and
bodies are never logged.

## 10. Model transport (`llm/`)

`OpenAITransport` calls `client.responses.create` with `store=False`, strict function tools, and
`include=["reasoning.encrypted_content"]`, and converts the result into `LLMResponse` (text,
function calls, raw output items, usage). `ResilientTransport` retries only transient errors with
backoff and never sleeps past the turn deadline. `OfflineFakeModel` (`MODEL_PROVIDER=fake`) is a
small deterministic tool-using agent for demos and tests, and also answers the guard checkpoints with
keyword heuristics (offline only; in production both are the real model). `ChaosTransport` injects faults into
tool calls (extra, missing, mistyped, or malformed arguments, hallucinated tools, mismatched or
duplicate call IDs) and into replies (injected text, style violations, refusals, truncation,
timeouts, rate limits). `tests/agent/test_agent_chaos.py` runs the agent over it for 30 seeds and
checks the guardrails hold on every turn, with and without faults injected into the guard.

## 11. Observability (`observability/`)

Each turn writes one nested JSON trace: the `agent` span with phase before/after and stop reason,
`llm` spans (model, token use, tool names; never prompt or output text), `tool.<name>` spans,
and events for guardrail decisions, phase transitions, memory, email consent and results.
`Redactor` replaces names, dates of birth, contact details, ID digits, claim text, email bodies,
and API keys before serialization, and once more over the serialized JSON.

```bash
.venv/bin/python -m insurance_claims.observability.trace_reader traces/ --last 5
```

## 12. HTTP contract (`web/app.py`)

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

## 13. Frontend (`web/static/`)

Vanilla HTML/CSS/JS on a light theme. Server text is rendered with `textContent` only. A send
queue gives every message a `client_turn_id` and sends them in order, so Enter pressed while a
session is starting or reconnecting is never lost and retries are idempotent. The four-step
progress bar follows `state.steps`; Send/Skip buttons appear under the email offer while it is
active. The session ID is kept in `localStorage` so a refresh resumes the conversation.

## 14. Tests

`tests/` mirrors the package (`agent/`, `claims/`, `llm/`, `mail/`, `persistence/`,
`observability/`, `web/`) plus `browser/` (Playwright, opt-in with `-m browser`). The agent tests
script a misbehaving model (verifying with two fields, inventing identity values, calling claim
tools early, leaking claim data, selecting another party's claim, inventing amounts, consenting
in the same turn as the offer) and assert that code stops every attempt. `tests/agent/test_guard.py`
reproduces the external review's findings (a relative verified with the policyholder's details, an
unrelated question treated as consent, a false summary, an off-topic answer, memory and refusals
that depended on the agent calling a tool), states the guard's verdict explicitly with a stub, and
checks fail-closed behaviour for every kind of guard failure. No unit test calls the network;
`evals/live_eval.py` runs scripted conversations against the real model as both agent and guard.
