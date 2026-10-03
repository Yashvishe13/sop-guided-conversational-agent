# Verification results

Date: 2026-10-03. Version 0.2.0. Model: `gpt-5.6-luna` (Responses API). Prompt set `2026-10-03.agent-4`.

This report covers the sample conversation from `instructions.md`, the safety rules in `plan.md`,
and how each was verified on the current ReAct agent (see `docs/DESIGN.md`).

## Summary

| Check | How | Result |
| --- | --- | --- |
| Unit, chaos, API, privacy suite | `pytest` with the offline fake model and scripted misbehaving models, no network | **1,630 passed** (incl. the agent under fault injection and an import check of every module) |
| Browser smoke and recovery | Playwright against the real app (`pytest -m browser`) | **9 passed** |
| Live model evaluation | Scripted multi-turn runs against `gpt-5.6-luna` (`evals/live_eval.py`) | **15 / 15 passed** |
| Docker | `docker compose up -d --build`, `/health`, full conversation over HTTP with the real model, trace reader | Pass |
| Chrome walkthrough | Claude in Chrome against the Docker app with the real model | Pass |

## The sample conversation

Caller, first message:

> I'm the policyholder. My name is Margaret Chen, policy POL-9921. I'm calling about my denied healthcare claim from January. DOB is 1985-03-15, SSN last four is 4472.

What happens in that one turn (from the redacted trace):

```
llm                 model calls note_caller_context(denied, healthcare, January) and verify_identity(...)
tool.verify_identity  values grounded in the caller's text; 3 PII fields (policy number not counted); matched P9
phase_transition    VERIFY_ID -> RESOLVE_INTENT (identity_verified); claim tools now on the menu
tool.list_my_claims / tool.select_claim   CL-2048 chosen from the remembered hint (CL-2011 is closed)
phase_transition    RESOLVE_INTENT -> PROCESS_CASE (claim_selected)
tool.get_claim_details  facts for the reply
reply guard         style, grounding, status, deadline checks pass
```

Reply (Chrome, after the cleanup):

> You're verified, Margaret. Claim CL-2048 was denied because the review file was missing the pathology report and the treating provider's office note. The appeal deadline was March 18, 2026, and has passed, but the claim record identifies those documents as needed for review. Do you have the pathology report, or can you request it from your provider?

"Yes I can request both from my doctor. That's all, thanks." then moved to Wrap up with a summary
card (claim ID, status, passed deadline, both documents), the masked on-file address, and Send/Skip
buttons. Send email reported the honest outbox result ("saved to the local demo outbox... nothing
was sent to your inbox").

## Safety rules and where they are enforced

| Requirement | Enforced by | Tests |
| --- | --- | --- |
| Three distinct correct PII fields of one policyholder; policy number never counts; no hint of which field failed | `verify_identity` guardrails + `claims/verification.py` | `tests/claims/test_verification.py`, `tests/agent/test_agent.py` |
| Model cannot invent identity values or set verified/phase/consent | Values must appear in the caller's messages; state changes only inside tools | `tests/agent/test_agent.py` (scripted misbehaving model) |
| No claim data before verification | Claim tools absent before verification; reply leak check over every claim in the dataset | `tests/agent/test_agent.py`, `tests/agent/test_agent_chaos.py` |
| Brute force across sessions | Per-policyholder failure ledger, fails closed | `tests/persistence/test_store.py`, `tests/agent/test_agent.py` |
| Representative is not the policyholder | Sticky `representative_declared`; `verify_identity` returns not_allowed | `tests/agent/test_agent.py` |
| Early hints remembered and reused | `note_caller_context` in any phase; returned on verification | `tests/agent/test_agent.py`, live `hint_reuse` |
| Answers only from the caller's claim and guidelines | Party-scoped tools; reply grounding (IDs, amounts, dates, status, claim descriptions, deadline wording, promises) | `tests/agent/test_guardrails.py`, `tests/agent/test_agent.py` |
| Off-topic declined, human after repeats; persuasion before handoff | `flag_off_topic`, `note_refusal`, `request_human` gate | `tests/agent/test_agent.py`, live `off_topic_loop` |
| No em dash or colon | `check_style` on replies and on the email summary | `tests/agent/test_guardrails.py`, `tests/agent/test_agent_chaos.py` |
| No internal vocabulary (phase names, party IDs, prompt words) | `check_internal_reference` in `ReplyGuard` | `tests/agent/test_agent.py`, `tests/agent/test_guardrails.py` |
| Explicit consent, on-record recipient, at most one send, truthful status | Offer then a later explicit choice; op key persisted before dispatch; ledger reconciliation | `tests/mail/test_sender.py`, `tests/web/test_email_recovery.py` |
| Durable memory, concurrency, migrations | Encrypted SQLite checkpoints, optimistic versions, turn idempotency, schema migrations (v1 -> v3) | `tests/persistence/test_store.py`, `tests/web/test_service.py`, `tests/web/test_api.py` |
| Traces without PII or secrets | Redaction before serialization; LLM spans keep metadata and tokens only | `tests/observability/` |
| Malformed tool arguments, unknown tools, wrong call IDs, injected text, timeouts | Call-ID pairing check, tool menu re-check, argument validation, reply guard, budgets, retries, safe fallback | `tests/agent/test_agent_chaos.py` (30 seeds), `tests/llm/test_chaos.py` |

## Robustness rounds (real model)

* Emotional support, following `instructions.md`: stress acknowledged first; "I already told you
  who I am" gets empathy, the reason for verification, and alternative fields; a second refusal
  stops the pressing and offers a human; anger and "guarantee it" get an honest no.
* 31 messy, realistic conversations: corrections, a claim ID given directly, another customer's
  claim ID (refused), several questions at once, a policyholder with no claims, aliases, many date
  and phone formats, a new-claim request (human), injection after verification, "submit this for
  me" (declined), memory recall, impersonation, brute force (locked after 3 attempts), Spanish.
* Defects found in those rounds were fixed with regression tests: claim descriptions with the
  wrong month or year, identity values passed with surrounding words, unexplained ignored fields,
  "skip" then "actually send it", and a premature human handoff.

## Code cleanup (0.2.0)

* Package split into `agent/`, `claims/`, `domain/`, `llm/`, `mail/`, `persistence/`,
  `observability/`, `web/`; tests mirror it. Each package has a docstring explaining its role.
* `agent/loop.py` reads top to bottom: handle a turn, the ReAct loop, tool execution, turning tool
  effects into messages. Reply checks moved to `agent/reply_guard.py`; all fixed texts are named
  constants at the top of the loop.
* Removed everything left over from the earlier extract-then-control pipeline: the structured
  output path (`text_format`, `responses.parse` tracing, sample signal schemas), utterance and
  intent based follow-up matching, the consent-scenario loader, migrations from the old checkpoint
  and database formats, and unused helpers and settings. Old saved conversations are reported as
  unrecoverable and the UI starts a new one.
* Found while removing: the internal-reference check (phase names, party IDs) was only reachable
  through the old `check_reply`, so the ReAct agent was not running it. It now runs in `ReplyGuard`
  on every reply, with a regression test.
* The fault injector was rebuilt for tool calls and now drives the real agent loop
  (`tests/agent/test_agent_chaos.py`); it immediately found the offline fake model crashing on a
  failed tool result (fixed).
* Added a project `ruff` config (lint and format pass) and a test that imports every module (it
  caught a stale import in the live eval script after the move). The project is now a git repository.

## Known limitations

* The email outbox is a local demo transport: it writes `.eml` files and always reports "saved,
  not delivered". SMTP is supported but cannot reconcile after a timeout, so an ambiguous SMTP
  timeout is reported as `delivery_unknown` and routed to a person rather than resent.
* A single server process is assumed (per-session locks are in-process).
* The per-policyholder lockout trades availability for safety: someone who knows a name and date
  of birth can lock that policyholder out for `PARTY_FAILURE_WINDOW_HOURS`.
* Without `STATE_ENCRYPTION_KEY`, the generated key sits in the same volume as the database.
* "Today" for deadline statements is the server's UTC date unless `APP_TODAY` is set.
* If the model is unavailable, the agent replies with a safe per-phase fallback; it does not
  answer claim questions without the model.
* Live model output varies between runs; the live evaluation is a monitoring signal, not an oracle.
