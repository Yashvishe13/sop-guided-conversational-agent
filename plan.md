# Implementation plan: SOP-guided insurance claims agent

## Goal and acceptance criteria

Build a FastAPI-backed Python conversational agent and a simple HTML/CSS/JavaScript chat UI for the fixture-backed insurance claims demo in `instructions.md`. Use `gpt-5.6-luna` as the backbone model. The application owns the workflow `VERIFY_ID -> RESOLVE_INTENT -> PROCESS_CASE -> POST_PROCESS`; the model extracts meaning and drafts natural language within each phase. A customer can ask clarifying questions, give partial answers, refuse, change topics, and return to a claim question without losing useful context.

The implementation is complete when a test user can open the browser UI, enter the Margaret Chen / `POL-9921` example as natural-language chat, and see the complete four-phase workflow: verification without early claim disclosure, remembered intent and case resolution, grounded claim processing, and a post-case email offer with send/skip choices. The same app must run from a documented Docker build using an API token supplied during setup. Unit and live evaluation must also show safe behavior for wrong identity, representatives, ambiguous claims, off-topic loops, emotional refusals, malformed model output, and tool failures.

## Skills to use during implementation

| Skill | Use in this project |
| --- | --- |
| `openai-agent` | Build the bounded OpenAI tool-calling loop, narrow tool schemas, dispatch, memory handoff, and explicit termination. |
| `openai-docs` | Verify the current Responses API, SDK, tool-calling, and structured-output syntax before coding; pin compatible dependencies and model configuration. |
| `agent-routing-and-guards` | Define phase transitions, protected gates, stop reasons, retry limits, and human handoff in application code. |
| `agent-structured-outputs` | Validate extracted identity fields, intent/case hints, affect/refusal signals, and proposed tool arguments before they affect state or actions. |
| `agent-durable-execution` | Make session memory and email operations recoverable across restarts, with checkpoints and idempotency. |
| `agent-security-boundaries` | Keep untrusted utterances and fixture text from changing authorization, enforce claim ownership, and redact PII in logs. |
| `agent-tracing` | Add nested JSON spans for each turn, model call, tool call, guard, and stop reason; make traces useful for debugging without recording raw secrets. |

The Responses API is the implementation target. Its function calls carry a `call_id` that must be paired with each function result, and strict schemas are supported for tools and structured outputs. The [official GPT-5.6 Luna model page](https://developers.openai.com/api/docs/models/gpt-5.6-luna) lists Responses, function calling, and structured outputs as supported. Also use the [official function-calling guide](https://developers.openai.com/api/docs/guides/function-calling) and [structured-output guide](https://developers.openai.com/api/docs/guides/structured-outputs).

## Architecture and work sequence

### 1. Establish the application contract and fixture adapters

- Create a small installable Python package under `apps/insurance_claims/` with a FastAPI app, typed domain models, and `pytest` tests. Add dependency metadata, a README with local and Docker setup/run commands, and `.gitignore` entries for `.env`, SQLite state, outbox artifacts, caches, and `traces/`. Accept the AI API auth token as `OPENAI_API_KEY` from the existing `.env` or a process environment variable; fail startup clearly when it is absent outside tests. Never commit, print, or return the token to the browser. Set `OPENAI_MODEL=gpt-5.6-luna` by default, with timeouts, retry and turn budgets, state path, and email transport configured on the server.
- Load and validate `policyholders.json`, `claims.json`, `representatives.json`, `required_document_guideline.json`, and `claim_schema.json` at startup. Normalize names, phones, emails, and dates for comparison while retaining canonical values. Treat fixture amounts as decimal strings and parse them with `Decimal` if arithmetic is needed.
- Put claim lookup behind a repository interface. Require an authenticated `party_id` in the executor, filter before returning any rows, and return a typed `not_found` or `ambiguous` result. Fixture descriptions are data, not instructions. Use an injected clock for date-sensitive statements; do not present an expired appeal deadline as upcoming.
- Define an email sender interface with a test/local outbox and a configurable real transport (for example SMTP). Real delivery requires a configured transport; never report an email as sent when it was only drafted or queued.

### 2. Build the FastAPI backend and browser chat UI

- Expose a small JSON API: `POST /api/sessions` starts a conversation; `POST /api/sessions/{id}/messages` accepts one user turn; `GET /api/sessions/{id}` restores permitted chat history and public workflow state after refresh; `GET /health` reports readiness. Keep the agent/controller behind a service interface so HTTP handlers contain no SOP logic. Use typed request/response models, input-size limits, consistent error payloads, and a request/turn ID for duplicate-submit handling.
- Serve a same-origin static `index.html`, CSS, and vanilla JavaScript from FastAPI. Provide a transcript, text input, send button, loading/error states, a visible four-step progress indicator, a new-conversation control, and explicit Send/Skip controls when the email offer is active. Show the current phase without showing protected claim data before verification. Make the interface keyboard-usable and readable on a phone-sized viewport.
- Keep the OpenAI token solely in server configuration; browser requests never carry it. Use an opaque, unguessable browser session credential tied to each conversation, protect session reads/writes from other clients, and set production-safe cookie/CSRF and transport settings. The client renders server-supplied text safely as text, never injected HTML. Refreshing the page should resume the same unexpired conversation.
- Package the FastAPI server, static UI, fixtures, and Python dependencies in a `Dockerfile`; add a compose example that passes `OPENAI_API_KEY` at runtime and mounts durable SQLite/outbox data. Document `docker compose up --build`, the local URL, how to supply the token without baking it into the image, how to run tests, and how to reset demo state. The repository and built image should be sufficient to run the demo.

### 3. Build durable session memory and the SOP controller

- Persist each session in SQLite, keyed by a stable opaque session ID: phase, version, turn index, candidate party reference, matched field **names** and protected comparison state, verification status/time, intent and case hints, selected case, conversation summary, unanswered questions, off-topic/refusal counters, consent decision, and pending email operation. Store a bounded recent turn history plus structured summary so a process restart retains context. Scope every read and write to the session/party; set a retention period and a deletion path.
- Keep raw PII out of summaries and traces. Store only what verification needs, protect the SQLite file with restrictive permissions, and use a deployment-appropriate encryption mechanism for durable PII. Persist state after each accepted turn and at each phase transition in one transaction with optimistic version checks to prevent concurrent turns from overwriting each other. Version the state schema and provide migration handling.
- Implement phase transitions as deterministic controller rules. The model may propose `identity_fields`, `intent`, `case_hint`, `affect`, `off_topic`, and `reply_intent`, but cannot set `verified`, `party_id`, phase, permission, or consent. A guard decides each transition. Keep verification active until at least **three distinct** fields from full name, DOB, phone, email, and the correct ID/SSN last four match one uniquely identified policyholder. A policy number helps find a candidate but is not one of the three PII fields. Alias values in fixtures can match their respective fields. Wrong or conflicting values never count. Do not leak which field failed or whether a named policyholder exists.
- Preserve useful hints even during `VERIFY_ID`; e.g., denial + healthcare + January remain untrusted hints until verification succeeds. After verification, use the saved hints to select among the authorized party's claims, including status and year where available. If more than one claim still fits, ask a targeted clarification. The representative fixture gives a relationship, not proof of claim access: offer human review rather than treating it as authorization.
- If a verified caller asks another case question during `POST_PROCESS`, return to `PROCESS_CASE` without repeating identity verification within the valid session. Define session expiry and require re-verification after expiry or a change of caller.

### 4. Add the model loop and grounded case handling

- Create a separate, versioned `apps/insurance_claims/prompts.toml` file and load it at application startup through a prompt loader. Keep prompt text out of Python route/controller code. Require named entries for global behavior, `VERIFY_ID`, `RESOLVE_INTENT`, `PROCESS_CASE`, `POST_PROCESS`, and any extraction or response-drafting task. Validate that every required entry exists and is nonempty; record only a prompt version/hash in traces. Package this file in the Docker image.
- Give each prompt a clear, phase-specific guideline and a common response style. The global guideline says to answer **only questions relevant to insurance claims and this customer-service workflow**. For unrelated requests, including general knowledge questions, politely decline and guide the customer back to their insurance claim; after repeated off-topic attempts, offer a human representative. It also says to sound natural and conversational, respond to emotion with empathy, and avoid em dashes (`—`) and colons (`:`) in every user-facing chat reply and generated email summary. Phase prompts state the allowed task and evidence, what must remain private, when to ask a clarification, and when to hand off. Prompt text guides the model, while code enforces scope, authorization, phase order, and consent.
- Use one agent with narrow tasks: extract conversational signals into a strict typed object, propose an allowed claim lookup/follow-up action, and draft a response from a compact evidence packet. Validate both schema and semantics; bound repair attempts and return a safe clarification or handoff if validation fails. Keep phase and tool availability controlled by code, not prompts.
- Offer read-only tools for authorized claim lookup and guideline lookup. Separate the email action from the model's tool menu: the application executes it only after explicit user consent and a verified recipient. Bound model steps, tool calls, tokens, elapsed time, and repeated no-progress attempts. Handle multiple tool calls, refusals, incomplete outputs, unknown tools, timeouts, rate limits, and empty responses with explicit stop reasons.
- In `PROCESS_CASE`, assemble evidence only from the selected authorized claim, schema definitions, and relevant document/follow-up guidance. Require response claims about status, denial reason, amounts, required documents, deadlines, and next steps to be supported by that evidence. Permit conversational paraphrase and empathy, but no invented outcomes, appeal rights, guarantees, or document-submission actions. Return a useful uncertainty statement or human handoff when the fixtures do not answer a question.
- Use a narrow insurance-service scope classifier with deterministic guard checks for sensitive actions. Politely decline unrelated questions. After a configurable small number of repeated irrelevant requests, offer a human representative. Detect frustration, anxiety, anger, confusion, and refusal; acknowledge emotion, explain the purpose of verification or consent, offer alternate allowed identity fields, and stop persuading after repeated refusal. Empathy never overrides a gate.
- Validate final user-facing text for the no-em-dash/no-colon style rule. If a draft violates it, request one bounded rewrite or use a safe fixed response; do not change claim facts or suppress an important warning to satisfy style. Structured API payloads, trace metadata, and timestamps are outside this prose-only rule.

### 5. Implement post-processing and optional email

- Build a structured summary from the verified conversation: topics discussed, selected claim's current status/outcome, source-backed follow-up items, and any uncertainty. Show or describe the proposed summary and ask clearly whether to send it. A skip decision closes the phase without sending.
- Accept explicit affirmative consent only in the active `POST_PROCESS` context. Send only to the verified policyholder email on record; if the user requests another address or is a representative without verified authority, route to human handling. No silent send, and no model-generated recipient or consent flag.
- Persist a stable email operation key, consent, destination reference, and draft hash before dispatch. Record a delivery receipt afterward. On timeout or restart, reconcile with the transport before retrying so one consent produces at most one send. If the transport cannot support reconciliation/idempotency, stop with an honest `delivery_unknown` status and route to manual review rather than risk duplicates.

### 6. Add tracing and operational safeguards

- Implement `agent-tracing`-style `span()`, `@traced`, OpenAI-call wrapping, and `walk()` in a small standard-library module. One nested JSON trace per turn/run should capture phase, state version, model/tool latency and token use, validated proposals, guard outcomes, source claim IDs, retries, and stop reason. Truncate large fields and use atomic file replacement on write.
- Adapt the skill's example to this PII-bearing workflow: redact or hash names, DOB, contact information, ID digits, claim text, email body, and API credentials **before** trace serialization. Store no raw model prompt/output when it contains those values. A trace should still explain why a gate held or a claim was selected. Test redaction using seeded secrets.
- Use structured errors, bounded retries with backoff for transient API failures, and clean user-facing recovery messages. Guard against stale state, duplicate turns, corrupt fixtures, missing configuration, and unavailable email transport. Keep control decisions and data-access checks functional if the model is unavailable.

## Verification and chaos testing

Unit tests use a deterministic fake model and fake tools. A seeded chaos layer mutates model replies and tool results, injects exceptions/timeouts, and reorders or duplicates calls; the tests assert **state and side effects**, not exact wording. Keep live API calls out of the unit suite so failures are reproducible and cheap.

| Test area | Required cases and invariant |
| --- | --- |
| Verification gate | Partial identity across turns; three distinct correct fields; aliases; wrong/conflicting fields; policy number not counted; adversarial model says `verified=true`; no claim data or claim-specific email before the gate. |
| Memory and recovery | January denial hint from the first utterance survives later verification and process restart; simultaneous turns cannot lose state; a corrupt/old checkpoint has explicit recovery behavior. |
| Case grounding | Correct `P9` January denial is found; unrelated party case and similar closed January claim are never exposed; ambiguous hints trigger clarification; unsupported claim assertions are blocked; expired dates are stated accurately using a frozen clock. |
| Conversational behavior | Clarification, refusal, anger, repeated off-topic prompts, prompt injection in user/fixture text, safe human handoff, and return to claim discussion after post-processing. |
| Prompts and response style | Every required prompt loads from `prompts.toml` and has an explicit guideline; missing or malformed entries fail clearly. Insurance-only answers, polite off-topic decline, natural wording, and absence of `—` and `:` are checked in chat replies and generated email summaries. |
| Email | Explicit yes, no, ambiguous answer, wrong recipient, send failure, timeout, duplicate turn, crash before/after dispatch; at most one authorized send and truthful delivery status. |
| Chaos/fault injection | Malformed JSON, extra/missing fields, invalid enum, hallucinated tool name, wrong `call_id`, parallel/duplicate function calls, empty/refusal/incomplete output, prompt-injection content, repeated same tool, API rate limit/timeout, and malformed fixture rows. No unsafe transition or external write may occur. |
| Trace privacy | Nested spans and token accounting are correct; seeded PII and `OPENAI_API_KEY` are absent from trace files, errors, and test snapshots. |
| HTTP and browser UI | FastAPI request validation, session isolation, duplicate submissions, reload/resume, text escaping, usable send/skip controls, and four-phase progress. Run a browser smoke test of the complete workflow with a fake model. |
| Docker delivery | Image builds; container starts with a supplied `OPENAI_API_KEY`; health and chat routes respond; SQLite data survives a container restart through the documented volume; the token is absent from image layers and static assets. |

Add a small **separate** opt-in live-model evaluation using `gpt-5.6-luna` and `OPENAI_API_KEY`: run fixed multi-turn scripts plus a few seeded phrasing variations, record pass/fail for gate compliance, case grounding, hint reuse, empathy, scope refusal, and email consent. Save redacted traces and report observed failures. Do not treat live-model output as a deterministic unit-test oracle. Run the regular unit suite, browser smoke test, and Docker startup check before declaring the implementation complete.

## Key deliverables

1. Docker image and repository with a `Dockerfile`, compose example, and clear local/Docker setup instructions that accept `OPENAI_API_KEY` at runtime.
2. FastAPI backend and simple HTML/CSS/JavaScript chat UI for a natural-language test conversation showing all four SOP phases.
3. `gpt-5.6-luna`-powered agent with a separate, loaded `prompts.toml`, clear per-prompt guidelines, a deterministic SOP controller, validated model proposals, authorized fixture tools, and grounded replies.
4. Persistent, versioned SQLite session memory that survives restarts and carries early case hints forward.
5. Optional, consent-gated email summary with idempotent delivery handling and truthful status.
6. Redacted nested JSON tracing with a reader for tool paths, guard decisions, and token use.
7. Deterministic unit/chaos suite, API/browser/Docker smoke checks, opt-in live-model evaluation scripts, and a results report covering the sample conversation and safety invariants.
