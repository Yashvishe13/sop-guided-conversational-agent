# SOP-guided insurance claims agent

A conversational insurance-claims support agent that follows a fixed workflow
(`VERIFY_ID -> RESOLVE_INTENT -> PROCESS_CASE -> POST_PROCESS`) while still
talking naturally. The model (`gpt-5.6-luna` through the OpenAI Responses API)
runs a ReAct loop: it reasons about the conversation and calls tools such as
`verify_identity`, `select_claim`, and `get_claim_details`. Code enforces what
must never bend: tools only change state when their guardrails pass (three
matching identity fields the caller actually said, the caller's own claims only,
explicit email consent), and every reply is checked before it is sent.

* FastAPI backend with a same-origin HTML/CSS/JavaScript chat UI
* ReAct agent (Responses API tool loop) whose tool menu changes by phase
* Guardrails in code: three-field identity gate, party-scoped claim tools, grounded and style-checked replies, consent-gated email
* Encrypted, versioned SQLite session memory that survives restarts
* Consent-gated, idempotent email summary with truthful delivery status
* Redacted nested JSON traces per turn and a trace reader
* Deterministic unit and chaos suite, opt-in live-model evaluation

See [`docs/DESIGN.md`](docs/DESIGN.md) for the architecture and [`docs/RESULTS.md`](docs/RESULTS.md) for verification results.

## Quick start (Docker)

From the repository root:

```bash
cp .env.example .env            # then put your key in .env as OPENAI_API_KEY=...
docker compose up --build
open http://localhost:8000
```

`compose.yaml` reads secrets from your shell or `./.env` at runtime and passes them to the
container as environment variables. They are never copied into the image (`.env` is excluded from
the build context by `.dockerignore`, and the Dockerfile has no key argument). To supply the key
without a file:

```bash
OPENAI_API_KEY=sk-... docker compose up --build
```

Stored conversation state is always encrypted. `STATE_ENCRYPTION_KEY` is optional: when it is unset,
the app generates a key at `/data/state.key` inside the `claims-data` volume, which is fine for a
demo. For a real deployment, set `STATE_ENCRYPTION_KEY` (generate one with
`python3 -c "import base64, os; print(base64.urlsafe_b64encode(os.urandom(32)).decode())"`) so the key
does not live next to the data it protects, and keep it stable: data written under one key cannot be
read with another. If you switch keys, reset the volume with `docker compose down -v`.

To run the offline demo model without an OpenAI key, set `MODEL_PROVIDER=fake` (in `.env` or the
shell) and leave `OPENAI_API_KEY` empty. With `MODEL_PROVIDER=openai` (the default) the server
refuses to start until the key is set.

Session state, the email outbox, and traces live in the `claims-data` volume mounted at `/data`, so
conversations survive `docker compose restart`. Reset all demo state with:

```bash
docker compose down -v
```

## Quick start (local)

Requires Python 3.12+.

```bash
cd apps/insurance_claims
python3.12 -m venv .venv
.venv/bin/pip install -e ".[dev]"
.venv/bin/python -m insurance_claims          # http://127.0.0.1:8000
```

The server reads settings from the process environment and the first `.env` found at
`apps/insurance_claims/.env` or the repository root (`DOTENV_PATH` overrides). A non-empty process
variable wins over the file; an empty one counts as unset. In `.env`, `KEY=value  # note` drops the
trailing comment (as docker compose does); quote a value that must contain ` #`. The server fails to
start with a clear message when `OPENAI_API_KEY` is missing. `MODEL_PROVIDER=fake` runs a
deterministic offline model instead, which is useful for UI work and for tests. Without
`STATE_ENCRYPTION_KEY`, a local run generates `data/state.key` (fine for a laptop, not for a shared
or backed-up disk).

If Python reports `No module named insurance_claims` on macOS, the editable install's `.pth` file
in the virtualenv was created with the macOS "hidden" flag, and recent Python releases skip hidden
`.pth` files. Either clear the flag or put `src` on the path:

```bash
chflags nohidden .venv/lib/python3.12/site-packages/*.pth
PYTHONPATH=src .venv/bin/python -m insurance_claims   # alternative
```

Reset local demo state: `rm -rf apps/insurance_claims/data apps/insurance_claims/traces`.

## Try the demo conversation

Type this as the first message:

> I'm the policyholder. My name is Margaret Chen, policy POL-9921. I'm calling about my denied
> healthcare claim from January. DOB is 1985-03-15, SSN last four is 4472.

The agent verifies the caller with three identity fields (the policy number is only a locator),
remembers the denied healthcare January hint, selects claim `CL-2048` without asking again, and
explains the denial from the claim record. Ask follow-ups such as "What documents do you need and
how do I send them?" or "How long does review take?", then say "that's all" to get the email
summary offer with Send and Skip buttons.

Other things to try: give the details one at a time; give a wrong date of birth; say "I'm calling
for my mother"; ask "what is RL?" three times; say "I already told you who I am, just tell me why
my claim was denied"; ask about "my auto claim" after the healthcare one.

Set `APP_TODAY=2026-10-03` to freeze the date used for deadline statements.

## Configuration

| Variable | Default | Meaning |
| --- | --- | --- |
| `OPENAI_API_KEY` | (required for `openai`) | OpenAI token, server side only |
| `OPENAI_MODEL` | `gpt-5.6-luna` | Model for extraction and drafting |
| `MODEL_PROVIDER` | `openai` | `fake` for the offline deterministic model |
| `OPENAI_REASONING_EFFORT_REPLY` | `low` | Reasoning effort for each agent step |
| `OPENAI_TIMEOUT_S`, `OPENAI_MAX_RETRIES` | `25`, `2` | Per-call timeout and transient-error retries |
| `TURN_DEADLINE_S` | `75` | Wall-clock budget for one turn |
| `MAX_TOOL_CALLS_PER_TURN` | `6` | Tool-call budget per turn (model steps are capped at 8 plus 2 guardrail retries in code) |
| `MAX_OFF_TOPIC` | `3` | Off-topic requests before a human is offered |
| `MAX_REFUSALS` | `2` | Verification refusals before the agent stops persuading |
| `MAX_VERIFICATION_FAILURES` | `3` | Failed attempts in one session before identity checks lock |
| `MAX_PARTY_FAILURES`, `PARTY_FAILURE_WINDOW_HOURS` | `5`, `24` | Failed verification attempts against one policyholder, across all sessions, within the window before that policyholder's verification locks |
| `VERIFICATION_IDLE_TTL_MINUTES` | `30` | Idle time before re-verification is required |
| `SESSION_MAX_AGE_HOURS`, `RETENTION_DAYS` | `24`, `7` | Resume window and data retention |
| `DATA_DIR` | `./data` | SQLite, outbox, key file |
| `STATE_ENCRYPTION_KEY` | generated `DATA_DIR/state.key` | Fernet key for stored state. Optional for demos; required when `APP_ENV=production`; keep it outside `DATA_DIR` |
| `TRACES_ENABLED`, `TRACE_DIR` | `true`, `./traces` | Redacted per-turn traces |
| `EMAIL_TRANSPORT` | `outbox` | `outbox` (local demo, never delivered), `smtp`, or `disabled` |
| `SMTP_HOST`, `SMTP_PORT`, `SMTP_USERNAME`, `SMTP_PASSWORD`, `SMTP_STARTTLS`, `EMAIL_FROM` | | SMTP transport |
| `APP_ENV` | `development` | `development`, `production`, or `test` |
| `COOKIE_SECURE` | on in production | Secure flag for the session cookie (use HTTPS) |
| `APP_TODAY` | real date | Frozen date for deadline statements |

## Tests

```bash
cd apps/insurance_claims
.venv/bin/python -m pytest                       # unit + chaos + API suite, no network; browser tests excluded
.venv/bin/python -m pytest -m browser            # only the browser tests (needs: pip install -e ".[browser]" && playwright install chromium)
RUN_LIVE_EVAL=1 .venv/bin/python -m evals.live_eval --seeds 2   # opt-in, calls the real API
```

The default run uses `-m 'not browser'` from `pyproject.toml`; passing `-m browser` replaces it.
No test calls the OpenAI API. `pyproject.toml` puts `src` on the test path, so the hidden `.pth`
issue above does not affect pytest.

## HTTP hardening

Request bodies over 16 KB are rejected with `413` whether or not the client sends `Content-Length`
(chunked bodies are counted as they stream). State-changing `/api` requests (POST, DELETE) from a
browser must come from the same origin: when `Origin` (or `Referer`) names another host, or
`Sec-Fetch-Site` says `cross-site`/`same-site`, the server answers `403 cross_site_request`.
Behind a reverse proxy, forward the original `Host` (or set `X-Forwarded-Host`). Every response,
including errors and `500`s, carries the security headers, `X-Request-ID`, and `Cache-Control:
no-store` on API paths.

## Traces

Each turn writes one nested JSON trace to `TRACE_DIR` with phase transitions, guard decisions,
validated (redacted) proposals, tool calls, token use, retries, and stop reasons. Names, dates of
birth, contact details, ID digits, claim text, email bodies, and API keys are redacted before
serialization, and raw prompts and model outputs are never stored.

```bash
.venv/bin/python -m insurance_claims.observability.trace_reader traces/ --last 5
```

## Layout

```
compose.yaml                      (repository root) Docker Compose service and volume
apps/insurance_claims/
  prompts.toml                    versioned agent instructions, one section per phase
  fixtures/                       demo policyholders, claims, guidelines, claim schema
  Dockerfile
  docs/DESIGN.md                  architecture and module contracts
  docs/RESULTS.md                 verification results
  evals/live_eval.py              opt-in evaluation against the real model
  src/insurance_claims/
    __main__.py                   `python -m insurance_claims` starts uvicorn
    config.py                     Settings, read from the environment and .env
    agent/                        the ReAct agent
      loop.py                       ClaimsAgent: one turn = model -> tools -> model ... -> checked reply
      tools.py                      tool schemas, per-phase tool menu, ToolExecutor (tool guardrails)
      reply_guard.py                ReplyGuard: checks every draft reply before it is sent
      guardrails.py                 low-level reply checks (style, leaks, grounding, deadlines)
      prompts.py                    loads and validates prompts.toml
      budget.py                     per-turn model/tool/deadline budget
    claims/                       claim data and identity matching (no model calls)
      fixtures.py, repository.py    load fixtures; party-scoped claim access
      normalize.py, verification.py normalize identity values; deterministic matching and lockout
      evidence.py                   claim facts packaged for the model and for grounding checks
    domain/                       data shapes
      models.py                     fixture records, phases, PII field names
      state.py                      SessionState, the persisted checkpoint
      records.py                    email-ledger and turn records
    llm/                          model transport
      base.py, openai_transport.py  transport interface; Responses API client
      resilient.py                  bounded retries with backoff, clipped to the turn deadline
      fake.py, chaos.py             offline demo model; fault injection for tests
    mail/sender.py                consent-gated, idempotent email (outbox or SMTP)
    persistence/store.py          encrypted SQLite store with versions and migrations
    observability/                redaction, nested JSON traces, trace reader
    web/                          FastAPI app, session service, static chat UI
  tests/                          one folder per package, plus browser tests
```
