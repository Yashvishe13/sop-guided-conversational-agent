"""SOP-guided insurance claims support agent.

Package map (read in this order):

* ``agent``          The ReAct agent: the tool loop (``loop``), the tools and their code guardrails
                     (``tools``), reply checks (``reply_guard`` on top of ``guardrails``), per-turn budgets, and the prompt loader.
* ``claims``         Trusted claim data: fixture loading, party-scoped repositories, identity
                     normalization and matching (``verification``), and the evidence packets replies are checked against.
* ``domain``         Shared vocabulary, the persisted ``SessionState``, and ledger records.
* ``llm``            Model transports: OpenAI Responses API, retries, an offline fake, and a chaos layer for tests.
* ``mail``           Consent-gated, idempotent email dispatch with truthful delivery status.
* ``persistence``    Encrypted SQLite store for sessions, turns, the email ledger, and verification failures.
* ``observability``  Redacted nested JSON traces per turn, and a trace reader CLI.
* ``web``            FastAPI app (HTTP contract, cookies, CSRF, limits), the conversation service, and the static UI.
* ``config``         All server-side settings, read from the environment or ``.env``.
"""

__version__ = "0.2.0"
