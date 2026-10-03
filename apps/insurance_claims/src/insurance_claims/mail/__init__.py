"""Email delivery for the post-case summary: transports (local outbox, SMTP) and an idempotent,
reconcilable dispatcher that sends at most once per consent and never reports a send it cannot prove."""
