"""Email dispatch: consent-gated, idempotent, reconcilable, PII-silent.

The ledger here is an in-memory fake of the ``EmailLedger`` protocol (the real
one lives in ``store.py``). Invariants checked throughout: one consent causes
at most one external delivery, a timeout is never treated as a failure, the
op row is claimed before the transport is called, failed/unknown rows are
never re-sent, the outbox is never reported as ``sent``, and no address,
subject, or body reaches logs, traces, reprs, or error messages.
"""

from __future__ import annotations

import email
import email.policy
import logging
import smtplib
import stat
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Callable

import pytest

from insurance_claims.config import Settings
from insurance_claims.domain.records import EmailOpRecord
from insurance_claims.mail import sender as emailer
from insurance_claims.mail.sender import (
    DeliveryReceipt,
    DisabledTransport,
    DispatchOutcome,
    EmailDispatcher,
    EmailMessage,
    EmailTransportError,
    OutboxTransport,
    ScriptedTransport,
    SimulatedCrash,
    SmtpTransport,
    make_op_key,
    summary_hash,
    transport_from_settings,
)
from insurance_claims.observability import tracing as tracing_module

TO = "margaret@email.com"
FROM = "claims-support@example.com"
SUBJECT = "Your claim summary subj-91c2"
BODY = (
    "Hello Margaret, thanks for calling about claim CL-2048. Marker body-7f3a.\n"
    "We discussed why the claim was denied and the documents still needed.\n"
    "Next steps include sending the original pathology report and the treating provider office note, "
    "which together usually take about ten business days to review once they arrive.\n"
    "Café résumé ✓ naïve\n"
)
NOW = datetime(2026, 10, 3, 12, 0, tzinfo=UTC)
LATER = NOW + timedelta(minutes=5)
DELIVERED = frozenset({"sent", "queued"})
SESSION = "sess-abc123"
PII_MARKERS = (TO, "margaret@", "body-7f3a", "subj-91c2", "pathology report", "Hello Margaret")


def fixed_clock() -> datetime:
    return NOW


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class MemoryLedger:
    """In-memory ``EmailLedger`` with compare-and-set semantics and fault hooks.

    Like ``store.SessionStore`` it stamps ``updated_at`` from its own clock on
    every write, keeps stored receipt/detail when given None, never re-arms a
    row to ``pending``, and never un-delivers.
    """

    def __init__(self) -> None:
        self.rows: dict[str, EmailOpRecord] = {}
        self.transitions: list[tuple[str, str, str]] = []
        self.refuse_claim = False
        self.on_claim: Callable[[MemoryLedger, str], None] | None = None
        self.fail_get = False
        self.fail_update_to: set[str] = set()
        self.raise_after_update_to: set[str] = set()
        self.clock: Callable[[], datetime] = fixed_clock

    def seed(self, op_key: str, *, draft_hash: str, status: str = "pending", **extra: Any) -> EmailOpRecord:
        row = EmailOpRecord(
            op_key=op_key,
            session_id=SESSION,
            recipient_ref="P9",
            draft_hash=draft_hash,
            status=status,  # type: ignore[arg-type]
            created_at=NOW,
            updated_at=NOW,
            **extra,
        )
        self.rows[op_key] = row
        return row

    def status(self, op_key: str) -> str:
        return self.rows[op_key].status

    def email_op_get(self, op_key: str) -> EmailOpRecord | None:
        if self.fail_get:
            raise RuntimeError("database is locked")
        row = self.rows.get(op_key)
        return replace(row) if row is not None else None

    def email_op_update(
        self,
        op_key: str,
        *,
        status: str,
        receipt_id: str | None = None,
        detail: str | None = None,
        expected_status: str | None = None,
    ) -> bool:
        if status in self.fail_update_to:
            raise RuntimeError("database is locked")
        row = self.rows.get(op_key)
        if row is None:
            return False
        if expected_status == "pending" and status == "dispatching":
            if self.on_claim is not None:
                self.on_claim(self, op_key)
            if self.refuse_claim:
                return False
        if expected_status is not None and row.status != expected_status:
            return False
        if row.status != status and (status == "pending" or (row.status in DELIVERED and status not in DELIVERED)):
            return False
        self.transitions.append((op_key, row.status, status))
        row.status = status  # type: ignore[assignment]
        row.receipt_id = receipt_id if receipt_id is not None else row.receipt_id
        row.detail = detail if detail is not None else row.detail
        row.updated_at = self.clock()
        if status in self.raise_after_update_to:  # the write landed, but the caller sees an error
            raise RuntimeError("disk I/O error")
        return True


class LedgerCheckingTransport(ScriptedTransport):
    """Records the ledger status at the moment ``send`` runs."""

    def __init__(self, ledger: MemoryLedger, behaviors: tuple[str, ...] = ("ok",), **kwargs: Any) -> None:
        super().__init__(behaviors, **kwargs)
        self.ledger = ledger
        self.status_at_send: list[str] = []

    def send(self, message: EmailMessage, *, timeout: float) -> DeliveryReceipt:
        self.status_at_send.append(self.ledger.status(message.op_key))
        return super().send(message, timeout=timeout)


class CountingOutbox(OutboxTransport):
    def __init__(self, outbox_dir: Path) -> None:
        super().__init__(outbox_dir, clock=fixed_clock)
        self.send_count = 0

    def send(self, message: EmailMessage, *, timeout: float) -> DeliveryReceipt:
        self.send_count += 1
        return super().send(message, timeout=timeout)


def seed_op(ledger: MemoryLedger, *, status: str = "pending", body: str = BODY, turn: int = 7, **extra: Any) -> str:
    draft = summary_hash(body)
    key = make_op_key(SESSION, draft, turn)
    ledger.seed(key, draft_hash=draft, status=status, **extra)
    return key


def make_dispatcher(transport: Any, ledger: Any, *, clock: Callable[[], datetime] = fixed_clock) -> EmailDispatcher:
    return EmailDispatcher(transport, ledger, from_address=FROM, timeout=5.0, clock=clock)


def later_clock() -> datetime:
    """Past the default claim TTL (60s for a 5s timeout): a claim stamped at NOW is orphaned."""
    return LATER


def stale_dispatcher(transport: Any, ledger: MemoryLedger) -> EmailDispatcher:
    """A restarted worker some minutes after the crash (its ledger writes are stamped then too)."""
    ledger.clock = later_clock
    return make_dispatcher(transport, ledger, clock=later_clock)


def run(dispatcher: EmailDispatcher, key: str, **overrides: str) -> DispatchOutcome:
    params = {"to_address": TO, "subject": SUBJECT, "body_text": BODY, **overrides}
    return dispatcher.dispatch(op_key=key, **params)


def message(op_key: str = "em_" + "a" * 32, **overrides: str) -> EmailMessage:
    fields = {"to_address": TO, "from_address": FROM, "subject": SUBJECT, "body_text": BODY, **overrides}
    return EmailMessage(op_key=op_key, **fields)


@pytest.fixture
def ledger() -> MemoryLedger:
    return MemoryLedger()


@pytest.fixture
def outbox_dir(tmp_path: Path) -> Path:
    return tmp_path / "data" / "outbox"


# ---------------------------------------------------------------------------
# Keys, hashes, value types
# ---------------------------------------------------------------------------


class TestKeys:
    def test_summary_hash_is_stable_sha256_of_exact_text(self) -> None:
        assert summary_hash(BODY) == summary_hash(BODY)
        assert len(summary_hash(BODY)) == 64
        assert summary_hash(BODY) != summary_hash(BODY + " ")
        assert summary_hash("") == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"

    def test_summary_hash_rejects_non_string(self) -> None:
        with pytest.raises(TypeError):
            summary_hash(None)  # type: ignore[arg-type]

    def test_make_op_key_stable_and_distinct_per_consent_turn(self) -> None:
        draft = summary_hash(BODY)
        key = make_op_key(SESSION, draft, 7)
        assert key == make_op_key(SESSION, draft, 7)
        assert key != make_op_key(SESSION, draft, 8)
        assert key != make_op_key("sess-other", draft, 7)
        assert key != make_op_key(SESSION, summary_hash("other"), 7)

    def test_make_op_key_is_filename_safe_and_hides_session_id(self) -> None:
        key = make_op_key(SESSION, summary_hash(BODY), 0)
        assert key.startswith("em_") and len(key) == 35
        assert key.replace("_", "").isalnum()
        assert SESSION not in key

    @pytest.mark.parametrize(
        ("session_id", "draft", "turn"),
        [("", "d", 1), ("  ", "d", 1), ("s", "", 1), ("s", "d", -1), ("s", "d", True), ("s", "d", "3")],
    )
    def test_make_op_key_rejects_bad_inputs(self, session_id: str, draft: str, turn: Any) -> None:
        with pytest.raises(ValueError):
            make_op_key(session_id, draft, turn)

    def test_message_repr_hides_address_subject_and_body(self) -> None:
        text = repr(message())
        for marker in PII_MARKERS:
            assert marker not in text

    def test_transport_error_carries_kind_and_certainty(self) -> None:
        exc = EmailTransportError("timeout", definitely_not_sent=False, message="smtp_send_timeout")
        assert exc.kind == "timeout" and exc.definitely_not_sent is False
        assert "smtp_send_timeout" in str(exc)
        with pytest.raises(ValueError):
            EmailTransportError("exploded", definitely_not_sent=True)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# OutboxTransport
# ---------------------------------------------------------------------------


def eml_files(directory: Path) -> list[Path]:
    return sorted(directory.glob("*.eml")) if directory.exists() else []


class TestOutboxTransport:
    def test_writes_one_private_rfc5322_file_with_queued_receipt(self, outbox_dir: Path) -> None:
        transport = OutboxTransport(outbox_dir, clock=fixed_clock)
        msg = message()
        receipt = transport.send(msg, timeout=1.0)

        files = eml_files(outbox_dir)
        assert files == [outbox_dir / f"{msg.op_key}.eml"]
        assert stat.S_IMODE(files[0].stat().st_mode) == 0o600
        assert stat.S_IMODE(outbox_dir.stat().st_mode) == 0o700

        parsed = email.message_from_bytes(files[0].read_bytes(), policy=email.policy.default)
        assert parsed["To"] == TO
        assert parsed["From"] == FROM
        assert parsed["Subject"] == SUBJECT
        assert parsed["Message-ID"] == f"<{msg.op_key}@example.com>"
        assert parsed["X-Claims-Op-Key"] == msg.op_key
        assert parsed.get_content().replace("\r\n", "\n") == BODY

        assert receipt.status == "queued"
        assert receipt.transport == "outbox"
        assert receipt.message_id == f"<{msg.op_key}@example.com>"
        assert receipt.op_key == msg.op_key
        assert receipt.at == NOW

    def test_is_never_reported_as_sent(self, outbox_dir: Path) -> None:
        transport = OutboxTransport(outbox_dir, clock=fixed_clock)
        assert transport.send(message(), timeout=1.0).status == "queued"
        assert transport.lookup(message().op_key).status == "queued"  # type: ignore[union-attr]

    def test_resend_same_key_returns_same_receipt_without_rewriting(self, outbox_dir: Path) -> None:
        transport = OutboxTransport(outbox_dir, clock=fixed_clock)
        first = transport.send(message(), timeout=1.0)
        path = eml_files(outbox_dir)[0]
        before = (path.read_bytes(), path.stat().st_mtime_ns, path.stat().st_ino)

        later = OutboxTransport(outbox_dir, clock=lambda: NOW + timedelta(hours=1))
        second = later.send(message(subject="Changed subject"), timeout=1.0)

        assert eml_files(outbox_dir) == [path]
        assert (path.read_bytes(), path.stat().st_mtime_ns, path.stat().st_ino) == before
        assert second.message_id == first.message_id
        assert second.at == first.at == NOW
        assert second.status == "queued"

    def test_lookup_missing_then_found(self, outbox_dir: Path) -> None:
        transport = OutboxTransport(outbox_dir, clock=fixed_clock)
        msg = message()
        assert transport.lookup(msg.op_key) is None
        transport.send(msg, timeout=1.0)
        found = transport.lookup(msg.op_key)
        assert found is not None
        assert (found.status, found.transport, found.op_key) == ("queued", "outbox", msg.op_key)
        assert found.message_id == f"<{msg.op_key}@example.com>"
        assert found.at == NOW

    def test_lookup_falls_back_to_mtime_when_date_header_missing(self, outbox_dir: Path) -> None:
        outbox_dir.mkdir(parents=True)
        key = "em_" + "b" * 32
        (outbox_dir / f"{key}.eml").write_bytes(b"Message-ID: <x@y>\r\n\r\nbody\r\n")
        found = OutboxTransport(outbox_dir).lookup(key)
        assert found is not None and found.message_id == "<x@y>"
        assert found.at.tzinfo is not None

    @pytest.mark.parametrize("bad_key", ["../escape", "a/b", "", "x" * 200, "em_key.eml", "..", "em key"])
    def test_rejects_unsafe_op_keys_without_writing(self, tmp_path: Path, bad_key: str) -> None:
        outbox = tmp_path / "outbox"
        transport = OutboxTransport(outbox)
        with pytest.raises(EmailTransportError) as info:
            transport.send(message(op_key=bad_key), timeout=1.0)
        assert info.value.kind == "rejected" and info.value.definitely_not_sent
        assert transport.lookup(bad_key) is None
        assert not any(p.is_file() for p in tmp_path.rglob("*"))

    @pytest.mark.parametrize(
        "overrides",
        [
            {"subject": "Hi\r\nBcc: attacker@evil.test"},
            {"subject": "Hi\nX-Injected: 1"},
            {"subject": "   "},
            {"subject": "s" * 201},
            {"to_address": "margaret@email.com, attacker@evil.test"},
            {"to_address": "margaret@email.com\r\nBcc: attacker@evil.test"},
            {"to_address": "Margaret <margaret@email.com>"},
            {"to_address": "not-an-address"},
            {"from_address": "bad sender"},
            {"body_text": "x" * 100_001},
            {"body_text": "nul\x00byte"},
        ],
    )
    def test_rejects_header_injection_and_invalid_fields(self, outbox_dir: Path, overrides: dict[str, str]) -> None:
        transport = OutboxTransport(outbox_dir)
        with pytest.raises(EmailTransportError) as info:
            transport.send(message(**overrides), timeout=1.0)
        assert info.value.kind == "rejected" and info.value.definitely_not_sent
        assert eml_files(outbox_dir) == []
        for marker in PII_MARKERS:
            assert marker not in str(info.value)

    def test_leaves_no_temp_files(self, outbox_dir: Path) -> None:
        transport = OutboxTransport(outbox_dir)
        transport.send(message(), timeout=1.0)
        transport.send(message(), timeout=1.0)
        assert [p.name for p in outbox_dir.iterdir()] == [f"{message().op_key}.eml"]

    def test_write_failure_is_definitely_not_sent(self, outbox_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        def boom(*args: Any, **kwargs: Any) -> Any:
            raise OSError(28, "No space left on device")

        monkeypatch.setattr(emailer.tempfile, "mkstemp", boom)
        with pytest.raises(EmailTransportError) as info:
            OutboxTransport(outbox_dir).send(message(), timeout=1.0)
        assert info.value.kind == "unavailable" and info.value.definitely_not_sent
        assert info.value.__cause__ is None and info.value.__suppress_context__
        assert eml_files(outbox_dir) == []

    def test_falls_back_to_replace_when_hard_links_unsupported(self, outbox_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        def no_links(*args: Any, **kwargs: Any) -> None:
            raise PermissionError(1, "Operation not permitted")

        monkeypatch.setattr(emailer.os, "link", no_links)
        receipt = OutboxTransport(outbox_dir, clock=fixed_clock).send(message(), timeout=1.0)
        files = eml_files(outbox_dir)
        assert len(files) == 1 and receipt.status == "queued"
        assert stat.S_IMODE(files[0].stat().st_mode) == 0o600
        assert [p.name for p in outbox_dir.iterdir()] == [files[0].name]

    def test_race_loser_returns_winners_file(self, outbox_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Another worker publishes between our lookup and our link: theirs wins, ours is discarded."""
        winner = OutboxTransport(outbox_dir, clock=fixed_clock)
        real_publish = OutboxTransport._publish

        def publish_after_rival(self: OutboxTransport, tmp_name: str, path: Path) -> bool:
            if not path.exists():
                rival = OutboxTransport(outbox_dir, clock=fixed_clock)
                monkeypatch.setattr(OutboxTransport, "_publish", real_publish)
                rival.send(message(subject="Rival subject"), timeout=1.0)
            return real_publish(self, tmp_name, path)

        monkeypatch.setattr(OutboxTransport, "_publish", publish_after_rival)
        receipt = winner.send(message(), timeout=1.0)
        files = eml_files(outbox_dir)
        assert len(files) == 1 and receipt.status == "queued"
        parsed = email.message_from_bytes(files[0].read_bytes(), policy=email.policy.default)
        assert parsed["Subject"] == "Rival subject"
        assert [p.name for p in outbox_dir.iterdir()] == [files[0].name]

    def test_message_id_falls_back_to_localhost_for_odd_sender_domain(self, outbox_dir: Path) -> None:
        receipt = OutboxTransport(outbox_dir).send(message(from_address="a@-bad-.x_y"), timeout=1.0)
        assert receipt.message_id == f"<{message().op_key}@localhost>"


# ---------------------------------------------------------------------------
# Dispatcher with the outbox
# ---------------------------------------------------------------------------


class TestDispatchOutbox:
    def test_dispatch_queues_once_and_records_receipt(self, ledger: MemoryLedger, outbox_dir: Path) -> None:
        key = seed_op(ledger)
        transport = CountingOutbox(outbox_dir)
        outcome = run(make_dispatcher(transport, ledger), key)

        assert outcome.status == "queued"
        assert outcome.status != "sent"
        assert outcome.resent is False
        assert outcome.receipt_id == f"<{key}@example.com>"
        assert ledger.status(key) == "queued"
        assert ledger.rows[key].receipt_id == outcome.receipt_id
        assert [t[1:] for t in ledger.transitions] == [("pending", "dispatching"), ("dispatching", "queued")]
        assert len(eml_files(outbox_dir)) == 1

    def test_same_op_key_again_does_not_write_or_send_again(self, ledger: MemoryLedger, outbox_dir: Path) -> None:
        key = seed_op(ledger)
        transport = CountingOutbox(outbox_dir)
        dispatcher = make_dispatcher(transport, ledger)
        first = run(dispatcher, key)
        second = run(dispatcher, key)
        third = run(make_dispatcher(CountingOutbox(outbox_dir), ledger), key)

        assert transport.send_count == 1
        assert (second.status, second.receipt_id, second.resent) == ("queued", first.receipt_id, False)
        assert third.status == "queued"
        assert len(eml_files(outbox_dir)) == 1

    def test_distinct_consents_produce_distinct_files(self, ledger: MemoryLedger, outbox_dir: Path) -> None:
        dispatcher = make_dispatcher(OutboxTransport(outbox_dir, clock=fixed_clock), ledger)
        k1, k2 = seed_op(ledger, turn=7), seed_op(ledger, turn=12)
        assert run(dispatcher, k1).status == run(dispatcher, k2).status == "queued"
        assert len(eml_files(outbox_dir)) == 2

    def test_crash_after_outbox_write_reconciles_to_queued(self, ledger: MemoryLedger, outbox_dir: Path) -> None:
        key = seed_op(ledger)
        OutboxTransport(outbox_dir, clock=fixed_clock).send(message(op_key=key), timeout=1.0)
        ledger.rows[key].status = "dispatching"  # the worker died before recording

        transport = CountingOutbox(outbox_dir)
        outcome = run(make_dispatcher(transport, ledger), key)
        assert outcome.status == "queued" and outcome.detail == "reconciled"
        assert transport.send_count == 0
        assert len(eml_files(outbox_dir)) == 1


# ---------------------------------------------------------------------------
# Dispatcher protocol with ScriptedTransport
# ---------------------------------------------------------------------------


class TestDispatchProtocol:
    def test_ok_sends_once_and_claims_before_sending(self, ledger: MemoryLedger) -> None:
        key = seed_op(ledger)
        transport = LedgerCheckingTransport(ledger)
        outcome = run(make_dispatcher(transport, ledger), key)

        assert transport.status_at_send == ["dispatching"]
        assert (outcome.status, outcome.resent, outcome.detail) == ("sent", False, "accepted")
        assert outcome.receipt_id == f"<{key}@scripted.invalid>"
        assert len(transport.delivered) == 1
        delivered = transport.delivered[0]
        assert (delivered.to_address, delivered.from_address, delivered.subject, delivered.body_text) == (
            TO,
            FROM,
            SUBJECT,
            BODY,
        )
        assert ledger.status(key) == "sent"

    def test_timeout_after_send_with_reconciliation_records_sent_once(self, ledger: MemoryLedger) -> None:
        key = seed_op(ledger)
        transport = ScriptedTransport(["timeout_after_send"], supports_reconciliation=True)
        outcome = run(make_dispatcher(transport, ledger), key)

        assert outcome.status == "sent"
        assert outcome.detail == "reconciled_after_timeout"
        assert outcome.resent is False
        assert len(transport.delivered) == 1
        assert transport.send_calls == [key]
        assert transport.lookup_calls == [key]
        assert ledger.status(key) == "sent"

    def test_timeout_after_send_without_reconciliation_is_unknown_and_never_retried(self, ledger: MemoryLedger) -> None:
        key = seed_op(ledger)
        transport = ScriptedTransport(["timeout_after_send"], supports_reconciliation=False)
        dispatcher = make_dispatcher(transport, ledger)
        outcome = run(dispatcher, key)

        assert (outcome.status, outcome.resent) == ("delivery_unknown", False)
        assert outcome.detail == "timeout_unreconcilable"
        assert transport.send_calls == [key]
        assert transport.lookup_calls == []
        assert ledger.status(key) == "delivery_unknown"

        again = run(dispatcher, key)
        assert again.status == "delivery_unknown"
        assert transport.send_calls == [key]
        assert len(transport.delivered) == 1

    def test_timeout_before_send_without_reconciliation_cannot_prove_failure(self, ledger: MemoryLedger) -> None:
        key = seed_op(ledger)
        transport = ScriptedTransport(["timeout_before_send"], supports_reconciliation=False)
        outcome = run(make_dispatcher(transport, ledger), key)
        assert outcome.status == "delivery_unknown"
        assert outcome.status != "failed"
        assert transport.delivered == [] and len(transport.send_calls) == 1
        assert ledger.status(key) == "delivery_unknown"

    def test_timeout_before_send_with_reconciliation_retries_once(self, ledger: MemoryLedger) -> None:
        key = seed_op(ledger)
        transport = ScriptedTransport(["timeout_before_send", "ok"])
        outcome = run(make_dispatcher(transport, ledger), key)
        assert (outcome.status, outcome.resent, outcome.detail) == ("sent", True, "sent_on_retry")
        assert len(transport.delivered) == 1
        assert transport.send_calls == [key, key]
        assert transport.lookup_calls == [key]

    def test_retry_is_bounded_to_one(self, ledger: MemoryLedger) -> None:
        key = seed_op(ledger)
        transport = ScriptedTransport(["timeout_before_send", "timeout_before_send", "ok"])
        outcome = run(make_dispatcher(transport, ledger), key)
        assert (outcome.status, outcome.resent) == ("delivery_unknown", True)
        assert transport.send_calls == [key, key]
        assert transport.lookup_calls == [key, key]
        assert transport.delivered == []
        assert ledger.status(key) == "delivery_unknown"

    def test_retry_that_times_out_after_delivery_is_reconciled(self, ledger: MemoryLedger) -> None:
        key = seed_op(ledger)
        transport = ScriptedTransport(["timeout_before_send", "timeout_after_send"])
        outcome = run(make_dispatcher(transport, ledger), key)
        assert (outcome.status, outcome.resent) == ("sent", True)
        assert outcome.detail == "reconciled_after_retry_timeout"
        assert len(transport.delivered) == 1

    def test_retry_rejected_is_failed(self, ledger: MemoryLedger) -> None:
        key = seed_op(ledger)
        transport = ScriptedTransport(["timeout_before_send", "reject"])
        outcome = run(make_dispatcher(transport, ledger), key)
        assert (outcome.status, outcome.resent, outcome.detail) == ("failed", True, "retry_rejected")
        assert transport.delivered == []

    @pytest.mark.parametrize(("behavior", "detail"), [("reject", "rejected"), ("unavailable", "unavailable")])
    def test_definite_failures_are_failed_without_lookup_or_retry(self, ledger: MemoryLedger, behavior: str, detail: str) -> None:
        key = seed_op(ledger)
        transport = ScriptedTransport([behavior, "ok"])
        outcome = run(make_dispatcher(transport, ledger), key)
        assert (outcome.status, outcome.detail, outcome.resent) == ("failed", detail, False)
        assert transport.delivered == [] and transport.send_calls == [key] and transport.lookup_calls == []
        assert ledger.status(key) == "failed"

    @pytest.mark.parametrize("status", ["failed", "delivery_unknown", "sent", "queued"])
    def test_final_rows_are_returned_and_never_resent(self, ledger: MemoryLedger, status: str) -> None:
        key = seed_op(ledger, status=status, receipt_id="<r@x>", detail="earlier")
        transport = ScriptedTransport()
        outcome = run(make_dispatcher(transport, ledger), key)
        assert (outcome.status, outcome.receipt_id, outcome.detail, outcome.resent) == (
            status,
            "<r@x>",
            "earlier",
            False,
        )
        assert transport.send_calls == [] and transport.lookup_calls == []
        assert ledger.transitions == []

    def test_failed_row_from_a_real_failure_is_not_resent(self, ledger: MemoryLedger) -> None:
        key = seed_op(ledger)
        transport = ScriptedTransport(["reject", "ok", "ok"])
        dispatcher = make_dispatcher(transport, ledger)
        assert run(dispatcher, key).status == "failed"
        assert run(dispatcher, key).status == "failed"
        assert transport.send_calls == [key]
        assert transport.delivered == []

    def test_missing_consent_row_never_sends(self, ledger: MemoryLedger) -> None:
        transport = ScriptedTransport()
        outcome = run(make_dispatcher(transport, ledger), make_op_key(SESSION, summary_hash(BODY), 1))
        assert (outcome.status, outcome.detail) == ("failed", "no_consent_record")
        assert transport.send_calls == []

    def test_body_must_match_consented_draft(self, ledger: MemoryLedger) -> None:
        key = seed_op(ledger)
        transport = ScriptedTransport()
        dispatcher = make_dispatcher(transport, ledger)
        outcome = run(dispatcher, key, body_text=BODY + "\nP.S. extra unconsented text")
        assert (outcome.status, outcome.detail) == ("failed", "draft_mismatch")
        assert transport.send_calls == []
        assert ledger.status(key) == "failed"
        assert run(dispatcher, key).status == "failed"
        assert transport.send_calls == []

    @pytest.mark.parametrize(
        ("overrides", "reason"),
        [
            ({"to_address": "margaret@email.com\r\nBcc: attacker@evil.test"}, "invalid_recipient"),
            ({"to_address": "margaret@email.com, attacker@evil.test"}, "invalid_recipient"),
            ({"subject": "Hi\r\nBcc: attacker@evil.test"}, "invalid_subject"),
        ],
    )
    def test_invalid_message_fails_before_claim_or_send(self, ledger: MemoryLedger, overrides: dict[str, str], reason: str) -> None:
        key = seed_op(ledger)
        transport = ScriptedTransport()
        outcome = run(make_dispatcher(transport, ledger), key, **overrides)
        assert (outcome.status, outcome.detail) == ("failed", reason)
        assert transport.send_calls == []
        assert [t[1:] for t in ledger.transitions] == [("pending", "failed")]

    def test_unexpected_transport_exception_is_not_retried(self, ledger: MemoryLedger) -> None:
        key = seed_op(ledger)

        class Buggy(ScriptedTransport):
            def send(self, message: EmailMessage, *, timeout: float) -> DeliveryReceipt:
                self.send_calls.append(message.op_key)
                raise RuntimeError("bug in transport")

        transport = Buggy()
        outcome = run(make_dispatcher(transport, ledger), key)
        assert (outcome.status, outcome.detail) == ("delivery_unknown", "transport_error_not_found")
        assert transport.send_calls == [key]
        assert ledger.status(key) == "delivery_unknown"

    def test_lookup_failure_after_timeout_is_unknown_without_retry(self, ledger: MemoryLedger) -> None:
        key = seed_op(ledger)
        transport = ScriptedTransport(["timeout_before_send", "ok"], lookup_fails=True)
        outcome = run(make_dispatcher(transport, ledger), key)
        assert (outcome.status, outcome.detail) == ("delivery_unknown", "timeout_lookup_failed")
        assert transport.send_calls == [key]

    def test_invalid_receipt_is_treated_as_unknown(self, ledger: MemoryLedger) -> None:
        key = seed_op(ledger)

        class WrongReceipt(ScriptedTransport):
            def send(self, message: EmailMessage, *, timeout: float) -> DeliveryReceipt:
                super().send(message, timeout=timeout)
                return DeliveryReceipt("em_other", "sent", self.name, None, NOW)

        transport = WrongReceipt()
        outcome = run(make_dispatcher(transport, ledger), key)
        assert (outcome.status, outcome.detail) == ("delivery_unknown", "invalid_receipt")
        assert run(make_dispatcher(transport, ledger), key).status == "delivery_unknown"
        assert len(transport.send_calls) == 1

    def test_queued_receipt_status_is_passed_through_not_upgraded(self, ledger: MemoryLedger) -> None:
        key = seed_op(ledger)
        outcome = run(make_dispatcher(ScriptedTransport(receipt_status="queued"), ledger), key)
        assert (outcome.status, outcome.detail) == ("queued", "queued_not_delivered")

    def test_disabled_transport_fails_honestly(self, ledger: MemoryLedger) -> None:
        key = seed_op(ledger)
        outcome = run(make_dispatcher(DisabledTransport(), ledger), key)
        assert (outcome.status, outcome.detail) == ("failed", "unavailable")
        assert ledger.status(key) == "failed"
        with pytest.raises(EmailTransportError) as info:
            DisabledTransport().send(message(), timeout=1.0)
        assert info.value.definitely_not_sent and info.value.kind == "unavailable"
        assert DisabledTransport().lookup("em_x") is None

    @pytest.mark.parametrize(
        "script",
        [
            ["ok"],
            ["timeout_after_send"],
            ["timeout_before_send"],
            ["timeout_before_send", "timeout_after_send"],
            ["reject"],
            ["unavailable"],
        ],
    )
    @pytest.mark.parametrize("reconcilable", [True, False])
    def test_one_consent_at_most_one_delivery_under_repeated_dispatch(
        self, ledger: MemoryLedger, script: list[str], reconcilable: bool
    ) -> None:
        key = seed_op(ledger)
        transport = ScriptedTransport(script, supports_reconciliation=reconcilable)
        dispatcher = make_dispatcher(transport, ledger)
        outcomes = [run(dispatcher, key) for _ in range(4)]
        calls_after_first = len(transport.send_calls)
        assert len(transport.delivered) <= 1
        assert calls_after_first <= 2
        assert len({o.status for o in outcomes}) == 1
        assert ledger.status(key) == outcomes[0].status


# ---------------------------------------------------------------------------
# Crash recovery, contention, ledger faults
# ---------------------------------------------------------------------------


class TestRecovery:
    def test_crash_after_send_then_reconcile_finds_delivery(self, ledger: MemoryLedger) -> None:
        key = seed_op(ledger)
        transport = ScriptedTransport(["crash_after_send"], supports_reconciliation=True)
        with pytest.raises(SimulatedCrash):
            run(make_dispatcher(transport, ledger), key)
        assert ledger.status(key) == "dispatching"

        restarted = make_dispatcher(transport, ledger)
        outcome = run(restarted, key)
        assert (outcome.status, outcome.detail, outcome.resent) == ("sent", "reconciled", False)
        assert len(transport.delivered) == 1 and transport.send_calls == [key]
        assert ledger.status(key) == "sent"

    def test_explicit_reconcile_of_dispatching_row(self, ledger: MemoryLedger) -> None:
        key = seed_op(ledger)
        transport = ScriptedTransport(["crash_after_send"])
        with pytest.raises(SimulatedCrash):
            run(make_dispatcher(transport, ledger), key)
        outcome = make_dispatcher(transport, ledger).reconcile(key)
        assert outcome is not None and outcome.status == "sent"
        assert transport.send_calls == [key]

    def test_crash_after_send_non_reconcilable_is_unknown_and_not_resent(self, ledger: MemoryLedger) -> None:
        key = seed_op(ledger)
        transport = ScriptedTransport(["crash_after_send"], supports_reconciliation=False)
        with pytest.raises(SimulatedCrash):
            run(make_dispatcher(transport, ledger), key)

        fresh = run(make_dispatcher(transport, ledger), key)  # the claim may still have a live owner
        assert (fresh.status, fresh.detail, fresh.final) == ("delivery_unknown", "dispatch_in_flight", False)
        assert ledger.status(key) == "dispatching"

        restarted = stale_dispatcher(transport, ledger)
        outcome = run(restarted, key)
        assert (outcome.status, outcome.detail, outcome.final) == ("delivery_unknown", "dispatch_interrupted", True)
        assert run(restarted, key).status == "delivery_unknown"
        assert restarted.reconcile(key).status == "delivery_unknown"  # type: ignore[union-attr]
        assert len(transport.delivered) == 1 and transport.send_calls == [key]
        assert transport.lookup_calls == []

    def test_crash_before_send_fresh_claim_is_in_flight_and_never_resent(self, ledger: MemoryLedger) -> None:
        key = seed_op(ledger)
        transport = ScriptedTransport(["crash_before_send"], supports_reconciliation=True)
        with pytest.raises(SimulatedCrash):
            run(make_dispatcher(transport, ledger), key)

        dispatcher = make_dispatcher(transport, ledger)
        for outcome in (run(dispatcher, key), dispatcher.reconcile(key)):
            assert outcome is not None
            assert (outcome.status, outcome.detail, outcome.final) == ("delivery_unknown", "dispatch_in_flight", False)
        assert transport.send_calls == [key] and transport.delivered == []
        assert ledger.status(key) == "dispatching"
        assert [t[1:] for t in ledger.transitions] == [("pending", "dispatching")]

    def test_reconcile_upgrades_unknown_when_proof_appears(self, ledger: MemoryLedger) -> None:
        key = seed_op(ledger)
        transport = ScriptedTransport(["timeout_after_send"], lookup_fails=True)
        dispatcher = make_dispatcher(transport, ledger)
        assert run(dispatcher, key).status == "delivery_unknown"

        transport.lookup_fails = False
        outcome = dispatcher.reconcile(key)
        assert outcome is not None and (outcome.status, outcome.detail) == ("sent", "reconciled")
        assert ledger.status(key) == "sent"
        assert transport.send_calls == [key]

    def test_reconcile_keeps_unknown_when_lookup_still_fails(self, ledger: MemoryLedger) -> None:
        key = seed_op(ledger, status="delivery_unknown", detail="timeout_lookup_failed")
        transport = ScriptedTransport(lookup_fails=True)
        outcome = make_dispatcher(transport, ledger).reconcile(key)
        assert outcome is not None and outcome.status == "delivery_unknown"
        assert transport.send_calls == []

    def test_reconcile_without_dispatch_returns_none(self, ledger: MemoryLedger) -> None:
        transport = ScriptedTransport()
        dispatcher = make_dispatcher(transport, ledger)
        assert dispatcher.reconcile("em_missing") is None
        assert dispatcher.reconcile(seed_op(ledger)) is None
        assert transport.send_calls == [] and transport.lookup_calls == []

    def test_reconcile_of_final_row_does_not_touch_transport(self, ledger: MemoryLedger) -> None:
        key = seed_op(ledger, status="failed", detail="rejected")
        transport = ScriptedTransport()
        outcome = make_dispatcher(transport, ledger).reconcile(key)
        assert outcome is not None and (outcome.status, outcome.detail) == ("failed", "rejected")
        assert transport.lookup_calls == []

    def test_cas_refused_means_no_send(self, ledger: MemoryLedger) -> None:
        key = seed_op(ledger)
        ledger.refuse_claim = True
        transport = ScriptedTransport()
        outcome = run(make_dispatcher(transport, ledger), key)
        # Still pending, so nobody claimed it and nothing was sent: failed, not unknown, and retryable.
        assert (outcome.status, outcome.detail, outcome.final) == ("failed", "ledger_conflict", False)
        assert transport.send_calls == []
        assert ledger.status(key) == "pending"

    def test_cas_lost_to_live_worker_reconciles_without_sending(self, ledger: MemoryLedger) -> None:
        key = seed_op(ledger)
        rival = ScriptedTransport()

        def rival_claims_and_sends(led: MemoryLedger, op_key: str) -> None:
            led.rows[op_key].status = "dispatching"
            rival.send(message(op_key=op_key), timeout=1.0)

        ledger.on_claim = rival_claims_and_sends
        transport = ScriptedTransport()
        transport.delivered = rival.delivered  # same provider, shared lookup view
        outcome = run(make_dispatcher(transport, ledger), key)
        assert (outcome.status, outcome.detail) == ("sent", "reconciled")
        assert transport.send_calls == []
        assert len(rival.delivered) == 1

    def test_cas_lost_non_reconcilable_reports_unknown(self, ledger: MemoryLedger) -> None:
        key = seed_op(ledger)
        ledger.on_claim = lambda led, op_key: setattr(led.rows[op_key], "status", "dispatching")
        transport = ScriptedTransport(supports_reconciliation=False)
        outcome = run(make_dispatcher(transport, ledger), key)
        assert (outcome.status, outcome.final) == ("delivery_unknown", False)  # the rival may still be sending
        assert transport.send_calls == []
        assert ledger.status(key) == "dispatching"

    def test_cas_lost_to_finished_worker_returns_its_result(self, ledger: MemoryLedger) -> None:
        key = seed_op(ledger)

        def finished(led: MemoryLedger, op_key: str) -> None:
            row = led.rows[op_key]
            row.status, row.receipt_id, row.detail = "sent", "<done@x>", "accepted"

        ledger.on_claim = finished
        transport = ScriptedTransport()
        outcome = run(make_dispatcher(transport, ledger), key)
        assert (outcome.status, outcome.receipt_id) == ("sent", "<done@x>")
        assert transport.send_calls == []

    def test_owner_upgrades_unknown_written_by_concurrent_reconcile(self, ledger: MemoryLedger) -> None:
        """A reconcile that sees the claim as expired marks it unknown; the owner's receipt then wins."""
        key = seed_op(ledger)
        holder: dict[str, EmailDispatcher] = {}

        class SlowTransport(ScriptedTransport):
            def send(self, message: EmailMessage, *, timeout: float) -> DeliveryReceipt:
                interim = holder["other"].reconcile(message.op_key)
                assert interim is not None and (interim.status, interim.detail) == (
                    "delivery_unknown",
                    "dispatch_interrupted",
                )
                return super().send(message, timeout=timeout)

        transport = SlowTransport()
        holder["other"] = make_dispatcher(ScriptedTransport(supports_reconciliation=False), ledger, clock=later_clock)
        outcome = run(make_dispatcher(transport, ledger), key)
        assert (outcome.status, outcome.detail) == ("sent", "accepted")
        assert ledger.status(key) == "sent"
        assert len(transport.delivered) == 1

    def test_ledger_read_failure_never_sends(self, ledger: MemoryLedger) -> None:
        key = seed_op(ledger)
        ledger.fail_get = True
        transport = ScriptedTransport()
        dispatcher = make_dispatcher(transport, ledger)
        outcome = run(dispatcher, key)
        assert (outcome.status, outcome.detail, outcome.final) == ("delivery_unknown", "ledger_unavailable", False)
        again = dispatcher.reconcile(key)
        assert again is not None and (again.detail, again.final) == ("ledger_unavailable", False)
        assert transport.send_calls == []

    def test_ledger_claim_failure_never_sends(self, ledger: MemoryLedger) -> None:
        """L12: a ledger error on the claim CAS sent nothing, so it is never reported as unknown."""
        key = seed_op(ledger)
        ledger.fail_update_to = {"dispatching"}
        transport = ScriptedTransport()
        dispatcher = make_dispatcher(transport, ledger)
        outcome = run(dispatcher, key)
        assert (outcome.status, outcome.detail, outcome.final) == ("failed", "ledger_unavailable", False)
        assert transport.send_calls == []
        assert ledger.status(key) == "pending"  # consistent: unclaimed, nothing sent, nothing settled
        assert ledger.transitions == []

        ledger.fail_update_to = set()  # the blip clears; the consent is honoured exactly once
        retried = run(dispatcher, key)
        assert (retried.status, retried.final) == ("sent", True)
        assert run(dispatcher, key).status == "sent"
        assert len(transport.delivered) == 1 and transport.send_calls == [key]

    def test_claim_that_landed_despite_an_error_is_never_sent_twice(self, ledger: MemoryLedger) -> None:
        """L12: the CAS committed but raised. Nothing was sent; the claim is in flight, then resumed once."""
        key = seed_op(ledger)
        ledger.raise_after_update_to = {"dispatching"}
        transport = ScriptedTransport()
        outcome = run(make_dispatcher(transport, ledger), key)
        assert (outcome.status, outcome.detail, outcome.final) == ("delivery_unknown", "dispatch_in_flight", False)
        assert transport.send_calls == []
        assert ledger.status(key) == "dispatching"

        ledger.raise_after_update_to = set()
        resumed = run(stale_dispatcher(transport, ledger), key)
        assert (resumed.status, resumed.final) == ("sent", True)
        assert len(transport.delivered) == 1 and transport.send_calls == [key]

    def test_claim_error_with_unreadable_ledger_is_unknown_but_not_final(self, ledger: MemoryLedger) -> None:
        key = seed_op(ledger)

        def down(led: MemoryLedger, op_key: str) -> None:
            led.fail_get = True
            raise RuntimeError("database is locked")

        ledger.on_claim = down
        transport = ScriptedTransport()
        outcome = run(make_dispatcher(transport, ledger), key)
        assert (outcome.status, outcome.detail, outcome.final) == ("delivery_unknown", "ledger_unavailable", False)
        assert transport.send_calls == []

    def test_record_failure_after_send_reports_truth_and_reconciles_later(self, ledger: MemoryLedger) -> None:
        key = seed_op(ledger)
        ledger.fail_update_to = {"sent"}
        transport = ScriptedTransport()
        outcome = run(make_dispatcher(transport, ledger), key)
        assert outcome.status == "sent"
        assert outcome.detail == "accepted|unrecorded"
        assert ledger.status(key) == "dispatching"

        ledger.fail_update_to = set()
        later = run(make_dispatcher(transport, ledger), key)
        assert later.status == "sent"
        assert len(transport.delivered) == 1 and transport.send_calls == [key]

    def test_draft_mismatch_with_ledger_down_is_failed_unrecorded(self, ledger: MemoryLedger) -> None:
        key = seed_op(ledger)
        ledger.fail_update_to = {"failed"}
        transport = ScriptedTransport()
        outcome = run(make_dispatcher(transport, ledger), key, body_text="tampered")
        assert (outcome.status, outcome.detail) == ("failed", "draft_mismatch|unrecorded")
        assert transport.send_calls == []

    def test_unknown_ledger_status_never_sends(self, ledger: MemoryLedger) -> None:
        key = seed_op(ledger, status="mystery")
        transport = ScriptedTransport()
        dispatcher = make_dispatcher(transport, ledger)
        assert run(dispatcher, key).detail == "unknown_ledger_status"
        assert dispatcher.reconcile(key).detail == "unknown_ledger_status"  # type: ignore[union-attr]
        assert transport.send_calls == [] and transport.lookup_calls == []


class CrashOnceOutbox(CountingOutbox):
    """Dies (like a process kill) on its first send, after the claim and before any file is written."""

    def __init__(self, outbox_dir: Path) -> None:
        super().__init__(outbox_dir)
        self.crashed = False

    def send(self, message: EmailMessage, *, timeout: float) -> DeliveryReceipt:
        if not self.crashed:
            self.crashed = True
            raise SimulatedCrash("killed before the outbox write")
        return super().send(message, timeout=timeout)


class TestOrphanResume:
    """C44: an orphaned claim on a reconcilable transport whose lookup proves nothing went out is resent once."""

    def test_orphan_with_nothing_sent_is_resumed_once(self, ledger: MemoryLedger) -> None:
        key = seed_op(ledger)
        transport = ScriptedTransport(["crash_before_send"], supports_reconciliation=True)
        with pytest.raises(SimulatedCrash):
            run(make_dispatcher(transport, ledger), key)
        assert ledger.status(key) == "dispatching" and transport.delivered == []

        restarted = stale_dispatcher(transport, ledger)
        outcome = run(restarted, key)
        assert (outcome.status, outcome.detail, outcome.resent, outcome.final) == ("sent", "sent_on_retry", True, True)
        assert len(transport.delivered) == 1 and transport.send_calls == [key, key]
        assert ledger.status(key) == "sent"
        assert [t[1:] for t in ledger.transitions] == [
            ("pending", "dispatching"),
            ("dispatching", "delivery_unknown"),  # the exclusive takeover, before the resend
            ("delivery_unknown", "sent"),
        ]

        assert run(restarted, key).status == "sent"
        assert restarted.reconcile(key).status == "sent"  # type: ignore[union-attr]
        assert transport.send_calls == [key, key] and len(transport.delivered) == 1

    def test_outbox_orphan_is_resumed_and_queued_once(self, ledger: MemoryLedger, outbox_dir: Path) -> None:
        key = seed_op(ledger)
        transport = CrashOnceOutbox(outbox_dir)
        with pytest.raises(SimulatedCrash):
            run(make_dispatcher(transport, ledger), key)
        assert ledger.status(key) == "dispatching" and eml_files(outbox_dir) == []

        fresh = run(make_dispatcher(transport, ledger), key)
        assert (fresh.status, fresh.final) == ("delivery_unknown", False)
        assert eml_files(outbox_dir) == []

        restarted = stale_dispatcher(transport, ledger)
        outcome = run(restarted, key)
        assert (outcome.status, outcome.final, outcome.resent) == ("queued", True, True)
        assert outcome.status != "sent"
        assert outcome.receipt_id == f"<{key}@example.com>"
        files = eml_files(outbox_dir)
        assert files == [outbox_dir / f"{key}.eml"]
        parsed = email.message_from_bytes(files[0].read_bytes(), policy=email.policy.default)
        assert parsed["To"] == TO and parsed.get_content().replace("\r\n", "\n") == BODY

        assert run(restarted, key).status == "queued"
        assert transport.send_count == 1 and len(eml_files(outbox_dir)) == 1

    def test_reconcile_never_resumes_an_orphan(self, ledger: MemoryLedger) -> None:
        key = seed_op(ledger)
        transport = ScriptedTransport(["crash_before_send"])
        with pytest.raises(SimulatedCrash):
            run(make_dispatcher(transport, ledger), key)

        restarted = stale_dispatcher(transport, ledger)
        outcome = restarted.reconcile(key)
        assert outcome is not None
        assert (outcome.status, outcome.detail, outcome.final) == ("delivery_unknown", "dispatch_interrupted", True)
        assert run(restarted, key).status == "delivery_unknown"  # settled unknown is never re-sent
        assert transport.send_calls == [key] and transport.delivered == []

    @pytest.mark.parametrize(
        ("transport_kwargs", "overrides"),
        [
            ({"supports_reconciliation": False}, {}),
            ({"lookup_fails": True}, {}),
            ({}, {"body_text": BODY + "\nP.S. unconsented"}),
            ({}, {"to_address": "margaret@email.com\r\nBcc: attacker@evil.test"}),
        ],
        ids=["not_reconcilable", "lookup_failed", "draft_mismatch", "invalid_message"],
    )
    def test_orphan_is_not_resent_without_proof_or_with_a_different_message(
        self, ledger: MemoryLedger, transport_kwargs: dict[str, Any], overrides: dict[str, str]
    ) -> None:
        key = seed_op(ledger)
        transport = ScriptedTransport(["crash_before_send"], **transport_kwargs)
        with pytest.raises(SimulatedCrash):
            run(make_dispatcher(transport, ledger), key)

        outcome = run(stale_dispatcher(transport, ledger), key, **overrides)
        assert (outcome.status, outcome.detail, outcome.final) == ("delivery_unknown", "dispatch_interrupted", True)
        assert transport.send_calls == [key] and transport.delivered == []
        assert ledger.status(key) == "delivery_unknown"

    def test_concurrent_resumers_send_at_most_once(self, ledger: MemoryLedger) -> None:
        key = seed_op(ledger)
        holder: dict[str, Any] = {}

        class RacingTransport(ScriptedTransport):
            def send(self, message: EmailMessage, *, timeout: float) -> DeliveryReceipt:
                if len(self.send_calls) == 1:  # the resend: a second recovery races it
                    holder["rival"] = run(make_dispatcher(holder["rival_transport"], ledger, clock=later_clock), key)
                return super().send(message, timeout=timeout)

        transport = RacingTransport(["crash_before_send"])
        holder["rival_transport"] = ScriptedTransport()
        holder["rival_transport"].delivered = transport.delivered  # same provider, shared lookup view
        with pytest.raises(SimulatedCrash):
            run(make_dispatcher(transport, ledger), key)

        outcome = run(stale_dispatcher(transport, ledger), key)
        rival = holder["rival"]
        assert (rival.status, rival.detail, rival.final) == ("delivery_unknown", "dispatch_in_flight", False)
        assert holder["rival_transport"].send_calls == []
        assert (outcome.status, outcome.final) == ("sent", True)
        assert len(transport.delivered) == 1 and ledger.status(key) == "sent"

    def test_resumer_that_dies_leaves_unknown_and_is_never_resent(self, ledger: MemoryLedger) -> None:
        key = seed_op(ledger)
        transport = ScriptedTransport(["crash_before_send", "crash_before_send"])
        with pytest.raises(SimulatedCrash):
            run(make_dispatcher(transport, ledger), key)
        with pytest.raises(SimulatedCrash):
            run(stale_dispatcher(transport, ledger), key)  # dies during its one resend
        assert ledger.status(key) == "delivery_unknown"

        mid = run(make_dispatcher(transport, ledger, clock=later_clock), key)
        assert (mid.status, mid.detail, mid.final) == ("delivery_unknown", "dispatch_in_flight", False)

        much_later = lambda: LATER + timedelta(minutes=5)  # noqa: E731
        ledger.clock = much_later
        final = run(make_dispatcher(transport, ledger, clock=much_later), key)
        assert (final.status, final.detail, final.final) == ("delivery_unknown", "resume_interrupted", True)
        assert run(make_dispatcher(transport, ledger, clock=much_later), key).status == "delivery_unknown"
        assert transport.send_calls == [key, key] and transport.delivered == []

    def test_takeover_ledger_error_sends_nothing(self, ledger: MemoryLedger) -> None:
        key = seed_op(ledger)
        transport = ScriptedTransport(["crash_before_send"])
        with pytest.raises(SimulatedCrash):
            run(make_dispatcher(transport, ledger), key)
        ledger.fail_update_to = {"delivery_unknown"}
        outcome = run(stale_dispatcher(transport, ledger), key)
        assert (outcome.status, outcome.detail, outcome.final) == ("delivery_unknown", "ledger_unavailable", False)
        assert transport.send_calls == [key]
        assert ledger.status(key) == "dispatching"


class TestInFlightClaims:
    """L14: a reconcile or dispatch from another worker never overwrites a live owner's claim."""

    @pytest.mark.parametrize("reconcilable", [True, False])
    @pytest.mark.parametrize("via", ["reconcile", "dispatch"])
    def test_concurrent_caller_leaves_a_live_send_alone(self, ledger: MemoryLedger, reconcilable: bool, via: str) -> None:
        key = seed_op(ledger)
        holder: dict[str, Any] = {}
        other_transport = ScriptedTransport(supports_reconciliation=reconcilable)
        other = make_dispatcher(other_transport, ledger)

        class SlowTransport(ScriptedTransport):
            def send(self, message: EmailMessage, *, timeout: float) -> DeliveryReceipt:
                holder["interim"] = other.reconcile(key) if via == "reconcile" else run(other, key)
                holder["status_during"] = ledger.status(key)
                return super().send(message, timeout=timeout)

        transport = SlowTransport()
        outcome = run(make_dispatcher(transport, ledger), key)

        interim = holder["interim"]
        assert interim is not None
        assert (interim.status, interim.detail, interim.final) == ("delivery_unknown", "dispatch_in_flight", False)
        assert holder["status_during"] == "dispatching"
        assert (outcome.status, outcome.detail, outcome.final) == ("sent", "accepted", True)
        assert [t[1:] for t in ledger.transitions] == [("pending", "dispatching"), ("dispatching", "sent")]
        assert other_transport.send_calls == [] and len(transport.delivered) == 1

        settled = other.reconcile(key)  # the next recovery sees the owner's definite result
        assert settled is not None and (settled.status, settled.final) == ("sent", True)

    def test_claim_ttl_default_and_override(self, ledger: MemoryLedger) -> None:
        assert make_dispatcher(ScriptedTransport(), ledger).claim_ttl_seconds == 60.0
        slow = EmailDispatcher(ScriptedTransport(), ledger, from_address=FROM, timeout=30, clock=fixed_clock)
        assert slow.claim_ttl_seconds == 120.0
        with pytest.raises(ValueError):
            EmailDispatcher(ScriptedTransport(), ledger, from_address=FROM, timeout=5, claim_ttl=0)

        key = seed_op(ledger, status="dispatching")
        just_after = lambda: NOW + timedelta(seconds=11)  # noqa: E731
        short = EmailDispatcher(
            ScriptedTransport(supports_reconciliation=False),
            ledger,
            from_address=FROM,
            timeout=5,
            clock=just_after,
            claim_ttl=10,
        )
        assert short.reconcile(key).detail == "dispatch_interrupted"  # type: ignore[union-attr]

    def test_unreadable_claim_age_is_treated_as_live(self, ledger: MemoryLedger) -> None:
        key = seed_op(ledger, status="dispatching")
        ledger.rows[key].updated_at = None  # type: ignore[assignment]
        transport = ScriptedTransport()
        outcome = run(stale_dispatcher(transport, ledger), key)
        assert (outcome.detail, outcome.final) == ("dispatch_in_flight", False)
        assert transport.send_calls == []


class TestRealLedger:
    """The same protocol against the real SQLite ``SessionStore`` (its CAS and transition rules)."""

    @pytest.fixture
    def store_and_key(self, tmp_path: Path) -> tuple[Any, list[datetime], str]:
        from cryptography.fernet import Fernet

        from insurance_claims.domain.state import SessionState
        from insurance_claims.persistence.store import SessionStore, StateCipher

        now = [NOW]
        store = SessionStore(tmp_path / "db.sqlite3", StateCipher(Fernet.generate_key()), clock=lambda: now[0])
        store.init_schema()
        state = SessionState(session_id=SESSION, created_at=NOW, last_activity_at=NOW)
        store.create(state, secret_hash="h", expires_at=NOW + timedelta(days=1))
        draft = summary_hash(BODY)
        key = make_op_key(SESSION, draft, 7)
        op = EmailOpRecord(key, SESSION, "P9", draft, "pending", NOW, NOW)
        store.commit(SESSION, expected_version=1, state=state, email_ops_create=[op])
        return store, now, key

    def test_outbox_orphan_resumes_once(self, store_and_key: tuple[Any, list[datetime], str], outbox_dir: Path) -> None:
        store, now, key = store_and_key
        transport = CrashOnceOutbox(outbox_dir)
        with pytest.raises(SimulatedCrash):
            run(make_dispatcher(transport, store, clock=lambda: now[0]), key)
        assert store.email_op_get(key).status == "dispatching"

        assert run(make_dispatcher(transport, store, clock=lambda: now[0]), key).final is False
        now[0] = LATER
        outcome = run(make_dispatcher(transport, store, clock=lambda: now[0]), key)
        assert (outcome.status, outcome.resent, outcome.final) == ("queued", True, True)
        assert store.email_op_get(key).status == "queued"
        assert transport.send_count == 1 and len(eml_files(outbox_dir)) == 1

    def test_reconcile_from_other_worker_does_not_mark_live_send_unknown(self, store_and_key: tuple[Any, list[datetime], str]) -> None:
        store, now, key = store_and_key
        other = make_dispatcher(ScriptedTransport(supports_reconciliation=False), store, clock=lambda: now[0])
        seen: dict[str, Any] = {}

        class SlowTransport(ScriptedTransport):
            def send(self, message: EmailMessage, *, timeout: float) -> DeliveryReceipt:
                now[0] = NOW + timedelta(seconds=1)
                seen["interim"] = other.reconcile(key)
                seen["during"] = store.email_op_get(key).status
                return super().send(message, timeout=timeout)

        transport = SlowTransport()
        outcome = run(make_dispatcher(transport, store, clock=lambda: now[0]), key)
        assert (seen["interim"].final, seen["during"]) == (False, "dispatching")
        assert (outcome.status, outcome.detail) == ("sent", "accepted")
        assert store.email_op_get(key).status == "sent"
        assert other.reconcile(key).status == "sent"
        assert len(transport.delivered) == 1

    def test_claim_error_keeps_row_pending_and_retry_sends_once(
        self, store_and_key: tuple[Any, list[datetime], str], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        store, now, key = store_and_key
        real_update = store.email_op_update
        calls = {"n": 0}

        def flaky(op_key: str, **kwargs: Any) -> bool:
            calls["n"] += 1
            if calls["n"] == 1:
                import sqlite3

                raise sqlite3.OperationalError("database is locked")
            return real_update(op_key, **kwargs)

        monkeypatch.setattr(store, "email_op_update", flaky)
        transport = ScriptedTransport()
        dispatcher = make_dispatcher(transport, store, clock=lambda: now[0])
        first = run(dispatcher, key)
        assert (first.status, first.final) == ("failed", False)
        assert store.email_op_get(key).status == "pending" and transport.send_calls == []
        second = run(dispatcher, key)
        assert (second.status, second.final) == ("sent", True)
        assert len(transport.delivered) == 1


# ---------------------------------------------------------------------------
# SmtpTransport (fake smtplib.SMTP, no network)
# ---------------------------------------------------------------------------


@dataclass
class SmtpScript:
    fail: dict[str, BaseException] = field(default_factory=dict)
    refused: dict[str, tuple[int, bytes]] = field(default_factory=dict)
    instances: list[Any] = field(default_factory=list)

    @property
    def last(self) -> Any:
        return self.instances[-1]


@pytest.fixture
def smtp(monkeypatch: pytest.MonkeyPatch) -> SmtpScript:
    script = SmtpScript()

    class FakeSMTP:
        def __init__(self, host: str, port: int = 0, timeout: float | None = None) -> None:
            if "connect" in script.fail:
                raise script.fail["connect"]
            self.host, self.port, self.timeout = host, port, timeout
            self.calls: list[str] = []
            self.sent: list[tuple[Any, str, list[str]]] = []
            self.tls_context: Any = None
            self.login_args: tuple[str, str] | None = None
            script.instances.append(self)

        def _step(self, name: str) -> None:
            self.calls.append(name)
            if name in script.fail:
                raise script.fail[name]

        def ehlo(self) -> None:
            self._step("ehlo")

        def starttls(self, context: Any = None) -> None:
            self.tls_context = context
            self._step("starttls")

        def login(self, user: str, password: str) -> None:
            self.login_args = (user, password)
            self._step("login")

        def send_message(self, msg: Any, from_addr: str | None = None, to_addrs: Any = None) -> dict:
            self._step("send_message")
            self.sent.append((msg, from_addr, list(to_addrs or [])))
            return dict(script.refused)

        def quit(self) -> None:
            self._step("quit")

        def close(self) -> None:
            self.calls.append("close")

    monkeypatch.setattr(smtplib, "SMTP", FakeSMTP)
    return script


def smtp_transport(**kwargs: Any) -> SmtpTransport:
    params = {"username": "mailer", "password": "smtp-secret-pw", "starttls": True, "clock": fixed_clock, **kwargs}
    return SmtpTransport("smtp.example.test", 2525, **params)


def smtp_error(transport: SmtpTransport, msg: EmailMessage | None = None) -> EmailTransportError:
    with pytest.raises(EmailTransportError) as info:
        transport.send(msg or message(), timeout=3.0)
    exc = info.value
    for marker in PII_MARKERS:
        assert marker not in str(exc) and marker not in repr(exc)
    assert "smtp-secret-pw" not in str(exc)
    # The original smtplib error may carry the address: it must never be chained visibly.
    assert exc.__cause__ is None
    assert exc.__context__ is None or exc.__suppress_context__
    return exc


class TestSmtpTransport:
    def test_success_uses_starttls_then_login_then_send(self, smtp: SmtpScript) -> None:
        msg = message()
        receipt = smtp_transport().send(msg, timeout=3.0)
        client = smtp.last

        assert (client.host, client.port, client.timeout) == ("smtp.example.test", 2525, 3.0)
        assert client.calls == ["ehlo", "starttls", "ehlo", "login", "send_message", "quit"]
        assert client.tls_context is not None
        assert client.login_args == ("mailer", "smtp-secret-pw")
        sent, from_addr, to_addrs = client.sent[0]
        assert (from_addr, to_addrs) == (FROM, [TO])
        assert sent["To"] == TO and sent["From"] == FROM and sent["Subject"] == SUBJECT
        assert sent["Message-ID"] == f"<{msg.op_key}@example.com>"
        assert sent.get_content() == BODY
        assert (receipt.status, receipt.transport, receipt.message_id) == ("sent", "smtp", sent["Message-ID"])
        assert receipt.at == NOW

    def test_plain_relay_without_tls_or_login(self, smtp: SmtpScript) -> None:
        smtp_transport(username=None, password=None, starttls=False).send(message(), timeout=3.0)
        assert smtp.last.calls == ["ehlo", "send_message", "quit"]

    def test_not_reconcilable(self, smtp: SmtpScript) -> None:
        transport = smtp_transport()
        assert transport.supports_reconciliation is False
        assert transport.lookup("em_x") is None

    def test_timeout_during_send_is_ambiguous(self, smtp: SmtpScript) -> None:
        smtp.fail["send_message"] = TimeoutError("timed out")
        exc = smtp_error(smtp_transport())
        assert (exc.kind, exc.definitely_not_sent) == ("timeout", False)

    @pytest.mark.parametrize("step", ["connect", "ehlo", "starttls", "login"])
    def test_timeout_before_send_is_definitely_not_sent(self, smtp: SmtpScript, step: str) -> None:
        smtp.fail[step] = TimeoutError("timed out")
        exc = smtp_error(smtp_transport())
        assert (exc.kind, exc.definitely_not_sent) == ("timeout", True)
        if smtp.instances:
            assert "send_message" not in smtp.last.calls

    @pytest.mark.parametrize(
        "error",
        [
            ConnectionRefusedError(61, "Connection refused"),
            OSError(8, "nodename nor servname provided"),
            smtplib.SMTPConnectError(421, b"service not available"),
        ],
    )
    def test_connection_failures_are_unavailable_and_not_sent(self, smtp: SmtpScript, error: Exception) -> None:
        smtp.fail["connect"] = error
        exc = smtp_error(smtp_transport())
        assert (exc.kind, exc.definitely_not_sent) == ("unavailable", True)

    def test_recipient_refused_is_rejected_and_hides_address(self, smtp: SmtpScript) -> None:
        smtp.fail["send_message"] = smtplib.SMTPRecipientsRefused({TO: (550, b"5.1.1 user unknown " + TO.encode())})
        exc = smtp_error(smtp_transport())
        assert (exc.kind, exc.definitely_not_sent) == ("rejected", True)
        assert smtp.last.calls[-1] == "quit"

    @pytest.mark.parametrize(
        ("error", "kind"),
        [
            (smtplib.SMTPDataError(554, b"5.7.1 message rejected"), "rejected"),
            (smtplib.SMTPSenderRefused(553, b"sender refused", FROM), "rejected"),
            (smtplib.SMTPDataError(451, b"4.3.0 try again later"), "unavailable"),
        ],
    )
    def test_server_replies_during_send_are_definite(self, smtp: SmtpScript, error: Exception, kind: str) -> None:
        smtp.fail["send_message"] = error
        exc = smtp_error(smtp_transport())
        assert (exc.kind, exc.definitely_not_sent) == (kind, True)

    def test_auth_failure_is_unavailable_not_rejected(self, smtp: SmtpScript) -> None:
        smtp.fail["login"] = smtplib.SMTPAuthenticationError(535, b"bad credentials")
        exc = smtp_error(smtp_transport())
        assert (exc.kind, exc.definitely_not_sent) == ("unavailable", True)
        assert "send_message" not in smtp.last.calls

    def test_starttls_unsupported_is_unavailable(self, smtp: SmtpScript) -> None:
        smtp.fail["starttls"] = smtplib.SMTPNotSupportedError("STARTTLS extension not supported")
        exc = smtp_error(smtp_transport())
        assert (exc.kind, exc.definitely_not_sent) == ("unavailable", True)

    @pytest.mark.parametrize("error", [smtplib.SMTPServerDisconnected("closed"), ConnectionResetError(54, "reset by peer")])
    def test_disconnect_during_send_is_ambiguous(self, smtp: SmtpScript, error: Exception) -> None:
        smtp.fail["send_message"] = error
        exc = smtp_error(smtp_transport())
        assert (exc.kind, exc.definitely_not_sent) == ("unavailable", False)

    def test_partial_refusal_is_rejected(self, smtp: SmtpScript) -> None:
        smtp.refused = {TO: (550, b"no")}
        exc = smtp_error(smtp_transport())
        assert (exc.kind, exc.definitely_not_sent) == ("rejected", True)

    def test_quit_failure_after_acceptance_still_sent(self, smtp: SmtpScript) -> None:
        smtp.fail["quit"] = smtplib.SMTPServerDisconnected("gone")
        receipt = smtp_transport().send(message(), timeout=3.0)
        assert receipt.status == "sent"
        assert smtp.last.calls[-2:] == ["quit", "close"]

    def test_header_injection_rejected_before_connecting(self, smtp: SmtpScript) -> None:
        exc = smtp_error(smtp_transport(), message(to_address="a@b.com\r\nBcc: attacker@evil.test"))
        assert (exc.kind, exc.definitely_not_sent) == ("rejected", True)
        assert smtp.instances == []

    def test_repr_and_construction_guard_credentials(self) -> None:
        transport = smtp_transport()
        assert "smtp-secret-pw" not in repr(transport)
        with pytest.raises(ValueError):
            SmtpTransport("  ")
        with pytest.raises(ValueError):
            SmtpTransport("smtp.example.test", 0)

    def test_dispatch_over_smtp_timeout_is_unknown_and_sent_once(self, smtp: SmtpScript, ledger: MemoryLedger) -> None:
        smtp.fail["send_message"] = TimeoutError("timed out")
        key = seed_op(ledger)
        dispatcher = make_dispatcher(smtp_transport(), ledger)
        outcome = run(dispatcher, key)
        assert (outcome.status, outcome.detail) == ("delivery_unknown", "timeout_unreconcilable")
        assert run(dispatcher, key).status == "delivery_unknown"
        assert sum(c.calls.count("send_message") for c in smtp.instances) == 1

    def test_dispatch_over_smtp_rejection_is_failed(self, smtp: SmtpScript, ledger: MemoryLedger) -> None:
        smtp.fail["send_message"] = smtplib.SMTPRecipientsRefused({TO: (550, b"unknown")})
        key = seed_op(ledger)
        outcome = run(make_dispatcher(smtp_transport(), ledger), key)
        assert (outcome.status, outcome.detail) == ("failed", "rejected")

    def test_dispatch_over_smtp_success_is_sent(self, smtp: SmtpScript, ledger: MemoryLedger) -> None:
        key = seed_op(ledger)
        outcome = run(make_dispatcher(smtp_transport(), ledger), key)
        assert (outcome.status, outcome.receipt_id) == ("sent", f"<{key}@example.com>")
        assert ledger.status(key) == "sent"


# ---------------------------------------------------------------------------
# Construction, settings, ScriptedTransport itself
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_dispatcher_validates_config(self, ledger: MemoryLedger) -> None:
        with pytest.raises(ValueError):
            EmailDispatcher(ScriptedTransport(), ledger, from_address=FROM, timeout=0, clock=fixed_clock)
        with pytest.raises(ValueError):
            EmailDispatcher(ScriptedTransport(), ledger, from_address="nope", timeout=5, clock=fixed_clock)

    def test_scripted_transport_validates_and_defaults_to_ok(self) -> None:
        with pytest.raises(ValueError):
            ScriptedTransport(["explode"])
        with pytest.raises(ValueError):
            ScriptedTransport(receipt_status="delivered")  # type: ignore[arg-type]
        transport = ScriptedTransport(["reject"])
        with pytest.raises(EmailTransportError):
            transport.send(message(), timeout=1.0)
        assert transport.send(message(), timeout=1.0).status == "sent"

    def test_transport_from_settings(self, tmp_path: Path) -> None:
        base = Settings(data_dir=tmp_path)
        outbox = transport_from_settings(base.with_overrides(email_transport="outbox"))
        assert isinstance(outbox, OutboxTransport) and outbox.outbox_dir == tmp_path / "outbox"
        smtp_settings = base.with_overrides(email_transport="smtp", smtp_host="mail.test", smtp_password="pw-xyz")
        smtp_t = transport_from_settings(smtp_settings)
        assert isinstance(smtp_t, SmtpTransport) and "pw-xyz" not in repr(smtp_t)
        assert isinstance(transport_from_settings(base.with_overrides(email_transport="disabled")), DisabledTransport)
        with pytest.raises(ValueError):
            transport_from_settings(base.with_overrides(email_transport="smtp", smtp_host=None))


# ---------------------------------------------------------------------------
# PII never reaches logs or traces
# ---------------------------------------------------------------------------


class TestNoPiiLeaks:
    def test_logs_and_trace_events_never_contain_address_subject_or_body(
        self,
        caplog: pytest.LogCaptureFixture,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        smtp: SmtpScript,
    ) -> None:
        events: list[tuple[str, dict[str, Any]]] = []
        monkeypatch.setattr(tracing_module, "event", lambda name, type="guard", **data: events.append((name, data)))
        caplog.set_level(logging.DEBUG)

        scripts: list[tuple[Any, list[str]]] = [
            (ScriptedTransport(["ok"]), []),
            (ScriptedTransport(["timeout_after_send"]), []),
            (ScriptedTransport(["timeout_before_send", "timeout_before_send"]), []),
            (ScriptedTransport(["timeout_after_send"], supports_reconciliation=False), []),
            (ScriptedTransport(["reject"]), []),
            (ScriptedTransport(["timeout_before_send"], lookup_fails=True), []),
            (OutboxTransport(tmp_path / "outbox", clock=fixed_clock), []),
            (DisabledTransport(), []),
        ]
        for turn, (transport, _) in enumerate(scripts):
            led = MemoryLedger()
            key = seed_op(led, turn=turn)
            run(make_dispatcher(transport, led), key)
            run(make_dispatcher(transport, led), key)

        for failure in (TimeoutError("t"), smtplib.SMTPRecipientsRefused({TO: (550, TO.encode())})):
            smtp.fail["send_message"] = failure
            led = MemoryLedger()
            run(make_dispatcher(smtp_transport(), led), seed_op(led))

        led = MemoryLedger()
        key = seed_op(led)
        run(make_dispatcher(ScriptedTransport(), led), key, to_address=TO + "\r\nBcc: x@evil.test")
        led = MemoryLedger()
        run(make_dispatcher(ScriptedTransport(), led), seed_op(led), body_text=BODY + "tampered")

        messages = [r.getMessage() for r in caplog.records]
        assert any("op_key=" in m for m in messages), "expected structured dispatch logs"
        blob = "\n".join(messages + [caplog.text, repr(events)])
        for marker in PII_MARKERS:
            assert marker not in blob
        assert "smtp-secret-pw" not in blob
        assert all(r.exc_info is None for r in caplog.records)
        assert events and all(name in {"email_dispatch", "email_reconcile"} for name, _ in events)

    def test_tracing_failure_does_not_break_dispatch(self, ledger: MemoryLedger, monkeypatch: pytest.MonkeyPatch) -> None:
        def broken(*args: Any, **kwargs: Any) -> None:
            raise RuntimeError("trace sink down")

        monkeypatch.setattr(tracing_module, "event", broken)
        key = seed_op(ledger)
        transport = ScriptedTransport()
        assert run(make_dispatcher(transport, ledger), key).status == "sent"
        assert len(transport.delivered) == 1
