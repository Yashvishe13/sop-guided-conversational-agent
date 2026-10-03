"""Consent-gated, idempotent, reconcilable email dispatch.

The agent (with its code guardrails) decides consent and the recipient (the verified address on
record) and creates a ``pending`` ledger row in the consent commit. This module
only executes that one logical write, at most once per consent:

* the row moves ``pending -> dispatching`` by compare-and-set *before* the
  transport is called, so a crash always leaves evidence of a possible send;
* a timeout is never treated as a failure: it is reconciled with ``lookup``
  (when the transport supports it) before at most one retry with the same key;
* a ``dispatching`` claim younger than ``claim_ttl`` may belong to a live
  owner: it is reported as in flight (``final=False``) and never overwritten.
  An older claim is an orphan. ``dispatch`` resumes it with at most one more
  send, but only on a reconcilable transport and only after a lookup that
  succeeded and found nothing. Otherwise it becomes ``delivery_unknown``;
* ``failed`` and ``delivery_unknown`` rows are never re-sent automatically;
* an op that was never claimed (still ``pending``) was never handed to the
  transport, so it is never reported as ``delivery_unknown``;
* the local outbox only *queues* a file and is always reported as ``queued``.

Nothing here logs, traces, or raises with an address, subject, or body. Only
the op key (a hash), statuses, and machine codes leave this module.
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
import smtplib
import ssl
import tempfile
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from email import policy as email_policy
from email.message import EmailMessage as MimeMessage
from email.message import Message
from email.parser import BytesHeaderParser
from email.utils import format_datetime, parsedate_to_datetime
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Final, Literal, Protocol

from insurance_claims.domain.records import EmailOpRecord
from insurance_claims.observability import tracing

if TYPE_CHECKING:
    from insurance_claims.config import Settings

__all__ = [
    "SCRIPT_BEHAVIORS",
    "DeliveryReceipt",
    "DisabledTransport",
    "DispatchOutcome",
    "EmailDispatcher",
    "EmailLedger",
    "EmailMessage",
    "EmailTransport",
    "EmailTransportError",
    "OutboxTransport",
    "ScriptedTransport",
    "SimulatedCrash",
    "SmtpTransport",
    "make_op_key",
    "summary_hash",
    "transport_from_settings",
]

logger = logging.getLogger(__name__)

Clock = Callable[[], datetime]
ReceiptStatus = Literal["sent", "queued"]
DispatchStatus = Literal["sent", "queued", "failed", "delivery_unknown"]
TransportErrorKind = Literal["timeout", "rejected", "unavailable"]

MAX_SUBJECT_CHARS: Final = 200
MAX_BODY_CHARS: Final = 100_000

_KINDS: Final = frozenset({"timeout", "rejected", "unavailable"})
_FINAL: Final = frozenset({"sent", "queued", "failed", "delivery_unknown"})
_OP_KEY_RE: Final = re.compile(r"^[A-Za-z0-9_-]{1,128}$")
_ADDR_CHARS: Final = r"[^@\s<>,;:\"()\[\]\\]+"
_ADDRESS_RE: Final = re.compile(rf"^{_ADDR_CHARS}@{_ADDR_CHARS}\.{_ADDR_CHARS}$")
_DOMAIN_RE: Final = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9.-]{0,251}[A-Za-z0-9])?$")


def _utcnow() -> datetime:
    return datetime.now(UTC)


def _aware(value: datetime) -> datetime:
    """Treat naive datetimes as UTC so headers and receipts are unambiguous."""
    return value if value.tzinfo is not None else value.replace(tzinfo=UTC)


# ---------------------------------------------------------------------------
# Value types
# ---------------------------------------------------------------------------


@dataclass
class EmailMessage:
    """One outgoing message. Address, subject, and body are kept out of ``repr`` (PII)."""

    op_key: str
    to_address: str = field(repr=False)
    from_address: str
    subject: str = field(repr=False)
    body_text: str = field(repr=False)


@dataclass
class DeliveryReceipt:
    """Proof that a transport accepted (``sent``) or locally queued (``queued``) a message."""

    op_key: str
    status: ReceiptStatus
    transport: str
    message_id: str | None
    at: datetime


class EmailTransportError(Exception):
    """A transport failure. ``message`` is a PII-free machine code.

    ``definitely_not_sent`` is True only when the transport can prove nothing
    left the system; otherwise the dispatcher must reconcile, never assume.
    """

    def __init__(self, kind: TransportErrorKind, *, definitely_not_sent: bool, message: str = "") -> None:
        if kind not in _KINDS:
            raise ValueError(f"unknown email transport error kind {kind!r}")
        self.kind: TransportErrorKind = kind
        self.definitely_not_sent = bool(definitely_not_sent)
        self.message = message or kind
        super().__init__(f"email transport {kind} ({self.message})")


class EmailTransport(Protocol):
    """``supports_reconciliation=True`` makes two promises. ``lookup(op_key)`` is
    authoritative, and ``send`` is idempotent by op key: a second send with the
    same key never makes a second delivery. The dispatcher's one retry, and its
    resume of an orphaned claim, depend on both. A transport that cannot keep
    both promises must set it to False, and its ambiguous outcomes then end as
    ``delivery_unknown``.
    """

    name: str
    supports_reconciliation: bool

    def send(self, message: EmailMessage, *, timeout: float) -> DeliveryReceipt: ...

    def lookup(self, op_key: str) -> DeliveryReceipt | None: ...


class EmailLedger(Protocol):
    """Durable op ledger (implemented by ``store.SessionStore``)."""

    def email_op_get(self, op_key: str) -> EmailOpRecord | None: ...

    def email_op_update(
        self,
        op_key: str,
        *,
        status: str,
        receipt_id: str | None = None,
        detail: str | None = None,
        expected_status: str | None = None,
    ) -> bool: ...


@dataclass
class DispatchOutcome:
    """Truthful result of one dispatch. ``resent`` is True when a second send call was made.

    ``final`` is False when the op is not settled yet and this call sent nothing.
    That happens when another claim on the op is still inside its in-flight
    window (detail ``dispatch_in_flight``), or when the ledger failed before
    the transport was called (detail ``ledger_unavailable``, or
    ``ledger_conflict``). The caller must not present ``status`` as the outcome.
    It should keep the consent pending and call ``dispatch`` (or ``reconcile``)
    again later. ``status`` still holds the most cautious true value for callers
    that ignore ``final``: ``failed`` when the op is provably unclaimed (nothing
    can have been sent), otherwise ``delivery_unknown``.
    """

    status: DispatchStatus
    receipt_id: str | None
    detail: str
    resent: bool
    final: bool = True


# ---------------------------------------------------------------------------
# Keys and hashes
# ---------------------------------------------------------------------------


def summary_hash(summary_text: str) -> str:
    """SHA-256 hex of the exact summary text the caller consented to."""
    if not isinstance(summary_text, str):
        raise TypeError("summary_text must be a string")
    return hashlib.sha256(summary_text.encode("utf-8")).hexdigest()


def make_op_key(session_id: str, draft_hash: str, consent_turn: int) -> str:
    """Stable idempotency key for one consented send (``em_`` + 32 hex chars).

    It is filename-safe and does not reveal the session ID.
    """
    if not isinstance(session_id, str) or not session_id.strip():
        raise ValueError("session_id is required")
    if not isinstance(draft_hash, str) or not draft_hash.strip():
        raise ValueError("draft_hash is required")
    if isinstance(consent_turn, bool) or not isinstance(consent_turn, int) or consent_turn < 0:
        raise ValueError("consent_turn must be a non-negative integer")
    material = "\x1f".join(("email-op/v1", session_id, draft_hash, str(consent_turn)))
    return "em_" + hashlib.sha256(material.encode("utf-8")).hexdigest()[:32]


# ---------------------------------------------------------------------------
# Message building (shared by transports)
# ---------------------------------------------------------------------------


def _has_line_break(value: str) -> bool:
    return any(ch in value for ch in ("\r", "\n", "\x00"))


def _check_message(message: EmailMessage) -> str | None:
    """Return a PII-free reason code when the message must not be sent, else None."""
    if not _OP_KEY_RE.fullmatch(message.op_key or ""):
        return "invalid_op_key"
    if not _ADDRESS_RE.fullmatch(message.to_address or ""):
        return "invalid_recipient"
    if not _ADDRESS_RE.fullmatch(message.from_address or ""):
        return "invalid_sender"
    subject = message.subject or ""
    if not subject.strip() or _has_line_break(subject) or len(subject) > MAX_SUBJECT_CHARS:
        return "invalid_subject"
    if "\x00" in (message.body_text or "") or len(message.body_text or "") > MAX_BODY_CHARS:
        return "invalid_body"
    return None


def _message_id(op_key: str, from_address: str) -> str:
    """Deterministic Message-ID derived from the op key (same consent, same ID)."""
    domain = from_address.rpartition("@")[2].strip().lower()
    if not _DOMAIN_RE.fullmatch(domain):
        domain = "localhost"
    return f"<{op_key}@{domain}>"


def _build_mime(message: EmailMessage, *, message_id: str, at: datetime) -> MimeMessage:
    mime = MimeMessage()
    mime["From"] = message.from_address
    mime["To"] = message.to_address
    mime["Subject"] = message.subject
    mime["Date"] = format_datetime(_aware(at).astimezone(UTC))
    mime["Message-ID"] = message_id
    mime["X-Claims-Op-Key"] = message.op_key
    mime.set_content(message.body_text)
    return mime


# ---------------------------------------------------------------------------
# Transports
# ---------------------------------------------------------------------------


class OutboxTransport:
    """Local demo transport: writes ``<outbox_dir>/<op_key>.eml`` and reports ``queued``.

    Writing a file is not delivery, so receipts are never ``sent``. Idempotent
    by file: an existing file for the op key is returned as the receipt and is
    never overwritten. Files are written atomically (temp file in the same
    directory, then an atomic no-clobber link, falling back to ``os.replace``)
    with mode 0600 in a 0700 directory.
    """

    name = "outbox"
    supports_reconciliation = True

    def __init__(self, outbox_dir: Path, *, clock: Clock | None = None) -> None:
        self._dir = Path(outbox_dir)
        self._clock = clock or _utcnow

    @property
    def outbox_dir(self) -> Path:
        return self._dir

    def path_for(self, op_key: str) -> Path:
        """Path of the ``.eml`` for ``op_key``; ValueError for unsafe keys (no traversal)."""
        if not _OP_KEY_RE.fullmatch(op_key or ""):
            raise ValueError("invalid op key")
        return self._dir / f"{op_key}.eml"

    def send(self, message: EmailMessage, *, timeout: float) -> DeliveryReceipt:
        reason = _check_message(message)
        if reason:
            raise EmailTransportError("rejected", definitely_not_sent=True, message=reason)
        existing = self.lookup(message.op_key)
        if existing is not None:
            return existing
        path = self.path_for(message.op_key)
        at = _aware(self._clock())
        message_id = _message_id(message.op_key, message.from_address)
        data = _build_mime(message, message_id=message_id, at=at).as_bytes(policy=email_policy.SMTP)
        try:
            self._ensure_dir()
            created = self._write_once(path, data)
        except OSError:
            if path.exists():  # published before the error surfaced
                return self._receipt_from_file(message.op_key, path)
            raise EmailTransportError("unavailable", definitely_not_sent=True, message="outbox_write_failed") from None
        if not created:  # another worker published first; theirs is the receipt
            return self._receipt_from_file(message.op_key, path)
        return DeliveryReceipt(message.op_key, "queued", self.name, message_id, at)

    def lookup(self, op_key: str) -> DeliveryReceipt | None:
        try:
            path = self.path_for(op_key)
        except ValueError:
            return None
        if not path.is_file():
            return None
        return self._receipt_from_file(op_key, path)

    # -- internals --------------------------------------------------------

    def _ensure_dir(self) -> None:
        self._dir.mkdir(mode=0o700, parents=True, exist_ok=True)
        try:
            os.chmod(self._dir, 0o700)
        except OSError:  # best effort (e.g. a mounted volume we do not own)
            pass

    def _write_once(self, path: Path, data: bytes) -> bool:
        """Atomically publish ``data`` at ``path`` unless it exists. True when we created it."""
        fd, tmp_name = tempfile.mkstemp(prefix=".tmp-", suffix=".eml", dir=self._dir)
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
            os.chmod(tmp_name, 0o600)
            return self._publish(tmp_name, path)
        finally:
            try:
                os.unlink(tmp_name)
            except FileNotFoundError:
                pass

    def _publish(self, tmp_name: str, path: Path) -> bool:
        try:
            os.link(tmp_name, path)  # atomic create-if-absent: never clobbers a receipt
        except FileExistsError:
            return False
        except OSError:  # hard links unsupported on this filesystem
            if path.exists():
                return False
            os.replace(tmp_name, path)
        self._fsync_dir()
        return True

    def _fsync_dir(self) -> None:
        try:
            dir_fd = os.open(self._dir, os.O_RDONLY)
        except OSError:
            return
        try:
            os.fsync(dir_fd)
        except OSError:
            pass
        finally:
            os.close(dir_fd)

    def _receipt_from_file(self, op_key: str, path: Path) -> DeliveryReceipt:
        try:
            with path.open("rb") as handle:
                headers = BytesHeaderParser(policy=email_policy.default).parse(handle)
            mtime = path.stat().st_mtime
        except OSError:
            raise EmailTransportError("unavailable", definitely_not_sent=False, message="outbox_read_failed") from None
        message_id = str(headers.get("Message-ID") or "").strip() or None
        return DeliveryReceipt(op_key, "queued", self.name, message_id, _header_date(headers, mtime))


def _header_date(headers: Message, fallback_ts: float) -> datetime:
    raw = headers.get("Date")
    if raw:
        try:
            return _aware(parsedate_to_datetime(str(raw)))
        except (TypeError, ValueError):
            pass
    return datetime.fromtimestamp(fallback_ts, tz=UTC)


class SmtpTransport:
    """Real delivery over SMTP (STARTTLS + optional login).

    SMTP offers no lookup by op key, so ``supports_reconciliation`` is False:
    an ambiguous failure ends as ``delivery_unknown``. Failures before the
    message is handed to ``send_message`` are definitely not sent; a timeout or
    disconnect during ``send_message`` is ambiguous; a 4xx/5xx reply to the
    envelope or data is a definite non-delivery.
    """

    name = "smtp"
    supports_reconciliation = False

    def __init__(
        self,
        host: str,
        port: int = 587,
        *,
        username: str | None = None,
        password: str | None = None,
        starttls: bool = True,
        clock: Clock | None = None,
        ssl_context: ssl.SSLContext | None = None,
    ) -> None:
        if not host or not host.strip():
            raise ValueError("SMTP host is required")
        if not 0 < int(port) < 65536:
            raise ValueError("SMTP port out of range")
        self._host = host.strip()
        self._port = int(port)
        self._username = username or None
        self._password = password
        self._starttls = starttls
        self._clock = clock or _utcnow
        self._ssl_context = ssl_context

    def __repr__(self) -> str:  # never include credentials
        return f"SmtpTransport(host={self._host!r}, port={self._port}, starttls={self._starttls})"

    def send(self, message: EmailMessage, *, timeout: float) -> DeliveryReceipt:
        reason = _check_message(message)
        if reason:
            raise EmailTransportError("rejected", definitely_not_sent=True, message=reason)
        at = _aware(self._clock())
        message_id = _message_id(message.op_key, message.from_address)
        mime = _build_mime(message, message_id=message_id, at=at)
        phase = "connect"
        try:
            client = smtplib.SMTP(self._host, self._port, timeout=timeout)
            try:
                phase = "handshake"
                client.ehlo()
                if self._starttls:
                    client.starttls(context=self._ssl_context or ssl.create_default_context())
                    client.ehlo()
                if self._username:
                    phase = "auth"
                    client.login(self._username, self._password or "")
                phase = "send"
                refused = client.send_message(mime, from_addr=message.from_address, to_addrs=[message.to_address])
                phase = "accepted"
            finally:
                _quit_quietly(client)
        except Exception as exc:
            raise _map_smtp_error(exc, phase) from None
        if refused:
            raise EmailTransportError("rejected", definitely_not_sent=True, message="smtp_recipient_refused")
        return DeliveryReceipt(message.op_key, "sent", self.name, message_id, at)

    def lookup(self, op_key: str) -> DeliveryReceipt | None:
        return None


def _quit_quietly(client: smtplib.SMTP) -> None:
    try:
        client.quit()
    except Exception:
        try:
            client.close()
        except Exception:
            pass


def _map_smtp_error(exc: BaseException, phase: str) -> EmailTransportError:
    """Map an smtplib/socket failure to a PII-free transport error."""
    in_send = phase == "send"
    prefix = f"smtp_{phase}"
    if isinstance(exc, TimeoutError):
        return EmailTransportError("timeout", definitely_not_sent=not in_send, message=f"{prefix}_timeout")
    if isinstance(exc, smtplib.SMTPRecipientsRefused):
        return EmailTransportError("rejected", definitely_not_sent=True, message=f"{prefix}_recipient_refused")
    if isinstance(exc, smtplib.SMTPResponseException):
        code = exc.smtp_code if isinstance(exc.smtp_code, int) else -1
        if in_send and 500 <= code < 600:
            return EmailTransportError("rejected", definitely_not_sent=True, message=f"{prefix}_{code}")
        if in_send and not 400 <= code < 500:
            return EmailTransportError("unavailable", definitely_not_sent=False, message=f"{prefix}_reply")
        return EmailTransportError("unavailable", definitely_not_sent=True, message=f"{prefix}_{code}")
    if isinstance(exc, smtplib.SMTPServerDisconnected):
        return EmailTransportError("unavailable", definitely_not_sent=not in_send, message=f"{prefix}_disconnected")
    return EmailTransportError("unavailable", definitely_not_sent=not in_send, message=f"{prefix}_error")


class DisabledTransport:
    """Email turned off: every send is a definite, honest failure."""

    name = "disabled"
    supports_reconciliation = False

    def send(self, message: EmailMessage, *, timeout: float) -> DeliveryReceipt:
        raise EmailTransportError("unavailable", definitely_not_sent=True, message="email_disabled")

    def lookup(self, op_key: str) -> DeliveryReceipt | None:
        return None


SCRIPT_BEHAVIORS: Final = frozenset(
    {
        "ok",
        "timeout_after_send",
        "timeout_before_send",
        "reject",
        "unavailable",
        "crash_before_send",
        "crash_after_send",
    }
)


class SimulatedCrash(BaseException):
    """Stands in for a process kill in tests: it escapes ``except Exception`` handlers."""


class ScriptedTransport:
    """Deterministic test transport.

    Each ``send`` consumes the next behavior (``"ok"`` once the script is
    exhausted). ``delivered`` records every message that actually went out,
    duplicates included, so tests can assert at-most-once delivery. It is
    deliberately *not* idempotent, even when reconcilable, so that a duplicate
    send by the dispatcher is never hidden.
    """

    def __init__(
        self,
        behaviors: Sequence[str] = ("ok",),
        *,
        supports_reconciliation: bool = True,
        receipt_status: ReceiptStatus = "sent",
        lookup_fails: bool = False,
        name: str = "scripted",
        clock: Clock | None = None,
    ) -> None:
        unknown = sorted(set(behaviors) - SCRIPT_BEHAVIORS)
        if unknown:
            raise ValueError(f"unknown scripted behaviors {unknown}")
        if receipt_status not in ("sent", "queued"):
            raise ValueError("receipt_status must be 'sent' or 'queued'")
        self.name = name
        self.supports_reconciliation = supports_reconciliation
        self.receipt_status: ReceiptStatus = receipt_status
        self.lookup_fails = lookup_fails
        self._behaviors = list(behaviors)
        self._clock = clock or _utcnow
        self.delivered: list[EmailMessage] = []
        self.send_calls: list[str] = []
        self.lookup_calls: list[str] = []

    def send(self, message: EmailMessage, *, timeout: float) -> DeliveryReceipt:
        self.send_calls.append(message.op_key)
        behavior = self._behaviors.pop(0) if self._behaviors else "ok"
        if behavior == "reject":
            raise EmailTransportError("rejected", definitely_not_sent=True, message="scripted_reject")
        if behavior == "unavailable":
            raise EmailTransportError("unavailable", definitely_not_sent=True, message="scripted_unavailable")
        if behavior == "timeout_before_send":
            raise EmailTransportError("timeout", definitely_not_sent=False, message="scripted_timeout")
        if behavior == "crash_before_send":
            raise SimulatedCrash("crash_before_send")
        self.delivered.append(message)
        if behavior == "timeout_after_send":
            raise EmailTransportError("timeout", definitely_not_sent=False, message="scripted_timeout")
        if behavior == "crash_after_send":
            raise SimulatedCrash("crash_after_send")
        return self._receipt(message.op_key)

    def lookup(self, op_key: str) -> DeliveryReceipt | None:
        self.lookup_calls.append(op_key)
        if self.lookup_fails:
            raise EmailTransportError("unavailable", definitely_not_sent=False, message="scripted_lookup_failed")
        if not self.supports_reconciliation:
            return None
        if any(m.op_key == op_key for m in self.delivered):
            return self._receipt(op_key)
        return None

    def _receipt(self, op_key: str) -> DeliveryReceipt:
        return DeliveryReceipt(op_key, self.receipt_status, self.name, f"<{op_key}@scripted.invalid>", self._clock())


def transport_from_settings(settings: Settings, *, clock: Clock | None = None) -> EmailTransport:
    """Build the configured transport (``outbox`` | ``smtp`` | ``disabled``)."""
    if settings.email_transport == "outbox":
        return OutboxTransport(settings.outbox_dir, clock=clock)
    if settings.email_transport == "smtp":
        if not settings.smtp_host:
            raise ValueError("EMAIL_TRANSPORT=smtp requires SMTP_HOST")
        return SmtpTransport(
            settings.smtp_host,
            settings.smtp_port,
            username=settings.smtp_username,
            password=settings.smtp_password,
            starttls=settings.smtp_starttls,
            clock=clock,
        )
    return DisabledTransport()


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

_RESUME_CLAIM: Final = "resume_in_flight"
"""``detail`` of a ``delivery_unknown`` row while an orphaned claim is being resumed."""

_MIN_CLAIM_TTL_S: Final = 60.0


class _Ledger(Enum):
    DOWN = "down"
    """The ledger could not be read: never send; the op is not settled."""


def _is_claimed(row: EmailOpRecord) -> bool:
    """True while someone may be sending: a ``dispatching`` claim or a resume in flight."""
    return row.status == "dispatching" or (row.status == "delivery_unknown" and row.detail == _RESUME_CLAIM)


def _unsettled(status: DispatchStatus, detail: str, *, resent: bool = False) -> DispatchOutcome:
    return DispatchOutcome(status, None, detail, resent, final=False)


class EmailDispatcher:
    """Executes one consented send per op key, at most once, with truthful status.

    The op row (``pending``) is created by the consent commit. Recipient and
    consent are decided by the agent and its guardrails; this class only guarantees the
    durable-write protocol: claim by CAS, send, record, reconcile ambiguity.

    ``claim_ttl`` (seconds) is how long a ``dispatching`` claim may belong to a
    live owner, which sends, looks up, and retries once. The default is
    ``max(60, 4 * timeout)``. Until a claim is that old, other callers report it
    as in flight and leave it alone. After that it counts as orphaned.
    """

    def __init__(
        self,
        transport: EmailTransport,
        ledger: EmailLedger,
        *,
        from_address: str,
        timeout: float,
        clock: Clock = _utcnow,
        claim_ttl: float | None = None,
    ) -> None:
        if not timeout or timeout <= 0:
            raise ValueError("timeout must be positive")
        if not _ADDRESS_RE.fullmatch(from_address or ""):
            raise ValueError("from_address is not a valid email address")
        if claim_ttl is None:
            claim_ttl = max(_MIN_CLAIM_TTL_S, 4.0 * float(timeout))
        if claim_ttl <= 0:
            raise ValueError("claim_ttl must be positive")
        self._transport = transport
        self._ledger = ledger
        self._from = from_address
        self._timeout = float(timeout)
        self._claim_ttl = timedelta(seconds=float(claim_ttl))
        self._clock = clock

    @property
    def claim_ttl_seconds(self) -> float:
        return self._claim_ttl.total_seconds()

    # -- public API -------------------------------------------------------

    def dispatch(self, *, op_key: str, to_address: str, subject: str, body_text: str) -> DispatchOutcome:
        """Send the consented summary for ``op_key`` unless it was already attempted.

        Use this, with the rebuilt consented message, for both ``pending`` and
        ``dispatching`` rows. A ``dispatching`` row is never re-sent blindly. A
        receipt found by lookup is recorded. A claim younger than ``claim_ttl``
        is reported as in flight (``final=False``). An orphaned claim is resumed
        with at most one more send, with the same op key, and only when the
        transport is reconcilable and a lookup succeeded and found nothing.
        Every other orphan becomes ``delivery_unknown``.
        """
        started = self._clock()
        outcome = self._dispatch(op_key, to_address, subject, body_text)
        self._report("email_dispatch", op_key, outcome, started)
        return outcome

    def reconcile(self, op_key: str) -> DispatchOutcome | None:
        """Resolve an in-flight or unknown op without ever sending. None when nothing was dispatched."""
        started = self._clock()
        row = self._get(op_key)
        if row is _Ledger.DOWN:
            outcome: DispatchOutcome | None = _unsettled("delivery_unknown", "ledger_unavailable")
        elif row is None or row.status == "pending":
            return None
        elif _is_claimed(row):
            outcome = self._resolve_claimed(row, None)
        elif row.status in ("sent", "queued", "failed"):
            outcome = _from_row(row)
        elif row.status == "delivery_unknown":
            outcome = self._recheck_unknown(row)
        else:
            outcome = DispatchOutcome("delivery_unknown", None, "unknown_ledger_status", False)
        self._report("email_reconcile", op_key, outcome, started)
        return outcome

    # -- dispatch protocol ------------------------------------------------

    def _dispatch(self, op_key: str, to_address: str, subject: str, body_text: str) -> DispatchOutcome:
        row = self._get(op_key)
        if row is _Ledger.DOWN:  # sent nothing now, but an earlier attempt may have
            return _unsettled("delivery_unknown", "ledger_unavailable")
        if row is None:
            return DispatchOutcome("failed", None, "no_consent_record", False)
        message = EmailMessage(op_key, to_address, self._from, subject, body_text)
        if _is_claimed(row):
            return self._resolve_claimed(row, message)
        if row.status in _FINAL:
            return _from_row(row)  # sent/queued: no resend; failed/unknown: never auto-resend
        if row.status != "pending":
            return DispatchOutcome("delivery_unknown", None, "unknown_ledger_status", False)
        if row.draft_hash != summary_hash(body_text):
            return self._settle_pending(op_key, "draft_mismatch")
        reason = _check_message(message)
        if reason:
            return self._settle_pending(op_key, reason)
        claimed = self._cas(op_key, "dispatching", expected="pending")
        if claimed is None:  # the claim may or may not have landed; nothing was sent
            return self._after_lost_claim(op_key, unclaimed_detail="ledger_unavailable")
        if not claimed:
            return self._after_lost_claim(op_key)
        return self._send(message)

    def _send(self, message: EmailMessage) -> DispatchOutcome:
        try:
            receipt = self._transport.send(message, timeout=self._timeout)
        except EmailTransportError as exc:
            if exc.definitely_not_sent:
                return self._record(message.op_key, "failed", None, exc.kind, resent=False)
            return self._resolve_ambiguous(message, exc.kind)
        except Exception as exc:
            logger.warning("email transport raised %s op_key=%s", type(exc).__name__, message.op_key)
            return self._resolve_ambiguous(message, "transport_error", allow_retry=False)
        return self._record_receipt(message.op_key, receipt, detail=None, resent=False)

    def _resolve_ambiguous(self, message: EmailMessage, kind: str, *, allow_retry: bool = True) -> DispatchOutcome:
        """A send may or may not have happened: look it up before at most one retry."""
        op_key = message.op_key
        if not self._transport.supports_reconciliation:
            return self._record(op_key, "delivery_unknown", None, f"{kind}_unreconcilable", resent=False)
        found, lookup_ok = self._lookup(op_key)
        if found is not None:
            return self._record_receipt(op_key, found, detail=f"reconciled_after_{kind}", resent=False)
        if not lookup_ok or not allow_retry:
            reason = "lookup_failed" if not lookup_ok else "not_found"
            return self._record(op_key, "delivery_unknown", None, f"{kind}_{reason}", resent=False)
        return self._retry_once(message)

    def _retry_once(self, message: EmailMessage, *, claim: str = "dispatching") -> DispatchOutcome:
        """The one extra send with the same op key. ``claim`` is the row status we hold it under."""
        op_key = message.op_key
        logger.info("email retry after confirmed non-delivery op_key=%s", op_key)
        try:
            receipt = self._transport.send(message, timeout=self._timeout)
        except EmailTransportError as exc:
            if exc.definitely_not_sent:
                return self._record(op_key, "failed", None, f"retry_{exc.kind}", resent=True, claim=claim)
            kind = f"retry_{exc.kind}"
        except Exception as exc:
            logger.warning("email transport raised %s op_key=%s", type(exc).__name__, op_key)
            kind = "retry_transport_error"
        else:
            return self._record_receipt(op_key, receipt, detail="sent_on_retry", resent=True, claim=claim)
        found, _ = self._lookup(op_key)
        if found is not None:
            return self._record_receipt(op_key, found, detail=f"reconciled_after_{kind}", resent=True, claim=claim)
        return self._record(op_key, "delivery_unknown", None, kind, resent=True, claim=claim)

    def _resolve_claimed(self, row: EmailOpRecord, message: EmailMessage | None) -> DispatchOutcome:
        """Someone claimed this op. Report it. Only an orphaned ``dispatching`` claim is ever resumed.

        A receipt found by lookup always wins. A claim inside ``claim_ttl`` may
        still be sending, so it is reported as in flight and left alone. That
        way a reconcile from another worker cannot replace the owner's result
        with ``delivery_unknown``. An older claim is resumed (see ``_resume``)
        when ``message`` is given, the transport is reconcilable, and the lookup
        succeeded and found nothing. Otherwise it is settled as
        ``delivery_unknown``.
        """
        op_key = row.op_key
        reconcilable = bool(self._transport.supports_reconciliation)
        lookup_ok = False
        if reconcilable:
            found, lookup_ok = self._lookup(op_key)
            if found is not None:
                return self._record_receipt(op_key, found, detail="reconciled", resent=False)
        if not self._claim_expired(row):
            return _unsettled("delivery_unknown", "dispatch_in_flight")
        if row.status == "dispatching":
            if message is not None and reconcilable and lookup_ok and self._resumable(row, message):
                return self._resume(message)
            return self._record(op_key, "delivery_unknown", None, "dispatch_interrupted", resent=False)
        # A resume whose worker died: its one extra send may have happened. Never send again.
        return self._record(op_key, "delivery_unknown", None, "resume_interrupted", resent=False, claim="delivery_unknown")

    def _resume(self, message: EmailMessage) -> DispatchOutcome:
        """Take over an orphaned claim, then send at most once more with the same op key.

        The takeover is a compare-and-set from ``dispatching`` to
        ``delivery_unknown`` (detail ``resume_in_flight``). Only one worker can
        win it, and a row never returns to ``dispatching``, so one consent gets
        at most one resend. If this worker dies mid-resend, the row is left
        ``delivery_unknown``, which is never re-sent.
        """
        op_key = message.op_key
        took = self._cas(op_key, "delivery_unknown", expected="dispatching", detail=_RESUME_CLAIM)
        if took is None:
            return _unsettled("delivery_unknown", "ledger_unavailable")
        if not took:
            return self._after_lost_claim(op_key)
        found, lookup_ok = self._lookup(op_key)  # look again now that we hold the op
        if found is not None:
            return self._record_receipt(op_key, found, detail="reconciled", resent=False, claim="delivery_unknown")
        if not lookup_ok:
            return self._record(op_key, "delivery_unknown", None, "dispatch_interrupted", resent=False, claim="delivery_unknown")
        logger.info("email resume of an interrupted dispatch op_key=%s", op_key)
        return self._retry_once(message, claim="delivery_unknown")

    def _resumable(self, row: EmailOpRecord, message: EmailMessage) -> bool:
        """Only the exact consented draft, as a valid message, may be resent."""
        if not isinstance(message.body_text, str):
            return False
        return row.draft_hash == summary_hash(message.body_text) and _check_message(message) is None

    def _claim_expired(self, row: EmailOpRecord) -> bool:
        """True when the claim is older than ``claim_ttl``. If its age cannot be read, the owner may be live."""
        try:
            return _aware(self._clock()) - _aware(row.updated_at) > self._claim_ttl
        except Exception:
            return False

    def _recheck_unknown(self, row: EmailOpRecord) -> DispatchOutcome:
        """Never sends. Upgrades a settled ``delivery_unknown`` when proof of delivery appears."""
        if self._transport.supports_reconciliation:
            found, _ = self._lookup(row.op_key)
            if found is not None:
                return self._record_receipt(row.op_key, found, detail="reconciled", resent=False)
        return _from_row(row)

    def _after_lost_claim(self, op_key: str, *, unclaimed_detail: str = "ledger_conflict") -> DispatchOutcome:
        """Our claim did not land. Report the op as it now stands; never send.

        If the op is still ``pending``, nobody claimed it, so nothing was handed
        to the transport. It is reported ``failed`` but not final, so the
        consent can be retried.
        """
        row = self._get(op_key)
        if row is _Ledger.DOWN:
            return _unsettled("delivery_unknown", "ledger_unavailable")
        if row is None:
            return DispatchOutcome("failed", None, "no_consent_record", False)
        if row.status == "pending":
            return _unsettled("failed", unclaimed_detail)
        if _is_claimed(row):
            return self._resolve_claimed(row, None)
        if row.status in _FINAL:
            return _from_row(row)
        return DispatchOutcome("delivery_unknown", None, "unknown_ledger_status", False)

    def _settle_pending(self, op_key: str, reason: str) -> DispatchOutcome:
        """Close a pending op as failed without sending (invalid draft or message)."""
        logger.warning("email op refused before send op_key=%s reason=%s", op_key, reason)
        closed = self._cas(op_key, "failed", expected="pending", detail=reason)
        if closed is None:
            return DispatchOutcome("failed", None, f"{reason}|unrecorded", False)
        if not closed:
            return self._after_lost_claim(op_key)
        return DispatchOutcome("failed", None, reason, False)

    # -- ledger helpers ---------------------------------------------------

    def _record_receipt(
        self,
        op_key: str,
        receipt: DeliveryReceipt,
        *,
        detail: str | None,
        resent: bool,
        claim: str = "dispatching",
    ) -> DispatchOutcome:
        if (
            not isinstance(receipt, DeliveryReceipt)
            or receipt.op_key != op_key
            or receipt.status
            not in (
                "sent",
                "queued",
            )
        ):
            logger.warning("email transport returned an invalid receipt op_key=%s", op_key)
            return self._record(op_key, "delivery_unknown", None, "invalid_receipt", resent=resent, claim=claim)
        default = "accepted" if receipt.status == "sent" else "queued_not_delivered"
        return self._record(op_key, receipt.status, receipt.message_id, detail or default, resent=resent, claim=claim)

    def _record(
        self,
        op_key: str,
        status: DispatchStatus,
        receipt_id: str | None,
        detail: str,
        *,
        resent: bool,
        claim: str = "dispatching",
    ) -> DispatchOutcome:
        """Persist a result for an op we hold. ``claim`` is the status our claim left on the row.

        Definitive evidence (a receipt or a definite failure) may also replace a
        ``delivery_unknown`` written by a concurrent reconcile or resume. If the
        ledger already holds a different settled result, the ledger wins. If
        another worker is resuming the op, an unknown result is reported as
        still in flight.
        """
        outcome = DispatchOutcome(status, receipt_id, detail, resent)
        try:
            if self._update(op_key, outcome, expected=claim):
                return outcome
            current = self._ledger.email_op_get(op_key)
            if current is not None and current.status == "delivery_unknown" and status != "delivery_unknown":
                if self._update(op_key, outcome, expected="delivery_unknown"):
                    return outcome
                current = self._ledger.email_op_get(op_key)
        except Exception as exc:
            logger.warning("email ledger write failed (%s) op_key=%s", type(exc).__name__, op_key)
            return DispatchOutcome(status, receipt_id, f"{detail}|unrecorded", resent)
        if current is None:
            return DispatchOutcome(status, receipt_id, f"{detail}|unrecorded", resent)
        if _is_claimed(current):
            if status == "delivery_unknown":
                return _unsettled("delivery_unknown", "dispatch_in_flight", resent=resent)
            return DispatchOutcome(status, receipt_id, f"{detail}|unrecorded", resent)
        if current.status in _FINAL:
            recorded = _from_row(current)
            recorded.resent = resent
            return recorded
        return DispatchOutcome(status, receipt_id, f"{detail}|unrecorded", resent)

    def _update(self, op_key: str, outcome: DispatchOutcome, *, expected: str) -> bool:
        return bool(
            self._ledger.email_op_update(
                op_key,
                status=outcome.status,
                receipt_id=outcome.receipt_id,
                detail=outcome.detail,
                expected_status=expected,
            )
        )

    def _cas(self, op_key: str, status: str, *, expected: str, detail: str | None = None) -> bool | None:
        """Compare-and-set; None when the ledger itself failed (never send then)."""
        try:
            return bool(self._ledger.email_op_update(op_key, status=status, detail=detail, expected_status=expected))
        except Exception as exc:
            logger.warning("email ledger CAS failed (%s) op_key=%s", type(exc).__name__, op_key)
            return None

    def _get(self, op_key: str) -> EmailOpRecord | Literal[_Ledger.DOWN] | None:
        try:
            return self._ledger.email_op_get(op_key)
        except Exception as exc:
            logger.warning("email ledger read failed (%s) op_key=%s", type(exc).__name__, op_key)
            return _Ledger.DOWN

    def _lookup(self, op_key: str) -> tuple[DeliveryReceipt | None, bool]:
        """(receipt or None, lookup_succeeded). A failed lookup proves nothing."""
        try:
            found = self._transport.lookup(op_key)
        except Exception as exc:
            logger.warning("email lookup failed (%s) op_key=%s", type(exc).__name__, op_key)
            return None, False
        if found is not None and (not isinstance(found, DeliveryReceipt) or found.op_key != op_key):
            logger.warning("email lookup returned a mismatched receipt op_key=%s", op_key)
            return None, False
        return found, True

    def _report(self, name: str, op_key: str, outcome: DispatchOutcome | None, started: datetime) -> None:
        if outcome is None:
            return
        elapsed_ms = None
        try:
            elapsed_ms = int((self._clock() - started).total_seconds() * 1000)
        except Exception:
            pass
        logger.info(
            "%s op_key=%s status=%s detail=%s resent=%s final=%s transport=%s",
            name,
            op_key,
            outcome.status,
            outcome.detail,
            outcome.resent,
            outcome.final,
            self._transport.name,
        )
        try:
            tracing.event(
                name,
                type="tool",
                op_key=op_key,
                status=outcome.status,
                detail=outcome.detail,
                resent=outcome.resent,
                final=outcome.final,
                transport=self._transport.name,
                elapsed_ms=elapsed_ms,
            )
        except Exception:
            pass


def _from_row(row: EmailOpRecord) -> DispatchOutcome:
    return DispatchOutcome(row.status, row.receipt_id, row.detail or f"recorded_{row.status}", False)  # type: ignore[arg-type]
