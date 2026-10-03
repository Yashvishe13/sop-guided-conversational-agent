"""Conversation service: the transactional shell around the claims agent.

Responsibilities (no SOP logic lives here):

* session lifecycle: opaque session IDs, a per-session secret held in an
  HttpOnly cookie (only its hash is stored), HMAC-derived CSRF tokens, expiry;
* turn idempotency: a ``client_turn_id`` is processed once; duplicates replay
  the stored response and a reused ID with a different body is rejected;
* concurrency: a per-session in-process lock plus optimistic versioning in
  SQLite, so simultaneous turns can never overwrite each other;
* checkpoints: one commit per accepted turn, and an extra commit that records
  email consent and the operation key *before* dispatch;
* recovery: on load, consented-but-unfinished email operations are reconciled
  with the ledger and transport before anything else happens; operations of
  sessions nobody can resume any more are closed (never sent unattended);
* visibility: what the browser gets back applies the verification idle TTL and
  hides messages from before the latest identity reset (the stored state is
  never changed by a read);
* retention: a time-based purge of sessions, turns, ledgers, traces and outbox files.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import secrets
import threading
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Callable

from insurance_claims.agent.loop import ClaimsAgent, TurnInput, fingerprint
from insurance_claims.config import Settings
from insurance_claims.domain.models import Phase
from insurance_claims.domain.records import EmailOpRecord, TurnRecord
from insurance_claims.domain.state import (
    CaseSelection,
    ChatMessage,
    EmailOffer,
    SessionState,
    VerificationState,
)
from insurance_claims.mail.sender import DispatchOutcome, EmailDispatcher
from insurance_claims.observability import tracing
from insurance_claims.persistence.store import (
    SessionCorrupt,
    SessionNotFound,
    SessionRow,
    SessionStore,
    VersionConflict,
)

logger = logging.getLogger(__name__)

"""Phase-log reasons that end a verified epoch: earlier messages are no longer shown."""
_PURGE_INTERVAL = timedelta(hours=1)
_ABANDONED_OP_GRACE = timedelta(minutes=5)
"""An unfinished email op of an expired session is only swept after this much ledger idleness."""
_EMAIL_PENDING_TEXT = "I'm still confirming whether your email summary went out. I'll let you know as soon as I can confirm it."
_TERMINAL_OP_STATUSES = frozenset({"sent", "queued", "failed"})


def _settled(result: DispatchOutcome) -> bool:
    """False when the dispatcher reports the op as not settled yet (nothing sent by that call)."""
    return bool(getattr(result, "final", True))


class ServiceError(Exception):
    def __init__(self, status: int, code: str, message: str) -> None:
        super().__init__(message)
        self.status = status
        self.code = code
        self.message = message


def _utf8(value: str) -> bytes:
    """Encode any str (including lone surrogates from odd headers) without raising."""
    return value.encode("utf-8", "surrogatepass")


def _hash(value: str) -> str:
    return hashlib.sha256(_utf8(value)).hexdigest()


def message_view(msg: ChatMessage) -> dict[str, Any]:
    return {"role": msg.role, "text": msg.text, "kind": msg.kind, "turn_index": msg.turn_index}


@dataclass
class _LockEntry:
    lock: threading.Lock
    users: int = 0


class ConversationService:
    def __init__(
        self,
        *,
        settings: Settings,
        store: SessionStore,
        agent: ClaimsAgent,
        dispatcher: EmailDispatcher,
        csrf_key: bytes,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self.settings = settings
        self.store = store
        self.agent = agent
        self.dispatcher = dispatcher
        self.csrf_key = csrf_key
        self.clock = clock or (lambda: datetime.now(UTC))
        self._locks: dict[str, _LockEntry] = {}
        self._locks_guard = threading.Lock()
        # Turn request hashes are keyed: a short answer such as four digits must not be
        # recoverable from the plain ``turns.request_hash`` column without a server secret.
        self._request_hash_key = hmac.new(csrf_key, b"turn-request-hash/v1", hashlib.sha256).digest()
        self._last_purge: datetime | None = None
        self._purge_guard = threading.Lock()

    # ------------------------------------------------------------------ auth helpers
    def csrf_token(self, session_id: str, secret: str) -> str:
        return hmac.new(self.csrf_key, _utf8(f"{session_id}.{secret}"), hashlib.sha256).hexdigest()

    def _request_hash(self, text: str | None, action: str | None) -> str:
        payload = _utf8(json.dumps({"text": text, "action": action}, sort_keys=True))
        return hmac.new(self._request_hash_key, payload, hashlib.sha256).hexdigest()

    def _load(self, session_id: str) -> SessionRow:
        try:
            return self.store.load(session_id)
        except SessionNotFound as exc:
            raise ServiceError(404, "session_not_found", "This conversation was not found. Please start a new one.") from exc
        except SessionCorrupt as exc:
            tracing.event("checkpoint_corrupt", reason=getattr(exc, "reason", "unknown"))
            raise ServiceError(409, "session_unrecoverable", "This conversation could not be restored. Please start a new one.") from exc

    def _authorize(self, session_id: str, secret: str | None, csrf: str | None = None, *, need_csrf: bool) -> None:
        """Check cookie, CSRF and expiry from the plain row columns, before anything is decrypted."""
        try:
            auth = self.store.load_auth(session_id)
        except (SessionNotFound, UnicodeEncodeError) as exc:
            raise ServiceError(404, "session_not_found", "This conversation was not found. Please start a new one.") from exc
        if not secret or not hmac.compare_digest(_hash(secret).encode(), _utf8(auth.secret_hash)):
            raise ServiceError(401, "unauthorized", "This conversation belongs to a different browser session.")
        if need_csrf and (not csrf or not hmac.compare_digest(_utf8(csrf), self.csrf_token(session_id, secret).encode())):
            raise ServiceError(403, "csrf_failed", "The request could not be verified. Please reload the page.")
        if self._now() > auth.expires_at:
            raise ServiceError(404, "session_expired", "This conversation has expired. Please start a new one.")

    @staticmethod
    def _conflict_error(exc: VersionConflict) -> ServiceError:
        if getattr(exc, "reason", "") == "session_missing":
            return ServiceError(404, "session_not_found", "This conversation was not found. Please start a new one.")
        return ServiceError(409, "version_conflict", "Another message changed this conversation first. Please retry.")

    def _now(self) -> datetime:
        now = self.clock()
        return now if now.tzinfo else now.replace(tzinfo=UTC)

    def _acquire(self, session_id: str) -> _LockEntry:
        with self._locks_guard:
            entry = self._locks.setdefault(session_id, _LockEntry(threading.Lock()))
            entry.users += 1
        if not entry.lock.acquire(blocking=False):
            self._release(session_id, entry, locked=False)
            raise ServiceError(409, "turn_in_progress", "Still working on your previous message.")
        return entry

    def _release(self, session_id: str, entry: _LockEntry, *, locked: bool = True) -> None:
        if locked:
            entry.lock.release()
        with self._locks_guard:
            entry.users -= 1
            if entry.users <= 0:
                self._locks.pop(session_id, None)

    # ------------------------------------------------------------------ visibility
    def _idle_expired(self, state: SessionState, now: datetime) -> bool:
        """Same rule as ``ClaimsAgent._expire_if_needed``: verified and idle past the TTL."""
        ttl = timedelta(minutes=self.settings.verification_idle_ttl_minutes)
        return state.verified and now - state.last_activity_at > ttl

    def _visible(self, state: SessionState, now: datetime) -> tuple[SessionState, int]:
        """(state as the browser may see it, first turn index whose messages it may see).

        Read-only: an idle-expired verified state is shown as reset (the stored
        checkpoint stays verified, so the next turn still expires it and says so),
        and messages from before the latest identity reset are hidden.
        """
        if self._idle_expired(state, now):
            view = state.model_copy(deep=True)
            old = view.verification
            view.verification = VerificationState(
                caller_role=old.caller_role,
                representative_declared=old.representative_declared,
                failed_attempts=old.failed_attempts,
            )
            view.case = CaseSelection()
            if view.email.status in ("offered", "none"):
                view.email = EmailOffer(send_count=view.email.send_count)
            view.phase = Phase.VERIFY_ID
            return view, state.turn_index + 1
        return state, max(state.last_reset_turn(), 0)

    # ------------------------------------------------------------------ payloads
    def _session_payload(self, session_id: str, secret: str, state: SessionState) -> dict[str, Any]:
        view, floor = self._visible(state, self._now())
        return {
            "session_id": session_id,
            "csrf_token": self.csrf_token(session_id, secret),
            "state": view.public_view(),
            # Turn 0 is the fixed greeting; it carries nothing from an earlier caller.
            "messages": [message_view(m) for m in state.history if m.turn_index == 0 or m.turn_index >= floor],
        }

    # ------------------------------------------------------------------ lifecycle
    def create_session(self) -> tuple[str, str, dict[str, Any]]:
        session_id = secrets.token_urlsafe(24)
        secret = secrets.token_urlsafe(32)
        with tracing.span("session_create", type="task", input={"session": fingerprint(session_id)}):
            state = self.agent.new_state(session_id)
            expires = self._now() + timedelta(hours=self.settings.session_max_age_hours)
            self.store.create(state, secret_hash=_hash(secret), expires_at=expires)
        self._maybe_purge()
        return session_id, secret, self._session_payload(session_id, secret, state)

    def get_session(self, session_id: str, secret: str | None) -> dict[str, Any]:
        self._maybe_purge()
        self._authorize(session_id, secret, need_csrf=False)
        state = self._load(session_id).state
        if self._needs_email_recovery(state):
            entry = self._acquire(session_id)
            try:
                row = self._load(session_id)  # re-read under the lock: another request may have recovered it
                state = row.state
                if self._needs_email_recovery(state):
                    state, _, _ = self._recover_email(session_id, state, row.version)
            except VersionConflict as exc:
                tracing.event("version_conflict", reason=getattr(exc, "reason", "stale_version"))
                raise self._conflict_error(exc) from exc
            finally:
                self._release(session_id, entry)
        return self._session_payload(session_id, secret or "", state)

    def delete_session(self, session_id: str, secret: str | None, csrf: str | None) -> None:
        # Authorized from plain columns only, so the owner can delete even an unreadable checkpoint.
        self._authorize(session_id, secret, csrf, need_csrf=True)
        self.store.delete(session_id)
        tracing.event("session_deleted", session=fingerprint(session_id))

    def purge(self) -> int:
        """Close abandoned email ops, then delete everything past retention (rows, traces, outbox)."""
        now = self._now()
        with self._purge_guard:
            self._last_purge = now
        retention = timedelta(days=self.settings.retention_days)
        try:
            self.sweep_email_ops()
        except Exception:
            tracing.event("email_sweep_error")
        removed = self.store.purge(
            retention=retention,
            failure_retention=timedelta(hours=self.settings.party_failure_window_hours),
        )
        removed += purge_old_files(self.settings.trace_dir, "*.json", retention, now=now)
        removed += purge_old_files(self.settings.outbox_dir, "*.eml", retention, now=now)
        return removed

    def _maybe_purge(self) -> None:
        """Run the retention purge at most once per interval, driven by any request."""
        now = self._now()
        with self._purge_guard:
            if self._last_purge is not None and now - self._last_purge < _PURGE_INTERVAL:
                return
            self._last_purge = now
        try:
            self.purge()
        except Exception:
            logger.warning("retention purge failed")
            tracing.event("purge_failed")

    def sweep_email_ops(self) -> int:
        """Close unfinished email ops whose session can no longer be resumed. Never sends.

        ``pending``: nothing was handed to the transport, so it is recorded ``failed``
        (``abandoned_before_dispatch``). ``dispatching``: reconciled by lookup when the
        transport supports it, otherwise (or when still unresolved) ``delivery_unknown``,
        and reported for manual review.
        """
        now = self._now()
        # Never touch a claim a live owner may still hold (the dispatcher's in-flight window).
        grace = max(_ABANDONED_OP_GRACE, timedelta(seconds=float(getattr(self.dispatcher, "claim_ttl_seconds", 0) or 0)))
        handled = 0
        for op in self.store.email_ops_abandoned(now=now, idle_before=now - grace):
            try:
                if op.status == "pending":
                    self.store.email_op_update(op.op_key, status="failed", detail="abandoned_before_dispatch", expected_status="pending")
                else:
                    result = self.dispatcher.reconcile(op.op_key)  # never sends
                    if result is not None and not _settled(result):
                        continue  # still in flight; a later sweep settles it
                    current = self.store.email_op_get(op.op_key)
                    if current is not None and current.status == "dispatching":
                        self.store.email_op_update(
                            op.op_key, status="delivery_unknown", detail="abandoned_in_dispatch", expected_status="dispatching"
                        )
                        current = self.store.email_op_get(op.op_key)
                    if current is not None and current.status == "delivery_unknown":
                        logger.warning("email operation needs manual review op_key=%s", op.op_key[:12])
                        tracing.event("email_needs_review", op_key=op.op_key[:12])
                handled += 1
            except Exception:
                tracing.event("email_sweep_error")
        return handled

    # ------------------------------------------------------------------ turns
    def post_turn(
        self,
        session_id: str,
        secret: str | None,
        csrf: str | None,
        *,
        client_turn_id: str,
        text: str | None,
        action: str | None,
    ) -> dict[str, Any]:
        self._maybe_purge()
        with tracing.span(
            "turn",
            type="turn",
            input={"session": fingerprint(session_id), "client_turn": fingerprint(client_turn_id), "kind": "action" if action else "text"},
        ) as root:
            self._authorize(session_id, secret, csrf, need_csrf=True)
            request_hash = self._request_hash(text, action)
            replay = self._replay(session_id, client_turn_id, request_hash)
            if replay is not None:
                root["output"] = {"duplicate": True}
                return replay
            entry = self._acquire(session_id)
            try:
                row = self._load(session_id)
                replay = self._replay(session_id, client_turn_id, request_hash, row.state)
                if replay is not None:
                    root["output"] = {"duplicate": True}
                    return replay
                state, version = row.state, row.version
                recovered: list[ChatMessage] = []
                if self._needs_email_recovery(state):
                    expired = self._idle_expired(state, self._now())
                    state, version, notices = self._recover_email(session_id, state, version)
                    # The notice belongs to the earlier verified epoch; an idle-expired caller does not see it.
                    recovered = [] if expired else notices
                outcome = self.agent.handle_turn(state, TurnInput(client_turn_id=client_turn_id, text=text, action=action))  # type: ignore[arg-type]
                messages = [*recovered, *outcome.messages]
                final_state = outcome.state
                if outcome.pending_email is not None:
                    pe = outcome.pending_email
                    now = self._now()
                    op = EmailOpRecord(
                        op_key=pe.op_key,
                        session_id=session_id,
                        recipient_ref=pe.recipient_ref,
                        draft_hash=pe.draft_hash,
                        status="pending",
                        created_at=now,
                        updated_at=now,
                    )
                    # Checkpoint consent + operation key BEFORE the external write.
                    version = self.store.commit(session_id, expected_version=version, state=final_state, email_ops_create=[op])
                    tracing.event("checkpoint", reason="email_consent_recorded", version=version)
                    result = self._dispatch(pe.op_key, pe.to_address, pe.subject, pe.body_text)
                    if _settled(result):
                        done = self.agent.complete_email(final_state, result)
                        final_state = done.state
                        messages.extend(done.messages)
                    else:
                        # Not settled: keep the consent pending; the next load recovers it.
                        messages.append(self._email_pending_notice(final_state))
                payload = {
                    "turn_index": final_state.turn_index,
                    "duplicate": False,
                    "messages": [message_view(m) for m in messages],
                    "state": final_state.public_view(),
                }
                record = TurnRecord(
                    session_id=session_id,
                    client_turn_id=client_turn_id,
                    turn_index=final_state.turn_index,
                    request_hash=request_hash,
                    response_json=json.dumps(payload),
                    created_at=self._now(),
                )
                version = self.store.commit(session_id, expected_version=version, state=final_state, turn=record)
                root["output"] = {
                    "phase": final_state.phase.value,
                    "version": version,
                    "stop_reason": outcome.stop_reason,
                    "email_status": final_state.email.status,
                }
                return payload
            except VersionConflict as exc:
                reason = getattr(exc, "reason", "stale_version")
                tracing.event("version_conflict", reason=reason)
                if reason == "duplicate_turn":
                    replay = self._replay(session_id, client_turn_id, request_hash)
                    if replay is not None:
                        return replay
                raise self._conflict_error(exc) from exc
            finally:
                self._release(session_id, entry)

    def _replay(self, session_id: str, client_turn_id: str, request_hash: str, state: SessionState | None = None) -> dict[str, Any] | None:
        try:
            record = self.store.get_turn(session_id, client_turn_id)
        except SessionCorrupt as exc:
            tracing.event("checkpoint_corrupt", reason=getattr(exc, "reason", "unknown"))
            raise ServiceError(409, "session_unrecoverable", "This conversation could not be restored. Please start a new one.") from exc
        if record is None:
            return None
        if not hmac.compare_digest(_utf8(record.request_hash), request_hash.encode()):
            raise ServiceError(409, "turn_id_reused", "This message ID was already used for a different message.")
        tracing.event("duplicate_turn_replayed")
        payload = json.loads(record.response_json)
        payload["duplicate"] = True
        current = state if state is not None else self._load(session_id).state
        view, floor = self._visible(current, self._now())
        if record.turn_index < floor:
            # The recorded reply belongs to an expired or replaced verified epoch.
            payload["messages"] = []
            payload["state"] = view.public_view()
        return payload

    # ------------------------------------------------------------------ email
    def _dispatch(self, op_key: str, to_address: str, subject: str, body: str) -> DispatchOutcome:
        with tracing.span("email_dispatch", type="tool", input={"op_key": op_key[:12]}) as node:
            try:
                result = self.dispatcher.dispatch(op_key=op_key, to_address=to_address, subject=subject, body_text=body)
            except Exception as exc:  # the dispatcher should not raise; never guess that nothing was sent
                node["error"] = exc.__class__.__name__
                result = DispatchOutcome(status="delivery_unknown", receipt_id=None, detail="dispatcher_error", resent=False)
            node["output"] = {"status": result.status, "resent": result.resent}
            return result

    def _email_pending_notice(self, state: SessionState) -> ChatMessage:
        """Response-only notice while a consented send is not settled (the consent stays pending)."""
        return ChatMessage(
            role="assistant", text=_EMAIL_PENDING_TEXT, turn_index=state.turn_index, at=self._now(), phase=state.phase, kind="notice"
        )

    @staticmethod
    def _needs_email_recovery(state: SessionState) -> bool:
        return state.email.status in ("consented", "dispatching") and bool(state.email.op_key)

    def _recover_email(self, session_id: str, state: SessionState, version: int) -> tuple[SessionState, int, list[ChatMessage]]:
        """Finish a consented op after a restart. Returns (state, version, notices for the caller)."""
        if not self._needs_email_recovery(state):
            return state, version, []
        op_key = state.email.op_key or ""
        with tracing.span("email_recovery", type="task", input={"op_key": op_key[:12]}) as node:
            record = self.store.email_op_get(op_key)
            if record is None:
                # Consent and the op row are committed together, so no row means no consent was durable.
                state = state.model_copy(deep=True)
                state.email.status = "offered"
                state.email.op_key = None
                node["output"] = {"result": "no_operation_row"}
                version = self.store.commit(session_id, expected_version=version, state=state)
                return state, version, []
            if record.status in _TERMINAL_OP_STATUSES:
                result = DispatchOutcome(status=record.status, receipt_id=record.receipt_id, detail=record.detail or "", resent=False)  # type: ignore[arg-type]
            else:
                pending = self.agent.rebuild_pending_email(state)
                if pending is not None:
                    # The dispatcher owns at-most-once: a claimed (``dispatching``) op is looked up
                    # first and resent once, same op key, only when the transport can prove nothing
                    # went out; otherwise it is reported in flight or ``delivery_unknown``.
                    result = self._dispatch(pending.op_key, pending.to_address, pending.subject, pending.body_text)
                elif record.status == "pending":
                    result = DispatchOutcome(status="delivery_unknown", receipt_id=None, detail="cannot_rebuild", resent=False)
                else:
                    result = self.dispatcher.reconcile(op_key) or DispatchOutcome(
                        status="delivery_unknown", receipt_id=None, detail="unreconciled", resent=False
                    )
            if not _settled(result):
                node["output"] = {"result": "not_settled", "detail": result.detail}
                return state, version, []
            done = self.agent.complete_email(state, result)
            # Background recovery is not caller activity: it must not extend the verification idle TTL.
            done.state.last_activity_at = state.last_activity_at
            node["output"] = {"result": result.status}
            version = self.store.commit(session_id, expected_version=version, state=done.state)
            return done.state, version, list(done.messages)

    # ------------------------------------------------------------------ health
    def health(self) -> dict[str, Any]:
        checks: dict[str, Any] = {}
        status = "ok"
        try:
            self.store.load("health-probe-" + "0" * 16)
            checks["database"] = "ok"
        except SessionNotFound:
            checks["database"] = "ok"
        except Exception:
            checks["database"] = "error"
            status = "error"
        return {"status": status, "checks": checks}


def purge_old_files(directory: Path, pattern: str, retention: timedelta, *, now: datetime | None = None) -> int:
    """Delete files matching ``pattern`` in ``directory`` last modified before ``now - retention``."""
    if not directory.is_dir():
        return 0
    reference = now or datetime.now(UTC)
    cutoff = reference.timestamp() - retention.total_seconds()
    removed = 0
    for path in directory.glob(pattern):
        try:
            if path.is_file() and path.stat().st_mtime < cutoff:
                path.unlink()
                removed += 1
        except OSError:
            continue
    return removed
