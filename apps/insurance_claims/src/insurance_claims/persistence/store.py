"""Durable, encrypted SQLite session store (checkpoints, turn ledger, email ledger).

One ``sessions`` row holds the Fernet-encrypted ``SessionState`` checkpoint and an
optimistic ``version`` counter. Each accepted turn is committed in a single
``BEGIN IMMEDIATE`` transaction together with its idempotency record (``turns``)
and any email-ledger writes, so a crash never leaves a half-applied turn.

Security invariants:

* Only encrypted blobs carry PII. Plain columns hold opaque session IDs, hashes,
  statuses, party-ID references, machine-code details, and timestamps.
* Exceptions never carry decrypted content or key material (``raise ... from None``).
* The database file is ``0600`` and its directory ``0700`` (best effort).
* A delivered email op (``sent``/``queued``) is never moved back to a resendable status.
"""

from __future__ import annotations

import copy
import json
import logging
import os
import sqlite3
import tempfile
from collections.abc import Callable, Iterable, Iterator, Sequence
from contextlib import closing, contextmanager, suppress
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, get_args

from cryptography.fernet import Fernet, InvalidToken
from pydantic import ValidationError

from insurance_claims.config import ConfigError, Settings
from insurance_claims.domain.records import EmailOpRecord, EmailOpStatus, TurnRecord
from insurance_claims.domain.state import STATE_SCHEMA_VERSION, SessionState

logger = logging.getLogger(__name__)

DB_SCHEMA_VERSION = 2
"""Version of the SQL table layout (``meta.db_schema_version``), independent of the state schema.
A database written with another version is refused at startup (reset it with ``docker compose down -v``)."""

_SUBJECT_MAX_CHARS = 200

EMAIL_OP_STATUSES: frozenset[str] = frozenset(get_args(EmailOpStatus))
_DELIVERED = frozenset({"sent", "queued"})
_DETAIL_MAX_CHARS = 500
_PRIVATE_FILE = 0o600
_PRIVATE_DIR = 0o700
_DB_TIME_FORMAT = "%Y-%m-%dT%H:%M:%S.%f+00:00"
"""Fixed-width UTC ISO format, so string order equals time order (used by ``purge``)."""

CORRUPT_REASONS = ("decrypt_failed", "invalid_json", "unsupported_schema", "validation_failed")


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class SessionNotFound(Exception):
    """No session row exists for the requested ID."""


class SessionCorrupt(Exception):
    """A stored checkpoint cannot be turned back into a ``SessionState``.

    ``reason`` is one of ``CORRUPT_REASONS``. ``detail`` holds field locations and
    error types only, never stored values.
    """

    def __init__(self, reason: str, detail: str = "") -> None:
        super().__init__(reason)
        self.reason = reason
        self.detail = detail


class VersionConflict(Exception):
    """A concurrency or idempotency check rejected a write; the transaction was rolled back.

    ``reason``: ``stale_version`` | ``session_missing`` | ``session_exists`` |
    ``duplicate_turn`` | ``duplicate_email_op`` | ``busy``.
    """

    def __init__(self, reason: str = "stale_version") -> None:
        super().__init__(reason)
        self.reason = reason


# ---------------------------------------------------------------------------
# Encryption
# ---------------------------------------------------------------------------


class StateCipher:
    """Fernet (AES-CBC + HMAC-SHA256) wrapper. The key never appears in repr or errors."""

    __slots__ = ("_fernet",)

    def __init__(self, key: str | bytes) -> None:
        try:
            self._fernet = Fernet(key)
        except (TypeError, ValueError):
            raise ConfigError("State encryption key is not a valid Fernet key") from None

    def __repr__(self) -> str:
        return "StateCipher(key=<redacted>)"

    def encrypt(self, data: bytes) -> bytes:
        """Encrypt and authenticate ``data``."""
        if not isinstance(data, bytes):
            raise TypeError("StateCipher.encrypt expects bytes")
        return self._fernet.encrypt(data)

    def decrypt(self, token: bytes) -> bytes:
        """Verify and decrypt ``token``; any failure is ``SessionCorrupt("decrypt_failed")``."""
        if isinstance(token, memoryview):
            token = token.tobytes()
        if not isinstance(token, bytes):
            raise SessionCorrupt("decrypt_failed")
        try:
            return self._fernet.decrypt(token)
        except (InvalidToken, TypeError, ValueError):
            raise SessionCorrupt("decrypt_failed") from None


def load_cipher(settings: Settings) -> StateCipher:
    """Use ``STATE_ENCRYPTION_KEY`` when set; else read or create ``settings.key_file`` (0600)."""
    if settings.state_encryption_key:
        try:
            return StateCipher(settings.state_encryption_key)
        except ConfigError:
            raise ConfigError("STATE_ENCRYPTION_KEY must be a Fernet key (32 url-safe base64-encoded bytes)") from None
    key_file = settings.key_file
    _ensure_private_dir(key_file.parent)
    key = _read_key_file(key_file)
    if key is None:
        key = _create_key_file(key_file)
    try:
        return StateCipher(key)
    except ConfigError:
        raise ConfigError(f"State key file is not a valid Fernet key: {key_file}") from None


def _read_key_file(path: Path) -> bytes | None:
    """Return the stripped key bytes, ``None`` when the file does not exist."""
    try:
        data = path.read_bytes()
    except FileNotFoundError:
        return None
    except OSError:
        raise ConfigError(f"Cannot read state key file: {path}") from None
    _chmod_best_effort(path, _PRIVATE_FILE)
    key = data.strip()
    if not key:
        raise ConfigError(f"State key file is empty: {path}")
    return key


def _create_key_file(path: Path) -> bytes:
    """Create a new key file atomically; if another process wins the race, use its key."""
    with suppress(FileExistsError):
        _publish_exclusive(path, Fernet.generate_key() + b"\n")
    key = _read_key_file(path)
    if key is None:
        raise ConfigError(f"Could not create state key file: {path}")
    return key


def _publish_exclusive(path: Path, data: bytes) -> None:
    """Write ``data`` to ``path`` (0600) only if it does not exist yet; never a partial file."""
    fd, tmp_name = tempfile.mkstemp(prefix=".tmp-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(tmp_name, _PRIVATE_FILE)
        try:
            os.link(tmp_name, path)  # atomic create-if-absent with full content
        except FileExistsError:
            raise
        except OSError:  # filesystem without hard links
            _write_exclusive(path, data)
        _fsync_dir(path.parent)
    finally:
        with suppress(FileNotFoundError):
            os.unlink(tmp_name)


def _write_exclusive(path: Path, data: bytes) -> None:
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, _PRIVATE_FILE)
    with os.fdopen(fd, "wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())


def _fsync_dir(path: Path) -> None:
    with suppress(OSError):
        fd = os.open(path, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)


def _ensure_private_dir(path: Path) -> None:
    path.mkdir(mode=_PRIVATE_DIR, parents=True, exist_ok=True)
    _chmod_best_effort(path, _PRIVATE_DIR)


def _chmod_best_effort(path: Path, mode: int) -> None:
    try:
        if path.exists() and (path.stat().st_mode & 0o777) != mode:
            os.chmod(path, mode)
    except OSError:
        logger.debug("could not restrict permissions on %s", path)


# ---------------------------------------------------------------------------
# State schema migration
# ---------------------------------------------------------------------------


# Upgrade functions for older checkpoints, keyed by the version they upgrade from
# (``{3: _v3_to_v4}`` when STATE_SCHEMA_VERSION becomes 4). Versions without a path are rejected.
_MIGRATIONS: dict[int, Callable[[dict[str, Any]], dict[str, Any]]] = {}


def migrate_state(raw: dict) -> tuple[dict, int | None]:
    """Upgrade a decrypted checkpoint dict; returns ``(current_dict, migrated_from)``.

    A checkpoint with no upgrade path to ``STATE_SCHEMA_VERSION`` is ``unsupported_schema``
    (the conversation cannot be resumed and the caller starts a new one). The input is never mutated.
    """
    if not isinstance(raw, dict):
        raise SessionCorrupt("invalid_json")
    version = raw.get("schema_version")
    supported = type(version) is int and (
        version == STATE_SCHEMA_VERSION or all(v in _MIGRATIONS for v in range(version, STATE_SCHEMA_VERSION))
    )
    if not supported or version > STATE_SCHEMA_VERSION:
        raise SessionCorrupt("unsupported_schema")
    data = copy.deepcopy(raw)
    if version == STATE_SCHEMA_VERSION:
        return data, None
    original = version
    while version < STATE_SCHEMA_VERSION:
        data = _MIGRATIONS[version](data)
        version += 1
    data["schema_version"] = STATE_SCHEMA_VERSION
    return data, original


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------


@dataclass
class SessionRow:
    state: SessionState
    version: int
    secret_hash: str
    created_at: datetime
    expires_at: datetime
    migrated_from: int | None


@dataclass(frozen=True)
class SessionAuth:
    """The plain (never encrypted) columns needed to authorize a request before decrypting."""

    secret_hash: str
    expires_at: datetime


_SCHEMA_SQL: tuple[str, ...] = (
    """CREATE TABLE IF NOT EXISTS meta (
        key TEXT PRIMARY KEY,
        value TEXT NOT NULL
    )""",
    """CREATE TABLE IF NOT EXISTS sessions (
        session_id TEXT PRIMARY KEY,
        secret_hash TEXT NOT NULL,
        version INTEGER NOT NULL CHECK (version >= 1),
        schema_version INTEGER NOT NULL,
        state_blob BLOB NOT NULL,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        expires_at TEXT NOT NULL
    )""",
    "CREATE INDEX IF NOT EXISTS idx_sessions_updated_at ON sessions(updated_at)",
    """CREATE TABLE IF NOT EXISTS turns (
        session_id TEXT NOT NULL,
        client_turn_id TEXT NOT NULL,
        turn_index INTEGER NOT NULL,
        request_hash TEXT NOT NULL,
        response_blob BLOB NOT NULL,
        created_at TEXT NOT NULL,
        PRIMARY KEY (session_id, client_turn_id)
    )""",
    "CREATE INDEX IF NOT EXISTS idx_turns_created_at ON turns(created_at)",
    f"""CREATE TABLE IF NOT EXISTS email_operations (
        op_key TEXT PRIMARY KEY,
        session_id TEXT NOT NULL,
        recipient_ref TEXT NOT NULL,
        draft_hash TEXT NOT NULL,
        status TEXT NOT NULL CHECK (status IN ({", ".join(f"'{s}'" for s in sorted(EMAIL_OP_STATUSES))})),
        receipt_id TEXT,
        detail TEXT,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
    )""",
    "CREATE INDEX IF NOT EXISTS idx_email_ops_session ON email_operations(session_id)",
    "CREATE INDEX IF NOT EXISTS idx_email_ops_updated_at ON email_operations(updated_at)",
    # Cross-session verification failures. ``subject`` is an opaque reference such as
    # ``party:P9`` (never a raw identity value).
    """CREATE TABLE IF NOT EXISTS verification_failures (
        subject TEXT NOT NULL,
        failed_at TEXT NOT NULL
    )""",
    "CREATE INDEX IF NOT EXISTS idx_vfail_subject ON verification_failures(subject, failed_at)",
)

_EMAIL_COLUMNS = "op_key, session_id, recipient_ref, draft_hash, status, receipt_id, detail, created_at, updated_at"


class SessionStore:
    """SQLite checkpoint store. Thread-safe: every operation opens its own connection."""

    def __init__(
        self,
        db_path: Path,
        cipher: StateCipher,
        *,
        clock: Callable[[], datetime],
        busy_timeout_s: float = 5.0,
    ) -> None:
        self._db_path = Path(db_path)
        self._uri = f"{self._db_path.resolve().as_uri()}?mode=rw"
        self._cipher = cipher
        self._clock = clock
        self._busy_timeout_s = busy_timeout_s

    @property
    def db_path(self) -> Path:
        return self._db_path

    # -- schema -------------------------------------------------------------

    def init_schema(self) -> None:
        """Create the database (file 0600, directory 0700), enable WAL, and create tables. Idempotent."""
        _ensure_private_dir(self._db_path.parent)
        if not self._db_path.exists():
            with suppress(FileExistsError):
                os.close(os.open(self._db_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, _PRIVATE_FILE))
        self._tighten_files()
        with closing(self._connect()) as conn:
            conn.execute("PRAGMA journal_mode=WAL")
        with self._write() as conn:
            for statement in _SCHEMA_SQL:
                conn.execute(statement)
            row = conn.execute("SELECT value FROM meta WHERE key = 'db_schema_version'").fetchone()
            if row is None:
                conn.execute("INSERT INTO meta (key, value) VALUES ('db_schema_version', ?)", (str(DB_SCHEMA_VERSION),))
            elif str(row[0]) != str(DB_SCHEMA_VERSION):
                raise ConfigError(
                    f"Session database schema version {row[0]!r} is not supported by this build (supports {DB_SCHEMA_VERSION})"
                )
        self._tighten_files()

    # -- sessions -----------------------------------------------------------

    def create(self, state: SessionState, *, secret_hash: str, expires_at: datetime) -> int:
        """Insert a new session at version 1. An existing ID raises ``VersionConflict('session_exists')``."""
        if not state.session_id:
            raise ValueError("state.session_id is required")
        if not isinstance(secret_hash, str) or not secret_hash:
            raise ValueError("secret_hash is required")
        blob = self._seal_state(state)
        now = _to_db_time(self._clock())
        try:
            with self._write() as conn:
                conn.execute(
                    "INSERT INTO sessions (session_id, secret_hash, version, schema_version, state_blob,"
                    " created_at, updated_at, expires_at) VALUES (?, ?, 1, ?, ?, ?, ?, ?)",
                    (
                        state.session_id,
                        secret_hash,
                        STATE_SCHEMA_VERSION,
                        blob,
                        now,
                        now,
                        _to_db_time(expires_at),
                    ),
                )
        except sqlite3.IntegrityError:
            raise VersionConflict("session_exists") from None
        return 1

    def load_auth(self, session_id: str) -> SessionAuth:
        """Read only ``secret_hash`` and ``expires_at`` (no decryption), for authorizing first."""
        with closing(self._connect()) as conn:
            row = conn.execute("SELECT secret_hash, expires_at FROM sessions WHERE session_id = ?", (session_id,)).fetchone()
        if row is None:
            raise SessionNotFound("session not found")
        return SessionAuth(secret_hash=row[0], expires_at=_row_time(row[1]))

    def load(self, session_id: str) -> SessionRow:
        """Decrypt, migrate, and validate the latest checkpoint."""
        with closing(self._connect()) as conn:
            row = conn.execute(
                "SELECT state_blob, version, secret_hash, created_at, expires_at FROM sessions WHERE session_id = ?",
                (session_id,),
            ).fetchone()
        if row is None:
            raise SessionNotFound("session not found")
        blob, version, secret_hash, created_at, expires_at = row
        state, migrated_from = self._open_state(blob, session_id)
        return SessionRow(
            state=state,
            version=version,
            secret_hash=secret_hash,
            created_at=_row_time(created_at),
            expires_at=_row_time(expires_at),
            migrated_from=migrated_from,
        )

    def commit(
        self,
        session_id: str,
        *,
        expected_version: int,
        state: SessionState,
        turn: TurnRecord | None = None,
        email_ops_create: Sequence[EmailOpRecord] = (),
        email_ops_update: Sequence[tuple[str, str, str | None, str | None]] = (),
    ) -> int:
        """Atomically write the checkpoint, turn record, and email ops; return the new version.

        Stale ``expected_version``, a missing session, a duplicate ``client_turn_id``, or a
        duplicate ``op_key`` roll everything back and raise ``VersionConflict``.
        """
        if type(expected_version) is not int or expected_version < 1:
            raise ValueError("expected_version must be a positive int")
        if state.session_id != session_id:
            raise ValueError("state belongs to a different session")
        if turn is not None:
            _check_turn(turn, session_id)
        for op in email_ops_create:
            _check_email_op(op, session_id)
        updates = [_check_update_tuple(item) for item in email_ops_update]
        state_blob = self._seal_state(state)
        turn_blob = self._cipher.encrypt(turn.response_json.encode("utf-8")) if turn is not None else None
        now = _to_db_time(self._clock())
        try:
            with self._write() as conn:
                self._bump_version(conn, session_id, expected_version, state_blob, now)
                if turn is not None and turn_blob is not None:
                    _insert_turn(conn, turn, turn_blob)
                for op in email_ops_create:
                    _insert_email_op(conn, op)
                for op_key, status, receipt_id, detail in updates:
                    _update_email_op_in_session(conn, session_id, op_key, status, receipt_id, detail, now)
        except sqlite3.OperationalError as exc:
            if _is_busy(exc):
                raise VersionConflict("busy") from None
            raise
        return expected_version + 1

    def delete(self, session_id: str) -> bool:
        """Delete the session, its turns, and its email ops. True when the session existed."""
        with self._write() as conn:
            conn.execute("DELETE FROM turns WHERE session_id = ?", (session_id,))
            conn.execute("DELETE FROM email_operations WHERE session_id = ?", (session_id,))
            removed = conn.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,)).rowcount
        return removed > 0

    def purge(self, *, retention: timedelta, failure_retention: timedelta | None = None) -> int:
        """Delete sessions idle longer than ``retention`` plus old or orphaned turns/ops.

        Verification failures are kept for ``max(retention, failure_retention)`` so a
        lockout window longer than the retention period is never cut short.
        Returns the number of sessions removed.
        """
        if retention < timedelta(0):
            raise ValueError("retention must not be negative")
        if failure_retention is not None and failure_retention < timedelta(0):
            raise ValueError("failure_retention must not be negative")
        now = self._clock()
        cutoff = _to_db_time(now - retention)
        failure_cutoff = _to_db_time(now - max(retention, failure_retention or timedelta(0)))
        with self._write() as conn:
            conn.execute("DELETE FROM verification_failures WHERE failed_at < ?", (failure_cutoff,))
            removed = conn.execute("DELETE FROM sessions WHERE updated_at < ?", (cutoff,)).rowcount
            conn.execute(
                "DELETE FROM turns WHERE created_at < ? OR session_id NOT IN (SELECT session_id FROM sessions)",
                (cutoff,),
            )
            conn.execute(
                "DELETE FROM email_operations WHERE updated_at < ? OR session_id NOT IN (SELECT session_id FROM sessions)",
                (cutoff,),
            )
        return removed

    # -- cross-session verification failures --------------------------------

    def record_verification_failure(self, subjects: Iterable[str], at: datetime) -> None:
        """Record one failed attempt for each distinct subject (opaque refs like ``party:P9``)."""
        unique = sorted({_check_subject(subject) for subject in subjects})
        if not unique:
            return
        stamp = _to_db_time(at)
        with self._write() as conn:
            conn.executemany(
                "INSERT INTO verification_failures (subject, failed_at) VALUES (?, ?)",
                [(subject, stamp) for subject in unique],
            )

    def count_verification_failures(self, subject: str, since: datetime) -> int:
        """Failed attempts recorded for ``subject`` at or after ``since``."""
        with closing(self._connect()) as conn:
            row = conn.execute(
                "SELECT COUNT(*) FROM verification_failures WHERE subject = ? AND failed_at >= ?",
                (_check_subject(subject), _to_db_time(since)),
            ).fetchone()
        return int(row[0])

    # -- turn ledger --------------------------------------------------------

    def get_turn(self, session_id: str, client_turn_id: str) -> TurnRecord | None:
        """Return the recorded turn (response decrypted) for duplicate-submit replay."""
        with closing(self._connect()) as conn:
            row = conn.execute(
                "SELECT turn_index, request_hash, response_blob, created_at FROM turns WHERE session_id = ? AND client_turn_id = ?",
                (session_id, client_turn_id),
            ).fetchone()
        if row is None:
            return None
        turn_index, request_hash, response_blob, created_at = row
        try:
            response_json = self._cipher.decrypt(response_blob).decode("utf-8")
        except UnicodeDecodeError:
            raise SessionCorrupt("invalid_json") from None
        return TurnRecord(
            session_id=session_id,
            client_turn_id=client_turn_id,
            turn_index=turn_index,
            request_hash=request_hash,
            response_json=response_json,
            created_at=_row_time(created_at),
        )

    # -- email ledger -------------------------------------------------------

    def email_op_get(self, op_key: str) -> EmailOpRecord | None:
        with closing(self._connect()) as conn:
            row = conn.execute(f"SELECT {_EMAIL_COLUMNS} FROM email_operations WHERE op_key = ?", (op_key,)).fetchone()
        return _email_op_from_row(row) if row is not None else None

    def email_op_update(
        self,
        op_key: str,
        *,
        status: str,
        receipt_id: str | None = None,
        detail: str | None = None,
        expected_status: str | None = None,
    ) -> bool:
        """Update an op's status; compare-and-set when ``expected_status`` is given.

        ``None`` for ``receipt_id``/``detail`` keeps the stored value. Returns False when the
        op is missing, the CAS fails, or the transition would un-deliver or re-arm the op.
        """
        _check_status(status)
        if expected_status is not None:
            _check_status(expected_status)
        with self._write() as conn:
            row = conn.execute("SELECT status FROM email_operations WHERE op_key = ?", (op_key,)).fetchone()
            if row is None:
                return False
            current = row[0]
            if expected_status is not None and current != expected_status:
                return False
            if not _transition_allowed(current, status):
                return False
            _write_email_status(conn, op_key, status, receipt_id, detail, _to_db_time(self._clock()))
            return True

    def email_ops_abandoned(self, *, now: datetime, idle_before: datetime) -> list[EmailOpRecord]:
        """Unfinished ops (``pending``/``dispatching``) whose session is expired or gone.

        No one can resume these through the API any more. ``idle_before`` skips ops
        touched recently, so an op owned by a turn that is still running is left alone.
        """
        columns = ", ".join(f"e.{name.strip()}" for name in _EMAIL_COLUMNS.split(","))
        with closing(self._connect()) as conn:
            rows = conn.execute(
                f"SELECT {columns} FROM email_operations e"
                " LEFT JOIN sessions s ON s.session_id = e.session_id"
                " WHERE e.status IN ('pending', 'dispatching') AND e.updated_at < ?"
                " AND (s.session_id IS NULL OR s.expires_at < ?)"
                " ORDER BY e.created_at, e.op_key",
                (_to_db_time(idle_before), _to_db_time(now)),
            ).fetchall()
        return [_email_op_from_row(row) for row in rows]

    # -- internals ----------------------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        """Open the existing database (never creates it, so it is never created world-readable)."""
        conn = sqlite3.connect(self._uri, uri=True, timeout=self._busy_timeout_s, isolation_level=None)
        try:
            conn.execute("PRAGMA synchronous = FULL")
            conn.execute("PRAGMA secure_delete = ON")
        except BaseException:
            conn.close()
            raise
        return conn

    @contextmanager
    def _write(self) -> Iterator[sqlite3.Connection]:
        """One atomic unit holding the write lock (``BEGIN IMMEDIATE``); rolls back on any error."""
        with closing(self._connect()) as conn:
            conn.execute("BEGIN IMMEDIATE")
            try:
                yield conn
            except BaseException:
                if conn.in_transaction:
                    conn.execute("ROLLBACK")
                raise
            conn.execute("COMMIT")

    def _tighten_files(self) -> None:
        for suffix in ("", "-wal", "-shm"):
            _chmod_best_effort(Path(f"{self._db_path}{suffix}"), _PRIVATE_FILE)

    def _seal_state(self, state: SessionState) -> bytes:
        if not isinstance(state, SessionState):
            raise TypeError("state must be a SessionState")
        payload = state.model_dump(mode="json")
        payload["schema_version"] = STATE_SCHEMA_VERSION  # the model is the current shape by construction
        data = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        return self._cipher.encrypt(data)

    def _open_state(self, blob: bytes, session_id: str) -> tuple[SessionState, int | None]:
        plaintext = self._cipher.decrypt(blob)
        try:
            raw = json.loads(plaintext)
        except (ValueError, RecursionError):
            raise SessionCorrupt("invalid_json") from None
        if not isinstance(raw, dict):
            raise SessionCorrupt("invalid_json")
        data, migrated_from = migrate_state(raw)
        try:
            state = SessionState.model_validate(data)
        except ValidationError as exc:
            raise SessionCorrupt("validation_failed", _safe_errors(exc)) from None
        if state.session_id != session_id:
            raise SessionCorrupt("validation_failed", "session_id does not match the row")
        return state, migrated_from

    @staticmethod
    def _bump_version(conn: sqlite3.Connection, session_id: str, expected_version: int, blob: bytes, now: str) -> None:
        cursor = conn.execute(
            "UPDATE sessions SET state_blob = ?, schema_version = ?, version = version + 1, updated_at = ?"
            " WHERE session_id = ? AND version = ?",
            (blob, STATE_SCHEMA_VERSION, now, session_id, expected_version),
        )
        if cursor.rowcount == 1:
            return
        exists = conn.execute("SELECT 1 FROM sessions WHERE session_id = ?", (session_id,)).fetchone()
        raise VersionConflict("stale_version" if exists else "session_missing")


# ---------------------------------------------------------------------------
# Row helpers
# ---------------------------------------------------------------------------


def _utc(value: datetime) -> datetime:
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)


def _to_db_time(value: datetime) -> str:
    if not isinstance(value, datetime):
        raise TypeError("expected a datetime")
    return _utc(value).strftime(_DB_TIME_FORMAT)


def _row_time(raw: str) -> datetime:
    try:
        return _utc(datetime.fromisoformat(raw))
    except (TypeError, ValueError):
        raise SessionCorrupt("validation_failed", "bad row timestamp") from None


def _safe_errors(exc: ValidationError, limit: int = 5) -> str:
    """Error locations and types only; ``include_input=False`` keeps stored values out."""
    parts = [
        f"{'.'.join(str(p) for p in err['loc'])}:{err['type']}"
        for err in exc.errors(include_input=False, include_url=False, include_context=False)[:limit]
    ]
    return "; ".join(parts)


def _is_busy(exc: sqlite3.OperationalError) -> bool:
    code = getattr(exc, "sqlite_errorcode", None)
    if isinstance(code, int):
        return (code & 0xFF) in (sqlite3.SQLITE_BUSY, sqlite3.SQLITE_LOCKED)
    message = str(exc).lower()
    return "locked" in message or "busy" in message


def _nonempty(value: object) -> bool:
    return isinstance(value, str) and bool(value)


def _check_turn(turn: TurnRecord, session_id: str) -> None:
    if turn.session_id != session_id:
        raise ValueError("turn belongs to a different session")
    if not _nonempty(turn.client_turn_id) or not isinstance(turn.request_hash, str):
        raise ValueError("turn record is incomplete")
    if not isinstance(turn.response_json, str) or type(turn.turn_index) is not int:
        raise ValueError("turn record has invalid types")


def _check_email_op(op: EmailOpRecord, session_id: str) -> None:
    if op.session_id != session_id:
        raise ValueError("email op belongs to a different session")
    if not all(_nonempty(v) for v in (op.op_key, op.recipient_ref, op.draft_hash)):
        raise ValueError("email op record is incomplete")
    _check_status(op.status)


def _check_update_tuple(item: tuple[str, str, str | None, str | None]) -> tuple[str, str, str | None, str | None]:
    if len(item) != 4:
        raise ValueError("email op update must be (op_key, status, receipt_id, detail)")
    op_key, status, receipt_id, detail = item
    if not _nonempty(op_key):
        raise ValueError("email op update needs an op_key")
    _check_status(status)
    return op_key, status, receipt_id, detail


def _check_subject(subject: object) -> str:
    if not isinstance(subject, str) or not subject.strip() or len(subject) > _SUBJECT_MAX_CHARS:
        raise ValueError("verification failure subject must be a short non-empty string")
    return subject


def _check_status(status: str) -> None:
    if status not in EMAIL_OP_STATUSES:
        raise ValueError(f"unknown email op status {status!r}")


def _transition_allowed(current: str, new: str) -> bool:
    """Same status is a no-op update; never re-arm to ``pending``; never un-deliver."""
    if current == new:
        return True
    if new == "pending":
        return False
    if current in _DELIVERED:
        return new in _DELIVERED
    return True


def _clip(detail: str | None) -> str | None:
    return None if detail is None else str(detail)[:_DETAIL_MAX_CHARS]


def _insert_turn(conn: sqlite3.Connection, turn: TurnRecord, blob: bytes) -> None:
    try:
        conn.execute(
            "INSERT INTO turns (session_id, client_turn_id, turn_index, request_hash, response_blob, created_at) VALUES (?, ?, ?, ?, ?, ?)",
            (
                turn.session_id,
                turn.client_turn_id,
                turn.turn_index,
                turn.request_hash,
                blob,
                _to_db_time(turn.created_at),
            ),
        )
    except sqlite3.IntegrityError:
        raise VersionConflict("duplicate_turn") from None


def _insert_email_op(conn: sqlite3.Connection, op: EmailOpRecord) -> None:
    try:
        conn.execute(
            f"INSERT INTO email_operations ({_EMAIL_COLUMNS}) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                op.op_key,
                op.session_id,
                op.recipient_ref,
                op.draft_hash,
                op.status,
                op.receipt_id,
                _clip(op.detail),
                _to_db_time(op.created_at),
                _to_db_time(op.updated_at),
            ),
        )
    except sqlite3.IntegrityError:
        raise VersionConflict("duplicate_email_op") from None


def _update_email_op_in_session(
    conn: sqlite3.Connection,
    session_id: str,
    op_key: str,
    status: str,
    receipt_id: str | None,
    detail: str | None,
    now: str,
) -> None:
    row = conn.execute("SELECT status FROM email_operations WHERE op_key = ? AND session_id = ?", (op_key, session_id)).fetchone()
    if row is None:
        raise ValueError("email op update targets no operation of this session")
    if not _transition_allowed(row[0], status):
        raise ValueError(f"email op transition {row[0]} -> {status} is not allowed")
    _write_email_status(conn, op_key, status, receipt_id, detail, now)


def _write_email_status(conn: sqlite3.Connection, op_key: str, status: str, receipt_id: str | None, detail: str | None, now: str) -> None:
    conn.execute(
        "UPDATE email_operations SET status = ?, receipt_id = COALESCE(?, receipt_id),"
        " detail = COALESCE(?, detail), updated_at = ? WHERE op_key = ?",
        (status, receipt_id, _clip(detail), now, op_key),
    )


def _email_op_from_row(row: Sequence[Any]) -> EmailOpRecord:
    op_key, session_id, recipient_ref, draft_hash, status, receipt_id, detail, created_at, updated_at = row
    return EmailOpRecord(
        op_key=op_key,
        session_id=session_id,
        recipient_ref=recipient_ref,
        draft_hash=draft_hash,
        status=status,
        created_at=_row_time(created_at),
        updated_at=_row_time(updated_at),
        receipt_id=receipt_id,
        detail=detail,
    )


__all__ = [
    "CORRUPT_REASONS",
    "DB_SCHEMA_VERSION",
    "SessionAuth",
    "SessionCorrupt",
    "SessionNotFound",
    "SessionRow",
    "SessionStore",
    "StateCipher",
    "VersionConflict",
    "load_cipher",
    "migrate_state",
]
