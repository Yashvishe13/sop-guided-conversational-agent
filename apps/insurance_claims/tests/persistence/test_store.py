"""Tests for the durable, encrypted SQLite session store."""

from __future__ import annotations

import json
import os
import sqlite3
import stat
import threading
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path

import pytest
from cryptography.fernet import Fernet
from pydantic import ValidationError

from insurance_claims.config import ConfigError, Settings
from insurance_claims.domain.models import Phase
from insurance_claims.domain.records import EmailOpRecord, TurnRecord
from insurance_claims.domain.state import (
    STATE_SCHEMA_VERSION,
    ChatMessage,
    SessionState,
)
from insurance_claims.persistence.store import (
    DB_SCHEMA_VERSION,
    SessionCorrupt,
    SessionNotFound,
    SessionStore,
    StateCipher,
    VersionConflict,
    load_cipher,
    migrate_state,
)

SEED_NAME = "Zebulon Quixotewright"
SEED_EMAIL = "zebulon.quixote@example.org"
SEED_PHONE = "6505550199"
T0 = datetime(2026, 10, 3, 12, 0, tzinfo=UTC)
EXPIRES = T0 + timedelta(hours=24)


class FakeClock:
    """Injectable, manually advanced UTC clock."""

    def __init__(self, now: datetime = T0) -> None:
        self.now = now

    def __call__(self) -> datetime:
        return self.now

    def advance(self, **delta: float) -> None:
        self.now += timedelta(**delta)


# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


@pytest.fixture
def key() -> bytes:
    return Fernet.generate_key()


@pytest.fixture
def cipher(key: bytes) -> StateCipher:
    return StateCipher(key)


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    return tmp_path / "data" / "sessions.sqlite3"


@pytest.fixture
def store(db_path: Path, cipher: StateCipher, clock: FakeClock) -> SessionStore:
    instance = SessionStore(db_path, cipher, clock=clock)
    instance.init_schema()
    return instance


def make_state(session_id: str = "sess-1", *, with_pii: bool = True) -> SessionState:
    history = [ChatMessage(role="assistant", text="Hello, how can I help today?", turn_index=0, at=T0, phase=Phase.VERIFY_ID)]
    if with_pii:
        history.append(
            ChatMessage(
                role="user",
                text=f"I am {SEED_NAME}, my email is {SEED_EMAIL} and my phone is {SEED_PHONE}",
                turn_index=1,
                at=T0 + timedelta(seconds=5),
                phase=Phase.VERIFY_ID,
            )
        )
    state = SessionState(
        session_id=session_id,
        created_at=T0,
        last_activity_at=T0 + timedelta(seconds=5),
        turn_index=len(history) - 1,
        history=history,
    )
    state.hints.question_summaries = ["why the January claim was denied"]
    state.hints.case_type = "healthcare"
    state.hints.month = 1
    state.hints.year = 2026
    state.hints.first_turn = 1
    state.hints.captured_in_phase = Phase.VERIFY_ID
    state.verification.field_matches = {"full_name": ["P9"], "email": ["P9"]}
    state.counters.off_topic_total = 1
    state.counters.refusals = 1
    return state


def create(store: SessionStore, session_id: str = "sess-1", **kwargs: object) -> SessionState:
    state = make_state(session_id, **kwargs)  # type: ignore[arg-type]
    assert store.create(state, secret_hash=f"hash-{session_id}", expires_at=EXPIRES) == 1
    return state


def evolve(state: SessionState, **changes: object) -> SessionState:
    updated = state.model_copy(deep=True)
    for name, value in changes.items():
        setattr(updated, name, value)
    return updated


def make_turn(session_id: str, client_turn_id: str = "turn-1", *, text: str = "ok", at: datetime = T0) -> TurnRecord:
    response = json.dumps({"turn_index": 2, "messages": [{"role": "assistant", "text": text}]})
    return TurnRecord(
        session_id=session_id,
        client_turn_id=client_turn_id,
        turn_index=2,
        request_hash="req-" + client_turn_id,
        response_json=response,
        created_at=at,
    )


def make_op(session_id: str, op_key: str = "op-1", *, status: str = "pending", at: datetime = T0) -> EmailOpRecord:
    return EmailOpRecord(
        op_key=op_key,
        session_id=session_id,
        recipient_ref="P9",
        draft_hash="d" * 16,
        status=status,  # type: ignore[arg-type]
        created_at=at,
        updated_at=at,
    )


def add_ops(store: SessionStore, state: SessionState, *ops: EmailOpRecord) -> int:
    """Commit email ops on a freshly created session (expected version 1)."""
    return store.commit(state.session_id, expected_version=1, state=state, email_ops_create=list(ops))


def raw_connect(db_path: Path) -> sqlite3.Connection:
    return sqlite3.connect(db_path, isolation_level=None)


def insert_raw_session(db_path: Path, session_id: str, blob: bytes, *, schema_version: int = STATE_SCHEMA_VERSION) -> None:
    stamp = T0.strftime("%Y-%m-%dT%H:%M:%S.%f+00:00")
    with raw_connect(db_path) as conn:
        conn.execute(
            "INSERT INTO sessions (session_id, secret_hash, version, schema_version, state_blob,"
            " created_at, updated_at, expires_at) VALUES (?, 'h', 1, ?, ?, ?, ?, ?)",
            (session_id, schema_version, blob, stamp, stamp, stamp),
        )
    conn.close()


def count_rows(db_path: Path, table: str, session_id: str) -> int:
    conn = raw_connect(db_path)
    try:
        return conn.execute(f"SELECT count(*) FROM {table} WHERE session_id = ?", (session_id,)).fetchone()[0]
    finally:
        conn.close()


def file_mode(path: Path) -> int:
    return stat.S_IMODE(path.stat().st_mode)


def seal(cipher: StateCipher, payload: object) -> bytes:
    return cipher.encrypt(json.dumps(payload).encode("utf-8"))


# ---------------------------------------------------------------------------
# Round trip and restart
# ---------------------------------------------------------------------------


def test_create_load_round_trip_preserves_state_with_history(store: SessionStore, clock: FakeClock) -> None:
    state = create(store, "sess-rt")

    row = store.load("sess-rt")

    assert row.state == state
    assert [m.text for m in row.state.history] == [m.text for m in state.history]
    assert row.state.hints.question_summaries == ["why the January claim was denied"]
    assert row.version == 1
    assert row.secret_hash == "hash-sess-rt"
    assert row.created_at == clock.now
    assert row.expires_at == EXPIRES
    assert row.created_at.tzinfo is not None and row.expires_at.utcoffset() == timedelta(0)
    assert row.migrated_from is None


def test_state_survives_a_new_store_instance(db_path: Path, key: bytes, clock: FakeClock) -> None:
    first = SessionStore(db_path, StateCipher(key), clock=clock)
    first.init_schema()
    create(first, "sess-restart")

    restarted = SessionStore(db_path, StateCipher(key), clock=clock)
    restarted.init_schema()
    row = restarted.load("sess-restart")

    assert row.state.hints.case_type == "healthcare"
    assert row.state.hints.month == 1
    assert row.state.phase is Phase.VERIFY_ID


def test_load_unknown_session_raises_not_found(store: SessionStore) -> None:
    with pytest.raises(SessionNotFound):
        store.load("missing")


def test_create_duplicate_session_id_conflicts_and_keeps_original(store: SessionStore) -> None:
    create(store, "sess-dup")
    other = evolve(make_state("sess-dup"), turn_index=99)

    with pytest.raises(VersionConflict) as excinfo:
        store.create(other, secret_hash="other-hash", expires_at=EXPIRES)

    assert excinfo.value.reason == "session_exists"
    row = store.load("sess-dup")
    assert row.secret_hash == "hash-sess-dup"
    assert row.state.turn_index != 99


def test_create_rejects_empty_secret_hash(store: SessionStore) -> None:
    with pytest.raises(ValueError):
        store.create(make_state("sess-x"), secret_hash="", expires_at=EXPIRES)


def test_naive_and_offset_datetimes_are_stored_as_utc(db_path: Path, cipher: StateCipher) -> None:
    naive_clock = FakeClock(datetime(2026, 10, 3, 12, 0))  # naive -> treated as UTC
    store = SessionStore(db_path, cipher, clock=naive_clock)
    store.init_schema()
    eastern = timezone(timedelta(hours=-5))
    expires = datetime(2026, 10, 4, 7, 0, tzinfo=eastern)
    store.create(make_state("sess-tz"), secret_hash="h", expires_at=expires)

    row = store.load("sess-tz")

    assert row.created_at == T0
    assert row.expires_at == expires
    assert row.expires_at.utcoffset() == timedelta(0)


# ---------------------------------------------------------------------------
# Optimistic versioning and atomic commit
# ---------------------------------------------------------------------------


def test_commit_increments_version_and_persists(store: SessionStore, clock: FakeClock) -> None:
    state = create(store, "sess-c")
    clock.advance(minutes=1)

    v2 = store.commit("sess-c", expected_version=1, state=evolve(state, phase=Phase.RESOLVE_INTENT))
    v3 = store.commit("sess-c", expected_version=2, state=evolve(state, phase=Phase.PROCESS_CASE))

    assert (v2, v3) == (2, 3)
    row = store.load("sess-c")
    assert row.version == 3
    assert row.state.phase is Phase.PROCESS_CASE


def test_stale_expected_version_raises_and_keeps_newer_state(store: SessionStore) -> None:
    state = create(store, "sess-stale")
    store.commit("sess-stale", expected_version=1, state=evolve(state, phase=Phase.RESOLVE_INTENT))

    with pytest.raises(VersionConflict) as excinfo:
        store.commit("sess-stale", expected_version=1, state=evolve(state, phase=Phase.POST_PROCESS))

    assert excinfo.value.reason == "stale_version"
    row = store.load("sess-stale")
    assert row.version == 2
    assert row.state.phase is Phase.RESOLVE_INTENT


def test_commit_to_missing_session_conflicts(store: SessionStore) -> None:
    with pytest.raises(VersionConflict) as excinfo:
        store.commit("ghost", expected_version=1, state=make_state("ghost"))
    assert excinfo.value.reason == "session_missing"


@pytest.mark.parametrize("bad_version", [0, -1, True, "1", 1.0])
def test_commit_rejects_invalid_expected_version(store: SessionStore, bad_version: object) -> None:
    state = create(store, "sess-v")
    with pytest.raises(ValueError):
        store.commit("sess-v", expected_version=bad_version, state=state)  # type: ignore[arg-type]


def test_commit_rejects_state_of_another_session(store: SessionStore) -> None:
    create(store, "sess-a")
    create(store, "sess-b")

    with pytest.raises(ValueError):
        store.commit("sess-a", expected_version=1, state=make_state("sess-b"))

    assert store.load("sess-a").version == 1
    assert store.load("sess-b").version == 1


def test_two_threads_same_version_exactly_one_succeeds(store: SessionStore) -> None:
    for round_number in range(5):
        session_id = f"sess-race-{round_number}"
        state = create(store, session_id)
        barrier = threading.Barrier(2)
        results: list[tuple[str, object, str]] = []
        guard = threading.Lock()

        def worker(label: str, base: SessionState = state, sid: str = session_id) -> None:
            candidate = evolve(base, hints=base.hints.model_copy(update={"question_summaries": [label]}))
            barrier.wait()
            outcome: tuple[str, object, str]
            try:
                outcome = ("ok", store.commit(sid, expected_version=1, state=candidate), label)
            except VersionConflict as exc:
                outcome = ("conflict", exc.reason, label)
            with guard:
                results.append(outcome)

        threads = [threading.Thread(target=worker, args=(f"writer-{i}",)) for i in range(2)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=30)

        winners = [r for r in results if r[0] == "ok"]
        losers = [r for r in results if r[0] == "conflict"]
        assert len(winners) == 1 and winners[0][1] == 2
        assert len(losers) == 1 and losers[0][1] == "stale_version"
        row = store.load(session_id)
        assert row.version == 2
        assert row.state.hints.question_summaries == [winners[0][2]]


def test_many_writers_with_retry_lose_no_updates(store: SessionStore) -> None:
    state = create(store, "sess-many")
    writers = 8
    barrier = threading.Barrier(writers)
    failures: list[str] = []

    def worker(label: str) -> None:
        barrier.wait()
        for _ in range(500):
            row = store.load("sess-many")
            updated = evolve(
                row.state, hints=row.state.hints.model_copy(update={"question_summaries": [*row.state.hints.question_summaries, label]})
            )
            try:
                store.commit("sess-many", expected_version=row.version, state=updated)
                return
            except VersionConflict:
                continue
        failures.append(label)

    threads = [threading.Thread(target=worker, args=(f"topic-{i}",)) for i in range(writers)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=60)

    assert failures == []
    row = store.load("sess-many")
    assert row.version == 1 + writers
    assert sorted(q for q in row.state.hints.question_summaries if q.startswith("topic-")) == sorted(f"topic-{i}" for i in range(writers))
    assert row.state.history == state.history


def test_lock_timeout_is_reported_as_busy_conflict(db_path: Path, cipher: StateCipher, clock: FakeClock) -> None:
    store = SessionStore(db_path, cipher, clock=clock, busy_timeout_s=0.05)
    store.init_schema()
    state = create(store, "sess-busy")
    blocker = raw_connect(db_path)
    blocker.execute("BEGIN IMMEDIATE")
    try:
        with pytest.raises(VersionConflict) as excinfo:
            store.commit("sess-busy", expected_version=1, state=evolve(state, phase=Phase.RESOLVE_INTENT))
        assert excinfo.value.reason == "busy"
    finally:
        blocker.execute("ROLLBACK")
        blocker.close()
    assert store.load("sess-busy").version == 1


# ---------------------------------------------------------------------------
# Turn ledger
# ---------------------------------------------------------------------------


def test_get_turn_returns_decrypted_response(store: SessionStore) -> None:
    state = create(store, "sess-t")
    turn = make_turn("sess-t", "turn-abc", text="Your claim CL-2048 was reviewed.")
    store.commit("sess-t", expected_version=1, state=state, turn=turn)

    loaded = store.get_turn("sess-t", "turn-abc")

    assert loaded == turn
    assert json.loads(loaded.response_json)["messages"][0]["text"] == "Your claim CL-2048 was reviewed."


def test_get_turn_unknown_or_other_session_returns_none(store: SessionStore) -> None:
    state = create(store, "sess-t1")
    create(store, "sess-t2")
    store.commit("sess-t1", expected_version=1, state=state, turn=make_turn("sess-t1", "turn-1"))

    assert store.get_turn("sess-t1", "nope") is None
    assert store.get_turn("sess-t2", "turn-1") is None


def test_duplicate_client_turn_id_conflicts_and_rolls_back_state(store: SessionStore) -> None:
    state = create(store, "sess-dt")
    first = make_turn("sess-dt", "turn-1", text="first answer")
    store.commit("sess-dt", expected_version=1, state=evolve(state, turn_index=2), turn=first)

    replay = make_turn("sess-dt", "turn-1", text="second answer")
    with pytest.raises(VersionConflict) as excinfo:
        store.commit(
            "sess-dt",
            expected_version=2,
            state=evolve(state, turn_index=3, phase=Phase.POST_PROCESS),
            turn=replay,
            email_ops_create=[make_op("sess-dt", "op-dt")],
        )

    assert excinfo.value.reason == "duplicate_turn"
    row = store.load("sess-dt")
    assert row.version == 2
    assert row.state.turn_index == 2
    assert row.state.phase is Phase.VERIFY_ID
    assert store.get_turn("sess-dt", "turn-1") == first
    assert store.email_op_get("op-dt") is None


def test_commit_rejects_turn_for_another_session(store: SessionStore) -> None:
    state = create(store, "sess-own")
    with pytest.raises(ValueError):
        store.commit("sess-own", expected_version=1, state=state, turn=make_turn("sess-other"))
    assert store.load("sess-own").version == 1
    assert store.get_turn("sess-other", "turn-1") is None


# ---------------------------------------------------------------------------
# Encryption at rest and file permissions
# ---------------------------------------------------------------------------


def test_db_file_and_wal_contain_no_plaintext_pii(store: SessionStore, db_path: Path) -> None:
    wal = Path(f"{db_path}-wal")
    keeper = raw_connect(db_path)  # an open connection keeps the WAL file from being checkpointed away
    try:
        keeper.execute("SELECT count(*) FROM sessions").fetchone()
        state = create(store, "sess-enc")
        turn = make_turn("sess-enc", text=f"Thanks {SEED_NAME}, a summary goes to {SEED_EMAIL}")
        store.commit("sess-enc", expected_version=1, state=state, turn=turn, email_ops_create=[make_op("sess-enc")])

        assert wal.exists() and wal.stat().st_size > 0
        raw = db_path.read_bytes() + wal.read_bytes()
        assert b"sess-enc" in raw  # control: the scan does see plaintext columns
        for secret in (SEED_NAME, SEED_EMAIL, SEED_PHONE, "Quixotewright", "Thanks"):
            assert secret.lower().encode() not in raw.lower()
    finally:
        keeper.close()

    raw_after = db_path.read_bytes() + (wal.read_bytes() if wal.exists() else b"")
    assert b"sess-enc" in raw_after
    for secret in (SEED_NAME, SEED_EMAIL, SEED_PHONE):
        assert secret.lower().encode() not in raw_after.lower()


def test_db_file_wal_and_dir_are_private(store: SessionStore, db_path: Path) -> None:
    assert file_mode(db_path) == 0o600
    assert file_mode(db_path.parent) == 0o700
    keeper = raw_connect(db_path)
    try:
        keeper.execute("SELECT count(*) FROM sessions").fetchone()  # opens the file
        create(store, "sess-perm")
        assert Path(f"{db_path}-wal").exists()
        for suffix in ("-wal", "-shm"):
            sidecar = Path(f"{db_path}{suffix}")
            if sidecar.exists():
                assert file_mode(sidecar) == 0o600
    finally:
        keeper.close()
    assert file_mode(db_path) == 0o600


def test_init_schema_tightens_existing_loose_db_file(db_path: Path, cipher: StateCipher, clock: FakeClock) -> None:
    db_path.parent.mkdir(parents=True)
    sqlite3.connect(db_path).close()
    os.chmod(db_path, 0o644)
    os.chmod(db_path.parent, 0o755)

    SessionStore(db_path, cipher, clock=clock).init_schema()

    assert file_mode(db_path) == 0o600
    assert file_mode(db_path.parent) == 0o700


def test_operations_before_init_never_create_the_db(db_path: Path, cipher: StateCipher, clock: FakeClock) -> None:
    store = SessionStore(db_path, cipher, clock=clock)
    db_path.parent.mkdir(parents=True)
    with pytest.raises(sqlite3.OperationalError):
        store.load("anything")
    assert not db_path.exists()


def test_init_schema_is_idempotent_and_records_db_version(store: SessionStore, db_path: Path) -> None:
    create(store, "sess-keep")
    store.init_schema()
    store.init_schema()

    conn = raw_connect(db_path)
    try:
        version = conn.execute("SELECT value FROM meta WHERE key = 'db_schema_version'").fetchone()[0]
        journal = conn.execute("PRAGMA journal_mode").fetchone()[0]
    finally:
        conn.close()
    assert version == str(DB_SCHEMA_VERSION)
    assert journal.lower() == "wal"
    assert store.load("sess-keep").version == 1


@pytest.mark.parametrize("version", ["1", "99", "x"])
def test_init_schema_refuses_other_db_schemas(store: SessionStore, db_path: Path, version: str) -> None:
    conn = raw_connect(db_path)
    conn.execute("UPDATE meta SET value = ? WHERE key = 'db_schema_version'", (version,))
    conn.close()
    with pytest.raises(ConfigError):
        store.init_schema()


# ---------------------------------------------------------------------------
# Corrupt and unsupported checkpoints
# ---------------------------------------------------------------------------


def test_migrate_state_current_version_returns_copy() -> None:
    current = make_state("sess-now").model_dump(mode="json")

    migrated, migrated_from = migrate_state(current)

    assert migrated_from is None
    assert migrated == current
    migrated["history"].clear()
    assert current["history"]


@pytest.mark.parametrize("version", [1, 2, 99, STATE_SCHEMA_VERSION + 1, 0, -1, "2", True, None, 2.0])  # 1 and 2: retired pipeline formats
def test_migrate_state_rejects_unsupported_versions(version: object) -> None:
    payload = make_state("sess-bad").model_dump(mode="json")
    payload["schema_version"] = version
    with pytest.raises(SessionCorrupt) as excinfo:
        migrate_state(payload)
    assert excinfo.value.reason == "unsupported_schema"


def test_migrate_state_rejects_non_dict() -> None:
    with pytest.raises(SessionCorrupt) as excinfo:
        migrate_state([1, 2])  # type: ignore[arg-type]
    assert excinfo.value.reason == "invalid_json"


def test_schema_version_99_blob_is_unsupported(store: SessionStore, cipher: StateCipher, db_path: Path) -> None:
    payload = make_state("sess-99").model_dump(mode="json")
    payload["schema_version"] = 99
    insert_raw_session(db_path, "sess-99", seal(cipher, payload), schema_version=99)

    with pytest.raises(SessionCorrupt) as excinfo:
        store.load("sess-99")

    assert excinfo.value.reason == "unsupported_schema"
    assert str(excinfo.value) == "unsupported_schema"


@pytest.mark.parametrize("blob", [b"not a fernet token", b"", b"gAAAAA" + b"x" * 80])
def test_garbage_blob_is_decrypt_failed(store: SessionStore, db_path: Path, blob: bytes) -> None:
    insert_raw_session(db_path, "sess-garbage", blob)
    with pytest.raises(SessionCorrupt) as excinfo:
        store.load("sess-garbage")
    assert excinfo.value.reason == "decrypt_failed"


def test_wrong_key_is_decrypt_failed_for_state_and_turns(store: SessionStore, db_path: Path, clock: FakeClock) -> None:
    state = create(store, "sess-key")
    store.commit("sess-key", expected_version=1, state=state, turn=make_turn("sess-key"))
    intruder = SessionStore(db_path, StateCipher(Fernet.generate_key()), clock=clock)

    with pytest.raises(SessionCorrupt) as load_error:
        intruder.load("sess-key")
    with pytest.raises(SessionCorrupt) as turn_error:
        intruder.get_turn("sess-key", "turn-1")

    assert load_error.value.reason == "decrypt_failed"
    assert turn_error.value.reason == "decrypt_failed"


@pytest.mark.parametrize("plaintext", [b"{not json", b"[1, 2, 3]", b'"just a string"', b"\xff\xfe\xfa"])
def test_non_object_json_is_invalid_json(store: SessionStore, cipher: StateCipher, db_path: Path, plaintext: bytes) -> None:
    insert_raw_session(db_path, "sess-json", cipher.encrypt(plaintext))
    with pytest.raises(SessionCorrupt) as excinfo:
        store.load("sess-json")
    assert excinfo.value.reason == "invalid_json"


def test_valid_json_failing_validation_never_leaks_values(store: SessionStore, cipher: StateCipher, db_path: Path) -> None:
    payload = make_state("sess-invalid").model_dump(mode="json")
    payload["history"][1]["at"] = SEED_EMAIL
    payload["verification"]["party_id"] = {"name": SEED_NAME}
    payload["verified"] = True
    with pytest.raises(ValidationError) as pydantic_error:
        SessionState.model_validate(payload)
    assert SEED_EMAIL in str(pydantic_error.value)  # control: pydantic's own message would leak
    insert_raw_session(db_path, "sess-invalid", seal(cipher, payload), schema_version=2)

    with pytest.raises(SessionCorrupt) as excinfo:
        store.load("sess-invalid")

    error = excinfo.value
    assert error.reason == "validation_failed"
    assert "history.1.at" in error.detail
    surfaced = f"{error} {error!r} {error.detail} {error.args}"
    for secret in (SEED_EMAIL, SEED_NAME, "Quixotewright"):
        assert secret not in surfaced
    assert error.__cause__ is None
    assert error.__suppress_context__ is True


def test_blob_moved_to_another_session_row_is_rejected(store: SessionStore, db_path: Path) -> None:
    create(store, "sess-victim")
    create(store, "sess-attacker")
    conn = raw_connect(db_path)
    try:
        blob = conn.execute("SELECT state_blob FROM sessions WHERE session_id = 'sess-victim'").fetchone()[0]
        conn.execute("UPDATE sessions SET state_blob = ? WHERE session_id = 'sess-attacker'", (blob,))
    finally:
        conn.close()

    with pytest.raises(SessionCorrupt) as excinfo:
        store.load("sess-attacker")

    assert excinfo.value.reason == "validation_failed"


# ---------------------------------------------------------------------------
# Cipher and key management
# ---------------------------------------------------------------------------


def test_cipher_round_trip_and_tamper_detection(cipher: StateCipher) -> None:
    token = cipher.encrypt(b"payload")
    assert cipher.decrypt(token) == b"payload"
    assert cipher.decrypt(memoryview(token)) == b"payload"
    tampered = token[:-4] + (b"AAAA" if token[-4:] != b"AAAA" else b"BBBB")
    for bad in (tampered, "not-bytes", None):
        with pytest.raises(SessionCorrupt) as excinfo:
            cipher.decrypt(bad)  # type: ignore[arg-type]
        assert excinfo.value.reason == "decrypt_failed"
    with pytest.raises(TypeError):
        cipher.encrypt("text")  # type: ignore[arg-type]


def test_cipher_hides_key(key: bytes) -> None:
    cipher = StateCipher(key.decode())
    assert key.decode() not in repr(cipher)
    bad_key = "definitely-not-a-fernet-key-SECRET42"
    with pytest.raises(ConfigError) as excinfo:
        StateCipher(bad_key)
    assert bad_key not in str(excinfo.value)
    assert excinfo.value.__cause__ is None


def test_load_cipher_creates_private_key_file_and_reuses_it(tmp_path: Path) -> None:
    settings = Settings(data_dir=tmp_path / "data", state_encryption_key=None)

    first = load_cipher(settings)

    key_file = settings.key_file
    assert key_file.is_file()
    assert file_mode(key_file) == 0o600
    assert file_mode(key_file.parent) == 0o700
    content = key_file.read_bytes()
    token = first.encrypt(b"remember me")

    second = load_cipher(settings)

    assert second.decrypt(token) == b"remember me"
    assert key_file.read_bytes() == content
    assert [p.name for p in key_file.parent.iterdir()] == ["state.key"]


def test_load_cipher_prefers_env_key_and_writes_no_file(tmp_path: Path) -> None:
    env_key = Fernet.generate_key().decode()
    settings = Settings(data_dir=tmp_path / "data", state_encryption_key=env_key)

    cipher = load_cipher(settings)

    assert StateCipher(env_key).decrypt(cipher.encrypt(b"x")) == b"x"
    assert not settings.key_file.exists()


def test_load_cipher_invalid_env_key_fails_without_echoing_it(tmp_path: Path) -> None:
    bad_key = "this-is-not-fernet-SECRET-123"
    with pytest.raises(ConfigError) as excinfo:
        load_cipher(Settings(data_dir=tmp_path / "data", state_encryption_key=bad_key))
    assert bad_key not in str(excinfo.value)
    assert "STATE_ENCRYPTION_KEY" in str(excinfo.value)
    assert excinfo.value.__cause__ is None


@pytest.mark.parametrize("content", [b"garbage-key-material", b"", b"   \n"])
def test_load_cipher_rejects_corrupt_key_file(tmp_path: Path, content: bytes) -> None:
    settings = Settings(data_dir=tmp_path / "data", state_encryption_key=None)
    settings.data_dir.mkdir()
    settings.key_file.write_bytes(content)

    with pytest.raises(ConfigError) as excinfo:
        load_cipher(settings)

    assert "garbage-key-material" not in str(excinfo.value)
    assert settings.key_file.read_bytes() == content  # never silently replaced


def test_load_cipher_tightens_loose_key_file(tmp_path: Path) -> None:
    settings = Settings(data_dir=tmp_path / "data", state_encryption_key=None)
    settings.data_dir.mkdir()
    settings.key_file.write_bytes(Fernet.generate_key())
    os.chmod(settings.key_file, 0o644)

    load_cipher(settings)

    assert file_mode(settings.key_file) == 0o600


def test_load_cipher_concurrent_first_use_agrees_on_one_key(tmp_path: Path) -> None:
    settings = Settings(data_dir=tmp_path / "data", state_encryption_key=None)
    workers = 8
    barrier = threading.Barrier(workers)
    ciphers: list[StateCipher] = []
    guard = threading.Lock()

    def worker() -> None:
        barrier.wait()
        made = load_cipher(settings)
        with guard:
            ciphers.append(made)

    threads = [threading.Thread(target=worker) for _ in range(workers)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)

    assert len(ciphers) == workers
    token = ciphers[0].encrypt(b"shared")
    assert all(c.decrypt(token) == b"shared" for c in ciphers)
    assert [p.name for p in settings.data_dir.iterdir()] == ["state.key"]


# ---------------------------------------------------------------------------
# Email ledger
# ---------------------------------------------------------------------------


def test_email_op_created_in_commit_and_cas_succeeds_once(store: SessionStore, clock: FakeClock) -> None:
    state = create(store, "sess-mail")
    store.commit("sess-mail", expected_version=1, state=state, email_ops_create=[make_op("sess-mail")])

    record = store.email_op_get("op-1")
    assert record is not None
    assert (record.status, record.session_id, record.recipient_ref) == ("pending", "sess-mail", "P9")
    assert record.created_at == T0

    clock.advance(seconds=30)
    assert store.email_op_update("op-1", status="dispatching", expected_status="pending") is True
    assert store.email_op_update("op-1", status="dispatching", expected_status="pending") is False
    dispatching = store.email_op_get("op-1")
    assert dispatching is not None and dispatching.status == "dispatching"
    assert dispatching.updated_at == clock.now

    assert store.email_op_update("op-1", status="sent", receipt_id="msg-1", expected_status="dispatching") is True
    sent = store.email_op_get("op-1")
    assert sent is not None and (sent.status, sent.receipt_id) == ("sent", "msg-1")


def test_email_op_cas_under_contention_has_one_owner(store: SessionStore) -> None:
    state = create(store, "sess-cas")
    store.commit("sess-cas", expected_version=1, state=state, email_ops_create=[make_op("sess-cas", "op-cas")])
    workers = 8
    barrier = threading.Barrier(workers)
    outcomes: list[bool] = []
    guard = threading.Lock()

    def worker() -> None:
        barrier.wait()
        won = store.email_op_update("op-cas", status="dispatching", expected_status="pending")
        with guard:
            outcomes.append(won)

    threads = [threading.Thread(target=worker) for _ in range(workers)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)

    assert sorted(outcomes) == [False] * (workers - 1) + [True]


def test_delivered_email_op_is_never_rearmed_or_undelivered(store: SessionStore) -> None:
    state = create(store, "sess-final")
    store.commit("sess-final", expected_version=1, state=state, email_ops_create=[make_op("sess-final", "op-f")])
    assert store.email_op_update("op-f", status="dispatching", expected_status="pending")
    assert store.email_op_update("op-f", status="sent", receipt_id="msg-9")

    for status in ("pending", "dispatching", "failed", "delivery_unknown"):
        assert store.email_op_update("op-f", status=status) is False
    assert store.email_op_update("op-f", status="sent", detail="confirmed") is True
    record = store.email_op_get("op-f")
    assert record is not None
    assert (record.status, record.receipt_id, record.detail) == ("sent", "msg-9", "confirmed")


def test_failed_or_unknown_op_cannot_return_to_pending(store: SessionStore) -> None:
    state = create(store, "sess-fu")
    store.commit(
        "sess-fu",
        expected_version=1,
        state=state,
        email_ops_create=[make_op("sess-fu", "op-a", status="failed"), make_op("sess-fu", "op-b")],
    )
    assert store.email_op_update("op-a", status="pending") is False
    assert store.email_op_update("op-b", status="dispatching", expected_status="pending")
    assert store.email_op_update("op-b", status="delivery_unknown")
    assert store.email_op_update("op-b", status="pending") is False
    assert store.email_op_update("op-b", status="sent", receipt_id="reconciled") is True  # reconciliation


def test_email_op_update_missing_or_invalid(store: SessionStore) -> None:
    assert store.email_op_update("nope", status="dispatching") is False
    with pytest.raises(ValueError):
        store.email_op_update("nope", status="delivered")
    with pytest.raises(ValueError):
        store.email_op_update("nope", status="sent", expected_status="bogus")


def test_email_op_detail_is_clipped(store: SessionStore) -> None:
    state = create(store, "sess-clip")
    store.commit("sess-clip", expected_version=1, state=state, email_ops_create=[make_op("sess-clip", "op-c")])
    assert store.email_op_update("op-c", status="failed", detail="x" * 5000)
    record = store.email_op_get("op-c")
    assert record is not None and record.detail is not None and len(record.detail) == 500


def test_commit_applies_email_op_updates_atomically(store: SessionStore) -> None:
    state = create(store, "sess-upd")
    store.commit("sess-upd", expected_version=1, state=state, email_ops_create=[make_op("sess-upd", "op-u")])

    version = store.commit(
        "sess-upd",
        expected_version=2,
        state=evolve(state, phase=Phase.POST_PROCESS),
        email_ops_update=[("op-u", "dispatching", None, "claimed")],
    )

    assert version == 3
    record = store.email_op_get("op-u")
    assert record is not None and (record.status, record.detail) == ("dispatching", "claimed")


@pytest.mark.parametrize(
    "update",
    [
        ("op-missing", "dispatching", None, None),  # no such op
        ("op-foreign", "dispatching", None, None),  # op of another session
        ("op-u", "pending", None, None),  # re-arming is forbidden
        ("op-u", "not-a-status", None, None),
    ],
)
def test_commit_with_bad_email_update_rolls_back(store: SessionStore, update: tuple[str, str, str | None, str | None]) -> None:
    state = create(store, "sess-roll")
    other = create(store, "sess-foreign")
    add_ops(store, state, make_op("sess-roll", "op-u", status="failed"))
    add_ops(store, other, make_op("sess-foreign", "op-foreign"))

    with pytest.raises(ValueError):
        store.commit(
            "sess-roll",
            expected_version=2,
            state=evolve(state, phase=Phase.POST_PROCESS),
            turn=make_turn("sess-roll", "turn-roll"),
            email_ops_update=[update],
        )

    row = store.load("sess-roll")
    assert row.version == 2 and row.state.phase is Phase.VERIFY_ID
    assert store.get_turn("sess-roll", "turn-roll") is None
    foreign = store.email_op_get("op-foreign")
    assert foreign is not None and foreign.status == "pending"


def test_duplicate_op_key_in_commit_conflicts_and_rolls_back(store: SessionStore) -> None:
    state = create(store, "sess-dop")
    store.commit("sess-dop", expected_version=1, state=state, email_ops_create=[make_op("sess-dop", "op-d")])

    with pytest.raises(VersionConflict) as excinfo:
        store.commit(
            "sess-dop",
            expected_version=2,
            state=evolve(state, phase=Phase.POST_PROCESS),
            turn=make_turn("sess-dop", "turn-dop"),
            email_ops_create=[make_op("sess-dop", "op-d")],
        )

    assert excinfo.value.reason == "duplicate_email_op"
    assert store.load("sess-dop").version == 2
    assert store.get_turn("sess-dop", "turn-dop") is None


def test_commit_rejects_email_op_for_another_session(store: SessionStore) -> None:
    state = create(store, "sess-x1")
    with pytest.raises(ValueError):
        store.commit("sess-x1", expected_version=1, state=state, email_ops_create=[make_op("sess-x2")])
    assert store.email_op_get("op-1") is None
    assert store.load("sess-x1").version == 1


# ---------------------------------------------------------------------------
# Delete and purge
# ---------------------------------------------------------------------------


def test_delete_removes_session_turns_and_ops_only_for_that_session(store: SessionStore, db_path: Path) -> None:
    for session_id in ("sess-del", "sess-stay"):
        state = create(store, session_id)
        store.commit(
            session_id,
            expected_version=1,
            state=state,
            turn=make_turn(session_id),
            email_ops_create=[make_op(session_id, f"op-{session_id}")],
        )

    assert store.delete("sess-del") is True

    for table in ("sessions", "turns", "email_operations"):
        assert count_rows(db_path, table, "sess-del") == 0
        assert count_rows(db_path, table, "sess-stay") == 1
    with pytest.raises(SessionNotFound):
        store.load("sess-del")
    assert store.get_turn("sess-del", "turn-1") is None
    assert store.email_op_get("op-sess-del") is None
    assert store.load("sess-stay").version == 2
    assert store.delete("sess-del") is False


def test_purge_removes_old_rows_only(store: SessionStore, clock: FakeClock, db_path: Path) -> None:
    old_state = create(store, "sess-old")
    store.commit(
        "sess-old",
        expected_version=1,
        state=old_state,
        turn=make_turn("sess-old", at=clock.now),
        email_ops_create=[make_op("sess-old", "op-old", at=clock.now)],
    )
    clock.advance(days=8)
    new_state = create(store, "sess-new")
    store.commit(
        "sess-new",
        expected_version=1,
        state=new_state,
        turn=make_turn("sess-new", at=clock.now),
        email_ops_create=[make_op("sess-new", "op-new", at=clock.now)],
    )

    assert store.purge(retention=timedelta(days=7)) == 1

    with pytest.raises(SessionNotFound):
        store.load("sess-old")
    assert store.get_turn("sess-old", "turn-1") is None
    assert store.email_op_get("op-old") is None
    assert store.load("sess-new").version == 2
    assert store.get_turn("sess-new", "turn-1") is not None
    assert store.email_op_get("op-new") is not None
    assert store.purge(retention=timedelta(days=7)) == 0


def test_purge_uses_last_update_time_and_strict_cutoff(store: SessionStore, clock: FakeClock) -> None:
    state = create(store, "sess-edge")
    stale = create(store, "sess-active")
    clock.advance(days=6)
    store.commit("sess-active", expected_version=1, state=stale)  # activity refreshes updated_at
    clock.advance(days=1)

    assert store.purge(retention=timedelta(days=7)) == 0  # sess-edge is exactly at the cutoff
    clock.advance(microseconds=1)
    assert store.purge(retention=timedelta(days=7)) == 1

    with pytest.raises(SessionNotFound):
        store.load("sess-edge")
    assert store.load("sess-active").state == stale
    assert state.session_id == "sess-edge"


def test_purge_removes_orphaned_turns_and_ops(store: SessionStore, db_path: Path) -> None:
    create(store, "sess-live")
    stamp = T0.strftime("%Y-%m-%dT%H:%M:%S.%f+00:00")
    conn = raw_connect(db_path)
    try:
        conn.execute(
            "INSERT INTO turns VALUES ('sess-gone', 't', 1, 'h', x'00', ?)",
            (stamp,),
        )
        conn.execute(
            "INSERT INTO email_operations VALUES ('op-orphan', 'sess-gone', 'P9', 'd', 'pending', NULL, NULL, ?, ?)",
            (stamp, stamp),
        )
    finally:
        conn.close()

    assert store.purge(retention=timedelta(days=7)) == 0

    assert count_rows(db_path, "turns", "sess-gone") == 0
    assert count_rows(db_path, "email_operations", "sess-gone") == 0
    assert store.load("sess-live").version == 1


def test_purge_rejects_negative_retention(store: SessionStore) -> None:
    with pytest.raises(ValueError):
        store.purge(retention=timedelta(days=-1))


# ---------------------------------------------------------------------------
# Cross-session verification failures (C4)
# ---------------------------------------------------------------------------


def test_verification_failures_are_counted_per_subject_across_sessions(store: SessionStore, clock: FakeClock) -> None:
    store.record_verification_failure(["party:P9", "party:P13"], clock.now)
    clock.advance(minutes=5)
    store.record_verification_failure(["party:P9", "party:P9"], clock.now)  # duplicates in one call count once

    since = T0 - timedelta(hours=1)
    assert store.count_verification_failures("party:P9", since) == 2
    assert store.count_verification_failures("party:P13", since) == 1
    assert store.count_verification_failures("party:P1", since) == 0
    # The window is inclusive at ``since`` and excludes older rows.
    assert store.count_verification_failures("party:P9", T0) == 2
    assert store.count_verification_failures("party:P9", T0 + timedelta(minutes=1)) == 1


def test_verification_failures_survive_a_new_store_instance(db_path: Path, key: bytes, clock: FakeClock) -> None:
    first = SessionStore(db_path, StateCipher(key), clock=clock)
    first.init_schema()
    first.record_verification_failure(["party:P9"], clock.now)
    second = SessionStore(db_path, StateCipher(key), clock=clock)
    second.init_schema()
    assert second.count_verification_failures("party:P9", T0 - timedelta(days=1)) == 1


def test_verification_failures_empty_or_invalid_subjects(store: SessionStore, clock: FakeClock) -> None:
    store.record_verification_failure([], clock.now)
    for bad in ("", "   ", "x" * 500, None):
        with pytest.raises(ValueError):
            store.record_verification_failure([bad], clock.now)  # type: ignore[list-item]
    with pytest.raises(ValueError):
        store.count_verification_failures("", clock.now)


def test_purge_keeps_failures_for_the_longer_of_retention_and_window(store: SessionStore, clock: FakeClock) -> None:
    store.record_verification_failure(["party:P9"], clock.now)
    clock.advance(days=8)
    store.record_verification_failure(["party:P13"], clock.now)

    store.purge(retention=timedelta(days=7), failure_retention=timedelta(days=10))
    assert store.count_verification_failures("party:P9", T0 - timedelta(days=1)) == 1  # window still needs it

    store.purge(retention=timedelta(days=7), failure_retention=timedelta(hours=24))
    assert store.count_verification_failures("party:P9", T0 - timedelta(days=1)) == 0
    assert store.count_verification_failures("party:P13", T0) == 1


# ---------------------------------------------------------------------------
# Authorization columns without decryption (L17)
# ---------------------------------------------------------------------------


def test_load_auth_reads_plain_columns_even_when_blob_is_corrupt(store: SessionStore, db_path: Path) -> None:
    create(store, "sess-auth")
    conn = raw_connect(db_path)
    try:
        conn.execute("UPDATE sessions SET state_blob = ? WHERE session_id = 'sess-auth'", (b"garbage",))
    finally:
        conn.close()

    auth = store.load_auth("sess-auth")
    assert auth.secret_hash == "hash-sess-auth"
    assert auth.expires_at == EXPIRES
    with pytest.raises(SessionCorrupt):
        store.load("sess-auth")
    with pytest.raises(SessionNotFound):
        store.load_auth("sess-missing")


# ---------------------------------------------------------------------------
# Abandoned email operations (C37)
# ---------------------------------------------------------------------------


def test_email_ops_abandoned_lists_unfinished_ops_of_expired_or_missing_sessions(
    store: SessionStore, clock: FakeClock, db_path: Path
) -> None:
    expired = make_state("sess-expired")
    store.create(expired, secret_hash="h1", expires_at=T0 + timedelta(hours=1))
    add_ops(
        store,
        expired,
        make_op("sess-expired", "op-pending"),
        make_op("sess-expired", "op-dispatching", status="dispatching"),
        make_op("sess-expired", "op-sent", status="sent"),
    )
    live = make_state("sess-live")
    store.create(live, secret_hash="h2", expires_at=T0 + timedelta(days=2))
    add_ops(store, live, make_op("sess-live", "op-live"))
    stamp = T0.strftime("%Y-%m-%dT%H:%M:%S.%f+00:00")
    conn = raw_connect(db_path)
    try:
        conn.execute(
            "INSERT INTO email_operations VALUES ('op-orphan', 'sess-gone', 'P9', 'd', 'pending', NULL, NULL, ?, ?)",
            (stamp, stamp),
        )
    finally:
        conn.close()

    now = T0 + timedelta(hours=2)
    keys = [op.op_key for op in store.email_ops_abandoned(now=now, idle_before=now)]
    assert sorted(keys) == ["op-dispatching", "op-orphan", "op-pending"]
    # Ops touched after ``idle_before`` are left for a possibly live owner.
    assert store.email_ops_abandoned(now=now, idle_before=T0) == []


# ---------------------------------------------------------------------------
# DB schema v1 -> v2 (C40, C4)
# ---------------------------------------------------------------------------
