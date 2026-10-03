"""Conversation service: what a read may show, auth order, request hashing, and retention."""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import time
from datetime import timedelta
from pathlib import Path

import pytest

from insurance_claims.mail.sender import ScriptedTransport
from insurance_claims.web.service import ServiceError
from tests.conftest import DEMO_UTTERANCE, Harness

CLAIM_TOKENS = ("CL-2048", "pathology", "4472", "1985-03-15")


class Crash(BaseException):
    """Simulates the process dying (not an Exception, so nothing catches it)."""


def _texts(view: dict) -> str:
    return " ".join(m["text"] for m in view["messages"])


def _db(h: Harness) -> sqlite3.Connection:
    return sqlite3.connect(h.runtime.settings.db_path)


# ---------------------------------------------------------------------------
# Idle TTL and identity resets on reads (C18, C19, C45)
# ---------------------------------------------------------------------------


def test_get_after_idle_ttl_shows_no_verified_state_or_earlier_transcript(harness, clock) -> None:
    h = harness()
    h.say(DEMO_UTTERANCE)
    clock.advance(minutes=31)

    view = h.service.get_session(h.session_id, h.secret)
    state = view["state"]
    assert state["verified"] is False
    assert "case" not in state
    assert state["phase"] == "VERIFY_ID"
    assert all(m["turn_index"] == 0 for m in view["messages"])
    assert not any(token in _texts(view) for token in CLAIM_TOKENS)
    # Read-only: the stored checkpoint is unchanged, so the next turn still expires it and says so.
    assert h.state.verified

    reply = h.last_reply(h.say("hello?")).lower()
    assert "expired" in reply
    after = h.service.get_session(h.session_id, h.secret)
    assert after["state"]["verified"] is False
    assert not any(token in _texts(after) for token in CLAIM_TOKENS)
    assert any("hello?" in m["text"] for m in after["messages"])


def test_get_within_idle_ttl_restores_the_full_transcript(harness, clock) -> None:
    h = harness()
    h.say(DEMO_UTTERANCE)
    clock.advance(minutes=10)
    view = h.service.get_session(h.session_id, h.secret)
    assert view["state"]["verified"] is True
    assert view["state"]["case"] == {"case_id": "CL-2048"}
    assert "CL-2048" in _texts(view)


def test_idle_expired_email_offer_is_not_shown_as_active(harness, clock) -> None:
    h = harness()
    h.say(DEMO_UTTERANCE)
    h.say("that's all")
    assert h.state.email.status == "offered"
    clock.advance(minutes=45)
    view = h.service.get_session(h.session_id, h.secret)
    assert view["state"]["email"]["offer_active"] is False
    assert view["state"]["verified"] is False


def test_caller_change_hides_the_previous_callers_messages(harness) -> None:
    h = harness()
    h.say(DEMO_UTTERANCE)
    h.say("My name is Ava Lopez, DOB 1990-08-21, SSN last four 9180.")
    view = h.service.get_session(h.session_id, h.secret)
    text = _texts(view)
    assert not any(token in text for token in CLAIM_TOKENS)
    assert "Ava Lopez" in text  # the reset turn itself is still shown


def test_replay_after_idle_expiry_does_not_return_the_old_verified_reply(harness, clock) -> None:
    h = harness()
    first = h.say(DEMO_UTTERANCE, turn_id="turn-demo")
    assert first["state"]["verified"] is True
    clock.advance(minutes=31)
    replay = h.say(DEMO_UTTERANCE, turn_id="turn-demo")
    assert replay["duplicate"] is True
    assert replay["messages"] == []
    assert replay["state"]["verified"] is False and "case" not in replay["state"]


def test_replay_inside_ttl_is_unchanged(harness, clock) -> None:
    h = harness()
    first = h.say(DEMO_UTTERANCE, turn_id="turn-demo")
    clock.advance(minutes=5)
    replay = h.say(DEMO_UTTERANCE, turn_id="turn-demo")
    assert replay["duplicate"] is True
    assert replay["messages"] == first["messages"]


# ---------------------------------------------------------------------------
# Keyed request hash (C40)
# ---------------------------------------------------------------------------


def test_request_hash_column_is_keyed_not_a_plain_digest(harness) -> None:
    h = harness()
    h.say("4472", turn_id="turn-short")
    conn = _db(h)
    try:
        stored = conn.execute("SELECT request_hash FROM turns WHERE client_turn_id = 'turn-short'").fetchone()[0]
    finally:
        conn.close()
    plain = hashlib.sha256(json.dumps({"text": "4472", "action": None}, sort_keys=True).encode()).hexdigest()
    assert stored and stored != plain

    assert h.say("4472", turn_id="turn-short")["duplicate"] is True
    with pytest.raises(ServiceError) as err:
        h.say("4473", turn_id="turn-short")
    assert err.value.code == "turn_id_reused"


# ---------------------------------------------------------------------------
# Header robustness and authorization order (L16, L17)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("csrf", ["\xe9\xe9", "é" * 64, "\ud800"])
def test_non_ascii_csrf_token_is_rejected_with_403(harness, csrf: str) -> None:
    h = harness()
    with pytest.raises(ServiceError) as err:
        h.service.post_turn(h.session_id, h.secret, csrf, client_turn_id="t1", text="hi", action=None)
    assert err.value.status == 403


@pytest.mark.parametrize("secret", ["\xe9\xe9", "\ud800abc"])
def test_non_ascii_cookie_is_rejected_with_401(harness, secret: str) -> None:
    h = harness()
    with pytest.raises(ServiceError) as err:
        h.service.get_session(h.session_id, secret)
    assert err.value.status == 401
    with pytest.raises(ServiceError) as err:
        h.service.post_turn(h.session_id, secret, h.csrf, client_turn_id="t1", text="hi", action=None)
    assert err.value.status == 401


def _corrupt(h: Harness) -> None:
    conn = _db(h)
    try:
        conn.execute("UPDATE sessions SET state_blob = ? WHERE session_id = ?", (b"garbage", h.session_id))
        conn.commit()
    finally:
        conn.close()


def test_corrupt_session_checks_the_cookie_before_decrypting(harness) -> None:
    h = harness()
    _corrupt(h)
    for secret in (None, "wrong-secret"):
        with pytest.raises(ServiceError) as err:
            h.service.get_session(h.session_id, secret)
        assert err.value.status == 401
    with pytest.raises(ServiceError) as err:
        h.service.get_session(h.session_id, h.secret)
    assert err.value.status == 409 and err.value.code == "session_unrecoverable"


def test_owner_can_delete_a_corrupt_session(harness) -> None:
    h = harness()
    _corrupt(h)
    h.service.delete_session(h.session_id, h.secret, h.csrf)
    conn = _db(h)
    try:
        assert conn.execute("SELECT COUNT(*) FROM sessions WHERE session_id = ?", (h.session_id,)).fetchone()[0] == 0
    finally:
        conn.close()


def test_delete_still_requires_cookie_and_csrf(harness) -> None:
    h = harness()
    with pytest.raises(ServiceError) as err:
        h.service.delete_session(h.session_id, "wrong", h.csrf)
    assert err.value.status == 401
    with pytest.raises(ServiceError) as err:
        h.service.delete_session(h.session_id, h.secret, "wrong")
    assert err.value.status == 403


# ---------------------------------------------------------------------------
# Time-based retention (C41)
# ---------------------------------------------------------------------------


def test_purge_runs_on_time_and_removes_outbox_and_trace_files(harness, clock) -> None:
    h = harness()  # the settings use the local outbox transport
    h.say(DEMO_UTTERANCE)
    h.say("that's all")
    h.act("email_send")
    settings = h.runtime.settings
    outbox = sorted(settings.outbox_dir.glob("*.eml"))
    traces = sorted(settings.trace_dir.glob("*.json"))
    assert outbox and traces

    clock.advance(days=8)
    h.service.create_session()  # any request may run the hourly purge; no 50-create counter

    assert not any(path.exists() for path in outbox)
    assert not any(path.exists() for path in traces)
    with pytest.raises(ServiceError):
        h.service.get_session(h.session_id, h.secret)
    conn = _db(h)
    try:
        assert conn.execute("SELECT COUNT(*) FROM sessions WHERE session_id = ?", (h.session_id,)).fetchone()[0] == 0
    finally:
        conn.close()


def test_purge_runs_at_most_once_per_interval(harness, clock, monkeypatch) -> None:
    h = harness()
    calls: list[int] = []
    real = h.service.purge
    monkeypatch.setattr(h.service, "purge", lambda: calls.append(1) or real())
    h.service.get_session(h.session_id, h.secret)
    h.say("hello")
    assert calls == []  # Runtime purged at startup
    clock.advance(minutes=61)
    h.service.get_session(h.session_id, h.secret)
    h.say("hello again")
    assert calls == [1]


def test_recent_files_are_kept(harness, clock, tmp_path: Path) -> None:
    h = harness()
    outbox = h.runtime.settings.outbox_dir
    outbox.mkdir(parents=True, exist_ok=True)
    keep = outbox / "em_recent.eml"
    keep.write_text("x")
    old = outbox / "em_old.eml"
    old.write_text("x")
    stamp = clock.now.timestamp() - 8 * 86400
    os.utime(old, (stamp, stamp))
    os.utime(keep, (time.time(), time.time()))
    h.service.purge()
    assert keep.exists() and not old.exists()


# ---------------------------------------------------------------------------
# Abandoned email operations (C37)
# ---------------------------------------------------------------------------


def _consent_then_crash(h: Harness, monkeypatch) -> None:
    h.say(DEMO_UTTERANCE)
    h.say("that's all")

    def die(*args, **kwargs):
        raise Crash()

    monkeypatch.setattr(h.runtime.dispatcher, "dispatch", die)
    with pytest.raises(Crash):
        h.act("email_send")
    monkeypatch.undo()


def test_pending_op_of_expired_session_is_closed_without_sending(harness, clock, monkeypatch) -> None:
    transport = ScriptedTransport(["ok"])
    h = harness(email_transport=transport)
    _consent_then_crash(h, monkeypatch)
    op_key = h.state.email.op_key
    assert h.service.store.email_op_get(op_key).status == "pending"

    clock.advance(hours=25)
    h.service.purge()

    op = h.service.store.email_op_get(op_key)
    assert op.status == "failed" and op.detail == "abandoned_before_dispatch"
    assert transport.send_calls == []


def test_dispatching_op_of_expired_session_is_flagged_unknown_without_sending(harness, clock) -> None:
    transport = ScriptedTransport(["crash_before_send"])
    h = harness(email_transport=transport)
    h.say(DEMO_UTTERANCE)
    h.say("that's all")
    with pytest.raises(BaseException):
        h.act("email_send")
    op_key = h.state.email.op_key
    assert h.service.store.email_op_get(op_key).status == "dispatching"

    clock.advance(hours=25)
    h.service.purge()

    assert h.service.store.email_op_get(op_key).status == "delivery_unknown"
    assert len(transport.send_calls) == 1 and transport.delivered == []


def test_sweep_leaves_ops_of_live_sessions_for_lazy_recovery(harness, clock, monkeypatch) -> None:
    transport = ScriptedTransport(["ok"])
    h = harness(email_transport=transport)
    _consent_then_crash(h, monkeypatch)
    op_key = h.state.email.op_key
    clock.advance(hours=2)
    h.service.purge()
    assert h.service.store.email_op_get(op_key).status == "pending"
    h.service.get_session(h.session_id, h.secret)
    assert len(transport.delivered) == 1
    assert h.service.store.email_op_get(op_key).status == "sent"


# ---------------------------------------------------------------------------
# Cross-session failure ledger (C4)
# ---------------------------------------------------------------------------


def test_failed_attempts_are_counted_across_sessions(make_runtime, settings, clock) -> None:
    import dataclasses

    strict = dataclasses.replace(settings, max_party_failures=2)
    runtime = make_runtime(settings_override=strict)
    wrong = "My name is Margaret Chen, DOB 1985-03-15, SSN last four is 0001."
    for _ in range(2):
        h = Harness(runtime)
        assert h.say(wrong)["state"]["verified"] is False

    since = clock.now - timedelta(hours=1)
    assert runtime.store.count_verification_failures("party:P9", since) == 2

    fresh = Harness(runtime)
    assert fresh.say(DEMO_UTTERANCE)["state"]["verified"] is False
    assert "CL-2048" not in " ".join(fresh.replies)
