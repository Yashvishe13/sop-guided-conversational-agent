"""Email durability through the real service: crashes, timeouts, and restarts.

One explicit consent must produce at most one external send, and the status
reported to the caller must be truthful (sent, queued, failed, or unknown).
"""

from __future__ import annotations

import pytest

from insurance_claims.mail.sender import ScriptedTransport
from tests.conftest import DEMO_UTTERANCE, Harness


class Crash(BaseException):
    """Simulates the process dying (not an Exception, so nothing catches it)."""


def _to_offer(h: Harness) -> None:
    h.say(DEMO_UTTERANCE)
    h.say("that's all")
    assert h.state.email.status == "offered"


def _restart(h: Harness, make_runtime, transport) -> Harness:
    fresh = Harness.__new__(Harness)
    fresh.runtime = make_runtime(email_transport=transport)
    fresh.service = fresh.runtime.service
    fresh.session_id, fresh.secret, fresh.csrf = h.session_id, h.secret, h.csrf
    fresh.replies = []
    return fresh


def test_crash_after_consent_before_dispatch_sends_once_on_recovery(harness, make_runtime, monkeypatch):
    transport = ScriptedTransport(["ok"])
    h = harness(email_transport=transport)
    _to_offer(h)

    def die(*args, **kwargs):
        raise Crash()

    monkeypatch.setattr(h.runtime.dispatcher, "dispatch", die)
    with pytest.raises(Crash):
        h.act("email_send")
    assert h.state.email.status == "consented"  # consent + op key were checkpointed first
    assert transport.delivered == []
    monkeypatch.undo()

    after = _restart(h, make_runtime, transport)
    view = after.service.get_session(after.session_id, after.secret)
    assert len(transport.delivered) == 1
    assert after.state.email.status == "sent"
    assert any("sent the summary" in m["text"] for m in view["messages"])
    after.service.get_session(after.session_id, after.secret)
    after.say("thanks")
    assert len(transport.delivered) == 1


def test_crash_after_dispatch_before_final_commit_never_resends(harness, make_runtime, monkeypatch):
    transport = ScriptedTransport(["ok"])
    h = harness(email_transport=transport)
    _to_offer(h)

    def die(*args, **kwargs):
        raise Crash()

    monkeypatch.setattr(h.runtime.agent, "complete_email", die)
    with pytest.raises(Crash):
        h.act("email_send")
    assert len(transport.delivered) == 1
    monkeypatch.undo()

    after = _restart(h, make_runtime, transport)
    after.service.get_session(after.session_id, after.secret)
    assert after.state.email.status == "sent"
    assert len(transport.delivered) == 1


def test_timeout_without_reconciliation_reports_unknown_and_does_not_retry(harness):
    transport = ScriptedTransport(["timeout_after_send"], supports_reconciliation=False)
    h = harness(email_transport=transport)
    _to_offer(h)
    payload = h.act("email_send")
    assert h.state.email.status == "delivery_unknown"
    assert transport.send_calls == [h.state.email.op_key]
    reply = h.last_reply(payload).lower()
    assert "can't confirm" in reply or "cannot confirm" in reply
    assert "sent the summary" not in reply
    assert h.state.handoff.offered
    h.act("email_send")
    assert len(transport.send_calls) == 1


def test_timeout_with_reconciliation_records_sent_once(harness):
    transport = ScriptedTransport(["timeout_after_send"], supports_reconciliation=True)
    h = harness(email_transport=transport)
    _to_offer(h)
    h.act("email_send")
    assert h.state.email.status == "sent"
    assert len(transport.delivered) == 1


def test_rejection_is_reported_as_failed(harness):
    transport = ScriptedTransport(["reject"])
    h = harness(email_transport=transport)
    _to_offer(h)
    payload = h.act("email_send")
    assert h.state.email.status == "failed"
    assert transport.delivered == []
    assert "couldn't be sent" in h.last_reply(payload).lower() or "could not be sent" in h.last_reply(payload).lower()


def test_real_transport_success_says_sent(harness):
    transport = ScriptedTransport(["ok"], receipt_status="sent")
    h = harness(email_transport=transport)
    _to_offer(h)
    payload = h.act("email_send")
    assert h.state.email.status == "sent"
    assert "sent the summary" in h.last_reply(payload).lower()
    message = transport.delivered[0]
    assert message.to_address == "margaret@email.com"
    assert "—" not in message.body_text and ":" not in message.body_text


def test_second_consent_after_new_wrap_up_is_a_new_operation(harness):
    transport = ScriptedTransport(["ok", "ok"])
    h = harness(email_transport=transport)
    _to_offer(h)
    h.act("email_send")
    h.say("One more question, what documents do I need?")
    h.say("that's all")
    assert h.state.email.status == "offered"
    h.act("email_send")
    assert len(transport.delivered) == 2
    assert len({m.op_key for m in transport.delivered}) == 2
    assert h.state.email.send_count == 2


def _consent_then_crash(h: Harness, monkeypatch) -> None:
    def die(*args, **kwargs):
        raise Crash()

    monkeypatch.setattr(h.runtime.dispatcher, "dispatch", die)
    with pytest.raises(Crash):
        h.act("email_send")
    monkeypatch.undo()


def _sent_notices(h: Harness) -> int:
    return sum(1 for m in h.state.history if "sent the summary" in m.text)


def test_concurrent_reload_with_a_stale_snapshot_does_not_record_the_send_twice(harness, make_runtime, monkeypatch):
    """L11: the second reload re-reads under the lock and finds nothing left to recover."""
    transport = ScriptedTransport(["ok"])
    h = harness(email_transport=transport)
    _to_offer(h)
    _consent_then_crash(h, monkeypatch)

    after = _restart(h, make_runtime, transport)
    service = after.service
    real_load = service._load
    calls = {"n": 0}

    def load(session_id):
        row = real_load(session_id)
        calls["n"] += 1
        if calls["n"] == 1:  # another tab recovers between our first read and our lock
            service.get_session(after.session_id, after.secret)
        return row

    monkeypatch.setattr(service, "_load", load)
    service.get_session(after.session_id, after.secret)
    monkeypatch.undo()

    assert len(transport.delivered) == 1
    assert after.state.email.status == "sent"
    assert after.state.email.send_count == 1
    assert _sent_notices(after) == 1


def test_recovered_result_is_in_the_turn_response_when_history_is_full(harness, make_runtime, settings, monkeypatch):
    """L13: the notice is returned even though history was trimmed to history_limit."""
    import dataclasses

    small = dataclasses.replace(settings, history_limit=4)
    transport = ScriptedTransport(["ok"])
    h = harness(email_transport=transport, settings_override=small)
    _to_offer(h)
    _consent_then_crash(h, monkeypatch)
    assert len(h.state.history) == 4

    fresh = Harness.__new__(Harness)
    fresh.runtime = make_runtime(email_transport=transport, settings_override=small)
    fresh.service = fresh.runtime.service
    fresh.session_id, fresh.secret, fresh.csrf = h.session_id, h.secret, h.csrf
    fresh.replies = []
    payload = fresh.say("thanks")
    assert len(transport.delivered) == 1
    assert any("sent the summary" in m["text"] for m in payload["messages"])


def test_recovery_does_not_extend_the_verification_idle_ttl(harness, make_runtime, clock, monkeypatch):
    transport = ScriptedTransport(["ok"])
    h = harness(email_transport=transport)
    _to_offer(h)
    _consent_then_crash(h, monkeypatch)
    clock.advance(minutes=45)

    after = _restart(h, make_runtime, transport)
    view = after.service.get_session(after.session_id, after.secret)
    assert len(transport.delivered) == 1  # the consented send still completes
    assert view["state"]["verified"] is False
    assert "CL-2048" not in " ".join(m["text"] for m in view["messages"])
    reply = after.last_reply(after.say("hello?")).lower()
    assert "expired" in reply
    assert after.state.verified is False


def test_interrupted_dispatch_is_resent_once_when_lookup_proves_nothing_went_out(harness, make_runtime, clock):
    """C44: an orphaned claim on a reconcilable transport is finished by recovery, at most once."""
    transport = ScriptedTransport(["crash_before_send", "ok"], supports_reconciliation=True)
    h = harness(email_transport=transport)
    _to_offer(h)
    with pytest.raises(BaseException):
        h.act("email_send")
    op_key = h.state.email.op_key
    assert h.service.store.email_op_get(op_key).status == "dispatching"
    assert transport.delivered == []

    # A claim inside the dispatcher's in-flight window is left alone and the consent stays pending.
    early = _restart(h, make_runtime, transport)
    early.service.get_session(early.session_id, early.secret)
    assert transport.delivered == []
    assert early.state.email.status == "consented"

    clock.advance(minutes=5)
    after = _restart(h, make_runtime, transport)
    after.service.get_session(after.session_id, after.secret)
    after.service.get_session(after.session_id, after.secret)
    after.say("thanks")
    assert len(transport.delivered) == 1
    assert after.state.email.status == "sent"
    assert after.state.email.send_count == 1


def test_interrupted_dispatch_without_reconciliation_is_never_resent(harness, make_runtime, clock):
    transport = ScriptedTransport(["crash_before_send", "ok"], supports_reconciliation=False)
    h = harness(email_transport=transport)
    _to_offer(h)
    with pytest.raises(BaseException):
        h.act("email_send")
    clock.advance(minutes=5)
    after = _restart(h, make_runtime, transport)
    view = after.service.get_session(after.session_id, after.secret)
    assert transport.delivered == [] and len(transport.send_calls) == 1
    assert after.state.email.status == "delivery_unknown"
    assert any("can't confirm" in m["text"].lower() or "cannot confirm" in m["text"].lower() for m in view["messages"])
