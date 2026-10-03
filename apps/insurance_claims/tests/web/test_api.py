"""HTTP contract: validation, session isolation, CSRF, duplicate submits, headers, static UI, health."""

from __future__ import annotations

import re
import threading
import uuid

import pytest
from fastapi.testclient import TestClient

from insurance_claims.config import ConfigError, Settings
from insurance_claims.web.app import COOKIE_NAME, STATIC_DIR, create_app
from tests.conftest import DEMO_UTTERANCE, SEEDED_KEY


@pytest.fixture
def app(settings, clock):
    return create_app(settings, clock=clock)


@pytest.fixture
def client(app):
    return TestClient(app)


def start(client: TestClient) -> tuple[str, str]:
    r = client.post("/api/sessions")
    assert r.status_code == 201
    body = r.json()
    return body["session_id"], body["csrf_token"]


def send(client: TestClient, sid: str, csrf: str, text: str | None = None, action: str | None = None, turn_id: str | None = None):
    body = {"client_turn_id": turn_id or uuid.uuid4().hex}
    if text is not None:
        body["text"] = text
    if action is not None:
        body["action"] = action
    return client.post(f"/api/sessions/{sid}/messages", json=body, headers={"X-CSRF-Token": csrf})


def test_create_session_sets_httponly_cookie_and_greets(client):
    r = client.post("/api/sessions")
    assert r.status_code == 201
    cookie = r.headers["set-cookie"]
    assert COOKIE_NAME in cookie and "HttpOnly" in cookie and "SameSite=strict" in cookie.replace("Strict", "strict")
    body = r.json()
    assert body["state"]["phase"] == "VERIFY_ID"
    assert body["messages"][0]["role"] == "assistant"
    assert re.fullmatch(r"[0-9a-f]{64}", body["csrf_token"])
    assert SEEDED_KEY not in r.text


def test_full_turn_and_resume(client):
    sid, csrf = start(client)
    r = send(client, sid, csrf, DEMO_UTTERANCE)
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["state"]["verified"] is True
    assert data["state"]["phase"] == "PROCESS_CASE"
    resumed = client.get(f"/api/sessions/{sid}")
    assert resumed.status_code == 200
    texts = [m["text"] for m in resumed.json()["messages"]]
    assert DEMO_UTTERANCE in texts
    assert resumed.json()["csrf_token"] == csrf


def test_session_isolation_between_clients(app):
    a, b = TestClient(app), TestClient(app)
    sid, csrf = start(a)
    start(b)
    assert b.get(f"/api/sessions/{sid}").status_code == 401
    r = b.post(f"/api/sessions/{sid}/messages", json={"client_turn_id": uuid.uuid4().hex, "text": "hi"}, headers={"X-CSRF-Token": csrf})
    assert r.status_code == 401
    assert r.json()["error"]["code"] == "unauthorized"


def test_missing_cookie_and_csrf(client, app):
    sid, _csrf = start(client)
    anon = TestClient(app)
    assert anon.get(f"/api/sessions/{sid}").status_code == 401
    r = client.post(f"/api/sessions/{sid}/messages", json={"client_turn_id": uuid.uuid4().hex, "text": "hi"})
    assert r.status_code == 403 and r.json()["error"]["code"] == "csrf_failed"
    r = client.post(
        f"/api/sessions/{sid}/messages", json={"client_turn_id": uuid.uuid4().hex, "text": "hi"}, headers={"X-CSRF-Token": "0" * 64}
    )
    assert r.status_code == 403


@pytest.mark.parametrize(
    "body",
    [
        {"text": "hi"},
        {"client_turn_id": "short", "text": "hi"},
        {"client_turn_id": "abcd1234!!", "text": "hi"},
        {"client_turn_id": uuid.uuid4().hex},
        {"client_turn_id": uuid.uuid4().hex, "text": "hi", "action": "email_send"},
        {"client_turn_id": uuid.uuid4().hex, "action": "delete_everything"},
        {"client_turn_id": uuid.uuid4().hex, "text": "hi", "verified": True},
        {"client_turn_id": uuid.uuid4().hex, "text": "   "},
    ],
)
def test_request_validation_envelope(client, body):
    sid, csrf = start(client)
    r = client.post(f"/api/sessions/{sid}/messages", json=body, headers={"X-CSRF-Token": csrf})
    assert r.status_code == 400
    err = r.json()["error"]
    assert err["code"] == "invalid_request" and err["request_id"]


def test_message_length_and_body_limits(client):
    sid, csrf = start(client)
    r = send(client, sid, csrf, "x" * 2001)
    assert r.status_code == 413
    r = client.post(
        f"/api/sessions/{sid}/messages",
        content=b'{"client_turn_id":"' + b"a" * 20 + b'","text":"' + b"y" * 20000 + b'"}',
        headers={"X-CSRF-Token": csrf, "Content-Type": "application/json"},
    )
    assert r.status_code == 413


def test_duplicate_submit_replays_and_reuse_rejected(client, settings):
    sid, csrf = start(client)
    first = send(client, sid, csrf, DEMO_UTTERANCE, turn_id="dup-turn-000001")
    second = send(client, sid, csrf, DEMO_UTTERANCE, turn_id="dup-turn-000001")
    assert first.status_code == second.status_code == 200
    assert second.json()["duplicate"] is True
    assert second.json()["messages"] == first.json()["messages"]
    history = client.get(f"/api/sessions/{sid}").json()["messages"]
    assert sum(1 for m in history if m["text"] == DEMO_UTTERANCE) == 1
    reused = send(client, sid, csrf, "something else", turn_id="dup-turn-000001")
    assert reused.status_code == 409 and reused.json()["error"]["code"] == "turn_id_reused"


def test_concurrent_turns_never_lose_state(app):
    client = TestClient(app)
    sid, csrf = start(client)
    results: list[int] = []

    def worker(i: int) -> None:
        c = TestClient(app)
        c.cookies = client.cookies
        r = send(c, sid, csrf, f"my phone is 650-521-283{i}")
        results.append(r.status_code)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert set(results) <= {200, 409}
    assert 200 in results
    history = client.get(f"/api/sessions/{sid}").json()
    user_msgs = [m for m in history["messages"] if m["role"] == "user"]
    assert len(user_msgs) == results.count(200)
    assert history["state"]["turn_index"] == results.count(200)


def test_html_is_returned_as_text_not_markup(client):
    sid, csrf = start(client)
    payload = "<script>alert(1)</script><img src=x onerror=alert(2)>"
    r = send(client, sid, csrf, payload)
    assert r.status_code == 200
    history = client.get(f"/api/sessions/{sid}").json()["messages"]
    assert any(m["text"] == payload for m in history)
    js = (STATIC_DIR / "app.js").read_text()
    assert not re.search(r"\.(innerHTML|outerHTML)\s*=|insertAdjacentHTML\s*\(|document\.write\s*\(", js)


def test_security_headers_and_static_ui(client):
    r = client.get("/")
    assert r.status_code == 200 and "text/html" in r.headers["content-type"]
    csp = r.headers["content-security-policy"]
    assert "default-src 'self'" in csp and "frame-ancestors 'none'" in csp and "unsafe-inline" not in csp
    assert r.headers["x-content-type-options"] == "nosniff"
    for asset in ("/static/app.js", "/static/app.css"):
        a = client.get(asset)
        assert a.status_code == 200
        assert SEEDED_KEY not in a.text and "OPENAI" not in a.text
    api = client.post("/api/sessions")
    assert api.headers["cache-control"] == "no-store"
    html = r.text
    assert 'src="/static/app.js"' in html
    assert re.search(r"<script(?![^>]*\bsrc=)[^>]*>", html) is None, "no inline scripts under the CSP"
    assert re.search(r"\sstyle=|\son[a-z]+=", html) is None, "no inline styles or handlers under the CSP"


def test_health_reports_without_secrets(client):
    r = client.get("/health")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "ok"
    assert data["checks"]["database"] == "ok"
    assert data["checks"]["model"]["configured"] is True
    assert SEEDED_KEY not in r.text


def test_delete_session(client):
    sid, csrf = start(client)
    assert client.delete(f"/api/sessions/{sid}").status_code == 403
    assert client.delete(f"/api/sessions/{sid}", headers={"X-CSRF-Token": csrf}).status_code == 204
    assert client.get(f"/api/sessions/{sid}").status_code in (401, 404)


def test_unknown_routes_and_bad_ids(client):
    assert client.get("/api/sessions/!!bad!!").status_code == 404
    assert client.get("/nope").json()["error"]["code"] == "not_found"


def test_email_buttons_flow_over_http(client, settings):
    sid, csrf = start(client)
    send(client, sid, csrf, DEMO_UTTERANCE)
    offer = send(client, sid, csrf, "that's all, thanks").json()
    assert offer["state"]["email"]["offer_active"] is True
    assert any(m["kind"] == "email_offer" for m in offer["messages"])
    done = send(client, sid, csrf, action="email_send").json()
    assert done["state"]["email"]["status"] == "queued"
    assert done["state"]["email"]["offer_active"] is False
    assert len(list(settings.outbox_dir.glob("*.eml"))) == 1


def test_missing_key_fails_startup(settings):
    with pytest.raises(ConfigError, match="OPENAI_API_KEY"):
        create_app(settings.with_overrides(openai_api_key=None, model_provider="openai"))


def test_settings_repr_hides_secrets():
    s = Settings(openai_api_key=SEEDED_KEY, smtp_password="hunter2", state_encryption_key="k" * 44)
    text = repr(s)
    assert SEEDED_KEY not in text and "hunter2" not in text and "k" * 44 not in text


# --- request body limit on streamed bodies (C13) -------------------------------------------------

SECURITY_HEADER_NAMES = ("content-security-policy", "x-content-type-options", "x-frame-options", "referrer-policy")


def _assert_enveloped(r, status: int, code: str, *, no_store: bool = True) -> None:
    assert r.status_code == status, r.text
    err = r.json()["error"]
    assert err["code"] == code
    assert err["request_id"] and r.headers["x-request-id"] == err["request_id"]
    for name in SECURITY_HEADER_NAMES:
        assert name in r.headers, name
    if no_store:
        assert r.headers["cache-control"] == "no-store"


def test_chunked_body_over_limit_is_rejected_without_processing(client):
    sid, csrf = start(client)
    padded = b'{"client_turn_id":"' + uuid.uuid4().hex.encode() + b'","text":"hello"' + b" " * 20_000 + b"}"

    def chunks():
        for i in range(0, len(padded), 1024):
            yield padded[i : i + 1024]

    r = client.post(
        f"/api/sessions/{sid}/messages",
        content=chunks(),  # a generator is sent with Transfer-Encoding: chunked and no Content-Length
        headers={"X-CSRF-Token": csrf, "Content-Type": "application/json"},
    )
    _assert_enveloped(r, 413, "payload_too_large")
    state = client.get(f"/api/sessions/{sid}").json()
    assert state["state"]["turn_index"] == 0
    assert not [m for m in state["messages"] if m["role"] == "user"]


def _asgi_post(app, path: str, chunks: list[bytes], headers: list[tuple[bytes, bytes]]) -> tuple[int, dict[str, str], bytes, int]:
    """Drive the ASGI app directly with a streamed body; return (status, headers, body, chunks_read)."""
    import anyio

    pulled = 0
    sent: list[dict] = []

    async def receive():
        nonlocal pulled
        if pulled < len(chunks):
            pulled += 1
            return {"type": "http.request", "body": chunks[pulled - 1], "more_body": pulled < len(chunks)}
        return {"type": "http.disconnect"}

    async def send(message):
        sent.append(message)

    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": path,
        "raw_path": path.encode(),
        "root_path": "",
        "query_string": b"",
        "headers": [(b"host", b"testserver"), (b"transfer-encoding", b"chunked"), *headers],
        "client": ("127.0.0.1", 5000),
        "server": ("testserver", 80),
    }
    anyio.run(app, scope, receive, send)
    start_msg = next(m for m in sent if m["type"] == "http.response.start")
    body = b"".join(m.get("body", b"") for m in sent if m["type"] == "http.response.body")
    response_headers = {k.decode().lower(): v.decode() for k, v in start_msg["headers"]}
    return start_msg["status"], response_headers, body, pulled


def test_streamed_body_stops_being_read_at_the_limit_even_without_a_session(app):
    """An unauthenticated multi-megabyte chunked body is cut off at the limit, before parsing or auth."""
    import json

    chunks = [b"{" + b" " * 4095] + [b" " * 4096] * 2047  # 8 MB if fully read
    status, headers, body, pulled = _asgi_post(
        app, "/api/sessions/" + "A" * 32 + "/messages", chunks, [(b"content-type", b"application/json")]
    )
    assert status == 413
    assert json.loads(body)["error"]["code"] == "payload_too_large"
    assert pulled <= 16_384 // 4096 + 1, f"read {pulled} chunks; the limit must stop the stream early"
    assert headers["cache-control"] == "no-store" and "content-security-policy" in headers
    assert headers["x-request-id"] == json.loads(body)["error"]["request_id"]


def test_streamed_body_under_the_limit_still_works(client):
    sid, csrf = start(client)
    raw = ('{"client_turn_id":"' + uuid.uuid4().hex + '","text":' + '"' + DEMO_UTTERANCE + '"}').encode()
    r = client.post(
        f"/api/sessions/{sid}/messages",
        content=(raw[i : i + 64] for i in range(0, len(raw), 64)),
        headers={"X-CSRF-Token": csrf, "Content-Type": "application/json"},
    )
    assert r.status_code == 200, r.text
    assert r.json()["state"]["verified"] is True


# --- cross-site state-changing requests (L15) -----------------------------------------------------


@pytest.mark.parametrize(
    "headers",
    [
        {"Origin": "http://evil.example"},
        {"Origin": "null"},
        {"Origin": "http://testserver.evil.example"},
        {"Origin": "http://testserver:8080"},
        {"Origin": "https://testserver:80"},
        {"Origin": "not a url"},
        {"Referer": "http://evil.example/attack.html"},
        {"Sec-Fetch-Site": "cross-site"},
        {"Sec-Fetch-Site": "same-site", "Origin": "http://testserver"},
    ],
)
def test_cross_site_session_creation_is_rejected_without_a_cookie(client, headers):
    r = client.post("/api/sessions", headers=headers, content=b"x=1")
    _assert_enveloped(r, 403, "cross_site_request")
    assert "set-cookie" not in r.headers
    assert COOKIE_NAME not in client.cookies


@pytest.mark.parametrize(
    "headers",
    [
        {},  # non-browser client: no Origin, Referer or Sec-Fetch-Site
        {"Origin": "http://testserver"},
        {"Origin": "http://TestServer:80"},
        {"Referer": "http://testserver/"},
        {"Sec-Fetch-Site": "same-origin", "Origin": "http://testserver"},
        {"Sec-Fetch-Site": "none"},
        {"Origin": "https://claims.example.com", "X-Forwarded-Host": "claims.example.com"},
    ],
)
def test_same_origin_and_non_browser_session_creation_allowed(client, headers):
    r = client.post("/api/sessions", headers=headers)
    assert r.status_code == 201, r.text
    assert COOKIE_NAME in r.headers["set-cookie"]


def test_cross_site_turn_and_delete_cannot_touch_an_existing_session(client):
    sid, csrf = start(client)
    evil = {"Origin": "http://evil.example", "X-CSRF-Token": csrf}
    r = client.post(f"/api/sessions/{sid}/messages", json={"client_turn_id": uuid.uuid4().hex, "text": "hi"}, headers=evil)
    _assert_enveloped(r, 403, "cross_site_request")
    r = client.delete(f"/api/sessions/{sid}", headers=evil)
    _assert_enveloped(r, 403, "cross_site_request")
    # Reads are not state-changing and stay available; the session is untouched.
    r = client.get(f"/api/sessions/{sid}", headers={"Origin": "http://evil.example"})
    assert r.status_code == 200 and r.json()["state"]["turn_index"] == 0
    assert send(client, sid, csrf, "hi").status_code == 200


def test_source_host_matching_handles_default_ports_and_ipv6():
    from starlette.datastructures import Headers

    from insurance_claims.web.app import _source_matches_host

    assert _source_matches_host("https://claims.example.com", Headers({"host": "claims.example.com:443"}))
    assert _source_matches_host("http://[::1]:8000", Headers({"host": "[::1]:8000"}))
    assert not _source_matches_host("http://[::1]:8001", Headers({"host": "[::1]:8000"}))
    assert not _source_matches_host("http://127.0.0.1:8765", Headers({"host": "localhost:8765"}))
    assert not _source_matches_host("http://evil.example", Headers({"host": "claims.example.com", "x-forwarded-host": ""}))
    assert not _source_matches_host("javascript://claims.example.com", Headers({"host": "claims.example.com"}))


# --- 500 responses and odd credentials keep the envelope (L16) ------------------------------------


def test_unhandled_error_is_enveloped_with_security_headers(app, client):
    sid, csrf = start(client)

    def boom(*args, **kwargs):
        raise RuntimeError("Margaret Chen 1985-03-15")

    app.state.runtime.service.post_turn = boom
    r = send(client, sid, csrf, "hello")
    _assert_enveloped(r, 500, "internal_error")
    assert "Margaret" not in r.text and "1985" not in r.text


def test_non_ascii_csrf_token_is_a_403_not_a_crash(client):
    sid, _csrf = start(client)
    bad = {"X-CSRF-Token": b"\xe9\xe9"}
    r = client.post(f"/api/sessions/{sid}/messages", json={"client_turn_id": uuid.uuid4().hex, "text": "hi"}, headers=bad)
    _assert_enveloped(r, 403, "csrf_failed")
    r = client.delete(f"/api/sessions/{sid}", headers=bad)
    _assert_enveloped(r, 403, "csrf_failed")
    assert client.get(f"/api/sessions/{sid}").status_code == 200


def test_runtime_wires_the_cross_session_failure_ledger(app):
    runtime = app.state.runtime
    assert runtime.agent.failure_ledger is runtime.store
