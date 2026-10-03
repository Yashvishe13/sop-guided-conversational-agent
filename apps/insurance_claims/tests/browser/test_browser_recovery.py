"""Browser regression tests for the chat client's recovery paths (static/app.js).

Opt-in like the smoke test: ``pytest -m browser``. Runs the real app with the deterministic fake
model and drives it through Playwright; lost responses are simulated with ``page.route``.

* An unreadable checkpoint (409 ``session_unrecoverable``) is a "session gone" case: the UI offers
  a new conversation (or starts one on reload) instead of "Still working" and endless retries.
* A turn whose response was lost is never labelled "Not sent", is never resent automatically, and
  the transcript is rebuilt from server history after the next turn.
* When a turn resets identity (verified true -> false), the transcript is rebuilt from server
  history, so messages the server now hides are not kept from the old local transcript.
"""

from __future__ import annotations

import json
import socket
import sqlite3
import threading
import time
from datetime import date
from typing import Any

import pytest

from tests.conftest import DEMO_UTTERANCE

pytestmark = pytest.mark.browser

playwright_sync = pytest.importorskip("playwright.sync_api")

DOCS_QUESTION = "What documents do I need and how do I submit them?"


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def server(settings, clock):
    import uvicorn

    from insurance_claims.web.app import create_app

    app = create_app(settings.with_overrides(frozen_today=date(2026, 10, 3)), clock=clock)
    port = _free_port()
    srv = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    thread = threading.Thread(target=srv.run, daemon=True)
    thread.start()
    for _ in range(100):
        if srv.started:
            break
        time.sleep(0.05)
    yield f"http://127.0.0.1:{port}", settings
    srv.should_exit = True
    thread.join(timeout=5)


@pytest.fixture
def page():
    with playwright_sync.sync_playwright() as p:
        try:
            browser = p.chromium.launch(channel="chrome", headless=True)
        except Exception:
            browser = p.chromium.launch(headless=True)
        context = browser.new_context(viewport={"width": 1100, "height": 900})
        pg = context.new_page()
        errors: list[str] = []
        pg.on("pageerror", lambda exc: errors.append(str(exc)))
        pg.errors = errors  # type: ignore[attr-defined]
        requests: list[tuple[str, str, str]] = []
        pg.on("request", lambda r: requests.append((r.method, r.url, r.post_data or "")) if "/api/" in r.url else None)
        pg.api_requests = requests  # type: ignore[attr-defined]
        yield pg
        browser.close()


# ---------------------------------------------------------------------- helpers


def _ready(page) -> None:
    page.wait_for_selector("#transcript .msg--assistant")
    page.wait_for_function("() => !document.getElementById('composer-input').disabled", timeout=10000)


def _idle(page) -> None:
    """Wait until no turn or resync is in flight."""
    page.wait_for_selector("#typing", state="hidden", timeout=15000)


def _send(page, text: str) -> None:
    """Send a message and wait for its answer and any transcript resync to finish."""
    page.fill("#composer-input", text)
    with page.expect_response(lambda r: r.request.method == "POST" and r.url.endswith("/messages"), timeout=15000):
        page.keyboard.press("Enter")
    _idle(page)


def _stored_id(page) -> str:
    return page.evaluate("() => localStorage.getItem('ic_session_id')")


def _ui_texts(page) -> list[str]:
    return page.evaluate("() => Array.from(document.querySelectorAll('#transcript .msg .msg__text')).map(n => n.textContent)")


def _ui_metas(page) -> list[str]:
    return page.locator("#transcript .msg__meta").all_inner_texts()


def _server_view(page) -> dict[str, Any]:
    return page.evaluate(
        "id => fetch('/api/sessions/' + encodeURIComponent(id), {credentials: 'same-origin'}).then(r => r.json())",
        _stored_id(page),
    )


def _turn_posts(page, needle: str) -> int:
    return sum(1 for m, u, body in page.api_requests if m == "POST" and u.endswith("/messages") and needle in body)


def _corrupt_checkpoint(db_path, session_id: str) -> None:
    """Simulate an unreadable checkpoint (corrupt blob, rotated key): load raises SessionCorrupt."""
    with sqlite3.connect(db_path) as conn:
        changed = conn.execute("UPDATE sessions SET state_blob = ? WHERE session_id = ?", (b"not-a-valid-token", session_id)).rowcount
    assert changed == 1


class _LoseNextTurnResponse:
    """Route handler: the next POST /messages reaches the server, but the browser never gets the reply."""

    def __init__(self) -> None:
        self.armed = False
        self.lost: dict[str, Any] | None = None

    def __call__(self, route) -> None:
        if self.armed and route.request.method == "POST":
            self.armed = False
            response = route.fetch()  # the server processes and commits the turn
            self.lost = {"status": response.status, "body": response.json()}
            route.abort("connectionreset")
        else:
            route.continue_()


# ---------------------------------------------------------------------- C38: unreadable checkpoint


def test_unreadable_checkpoint_on_send_offers_new_conversation(server, page):
    url, settings = server
    page.goto(url)
    _ready(page)
    _send(page, "hello")
    old_id = _stored_id(page)
    _corrupt_checkpoint(settings.db_path, old_id)

    page.fill("#composer-input", "my claim number is 123")
    page.keyboard.press("Enter")
    page.wait_for_selector("#error-banner:not([hidden])", timeout=10000)

    banner = page.locator("#error-message").inner_text()
    assert "could not be restored" in banner
    assert "still working" not in banner.lower()
    assert page.locator("#error-retry").inner_text() == "Start new conversation"
    assert page.locator("#composer-input").is_disabled()
    assert _turn_posts(page, "my claim number is 123") == 1  # never retried into a dead session

    page.click("#error-retry")
    page.wait_for_function(
        "old => !document.getElementById('composer-input').disabled && localStorage.getItem('ic_session_id') !== old",
        arg=old_id,
        timeout=10000,
    )
    assert page.locator("#error-banner").is_hidden()
    assert page.input_value("#composer-input") == "my claim number is 123"  # typed text is kept
    assert "hello" not in _ui_texts(page)
    assert not page.errors, page.errors  # type: ignore[attr-defined]


def test_unreadable_checkpoint_on_reload_starts_new_conversation(server, page):
    url, settings = server
    page.goto(url)
    _ready(page)
    _send(page, "hello")
    old_id = _stored_id(page)
    _corrupt_checkpoint(settings.db_path, old_id)

    page.reload()
    page.wait_for_function(
        "old => !document.getElementById('composer-input').disabled && localStorage.getItem('ic_session_id') !== old",
        arg=old_id,
        timeout=8000,
    )
    page.wait_for_selector("#transcript .msg--assistant")
    assert page.locator("#error-banner").is_hidden()
    assert "hello" not in _ui_texts(page)
    assert not page.errors, page.errors  # type: ignore[attr-defined]


# ---------------------------------------------------------------------- C39: lost turn responses


def test_lost_identity_response_then_new_message_shows_server_history(server, page):
    url, _ = server
    lose = _LoseNextTurnResponse()
    page.route("**/api/sessions/*/messages", lose)
    page.goto(url)
    _ready(page)

    lose.armed = True
    page.fill("#composer-input", DEMO_UTTERANCE)
    page.keyboard.press("Enter")
    page.wait_for_selector("#error-banner:not([hidden])", timeout=15000)
    assert lose.lost is not None and lose.lost["status"] == 200  # the server did process it
    # The outcome is unknown, so the bubble must not claim it was not sent.
    assert _ui_metas(page) == ["May not have been sent"]

    _send(page, DOCS_QUESTION)  # typed instead of pressing Try again

    server_texts = [m["text"] for m in _server_view(page)["messages"]]
    assert _ui_texts(page) == server_texts  # rebuilt from server history, nothing missing or stale
    assert any("verified" in t for t in server_texts)
    assert _ui_metas(page) == []  # no "Not sent" left on a processed turn
    assert page.locator("#verified-badge").is_visible()
    assert _turn_posts(page, "Margaret Chen") == 1  # the superseded turn was not resent
    assert not page.errors, page.errors  # type: ignore[attr-defined]


def test_lost_email_send_response_is_not_resent_and_delivery_notice_appears(server, page):
    url, settings = server
    lose = _LoseNextTurnResponse()
    page.route("**/api/sessions/*/messages", lose)
    page.goto(url)
    _ready(page)
    _send(page, DEMO_UTTERANCE)
    _send(page, DOCS_QUESTION)
    _send(page, "No, that's all, thanks.")
    page.get_by_role("button", name="Send email").wait_for(timeout=10000)

    lose.armed = True
    page.get_by_role("button", name="Send email").click()
    page.wait_for_selector("#error-banner:not([hidden])", timeout=15000)
    assert len(list(settings.outbox_dir.glob("*.eml"))) == 1
    assert _ui_metas(page) == ["May not have been sent"]

    _send(page, "ok thanks")

    server_view = _server_view(page)
    server_texts = [m["text"] for m in server_view["messages"]]
    ui_texts = _ui_texts(page)
    assert ui_texts == server_texts
    assert any("outbox" in t.lower() for t in ui_texts)  # the delivery notice is shown
    assert "Send the email summary" in ui_texts
    assert _ui_metas(page) == []
    assert server_view["state"]["email"]["status"] in ("sent", "queued")
    assert _turn_posts(page, "email_send") == 1  # never resent automatically
    assert len(list(settings.outbox_dir.glob("*.eml"))) == 1
    assert page.get_by_role("button", name="Send email").count() == 0
    assert not page.errors, page.errors  # type: ignore[attr-defined]


def test_unsent_turn_superseded_by_new_message_drops_out(server, page):
    """A turn the server refused (definitely not processed) keeps "Not sent" and is not resent."""
    url, _ = server
    state = {"refuse": False}

    def refuse_once(route) -> None:
        if state["refuse"] and route.request.method == "POST":
            state["refuse"] = False
            route.fulfill(
                status=429,
                content_type="application/json",
                body=json.dumps({"error": {"code": "rate_limited", "message": "Slow down."}}),
            )
        else:
            route.continue_()

    page.route("**/api/sessions/*/messages", refuse_once)
    page.goto(url)
    _ready(page)
    state["refuse"] = True
    page.fill("#composer-input", "first try")
    page.keyboard.press("Enter")
    page.wait_for_selector("#error-banner:not([hidden])", timeout=10000)
    assert _ui_metas(page) == ["Not sent"]

    _send(page, "second try")
    assert _ui_texts(page) == [m["text"] for m in _server_view(page)["messages"]]
    assert "first try" not in _ui_texts(page)
    assert _turn_posts(page, "first try") == 1


# ---------------------------------------------------------------------- identity reset re-render


def test_identity_reset_rebuilds_transcript_from_server_history(server, page, clock):
    url, _ = server
    hide = {"on": False}

    def hide_pre_reset_history(route) -> None:
        # Stand-in for the service hiding pre-reset history: keep only the latest (reset) turn.
        if hide["on"] and route.request.method == "GET":
            response = route.fetch()
            data = response.json()
            floor = data["state"]["turn_index"]
            data["messages"] = [m for m in data["messages"] if m["turn_index"] >= floor]
            route.fulfill(status=response.status, content_type="application/json", body=json.dumps(data))
        else:
            route.continue_()

    page.route("**/api/sessions/*", hide_pre_reset_history)
    page.goto(url)
    _ready(page)
    _send(page, DEMO_UTTERANCE)
    assert page.locator("#verified-badge").is_visible()
    assert any("CL-2048" in t for t in _ui_texts(page))

    clock.advance(minutes=31)  # past the verification idle TTL
    hide["on"] = True
    gets_before = sum(1 for m, u, _ in page.api_requests if m == "GET")
    _send(page, DOCS_QUESTION)

    assert sum(1 for m, u, _ in page.api_requests if m == "GET") > gets_before
    assert page.locator("#verified-badge").is_hidden()
    assert page.locator("#case-chip").is_hidden()
    ui_texts = _ui_texts(page)
    assert not any("CL-2048" in t for t in ui_texts), ui_texts
    assert ui_texts == [m["text"] for m in _server_view(page)["messages"]]

    # A reload renders the same server view cleanly.
    page.reload()
    _ready(page)
    assert _ui_texts(page) == ui_texts
    assert page.locator("#verified-badge").is_hidden()
    assert not page.errors, page.errors  # type: ignore[attr-defined]
