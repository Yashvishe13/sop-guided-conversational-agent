"""Browser smoke test of the complete workflow with the deterministic fake model.

Opt-in: ``pytest -m browser`` (needs ``pip install -e ".[browser]"``). Uses the
installed Google Chrome when available, otherwise Playwright's Chromium.
"""

from __future__ import annotations

import socket
import threading
import time
from datetime import date

import pytest

from tests.conftest import DEMO_UTTERANCE

pytestmark = pytest.mark.browser

playwright_sync = pytest.importorskip("playwright.sync_api")


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def server(settings):
    import uvicorn

    from insurance_claims.web.app import create_app

    app = create_app(settings.with_overrides(frozen_today=date(2026, 10, 3)))
    port = _free_port()
    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
    srv = uvicorn.Server(config)
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
        pg.on("console", lambda msg: errors.append(msg.text) if msg.type == "error" else None)
        pg.errors = errors  # type: ignore[attr-defined]
        yield pg
        browser.close()


def _send(page, text: str) -> None:
    before = page.locator("#transcript .msg--assistant").count()
    page.fill("#composer-input", text)
    page.keyboard.press("Enter")
    page.wait_for_function("n => document.querySelectorAll('#transcript .msg--assistant').length > n", arg=before, timeout=15000)


def _last_assistant(page) -> str:
    return page.locator("#transcript .msg--assistant").last.inner_text()


def test_complete_workflow_in_browser(server, page):
    url, settings = server
    page.goto(url)
    page.wait_for_selector("#transcript .msg--assistant")
    assert "verify" in _last_assistant(page).lower() or "details" in _last_assistant(page).lower()
    assert page.locator("#progress [aria-current='step']").count() == 1

    _send(page, DEMO_UTTERANCE)
    reply = _last_assistant(page)
    assert "CL-2048" in reply
    assert page.locator("#verified-badge").is_visible()
    active = page.locator("#progress [aria-current='step']").inner_text().lower()
    assert "review" in active or "claim" in active

    _send(page, "What documents do I need and how do I submit them?")
    assert "pathology" in _last_assistant(page).lower()

    _send(page, "<b>bold?</b> is this rendered")
    _send(page, "No, that's all, thanks.")
    # HTML typed by the user is shown as text, never rendered
    user_texts = page.locator("#transcript .msg--user").all_inner_texts()
    assert any("<b>bold?</b>" in t for t in user_texts)
    assert page.locator("#transcript b").count() == 0

    send_btn = page.get_by_role("button", name="Send email")
    skip_btn = page.get_by_role("button", name="Skip")
    send_btn.wait_for(timeout=10000)
    assert skip_btn.is_visible()
    before = page.locator("#transcript .msg--assistant").count()
    send_btn.click()
    page.wait_for_function("n => document.querySelectorAll('#transcript .msg--assistant').length > n", arg=before, timeout=15000)
    assert "outbox" in _last_assistant(page).lower()
    assert len(list(settings.outbox_dir.glob("*.eml"))) == 1
    assert page.get_by_role("button", name="Send email").count() == 0

    # reload resumes the same conversation
    count = page.locator("#transcript .msg").count()
    page.reload()
    page.wait_for_selector("#transcript .msg--assistant")
    page.wait_for_function("n => document.querySelectorAll('#transcript .msg').length >= n", arg=count, timeout=10000)
    assert page.locator("#verified-badge").is_visible()
    assert not page.errors, page.errors  # type: ignore[attr-defined]


def test_mobile_layout_has_no_horizontal_scroll(server, page):
    url, _ = server
    page.set_viewport_size({"width": 360, "height": 740})
    page.goto(url)
    page.wait_for_selector("#transcript .msg--assistant")
    _send(page, "What is RL?")
    overflow = page.evaluate("document.documentElement.scrollWidth - document.documentElement.clientWidth")
    assert overflow <= 0
    assert page.locator("#send-button").is_visible()
    page.fill("#composer-input", "typed with keyboard")
    reached = set()
    page.evaluate("document.body.focus()")
    for _ in range(25):
        page.keyboard.press("Tab")
        reached.add(page.evaluate("document.activeElement ? document.activeElement.id : ''"))
    assert {"composer-input", "send-button", "new-conversation"} <= reached, reached


def test_enter_during_new_conversation_is_queued_and_sent_once(server, page):
    url, _ = server
    page.goto(url)
    page.wait_for_selector("#transcript .msg--assistant")
    posts: list[str] = []
    page.on("request", lambda r: posts.append(r.post_data or "") if r.method == "POST" and r.url.endswith("/messages") else None)
    page.evaluate("window.confirm = () => true")
    page.click("#new-conversation")
    page.fill("#composer-input", "hello there")  # typed while the new session is still being created
    page.keyboard.press("Enter")
    page.wait_for_function("n => document.querySelectorAll('#transcript .msg--assistant').length >= n", arg=2, timeout=15000)
    assert sum("hello there" in p for p in posts) == 1
    assert page.locator("#transcript .msg--user").count() == 1
