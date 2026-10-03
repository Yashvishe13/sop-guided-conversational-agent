"""Shared fixtures for integration-level tests (deterministic fake model, temp storage, frozen clock)."""

from __future__ import annotations

import uuid
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any, Callable

import pytest
from cryptography.fernet import Fernet

from insurance_claims.config import APP_ROOT, Settings

TODAY = date(2026, 10, 3)
SEEDED_KEY = "sk-test-SEEDED-0123456789abcdefABCDEF"
DEMO_UTTERANCE = (
    "I'm the policyholder. My name is Margaret Chen, policy POL-9921. I'm calling about my denied healthcare "
    "claim from January. DOB is 1985-03-15, SSN last four is 4472."
)


class FakeClock:
    def __init__(self, start: datetime | None = None) -> None:
        self.now = start or datetime(2026, 10, 3, 15, 0, tzinfo=UTC)

    def __call__(self) -> datetime:
        return self.now

    def advance(self, **kwargs: float) -> None:
        self.now = self.now + timedelta(**kwargs)


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        openai_api_key=SEEDED_KEY,
        model_provider="fake",
        fixtures_dir=APP_ROOT / "fixtures",
        prompts_path=APP_ROOT / "prompts.toml",
        data_dir=tmp_path / "data",
        trace_dir=tmp_path / "traces",
        traces_enabled=True,
        state_encryption_key=Fernet.generate_key().decode(),
        frozen_today=TODAY,
        environment="test",
        email_transport="outbox",
    )


class Harness:
    """Drives the real service stack (agent, SQLite store, email dispatcher) without HTTP."""

    def __init__(self, runtime: Any) -> None:
        self.runtime = runtime
        self.service = runtime.service
        self.session_id, self.secret, self.created = self.service.create_session()
        self.csrf = self.service.csrf_token(self.session_id, self.secret)
        self.replies: list[str] = []

    def say(self, text: str, turn_id: str | None = None) -> dict[str, Any]:
        payload = self.service.post_turn(
            self.session_id, self.secret, self.csrf, client_turn_id=turn_id or uuid.uuid4().hex, text=text, action=None
        )
        self.replies.extend(m["text"] for m in payload["messages"] if m["role"] == "assistant")
        return payload

    def act(self, action: str, turn_id: str | None = None) -> dict[str, Any]:
        payload = self.service.post_turn(
            self.session_id, self.secret, self.csrf, client_turn_id=turn_id or uuid.uuid4().hex, text=None, action=action
        )
        self.replies.extend(m["text"] for m in payload["messages"] if m["role"] == "assistant")
        return payload

    @property
    def state(self):
        return self.service.store.load(self.session_id).state

    def last_reply(self, payload: dict[str, Any]) -> str:
        return " ".join(m["text"] for m in payload["messages"] if m["role"] == "assistant")


@pytest.fixture
def make_runtime(settings: Settings, clock: FakeClock) -> Callable[..., Any]:
    from insurance_claims.web.app import Runtime

    def _make(model: Any = None, email_transport: Any = None, settings_override: Settings | None = None) -> Any:
        return Runtime(settings_override or settings, model=model, email_transport=email_transport, clock=clock)

    return _make


@pytest.fixture
def harness(make_runtime: Callable[..., Any]) -> Callable[..., Harness]:
    def _make(model: Any = None, email_transport: Any = None, settings_override: Settings | None = None) -> Harness:
        return Harness(make_runtime(model=model, email_transport=email_transport, settings_override=settings_override))

    return _make
