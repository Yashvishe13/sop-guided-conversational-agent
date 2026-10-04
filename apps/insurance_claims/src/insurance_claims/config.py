"""Server-side configuration.

Everything security relevant (API token, model, budgets, storage paths, email
transport) lives here and only here. Nothing in this module is ever sent to
the browser. The API token is kept out of ``repr`` so it cannot leak through
logs, tracebacks, or trace files.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field, replace
from datetime import date
from pathlib import Path
from typing import Literal, Mapping

APP_ROOT = Path(__file__).resolve().parents[2]
"""``apps/insurance_claims`` in a checkout, ``/app`` in the Docker image."""


class ConfigError(RuntimeError):
    """Raised when the server cannot start because configuration is missing or invalid."""


ModelProvider = Literal["openai", "fake"]
EmailTransportName = Literal["outbox", "smtp", "disabled"]
Environment = Literal["development", "production", "test"]


def _bool(raw: str | None, default: bool) -> bool:
    if raw is None or raw.strip() == "":
        return default
    value = raw.strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    raise ConfigError(f"Expected a boolean value, got {raw!r}")


def _int(raw: str | None, default: int, *, minimum: int = 0) -> int:
    if raw is None or raw.strip() == "":
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ConfigError(f"Expected an integer, got {raw!r}") from exc
    if value < minimum:
        raise ConfigError(f"Expected an integer >= {minimum}, got {value}")
    return value


def _float(raw: str | None, default: float, *, minimum: float = 0.0) -> float:
    if raw is None or raw.strip() == "":
        return default
    try:
        value = float(raw)
    except ValueError as exc:
        raise ConfigError(f"Expected a number, got {raw!r}") from exc
    if value < minimum:
        raise ConfigError(f"Expected a number >= {minimum}, got {value}")
    return value


def _path(raw: str | None, default: Path) -> Path:
    if raw is None or raw.strip() == "":
        return default
    return Path(raw).expanduser().resolve()


_INLINE_COMMENT = re.compile(r"\s#")


def _dotenv_value(raw: str) -> str:
    """Parse the right-hand side of a ``.env`` line the way docker compose and python-dotenv do.

    A quoted value ends at its closing quote and may be followed by a ``# comment``. In an unquoted
    value, ``#`` starts a comment only when whitespace precedes it (``KEY=abc  # note`` is ``abc``,
    ``KEY=a#b`` stays ``a#b``). Quote a value that must contain `` #``.
    """
    value = raw.strip()
    if value[:1] in {"'", '"'}:
        quote = value[0]
        end = value.find(quote, 1)
        while end != -1:
            rest = value[end + 1 :].strip()
            if not rest or rest.startswith("#"):
                return value[1:end]
            end = value.find(quote, end + 1)
    return _INLINE_COMMENT.split(value, maxsplit=1)[0].strip()


def load_dotenv_file(path: Path, env: dict[str, str]) -> None:
    """Minimal ``.env`` reader: KEY=VALUE lines, ``#`` comments (whole-line and inline), optional quotes.

    A non-empty value already present in ``env`` wins, so a real process environment variable
    always overrides the file; an empty one (``OPENAI_API_KEY=`` exported by a shell or compose)
    counts as unset and the file fills it. The file content is never logged.
    """
    if not path.is_file():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export ") :]
        key, _, raw_value = line.partition("=")
        key = key.strip()
        if key and not (env.get(key) or "").strip():
            env[key] = _dotenv_value(raw_value)


@dataclass(frozen=True)
class Settings:
    # --- model -----------------------------------------------------------
    openai_api_key: str | None = field(default=None, repr=False)
    openai_model: str = "gpt-5.6-luna"
    model_provider: ModelProvider = "openai"
    reasoning_effort_reply: str = "low"
    """Reasoning effort for every agent step (OPENAI_REASONING_EFFORT_REPLY)."""
    model_timeout_s: float = 25.0
    model_max_retries: int = 2
    model_backoff_base_s: float = 0.5
    max_output_tokens_reply: int = 1500
    reasoning_effort_guard: str = "medium"
    """Reasoning effort for the guard model's verdicts (OPENAI_REASONING_EFFORT_GUARD)."""
    max_output_tokens_guard: int = 2000

    # --- per-turn budgets -------------------------------------------------
    turn_deadline_s: float = 150.0
    max_tool_calls_per_turn: int = 6
    """Tool calls per turn; the agent adds a small allowance for verification and lookups."""

    # --- SOP policy knobs -------------------------------------------------
    max_off_topic: int = 3
    """Off-topic turns in a session before a human representative is offered."""
    max_refusals: int = 2
    """Verification refusals before the agent stops persuading and offers a human."""
    max_verification_failures: int = 3
    """Failed full verification attempts before identity checks lock for the session."""
    max_party_failures: int = 5
    """Failed verification attempts against one policyholder (across all sessions) before that party locks."""
    party_failure_window_hours: int = 24
    max_turns_per_session: int = 80
    history_limit: int = 40
    max_message_chars: int = 2000

    # --- sessions ---------------------------------------------------------
    verification_idle_ttl_minutes: int = 30
    """Idle time after which a verified caller must verify again."""
    session_max_age_hours: int = 24
    """After this, a conversation can no longer be resumed."""
    retention_days: int = 7
    """Stored sessions, turns, and email ledger rows older than this are purged."""
    cookie_secure: bool = False
    frozen_today: date | None = None
    """Inject a fixed 'today' (APP_TODAY=YYYY-MM-DD) for demos and tests."""

    # --- storage ----------------------------------------------------------
    app_root: Path = APP_ROOT
    fixtures_dir: Path = APP_ROOT / "fixtures"
    prompts_path: Path = APP_ROOT / "prompts.toml"
    sop_path: Path = APP_ROOT / "sop.toml"
    """The SOP definition: phase order, tools per phase, transitions (SOP_PATH)."""
    data_dir: Path = APP_ROOT / "data"
    state_encryption_key: str | None = field(default=None, repr=False)
    traces_enabled: bool = True
    trace_dir: Path = APP_ROOT / "traces"

    # --- email ------------------------------------------------------------
    email_transport: EmailTransportName = "outbox"
    email_from: str = "claims-support@example.com"
    email_timeout_s: float = 10.0
    smtp_host: str | None = None
    smtp_port: int = 587
    smtp_username: str | None = None
    smtp_password: str | None = field(default=None, repr=False)
    smtp_starttls: bool = True

    environment: Environment = "development"

    # ---------------------------------------------------------------------
    @property
    def db_path(self) -> Path:
        return self.data_dir / "sessions.sqlite3"

    @property
    def outbox_dir(self) -> Path:
        return self.data_dir / "outbox"

    @property
    def key_file(self) -> Path:
        return self.data_dir / "state.key"

    def with_overrides(self, **changes: object) -> Settings:
        return replace(self, **changes)  # type: ignore[arg-type]

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None, *, dotenv: Path | None = None) -> Settings:
        """Build settings from environment variables (and optionally a ``.env`` file).

        Recognised variables are documented in the README. Unknown variables are ignored.
        """
        merged: dict[str, str] = dict(os.environ if env is None else env)
        if dotenv is not None:
            load_dotenv_file(dotenv, merged)
        get = merged.get

        provider = (get("MODEL_PROVIDER") or "openai").strip().lower()
        if provider not in {"openai", "fake"}:
            raise ConfigError("MODEL_PROVIDER must be 'openai' or 'fake'")
        transport = (get("EMAIL_TRANSPORT") or "outbox").strip().lower()
        if transport not in {"outbox", "smtp", "disabled"}:
            raise ConfigError("EMAIL_TRANSPORT must be 'outbox', 'smtp', or 'disabled'")
        environment = (get("APP_ENV") or "development").strip().lower()
        if environment not in {"development", "production", "test"}:
            raise ConfigError("APP_ENV must be 'development', 'production', or 'test'")
        today_raw = (get("APP_TODAY") or "").strip()
        try:
            frozen_today = date.fromisoformat(today_raw) if today_raw else None
        except ValueError as exc:
            raise ConfigError("APP_TODAY must be an ISO date like 2026-10-03") from exc

        data_dir = _path(get("DATA_DIR"), APP_ROOT / "data")
        return cls(
            openai_api_key=(get("OPENAI_API_KEY") or "").strip() or None,
            openai_model=(get("OPENAI_MODEL") or "gpt-5.6-luna").strip(),
            model_provider=provider,  # type: ignore[arg-type]
            reasoning_effort_reply=(get("OPENAI_REASONING_EFFORT_REPLY") or "low").strip(),
            reasoning_effort_guard=(get("OPENAI_REASONING_EFFORT_GUARD") or "medium").strip(),
            model_timeout_s=_float(get("OPENAI_TIMEOUT_S"), 25.0, minimum=1.0),
            model_max_retries=_int(get("OPENAI_MAX_RETRIES"), 2),
            turn_deadline_s=_float(get("TURN_DEADLINE_S"), 150.0, minimum=5.0),
            max_tool_calls_per_turn=_int(get("MAX_TOOL_CALLS_PER_TURN"), 6, minimum=0),
            max_off_topic=_int(get("MAX_OFF_TOPIC"), 3, minimum=1),
            max_refusals=_int(get("MAX_REFUSALS"), 2, minimum=1),
            max_verification_failures=_int(get("MAX_VERIFICATION_FAILURES"), 3, minimum=1),
            max_party_failures=_int(get("MAX_PARTY_FAILURES"), 5, minimum=1),
            party_failure_window_hours=_int(get("PARTY_FAILURE_WINDOW_HOURS"), 24, minimum=1),
            max_turns_per_session=_int(get("MAX_TURNS_PER_SESSION"), 80, minimum=5),
            history_limit=_int(get("HISTORY_LIMIT"), 40, minimum=4),
            max_message_chars=_int(get("MAX_MESSAGE_CHARS"), 2000, minimum=50),
            verification_idle_ttl_minutes=_int(get("VERIFICATION_IDLE_TTL_MINUTES"), 30, minimum=1),
            session_max_age_hours=_int(get("SESSION_MAX_AGE_HOURS"), 24, minimum=1),
            retention_days=_int(get("RETENTION_DAYS"), 7, minimum=1),
            cookie_secure=_bool(get("COOKIE_SECURE"), environment == "production"),
            frozen_today=frozen_today,
            fixtures_dir=_path(get("FIXTURES_DIR"), APP_ROOT / "fixtures"),
            prompts_path=_path(get("PROMPTS_PATH"), APP_ROOT / "prompts.toml"),
            sop_path=_path(get("SOP_PATH"), APP_ROOT / "sop.toml"),
            data_dir=data_dir,
            state_encryption_key=(get("STATE_ENCRYPTION_KEY") or "").strip() or None,
            traces_enabled=_bool(get("TRACES_ENABLED"), True),
            trace_dir=_path(get("TRACE_DIR"), APP_ROOT / "traces"),
            email_transport=transport,  # type: ignore[arg-type]
            email_from=(get("EMAIL_FROM") or "claims-support@example.com").strip(),
            email_timeout_s=_float(get("EMAIL_TIMEOUT_S"), 10.0, minimum=1.0),
            smtp_host=(get("SMTP_HOST") or "").strip() or None,
            smtp_port=_int(get("SMTP_PORT"), 587, minimum=1),
            smtp_username=(get("SMTP_USERNAME") or "").strip() or None,
            smtp_password=(get("SMTP_PASSWORD") or "") or None,
            smtp_starttls=_bool(get("SMTP_STARTTLS"), True),
            environment=environment,  # type: ignore[arg-type]
        )

    def validate_runtime(self) -> None:
        """Fail fast, with a clear message, when the server cannot run safely."""
        if self.model_provider == "openai" and not self.openai_api_key:
            raise ConfigError(
                "OPENAI_API_KEY is not set. Put it in the repository .env file or pass it as an "
                "environment variable (docker compose reads it at runtime). Set MODEL_PROVIDER=fake "
                "to run the offline demo model instead."
            )
        if self.email_transport == "smtp" and not self.smtp_host:
            raise ConfigError("EMAIL_TRANSPORT=smtp requires SMTP_HOST")
        if self.environment == "production" and not self.state_encryption_key:
            raise ConfigError("APP_ENV=production requires STATE_ENCRYPTION_KEY (a Fernet key)")
        if not self.fixtures_dir.is_dir():
            raise ConfigError(f"Fixture directory not found: {self.fixtures_dir}")
        if not self.prompts_path.is_file():
            raise ConfigError(f"Prompt file not found: {self.prompts_path}")
        if not self.sop_path.is_file():
            raise ConfigError(f"SOP file not found: {self.sop_path}")
