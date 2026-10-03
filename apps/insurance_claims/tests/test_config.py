"""Configuration loading (.env parsing) and the documented deployment files (compose, .env.example, README)."""

from __future__ import annotations

import re
from datetime import date
from pathlib import Path

import pytest

from insurance_claims.config import APP_ROOT, Settings, load_dotenv_file

REPO_ROOT = APP_ROOT.parents[1]
COMPOSE = REPO_ROOT / "compose.yaml"
ENV_EXAMPLE = REPO_ROOT / ".env.example"
README = APP_ROOT / "README.md"


def _parse(tmp_path: Path, text: str, env: dict[str, str] | None = None) -> dict[str, str]:
    path = tmp_path / ".env"
    path.write_text(text, encoding="utf-8")
    merged = dict(env or {})
    load_dotenv_file(path, merged)
    return merged


# --- .env parsing (L18) ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("line", "key", "expected"),
    [
        # The optional lines of the original .env.example, uncommented as the README suggests.
        ('MODEL_PROVIDER=fake        # or "fake" for the offline deterministic demo model', "MODEL_PROVIDER", "fake"),
        ("EMAIL_TRANSPORT=outbox       # outbox (local demo, nothing delivered) | smtp | disabled", "EMAIL_TRANSPORT", "outbox"),
        ('APP_TODAY=2026-10-03         # freeze "today" for demos', "APP_TODAY", "2026-10-03"),
        ("KEY=value\t# tab before the comment", "KEY", "value"),
        ("KEY=a#b", "KEY", "a#b"),
        ("KEY=sk-abc#def  # comment", "KEY", "sk-abc#def"),
        ('KEY="quoted # not a comment"', "KEY", "quoted # not a comment"),
        ('KEY="quoted"  # trailing comment', "KEY", "quoted"),
        ("KEY='single'  # trailing comment", "KEY", "single"),
        ('KEY="a"b"', "KEY", 'a"b'),
        ("export KEY=exported  # comment", "KEY", "exported"),
        ("KEY=", "KEY", ""),
        ("KEY=   ", "KEY", ""),
        ("KEY=#literal", "KEY", "#literal"),
    ],
)
def test_dotenv_values_drop_inline_comments(tmp_path, line, key, expected):
    assert _parse(tmp_path, line + "\n")[key] == expected


def test_uncommented_example_lines_start_the_app(tmp_path):
    text = (
        "OPENAI_API_KEY=sk-test-123\n"
        '# MODEL_PROVIDER=openai        # or "fake" for the offline deterministic demo model\n'
        'MODEL_PROVIDER=fake        # or "fake" for the offline deterministic demo model\n'
        "EMAIL_TRANSPORT=disabled       # outbox (local demo, nothing delivered) | smtp | disabled\n"
        'APP_TODAY=2026-10-03         # freeze "today" for demos\n'
    )
    path = tmp_path / ".env"
    path.write_text(text, encoding="utf-8")
    settings = Settings.from_env({}, dotenv=path)
    assert settings.model_provider == "fake"
    assert settings.email_transport == "disabled"
    assert settings.frozen_today == date(2026, 10, 3)
    assert settings.openai_api_key == "sk-test-123"


def test_process_environment_wins_but_empty_values_count_as_unset(tmp_path):
    merged = _parse(
        tmp_path,
        "OPENAI_API_KEY=sk-from-file\nMODEL_PROVIDER=fake\nAPP_TODAY=2026-10-03\n",
        env={"OPENAI_API_KEY": "", "MODEL_PROVIDER": "openai", "APP_TODAY": "  "},
    )
    assert merged["OPENAI_API_KEY"] == "sk-from-file"  # an empty shell export no longer shadows .env
    assert merged["MODEL_PROVIDER"] == "openai"  # a real value still overrides the file
    assert merged["APP_TODAY"] == "2026-10-03"


def test_env_example_parses_with_every_optional_line_uncommented(tmp_path):
    lines = []
    for raw in ENV_EXAMPLE.read_text(encoding="utf-8").splitlines():
        match = re.fullmatch(r"#\s?([A-Z][A-Z0-9_]*=.*)", raw.strip())
        lines.append(match.group(1) if match else raw)
    path = tmp_path / ".env"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    env: dict[str, str] = {}
    load_dotenv_file(path, env)
    assert {"OPENAI_API_KEY", "STATE_ENCRYPTION_KEY", "MODEL_PROVIDER", "EMAIL_TRANSPORT", "APP_TODAY"} <= env.keys()
    settings = Settings.from_env({}, dotenv=path)  # raises ConfigError on a malformed value
    assert settings.model_provider == "openai" and settings.email_transport == "outbox"
    assert settings.frozen_today == date(2026, 10, 3)


# --- deployment files (C46, L19) ------------------------------------------------------------------


def _compose_env() -> dict[str, str]:
    """``KEY: value`` pairs under the claims-agent ``environment:`` block (no YAML dependency)."""
    text = COMPOSE.read_text(encoding="utf-8")
    block = text.split("environment:", 1)[1].split("volumes:", 1)[0]
    pairs = re.findall(r"^\s+([A-Z][A-Z0-9_]*):\s*(.+?)\s*$", block, flags=re.MULTILINE)
    return dict(pairs)


def test_compose_state_key_is_optional():
    env = _compose_env()
    # Optional for the demo: when unset the app generates a key file inside /data.
    assert env["STATE_ENCRYPTION_KEY"] == "${STATE_ENCRYPTION_KEY:-}"
    # APP_ENV reaches the container so the production guard can be switched on from .env.
    assert env["APP_ENV"].startswith("${APP_ENV")


def test_compose_runs_the_fake_model_without_an_openai_key():
    env = _compose_env()
    # The app's validate_runtime enforces the key only for MODEL_PROVIDER=openai.
    assert env["OPENAI_API_KEY"] == "${OPENAI_API_KEY:-}"
    assert env["MODEL_PROVIDER"] == "${MODEL_PROVIDER:-openai}"


def test_compose_empty_openai_key_still_fails_clearly_for_openai(settings):
    from insurance_claims.config import ConfigError

    with pytest.raises(ConfigError, match="MODEL_PROVIDER=fake"):
        settings.with_overrides(openai_api_key=None, model_provider="openai").validate_runtime()
    settings.with_overrides(openai_api_key=None, model_provider="fake").validate_runtime()


def test_env_example_and_readme_document_the_state_key():
    example = ENV_EXAMPLE.read_text(encoding="utf-8")
    assert re.search(r"^STATE_ENCRYPTION_KEY=", example, flags=re.MULTILINE)
    assert "urlsafe_b64encode" in example
    readme = README.read_text(encoding="utf-8")
    assert "STATE_ENCRYPTION_KEY" in readme and "urlsafe_b64encode" in readme
    assert "docker compose down -v" in readme
    for name in ("MAX_PARTY_FAILURES", "PARTY_FAILURE_WINDOW_HOURS", "APP_ENV"):
        assert f"`{name}`" in readme, name
    assert "-m browser" in readme and "PYTHONPATH=src" in readme and "chflags nohidden" in readme


def test_readme_relative_links_resolve():
    readme = README.read_text(encoding="utf-8")
    targets = re.findall(r"\]\((?!https?://|#)([^)#\s]+)", readme)
    assert "docs/RESULTS.md" in targets  # the results report stays linked
    missing = [t for t in targets if not (README.parent / t).exists()]
    assert not missing, f"README links to missing files: {missing}"
