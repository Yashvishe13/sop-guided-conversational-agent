"""Versioned prompt text, loaded and validated once at startup.

Prompt text lives only in ``prompts.toml``. This module reads that file,
checks that every required entry exists and is a non-empty string, and
assembles the instructions for one model call. Traces should record only
``PromptSet.version`` and ``PromptSet.sha256``, never the prompt text.

TOML layout::

    version = "..."
    [global]            guideline = "..."
    [style]             guideline = "..."
    [phases.<PHASE>]    guideline = "..."   (one table per workflow phase)
    [tasks.<name>]      guideline = "..."   (one table per model task)
"""

from __future__ import annotations

import hashlib
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

from insurance_claims.domain.models import Phase

__all__ = [
    "GUARD_TASKS",
    "MAX_PROMPT_FILE_BYTES",
    "REQUIRED_PHASES",
    "REQUIRED_TASKS",
    "PromptError",
    "PromptSet",
    "load_prompts",
]


class PromptError(RuntimeError):
    """The prompt file is missing, malformed, or incomplete (startup must fail)."""


REQUIRED_PHASES: tuple[str, ...] = ("VERIFY_ID", "RESOLVE_INTENT", "PROCESS_CASE", "POST_PROCESS")
GUARD_TASKS: tuple[str, ...] = ("guard_caller", "guard_consent", "guard_summary", "guard_document", "guard_reply")
REQUIRED_TASKS: tuple[str, ...] = ("agent", *GUARD_TASKS)
KNOWN_PHASES: frozenset[str] = frozenset(p.value for p in Phase)
MAX_PROMPT_FILE_BYTES = 512 * 1024


@dataclass(frozen=True)
class PromptSet:
    """Validated, immutable prompt text plus its version and file hash."""

    version: str
    sha256: str
    global_guideline: str
    style_guideline: str
    phases: Mapping[str, str]
    tasks: Mapping[str, str]

    def guard_instructions(self, task: str) -> str:
        """Instructions for one guard checkpoint: the guard prompt alone, without the agent's prompts."""
        if task not in GUARD_TASKS:
            raise PromptError(f"Unknown guard task {task!r}")
        return self.tasks[task]

    def instructions(self, task: str, phase: str) -> str:
        """Assemble the instructions for one model call (global, style, phase, task)."""
        task_key, phase_key = str(task), str(phase)
        task_text = self.tasks.get(task_key)
        if task_text is None:
            raise PromptError(f"Unknown prompt task {task_key!r}")
        phase_text = self.phases.get(phase_key)
        if phase_text is None:
            raise PromptError(f"Unknown prompt phase {phase_key!r}")
        return (
            f"# Global guideline\n{self.global_guideline}\n\n"
            f"# Response style\n{self.style_guideline}\n\n"
            f"# Current phase {phase_key}\n{phase_text}\n\n"
            f"# Task\n{task_text}"
        )


def load_prompts(path: Path) -> PromptSet:
    """Load and validate ``prompts.toml``; raise ``PromptError`` naming any problem."""
    raw = _read_bytes(Path(path))
    document = _parse(raw)
    return PromptSet(
        version=_required_string(document, "version", "version"),
        sha256=hashlib.sha256(raw).hexdigest(),
        global_guideline=_guideline(document, "global"),
        style_guideline=_guideline(document, "style"),
        phases=_guideline_section(document, "phases", REQUIRED_PHASES, allowed=KNOWN_PHASES),
        tasks=_guideline_section(document, "tasks", REQUIRED_TASKS, allowed=None),
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _read_bytes(path: Path) -> bytes:
    """Read the file, failing clearly on absence, unreadability, or excessive size."""
    try:
        size = path.stat().st_size
    except FileNotFoundError:
        raise PromptError(f"Prompt file not found: {path}") from None
    except OSError as exc:
        raise PromptError(f"Cannot access prompt file {path}: {exc.strerror}") from None
    if not path.is_file():
        raise PromptError(f"Prompt path is not a file: {path}")
    if size > MAX_PROMPT_FILE_BYTES:
        raise PromptError(f"Prompt file is too large ({size} bytes, limit {MAX_PROMPT_FILE_BYTES})")
    try:
        return path.read_bytes()
    except OSError as exc:
        raise PromptError(f"Cannot read prompt file {path}: {exc.strerror}") from None


def _parse(raw: bytes) -> dict[str, Any]:
    """Decode UTF-8 and parse TOML."""
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        raise PromptError("Prompt file is not valid UTF-8") from None
    try:
        return tomllib.loads(text)
    except tomllib.TOMLDecodeError as exc:
        raise PromptError(f"Malformed TOML in prompt file: {exc}") from None


def _required_string(container: Mapping[str, Any], key: str, dotted: str) -> str:
    """Return ``container[key]`` stripped; it must exist, be a string, and be non-empty."""
    if key not in container:
        raise PromptError(f"Prompt file is missing required entry '{dotted}'")
    value = container[key]
    if not isinstance(value, str):
        raise PromptError(f"Prompt entry '{dotted}' must be a string, got {type(value).__name__}")
    text = value.strip()
    if not text:
        raise PromptError(f"Prompt entry '{dotted}' is empty")
    return text


def _table(container: Mapping[str, Any], key: str, dotted: str) -> Mapping[str, Any]:
    """Return ``container[key]``; it must exist and be a TOML table."""
    if key not in container:
        raise PromptError(f"Prompt file is missing required table '{dotted}'")
    value = container[key]
    if not isinstance(value, dict):
        raise PromptError(f"Prompt entry '{dotted}' must be a table, got {type(value).__name__}")
    return value


def _guideline(container: Mapping[str, Any], key: str, prefix: str = "") -> str:
    """Return the ``guideline`` string of the table ``container[key]``."""
    dotted = f"{prefix}{key}"
    table = _table(container, key, dotted)
    return _required_string(table, "guideline", f"{dotted}.guideline")


def _guideline_section(
    document: Mapping[str, Any],
    section: str,
    required: tuple[str, ...],
    *,
    allowed: frozenset[str] | None,
) -> Mapping[str, str]:
    """Validate ``[section.<name>]`` tables: every required name present, every entry well formed."""
    tables = _table(document, section, section)
    for name in required:
        if name not in tables:
            raise PromptError(f"Prompt file is missing required entry '{section}.{name}.guideline'")
    if allowed is not None:
        unknown = sorted(set(tables) - allowed)
        if unknown:
            raise PromptError(f"Prompt file has unknown {section} entries: {', '.join(unknown)}")
    texts = {name: _guideline(tables, name, f"{section}.") for name in tables}
    return MappingProxyType(texts)
