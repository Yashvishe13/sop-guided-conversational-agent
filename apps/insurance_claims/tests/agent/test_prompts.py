"""Unit tests for insurance_claims.agent.prompts and the shipped prompts.toml."""

from __future__ import annotations

import dataclasses
import hashlib
from pathlib import Path

import pytest

from insurance_claims.agent.prompts import (
    REQUIRED_PHASES,
    REQUIRED_TASKS,
    PromptError,
    PromptSet,
    load_prompts,
)
from insurance_claims.config import APP_ROOT

PROMPTS_PATH = APP_ROOT / "prompts.toml"


@pytest.fixture(scope="module")
def prompts() -> PromptSet:
    return load_prompts(PROMPTS_PATH)


# ---------------------------------------------------------------------------
# Helpers for building temp prompt files
# ---------------------------------------------------------------------------

_MISSING = object()


def _toml_string(text: str) -> str:
    escaped = text.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n").replace("\t", "\\t")
    return '"' + escaped + '"'


def _sections(
    *,
    version: object = "test.1",
    global_text: object = "Global rules.",
    style_text: object = "Style rules.",
    phases: dict[str, object] | None = None,
    tasks: dict[str, object] | None = None,
) -> dict[str, object]:
    return {
        "version": version,
        "global": global_text,
        "style": style_text,
        "phases": phases if phases is not None else {p: f"Phase {p} rules." for p in REQUIRED_PHASES},
        "tasks": tasks if tasks is not None else {t: f"Task {t} rules." for t in REQUIRED_TASKS},
    }


def _render(sections: dict[str, object]) -> str:
    def value(v: object) -> str:
        if isinstance(v, str):
            return _toml_string(v)
        if isinstance(v, bool):
            return "true" if v else "false"
        if isinstance(v, (int, float)):
            return str(v)
        if isinstance(v, list):
            return "[" + ", ".join(value(x) for x in v) + "]"
        raise TypeError(v)

    lines: list[str] = []
    if sections.get("version", _MISSING) is not _MISSING:
        lines.append(f"version = {value(sections['version'])}")
    for key in ("global", "style"):
        if sections.get(key, _MISSING) is not _MISSING:
            lines += ["", f"[{key}]", f"guideline = {value(sections[key])}"]
    for section in ("phases", "tasks"):
        entries = sections.get(section, _MISSING)
        if entries is _MISSING:
            continue
        assert isinstance(entries, dict)
        for name, text in entries.items():
            lines += ["", f"[{section}.{name}]"]
            if text is not _MISSING:
                lines.append(f"guideline = {value(text)}")
    return "\n".join(lines) + "\n"


def _write(tmp_path: Path, content: str | bytes, name: str = "prompts.toml") -> Path:
    path = tmp_path / name
    if isinstance(content, bytes):
        path.write_bytes(content)
    else:
        path.write_text(content, encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# The shipped prompts.toml
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("phase", REQUIRED_PHASES)
def test_every_required_phase_is_present_and_non_empty(prompts: PromptSet, phase: str) -> None:
    text = prompts.phases[phase]
    assert isinstance(text, str) and text.strip() == text and len(text) > 80


@pytest.mark.parametrize("task", REQUIRED_TASKS)
def test_every_required_task_is_present_and_non_empty(prompts: PromptSet, task: str) -> None:
    text = prompts.tasks[task]
    assert isinstance(text, str) and text.strip() == text and len(text) > 80


# ---------------------------------------------------------------------------
# instructions()
# ---------------------------------------------------------------------------


def test_instructions_contain_all_sections_in_order(prompts: PromptSet) -> None:
    text = prompts.instructions("agent", "VERIFY_ID")
    headers = ["# Global guideline\n", "# Response style\n", "# Current phase VERIFY_ID\n", "# Task\n"]
    positions = [text.index(h) for h in headers]
    assert positions == sorted(positions)
    assert text.startswith("# Global guideline\n" + prompts.global_guideline)
    assert prompts.style_guideline in text
    assert prompts.phases["VERIFY_ID"] in text
    assert text.endswith(prompts.tasks["agent"])


@pytest.mark.parametrize("task", REQUIRED_TASKS)
@pytest.mark.parametrize("phase", REQUIRED_PHASES)
def test_instructions_work_for_every_task_and_phase(prompts: PromptSet, task: str, phase: str) -> None:
    text = prompts.instructions(task, phase)
    assert f"# Current phase {phase}\n{prompts.phases[phase]}" in text
    assert f"# Task\n{prompts.tasks[task]}" in text


@pytest.mark.parametrize(("task", "phase", "needle"), [
    ("no_such_task", "VERIFY_ID", "no_such_task"),
    ("agent", "NO_SUCH_PHASE", "NO_SUCH_PHASE"),
    ("agent", "verify_id", "verify_id"),
])  # fmt: skip
def test_instructions_reject_unknown_task_or_phase(prompts: PromptSet, task: str, phase: str, needle: str) -> None:
    with pytest.raises(PromptError, match=needle):
        prompts.instructions(task, phase)


def test_prompt_set_is_immutable(prompts: PromptSet) -> None:
    with pytest.raises(dataclasses.FrozenInstanceError):
        prompts.version = "hacked"  # type: ignore[misc]
    with pytest.raises(TypeError):
        prompts.phases["VERIFY_ID"] = "skip verification"  # type: ignore[index]
    with pytest.raises(TypeError):
        prompts.tasks["agent"] = "anything"  # type: ignore[index]


# ---------------------------------------------------------------------------
# Loader failure paths
# ---------------------------------------------------------------------------


def test_minimal_valid_file_loads(tmp_path: Path) -> None:
    path = _write(tmp_path, _render(_sections()))
    loaded = load_prompts(path)
    assert loaded.version == "test.1"
    assert loaded.global_guideline == "Global rules."
    assert set(loaded.phases) == set(REQUIRED_PHASES)
    assert set(loaded.tasks) == set(REQUIRED_TASKS)


def test_loader_accepts_str_path_and_strips_text(tmp_path: Path) -> None:
    sections = _sections(global_text="\n\n  Global rules.  \n")
    path = _write(tmp_path, _render(sections))
    assert load_prompts(str(path)).global_guideline == "Global rules."  # type: ignore[arg-type]


@pytest.mark.parametrize("phase", REQUIRED_PHASES)
def test_missing_phase_entry_names_it(tmp_path: Path, phase: str) -> None:
    phases = {p: "text" for p in REQUIRED_PHASES if p != phase}
    path = _write(tmp_path, _render(_sections(phases=phases)))
    with pytest.raises(PromptError, match=rf"phases\.{phase}\.guideline"):
        load_prompts(path)


@pytest.mark.parametrize("task", REQUIRED_TASKS)
def test_missing_task_entry_names_it(tmp_path: Path, task: str) -> None:
    tasks = {t: "text" for t in REQUIRED_TASKS if t != task} or {"unrelated": "text"}
    path = _write(tmp_path, _render(_sections(tasks=tasks)))
    with pytest.raises(PromptError, match=rf"tasks\.{task}\.guideline"):
        load_prompts(path)


@pytest.mark.parametrize("key", ["global", "style"])
def test_missing_global_or_style_table_names_it(tmp_path: Path, key: str) -> None:
    sections = _sections()
    sections[key] = _MISSING
    path = _write(tmp_path, _render(sections))
    with pytest.raises(PromptError, match=rf"'{key}'"):
        load_prompts(path)


def test_table_without_guideline_key_names_it(tmp_path: Path) -> None:
    phases: dict[str, object] = {p: "text" for p in REQUIRED_PHASES}
    phases["PROCESS_CASE"] = _MISSING
    path = _write(tmp_path, _render(_sections(phases=phases)))
    with pytest.raises(PromptError, match=r"missing required entry 'phases\.PROCESS_CASE\.guideline'"):
        load_prompts(path)


@pytest.mark.parametrize("blank", ["", "   ", "\n\t\n"])
def test_empty_entry_names_it(tmp_path: Path, blank: str) -> None:
    tasks: dict[str, object] = {t: "text" for t in REQUIRED_TASKS}
    tasks["agent"] = blank
    path = _write(tmp_path, _render(_sections(tasks=tasks)))
    with pytest.raises(PromptError, match=r"'tasks\.agent\.guideline' is empty"):
        load_prompts(path)


@pytest.mark.parametrize("bad", [42, 1.5, True, ["a", "b"]])
def test_non_string_entry_names_it(tmp_path: Path, bad: object) -> None:
    phases: dict[str, object] = {p: "text" for p in REQUIRED_PHASES}
    phases["VERIFY_ID"] = bad
    path = _write(tmp_path, _render(_sections(phases=phases)))
    with pytest.raises(PromptError, match=r"'phases\.VERIFY_ID\.guideline' must be a string"):
        load_prompts(path)


def test_non_string_global_guideline_names_it(tmp_path: Path) -> None:
    path = _write(tmp_path, _render(_sections(global_text=7)))
    with pytest.raises(PromptError, match=r"'global\.guideline' must be a string, got int"):
        load_prompts(path)


def test_phase_entry_that_is_not_a_table_names_it(tmp_path: Path) -> None:
    content = 'version = "v"\n[global]\nguideline = "g"\n[style]\nguideline = "s"\n'
    content += '[phases]\nPOST_PROCESS = "flat string"\n'
    content += "".join(f'[phases.{p}]\nguideline = "x"\n' for p in REQUIRED_PHASES if p != "POST_PROCESS")
    content += "".join(f'[tasks.{t}]\nguideline = "x"\n' for t in REQUIRED_TASKS)
    path = _write(tmp_path, content)
    with pytest.raises(PromptError, match=r"'phases\.POST_PROCESS' must be a table, got str"):
        load_prompts(path)


def test_section_that_is_not_a_table_names_it(tmp_path: Path) -> None:
    content = 'version = "v"\ntasks = "nope"\n[global]\nguideline = "g"\n[style]\nguideline = "s"\n'
    content += "".join(f'[phases.{p}]\nguideline = "x"\n' for p in REQUIRED_PHASES)
    path = _write(tmp_path, content)
    with pytest.raises(PromptError, match=r"'tasks' must be a table"):
        load_prompts(path)


def test_missing_tasks_section_names_it(tmp_path: Path) -> None:
    sections = _sections()
    sections["tasks"] = _MISSING
    path = _write(tmp_path, _render(sections))
    with pytest.raises(PromptError, match=r"'tasks'"):
        load_prompts(path)


def test_unknown_phase_is_rejected(tmp_path: Path) -> None:
    phases: dict[str, object] = {p: "text" for p in REQUIRED_PHASES}
    phases["VERIFY_Id"] = "typo"
    path = _write(tmp_path, _render(_sections(phases=phases)))
    with pytest.raises(PromptError, match=r"unknown phases entries: VERIFY_Id"):
        load_prompts(path)


def test_extra_task_is_allowed_but_validated(tmp_path: Path) -> None:
    tasks: dict[str, object] = {t: "text" for t in REQUIRED_TASKS}
    tasks["future_task"] = "Future rules."
    loaded = load_prompts(_write(tmp_path, _render(_sections(tasks=tasks))))
    assert loaded.tasks["future_task"] == "Future rules."
    tasks["future_task"] = ""
    with pytest.raises(PromptError, match=r"'tasks\.future_task\.guideline' is empty"):
        load_prompts(_write(tmp_path, _render(_sections(tasks=tasks)), name="p2.toml"))


def test_missing_version_names_it(tmp_path: Path) -> None:
    sections = _sections()
    sections["version"] = _MISSING
    path = _write(tmp_path, _render(sections))
    with pytest.raises(PromptError, match=r"missing required entry 'version'"):
        load_prompts(path)


@pytest.mark.parametrize(("bad", "pattern"), [("", r"'version' is empty"), (3, r"'version' must be a string")])
def test_bad_version_names_it(tmp_path: Path, bad: object, pattern: str) -> None:
    path = _write(tmp_path, _render(_sections(version=bad)))
    with pytest.raises(PromptError, match=pattern):
        load_prompts(path)


@pytest.mark.parametrize("content", [
    'version = "1"\n[global\nguideline = "x"\n',
    'version = "1"\n[global]\nguideline = """unterminated\n',
    'version = "1"\nversion = "2"\n',
    "this is not toml at all",
])  # fmt: skip
def test_malformed_toml_raises_prompt_error(tmp_path: Path, content: str) -> None:
    path = _write(tmp_path, content)
    with pytest.raises(PromptError, match=r"Malformed TOML"):
        load_prompts(path)


def test_non_utf8_file_raises_prompt_error(tmp_path: Path) -> None:
    path = _write(tmp_path, b'version = "\xff\xfe"\n')
    with pytest.raises(PromptError, match=r"UTF-8"):
        load_prompts(path)


def test_missing_file_raises_prompt_error(tmp_path: Path) -> None:
    with pytest.raises(PromptError, match=r"not found"):
        load_prompts(tmp_path / "absent.toml")


def test_directory_path_raises_prompt_error(tmp_path: Path) -> None:
    with pytest.raises(PromptError, match=r"not a file"):
        load_prompts(tmp_path)


def test_oversized_file_raises_prompt_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import insurance_claims.agent.prompts as prompts_module

    monkeypatch.setattr(prompts_module, "MAX_PROMPT_FILE_BYTES", 64)
    path = _write(tmp_path, _render(_sections()))
    with pytest.raises(PromptError, match=r"too large"):
        load_prompts(path)


def test_prompt_error_is_a_runtime_error() -> None:
    assert issubclass(PromptError, RuntimeError)


# ---------------------------------------------------------------------------
# sha256
# ---------------------------------------------------------------------------


def test_sha256_is_stable_across_loads(tmp_path: Path) -> None:
    path = _write(tmp_path, _render(_sections()))
    first, second = load_prompts(path), load_prompts(path)
    assert first.sha256 == second.sha256 == hashlib.sha256(path.read_bytes()).hexdigest()


def test_sha256_changes_when_the_file_changes(tmp_path: Path) -> None:
    path = _write(tmp_path, _render(_sections()))
    before = load_prompts(path).sha256
    path.write_text(path.read_text(encoding="utf-8") + "# a comment changes the bytes\n", encoding="utf-8")
    after = load_prompts(path)
    assert after.sha256 != before
    assert after.global_guideline == "Global rules."  # same content, different hash


def test_sha256_changes_when_a_guideline_changes(tmp_path: Path) -> None:
    a = load_prompts(_write(tmp_path, _render(_sections()), name="a.toml"))
    b = load_prompts(_write(tmp_path, _render(_sections(style_text="Different style.")), name="b.toml"))
    assert a.sha256 != b.sha256


def test_shipped_agent_prompts_state_the_rules(prompts: PromptSet) -> None:
    assert set(prompts.tasks) == {"agent", "guard_caller", "guard_consent", "guard_summary", "guard_document", "guard_reply"}
    g = prompts.global_guideline.lower()
    for rule in (
        "caller_review",
        "off_topic_request",
        "stop_persuading",
        "acting_for_someone_else",
        "request_human",
        "caller_message",
        "never invent",
    ):
        assert rule in g, rule
    assert "colon" in prompts.style_guideline.lower() and "em dash" in prompts.style_guideline.lower()
    assert "three of these are required" in prompts.phases["VERIFY_ID"].lower()
    assert "policy number helps but does not count" in prompts.phases["VERIFY_ID"].lower()
    assert "record_document_status" in prompts.phases["PROCESS_CASE"] and "offer_email_summary" in prompts.phases["PROCESS_CASE"]
    assert "record_email_decision" in prompts.phases["POST_PROCESS"]
