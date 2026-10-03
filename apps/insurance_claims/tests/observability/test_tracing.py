"""Tracing: nesting, errors, async, OpenAI wrapper privacy, redaction before write, atomic files, reader CLI."""

from __future__ import annotations

import asyncio
import contextvars
import hashlib
import json
import os
import re
import stat
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from insurance_claims.config import APP_ROOT
from insurance_claims.domain.models import Claim, Policyholder, Representative
from insurance_claims.observability import trace_reader, tracing
from insurance_claims.observability.redaction import Redactor, build_fixture_redactor

FIXTURES = APP_ROOT / "fixtures"
SEEDED_KEY = "sk-test-SEEDED-0123456789"
DENIAL_REASON = "the review file did not include the pathology report and the treating provider office note"
FILE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2}\.\d{3}Z_[A-Za-z0-9_.\-]+_[0-9a-f]{6}\.json$")


# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _isolate_tracing(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every test starts disabled with zero failures; global config is restored afterwards."""
    monkeypatch.setattr(tracing, "_CONFIG", tracing._CONFIG)
    monkeypatch.setattr(tracing, "write_failures", 0)
    tracing.configure(trace_dir=Path("unused-traces"), enabled=False)


@pytest.fixture
def trace_dir(tmp_path: Path) -> Path:
    directory = tmp_path / "traces"
    tracing.configure(trace_dir=directory, enabled=True, redactor=Redactor(salt="unit"))
    return directory


@pytest.fixture(scope="module")
def bundle() -> SimpleNamespace:
    def load(name: str, model: type) -> tuple:
        return tuple(model(**row) for row in json.loads((FIXTURES / name).read_text(encoding="utf-8")))

    return SimpleNamespace(
        policyholders=load("policyholders.json", Policyholder),
        claims=load("claims.json", Claim),
        representatives=load("representatives.json", Representative),
    )


def _files(directory: Path) -> list[Path]:
    if not directory.exists():
        return []
    return sorted(directory.glob("*.json"), key=lambda p: (p.name[:24], p.stat().st_mtime_ns, p.name))


def _single(directory: Path) -> tuple[dict[str, Any], str]:
    files = _files(directory)
    assert len(files) == 1, [f.name for f in files]
    text = files[0].read_text(encoding="utf-8")
    return json.loads(text), text


def _fake_response(**overrides: Any) -> SimpleNamespace:
    fields: dict[str, Any] = {
        "status": "completed",
        "model": "gpt-5.6-luna-2026-09-01",
        "output": [
            SimpleNamespace(type="reasoning", summary=[SimpleNamespace(text="RAW-REASONING-TEXT")]),
            SimpleNamespace(type="function_call", name="get_claim_details", call_id="c1", arguments='{"case_id": "RAW-ARG"}'),
            SimpleNamespace(type="message", content=[SimpleNamespace(type="output_text", text="RAW-MODEL-OUTPUT Margaret Chen")]),
        ],
        "usage": SimpleNamespace(
            input_tokens=120,
            output_tokens=30,
            total_tokens=150,
            input_tokens_details=SimpleNamespace(cached_tokens=64),
            output_tokens_details=SimpleNamespace(reasoning_tokens=12),
        ),
        "incomplete_details": None,
    }
    fields.update(overrides)
    return SimpleNamespace(**fields)


def _request_kwargs() -> dict[str, Any]:
    return {
        "model": "gpt-5.6-luna",
        "instructions": "SECRET-INSTRUCTIONS-TEXT about Margaret Chen",
        "input": [
            {"role": "user", "content": "RAW-USER-TEXT my DOB is 1985-03-15"},
            {"type": "function_call_output", "call_id": "c0", "output": "RAW-TOOL-OUTPUT"},
        ],
        "tools": [{"type": "function", "name": "get_claim_details", "parameters": {}}, {"type": "web_search"}],
        "max_output_tokens": 800,
        "reasoning": {"effort": "low"},
        "tool_choice": "auto",
        "parallel_tool_calls": False,
        "store": False,
        "metadata": {"user": "RAW-METADATA-USER"},
    }


def _fake_client(response: Any = None) -> tuple[SimpleNamespace, list[dict[str, Any]]]:
    calls: list[dict[str, Any]] = []

    def create(**kwargs: Any) -> Any:
        calls.append(kwargs)
        return response if response is not None else _fake_response()

    return SimpleNamespace(responses=SimpleNamespace(create=create)), calls


RAW_STRINGS = (
    "SECRET-INSTRUCTIONS-TEXT",
    "RAW-USER-TEXT",
    "RAW-TOOL-OUTPUT",
    "RAW-MODEL-OUTPUT",
    "RAW-ARG",
    "RAW-REASONING-TEXT",
    "RAW-METADATA-USER",
)


# ---------------------------------------------------------------------------
# Nesting
# ---------------------------------------------------------------------------


def test_nested_spans_write_one_correctly_nested_file(trace_dir: Path) -> None:
    with tracing.span("turn", type="task", input={"turn": 1}, metadata={"phase": "VERIFY_ID"}) as root:
        assert tracing.current_span() is root
        with tracing.span("agent", type="function") as agent:
            assert tracing.current_span() is agent
            with tracing.span("responses.create", type="llm"):
                pass
            assert tracing.current_span() is agent
        with tracing.span("tool.select_claim"):
            pass
    assert tracing.current_span() is None

    trace, _ = _single(trace_dir)
    assert list(trace) == ["name", "type", "start", "duration_ms", "input", "output", "metadata", "children"]
    assert (trace["name"], trace["type"], trace["input"], trace["metadata"]) == ("turn", "task", {"turn": 1}, {"phase": "VERIFY_ID"})
    assert [c["name"] for c in trace["children"]] == ["agent", "tool.select_claim"]
    assert [c["name"] for c in trace["children"][0]["children"]] == ["responses.create"]
    assert trace["children"][1]["children"] == []
    assert all(isinstance(n["duration_ms"], float) and n["duration_ms"] >= 0 for n in tracing.walk(trace))
    assert FILE_RE.match(_files(trace_dir)[0].name)


def test_sibling_after_nested_span_keeps_the_right_parent(trace_dir: Path) -> None:
    with tracing.span("root"):
        with tracing.span("a"):
            with tracing.span("a.1"), tracing.span("a.1.x"):
                pass
            with tracing.span("a.2"):
                pass
        with tracing.span("b"):
            tracing.event("b.guard", decision="hold")
    trace, _ = _single(trace_dir)
    a, b = trace["children"]
    assert [c["name"] for c in a["children"]] == ["a.1", "a.2"]
    assert [c["name"] for c in a["children"][0]["children"]] == ["a.1.x"]
    assert b["name"] == "b" and [c["name"] for c in b["children"]] == ["b.guard"]


def test_each_root_writes_its_own_file(trace_dir: Path) -> None:
    for index in range(3):
        with tracing.span(f"turn{index}"):
            pass
    assert len(_files(trace_dir)) == 3


def test_exception_is_recorded_reraised_and_context_restored(trace_dir: Path) -> None:
    with pytest.raises(ValueError, match="boom"), tracing.span("root"):
        with tracing.span("ok"):
            pass
        with tracing.span("bad"):
            raise ValueError("boom")
    assert tracing.current_span() is None
    trace, _ = _single(trace_dir)
    assert trace["error"] == "ValueError: boom"
    ok, bad = trace["children"]
    assert "error" not in ok and bad["error"] == "ValueError: boom"
    with tracing.span("next"):
        pass
    assert len(_files(trace_dir)) == 2  # the failed root did not swallow the next root


def test_exception_inside_child_does_not_break_sibling_parenting(trace_dir: Path) -> None:
    with tracing.span("root"):
        with pytest.raises(KeyError), tracing.span("child"):
            raise KeyError("missing")
        with tracing.span("sibling"):
            pass
    trace, _ = _single(trace_dir)
    assert [c["name"] for c in trace["children"]] == ["child", "sibling"]
    assert "error" not in trace


def test_input_is_snapshotted_at_capture_time(trace_dir: Path) -> None:
    payload = {"items": [1]}
    with tracing.span("root", input=payload):
        payload["items"].append(2)
        payload["late"] = True
    trace, _ = _single(trace_dir)
    assert trace["input"] == {"items": [1]}


def test_restore_falls_back_when_token_belongs_to_another_context() -> None:
    foreign = contextvars.copy_context()
    token = foreign.run(tracing._current.set, {"name": "elsewhere"})
    tracing._restore(token, None)
    assert tracing.current_span() is None


# ---------------------------------------------------------------------------
# @traced
# ---------------------------------------------------------------------------


def test_traced_binds_args_by_name_and_captures_output(trace_dir: Path) -> None:
    @tracing.traced(type="tool")
    def lookup(case_id: str, *, include_docs: bool = False) -> dict[str, Any]:
        return {"case_id": case_id, "docs": include_docs}

    @tracing.traced
    def agent(question: str) -> str:
        lookup("CL-2048")
        return "done"

    assert agent("status?") == "done"
    trace, _ = _single(trace_dir)
    assert (trace["name"], trace["type"], trace["input"], trace["output"]) == ("agent", "function", {"question": "status?"}, "done")
    tool = trace["children"][0]
    assert (tool["name"], tool["type"]) == ("lookup", "tool")
    assert tool["input"] == {"case_id": "CL-2048", "include_docs": False}
    assert tool["output"] == {"case_id": "CL-2048", "docs": False}


def test_traced_on_methods_skips_self_and_honours_name_exclude_and_set_output(trace_dir: Path) -> None:
    class Agent:
        @tracing.traced(type="task", name="handle_turn", exclude=("utterance",))
        def handle(self, utterance: str, turn: int) -> dict[str, str]:
            tracing.set_output({"stop_reason": "completed"})
            return {"raw": "should not replace the explicit output"}

        @tracing.traced(capture_output=False)
        def secret_result(self) -> str:
            return "sensitive"

    agent = Agent()
    with tracing.span("root"):
        agent.handle("my name is Margaret", 3)
        agent.secret_result()
    trace, _ = _single(trace_dir)
    handled, hidden = trace["children"]
    assert handled["name"] == "handle_turn" and handled["type"] == "task"
    assert handled["input"] == {"utterance": "[omitted]", "turn": 3}
    assert handled["output"] == {"stop_reason": "completed"}
    assert hidden["input"] == {} and hidden["output"] == "[omitted]"


def test_traced_records_and_reraises_errors(trace_dir: Path) -> None:
    @tracing.traced
    def fails(x: int) -> None:
        raise RuntimeError(f"bad {x}")

    with pytest.raises(RuntimeError, match="bad 7"):
        fails(7)
    trace, _ = _single(trace_dir)
    assert trace["error"] == "RuntimeError: bad 7" and trace["input"] == {"x": 7}


def test_traced_preserves_metadata_and_handles_bad_arguments(trace_dir: Path) -> None:
    @tracing.traced
    def documented(a: int) -> int:
        """Docstring kept."""
        return a

    assert documented.__name__ == "documented" and documented.__doc__ == "Docstring kept."
    with pytest.raises(TypeError):
        documented(1, 2)  # type: ignore[call-arg]
    trace, _ = _single(trace_dir)
    assert trace["input"] == {"bind_error": "arguments do not match the signature"}
    assert trace["error"].startswith("TypeError")


def test_async_traced_nests_concurrent_children(trace_dir: Path) -> None:
    @tracing.traced(type="tool")
    async def fetch(case_id: str) -> str:
        await asyncio.sleep(0.001)
        with tracing.span("inner"):
            await asyncio.sleep(0)
        return f"ok {case_id}"

    @tracing.traced(type="task")
    async def run_turn(text: str) -> list[str]:
        return list(await asyncio.gather(fetch("CL-1"), fetch("CL-2")))

    assert asyncio.run(run_turn("hi")) == ["ok CL-1", "ok CL-2"]
    trace, _ = _single(trace_dir)
    assert trace["name"] == "run_turn" and trace["output"] == ["ok CL-1", "ok CL-2"]
    assert sorted(c["input"]["case_id"] for c in trace["children"]) == ["CL-1", "CL-2"]
    assert all([g["name"] for g in c["children"]] == ["inner"] for c in trace["children"])


def test_async_traced_records_errors(trace_dir: Path) -> None:
    @tracing.traced
    async def broken() -> None:
        raise LookupError("nope")

    with pytest.raises(LookupError):
        asyncio.run(broken())
    trace, _ = _single(trace_dir)
    assert trace["error"] == "LookupError: nope"


# ---------------------------------------------------------------------------
# annotate / set_output / event / add_tokens
# ---------------------------------------------------------------------------


def test_helpers_are_noops_outside_a_span(trace_dir: Path) -> None:
    tracing.annotate(phase="VERIFY_ID")
    tracing.set_output("x")
    tracing.add_tokens(1, 2, 3)
    assert tracing.current_span() is None
    assert _files(trace_dir) == []


def test_annotate_set_output_event_and_tokens(trace_dir: Path) -> None:
    with tracing.span("turn", type="task") as node:
        node["metadata"] = None  # callers may clobber metadata; annotate must cope
        tracing.annotate(phase="VERIFY_ID", state_version=4)
        tracing.annotate(stop_reason="awaiting_identity")
        tracing.add_tokens(10, 5, 15)
        tracing.add_tokens(1, 1, 2)
        tracing.add_tokens("x", None, True)  # type: ignore[arg-type]
        tracing.event("verification_gate", status="pending", provided=2)
        tracing.event("llm_retry", type="retry", attempt=1)
        tracing.set_output({"stop_reason": "completed"})
    trace, _ = _single(trace_dir)
    assert trace["metadata"] == {"phase": "VERIFY_ID", "state_version": 4, "stop_reason": "awaiting_identity"}
    assert trace["tokens"] == {"input": 11, "output": 6, "total": 17}
    assert trace["output"] == {"stop_reason": "completed"}
    gate, retry = trace["children"]
    assert (gate["type"], gate["duration_ms"], gate["output"]) == ("guard", 0.0, {"status": "pending", "provided": 2})
    assert retry["type"] == "retry" and retry["output"] == {"attempt": 1}


def test_event_outside_a_span_is_its_own_root(trace_dir: Path) -> None:
    tracing.event("orphan_guard", decision="block")
    trace, _ = _single(trace_dir)
    assert trace["name"] == "orphan_guard" and trace["output"] == {"decision": "block"}


def test_children_beyond_the_cap_are_counted(trace_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(tracing, "MAX_CHILDREN", 3)
    with tracing.span("root"):
        for index in range(5):
            tracing.event(f"e{index}")
    trace, _ = _single(trace_dir)
    assert len(trace["children"]) == 3 and trace["dropped_children"] == 2


def test_thread_without_context_starts_a_new_root_but_copied_context_nests(trace_dir: Path) -> None:
    with tracing.span("root"):
        plain = threading.Thread(target=lambda: tracing.event("from_plain_thread"))
        plain.start()
        plain.join()
        ctx = contextvars.copy_context()
        nested = threading.Thread(target=lambda: ctx.run(tracing.event, "from_copied_context"))
        nested.start()
        nested.join()
    traces = [json.loads(p.read_text()) for p in _files(trace_dir)]
    root = next(t for t in traces if t["name"] == "root")
    assert [c["name"] for c in root["children"]] == ["from_copied_context"]
    assert {t["name"] for t in traces} == {"root", "from_plain_thread"}


# ---------------------------------------------------------------------------
# wrap_openai
# ---------------------------------------------------------------------------


def test_wrap_openai_records_metadata_tokens_and_no_raw_content(trace_dir: Path) -> None:
    client, calls = _fake_client()
    wrapped = tracing.wrap_openai(client)
    assert wrapped is client
    kwargs = _request_kwargs()
    with tracing.span("turn", type="task"):
        response = client.responses.create(**kwargs)
    assert response.status == "completed" and calls == [kwargs]  # call passes through untouched

    trace, text = _single(trace_dir)
    llm = trace["children"][0]
    assert (llm["name"], llm["type"], llm["input"]) == ("responses.create", "llm", None)
    assert llm["tokens"] == {"input": 120, "output": 30, "total": 150}
    meta = llm["metadata"]
    assert meta["model"] == "gpt-5.6-luna" and meta["max_output_tokens"] == 800
    assert meta["reasoning"] == {"effort": "low"}
    assert meta["tool_names"] == ["get_claim_details", "web_search"]
    assert meta["prompt_sha256"] == hashlib.sha256(kwargs["instructions"].encode()).hexdigest()
    assert meta["input_items"] == 2 and meta["input_item_types"] == {"message": 1, "function_call_output": 1}
    assert (meta["reasoning_tokens"], meta["cached_tokens"], meta["response_model"]) == (12, 64, "gpt-5.6-luna-2026-09-01")
    assert llm["output"] == {
        "status": "completed",
        "item_types": ["reasoning", "function_call", "message"],
        "function_calls": ["get_claim_details"],
        "has_refusal": False,
        "text_chars": len("RAW-MODEL-OUTPUT Margaret Chen"),
    }
    for raw in RAW_STRINGS:
        assert raw not in text
    assert tracing.summarize(trace)["tokens"] == {"input": 120, "output": 30, "total": 150}


def test_wrap_openai_is_idempotent(trace_dir: Path) -> None:
    client, _ = _fake_client()
    tracing.wrap_openai(tracing.wrap_openai(client))
    with tracing.span("turn"):
        client.responses.create(model="m", input="hi")
    trace, _ = _single(trace_dir)
    assert [c["name"] for c in trace["children"]] == ["responses.create"]


def test_wrap_openai_without_responses_or_parse_is_harmless() -> None:
    bare = SimpleNamespace(chat=SimpleNamespace())
    assert tracing.wrap_openai(bare) is bare
    client, _ = _fake_client()
    tracing.wrap_openai(client)
    assert not hasattr(client.responses, "parse")


def test_wrap_openai_error_records_class_and_status_only(trace_dir: Path) -> None:
    class FakeRateLimitError(Exception):
        status_code = 429
        code = "rate_limit_exceeded"

    def create(**_: Any) -> Any:
        raise FakeRateLimitError("Request echoed RAW-USER-TEXT and margaret@email.com")

    client = tracing.wrap_openai(SimpleNamespace(responses=SimpleNamespace(create=create)))
    with pytest.raises(FakeRateLimitError), tracing.span("turn"):
        client.responses.create(model="m", input="RAW-USER-TEXT")
    trace, text = _single(trace_dir)
    expected = "FakeRateLimitError status=429 code=rate_limit_exceeded"
    assert trace["children"][0]["error"] == expected
    assert trace["error"] == expected  # the parent span uses the sanitized description too
    assert "RAW-USER-TEXT" not in text and "margaret" not in text


def test_wrap_openai_handles_missing_or_odd_usage_and_stream(trace_dir: Path) -> None:
    odd = _fake_response(usage=SimpleNamespace(input_tokens="12", output_tokens=None, total_tokens=None), status=None)
    client, _ = _fake_client(odd)
    tracing.wrap_openai(client)
    with tracing.span("turn"):
        client.responses.create(model="m", input=[])
        client.responses.create(model="m", input=[], stream=True)
    trace, _ = _single(trace_dir)
    first, streamed = trace["children"]
    assert first["tokens"] == {"input": 0, "output": 0, "total": 0} and first["output"]["status"] is None
    assert streamed["output"] == {"status": "stream_not_captured"}

    no_usage, _ = _fake_client(_fake_response(usage=None, output=None, incomplete_details=SimpleNamespace(reason="max_output_tokens")))
    tracing.wrap_openai(no_usage)
    with tracing.span("turn2"):
        no_usage.responses.create(model="m")
    second = next(t for t in (json.loads(f.read_text()) for f in _files(trace_dir)) if t["name"] == "turn2")
    llm = next(n for n in tracing.walk(second) if n["type"] == "llm")
    assert "tokens" not in llm and llm["output"]["item_types"] == [] and llm["output"]["incomplete_reason"] == "max_output_tokens"


def test_wrap_openai_detects_refusals(trace_dir: Path) -> None:
    refusal = _fake_response(output=[SimpleNamespace(type="message", content=[SimpleNamespace(type="refusal", refusal="RAW-REFUSAL")])])
    client, _ = _fake_client(refusal)
    tracing.wrap_openai(client)
    with tracing.span("turn"):
        client.responses.create(model="m", input="x")
    trace, text = _single(trace_dir)
    assert trace["children"][0]["output"]["has_refusal"] is True and "RAW-REFUSAL" not in text


def test_wrap_openai_async_client(trace_dir: Path) -> None:
    async def create(**_: Any) -> Any:
        await asyncio.sleep(0)
        return _fake_response()

    client = tracing.wrap_openai(SimpleNamespace(responses=SimpleNamespace(create=create)))

    async def run() -> Any:
        with tracing.span("turn"):
            return await client.responses.create(model="m", instructions="SECRET-INSTRUCTIONS-TEXT", input="RAW-USER-TEXT")

    assert asyncio.run(run()).status == "completed"
    trace, text = _single(trace_dir)
    assert trace["children"][0]["tokens"]["total"] == 150 and "SECRET-INSTRUCTIONS-TEXT" not in text


def test_wrap_openai_on_real_sdk_client_object() -> None:
    openai = pytest.importorskip("openai")
    client = openai.OpenAI(api_key="sk-test-not-used-0000000000", max_retries=0)
    tracing.wrap_openai(client)
    assert getattr(client.responses.create, tracing._TRACED_FLAG) is True


# ---------------------------------------------------------------------------
# Privacy: seeded secrets never reach the file
# ---------------------------------------------------------------------------


def test_seeded_secrets_never_reach_the_trace_file(trace_dir: Path, bundle: SimpleNamespace) -> None:
    tracing.configure(
        trace_dir=trace_dir, enabled=True, redactor=build_fixture_redactor(bundle, secrets=[SEEDED_KEY, None], salt="seeded-test")
    )
    utterance = (
        "I'm the policyholder. My name is Margaret Chen, policy POL-9921. DOB is 1985-03-15, SSN last four is 4472. "
        f"Email margaret@email.com, phone +16505212836. key {SEEDED_KEY}"
    )

    class Opaque:
        def __str__(self) -> str:
            return f"Opaque(Margaret Chen, (650) 521-2836, {SEEDED_KEY})"

    with (
        pytest.raises(RuntimeError),
        tracing.span(
            "turn",
            type="task",
            input={"utterance": utterance, "note": utterance, "holder": bundle.policyholders[0], "obj": Opaque()},
            metadata={"case_id": "CL-2048", "phase": "VERIFY_ID", "credential": SEEDED_KEY, "Margaret Chen": "as a key"},
        ),
    ):
        tracing.annotate(
            identity={"full_name": "Margaret Chen", "dob": "1985-03-15", "phone": "+16505212836",
                      "email": "margaret@email.com", "id_last4": "4472"},
            fields=["full_name", "dob", "phone", "email", "id_last4"],
            matched_fields=3,
        )  # fmt: skip
        tracing.event("verification_gate", status="verified", provided=["full_name", "dob", "id_last4"], case_id="CL-2048")
        with tracing.span("evidence", type="tool", input={"case_id": "CL-2048"}):
            tracing.set_output({"case_id": "CL-2048", "denial_reason": DENIAL_REASON, "why": f"because {DENIAL_REASON}"})
        with tracing.span("reply_guard"):
            tracing.set_output(f"Hi Margaret, {DENIAL_REASON}. Reach you at 650.521.2836? key={SEEDED_KEY}")
            raise RuntimeError(f"draft failed for Margaret Chen ({SEEDED_KEY}) {DENIAL_REASON} 1985-03-15 4472")

    trace, text = _single(trace_dir)
    for secret in (
        "Margaret Chen", "Margaret", "Chen", "margaret@email.com", "+16505212836", "6505212836", "650.521.2836",
        "1985-03-15", "4472", DENIAL_REASON, "pathology report and the treating", SEEDED_KEY, "SEEDED", "POL-9921", "9921",
    ):  # fmt: skip
        assert secret.lower() not in text.lower(), f"{secret!r} leaked into the trace file"
    assert "CL-2048" in text
    for field_name in ("full_name", "dob", "phone", "email", "id_last4", "case_id", "utterance"):
        assert f'"{field_name}"' in text
    assert trace["metadata"]["phase"] == "VERIFY_ID" and trace["metadata"]["matched_fields"] == 3
    assert trace["error"].startswith("RuntimeError: draft failed for [PII:h:")
    assert "[REDACTED]" in text and "[PII:h:" in text


def test_default_redactor_still_scrubs_generic_pii(trace_dir: Path) -> None:
    tracing.configure(trace_dir=trace_dir, enabled=True)  # no fixture redactor supplied
    with tracing.span("turn", input={"msg": "mail bob@corp.io, call 415-555-0199, key sk-abcdefghijklmnop"}):
        pass
    _, text = _single(trace_dir)
    for leak in ("bob@corp.io", "555-0199", "sk-abcdefghijklmnop"):
        assert leak not in text


# ---------------------------------------------------------------------------
# Writing: truncation, atomicity, failure handling, disabled mode
# ---------------------------------------------------------------------------


def test_truncates_long_strings(trace_dir: Path) -> None:
    with tracing.span("turn", input={"blob": "x" * 50_000, "short": "y" * 10}):
        tracing.set_output(["z" * 4001])
    trace, _ = _single(trace_dir)
    blob = trace["input"]["blob"]
    assert len(blob) == tracing.MAX_STRING_CHARS + len(tracing.TRUNCATION_MARK)
    assert blob.endswith(tracing.TRUNCATION_MARK) and set(blob[: tracing.MAX_STRING_CHARS]) == {"x"}
    assert trace["input"]["short"] == "y" * 10
    assert trace["output"][0].endswith(tracing.TRUNCATION_MARK)


def test_redaction_happens_before_truncation(trace_dir: Path) -> None:
    tracing.configure(trace_dir=trace_dir, enabled=True, redactor=Redactor(["Margaret Chen"], salt="t"))
    # The raw name straddles the 4000-char cut; truncating first would leave "Marga" behind.
    with tracing.span("turn", input={"blob": "x" * 3990 + " Margaret Chen " + "x" * 100}):
        pass
    _, text = _single(trace_dir)
    assert "Marg" not in text and "[PII:h:" in text


def test_capture_limits_bound_memory(trace_dir: Path) -> None:
    with tracing.span("turn", input={"many": list(range(tracing.CAPTURE_MAX_ITEMS + 5)), "huge": "q" * 100_000}) as node:
        assert len(node["input"]["many"]) == tracing.CAPTURE_MAX_ITEMS + 1
        assert len(node["input"]["huge"]) == tracing.CAPTURE_MAX_STRING + len(tracing.TRUNCATION_MARK)
    trace, _ = _single(trace_dir)
    assert trace["input"]["many"][-1] == "[5 more items]"


def test_atomic_write_leaves_no_temp_files_and_private_modes(trace_dir: Path) -> None:
    for index in range(5):
        with tracing.span(f"turn{index}"):
            tracing.event("gate")
    names = sorted(p.name for p in trace_dir.iterdir())
    assert len(names) == 5 and all(FILE_RE.match(n) for n in names)
    assert not any(n.startswith(".") or n.endswith(".tmp") for n in names)
    for path in trace_dir.iterdir():
        assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert stat.S_IMODE(trace_dir.stat().st_mode) == 0o700


def test_failed_replace_cleans_up_temp_file_and_counts(trace_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def broken_replace(*_: Any) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(tracing.os, "replace", broken_replace)
    with tracing.span("turn"):
        pass
    assert tracing.write_failures == 1
    assert list(trace_dir.iterdir()) == []


def test_write_failure_is_swallowed_and_counted(tmp_path: Path) -> None:
    blocker = tmp_path / "not-a-dir"
    blocker.write_text("file in the way")
    tracing.configure(trace_dir=blocker, enabled=True)
    with tracing.span("turn") as node:
        tracing.event("gate")
    assert node["children"][0]["name"] == "gate"
    tracing.event("orphan")
    assert tracing.write_failures == 2
    assert blocker.read_text() == "file in the way"


@pytest.mark.skipif(not hasattr(os, "geteuid") or os.geteuid() == 0, reason="root ignores directory permissions")
def test_unwritable_directory_is_swallowed_and_counted(tmp_path: Path) -> None:
    locked = tmp_path / "locked"
    locked.mkdir()
    locked.chmod(0o500)
    try:
        tracing.configure(trace_dir=locked, enabled=True)
        with tracing.span("turn"):
            pass
        assert tracing.write_failures == 1 and list(locked.iterdir()) == []
    finally:
        locked.chmod(0o700)


def test_unserializable_and_cyclic_values_never_raise(trace_dir: Path) -> None:
    class Hostile:
        def __str__(self) -> str:
            raise RuntimeError("cannot print")

        def __repr__(self) -> str:
            raise RuntimeError("cannot repr")

    loop: dict[str, Any] = {}
    loop["self"] = loop
    with tracing.span("turn", input={"hostile": Hostile(), "loop": loop, "nan": float("nan"), "bytes": b"\x00\x01", "set": {3, 1}}):
        tracing.annotate(when=__import__("datetime").date(2026, 1, 12), amount=__import__("decimal").Decimal("1450.00"))
    trace, _ = _single(trace_dir)
    assert trace["input"]["hostile"] == "<unserializable Hostile>"
    assert trace["input"]["loop"]["self"] == "[cycle]"
    assert trace["input"]["nan"] == "nan" and trace["input"]["bytes"] == "<2 bytes>" and trace["input"]["set"] == [1, 3]
    assert trace["metadata"] == {"when": "2026-01-12", "amount": "1450.00"}


def test_disabled_tracing_writes_nothing_but_spans_still_work(tmp_path: Path) -> None:
    directory = tmp_path / "traces"
    tracing.configure(trace_dir=directory, enabled=False)
    assert tracing.is_enabled() is False
    with tracing.span("turn") as root:
        with tracing.span("child"):
            tracing.annotate(phase="VERIFY_ID")
            tracing.add_tokens(1, 1, 2)
        tracing.event("gate", ok=True)
    tracing.event("orphan")
    assert [c["name"] for c in root["children"]] == ["child", "gate"]
    assert root["children"][0]["tokens"] == {"input": 1, "output": 1, "total": 2}
    assert not directory.exists() and tracing.write_failures == 0


def test_deeply_nested_spans_keep_their_captured_values(trace_dir: Path) -> None:
    def nest(level: int) -> None:
        if level == 12:
            tracing.set_output({"a": {"b": {"c": {"d": {"e": "leaf"}}}}})
            return
        with tracing.span(f"level{level}"):
            nest(level + 1)

    with tracing.span("root"):
        nest(0)
    trace, _ = _single(trace_dir)
    deepest = list(tracing.walk(trace))[-1]
    assert deepest["name"] == "level11"
    assert deepest["output"] == {"a": {"b": {"c": {"d": {"e": "leaf"}}}}}


def test_root_name_is_sanitized_in_file_name(trace_dir: Path) -> None:
    with tracing.span("turn/../../etc passwd"):
        pass
    (path,) = _files(trace_dir)
    assert path.parent == trace_dir and FILE_RE.match(path.name) and "/" not in path.name


# ---------------------------------------------------------------------------
# walk / summarize
# ---------------------------------------------------------------------------


def _sample_trace() -> dict[str, Any]:
    def node(name: str, kind: str, children: list | None = None, **extra: Any) -> dict[str, Any]:
        base = {"name": name, "type": kind, "start": "s", "duration_ms": 1.0, "input": None, "output": None,
                "metadata": {}, "children": children or []}  # fmt: skip
        return {**base, **extra}

    return node(
        "turn",
        "task",
        [
            node("agent", "function", [node("responses.create", "llm", tokens={"input": 100, "output": 20, "total": 120})]),
            node("verification_gate", "guard", output={"status": "verified", "provided": 3}),
            node("phase_transition", "guard", output={"from_phase": "VERIFY_ID", "to_phase": "RESOLVE_INTENT"}),
            node("tool.get_claim_details", "tool", output={"ok": True}),
            node("tool.get_followup_guidance", "tool", [node("bad", "function", error="ValueError: x")], error="ValueError: x"),
            node("responses.create", "llm", tokens={"input": 50, "output": 10, "total": 60}),
            node("tool.get_claim_details", "function", output={"stop_reason": "final_answer"}),
        ],
        input={"phase_before": "VERIFY_ID"},
        metadata={"phase": "RESOLVE_INTENT", "stop_reason": "completed"},
    )


def test_walk_is_preorder_and_tolerates_bad_children() -> None:
    trace = _sample_trace()
    names = [n["name"] for n in tracing.walk(trace)]
    assert names[:3] == ["turn", "agent", "responses.create"] and names[-1] == "tool.get_claim_details"
    assert [n["name"] for n in tracing.walk({"name": "x", "children": ["junk", None, {"name": "y"}]})] == ["x", "y"]
    assert [n["name"] for n in tracing.walk({"name": "solo"})] == ["solo"]


def test_summarize_reports_path_guards_tokens_stops_phases_errors() -> None:
    summary = tracing.summarize(_sample_trace())
    assert summary == {
        "tool_path": ["tool.get_claim_details", "tool.get_followup_guidance"],
        "guards": [
            {"name": "verification_gate", "output": {"status": "verified", "provided": 3}},
            {"name": "phase_transition", "output": {"from_phase": "VERIFY_ID", "to_phase": "RESOLVE_INTENT"}},
        ],
        "tokens": {"input": 150, "output": 30, "total": 180},
        "llm_calls": 2,
        "stop_reasons": ["completed", "final_answer"],
        "phases": ["VERIFY_ID", "RESOLVE_INTENT"],
        "errors": [{"name": "bad", "error": "ValueError: x"}],
    }


def test_summarize_orders_phases_like_the_agent_records_them() -> None:
    # Shape the agent writes: root "phase" is annotated at the end of the turn (final phase),
    # the agent span input carries "phase_before", transitions are guard events.
    trace = {
        "name": "turn", "type": "turn", "metadata": {"phase": "PROCESS_CASE", "stop_reason": "completed"},
        "children": [
            {"name": "agent", "type": "task", "input": {"phase_before": "VERIFY_ID"},
             "output": {"phase_after": "PROCESS_CASE", "stop_reason": "completed"}, "children": [
                {"name": "phase_transition", "type": "guard", "output": {"from_phase": "VERIFY_ID", "to_phase": "RESOLVE_INTENT"}},
                {"name": "phase_transition", "type": "guard", "output": {"from_phase": "RESOLVE_INTENT", "to_phase": "PROCESS_CASE"}},
                {"name": "compose_reply", "type": "task", "input": {"phase": "VERIFY_ID"}, "output": {"stop_reason": "completed"}},
                {"name": "turn_reply", "type": "guard", "output": {"stop_reason": "fallback"}},
            ]},
        ],
    }  # fmt: skip
    summary = tracing.summarize(trace)
    assert summary["phases"] == ["VERIFY_ID", "RESOLVE_INTENT", "PROCESS_CASE"]
    assert summary["stop_reasons"] == ["completed", "fallback"]
    assert tracing.summarize({"name": "t", "metadata": {"phase": "VERIFY_ID"}})["phases"] == ["VERIFY_ID"]
    no_transition = {"name": "t", "input": {"phase_before": "POST_PROCESS"}, "metadata": {"phase_after": "PROCESS_CASE"}}
    assert tracing.summarize(no_transition)["phases"] == ["POST_PROCESS", "PROCESS_CASE"]


def test_summarize_tolerates_garbage() -> None:
    empty = tracing.summarize("not a trace")  # type: ignore[arg-type]
    assert empty["tool_path"] == [] and empty["tokens"] == {"input": 0, "output": 0, "total": 0}
    odd = tracing.summarize({"name": "x", "type": "llm", "tokens": {"input": "many", "total": 5}, "metadata": "nope"})
    assert odd["llm_calls"] == 1 and odd["tokens"] == {"input": 0, "output": 0, "total": 5}


def test_summarize_of_a_written_trace(trace_dir: Path) -> None:
    client, _ = _fake_client()
    tracing.wrap_openai(client)

    @tracing.traced(type="tool")
    def get_claim_details(case_id: str) -> dict[str, str]:
        return {"case_id": case_id}

    with tracing.span("turn", type="task", metadata={"phase": "PROCESS_CASE"}):
        client.responses.create(model="m", input="x")
        get_claim_details("CL-2048")
        client.responses.create(model="m", input="y")
        tracing.event("grounding_check", passed=True)
        tracing.annotate(stop_reason="completed")
    trace, _ = _single(trace_dir)
    summary = tracing.summarize(trace)
    assert summary["tool_path"] == ["get_claim_details"] and summary["llm_calls"] == 2
    assert summary["tokens"] == {"input": 240, "output": 60, "total": 300}
    assert summary["guards"] == [{"name": "grounding_check", "output": {"passed": True}}]
    assert summary["stop_reasons"] == ["completed"] and summary["phases"] == ["PROCESS_CASE"]


# ---------------------------------------------------------------------------
# trace_reader CLI
# ---------------------------------------------------------------------------


def _write_two_traces(trace_dir: Path) -> None:
    client, _ = _fake_client()
    tracing.wrap_openai(client)
    with tracing.span("turn", type="task", metadata={"phase": "VERIFY_ID"}):
        client.responses.create(model="m", input="x")
        tracing.event("verification_gate", status="pending", provided=2)
        tracing.annotate(stop_reason="awaiting_identity")
    with tracing.span("turn", type="task", metadata={"phase": "PROCESS_CASE"}):
        with tracing.span("tool.get_claim_details", type="tool"):
            pass
        with pytest.raises(ValueError), tracing.span("reply_guard"):
            raise ValueError("draft rejected")
        tracing.annotate(stop_reason="fallback_template")


def test_trace_reader_prints_a_summary_for_a_directory(trace_dir: Path, capsys: pytest.CaptureFixture[str]) -> None:
    _write_two_traces(trace_dir)
    assert trace_reader.main([str(trace_dir)]) == 0
    out = capsys.readouterr().out
    assert out.count("== 20") == 2
    for expected in (
        "phases: VERIFY_ID",
        "stop reasons: awaiting_identity",
        "verification_gate",
        '"status": "pending"',
        "tool path: (no tool calls)",
        "tool path: tool.get_claim_details",
        "llm calls: 1",
        "tokens: input 120, output 30, total 150",
        "reply_guard: ValueError: draft rejected",
        "stop reasons: fallback_template",
        "== totals: 2 traces, 1 llm calls, tokens input 120 output 30 total 150, 1 with errors",
    ):
        assert expected in out, expected


def test_trace_reader_last_file_and_json_modes(trace_dir: Path, capsys: pytest.CaptureFixture[str]) -> None:
    _write_two_traces(trace_dir)
    assert trace_reader.main([str(trace_dir), "--last", "1"]) == 0
    out = capsys.readouterr().out
    assert out.count("== 20") == 1 and "fallback_template" in out and "totals" not in out

    newest = _files(trace_dir)[-1]
    assert trace_reader.main([str(newest), "--json"]) == 0
    line = capsys.readouterr().out.strip()
    payload = json.loads(line)
    assert payload["file"] == newest.name and payload["stop_reasons"] == ["fallback_template"]


def test_trace_reader_error_paths(tmp_path: Path, trace_dir: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert trace_reader.main([str(tmp_path / "missing")]) == 2
    assert trace_reader.main([str(tmp_path), "--last", "0"]) == 2
    empty = tmp_path / "empty"
    empty.mkdir()
    assert trace_reader.main([str(empty)]) == 0
    assert "no trace files" in capsys.readouterr().out

    _write_two_traces(trace_dir)
    (trace_dir / "corrupt.json").write_text("{not json")
    (trace_dir / "list.json").write_text("[1, 2]")
    (trace_dir / ".trace-abc.tmp").write_text("partial")
    assert trace_reader.main([str(trace_dir)]) == 1
    captured = capsys.readouterr()
    assert captured.out.count("== 20") == 2
    assert "skipped corrupt.json" in captured.err and "skipped list.json" in captured.err


def test_trace_reader_rescrubs_foreign_files(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    foreign = tmp_path / "foreign.json"
    foreign.write_text(json.dumps({"name": "x", "type": "guard", "children": [
        {"name": "g", "type": "guard", "output": {"note": "mail bob@corp.io key sk-abcdefghijklmn"}, "children": []}]}))  # fmt: skip
    assert trace_reader.main([str(foreign)]) == 0
    out = capsys.readouterr().out
    assert "bob@corp.io" not in out and "sk-abcdefghijklmn" not in out and "[REDACTED]" in out


def test_trace_reader_module_runs_as_script(trace_dir: Path) -> None:
    import subprocess
    import sys

    _write_two_traces(trace_dir)
    env = {**os.environ, "PYTHONPATH": str(APP_ROOT / "src")}
    result = subprocess.run(
        [sys.executable, "-m", "insurance_claims.observability.trace_reader", str(trace_dir), "--last", "1"],
        capture_output=True, text=True, env=env, timeout=60, check=False,
    )  # fmt: skip
    assert result.returncode == 0, result.stderr
    assert "fallback_template" in result.stdout
