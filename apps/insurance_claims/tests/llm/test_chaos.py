"""ChaosTransport: seeded, reproducible fault injection over a tool-calling model."""

from __future__ import annotations

import json
import threading
from typing import Any

import pytest

from insurance_claims.llm.base import FunctionCall, LLMError, LLMResponse, LLMUsage, ModelTransport
from insurance_claims.llm.chaos import (
    BAD_TOOL_ARGS,
    CHAOS_REFUSAL,
    ENUM_KEYS,
    EXTRA_FIELDS,
    HALLUCINATED_TOOLS,
    INJECTION_TEXT,
    INVALID_ENUM,
    MODES,
    RAISE_MODES,
    ChaosTransport,
)
from insurance_claims.llm.resilient import ResilientTransport
from insurance_claims.observability import tracing

NON_RAISE = [m for m in MODES if m not in RAISE_MODES]
CALL_MODES = [
    "extra_field",
    "missing_field",
    "invalid_enum",
    "wrong_types",
    "bad_tool_args",
    "hallucinated_tool",
    "wrong_call_id",
    "duplicate_calls",
    "parallel_calls",
]
PROSE_MODES = ["prompt_injection_text", "colon_and_em_dash"]
NOTE_ARGS = {"reason": "denied claim question", "case_type": "healthcare", "status": "denied", "month": 1, "year": 2026}


# ---------------------------------------------------------------------- builders


def message_item(text: str) -> dict[str, Any]:
    return {
        "type": "message",
        "id": "msg_1",
        "role": "assistant",
        "status": "completed",
        "content": [{"type": "output_text", "text": text, "annotations": []}],
    }


def prose_response(text: str = "Your claim was denied, and the appeal deadline has passed.") -> LLMResponse:
    return LLMResponse(status="completed", text=text, raw_output_items=[message_item(text)], usage=LLMUsage(10, 5, 15), model="inner-model")


def tool_response() -> LLMResponse:
    calls = [
        FunctionCall(call_id="call_a", name="note_caller_context", arguments=json.dumps(NOTE_ARGS), item_id="fc_a"),
        FunctionCall(call_id="call_b", name="select_claim", arguments='{"case_id": "CL-2048"}', item_id="fc_b"),
    ]
    raw = [{"type": "reasoning", "id": "rs_1", "summary": [], "encrypted_content": "ENC"}]
    raw += [
        {"type": "function_call", "id": c.item_id, "call_id": c.call_id, "name": c.name, "arguments": c.arguments, "status": "completed"}
        for c in calls
    ]
    return LLMResponse(status="completed", function_calls=calls, raw_output_items=raw, model="inner-model")


class RecordingInner:
    model_name = "inner-model"

    def __init__(self, response: LLMResponse) -> None:
        self.response = response
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> LLMResponse:
        self.calls.append(kwargs)
        return self.response


def run(chaos: ChaosTransport, **extra: Any) -> LLMResponse:
    return chaos.create(task="agent", instructions="rules", input=[{"role": "user", "content": "hi"}], **extra)


def mutate(mode: str, response: LLMResponse, seed: int = 7) -> tuple[LLMResponse, ChaosTransport]:
    chaos = ChaosTransport(RecordingInner(response), seed=seed, rate=1.0, modes=[mode])
    return run(chaos), chaos


def first_args(resp: LLMResponse) -> Any:
    return json.loads(resp.function_calls[0].arguments)


def raw_calls(resp: LLMResponse) -> list[dict[str, Any]]:
    return [i for i in resp.raw_output_items if i["type"] == "function_call"]


def raw_texts(resp: LLMResponse) -> list[str]:
    return [p["text"] for i in resp.raw_output_items if i["type"] == "message" for p in i["content"] if p["type"] == "output_text"]


# ------------------------------------------------------------- configuration


def test_rate_zero_is_a_pure_pass_through() -> None:
    original = tool_response()
    inner = RecordingInner(original)
    chaos = ChaosTransport(inner, seed=1, rate=0.0)
    for _ in range(50):
        assert run(chaos) is original
    assert chaos.applied == [] and chaos.calls == 50 and len(inner.calls) == 50


def test_all_kwargs_are_forwarded_to_inner() -> None:
    inner = RecordingInner(prose_response())
    tools = [{"type": "function", "name": "select_claim"}]
    run(
        ChaosTransport(inner, seed=1, rate=0.0),
        tools=tools,
        max_output_tokens=55,
        reasoning_effort="low",
        tool_choice="auto",
        parallel_tool_calls=False,
        timeout=4.0,
    )
    kw = inner.calls[0]
    assert kw["task"] == "agent" and kw["tools"] is tools
    assert (kw["max_output_tokens"], kw["reasoning_effort"], kw["tool_choice"], kw["parallel_tool_calls"], kw["timeout"]) == (
        55,
        "low",
        "auto",
        False,
        4.0,
    )


@pytest.mark.parametrize("rate", [-0.1, 1.01])
def test_invalid_rate_rejected(rate: float) -> None:
    with pytest.raises(ValueError):
        ChaosTransport(RecordingInner(prose_response()), seed=1, rate=rate)


def test_unknown_or_empty_modes_rejected() -> None:
    with pytest.raises(ValueError, match="unknown chaos modes"):
        ChaosTransport(RecordingInner(prose_response()), seed=1, modes=["refusal", "delete_database"])
    with pytest.raises(ValueError):
        ChaosTransport(RecordingInner(prose_response()), seed=1, modes=[])


def test_protocol_and_model_name_passthrough() -> None:
    chaos = ChaosTransport(RecordingInner(prose_response()), seed=1)
    assert isinstance(chaos, ModelTransport)
    assert chaos.model_name == "inner-model" and chaos.modes == MODES


def test_same_seed_gives_identical_runs_and_different_seeds_differ() -> None:
    def trace(seed: int) -> tuple[list[tuple[str, str]], list[str]]:
        chaos = ChaosTransport(RecordingInner(tool_response()), seed=seed, rate=0.5)
        outputs: list[str] = []
        for _ in range(60):
            try:
                outputs.append(repr(run(chaos)))
            except LLMError as exc:
                outputs.append(f"raised:{exc.kind}")
        return chaos.applied, outputs

    first = trace(42)
    assert first == trace(42)
    assert 10 < len(first[0]) < 50  # roughly rate 0.5
    assert trace(43) != first


def test_mode_choice_is_independent_of_inner_output() -> None:
    modes = ["empty_output", "refusal", "incomplete"]
    a = ChaosTransport(RecordingInner(tool_response()), seed=9, rate=0.4, modes=modes)
    b = ChaosTransport(RecordingInner(prose_response()), seed=9, rate=0.4, modes=modes)
    for _ in range(40):
        run(a)
        run(b)
    assert [m for _, m in a.applied] == [m for _, m in b.applied]


def test_inner_response_is_never_mutated() -> None:
    original = tool_response()
    snapshot = repr(original)
    chaos = ChaosTransport(RecordingInner(original), seed=3, rate=1.0, modes=NON_RAISE)
    for _ in range(40):
        run(chaos)
    assert repr(original) == snapshot


def test_applied_records_task_and_mode_and_emits_trace_event(monkeypatch: pytest.MonkeyPatch) -> None:
    events: list[tuple[str, dict[str, Any]]] = []
    monkeypatch.setattr(tracing, "event", lambda name, **data: events.append((name, data)))
    _, chaos = mutate("refusal", prose_response())
    assert chaos.applied == [("agent", "refusal")]
    assert events == [("chaos", {"type": "chaos", "task": "agent", "mode": "refusal", "call_index": 0})]


@pytest.mark.parametrize(("mode", "kind", "status"), [(m, k, s) for m, (k, s) in RAISE_MODES.items()])
def test_raise_modes_raise_without_calling_inner(mode: str, kind: str, status: int | None) -> None:
    inner = RecordingInner(prose_response())
    chaos = ChaosTransport(inner, seed=1, rate=1.0, modes=[mode])
    with pytest.raises(LLMError) as err:
        run(chaos)
    assert err.value.kind == kind and err.value.status_code == status
    assert inner.calls == [] and chaos.applied == [("agent", mode)]


def test_resilient_over_chaos_retries_injected_faults_deterministically() -> None:
    def attempt() -> tuple[str, int]:
        chaos = ChaosTransport(RecordingInner(prose_response()), seed=5, rate=0.7, modes=["raise_timeout", "raise_server"])
        resilient = ResilientTransport(chaos, max_retries=6, backoff_base_s=0.0, sleep=lambda _: None)
        return run(resilient).text, chaos.calls

    assert attempt() == attempt()
    assert attempt()[0] == prose_response().text


# ------------------------------------------------------------- tool-call modes


def test_extra_field_adds_protected_fields_and_syncs_raw_item() -> None:
    out, _ = mutate("extra_field", tool_response())
    args = first_args(out)
    assert args["verified"] is True and args["party_id"] == "P9" and args["reason"] == NOTE_ARGS["reason"]
    assert raw_calls(out)[0]["arguments"] == out.function_calls[0].arguments
    assert EXTRA_FIELDS.keys() <= args.keys()


def test_missing_field_drops_exactly_one_argument() -> None:
    out, _ = mutate("missing_field", tool_response())
    assert len(first_args(out)) == len(NOTE_ARGS) - 1


def test_invalid_enum_prefers_enum_valued_arguments() -> None:
    out, _ = mutate("invalid_enum", tool_response())
    changed = [k for k, v in first_args(out).items() if v == INVALID_ENUM]
    assert len(changed) == 1 and changed[0] in ENUM_KEYS


def test_wrong_types_changes_one_argument_type() -> None:
    out, _ = mutate("wrong_types", tool_response())
    args = first_args(out)
    assert sum(type(args[k]) is not type(NOTE_ARGS[k]) for k in NOTE_ARGS) == 1


def test_bad_tool_args_corrupts_one_call_and_its_raw_item() -> None:
    out, _ = mutate("bad_tool_args", tool_response())
    bad = [c for c in out.function_calls if c.arguments in BAD_TOOL_ARGS]
    assert len(bad) == 1
    assert {i["call_id"]: i["arguments"] for i in raw_calls(out)}[bad[0].call_id] == bad[0].arguments


def test_hallucinated_tool_appends_forbidden_call() -> None:
    out, _ = mutate("hallucinated_tool", tool_response())
    extra = out.function_calls[-1]
    assert extra.name in HALLUCINATED_TOOLS and len(out.function_calls) == 3
    assert raw_calls(out)[-1]["call_id"] == extra.call_id


def test_wrong_call_id_desyncs_one_call_from_its_output_item() -> None:
    out, _ = mutate("wrong_call_id", tool_response())
    assert {c.call_id for c in out.function_calls} != {i["call_id"] for i in raw_calls(out)}


def test_duplicate_calls_repeat_a_call_id() -> None:
    out, _ = mutate("duplicate_calls", tool_response())
    ids = [c.call_id for c in out.function_calls]
    assert len(ids) == 3 and len(set(ids)) == 2


def test_parallel_calls_add_a_sibling_with_new_id() -> None:
    out, _ = mutate("parallel_calls", tool_response())
    assert len({c.call_id for c in out.function_calls}) == 3
    assert out.function_calls[-1].name in {"note_caller_context", "select_claim"}


# ------------------------------------------------------------- prose and generic modes


def test_prompt_injection_prefixes_prose_and_fills_empty_text() -> None:
    out, _ = mutate("prompt_injection_text", prose_response())
    assert out.text.startswith(INJECTION_TEXT) and raw_texts(out) == [out.text]
    empty, _ = mutate("prompt_injection_text", prose_response(""))
    assert empty.text == INJECTION_TEXT


def test_colon_and_em_dash_in_prose() -> None:
    out, _ = mutate("colon_and_em_dash", prose_response())
    assert ":" in out.text and "—" in out.text and raw_texts(out) == [out.text]


def test_empty_output_clears_everything_but_reasoning() -> None:
    out, _ = mutate("empty_output", tool_response())
    assert out.text == "" and out.refusal is None and out.function_calls == []
    assert [i["type"] for i in out.raw_output_items] == ["reasoning"]


def test_refusal_replaces_output_with_refusal_message() -> None:
    out, _ = mutate("refusal", tool_response())
    assert out.refusal == CHAOS_REFUSAL and out.function_calls == [] and out.text == ""


def test_incomplete_truncates_and_drops_calls() -> None:
    out, _ = mutate("incomplete", prose_response("abcdefgh"))
    assert (out.status, out.incomplete_reason, out.text) == ("incomplete", "max_output_tokens", "abcd")
    calls, _ = mutate("incomplete", tool_response())
    assert calls.function_calls == [] and raw_calls(calls) == []


# ------------------------------------------------------------- applicability and determinism


@pytest.mark.parametrize("mode", NON_RAISE)
def test_every_mode_is_deterministic_for_a_fixed_seed(mode: str) -> None:
    source = prose_response() if mode in PROSE_MODES else tool_response()
    assert repr(mutate(mode, source, seed=11)[0]) == repr(mutate(mode, source, seed=11)[0])


@pytest.mark.parametrize("mode", CALL_MODES)
def test_call_mode_on_prose_passes_through_unrecorded(mode: str) -> None:
    original = prose_response()
    out, chaos = mutate(mode, original)
    assert out is original and chaos.applied == []


@pytest.mark.parametrize("mode", PROSE_MODES)
def test_prose_mode_on_tool_calls_passes_through_unrecorded(mode: str) -> None:
    original = tool_response()
    out, chaos = mutate(mode, original)
    assert out is original and chaos.applied == []


def test_inapplicable_mode_falls_back_to_next_configured_mode() -> None:
    chaos = ChaosTransport(RecordingInner(prose_response()), seed=1, rate=1.0, modes=["wrong_call_id", "missing_field", "refusal"])
    for _ in range(12):
        assert run(chaos).refusal == CHAOS_REFUSAL
    assert {mode for _, mode in chaos.applied} == {"refusal"}


def test_full_rate_with_all_modes_mutates_or_raises_every_call() -> None:
    chaos = ChaosTransport(RecordingInner(tool_response()), seed=2, rate=1.0)
    for _ in range(80):
        try:
            run(chaos)
        except LLMError:
            pass
    assert len(chaos.applied) == 80 and {m for _, m in chaos.applied} <= set(MODES)


def test_concurrent_calls_are_counted_exactly() -> None:
    chaos = ChaosTransport(RecordingInner(prose_response()), seed=4, rate=0.5, modes=["refusal", "empty_output"])

    def worker() -> None:
        for _ in range(25):
            run(chaos)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert chaos.calls == 200 and 0 < len(chaos.applied) < 200
