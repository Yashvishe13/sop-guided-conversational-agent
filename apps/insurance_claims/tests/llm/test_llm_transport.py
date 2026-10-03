"""OpenAITransport (SDK mapping, request shape, error hygiene) and ResilientTransport (retries, deadline)."""

from __future__ import annotations

import json
from typing import Any

import httpx
import httpx2
import openai
import pytest
from openai.types.responses import Response

from insurance_claims.config import ConfigError, Settings
from insurance_claims.llm.base import LLMError, LLMResponse, ModelTransport
from insurance_claims.llm.openai_transport import (
    ENCRYPTED_REASONING,
    OpenAITransport,
    map_exception,
    to_llm_response,
)
from insurance_claims.llm.resilient import (
    MAX_BACKOFF_S,
    ResilientTransport,
    backoff_delay,
)
from insurance_claims.observability import tracing

FAKE_KEY = "sk-test-FAKEKEY0123456789abcdef"
PII_INPUT = "My name is Margaret Chen, born 1985-03-15"


# ---------------------------------------------------------------------- builders


def make_response(
    output: list[dict[str, Any]],
    *,
    status: str | None = "completed",
    usage: dict[str, Any] | None = None,
    incomplete_reason: str | None = None,
    error: dict[str, Any] | None = None,
) -> Response:
    payload: dict[str, Any] = {
        "id": "resp_123",
        "created_at": 1_700_000_000,
        "model": "gpt-5.6-luna",
        "object": "response",
        "parallel_tool_calls": True,
        "tool_choice": "auto",
        "tools": [],
        "output": output,
    }
    if status is not None:
        payload["status"] = status
    if usage is not None:
        payload["usage"] = usage
    if incomplete_reason is not None:
        payload["incomplete_details"] = {"reason": incomplete_reason}
    if error is not None:
        payload["error"] = error
    return Response.model_validate(payload)


def message(*parts: dict[str, Any], msg_id: str = "msg_1", phase: str | None = None) -> dict[str, Any]:
    item: dict[str, Any] = {
        "type": "message",
        "id": msg_id,
        "role": "assistant",
        "status": "completed",
        "content": list(parts),
    }
    if phase is not None:
        item["phase"] = phase
    return item


def text_part(text: str) -> dict[str, Any]:
    return {"type": "output_text", "text": text, "annotations": []}


def refusal_part(text: str) -> dict[str, Any]:
    return {"type": "refusal", "refusal": text}


def call_item(call_id: str, name: str, arguments: str, item_id: str | None = None) -> dict[str, Any]:
    item = {"type": "function_call", "call_id": call_id, "name": name, "arguments": arguments, "status": "completed"}
    if item_id:
        item["id"] = item_id
    return item


def reasoning_item(rs_id: str = "rs_1", encrypted: str = "gAAAAencrypted") -> dict[str, Any]:
    return {"type": "reasoning", "id": rs_id, "summary": [], "encrypted_content": encrypted}


USAGE = {
    "input_tokens": 120,
    "output_tokens": 45,
    "total_tokens": 165,
    "input_tokens_details": {"cached_tokens": 64, "cache_write_tokens": 0},
    "output_tokens_details": {"reasoning_tokens": 30},
}


class FakeResponses:
    def __init__(self, result: Any = None, exc: BaseException | None = None) -> None:
        self.result = result
        self.exc = exc
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        if self.exc is not None:
            raise self.exc
        return self.result


class FakeClient:
    def __init__(self, result: Any = None, exc: BaseException | None = None) -> None:
        self.responses = FakeResponses(result, exc)


def settings(**overrides: Any) -> Settings:
    return Settings(openai_api_key=FAKE_KEY, model_timeout_s=25.0).with_overrides(**overrides)


def transport_with(result: Any = None, exc: BaseException | None = None) -> tuple[OpenAITransport, FakeClient]:
    client = FakeClient(result, exc)
    return OpenAITransport(settings(), client=client), client


def call(transport: Any, **overrides: Any) -> LLMResponse:
    kwargs: dict[str, Any] = {
        "task": "agent",
        "instructions": "Help the caller.",
        "input": [{"role": "user", "content": PII_INPUT}],
    }
    kwargs.update(overrides)
    return transport.create(**kwargs)


# ------------------------------------------------------------ response mapping


def test_message_text_is_concatenated_and_usage_mapped() -> None:
    resp = make_response([message(text_part('{"a": '), text_part("1}"))], usage=USAGE)
    transport, _ = transport_with(resp)
    out = call(transport)
    assert out.status == "completed"
    assert out.text == '{"a": 1}'
    assert json.loads(out.text) == {"a": 1}
    assert out.refusal is None
    assert out.function_calls == []
    assert out.response_id == "resp_123"
    assert out.model == "gpt-5.6-luna"
    assert out.incomplete_reason is None
    assert out.usage.as_dict() == {"input": 120, "output": 45, "total": 165, "reasoning": 30, "cached": 64}


def test_text_matches_sdk_output_text_for_plain_messages() -> None:
    resp = make_response([message(text_part("Hello ")), message(text_part("there"), msg_id="msg_2")])
    assert to_llm_response(resp).text == resp.output_text == "Hello there"


def test_refusal_content_is_surfaced_and_not_mixed_into_text() -> None:
    resp = make_response([message(refusal_part("I can't help with that."))])
    out = to_llm_response(resp)
    assert out.refusal == "I can't help with that."
    assert out.text == ""


def test_multiple_function_calls_preserve_order_and_raw_arguments() -> None:
    resp = make_response(
        [
            reasoning_item(),
            call_item("call_a", "get_claim", '{"case_id": "CL-1"}', item_id="fc_a"),
            call_item("call_b", "list_documents", "{not json", item_id="fc_b"),
        ]
    )
    out = to_llm_response(resp)
    assert [(c.call_id, c.name, c.arguments, c.item_id) for c in out.function_calls] == [
        ("call_a", "get_claim", '{"case_id": "CL-1"}', "fc_a"),
        ("call_b", "list_documents", "{not json", "fc_b"),
    ]
    assert out.text == ""


def test_raw_output_items_echo_reasoning_and_calls_in_api_shape() -> None:
    resp = make_response(
        [
            reasoning_item("rs_9", "ENC=="),
            call_item("call_a", "get_claim", "{}", item_id="fc_a"),
            message(text_part("done")),
        ]
    )
    raw = to_llm_response(resp).raw_output_items
    assert [item["type"] for item in raw] == ["reasoning", "function_call", "message"]
    assert raw[0] == {"id": "rs_9", "summary": [], "type": "reasoning", "encrypted_content": "ENC=="}
    assert raw[1]["call_id"] == "call_a" and raw[1]["arguments"] == "{}"
    assert "async_" not in raw[1]
    json.dumps(raw)  # plain JSON-serializable dicts, safe to echo as input
    for item in raw:
        assert None not in item.values()


def test_function_call_alias_fields_use_api_names() -> None:
    item = call_item("call_a", "get_claim", "{}")
    item["async"] = False
    raw = to_llm_response(make_response([item])).raw_output_items
    assert raw[0]["async"] is False and "async_" not in raw[0]


def test_incomplete_status_and_reason() -> None:
    resp = make_response([message(text_part('{"identity": '))], status="incomplete", incomplete_reason="max_output_tokens")
    out = to_llm_response(resp)
    assert out.status == "incomplete"
    assert out.incomplete_reason == "max_output_tokens"
    assert out.text == '{"identity": '


def test_missing_status_defaults_to_completed() -> None:
    resp = make_response([message(text_part("hi"))], status=None)
    assert resp.status is None
    assert to_llm_response(resp).status == "completed"


@pytest.mark.parametrize(
    ("status", "expected", "reason"),
    [
        ("failed", "failed", "server_error"),
        ("cancelled", "failed", "server_error"),
        ("in_progress", "incomplete", "in_progress"),
        ("queued", "incomplete", "queued"),
    ],
)
def test_non_terminal_and_failed_statuses(status: str, expected: str, reason: str) -> None:
    error = {"code": "server_error", "message": f"echo {PII_INPUT}"} if expected == "failed" else None
    out = to_llm_response(make_response([], status=status, error=error))
    assert out.status == expected
    assert out.incomplete_reason == reason
    assert "Margaret" not in str(out)


def test_missing_usage_yields_zeros() -> None:
    out = to_llm_response(make_response([message(text_part("x"))]))
    assert out.usage.as_dict() == {"input": 0, "output": 0, "total": 0, "reasoning": 0, "cached": 0}


def test_usage_with_missing_details_is_tolerated() -> None:
    resp = make_response([])
    resp.usage = type(
        "U", (), {"input_tokens": 5, "output_tokens": 2, "total_tokens": 7, "output_tokens_details": None, "input_tokens_details": None}
    )()
    assert to_llm_response(resp).usage.as_dict() == {"input": 5, "output": 2, "total": 7, "reasoning": 0, "cached": 0}


def test_final_answer_phase_wins_over_commentary() -> None:
    resp = make_response(
        [
            message(text_part("Let me check that claim."), msg_id="m1", phase="commentary"),
            message(text_part('{"ok": true}'), msg_id="m2", phase="final_answer"),
        ]
    )
    assert to_llm_response(resp).text == '{"ok": true}'


def test_model_constructed_response_and_dict_items_are_supported() -> None:
    resp = Response.model_construct(id="r1", model="m", output=[message(text_part("plain dict"))], status="completed")
    out = to_llm_response(resp)
    assert out.text == "plain dict"
    assert out.raw_output_items[0]["type"] == "message"


def test_unparseable_response_becomes_unknown_llm_error() -> None:
    class Weird:
        type = "message"
        content = None

    transport, _ = transport_with(Response.model_construct(id="r", model="m", output=[Weird()], status="completed"))
    with pytest.raises(LLMError) as info:
        call(transport)
    assert info.value.kind == "unknown"
    assert not info.value.retryable


# ---------------------------------------------------------------- request shape


def test_minimal_request_shape() -> None:
    transport, client = transport_with(make_response([message(text_part("ok"))]))
    call(transport, max_output_tokens=321)
    (kwargs,) = client.responses.calls
    assert kwargs == {
        "model": "gpt-5.6-luna",
        "instructions": "Help the caller.",
        "input": [{"role": "user", "content": PII_INPUT}],
        "store": False,
        "max_output_tokens": 321,
        "timeout": 25.0,
    }
    assert "task" not in kwargs


def test_reasoning_and_timeout_are_passed_through() -> None:
    transport, client = transport_with(make_response([message(text_part("Hello"))]))
    call(transport, reasoning_effort="low", timeout=7.5)
    kwargs = client.responses.calls[0]
    assert "text" not in kwargs
    assert kwargs["reasoning"] == {"effort": "low"}
    assert kwargs["timeout"] == 7.5
    assert kwargs["store"] is False
    assert "include" not in kwargs and "tools" not in kwargs


@pytest.mark.parametrize("effort", [None, ""])
def test_reasoning_omitted_without_effort(effort: str | None) -> None:
    transport, client = transport_with(make_response([]))
    call(transport, reasoning_effort=effort)
    assert "reasoning" not in client.responses.calls[0]


def test_tools_add_encrypted_reasoning_include_and_tool_options() -> None:
    transport, client = transport_with(make_response([]))
    tools = [{"type": "function", "name": "get_claim", "parameters": {"type": "object"}, "strict": True}]
    call(transport, task="process_case", tools=tools, tool_choice="auto", parallel_tool_calls=False)
    kwargs = client.responses.calls[0]
    assert kwargs["tools"] is tools
    assert kwargs["include"] == [ENCRYPTED_REASONING]
    assert kwargs["tool_choice"] == "auto"
    assert kwargs["parallel_tool_calls"] is False
    assert kwargs["store"] is False


def test_forced_tool_choice_dict_is_passed_verbatim() -> None:
    transport, client = transport_with(make_response([]))
    choice = {"type": "function", "name": "get_claim"}
    call(transport, tools=[{"type": "function", "name": "get_claim"}], tool_choice=choice)
    assert client.responses.calls[0]["tool_choice"] is choice


def test_empty_tools_list_means_no_tools_and_no_tool_options() -> None:
    transport, client = transport_with(make_response([]))
    call(transport, tools=[], tool_choice="auto", parallel_tool_calls=True)
    kwargs = client.responses.calls[0]
    for key in ("tools", "include", "tool_choice", "parallel_tool_calls"):
        assert key not in kwargs


def test_transport_satisfies_protocol_and_hides_key_in_repr() -> None:
    transport, _ = transport_with(make_response([]))
    assert isinstance(transport, ModelTransport)
    assert transport.model_name == "gpt-5.6-luna"
    assert FAKE_KEY not in repr(transport) and FAKE_KEY not in str(transport)


def test_real_client_is_built_without_sdk_retries_and_traced(monkeypatch: pytest.MonkeyPatch) -> None:
    wrapped: list[Any] = []

    def fake_wrap(client: Any) -> Any:
        wrapped.append(client)
        return client

    monkeypatch.setattr(tracing, "wrap_openai", fake_wrap)
    transport = OpenAITransport(settings(model_timeout_s=12.0))
    client = transport._client
    assert isinstance(client, openai.OpenAI)
    assert client.max_retries == 0
    assert client.timeout == 12.0
    assert wrapped == [client]
    assert FAKE_KEY not in repr(transport)


def test_missing_key_without_client_fails_clearly() -> None:
    with pytest.raises(ConfigError) as info:
        OpenAITransport(Settings(openai_api_key=None))
    assert "OPENAI_API_KEY" in str(info.value)


def test_injected_client_works_without_key() -> None:
    transport = OpenAITransport(Settings(openai_api_key=None), client=FakeClient(make_response([])))
    assert call(transport).status == "completed"


# ---------------------------------------------------------------- error mapping

REQUEST = httpx2.Request(
    "POST",
    "https://api.openai.com/v1/responses",
    headers={"Authorization": f"Bearer {FAKE_KEY}"},
    content=json.dumps({"input": PII_INPUT}).encode(),
)


def status_error(cls: type, status: int, *, code: str | None = None, headers: dict[str, str] | None = None) -> Exception:
    response = httpx2.Response(status, request=REQUEST, headers={"x-request-id": "req_abc123", **(headers or {})})
    body = {
        "message": f"Incorrect API key provided: {FAKE_KEY}. Input was: {PII_INPUT}",
        "type": "invalid_request_error",
        "code": code,
        "param": "input",
    }
    return cls(f"Error code: {status} - {body}", response=response, body=body)


ERROR_CASES: list[tuple[str, Any, str, int | None]] = [
    ("timeout", lambda: openai.APITimeoutError(request=REQUEST), "timeout", None),
    ("connection", lambda: openai.APIConnectionError(message=f"conn failed {FAKE_KEY}", request=REQUEST), "connection", None),
    ("rate_limit", lambda: status_error(openai.RateLimitError, 429, code="rate_limit_exceeded"), "rate_limit", 429),
    ("quota", lambda: status_error(openai.RateLimitError, 429, code="insufficient_quota"), "budget", 429),
    ("internal", lambda: status_error(openai.InternalServerError, 500), "server", 500),
    ("bad_gateway", lambda: status_error(openai.InternalServerError, 502), "server", 502),
    ("generic_5xx", lambda: status_error(openai.APIStatusError, 503), "server", 503),
    ("auth", lambda: status_error(openai.AuthenticationError, 401, code="invalid_api_key"), "auth", 401),
    ("permission", lambda: status_error(openai.PermissionDeniedError, 403), "auth", 403),
    ("bad_request", lambda: status_error(openai.BadRequestError, 400, code="invalid_value"), "bad_request", 400),
    ("not_found", lambda: status_error(openai.NotFoundError, 404, code="model_not_found"), "bad_request", 404),
    ("conflict", lambda: status_error(openai.ConflictError, 409), "bad_request", 409),
    ("unprocessable", lambda: status_error(openai.UnprocessableEntityError, 422), "bad_request", 422),
    ("request_timeout", lambda: status_error(openai.APIStatusError, 408), "timeout", 408),
    ("validation", lambda: openai.APIResponseValidationError(httpx2.Response(200, request=REQUEST), {"x": PII_INPUT}), "unknown", 200),
    ("sdk_generic", lambda: openai.OpenAIError(f"boom {FAKE_KEY}"), "unknown", None),
    ("runtime", lambda: RuntimeError(f"unexpected {FAKE_KEY} {PII_INPUT}"), "unknown", None),
    ("httpx2_timeout", lambda: httpx2.ReadTimeout("read timed out", request=REQUEST), "timeout", None),
    ("httpx_connect", lambda: httpx.ConnectError("refused"), "connection", None),
    ("builtin_timeout", lambda: TimeoutError("slow"), "timeout", None),
]


@pytest.mark.parametrize(("label", "factory", "kind", "status_code"), ERROR_CASES, ids=[c[0] for c in ERROR_CASES])
def test_sdk_exceptions_map_to_llm_error_kinds(label: str, factory: Any, kind: str, status_code: int | None) -> None:
    transport, _ = transport_with(exc=factory())
    with pytest.raises(LLMError) as info:
        call(transport)
    err = info.value
    assert err.kind == kind
    assert err.status_code == status_code
    assert err.retryable is (kind in {"timeout", "rate_limit", "connection", "server"})
    rendered = " ".join([str(err), repr(err), repr(err.args)])
    assert FAKE_KEY not in rendered
    assert "sk-test" not in rendered
    assert "Margaret" not in rendered and "1985" not in rendered
    assert err.__cause__ is None and err.__suppress_context__ is True


def test_error_message_keeps_safe_diagnostics() -> None:
    err = map_exception(status_error(openai.BadRequestError, 400, code="invalid_value"), secrets=[FAKE_KEY])
    message = str(err)
    assert message.startswith("bad_request: BadRequestError")
    assert "status=400" in message
    assert "code=invalid_value" in message
    assert "param=input" in message
    assert "request_id=req_abc123" in message


def test_secret_in_error_code_is_scrubbed() -> None:
    exc = status_error(openai.BadRequestError, 400, code=FAKE_KEY)
    err = map_exception(exc, secrets=[FAKE_KEY])
    assert FAKE_KEY not in str(err)
    assert "[REDACTED]" in str(err)


def test_unknown_sk_style_token_is_scrubbed_even_without_secret_list() -> None:
    exc = status_error(openai.BadRequestError, 400, code="sk-otherKEY1234567890")
    assert "sk-otherKEY" not in str(map_exception(exc))


def test_llm_error_from_client_passes_through_unchanged() -> None:
    original = LLMError("deadline", "inner deadline")
    transport, _ = transport_with(exc=original)
    with pytest.raises(LLMError) as info:
        call(transport)
    assert info.value is original


@pytest.mark.parametrize(
    ("headers", "expected"),
    [
        ({"retry-after-ms": "1500"}, 1.5),
        ({"retry-after": "3"}, 3.0),
        ({"retry-after": "9999"}, 60.0),
        ({"retry-after": "Wed, 21 Oct 2026 07:28:00 GMT"}, None),
        ({}, None),
    ],
)
def test_retry_after_hint_is_parsed(headers: dict[str, str], expected: float | None) -> None:
    err = map_exception(status_error(openai.RateLimitError, 429, headers=headers))
    assert getattr(err, "retry_after_s", None) == expected


# ------------------------------------------------------------ ResilientTransport


class ScriptedInner:
    """Returns or raises the scripted outcomes in order; records each call's kwargs."""

    model_name = "scripted-model"

    def __init__(self, outcomes: list[Any], clock: FakeClock | None = None, latency: float = 0.0) -> None:
        self.outcomes = list(outcomes)
        self.calls: list[dict[str, Any]] = []
        self.clock = clock
        self.latency = latency

    def create(self, **kwargs: Any) -> LLMResponse:
        self.calls.append(kwargs)
        if self.clock is not None:
            self.clock.now += self.latency
        outcome = self.outcomes.pop(0) if len(self.outcomes) > 1 else self.outcomes[0]
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


class FakeClock:
    def __init__(self, end: float | None = None) -> None:
        self.now = 0.0
        self.end = end
        self.sleeps: list[float] = []

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds
        if self.end is not None:
            assert self.now <= self.end, "slept past the deadline"

    def remaining(self) -> float | None:
        return None if self.end is None else self.end - self.now


OK = LLMResponse(status="completed", text="ok")


@pytest.fixture
def events(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, dict[str, Any]]]:
    recorded: list[tuple[str, dict[str, Any]]] = []

    def fake_event(name: str, type: str = "guard", **data: Any) -> None:
        recorded.append((name, {"type": type, **data}))

    monkeypatch.setattr(tracing, "event", fake_event)
    return recorded


def resilient(inner: Any, clock: FakeClock, *, max_retries: int = 2, base: float = 0.5) -> ResilientTransport:
    return ResilientTransport(inner, max_retries=max_retries, backoff_base_s=base, sleep=clock.sleep, deadline=clock.remaining)


def test_success_needs_no_retry(events: list) -> None:
    clock = FakeClock()
    inner = ScriptedInner([OK])
    assert call(resilient(inner, clock)) is OK
    assert clock.sleeps == [] and len(inner.calls) == 1 and events == []


def test_retryable_error_is_retried_with_backoff(events: list) -> None:
    clock = FakeClock()
    inner = ScriptedInner([LLMError("timeout"), LLMError("server", status_code=503), OK])
    out = call(resilient(inner, clock), max_output_tokens=77)
    assert out is OK
    assert len(inner.calls) == 3
    assert clock.sleeps == [backoff_delay(0, 0.5), backoff_delay(1, 0.5)]
    for kwargs in inner.calls:
        assert kwargs["task"] == "agent" and kwargs["max_output_tokens"] == 77
    retries = [data for name, data in events if name == "llm_retry"]
    assert [(r["attempt"], r["kind"]) for r in retries] == [(1, "timeout"), (2, "server")]
    assert retries[1]["status_code"] == 503
    assert all(PII_INPUT not in json.dumps(data) for _, data in events)


def test_gives_up_after_max_retries_and_reraises_last_error(events: list) -> None:
    clock = FakeClock()
    final = LLMError("rate_limit", status_code=429)
    inner = ScriptedInner([LLMError("rate_limit"), LLMError("rate_limit"), final])
    with pytest.raises(LLMError) as info:
        call(resilient(inner, clock, max_retries=2))
    assert info.value is final
    assert len(inner.calls) == 3
    assert len(clock.sleeps) == 2
    assert [name for name, _ in events][-1] == "llm_retry_exhausted"


@pytest.mark.parametrize("kind", ["bad_request", "auth", "budget", "unknown", "deadline"])
def test_non_retryable_errors_are_not_retried(kind: str) -> None:
    clock = FakeClock()
    inner = ScriptedInner([LLMError(kind), OK])
    with pytest.raises(LLMError) as info:
        call(resilient(inner, clock))
    assert info.value.kind == kind
    assert len(inner.calls) == 1 and clock.sleeps == []


def test_zero_retries_means_single_attempt() -> None:
    clock = FakeClock()
    inner = ScriptedInner([LLMError("connection"), OK])
    with pytest.raises(LLMError):
        call(resilient(inner, clock, max_retries=0))
    assert len(inner.calls) == 1 and clock.sleeps == []


def test_non_llm_exceptions_propagate_without_retry() -> None:
    clock = FakeClock()
    inner = ScriptedInner([ValueError("bug"), OK])
    with pytest.raises(ValueError):
        call(resilient(inner, clock))
    assert len(inner.calls) == 1


def test_backoff_is_exponential_deterministic_and_capped() -> None:
    delays = [backoff_delay(a, 0.5) for a in range(8)]
    assert delays == [backoff_delay(a, 0.5) for a in range(8)]
    assert delays[0] == 0.5
    for attempt, delay in enumerate(delays):
        raw = 0.5 * 2**attempt
        assert delay <= MAX_BACKOFF_S
        assert min(raw, MAX_BACKOFF_S) <= delay <= min(raw * 1.1, MAX_BACKOFF_S)
    assert delays[-1] == MAX_BACKOFF_S
    assert backoff_delay(3, 0.0) == 0.0


def test_deadline_stops_retry_instead_of_sleeping_past_it(events: list) -> None:
    clock = FakeClock(end=1.0)
    inner = ScriptedInner([LLMError("server")], clock=clock, latency=0.3)
    with pytest.raises(LLMError) as info:
        call(resilient(inner, clock, max_retries=5))
    assert info.value.kind == "deadline"
    assert not info.value.retryable
    assert clock.sleeps == [0.5]
    assert len(inner.calls) == 2
    assert inner.calls[0]["timeout"] == pytest.approx(1.0)
    assert inner.calls[1]["timeout"] == pytest.approx(0.2)
    assert events[-1][0] == "llm_deadline"


def test_deadline_already_passed_does_not_call_inner() -> None:
    clock = FakeClock(end=0.0)
    inner = ScriptedInner([OK])
    with pytest.raises(LLMError) as info:
        call(resilient(inner, clock))
    assert info.value.kind == "deadline"
    assert inner.calls == [] and clock.sleeps == []


def test_backoff_equal_to_remaining_time_is_refused() -> None:
    clock = FakeClock(end=0.5)
    inner = ScriptedInner([LLMError("timeout")])
    with pytest.raises(LLMError) as info:
        call(resilient(inner, clock, base=0.5))
    assert info.value.kind == "deadline"
    assert clock.sleeps == []


def test_attempt_timeout_is_clipped_to_remaining_time() -> None:
    clock = FakeClock(end=10.0)
    inner = ScriptedInner([OK])
    call(resilient(inner, clock), timeout=25.0)
    assert inner.calls[0]["timeout"] == 10.0
    clock2 = FakeClock(end=100.0)
    inner2 = ScriptedInner([OK])
    call(resilient(inner2, clock2), timeout=25.0)
    assert inner2.calls[0]["timeout"] == 25.0


def test_no_deadline_passes_timeout_through() -> None:
    inner = ScriptedInner([OK])
    transport = ResilientTransport(inner, max_retries=1, backoff_base_s=0.1, sleep=lambda s: None)
    call(transport)
    assert inner.calls[0]["timeout"] is None
    call(transport, timeout=3.0)
    assert inner.calls[1]["timeout"] == 3.0


def test_retry_after_hint_stretches_backoff_but_is_capped() -> None:
    clock = FakeClock()
    slow = LLMError("rate_limit", status_code=429)
    slow.retry_after_s = 3.0  # type: ignore[attr-defined]
    huge = LLMError("rate_limit", status_code=429)
    huge.retry_after_s = 60.0  # type: ignore[attr-defined]
    inner = ScriptedInner([slow, huge, OK])
    call(resilient(inner, clock))
    assert clock.sleeps == [3.0, MAX_BACKOFF_S]


def test_invalid_resilient_configuration_is_rejected() -> None:
    with pytest.raises(ValueError):
        ResilientTransport(ScriptedInner([OK]), max_retries=-1, backoff_base_s=0.5)
    with pytest.raises(ValueError):
        ResilientTransport(ScriptedInner([OK]), max_retries=1, backoff_base_s=-0.1)


def test_resilient_over_openai_transport_retries_sdk_rate_limits() -> None:
    clock = FakeClock()
    client = FakeClient(exc=status_error(openai.RateLimitError, 429, headers={"retry-after-ms": "250"}))
    transport = resilient(OpenAITransport(settings(), client=client), clock, max_retries=1, base=0.1)
    with pytest.raises(LLMError) as info:
        call(transport)
    assert info.value.kind == "rate_limit"
    assert len(client.responses.calls) == 2
    assert clock.sleeps == [0.25]
    assert FAKE_KEY not in str(info.value)


# ------------------------------------------- wire level: real SDK, mocked HTTP (no network)


class WireServer:
    """httpx2 MockTransport handler that replays scripted HTTP outcomes and records requests."""

    def __init__(self, outcomes: list[Any]) -> None:
        self.outcomes = list(outcomes)
        self.requests: list[httpx2.Request] = []

    def __call__(self, request: httpx2.Request) -> httpx2.Response:
        self.requests.append(request)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        status, body, headers = outcome
        return httpx2.Response(status, json=body, headers=headers)

    def body(self, index: int) -> dict[str, Any]:
        return json.loads(self.requests[index].content)


def wire_transport(server: WireServer) -> OpenAITransport:
    client = openai.OpenAI(api_key=FAKE_KEY, max_retries=0, http_client=httpx2.Client(transport=httpx2.MockTransport(server)))
    return OpenAITransport(settings(), client=client)


def wire_response(output: list[dict[str, Any]]) -> tuple[int, dict[str, Any], dict[str, str]]:
    return 200, make_response(output, usage=USAGE).model_dump(mode="json", exclude_none=True, by_alias=True), {}


def test_wire_request_body_and_key_only_in_auth_header() -> None:
    server = WireServer([wire_response([message(text_part("Hello there"))])])
    out = call(wire_transport(server), reasoning_effort="low")
    assert out.text == "Hello there" and out.usage.cached_tokens == 64
    body = server.body(0)
    assert body["store"] is False
    assert body["model"] == "gpt-5.6-luna"
    assert body["reasoning"] == {"effort": "low"}
    assert "task" not in body and "include" not in body
    assert server.requests[0].headers["authorization"] == f"Bearer {FAKE_KEY}"
    assert FAKE_KEY not in server.requests[0].content.decode()


def test_wire_tool_loop_round_trips_reasoning_and_call_ids() -> None:
    first = wire_response(
        [reasoning_item("rs_1", "ENCRYPTED=="), call_item("call_7", "get_claim", '{"case_id": "CL-2048"}', item_id="fc_7")]
    )
    second = wire_response([message(text_part("Your claim CL-2048 was denied."))])
    server = WireServer([first, second])
    transport = wire_transport(server)
    tools = [
        {
            "type": "function",
            "name": "get_claim",
            "parameters": {
                "type": "object",
                "properties": {"case_id": {"type": "string"}},
                "required": ["case_id"],
                "additionalProperties": False,
            },
            "strict": True,
        }
    ]
    turn1 = call(transport, task="process_case", tools=tools)
    assert server.body(0)["include"] == [ENCRYPTED_REASONING]
    (fc,) = turn1.function_calls
    follow_up = [*turn1.raw_output_items, {"type": "function_call_output", "call_id": fc.call_id, "output": '{"status": "denied"}'}]
    turn2 = call(transport, task="process_case", tools=tools, input=[{"role": "user", "content": "status?"}, *follow_up])
    assert turn2.text == "Your claim CL-2048 was denied."
    echoed = server.body(1)["input"]
    assert echoed[1] == {"id": "rs_1", "summary": [], "type": "reasoning", "encrypted_content": "ENCRYPTED=="}
    assert echoed[2]["type"] == "function_call" and echoed[2]["call_id"] == "call_7"
    assert echoed[3] == {"type": "function_call_output", "call_id": "call_7", "output": '{"status": "denied"}'}


@pytest.mark.parametrize(
    ("status", "code", "kind"),
    [
        (401, "invalid_api_key", "auth"),
        (429, "rate_limit_exceeded", "rate_limit"),
        (500, "server_error", "server"),
        (400, "invalid_value", "bad_request"),
    ],
)
def test_wire_http_errors_are_mapped_once_without_sdk_retries(status: int, code: str, kind: str) -> None:
    body = {
        "error": {
            "message": f"Incorrect API key provided: {FAKE_KEY}; input {PII_INPUT}",
            "type": "invalid_request_error",
            "code": code,
            "param": None,
        }
    }
    server = WireServer([(status, body, {"retry-after-ms": "750"})])
    with pytest.raises(LLMError) as info:
        call(wire_transport(server))
    err = info.value
    assert (err.kind, err.status_code) == (kind, status)
    assert len(server.requests) == 1
    assert f"code={code}" in str(err)
    assert FAKE_KEY not in str(err) and "Margaret" not in str(err)
    if kind == "rate_limit":
        assert err.retry_after_s == 0.75  # type: ignore[attr-defined]


def test_wire_read_timeout_maps_to_timeout() -> None:
    server = WireServer([httpx2.ReadTimeout("timed out")])
    with pytest.raises(LLMError) as info:
        call(wire_transport(server))
    assert info.value.kind == "timeout" and info.value.retryable
