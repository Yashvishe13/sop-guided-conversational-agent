"""OpenAI Responses API transport.

One call in, one normalized :class:`LLMResponse` out. This module never
retries (the client is built with ``max_retries=0``; retries live in
:mod:`insurance_claims.llm.resilient`) and never stores anything provider-side
(``store=False``). Reasoning items come back with ``encrypted_content`` when
tools are offered, so a tool loop can echo them with the function outputs.

Errors are mapped to :class:`LLMError` kinds. Exception messages are built
only from safe metadata (class name, HTTP status, provider error code/param,
request ID): the provider's free-text message is dropped because it can echo
request content, and the API key is scrubbed as a last line of defence.
"""

from __future__ import annotations

import importlib
import re
from typing import Any, Iterable, Mapping

import openai

from insurance_claims.config import ConfigError, Settings
from insurance_claims.llm.base import (
    FunctionCall,
    LLMError,
    LLMErrorKind,
    LLMResponse,
    LLMUsage,
    ModelTask,
)
from insurance_claims.observability import tracing

ENCRYPTED_REASONING = "reasoning.encrypted_content"
MAX_RETRY_AFTER_S = 60.0

_SAFE_TOKEN = re.compile(r"[^A-Za-z0-9_.\-\[\]]")
_KEY_PATTERN = re.compile(r"sk-[A-Za-z0-9_\-]{8,}")


def _exception_types(*names: str) -> tuple[type[BaseException], ...]:
    """Collect ``httpx``/``httpx2`` exception classes that exist in this environment."""
    found: list[type[BaseException]] = []
    for module_name in ("httpx2", "httpx"):
        try:
            module = importlib.import_module(module_name)
        except ImportError:  # pragma: no cover - both ship with the SDK today
            continue
        found.extend(getattr(module, name) for name in names if hasattr(module, name))
    return tuple(found)


_HTTP_TIMEOUTS = _exception_types("TimeoutException")
_HTTP_TRANSPORT = _exception_types("TransportError")
_STATUS_MAP: dict[str, str] = {
    "completed": "completed",
    "incomplete": "incomplete",
    "in_progress": "incomplete",
    "queued": "incomplete",
    "failed": "failed",
    "cancelled": "failed",
}


class OpenAITransport:
    """:class:`ModelTransport` over ``client.responses.create``."""

    def __init__(self, settings: Settings, client: Any | None = None) -> None:
        self.model_name: str = settings.openai_model
        self._default_timeout = settings.model_timeout_s
        self._secrets: tuple[str, ...] = tuple(s for s in (settings.openai_api_key,) if s)
        self._client = client if client is not None else _build_client(settings)

    def __repr__(self) -> str:
        return f"OpenAITransport(model={self.model_name!r})"

    def create(
        self,
        *,
        task: ModelTask,
        instructions: str,
        input: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        text_format: dict[str, Any] | None = None,
        max_output_tokens: int = 1000,
        reasoning_effort: str | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        parallel_tool_calls: bool | None = None,
        timeout: float | None = None,
    ) -> LLMResponse:
        """Run one Responses API call. ``task`` stays local and is never sent."""
        kwargs = self.build_request(
            instructions=instructions,
            input=input,
            tools=tools,
            text_format=text_format,
            max_output_tokens=max_output_tokens,
            reasoning_effort=reasoning_effort,
            tool_choice=tool_choice,
            parallel_tool_calls=parallel_tool_calls,
            timeout=timeout,
        )
        try:
            response = self._client.responses.create(**kwargs)
        except Exception as exc:
            raise map_exception(exc, secrets=self._secrets) from None
        try:
            return to_llm_response(response, default_model=self.model_name)
        except Exception as exc:
            raise LLMError("unknown", f"unparseable provider response ({type(exc).__name__})") from None

    def build_request(
        self,
        *,
        instructions: str,
        input: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        text_format: dict[str, Any] | None,
        max_output_tokens: int,
        reasoning_effort: str | None,
        tool_choice: str | dict[str, Any] | None,
        parallel_tool_calls: bool | None,
        timeout: float | None,
    ) -> dict[str, Any]:
        """Keyword arguments for ``responses.create`` (no secrets, no task metadata)."""
        kwargs: dict[str, Any] = {
            "model": self.model_name,
            "instructions": instructions,
            "input": input,
            "store": False,
            "max_output_tokens": max_output_tokens,
            "timeout": timeout if timeout is not None else self._default_timeout,
        }
        if reasoning_effort:
            kwargs["reasoning"] = {"effort": reasoning_effort}
        if text_format is not None:
            kwargs["text"] = text_format
        if tools:
            kwargs["tools"] = tools
            kwargs["include"] = [ENCRYPTED_REASONING]
            if tool_choice is not None:
                kwargs["tool_choice"] = tool_choice
            if parallel_tool_calls is not None:
                kwargs["parallel_tool_calls"] = parallel_tool_calls
        return kwargs


def _build_client(settings: Settings) -> Any:
    """Real SDK client: no SDK-level retries, server-side timeout, traced."""
    if not settings.openai_api_key:
        raise ConfigError("OPENAI_API_KEY is not set; set it or use MODEL_PROVIDER=fake")
    client = openai.OpenAI(
        api_key=settings.openai_api_key,
        max_retries=0,
        timeout=settings.model_timeout_s,
    )
    return tracing.wrap_openai(client)


# ---------------------------------------------------------------- response mapping


def to_llm_response(response: Any, *, default_model: str = "") -> LLMResponse:
    """Normalize an SDK ``Response`` (or a close fake) into :class:`LLMResponse`."""
    output = list(getattr(response, "output", None) or [])
    raw_status = getattr(response, "status", None)
    status = _STATUS_MAP.get(raw_status or "completed", "failed")
    return LLMResponse(
        status=status,  # type: ignore[arg-type]
        text=_collect_text(output),
        refusal=_collect_refusal(output),
        function_calls=_collect_calls(output),
        raw_output_items=[_dump_item(item) for item in output],
        usage=_map_usage(getattr(response, "usage", None)),
        incomplete_reason=_incomplete_reason(response, raw_status),
        response_id=getattr(response, "id", None),
        model=getattr(response, "model", None) or default_model,
    )


def _field(obj: Any, name: str, default: Any = None) -> Any:
    if isinstance(obj, Mapping):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _messages(output: Iterable[Any]) -> list[Any]:
    return [item for item in output if _field(item, "type") == "message"]


def _collect_text(output: list[Any]) -> str:
    """Join ``output_text`` parts; when phases are labelled, keep only the final answer."""
    messages = _messages(output)
    finals = [m for m in messages if _field(m, "phase") == "final_answer"]
    chosen = finals or messages
    parts: list[str] = []
    for message in chosen:
        for part in _field(message, "content", None) or []:
            if _field(part, "type") == "output_text" and _field(part, "text"):
                parts.append(_field(part, "text"))
    return "".join(parts)


def _collect_refusal(output: list[Any]) -> str | None:
    parts = [
        _field(part, "refusal")
        for message in _messages(output)
        for part in (_field(message, "content", None) or [])
        if _field(part, "type") == "refusal" and _field(part, "refusal")
    ]
    return "\n".join(parts) if parts else None


def _collect_calls(output: list[Any]) -> list[FunctionCall]:
    return [
        FunctionCall(
            call_id=_field(item, "call_id") or "",
            name=_field(item, "name") or "",
            arguments=_field(item, "arguments") or "",
            item_id=_field(item, "id"),
        )
        for item in output
        if _field(item, "type") == "function_call"
    ]


def _dump_item(item: Any) -> dict[str, Any]:
    """Plain dict for echoing back as input (API field names, no nulls)."""
    if hasattr(item, "model_dump"):
        return item.model_dump(exclude_none=True, by_alias=True, mode="json")
    if isinstance(item, Mapping):
        return {k: v for k, v in item.items() if v is not None}
    raise TypeError(f"unsupported output item {type(item).__name__}")


def _int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _map_usage(usage: Any) -> LLMUsage:
    if usage is None:
        return LLMUsage()
    return LLMUsage(
        input_tokens=_int(_field(usage, "input_tokens")),
        output_tokens=_int(_field(usage, "output_tokens")),
        total_tokens=_int(_field(usage, "total_tokens")),
        reasoning_tokens=_int(_field(_field(usage, "output_tokens_details"), "reasoning_tokens")),
        cached_tokens=_int(_field(_field(usage, "input_tokens_details"), "cached_tokens")),
    )


def _incomplete_reason(response: Any, raw_status: str | None) -> str | None:
    details = getattr(response, "incomplete_details", None)
    reason = _field(details, "reason") if details is not None else None
    if reason:
        return str(reason)
    if raw_status in {"in_progress", "queued"}:
        return raw_status
    if raw_status in {"failed", "cancelled"}:
        error = getattr(response, "error", None)
        code = _field(error, "code") if error is not None else None
        return _safe_token(code) or raw_status
    return None


# ---------------------------------------------------------------- error mapping


def map_exception(exc: BaseException, *, secrets: Iterable[str] = ()) -> LLMError:
    """Translate an SDK/transport exception into an :class:`LLMError` without leaking secrets."""
    if isinstance(exc, LLMError):
        return exc
    kind, status_code = _classify(exc)
    message = _scrub(_describe(exc, kind, status_code), secrets)
    error = LLMError(kind, message, status_code=status_code)
    retry_after = _retry_after_seconds(exc)
    if retry_after is not None:
        error.retry_after_s = retry_after  # type: ignore[attr-defined]
    return error


def _classify(exc: BaseException) -> tuple[LLMErrorKind, int | None]:
    if isinstance(exc, openai.APITimeoutError):
        return "timeout", None
    if isinstance(exc, openai.APIConnectionError):
        return "connection", None
    if isinstance(exc, openai.APIStatusError):
        return _classify_status(exc), exc.status_code
    if isinstance(exc, openai.APIResponseValidationError):
        return "unknown", exc.status_code
    if isinstance(exc, (TimeoutError, _HTTP_TIMEOUTS)):
        return "timeout", None
    if isinstance(exc, (ConnectionError, _HTTP_TRANSPORT)):
        return "connection", None
    return "unknown", None


def _classify_status(exc: openai.APIStatusError) -> LLMErrorKind:
    status = exc.status_code
    if isinstance(exc, openai.RateLimitError) or status == 429:
        return "budget" if exc.code == "insufficient_quota" else "rate_limit"
    if isinstance(exc, (openai.AuthenticationError, openai.PermissionDeniedError)) or status in {401, 403}:
        return "auth"
    if isinstance(exc, openai.InternalServerError) or status >= 500:
        return "server"
    if status == 408:
        return "timeout"
    return "bad_request"


def _safe_token(value: Any, limit: int = 64) -> str | None:
    if value is None:
        return None
    text = _SAFE_TOKEN.sub("", str(value))[:limit]
    return text or None


def _describe(exc: BaseException, kind: str, status_code: int | None) -> str:
    """Safe one-line description: no provider free text, no request body."""
    parts = [f"{kind}: {type(exc).__name__}"]
    if status_code is not None:
        parts.append(f"status={status_code}")
    if isinstance(exc, openai.APIError):
        for label, value in (("code", exc.code), ("type", exc.type), ("param", exc.param)):
            token = _safe_token(value)
            if token:
                parts.append(f"{label}={token}")
    request_id = _safe_token(getattr(exc, "request_id", None))
    if request_id:
        parts.append(f"request_id={request_id}")
    return " ".join(parts)


def _scrub(message: str, secrets: Iterable[str]) -> str:
    for secret in secrets:
        if secret:
            message = message.replace(secret, "[REDACTED]")
    return _KEY_PATTERN.sub("[REDACTED]", message)


def _retry_after_seconds(exc: BaseException) -> float | None:
    """Server-suggested wait from ``retry-after-ms`` / ``retry-after`` (seconds only), capped."""
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    if headers is None:
        return None
    for name, scale in (("retry-after-ms", 0.001), ("retry-after", 1.0)):
        raw = headers.get(name)
        if raw is None:
            continue
        try:
            seconds = float(raw) * scale
        except (TypeError, ValueError):
            continue
        if seconds >= 0:
            return min(seconds, MAX_RETRY_AFTER_S)
    return None
