"""Provider-neutral model transport contract.

The agent never touches the OpenAI SDK directly. It calls a
:class:`ModelTransport`, which returns a normalized :class:`LLMResponse`.
That indirection lets unit tests use a deterministic fake model and a seeded
chaos layer that mutates replies, and keeps every retry, budget, and stop
reason in application code.

Implementations:

* ``llm.openai_transport.OpenAITransport`` - Responses API (``store=False``).
* ``llm.fake.OfflineFakeModel`` - deterministic offline model.
* ``llm.chaos.ChaosTransport`` - wraps any transport and injects faults.
* ``llm.resilient.ResilientTransport`` - bounded retries with backoff.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, runtime_checkable

ModelTask = str
"""Local label for a model call (e.g. "agent"); used by fakes, chaos, and traces, never sent to the provider."""

LLMErrorKind = Literal[
    "timeout",
    "rate_limit",
    "connection",
    "server",
    "bad_request",
    "auth",
    "deadline",
    "budget",
    "unknown",
]

RETRYABLE_KINDS: frozenset[str] = frozenset({"timeout", "rate_limit", "connection", "server"})


class LLMError(Exception):
    """A model call failed. ``retryable`` is derived from ``kind``. Message never contains secrets."""

    def __init__(self, kind: LLMErrorKind, message: str = "", *, status_code: int | None = None) -> None:
        super().__init__(message or kind)
        self.kind: LLMErrorKind = kind
        self.status_code = status_code

    @property
    def retryable(self) -> bool:
        return self.kind in RETRYABLE_KINDS


@dataclass
class FunctionCall:
    call_id: str
    name: str
    arguments: str
    """Raw JSON string exactly as the model produced it (may be malformed)."""
    item_id: str | None = None


@dataclass
class LLMUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    reasoning_tokens: int = 0
    cached_tokens: int = 0

    def add(self, other: LLMUsage) -> None:
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens
        self.total_tokens += other.total_tokens
        self.reasoning_tokens += other.reasoning_tokens
        self.cached_tokens += other.cached_tokens

    def as_dict(self) -> dict[str, int]:
        return {
            "input": self.input_tokens,
            "output": self.output_tokens,
            "total": self.total_tokens,
            "reasoning": self.reasoning_tokens,
            "cached": self.cached_tokens,
        }


@dataclass
class LLMResponse:
    status: Literal["completed", "incomplete", "failed"]
    text: str = ""
    """Concatenated ``output_text`` of message items ('' when the model produced none)."""
    refusal: str | None = None
    function_calls: list[FunctionCall] = field(default_factory=list)
    raw_output_items: list[dict[str, Any]] = field(default_factory=list)
    """Output items as plain dicts, echoed back verbatim in tool loops (reasoning items included)."""
    usage: LLMUsage = field(default_factory=LLMUsage)
    incomplete_reason: str | None = None
    response_id: str | None = None
    model: str = ""


@runtime_checkable
class ModelTransport(Protocol):
    """One model call. Implementations must not retry internally (see ResilientTransport)."""

    model_name: str

    def create(
        self,
        *,
        task: ModelTask,
        instructions: str,
        input: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        max_output_tokens: int = 1000,
        reasoning_effort: str | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        parallel_tool_calls: bool | None = None,
        timeout: float | None = None,
    ) -> LLMResponse:
        """Run one Responses API style call.

        ``task`` is local metadata (used by fakes, chaos, and traces); it is never sent to the provider.
        Raises :class:`LLMError` on transport/provider failure.
        """
        ...
