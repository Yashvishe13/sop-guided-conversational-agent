"""Bounded retries with exponential backoff around any :class:`ModelTransport`.

Only transient failures (``LLMError.retryable``: timeout, rate limit,
connection, server) are retried. Each attempt's timeout is clipped to the time
left before the turn deadline, and the transport never sleeps past that
deadline: when the next backoff would cross it, it raises
``LLMError("deadline")`` instead. Every retry is recorded as a
``llm_retry`` trace event.
"""

from __future__ import annotations

import time
from typing import Any, Callable

from insurance_claims.llm.base import LLMError, LLMResponse, ModelTask, ModelTransport
from insurance_claims.observability import tracing

MAX_BACKOFF_S = 8.0
JITTER_FRACTION = 0.1
_GOLDEN = 0.6180339887498949

DeadlineFn = Callable[[], float | None]
"""Returns the seconds remaining before the turn deadline, or ``None`` for no deadline."""


def backoff_delay(attempt: int, base_s: float, *, cap_s: float = MAX_BACKOFF_S) -> float:
    """``base * 2**attempt`` plus up to 10% deterministic jitter, capped at ``cap_s``."""
    raw = max(0.0, base_s) * (2 ** min(max(0, attempt), 32))
    jitter = raw * JITTER_FRACTION * ((attempt * _GOLDEN) % 1.0)
    return min(cap_s, raw + jitter)


class ResilientTransport:
    """Retry wrapper. The inner transport must not retry on its own."""

    def __init__(
        self,
        inner: ModelTransport,
        *,
        max_retries: int,
        backoff_base_s: float,
        sleep: Callable[[float], None] = time.sleep,
        deadline: DeadlineFn = lambda: None,
    ) -> None:
        if max_retries < 0:
            raise ValueError("max_retries must be >= 0")
        if backoff_base_s < 0:
            raise ValueError("backoff_base_s must be >= 0")
        self.inner = inner
        self.max_retries = max_retries
        self.backoff_base_s = backoff_base_s
        self._sleep = sleep
        self._deadline = deadline

    @property
    def model_name(self) -> str:
        return self.inner.model_name

    def __repr__(self) -> str:
        return f"ResilientTransport({self.inner!r}, max_retries={self.max_retries})"

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
        """Call the inner transport, retrying transient failures within the deadline."""
        request = {
            "task": task,
            "instructions": instructions,
            "input": input,
            "tools": tools,
            "max_output_tokens": max_output_tokens,
            "reasoning_effort": reasoning_effort,
            "tool_choice": tool_choice,
            "parallel_tool_calls": parallel_tool_calls,
        }
        attempt = 0
        while True:
            call_timeout = self._attempt_timeout(task, timeout, attempt)
            try:
                return self.inner.create(**request, timeout=call_timeout)
            except LLMError as exc:
                self._before_retry(task, exc, attempt)
                attempt += 1

    # ------------------------------------------------------------------ helpers

    def _remaining(self) -> float | None:
        remaining = self._deadline()
        return None if remaining is None else float(remaining)

    def _attempt_timeout(self, task: str, timeout: float | None, attempt: int) -> float | None:
        """Per-attempt timeout clipped to the deadline; raise when no time is left."""
        remaining = self._remaining()
        if remaining is None:
            return timeout
        if remaining <= 0:
            tracing.event("llm_deadline", type="retry", task=task, attempt=attempt, remaining_s=0.0)
            raise LLMError("deadline", f"turn deadline reached before {task} attempt {attempt + 1}")
        return remaining if timeout is None else min(timeout, remaining)

    def _before_retry(self, task: str, exc: LLMError, attempt: int) -> None:
        """Re-raise when the error is final; otherwise wait out the backoff."""
        if not exc.retryable:
            raise exc
        if attempt >= self.max_retries:
            tracing.event("llm_retry_exhausted", type="retry", task=task, attempts=attempt + 1, kind=exc.kind)
            raise exc
        delay = self._delay(exc, attempt)
        remaining = self._remaining()
        if remaining is not None and delay >= remaining:
            tracing.event(
                "llm_deadline",
                type="retry",
                task=task,
                attempt=attempt + 1,
                kind=exc.kind,
                delay_s=round(delay, 3),
                remaining_s=round(max(remaining, 0.0), 3),
            )
            raise LLMError("deadline", f"retry of {task} after {exc.kind} would exceed the turn deadline") from exc
        tracing.event(
            "llm_retry",
            type="retry",
            task=task,
            attempt=attempt + 1,
            kind=exc.kind,
            status_code=exc.status_code,
            delay_s=round(delay, 3),
        )
        self._sleep(delay)

    def _delay(self, exc: LLMError, attempt: int) -> float:
        """Backoff for this attempt, stretched to a server ``retry-after`` hint when given."""
        delay = backoff_delay(attempt, self.backoff_base_s)
        hint = getattr(exc, "retry_after_s", None)
        if isinstance(hint, (int, float)) and hint > delay:
            delay = min(float(hint), MAX_BACKOFF_S)
        return delay
