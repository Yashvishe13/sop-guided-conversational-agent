"""Per-turn budgets: model calls, tool calls, elapsed time, and token accounting.

Every model call in a turn goes through :meth:`TurnBudget.call`, so a turn can
never exceed its call or time budget, and the remaining time bounds each
request's timeout. The active budget's deadline is also published in a context
variable so the retry layer never sleeps past it.
"""

from __future__ import annotations

import time
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, Callable

from insurance_claims.llm.base import LLMError, LLMResponse, LLMUsage, ModelTransport
from insurance_claims.observability import tracing

_deadline: ContextVar[float | None] = ContextVar("turn_deadline", default=None)


def remaining_seconds() -> float | None:
    deadline = _deadline.get()
    return None if deadline is None else max(0.0, deadline - time.monotonic())


class BudgetExceeded(Exception):
    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


@dataclass
class TurnBudget:
    max_model_calls: int
    max_tool_calls: int
    deadline_s: float
    model_timeout_s: float
    clock: Callable[[], float] = time.monotonic
    model_calls: int = 0
    tool_calls: int = 0
    usage: LLMUsage = field(default_factory=LLMUsage)
    started: float = 0.0

    def __post_init__(self) -> None:
        self.started = self.clock()
        self._token = None

    @property
    def deadline(self) -> float:
        return self.started + self.deadline_s

    def remaining(self) -> float:
        return max(0.0, self.deadline - self.clock())

    def __enter__(self) -> TurnBudget:
        self._token = _deadline.set(self.deadline)
        return self

    def __exit__(self, *exc: Any) -> None:
        if self._token is not None:
            _deadline.reset(self._token)
            self._token = None

    def call(self, transport: ModelTransport, **kwargs: Any) -> LLMResponse:
        if self.model_calls >= self.max_model_calls:
            tracing.event("budget_exhausted", kind="model_calls", used=self.model_calls)
            raise BudgetExceeded("budget_model_calls")
        remaining = self.remaining()
        if remaining <= 1.0:
            tracing.event("budget_exhausted", kind="time", remaining_s=round(remaining, 2))
            raise BudgetExceeded("budget_time")
        self.model_calls += 1
        kwargs["timeout"] = min(self.model_timeout_s, remaining)
        try:
            response = transport.create(**kwargs)
        except LLMError:
            raise
        self.usage.add(response.usage)
        return response

    def charge_tool(self) -> bool:
        if self.tool_calls >= self.max_tool_calls:
            return False
        self.tool_calls += 1
        return True

    def snapshot(self) -> dict[str, Any]:
        return {
            "model_calls": self.model_calls,
            "tool_calls": self.tool_calls,
            "elapsed_ms": round((self.clock() - self.started) * 1000, 1),
            "tokens": self.usage.as_dict(),
        }
