"""The guard: an independent model reviewer that the application consults on every message in,
every reply out, and every action that changes what the application believes.

The agent decides what to say and which tool to call. Some decisions are too important to rest on
the agent alone, and cannot be made reliably with keyword rules. For those, code asks the guard:

=====================  ===========================================  ==========================================
Checkpoint             Question                                     What code does with the answer
=====================  ===========================================  ==========================================
``review_caller``      Every caller message, before the agent       Who is speaking (own account, someone
                       runs: who is speaking, did they refuse        else's, a new person); counts refusals and
                       verification, is it off topic, what claim     off-topic requests; remembers the claim the
                       do they mean?                                 caller described
``judge_consent``      ``record_email_decision``: what did the       The decision is recorded only if the guard
                       caller's own words choose?                    and the agent agree
``judge_summary``      ``offer_email_summary``: is every statement   An unsupported or incomplete summary is
                       supported by the record?                      rejected with the problems listed
``judge_document``     ``record_document_status``: what did the      The status is stored only if the guard
                       caller say about this document?               and the agent agree
``judge_reply``        Every draft reply, in every phase: is each    A reply with any problem is blocked and
                       claim fact supported by the record and the    the agent must rewrite it; no model text
                       tool results, nothing disclosed before        reaches the caller without this review
                       verification, no promise or invented action,
                       nothing out of scope or internal?
=====================  ===========================================  ==========================================

Design, following the pattern of a permission classifier:

* **Independent.** The guard has its own prompts (``[tasks.guard_*]`` in ``prompts.toml``), its own
  model transport, and sees only the data it judges. It never sees the agent's reasoning or the
  arguments of its tool calls; to fact check a reply it gets the record and the claim tool results
  the agent read this turn, as evidence.
* **Untrusted input stays data.** The material is sent as one JSON document, so caller text cannot
  close a tag or pose as instructions; the prompts say to ignore instructions inside it.
* **Strict output.** Each verdict is a JSON object with a fixed schema (Responses API structured
  output), validated again here.
* **Fails closed.** Any error, refusal, timeout, or invalid verdict returns ``None``, and every call
  site treats ``None`` as "not allowed".
"""

from __future__ import annotations

import json
from typing import Any, Literal, TypeVar

from pydantic import BaseModel, ConfigDict, ValidationError

from insurance_claims.agent.prompts import PromptSet
from insurance_claims.llm.base import ModelTransport
from insurance_claims.observability import tracing

T = TypeVar("T", bound=BaseModel)


# ---------------------------------------------------------------------- verdicts


class _Verdict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ClaimMentioned(_Verdict):
    case_type: Literal["healthcare", "dental", "auto", "other"] | None
    status: Literal["denied", "open", "closed"] | None
    month: int | None
    year: int | None


class CallerReview(_Verdict):
    speaker: Literal["account_holder", "acting_for_someone_else", "unclear"]
    different_person: bool
    refused_verification: bool
    off_topic_request: bool
    claim_mentioned: ClaimMentioned
    identity_fields_given: list[Literal["full_name", "dob", "phone", "email", "id_last4", "policy_number"]]
    reason: str | None
    rationale: str


class ConsentVerdict(_Verdict):
    decision: Literal["send", "skip", "unclear"]
    rationale: str


class SummaryVerdict(_Verdict):
    supported: bool
    problems: list[str]


ReplyProblemCategory = Literal[
    "unsupported_fact",
    "disclosed_before_verification",
    "promise_or_invented_action",
    "out_of_scope",
    "internal_details",
    "missing_required",
]
DOC_STATUSES = ("has_it", "can_request", "cannot_obtain", "already_sent", "unknown")


class ReplyProblem(_Verdict):
    category: ReplyProblemCategory
    detail: str


class ReplyVerdict(_Verdict):
    allowed: bool
    problems: list[ReplyProblem]


class DocumentVerdict(_Verdict):
    status: Literal["has_it", "can_request", "cannot_obtain", "already_sent", "unknown"]
    rationale: str


# ---------------------------------------------------------------------- strict JSON schemas


def _nullable(kind: str, enum: list[str] | None = None) -> dict[str, Any]:
    out: dict[str, Any] = {"type": [kind, "null"]}
    if enum:
        out["enum"] = [*enum, None]
    return out


def _object(props: dict[str, Any]) -> dict[str, Any]:
    return {"type": "object", "properties": props, "required": list(props), "additionalProperties": False}


_STRINGS = {"type": "array", "items": {"type": "string"}}
SCHEMAS: dict[str, dict[str, Any]] = {
    "guard_caller": _object(
        {
            "speaker": {"type": "string", "enum": ["account_holder", "acting_for_someone_else", "unclear"]},
            "different_person": {"type": "boolean"},
            "refused_verification": {"type": "boolean"},
            "off_topic_request": {"type": "boolean"},
            "claim_mentioned": _object(
                {
                    "case_type": _nullable("string", ["healthcare", "dental", "auto", "other"]),
                    "status": _nullable("string", ["denied", "open", "closed"]),
                    "month": _nullable("integer"),
                    "year": _nullable("integer"),
                }
            ),
            "identity_fields_given": {
                "type": "array",
                "items": {"type": "string", "enum": ["full_name", "dob", "phone", "email", "id_last4", "policy_number"]},
            },
            "reason": _nullable("string"),
            "rationale": {"type": "string"},
        }
    ),
    "guard_consent": _object({"decision": {"type": "string", "enum": ["send", "skip", "unclear"]}, "rationale": {"type": "string"}}),
    "guard_summary": _object({"supported": {"type": "boolean"}, "problems": _STRINGS}),
    "guard_document": _object({"status": {"type": "string", "enum": list(DOC_STATUSES)}, "rationale": {"type": "string"}}),
    "guard_reply": _object(
        {
            "allowed": {"type": "boolean"},
            "problems": {
                "type": "array",
                "items": _object(
                    {
                        "category": {
                            "type": "string",
                            "enum": [
                                "unsupported_fact",
                                "disclosed_before_verification",
                                "promise_or_invented_action",
                                "out_of_scope",
                                "internal_details",
                                "missing_required",
                            ],
                        },
                        "detail": {"type": "string"},
                    }
                ),
            },
        }
    ),
}


# ---------------------------------------------------------------------- the guard


class Guard:
    def __init__(
        self, *, model: ModelTransport, prompts: PromptSet, reasoning_effort: str | None, max_output_tokens: int, timeout_s: float
    ) -> None:
        self.model = model
        self.prompts = prompts
        self.reasoning_effort = reasoning_effort
        self.max_output_tokens = max_output_tokens
        self.timeout_s = timeout_s

    def review_caller(self, transcript: list[dict[str, str]]) -> CallerReview | None:
        """``transcript``: ``[{"speaker": "caller" | "representative", "text": ...}]``, oldest first."""
        return self._ask("guard_caller", {"transcript": transcript}, CallerReview)

    def judge_consent(self, *, offer: str, caller_replies: list[str]) -> ConsentVerdict | None:
        return self._ask("guard_consent", {"offer": offer, "caller_replies": caller_replies}, ConsentVerdict)

    def judge_summary(
        self,
        *,
        summary: str,
        claim: dict[str, Any],
        document_status: dict[str, str],
        guidance: list[str],
        today: str,
        transcript: list[dict[str, str]],
    ) -> SummaryVerdict | None:
        payload = {
            "summary": summary,
            "claim": claim,
            "document_status": document_status,
            "guidance": guidance,
            "today": today,
            "transcript": transcript,
        }
        return self._ask("guard_summary", payload, SummaryVerdict)

    def judge_document(self, *, document: str, caller_messages: list[str]) -> DocumentVerdict | None:
        return self._ask("guard_document", {"document": document, "caller_messages": caller_messages}, DocumentVerdict)

    def judge_reply(
        self,
        *,
        transcript: list[dict[str, str]],
        reply: str,
        verified: bool,
        record: dict[str, Any] | None,
        tool_results: list[dict[str, Any]],
        application_state: dict[str, Any],
        required_in_reply: list[str] | None = None,
    ) -> ReplyVerdict | None:
        """``record`` is None before verification (nothing about any claim may be said then)."""
        payload = {
            "transcript": transcript,
            "reply": reply,
            "caller_verified": verified,
            "application_state": application_state,
            "record": record,
            "tool_results": tool_results,
            "required_in_reply": required_in_reply or [],
        }
        return self._ask("guard_reply", payload, ReplyVerdict)

    # ------------------------------------------------------------------ one checkpoint call
    def _ask(self, task: str, payload: dict[str, Any], model_cls: type[T]) -> T | None:
        with tracing.span(f"guard.{task.removeprefix('guard_')}", type="guard") as node:
            verdict, failure = self._call(task, payload, model_cls)
            # Only enum and boolean fields go into the trace; free text may quote the caller.
            node["output"] = {"available": verdict is not None, "failure": failure, **(_trace_view(verdict) if verdict else {})}
            return verdict

    def _call(self, task: str, payload: dict[str, Any], model_cls: type[T]) -> tuple[T | None, str | None]:
        try:
            resp = self.model.create(
                task=task,
                instructions=self.prompts.guard_instructions(task),
                input=[{"role": "user", "content": json.dumps(payload, ensure_ascii=False, default=str)}],
                text_format={"format": {"type": "json_schema", "name": task, "schema": SCHEMAS[task], "strict": True}},
                max_output_tokens=self.max_output_tokens,
                reasoning_effort=self.reasoning_effort,
                timeout=self.timeout_s,
            )
        except Exception as exc:  # fail closed on any transport problem
            return None, f"error:{getattr(exc, 'kind', type(exc).__name__)}"
        if resp.status != "completed" or resp.refusal or not resp.text:
            return None, "no_verdict"
        try:
            return model_cls.model_validate_json(resp.text), None
        except ValidationError:
            return None, "invalid_verdict"


def _trace_view(verdict: BaseModel) -> dict[str, Any]:
    data = verdict.model_dump()
    out = {k: v for k, v in data.items() if isinstance(v, bool) or k in ("speaker", "decision", "status")}
    if "problems" in data:
        out["problem_count"] = len(data["problems"])
        categories = sorted({p["category"] for p in data["problems"] if isinstance(p, dict)})
        if categories:
            out["problem_categories"] = categories
    return out
