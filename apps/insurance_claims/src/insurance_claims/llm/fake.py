"""Deterministic offline stand-in for the model (``MODEL_PROVIDER=fake``) and for tests.

It behaves like a simple tool-using agent so the real ReAct loop, tool executor, and
guardrails run end to end without network access. It is deliberately simple; the
production understanding comes from the real model.
"""

from __future__ import annotations

import json
import re
from typing import Any

from insurance_claims.llm.base import FunctionCall, LLMResponse, LLMUsage, ModelTask
from insurance_claims.observability import tracing

_NAME = re.compile(r"(?:name is|i'?m|this is)\s+([A-Z][a-z]+\s+[A-Z][a-z]+)")
_DOB = re.compile(r"\b(19\d\d-\d\d-\d\d)\b")
_LAST4 = re.compile(r"(?:last\s*(?:4|four)|ssn|social)\D{0,20}(\d{4})\b", re.IGNORECASE)
_PHONE = re.compile(r"\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}")
_EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")
_DONE = re.compile(r"\b(that'?s all|nothing else|i'?m done|no,? thanks)\b", re.IGNORECASE)
_OFF_TOPIC = re.compile(r"\bwhat is rl\b|\bweather\b|\bpython\b|\bworld cup\b", re.IGNORECASE)
_HUMAN = re.compile(r"\b(human|representative|real person|supervisor)\b", re.IGNORECASE)
_TYPES = {"healthcare": ("health", "medical"), "auto": ("auto", "car"), "dental": ("dental",)}


def _msg(text: str) -> LLMResponse:
    item = {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": text}]}
    return LLMResponse(status="completed", text=text, raw_output_items=[item], usage=LLMUsage(10, 10, 20))


class OfflineFakeModel:
    def __init__(self, *, model_name: str = "fake-agent-1", today: Any = None) -> None:
        self.model_name = model_name
        self.today = today
        self.calls: list[str] = []
        self._n = 0

    def create(
        self,
        *,
        task: ModelTask,
        instructions: str,
        input: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        tool_choice: Any = None,
        **_: Any,
    ) -> LLMResponse:
        self.calls.append(task)
        with tracing.span("fake.responses.create", type="llm") as node:
            node["tokens"] = {"input": 10, "output": 10, "total": 20}
            return self._decide(instructions, input, {t["name"] for t in tools or []}, tool_choice)

    # ------------------------------------------------------------------
    def _call(self, name: str, args: dict[str, Any]) -> LLMResponse:
        self._n += 1
        fc = FunctionCall(call_id=f"call_{self._n}", name=name, arguments=json.dumps(args))
        item = {"type": "function_call", "call_id": fc.call_id, "name": name, "arguments": fc.arguments}
        return LLMResponse(status="completed", function_calls=[fc], raw_output_items=[item], usage=LLMUsage(10, 10, 20))

    def _decide(self, instructions: str, items: list[dict[str, Any]], tools: set[str], tool_choice: Any) -> LLMResponse:
        ctx = (
            json.loads(instructions.split("(from the application, trusted)\n", 1)[1])
            if "(from the application, trusted)\n" in instructions
            else {}
        )
        callers = [i["content"] for i in items if i.get("role") == "user"]
        last = callers[-1] if callers else ""
        # results of tool calls made since the caller's latest message
        idx = max((k for k, i in enumerate(items) if i.get("role") == "user"), default=-1)
        done = {}
        names = {i.get("call_id"): i.get("name") for i in items[idx + 1 :] if i.get("type") == "function_call"}
        for i in items[idx + 1 :]:
            if i.get("type") == "function_call_output":
                out = json.loads(i["output"])
                if "error" not in out:  # a failed call counts as not done; the loop's step budget bounds retries
                    done[names.get(i["call_id"])] = out
        guard = next((i["content"] for i in items[idx + 1 :] if i.get("role") == "developer" and "GUARDRAIL" in i.get("content", "")), None)
        if tool_choice == "none" or guard:
            return _msg("Thanks for your patience. How else can I help with your claim?")

        if ctx.get("verified") and "report_caller_change" in tools and "report_caller_change" not in done:
            m = _NAME.search(last)
            if m and ctx.get("caller_first_name") and not m.group(1).startswith(ctx["caller_first_name"]):
                return self._call("report_caller_change", {"acting_for_someone_else": False})
        if "report_caller_change" in done:
            return _msg("I'll need to verify who I'm speaking with now before we continue.")
        if _HUMAN.search(last) and "request_human" in tools and "request_human" not in done:
            return self._call("request_human", {"reason": "caller_asked"})
        if "request_human" in done:
            return _msg("I've flagged this for a human claims representative to follow up with you.")
        if _OFF_TOPIC.search(last) and "flag_off_topic" not in done:
            return self._call("flag_off_topic", {})
        if "flag_off_topic" in done:
            extra = " I can also connect you with a human representative." if done["flag_off_topic"].get("offer_human") else ""
            return _msg("I can only help with insurance claim questions here." + extra)

        hint = {t: True for t, words in _TYPES.items() if any(w in last.lower() for w in words)}
        if "note_caller_context" not in done and ("claim" in last.lower()) and not ctx.get("verified"):
            ctype = next(iter(hint), None)
            status = "denied" if "denied" in last.lower() else None
            month = 1 if "january" in last.lower() else None
            return self._call(
                "note_caller_context",
                {"reason": "question about a claim", "case_type": ctype, "status": status, "month": month, "year": None},
            )

        if "verify_identity" in tools:
            v = done.get("verify_identity")
            if v is None:
                text = "\n".join(callers)
                args = {
                    "full_name": (m.group(1) if (m := _NAME.search(text)) else None),
                    "dob": (m.group(1) if (m := _DOB.search(text)) else None),
                    "phone": (m.group(0) if (m := _PHONE.search(text)) else None),
                    "email": (m.group(0) if (m := _EMAIL.search(text)) else None),
                    "id_last4": (m.group(1) if (m := _LAST4.search(text)) else None),
                    "policy_number": None,
                    "caller_is_policyholder": not re.search(r"\b(?:for|behalf of) my (?:mother|father|mom|dad)\b", text, re.IGNORECASE),
                }
                if (
                    sum(1 for k in ("full_name", "dob", "phone", "email", "id_last4") if args[k]) >= 3
                    or args["caller_is_policyholder"] is False
                ):
                    return self._call("verify_identity", args)
                return _msg(
                    "To protect your privacy, could you share three details such as your full name, date of birth, phone or email on file, or the last four digits of your SSN?"
                )
            return _msg(v.get("message", "Could you double check those details?").split(".")[0] + ".")

        if "list_my_claims" in tools and "list_my_claims" not in done and not ctx.get("selected_claim"):
            return self._call("list_my_claims", {})
        if "list_my_claims" in done and "select_claim" not in done:
            claims = done["list_my_claims"].get("claims", [])
            rc = done["list_my_claims"].get("remembered_context", {})
            fits = [c for c in claims if all(c.get(k) == rc[k] for k in ("case_type", "status") if rc.get(k))]
            if rc.get("month"):
                fits = [c for c in fits if int(c["created_at"][5:7]) == rc["month"]]
            if len(fits) == 1:
                return self._call("select_claim", {"case_id": fits[0]["case_id"]})
            return _msg(
                "Which claim would you like to talk about? I see "
                + ", ".join(f"your {c['case_type']} claim {c['case_id']}" for c in claims)
                + "."
            )

        if ctx.get("selected_claim") or "select_claim" in done:
            if _DONE.search(last) and "offer_email_summary" in tools and "offer_email_summary" not in done:
                d = done.get("get_claim_details")
                if d is None:
                    return self._call("get_claim_details", {})
                case = d["case"]
                docs = " and ".join(case.get("documents_needed") or [])
                body = f"Hello, thank you for contacting claims support about your {case['case_type']} claim {case['case_id']}. The claim is currently {case['status']}."
                if docs:
                    body += f" The next step is to send the {docs} through the member portal or claim upload link."
                body += " Claims Support Team"
                return self._call("offer_email_summary", {"summary": body})
            if "offer_email_summary" in done:
                return _msg("I've prepared a summary of our conversation. Would you like me to email it to the address on file?")
            if "record_email_decision" in tools and "record_email_decision" not in done:
                if re.search(r"\b(yes|send)\b", last, re.IGNORECASE):
                    return self._call("record_email_decision", {"decision": "send"})
                if re.search(r"\b(no|skip)\b", last, re.IGNORECASE):
                    return self._call("record_email_decision", {"decision": "skip"})
            if "record_email_decision" in done:
                return _msg("Got it.")
            d = done.get("get_claim_details")
            if d is None and "get_claim_details" in tools:
                return self._call("get_claim_details", {})
            if d:
                case = d["case"]
                text = f"Thanks, you're verified. Your {case['case_type']} claim {case['case_id']} is {case['status']}."
                if case.get("denial_reason"):
                    text = f"Thanks, you're verified. Your {case['case_type']} claim {case['case_id']} was denied because {case['denial_reason']}."
                if case.get("documents_needed"):
                    text += " Do you have the " + " and the ".join(case["documents_needed"]) + "?"
                return _msg(text)
        return _msg("How can I help with your claim today?")
