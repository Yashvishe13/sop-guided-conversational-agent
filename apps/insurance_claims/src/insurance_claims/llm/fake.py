"""Deterministic offline stand-in for the model (``MODEL_PROVIDER=fake``) and for tests.

It behaves like a simple tool-using agent so the real ReAct loop, tool executor, and
guardrails run end to end without network access, and it answers the guard's checkpoints
(tasks ``guard_*``) with keyword heuristics. It is deliberately simple; in production both
the agent and the guard are the real model.
"""

from __future__ import annotations

import json
import re
from typing import Any

from insurance_claims.llm.base import FunctionCall, LLMResponse, LLMUsage, ModelTask
from insurance_claims.observability import tracing

_NAME = re.compile(r"(?:name is|i'?m|this is)\s+([A-Z][a-z]+\s+[A-Z][a-z]+)")
_MONTH_NAMES = "january|february|march|april|may|june|july|august|september|october|november|december"
_DOB = re.compile(
    rf"\b(19\d\d-\d\d-\d\d|(?:{_MONTH_NAMES})\s+\d{{1,2}}(?:st|nd|rd|th)?,?\s+19\d\d|\d{{1,2}}/\d{{1,2}}/19\d\d)\b", re.IGNORECASE
)
_LAST4 = re.compile(r"(?:last\s*(?:4|four)|ssn|social)\D{0,20}(\d{4})\b", re.IGNORECASE)
_PHONE = re.compile(r"\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}")
_EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")
_DONE = re.compile(r"\b(that'?s all|nothing else|i'?m done|no,? thanks)\b", re.IGNORECASE)
_OFF_TOPIC = re.compile(r"\bwhat is rl\b|\bweather\b|\bpython\b|\bworld cup\b", re.IGNORECASE)
_HUMAN = re.compile(r"\b(human|representative|real person|supervisor)\b", re.IGNORECASE)
_TYPES = {"healthcare": ("health", "medical"), "auto": ("auto", "car"), "dental": ("dental",)}
_RELATIVE = r"(?:mother|father|mom|dad|wife|husband|son|daughter|sister|brother|client)"
_FOR_SOMEONE = re.compile(rf"\b(?:for|behalf of) my {_RELATIVE}\b|\bmy {_RELATIVE}'?s (?:claim|policy|account)\b", re.IGNORECASE)
_REFUSAL = re.compile(r"already told you|not giving|won'?t give|why do i (?:have|need) to|just tell me", re.IGNORECASE)
_YES = re.compile(r"\b(?:yes|yeah|sure|send it|go ahead|please do)\b", re.IGNORECASE)
_NO = re.compile(r"\b(?:no|nope|skip|don'?t)\b", re.IGNORECASE)
_UNSUPPORTED = re.compile(r"already (?:submitted|sent|received)|no (?:further )?action (?:is )?needed|will be approved", re.IGNORECASE)
_ON_FILE = re.compile(
    r"\b(?:complete and on file|on file|we(?:'ve| have) received|no longer needed|nothing (?:else|more) is needed)\b", re.IGNORECASE
)
_DOC_SAYS = (
    ("already_sent", re.compile(r"already (?:sent|uploaded|faxed|submitted)", re.IGNORECASE)),
    ("cannot_obtain", re.compile(r"can'?t get|cannot get|won'?t (?:re)?send|refuses?", re.IGNORECASE)),
    ("can_request", re.compile(r"\bcan (?:get|request|ask)\b|\bwill (?:get|request|ask)\b", re.IGNORECASE)),
    ("has_it", re.compile(r"\bi (?:have|got) (?:it|the|both)\b", re.IGNORECASE)),
)
_OUT_OF_SCOPE = re.compile(r"reinforcement learning|machine learning|\bdef \w+\(|world cup", re.IGNORECASE)


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
            if str(task).startswith("guard_"):
                return _json(_guard_verdict(str(task), json.loads(input[0]["content"])))
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

        if _HUMAN.search(last) and "request_human" in tools and "request_human" not in done:
            return self._call("request_human", {"reason": "caller_asked"})
        if "request_human" in done:
            return _msg("I've flagged this for a human claims representative to follow up with you.")
        review = ctx.get("caller_review") if isinstance(ctx.get("caller_review"), dict) else {}
        if review.get("off_topic_request"):
            extra = " I can also connect you with a human representative." if ctx.get("off_topic", {}).get("offer_human") else ""
            return _msg("I can only help with insurance claim questions here." + extra)
        if review.get("speaker") == "acting_for_someone_else" and not ctx.get("verified"):
            return _msg(
                "I can only discuss claim details with the policyholder. I can connect you with a representative who can review authorization."
            )
        if review.get("refused_verification") and not ctx.get("verified"):
            if ctx.get("verification", {}).get("stop_persuading"):
                return _msg("I understand, and I won't ask again. Would you like me to connect you with a human representative?")
            return _msg(
                "I understand this is frustrating. Verification protects your private claim information. "
                "Any three details work, such as your full name, date of birth, phone or email on file, or the last four digits of your SSN."
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
                }
                if sum(1 for k in ("full_name", "dob", "phone", "email", "id_last4") if args[k]) >= 3:
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


def _json(data: dict[str, Any]) -> LLMResponse:
    text = json.dumps(data)
    item = {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": text}]}
    return LLMResponse(status="completed", text=text, raw_output_items=[item], usage=LLMUsage(10, 10, 20))


def _guard_verdict(task: str, payload: dict[str, Any]) -> dict[str, Any]:
    """Keyword stand-in for the guard model (offline demo and tests only)."""
    if task == "guard_caller":
        callers = [e["text"] for e in payload["transcript"] if e["speaker"] == "caller"]
        everything, last = " ".join(callers), (callers[-1] if callers else "")
        low = everything.lower()
        case_type = next((t for t, words in _TYPES.items() if any(w in low for w in words)), None)
        names = [m.group(1) for c in callers if (m := _NAME.search(c))]
        return {
            "speaker": "acting_for_someone_else" if _FOR_SOMEONE.search(everything) else "account_holder",
            "different_person": len(names) >= 2 and _NAME.search(last) is not None and names[-1] != names[0],
            "refused_verification": bool(_REFUSAL.search(last)),
            "off_topic_request": bool(_OFF_TOPIC.search(last)),
            "claim_mentioned": {
                "case_type": case_type,
                "status": "denied" if "denied" in low else None,
                "month": 1 if "january" in low else None,
                "year": None,
            },
            "identity_fields_given": [
                name
                for name, pattern in (("full_name", _NAME), ("dob", _DOB), ("phone", _PHONE), ("email", _EMAIL), ("id_last4", _LAST4))
                if pattern.search(everything)
            ],
            "reason": "question about a claim" if "claim" in low else None,
            "rationale": "keyword heuristic",
        }
    if task == "guard_consent":
        last = (payload["caller_replies"] or [""])[-1]
        decision = "unclear" if "?" in last else "skip" if _NO.search(last) else "send" if _YES.search(last) else "unclear"
        return {"decision": decision, "rationale": "keyword heuristic"}
    if task == "guard_summary":
        bad = _UNSUPPORTED.search(payload["summary"])
        return {"supported": not bad, "problems": [f"unsupported statement: {bad.group(0)}"] if bad else []}
    if task == "guard_document":
        latest = " ".join(payload["caller_messages"][-3:])
        status = next((name for name, pattern in _DOC_SAYS if pattern.search(latest)), "unknown")
        return {"status": status, "rationale": "keyword heuristic"}
    reply, problems = payload["reply"], []
    if bad := _OUT_OF_SCOPE.search(reply):
        problems.append({"category": "out_of_scope", "detail": f"answers an unrelated request ({bad.group(0)})"})
    selected = (payload.get("record") or {}).get("selected_claim") or {}
    outstanding = [d for d, st in (selected.get("document_status") or {}).items() if st != "already_sent"]
    if outstanding and (bad := _ON_FILE.search(reply)):
        problems.append({"category": "unsupported_fact", "detail": f"says '{bad.group(0)}' but documents are still required"})
    if payload.get("required_in_reply") and not re.search(r"\b(?:human|representative|person)\b", reply, re.IGNORECASE):
        problems.append({"category": "missing_required", "detail": "does not offer a human representative"})
    return {"allowed": not problems, "problems": problems}
