"""Guardrail tests for the ReAct claims agent.

A scripted model plays a (sometimes misbehaving) agent: it calls tools with chosen
arguments and drafts replies. The tests assert that the code-side guardrails decide
phase, verification, data access, and email outcomes no matter what the model does.
"""

from __future__ import annotations

import json
from typing import Any, Callable

from insurance_claims.domain.models import Phase
from insurance_claims.llm.base import FunctionCall, LLMResponse, LLMUsage
from tests.conftest import DEMO_UTTERANCE

Step = Callable[[list[dict[str, Any]]], LLMResponse]


def call(name: str, **args: Any) -> LLMResponse:
    fc = FunctionCall(call_id=f"c_{name}_{len(json.dumps(args))}", name=name, arguments=json.dumps(args))
    return LLMResponse(
        status="completed",
        function_calls=[fc],
        raw_output_items=[{"type": "function_call", "call_id": fc.call_id, "name": name, "arguments": fc.arguments}],
        usage=LLMUsage(1, 1, 2),
    )


def say(text: str) -> LLMResponse:
    return LLMResponse(
        status="completed",
        text=text,
        raw_output_items=[{"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": text}]}],
        usage=LLMUsage(1, 1, 2),
    )


class Scripted:
    """Returns the scripted responses in order; records every request and the tool outputs it saw."""

    model_name = "scripted"

    def __init__(self, *responses: LLMResponse) -> None:
        self.responses = list(responses)
        self.requests: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> LLMResponse:
        self.requests.append(kwargs)
        return self.responses.pop(0) if self.responses else say("How else can I help with your claim?")

    def outputs(self) -> list[dict[str, Any]]:
        last = self.requests[-1]["input"]
        return [json.loads(i["output"]) for i in last if i.get("type") == "function_call_output"]

    def tools_offered(self, n: int) -> set[str]:
        return {t["name"] for t in self.requests[n]["tools"]}


IDENTITY = dict(
    full_name="Margaret Chen", dob="1985-03-15", phone=None, email=None, id_last4="4472", policy_number=None, caller_is_policyholder=True
)


def test_three_fields_are_required_before_matching(harness):
    model = Scripted(
        call("verify_identity", **{**IDENTITY, "id_last4": None, "policy_number": "POL-9921"}), say("Could you share one more detail?")
    )
    h = harness(model=model)
    h.say("I'm Margaret Chen, born 1985-03-15, policy POL-9921.")
    out = model.outputs()[0]
    assert out["status"] == "need_more_information" and out["fields_still_needed"] == 1
    assert not h.state.verified and h.state.phase == Phase.VERIFY_ID


def test_values_the_caller_never_said_are_ignored(harness):
    model = Scripted(call("verify_identity", **IDENTITY), say("Could you share your details?"))
    h = harness(model=model)
    h.say("Hi, I want to know about my claim.")  # the model invents Margaret's identity
    out = model.outputs()[0]
    assert out["status"] == "need_more_information" and set(out["ignored_fields"]) >= {"full_name", "dob", "id_last4"}
    assert not h.state.verified


def test_verification_succeeds_only_through_the_tool(harness):
    model = Scripted(call("verify_identity", **IDENTITY), say("Thanks Margaret, you're verified."))
    h = harness(model=model)
    h.say(DEMO_UTTERANCE)
    assert h.state.verified and h.state.verification.party_id == "P9"
    assert h.state.phase == Phase.RESOLVE_INTENT


def test_claim_tools_are_not_offered_or_usable_before_verification(harness):
    model = Scripted(call("get_claim_details"), say("Let me check that."))
    h = harness(model=model)
    h.say("Why was my claim denied?")
    assert "get_claim_details" not in model.tools_offered(0) and "verify_identity" in model.tools_offered(0)
    assert model.outputs()[0]["error"] == "not_allowed_now"
    assert h.state.phase == Phase.VERIFY_ID


def test_unverified_reply_with_claim_data_is_blocked(harness):
    model = Scripted(
        say("Your claim CL-2048 was denied for a missing pathology report."), say("Your claim CL-2048 was denied."), say("CL-2048 again.")
    )
    h = harness(model=model)
    payload = h.say("Why was my claim denied?")
    reply = h.last_reply(payload)
    assert "CL-2048" not in reply and "pathology" not in reply.lower()


def test_style_guardrail_asks_for_a_rewrite(harness):
    model = Scripted(say("Sure: here is what I need."), say("Sure, here is what I need. Could you share your full name?"))
    h = harness(model=model)
    payload = h.say("hello")
    assert h.last_reply(payload) == "Sure, here is what I need. Could you share your full name?"
    feedback = [i for i in model.requests[1]["input"] if i.get("role") == "developer"]
    assert feedback and "GUARDRAIL" in feedback[-1]["content"]


def test_representative_is_never_verified(harness):
    model = Scripted(
        call("verify_identity", **{**IDENTITY, "caller_is_policyholder": False}), say("I can connect you with a representative.")
    )
    h = harness(model=model)
    h.say("I'm calling for my mother Margaret Chen, DOB 1985-03-15, SSN last four 4472.")
    assert model.outputs()[0]["status"] == "not_allowed"
    assert not h.state.verified and h.state.verification.representative_declared


def test_other_partys_claim_cannot_be_selected(harness):
    model = Scripted(call("verify_identity", **IDENTITY), call("select_claim", case_id="CL-3001"), say("Which claim do you mean?"))
    h = harness(model=model)
    h.say(DEMO_UTTERANCE)
    assert model.outputs()[-1]["error"] == "not_found"
    assert h.state.case.case_id is None


def test_invented_amount_is_blocked_after_verification(harness):
    model = Scripted(
        call("verify_identity", **IDENTITY),
        call("select_claim", case_id="CL-2048"),
        call("get_claim_details"),
        say("You're verified. Good news, you will be paid $9,999.00 next week."),
        say("You're verified. Your healthcare claim CL-2048 was denied because the pathology report and office note were missing."),
    )
    h = harness(model=model)
    payload = h.say(DEMO_UTTERANCE)
    reply = h.last_reply(payload)
    assert "9,999" not in reply and "CL-2048" in reply
    assert h.state.phase == Phase.PROCESS_CASE


def test_incomplete_email_summary_is_rejected(harness):
    model = Scripted(
        call("verify_identity", **IDENTITY), call("select_claim", case_id="CL-2048"), call("get_claim_details"),
        call("offer_email_summary", summary="Thanks for calling, have a nice day. Claims Support Team"),
        say("You're verified. Your healthcare claim CL-2048 was denied because the pathology report and office note were missing."),
    )  # fmt: skip
    h = harness(model=model)
    h.say(DEMO_UTTERANCE)
    assert model.outputs()[-1]["error"] == "summary_rejected"
    assert h.state.email.status == "none" and h.state.phase == Phase.PROCESS_CASE


def test_consent_cannot_be_recorded_in_the_same_turn_as_the_offer(harness):
    summary = "We discussed your healthcare claim CL-2048, which is denied. Please send the pathology report and the office note through the member portal. Claims Support Team"
    model = Scripted(
        call("verify_identity", **IDENTITY), call("select_claim", case_id="CL-2048"), call("get_claim_details"),
        call("offer_email_summary", summary=summary), call("record_email_decision", decision="send"),
        say("You're verified. Your healthcare claim CL-2048 was denied because the pathology report and office note were missing. I've prepared a summary."),
    )  # fmt: skip
    h = harness(model=model)
    h.say(DEMO_UTTERANCE)
    assert h.state.email.status == "offered"
    outbox = h.runtime.settings.outbox_dir
    assert not outbox.exists() or not list(outbox.glob("*.eml"))


def test_offline_agent_runs_the_demo_end_to_end(harness):
    h = harness()
    payload = h.say(DEMO_UTTERANCE)
    state = h.state
    assert state.verified and state.case.case_id == "CL-2048" and state.phase == Phase.PROCESS_CASE
    assert "denied because" in h.last_reply(payload)
    h.say("that's all")
    assert h.state.email.status == "offered"
    h.act("email_send")
    assert h.state.email.status == "queued"


def test_switching_claims_after_the_offer_withdraws_it(harness):
    h = harness()
    h.say("My name is Margaret Chen, DOB 1985-03-15, SSN last four 4472. Checking on my auto claim.")
    h.say("that's all")
    assert h.state.email.status == "offered"
    model = h.runtime.agent.model
    h.runtime.agent.model = Scripted(
        call("select_claim", case_id="CL-2048"),
        call("get_claim_details"),
        say("Your healthcare claim CL-2048 was denied because the pathology report and office note were missing."),
    )
    h.say("What about my healthcare claim?")
    assert h.state.email.status == "none" and h.state.phase == Phase.PROCESS_CASE and h.state.case.case_id == "CL-2048"
    h.runtime.agent.model = model


def test_goodbye_without_offering_the_summary_is_blocked(harness):
    summary = "We discussed your healthcare claim CL-2048, which is denied. Please send the pathology report and the office note through the member portal. Claims Support Team"
    h = harness()
    h.say(DEMO_UTTERANCE)
    h.runtime.agent.model = Scripted(
        say("You're welcome, Margaret. Take care."),
        call("offer_email_summary", summary=summary),
        say("I've prepared a summary. Would you like it emailed to the address on file?"),
    )
    h.say("Thanks, that's all.")
    assert h.state.email.status == "offered" and h.state.phase == Phase.POST_PROCESS


def test_wrong_month_year_is_blocked(harness):
    model = Scripted(
        call("verify_identity", **IDENTITY), call("list_my_claims"),
        say("You're verified. Is it the closed healthcare claim from January 2026?"),
        say("You're verified. Is it the closed healthcare claim from January 2025 or the denied one from January 2026?"),
    )  # fmt: skip
    h = harness(model=model)
    payload = h.say(DEMO_UTTERANCE)
    assert h.last_reply(payload).endswith("denied one from January 2026?")


def test_ignored_identity_field_comes_with_a_fix(harness):
    model = Scripted(call("verify_identity", **{**IDENTITY, "dob": "3/15/85"}), say("Could you give your full date of birth?"))
    h = harness(model=model)
    h.say("hey its margaret chen dob 3/15/85 ssn 4472")
    out = model.outputs()[0]
    assert "dob" in out["ignored_fields"] and "four digit year" in out["how_to_fix_ignored_fields"]["dob"]


def test_email_can_be_reoffered_after_skip(harness):
    h = harness()
    h.say("My name is Margaret Chen, DOB 1985-03-15, SSN last four 4472. Checking on my auto claim.")
    h.say("that's all")
    h.act("email_skip")
    assert h.state.email.status == "skipped"
    summary = "We discussed your auto claim CL-2102, which is open and in progress. There are no outstanding documents. Claims Support Team"
    h.runtime.agent.model = Scripted(
        call("offer_email_summary", summary=summary), say("Sure, here it is again. Would you like it emailed?")
    )
    h.say("actually yes please send it")
    assert h.state.email.status == "offered"


def test_identity_values_wrapped_in_phrases_are_accepted(harness):
    args = {**IDENTITY, "dob": "Born March 15, 1985", "id_last4": "SSN ending 4472"}
    model = Scripted(call("verify_identity", **args), say("Thanks, you're verified."))
    h = harness(model=model)
    h.say("Margaret Chen. Born March 15, 1985 and SSN ending 4472.")
    assert model.outputs()[0]["status"] == "verified" and h.state.verified


def test_two_digit_year_and_spanish_dates_are_grounded(harness):
    for text, dob in (
        ("margaret chen dob 3/15/85 ssn 4472", "1985-03-15"),
        ("Soy Margaret Chen, nací el 15 de marzo de 1985, seguro social 4472", "15 de marzo de 1985"),
    ):
        model = Scripted(call("verify_identity", **{**IDENTITY, "dob": dob}), say("Gracias."))
        h = harness(model=model)
        h.say(text)
        assert h.state.verified, text


def test_first_refusal_cannot_be_escalated_but_second_can(harness):
    # Refusals are counted by code from the guard's review, so the agent cannot skip or fake the count.
    model = Scripted(
        call("request_human", reason="repeated_refusal"),
        say("I understand. Verification protects your claim, and any three details work."),
    )
    h = harness(model=model)
    h.say("I already told you who I am. This is ridiculous.")
    assert model.outputs()[0]["error"] == "too_early"
    assert h.state.counters.refusals == 1 and not h.state.handoff.requested
    model2 = Scripted(call("request_human", reason="repeated_refusal"), say("I won't keep asking. A representative will follow up."))
    h.runtime.agent.model = model2
    h.say("I said no. I'm not giving you anything.")
    assert h.state.counters.refusals == 2 and model2.outputs()[0]["status"] == "handoff_requested" and h.state.handoff.requested


def test_internal_vocabulary_never_reaches_the_caller(harness):
    model = Scripted(
        say("You are still in the VERIFY_ID phase, so I need more details."),
        say("I still need a couple more details. Could you share your full name and date of birth?"),
    )
    h = harness(model=model)
    payload = h.say("hello")
    assert h.last_reply(payload) == "I still need a couple more details. Could you share your full name and date of birth?"
    feedback = [i for i in model.requests[1]["input"] if i.get("role") == "developer"]
    assert "internal_reference" in feedback[-1]["content"]


def test_offline_demo_follows_the_readme_walkthrough(harness):
    # The root README tells reviewers to type these exact messages; they must work without an API key too.
    h = harness()
    h.say("Hi, I'm calling about my denied healthcare claim from January.")
    h.say("My name is Margaret Chen, date of birth March 15, 1985, SSN last four 4472.")
    assert h.state.verified and h.state.case.case_id == "CL-2048"
    h.say("That's all, thanks.")
    assert h.state.email.status == "offered"
    h.act("email_send")
    assert h.state.email.status == "queued"
