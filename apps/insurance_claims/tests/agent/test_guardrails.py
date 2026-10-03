"""Unit tests for insurance_claims.agent.guardrails (style, markup, leaks, grounding)."""

from __future__ import annotations

import dataclasses
import random
import re
import time
from collections.abc import Sequence
from datetime import date
from decimal import Decimal

import pytest

from insurance_claims.agent.guardrails import (
    COLONS,
    DEADLINE_MISSTATED,
    EM_DASHES,
    EMAIL_STATUS_MISSTATED,
    FORBIDDEN_PROMISE,
    GROUNDING_CODES,
    INTERNAL_REFERENCE,
    LEAK_CLAIM_DATA,
    STATUS_CONTRADICTED,
    STYLE_CODES,
    STYLE_COLON,
    STYLE_EM_DASH,
    UNSUPPORTED_AMOUNT,
    UNSUPPORTED_CASE_ID,
    UNSUPPORTED_DATE,
    GroundingContext,
    Violation,
    check_email_status,
    check_grounding,
    check_internal_reference,
    check_pre_verification_leak,
    check_style,
    mechanical_style_fix,
    sanitize_markup,
)
from insurance_claims.claims.evidence import GroundingTokens

TODAY = date(2026, 10, 3)
DEADLINE = date(2026, 3, 18)

# Secret tokens in the shape evidence.secret_claim_tokens produces (lowercased) for the fixture claims.
SECRET_TOKENS = [
    "cl-2048", "cl-2011", "cl-1899", "cl-2102", "cl-3001",
    "1450.00", "1,450.00", "1450", "780.00", "780", "3200.00", "3,200.00", "3200", "1200.00", "1,200.00", "1200",
    "the review file did not include the pathology report and the treating provider office note",
    "the submitted materials did not include the treating provider diagnosis report",
    "pathology report", "office note", "diagnosis report",
    "2026-03-18", "march 18, 2026", "march 18", "2026-01-12", "january 12, 2026", "january 12",
    "2026-03-01", "march 1, 2026", "march 1", "2026-04-15", "april 15, 2026", "april 15",
]  # fmt: skip

COMPLIANT_CASE_REPLY = (
    "I'm sorry this has been stressful. Your healthcare claim CL-2048 from January 12, 2026 was denied because "
    "the review file did not include the pathology report and the treating provider office note. The allowed "
    "amount was $1,450.00 and nothing has been paid yet, so the net pay is $0.00. The appeal deadline was "
    "March 18, 2026, which has already passed, so a claims representative can review what options remain. "
    "Once the missing documents are received, the review usually takes less than a week. I can't guarantee "
    "the outcome, but a complete and readable copy of each document gives the reviewer what they asked for."
)


def check_reply(text: str, *, verified: bool, secret_tokens: Sequence[str], grounding: GroundingContext | None) -> list[Violation]:
    """The checks ReplyGuard combines, in its order (style, internal references, leak, grounding)."""
    violations = check_style(text) + check_internal_reference(text)
    if not verified:
        violations += check_pre_verification_leak(text, secret_tokens)
    if grounding is not None:
        violations += check_grounding(text, grounding)
    return violations


def tokens(
    *,
    amounts: tuple[str, ...] = ("0.00", "1450.00"),
    dates: tuple[date, ...] = (date(2026, 1, 12), DEADLINE),
) -> GroundingTokens:
    return GroundingTokens(
        case_ids=frozenset({"CL-2048"}),
        amounts=frozenset(Decimal(a) for a in amounts),
        dates=frozenset(dates),
        documents=frozenset({"pathology report", "office note"}),
    )


def ctx(
    *,
    allowed: tuple[str, ...] = ("CL-2048",),
    deadline: date | None = DEADLINE,
    passed: bool = True,
    grounding_tokens: GroundingTokens | None = None,
    case_status: str | None = None,
    email_status: str | None = None,
) -> GroundingContext:
    return GroundingContext(
        tokens=grounding_tokens or tokens(),
        allowed_case_ids=frozenset(allowed),
        today=TODAY,
        deadline=deadline,
        deadline_passed=passed,
        case_status=case_status,
        email_status=email_status,
    )


def codes(violations: list[Violation]) -> list[str]:
    return [v.code for v in violations]


def words_and_numbers(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9$]+", text)


# ---------------------------------------------------------------------------
# Data classes and code groups
# ---------------------------------------------------------------------------


def test_violation_and_context_are_frozen() -> None:
    v = Violation("x", "y")
    with pytest.raises(dataclasses.FrozenInstanceError):
        v.code = "z"  # type: ignore[misc]
    c = ctx()
    with pytest.raises(dataclasses.FrozenInstanceError):
        c.deadline_passed = False  # type: ignore[misc]


def test_code_groups_are_disjoint_and_complete() -> None:
    assert {STYLE_EM_DASH, STYLE_COLON} == STYLE_CODES
    assert {
        UNSUPPORTED_CASE_ID, UNSUPPORTED_AMOUNT, UNSUPPORTED_DATE, FORBIDDEN_PROMISE, DEADLINE_MISSTATED,
        STATUS_CONTRADICTED, EMAIL_STATUS_MISSTATED,
    } == GROUNDING_CODES  # fmt: skip
    assert not STYLE_CODES & GROUNDING_CODES
    assert all(code.startswith("style_") for code in STYLE_CODES)
    assert not any(code.startswith("style_") for code in GROUNDING_CODES | {LEAK_CLAIM_DATA, INTERNAL_REFERENCE})


# ---------------------------------------------------------------------------
# check_style
# ---------------------------------------------------------------------------


def test_check_style_accepts_clean_text() -> None:
    assert check_style("Thanks, Margaret. Your claim was denied on March 18, 2026, and the amount is $1,450.00.") == []
    assert check_style("") == []


def test_check_style_flags_em_dash() -> None:
    result = check_style("Your claim \u2014 the one from January \u2014 was denied.")
    assert codes(result) == [STYLE_EM_DASH]
    assert "2" in result[0].detail


@pytest.mark.parametrize("char", list(EM_DASHES))
def test_check_style_flags_every_em_dash_variant(char: str) -> None:
    assert codes(check_style(f"word{char}word")) == [STYLE_EM_DASH]


@pytest.mark.parametrize("char", [":", "\uff1a", "\ufe55", "\ua789", "\u2236"])
def test_check_style_flags_colon_variants(char: str) -> None:
    assert codes(check_style(f"Status{char} denied")) == [STYLE_COLON]


def test_check_style_flags_times_written_with_colons() -> None:
    assert codes(check_style("Call back at 3:30 pm.")) == [STYLE_COLON]


def test_check_style_reports_both_codes_in_order() -> None:
    assert codes(check_style("Note: denied \u2014 sorry")) == [STYLE_EM_DASH, STYLE_COLON]


def test_check_style_allows_hyphen_and_en_dash() -> None:
    assert check_style("Claim CL-2048 covers 2025\u20132026 and a follow-up.") == []


def test_style_details_never_echo_text() -> None:
    secret = "Margaret Chen 1985-03-15"
    for v in check_style(f"{secret}: \u2014"):
        assert "Margaret" not in v.detail and "1985" not in v.detail


# ---------------------------------------------------------------------------
# mechanical_style_fix
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("raw", "expected"), [
    ("Your claim\u2014CL-2048\u2014was denied.", "Your claim, CL-2048, was denied."),
    ("Your claim \u2014 CL-2048 \u2014 was denied.", "Your claim, CL-2048, was denied."),
    ("Status: denied.", "Status, denied."),
    ("Status:denied", "Status, denied"),
    ("Status\uff1a denied", "Status, denied"),
    ("Here is what I found:\nYour claim was denied.", "Here is what I found.\nYour claim was denied."),
    ("\u2014 Hello there", "Hello there"),
    ("Hello there \u2014", "Hello there."),
    ("Wait...: no", "Wait... no"),
    ("Denied. \u2014 Next steps follow.", "Denied. Next steps follow."),
    ("Denied \u2014.", "Denied."),
    ("(note: )", "(note)"),
    ("A \u2014\u2014 B", "A, B"),
    ("A : \u2014 B", "A, B"),
    ("Hi Margaret,\nYour claim\uff1adenied", "Hi Margaret,\nYour claim, denied"),
])  # fmt: skip
def test_mechanical_style_fix_examples(raw: str, expected: str) -> None:
    assert mechanical_style_fix(raw) == expected


def test_mechanical_style_fix_rewrites_times_without_colons() -> None:
    assert mechanical_style_fix("We open at 9:00 and close at 5:30 pm.") == "We open at 9 and close at 5.30 pm."
    assert check_style(mechanical_style_fix("at 10\uff1a45")) == []


def test_mechanical_style_fix_keeps_every_word_and_number() -> None:
    raw = (
        "Here is the summary: your claim CL-2048 \u2014 filed January 12, 2026 \u2014 was denied; "
        "allowed amount: $1,450.00, net pay: $0.00. Deadline: March 18, 2026 (already passed)."
    )
    fixed = mechanical_style_fix(raw)
    assert check_style(fixed) == []
    assert words_and_numbers(fixed) == words_and_numbers(raw)
    assert "$1,450.00" in fixed and "$0.00" in fixed and "CL-2048" in fixed and "March 18, 2026" in fixed


def test_mechanical_style_fix_leaves_clean_text_alone() -> None:
    clean = "Thanks for waiting. Your claim CL-2048 was denied, and the deadline has passed."
    assert mechanical_style_fix(clean) == clean


def test_mechanical_style_fix_preserves_line_structure_of_emails() -> None:
    body = "Hello Margaret,\n\nWe discussed your claim.\n\nClaims Support Team"
    assert mechanical_style_fix(body) == body


def test_mechanical_style_fix_strips_injected_sentinel_characters() -> None:
    assert mechanical_style_fix("a\ue000b") == "ab"


def test_mechanical_style_fix_fuzz_always_style_clean_and_idempotent() -> None:
    rng = random.Random(20261003)
    alphabet = list("abc XYZ019$.,;!?()'\"\n\t-") + list(EM_DASHES) + list(COLONS) + ["\u2013", "\ue000"]
    for _ in range(3000):
        raw = "".join(rng.choice(alphabet) for _ in range(rng.randint(0, 40)))
        fixed = mechanical_style_fix(raw)
        assert check_style(fixed) == [], repr(raw)
        assert mechanical_style_fix(fixed) == fixed, repr(raw)
        kept = re.findall(r"[A-Za-z]+", raw.replace("\ue000", ""))  # the marker char is dropped first
        assert re.findall(r"[A-Za-z]+", fixed) == kept, repr(raw)


# ---------------------------------------------------------------------------
# sanitize_markup
# ---------------------------------------------------------------------------


def test_sanitize_markup_strips_bold_and_italics() -> None:
    assert sanitize_markup("Your claim is **denied** and __closed__.") == "Your claim is denied and closed."
    assert sanitize_markup("This is *important* and _urgent_.") == "This is important and urgent."


def test_sanitize_markup_strips_headings_bullets_quotes_and_code() -> None:
    raw = (
        "## Claim summary\n"
        "**Status** denied\n"
        "- the pathology report\n"
        "* the office note\n"
        "+ a third item\n"
        "\u2022 a fourth item\n"
        "1. upload them\n"
        "2) wait for review\n"
        "> quoted text\n"
        "Use `the portal` now"
    )
    assert sanitize_markup(raw) == (
        "Claim summary\nStatus denied\nthe pathology report\nthe office note\na third item\n"
        "a fourth item\nupload them\nwait for review\nquoted text\nUse the portal now"
    )


def test_sanitize_markup_removes_code_fences_rules_and_links() -> None:
    raw = "```text\nHello\n```\n---\nSee [the portal](https://example.com/portal) today."
    assert sanitize_markup(raw) == "Hello\n\nSee the portal today."


def test_sanitize_markup_collapses_spaces_and_blank_lines() -> None:
    assert sanitize_markup("  Hello    there \t friend  \n\n\n\nBye  ") == "Hello there friend\n\nBye"


def test_sanitize_markup_keeps_claim_facts_and_identifiers() -> None:
    text = "Claim CL-2048 has $1,450.00 allowed, email margaret_chen@email.com, rate 5 * 3."
    assert sanitize_markup(text) == text


def test_sanitize_markup_keeps_plain_text_and_hyphenated_words() -> None:
    text = "A follow-up is needed.\nThe re-review starts after upload."
    assert sanitize_markup(text) == text


def test_sanitize_markup_then_style_fix_yields_clean_prose() -> None:
    raw = "**Status:** denied\n- **Deadline:** March 18, 2026 \u2014 passed"
    cleaned = mechanical_style_fix(sanitize_markup(raw))
    assert cleaned == "Status, denied\nDeadline, March 18, 2026, passed"
    assert check_style(cleaned) == []


# ---------------------------------------------------------------------------
# check_pre_verification_leak
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("reply", [
    "Thanks, I've noted that you're calling about a denied healthcare claim from January, and I'll look at it right after verification.",
    "To protect your private claim information, I need three details, such as your full name and date of birth.",
    "I need 3 of these 5 details before we continue.",
    "I could not verify you with the details provided. Could you share your phone number on file?",
    "Thanks for that. Just one more detail please.",
    "Got it, March 15 is noted.",
    "Please hold on.",
])  # fmt: skip
def test_leak_check_allows_safe_verification_replies(reply: str) -> None:
    assert check_pre_verification_leak(reply, SECRET_TOKENS) == []


@pytest.mark.parametrize("reply", [
    "Your claim CL-2048 was denied.",
    "Your claim cl 2048 was denied.",
    "Your claim CL2048 was denied.",
    "Your claim CL\u20132048 was denied.",
    "The allowed amount is $1,450.00.",
    "The allowed amount is $1,450.",
    "The allowed amount is 1450 dollars.",
    "The appeal deadline was March 18, 2026.",
    "The appeal deadline was Mar. 18th.",
    "The appeal deadline was 18 March 2026.",
    "The appeal deadline was 3/18/2026.",
    "The appeal deadline was 2026-03-18.",
    "It was missing the PATHOLOGY REPORT.",
    "They need the office note.",
    "The claim was filed on January 12.",
    "The submitted materials did not include the treating provider diagnosis report.",
    "Your claim \uff23\uff2c-2048 was denied.",
])  # fmt: skip
def test_leak_check_catches_claim_data_in_many_formats(reply: str) -> None:
    result = check_pre_verification_leak(reply, SECRET_TOKENS)
    assert codes(result) == [LEAK_CLAIM_DATA]


def test_leak_check_respects_word_and_number_boundaries() -> None:
    # "march 1" is a token (another party's claim date); "March 15" and "1450x" are not matches.
    assert check_pre_verification_leak("Your birthday is March 15, 1985.", ["march 1"]) == []
    assert check_pre_verification_leak("Reference 14500 noted.", ["1450"]) == []
    assert check_pre_verification_leak("Reference 21450 noted.", ["1450"]) == []
    assert check_pre_verification_leak("The note was a footnote.", ["office note"]) == []


def test_leak_check_detail_never_contains_the_token() -> None:
    result = check_pre_verification_leak("Your claim CL-2048 was denied for the pathology report.", SECRET_TOKENS)
    assert len(result) == 1
    detail = result[0].detail.lower()
    assert "2048" not in detail and "pathology" not in detail
    assert "2" in detail  # counts only


def test_leak_check_ignores_empty_short_and_non_string_tokens() -> None:
    assert check_pre_verification_leak("a b c", ["", " ", "a", None]) == []  # type: ignore[list-item]
    assert check_pre_verification_leak("anything", []) == []


# ---------------------------------------------------------------------------
# check_grounding: case IDs
# ---------------------------------------------------------------------------


def test_compliant_case_reply_has_no_violations() -> None:
    assert check_grounding(COMPLIANT_CASE_REPLY, ctx()) == []
    assert check_reply(COMPLIANT_CASE_REPLY, verified=True, secret_tokens=SECRET_TOKENS, grounding=ctx()) == []


def test_unsupported_case_id_is_flagged_with_its_id() -> None:
    result = check_grounding("Your claim CL-3001 was denied.", ctx())
    assert codes(result) == [UNSUPPORTED_CASE_ID]
    assert "CL-3001" in result[0].detail


@pytest.mark.parametrize("text", ["About cl 3001.", "About CL3001.", "About cl-3001 and CL-2048."])
def test_unsupported_case_id_variants_are_flagged(text: str) -> None:
    assert codes(check_grounding(text, ctx())) == [UNSUPPORTED_CASE_ID]


@pytest.mark.parametrize("text", ["About CL-2048.", "About cl 2048.", "About CL2048."])
def test_allowed_case_id_variants_pass(text: str) -> None:
    assert check_grounding(text, ctx()) == []


def test_allowed_case_ids_are_normalized() -> None:
    assert check_grounding("About CL-2048.", ctx(allowed=("cl 2048",))) == []


def test_any_case_id_is_unsupported_when_none_allowed() -> None:
    assert codes(check_grounding("About CL-2048.", ctx(allowed=()))) == [UNSUPPORTED_CASE_ID]


# ---------------------------------------------------------------------------
# check_grounding: amounts
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("text", [
    "The allowed amount is $1,450.00.",
    "The allowed amount is 1450.00.",
    "The allowed amount is 1,450.00.",
    "The allowed amount is $1450.",
    "The allowed amount is 1450 dollars.",
    "The allowed amount is USD 1,450.00.",
    "Nothing was paid, so that is $0.",
    "Net pay was $0.00 and the fee was $1,450.00.",
])  # fmt: skip
def test_supported_amounts_in_several_formats_pass(text: str) -> None:
    assert check_grounding(text, ctx()) == []


@pytest.mark.parametrize("text", [
    "You were paid $999.00.",
    "You will receive 999.00 soon.",
    "The fee was $1,400.",
    "The fee was 1,450.50.",
    "The fee was 2500 dollars.",
    "The fee was $1450.01.",
])  # fmt: skip
def test_unsupported_amounts_are_flagged(text: str) -> None:
    result = check_grounding(text, ctx())
    assert UNSUPPORTED_AMOUNT in codes(result)


def test_unsupported_amount_detail_hides_the_value() -> None:
    result = check_grounding("You were paid $999.00 and $888.00.", ctx())
    assert codes(result) == [UNSUPPORTED_AMOUNT]
    assert "999" not in result[0].detail and "2" in result[0].detail


@pytest.mark.parametrize("text", [
    "It usually takes 5 to 7 days.",
    "Upload 2 documents in 1 PDF.",
    "Call at 3.30 pm.",
    "The claim CL-2048 was filed in 2026.",
    "On 2026-03-18 the deadline passed.",
    "Your last four digits 4472 were noted.",
])  # fmt: skip
def test_non_amount_numbers_are_not_treated_as_amounts(text: str) -> None:
    assert UNSUPPORTED_AMOUNT not in codes(check_grounding(text, ctx()))


def test_zero_is_unsupported_when_not_in_the_evidence() -> None:
    no_zero = ctx(grounding_tokens=tokens(amounts=("1450.00",)))
    assert codes(check_grounding("You were paid $0.", no_zero)) == [UNSUPPORTED_AMOUNT]


# ---------------------------------------------------------------------------
# check_grounding: dates
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("text", [
    "The appeal deadline March 18, 2026 has passed.",
    "The appeal deadline 2026-03-18 has passed.",
    "The appeal deadline March 18th has passed.",
    "The appeal deadline Mar 18, 2026 has passed.",
    "The appeal deadline 3/18/2026 has passed.",
    "The appeal deadline 18 March 2026 has passed.",
    "The appeal deadline the 18th of March has passed.",
    "It was filed on January 12.",
    "As of today, October 3, 2026, nothing has changed.",
])  # fmt: skip
def test_supported_dates_pass(text: str) -> None:
    assert check_grounding(text, ctx()) == []


@pytest.mark.parametrize("text", [
    "It was reviewed on March 19, 2026.",
    "It was reviewed on 2026-03-19.",
    "It was reviewed on March 19th.",
    "It was reviewed on 4/2/2026.",
    "It was reviewed on March 18, 2025.",
    "It was reviewed on February 30, 2026.",
])  # fmt: skip
def test_unsupported_dates_are_flagged(text: str) -> None:
    result = check_grounding(text, ctx())
    assert codes(result) == [UNSUPPORTED_DATE]


def test_unsupported_date_detail_hides_the_value() -> None:
    result = check_grounding("Born March 15, 1985.", ctx())
    assert codes(result) == [UNSUPPORTED_DATE]
    assert "1985" not in result[0].detail and "march" not in result[0].detail.lower()


@pytest.mark.parametrize("text", [
    "Your claim from January was denied.",
    "Your claim from January 2026 was denied.",
    "These 2 may apply to your file.",
    "Review takes less than a week.",
])  # fmt: skip
def test_month_or_year_alone_is_not_a_date(text: str) -> None:
    assert UNSUPPORTED_DATE not in codes(check_grounding(text, ctx()))


# ---------------------------------------------------------------------------
# check_grounding: forbidden promises
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("text", [
    "I guarantee the claim will be reviewed quickly.",
    "This is guaranteed to work.",
    "Your claim will be approved once you upload them.",
    "It should be approved next week.",
    "The denial is going to be overturned.",
    "It will definitely work out.",
    "I'm 100% sure about this.",
    "I've submitted your documents.",
    "I have uploaded the pathology report for you.",
    "I just filed an appeal for you.",
    "We have submitted the paperwork.",
    "I went ahead and submitted everything.",
    "I've sent your documents to the review team.",
    "I'll submit the office note for you.",
    "I will upload them today.",
    "I'll send the documents over.",
    "Let me file that appeal now.",
    "I can submit it for you.",
    "You are entitled to the full amount.",
    "You're entitled to a refund.",
    "You have the right to appeal.",
    "We will approve your claim.",
    "I promise this gets fixed.",
    "I\u2019ve submitted your documents.",
])  # fmt: skip
def test_forbidden_promises_are_flagged(text: str) -> None:
    assert FORBIDDEN_PROMISE in codes(check_grounding(text, ctx()))


@pytest.mark.parametrize("text", [
    "I can't guarantee the outcome.",
    "I cannot guarantee how the review will go.",
    "There's no guarantee, but a complete file helps.",
    "Approval is not guaranteed.",
    "I can't say whether it will be approved.",
    "I'm not sure it will be paid.",
    "I'm not 100% sure about that.",
    "I can't approve your claim myself.",
    "Only the review team can approve your claim.",
    "I can't submit documents for you, but you can upload them through the member portal.",
    "I haven't submitted anything on your behalf.",
    "Once you've submitted the documents, the review restarts.",
    "Your claim was filed on January 12, 2026.",
    "I've sent your claim summary to the email on file.",
    "I'll send you a summary by email if you'd like.",
    "I've submitted a request for a representative to call you.",
    "I'll file a request for a human representative.",
    "I've flagged this for a human representative.",
    "Once the files are received, the claim will be reviewed again.",
    "The claim may be approved after review.",
])  # fmt: skip
def test_hedged_or_harmless_wording_is_allowed(text: str) -> None:
    assert FORBIDDEN_PROMISE not in codes(check_grounding(text, ctx()))


def test_promise_detail_uses_fixed_labels() -> None:
    result = check_grounding("I guarantee it. I've submitted your documents.", ctx())
    assert codes(result) == [FORBIDDEN_PROMISE]
    assert result[0].detail == "forbidden wording claimed_action guarantee"


# ---------------------------------------------------------------------------
# check_grounding: deadline wording
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("text", [
    "You have until March 18, 2026 to appeal.",
    "You can still appeal by March 18, 2026.",
    "The appeal deadline is coming up on March 18, 2026.",
    "Make sure to appeal before March 18th.",
    "The claim was denied, and you have until March 18, 2026 to appeal.",
    "Your appeal deadline is March 18, 2026, so you still have time.",
    "You still have time, and the appeal deadline is March 18, 2026.",
    "There are days remaining before the appeal deadline.",
])  # fmt: skip
def test_passed_deadline_described_as_open_is_flagged(text: str) -> None:
    assert DEADLINE_MISSTATED in codes(check_grounding(text, ctx()))


@pytest.mark.parametrize("text", [
    "The appeal deadline was March 18, 2026, which has already passed.",
    "The appeal deadline of March 18, 2026 has passed.",
    "You had until March 18, 2026 to appeal, but that date has passed.",
    "The deadline to appeal by March 18, 2026 has expired.",
    "The appeal deadline listed on the claim was March 18, 2026.",
    "Please upload the documents within a week.",
    "You can still upload the documents through the member portal.",
])  # fmt: skip
def test_passed_deadline_described_accurately_is_allowed(text: str) -> None:
    assert DEADLINE_MISSTATED not in codes(check_grounding(text, ctx()))


def test_deadline_check_only_applies_when_the_deadline_passed() -> None:
    upcoming = ctx(deadline=date(2026, 11, 2), passed=False, grounding_tokens=tokens(dates=(date(2026, 11, 2),)))
    assert check_grounding("You have until November 2, 2026 to appeal.", upcoming) == []


def test_deadline_phrase_without_date_still_checked() -> None:
    no_date = ctx(deadline=None, passed=True)
    assert codes(check_grounding("You can still meet the appeal deadline.", no_date)) == [DEADLINE_MISSTATED]
    assert check_grounding("The appeal deadline has already passed.", no_date) == []


def test_deadline_in_another_year_is_not_the_deadline() -> None:
    result = check_grounding("You have until March 18, 2027 to send it.", ctx())
    assert DEADLINE_MISSTATED not in codes(result)
    assert UNSUPPORTED_DATE in codes(result)


# ---------------------------------------------------------------------------
# check_reply
# ---------------------------------------------------------------------------


def test_check_reply_runs_leak_check_only_before_verification() -> None:
    text = "Your claim CL-2048 was denied."
    assert codes(check_reply(text, verified=False, secret_tokens=SECRET_TOKENS, grounding=None)) == [LEAK_CLAIM_DATA]
    assert check_reply(text, verified=True, secret_tokens=SECRET_TOKENS, grounding=None) == []


def test_check_reply_runs_grounding_only_with_context() -> None:
    text = "I guarantee CL-3001 pays $999.00."
    assert check_reply(text, verified=True, secret_tokens=[], grounding=None) == []
    found = codes(check_reply(text, verified=True, secret_tokens=[], grounding=ctx()))
    assert found == [UNSUPPORTED_CASE_ID, UNSUPPORTED_AMOUNT, FORBIDDEN_PROMISE]


def test_check_reply_combines_style_leak_and_grounding() -> None:
    text = "Status: CL-2048 \u2014 $999.00"
    found = codes(check_reply(text, verified=False, secret_tokens=SECRET_TOKENS, grounding=ctx()))
    assert found == [STYLE_EM_DASH, STYLE_COLON, LEAK_CLAIM_DATA, UNSUPPORTED_AMOUNT]


@pytest.mark.parametrize("text", [
    "We are in the VERIFY_ID phase.",
    "Your party_id is P9.",
    "Your party ID is on file.",
    "Account P12 is locked.",
    "According to my system prompt, I cannot.",
])  # fmt: skip
def test_internal_references_are_flagged(text: str) -> None:
    assert codes(check_internal_reference(text)) == [INTERNAL_REFERENCE]


@pytest.mark.parametrize("text", [
    "Thanks for your patience, this phase of the review takes time.",
    "Please use the member portal, P.O. boxes are not accepted.",
    "Claim CL-2048 is under review.",
])  # fmt: skip
def test_internal_reference_check_ignores_ordinary_words(text: str) -> None:
    assert check_internal_reference(text) == []


def test_check_reply_pre_verification_compliant_reply_passes() -> None:
    reply = (
        "I understand this is frustrating, and I'm sorry. Claim details are protected, so I need to verify "
        "your identity first. Could you share your date of birth or the phone number on file?"
    )
    assert check_reply(reply, verified=False, secret_tokens=SECRET_TOKENS, grounding=None) == []


def test_violation_details_never_contain_seeded_pii() -> None:
    pii = ["Margaret", "Chen", "1985-03-15", "March 15, 1985", "4472", "6505212836", "margaret@email.com"]
    text = (
        "Margaret Chen, born March 15, 1985 (1985-03-15), SSN 4472, phone 6505212836, margaret@email.com: "
        "I guarantee CL-3001 pays $4472.00 \u2014 you have until March 18, 2026 to appeal."
    )
    violations = check_reply(text, verified=False, secret_tokens=SECRET_TOKENS + ["4472"], grounding=ctx())
    assert {STYLE_COLON, STYLE_EM_DASH, LEAK_CLAIM_DATA, UNSUPPORTED_AMOUNT, UNSUPPORTED_DATE} <= set(codes(violations))
    for violation in violations:
        for value in pii:
            assert value.lower() not in violation.detail.lower()


# ---------------------------------------------------------------------------
# Robustness: model output can be adversarial, so checks must stay fast
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("text", [
    " " * 4000,
    "\t" * 4000 + "x",
    "[" * 4000,
    "[a](" * 1000,
    ":" * 4000,
    " :" * 2000,
    " and" * 1000,
    "1," * 2000,
    "*a " * 1300,
    "march 1 " * 500,
])  # fmt: skip
def test_validators_stay_fast_on_pathological_input(text: str) -> None:
    start = time.perf_counter()
    sanitize_markup(text)
    mechanical_style_fix(text)
    check_reply(text, verified=False, secret_tokens=SECRET_TOKENS, grounding=ctx())
    assert time.perf_counter() - start < 0.5


# ---------------------------------------------------------------------------
# Regression: passed deadline described as open or current (C9, C43)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("text", [
    "The appeal deadline is March 18, 2026.",
    "The deadline to appeal is March 18, 2026, so please hurry.",
    "Your appeal deadline is March 18, 2026. You still have time to file an appeal.",
    "The appeal deadline listed was March 18, 2026, but you still have time to appeal.",
    "The appeal deadline was March 18, 2026, so you still have time.",
    "The appeal deadline was March 18, 2026. You still have time.",
    "Since your claim CL-2048 was denied in January, you have until March 18 to appeal.",
    "Since your claim was denied in January, the deadline to appeal is March 18.",
    "You have until 3/18 to appeal.",
    "The last day to appeal is 3/18.",
    "You can still file an appeal.",
    "The claim was denied, but you can still appeal.",
    "It is not too late to appeal.",
    "Your appeal window is still open.",
    "The appeal deadline is today.",
    "You still have over two weeks left before the appeal deadline.",
    "The appeal deadline is over two weeks away, on March 18, 2026.",
    "Your appeal deadline is March 18, 2026, so you have time left.",
])  # fmt: skip
def test_regression_passed_deadline_read_as_open_or_current_is_flagged(text: str) -> None:
    assert DEADLINE_MISSTATED in codes(check_grounding(text, ctx()))


@pytest.mark.parametrize("text", [
    "The appeal deadline is March 18, 2026, and it has already passed.",
    "A claims representative can tell you whether you can still appeal.",
    "You can no longer file an appeal on your own, but a representative can review your options.",
    "If you still want to appeal, a claims representative can review your options.",
    "Your claim was denied on January 12, 2026, and the appeal deadline listed was March 18, 2026, which has already passed.",
    "The appeal deadline listed on the claim was March 18, 2026, and that date has already passed, so a claims "
    "representative would need to review any remaining options with you.",
    "The appeal deadline of March 18, 2026 has gone by, so a claims representative would need to review what options are left with you.",
    "The appeal deadline of March 18, 2026 has come and gone, so a claims representative would need to review what options are left with you.",
    "The appeal deadline of March 18, 2026 came and went, so a claims representative would need to review what options are left.",
    "The appeal deadline of March 18, 2026 is over, so please talk to a claims representative before you send anything.",
    "The appeal deadline of March 18, 2026 is behind you, so a claims representative can explain what options are left.",
    "The appeal deadline, March 18, 2026, has come and gone, so a representative would need to review what options are left.",
    "Now that the appeal deadline has gone by, a claims representative can review any options left.",
    "Since the March 18, 2026 appeal deadline is behind you, a representative can look at what options are left.",
    "Because the appeal deadline is over, a claims representative would need to check with you before any next step.",
    "The appeal deadline passed on March 18, 2026. You can still upload the pathology report and the office note through the member portal.",
    "The appeal deadline passed on March 18, 2026. Please upload the documents before you call back.",
    "You still have time to upload the documents this week.",
])  # fmt: skip
def test_regression_passed_deadline_stated_accurately_is_allowed(text: str) -> None:
    assert DEADLINE_MISSTATED not in codes(check_grounding(text, ctx()))


def test_regression_appeal_still_open_wording_is_fine_before_the_deadline() -> None:
    upcoming = ctx(deadline=date(2026, 11, 2), passed=False, grounding_tokens=tokens(dates=(date(2026, 11, 2),)))
    for text in ("You can still file an appeal.", "The appeal deadline is November 2, 2026.", "You have until 11/2 to appeal."):
        assert DEADLINE_MISSTATED not in codes(check_grounding(text, upcoming)), text


# ---------------------------------------------------------------------------
# Regression: status and outcome grounding (C10)
# ---------------------------------------------------------------------------

DENIED = {"case_status": "denied"}
CLOSED_2011 = {
    "allowed": ("CL-2011",),
    "deadline": None,
    "passed": False,
    "case_status": "closed",
    "grounding_tokens": tokens(amounts=("780.00", "800.00"), dates=(date(2025, 1, 28),)),
}
OPEN_2102 = {
    "allowed": ("CL-2102",),
    "deadline": None,
    "passed": False,
    "case_status": "open",
    "grounding_tokens": tokens(amounts=("3200.00", "3500.00", "0.00"), dates=(date(2026, 2, 28),)),
}


@pytest.mark.parametrize("text", [
    "Good news, your claim CL-2048 has been approved and paid.",
    "Your claim was reprocessed and the denial has been overturned.",
    "Your claim CL-2048 is approved and the payment of $1,450.00 has been issued.",
    "Your claim is now being reviewed again.",
    "Your claim is now under review again.",
    "Your claim is still open.",
    "There's no doubt your claim has been approved.",
])  # fmt: skip
def test_regression_status_contradicting_a_denied_claim_is_flagged(text: str) -> None:
    assert STATUS_CONTRADICTED in codes(check_grounding(text, ctx(**DENIED)))
    # Without a selected-claim status the status check does not run.
    assert STATUS_CONTRADICTED not in codes(check_grounding(text, ctx()))


@pytest.mark.parametrize(("kwargs", "text"), [
    (OPEN_2102, "Your claim CL-2102 has been approved and paid."),
    (OPEN_2102, "Your claim was denied."),
    (OPEN_2102, "Your claim CL-2102 is closed."),
    (CLOSED_2011, "Your claim CL-2011 was denied."),
    (CLOSED_2011, "Your claim is still open."),
    (CLOSED_2011, "Your claim has been reopened."),
])  # fmt: skip
def test_regression_status_contradicting_open_or_closed_claim_is_flagged(kwargs: dict, text: str) -> None:
    assert STATUS_CONTRADICTED in codes(check_grounding(text, ctx(**kwargs)))


GOOD_DENIED_REPLIES = [
    COMPLIANT_CASE_REPLY,
    "Your claim was denied, and nothing has been paid so far.",
    "No payment has been issued on this claim.",
    "Your claim hasn't been approved.",
    "It was not approved because the pathology report was missing.",
    "Once the documents are received, the claim goes back into review.",
    "After you send them, your claim will be reviewed again, but approval isn't guaranteed.",
    "I can't tell you whether it will be approved.",
    "If it's approved after review, the payment would follow the normal process.",
    "The claim has not been paid, and the net payment is $0.00.",
    "Your healthcare claim CL-2048 from January 12, 2026 is currently denied.",
    "I'm sorry this has been so stressful. A claims representative can review whether any options remain.",
    "You can upload the pathology report and the office note through the member portal.",
    "Once you've submitted the documents, the review restarts.",
    "Once we've received the documents, the review usually restarts.",
    "Scanned copies are accepted as long as they're complete and readable.",
    "You were paid $0.00 on this claim so far.",
    "We talked about why the claim was denied and the documents still needed.",
    "I have no doubt a complete file helps the reviewer.",
    "The appeal deadline listed on the claim was March 18, 2026.",
]
GOOD_CLOSED_REPLIES = [
    "Your claim was settled and has been paid.",
    "The claim is closed with a net payment of $780.00.",
    "Your healthcare claim CL-2011 from January 28, 2025 is currently closed.",
    "We talked about why the claim was denied, and it is in fact closed with a payment.",
]
GOOD_OPEN_REPLIES = [
    "Your claim is still open and in progress.",
    "Your claim is under review right now.",
    "Your claim is being reviewed.",
    "Your auto claim CL-2102 from February 28, 2026 is currently open.",
    "Nothing has been paid yet, so the net payment so far is $0.00.",
    "It hasn't been approved or paid yet.",
    "The allowed maximum for this claim is $3,500.00, the expected reimbursement is $3,200.00, and the net payment so far is $0.00.",
]


@pytest.mark.parametrize(("kwargs", "text"),
    [(DENIED, t) for t in GOOD_DENIED_REPLIES]
    + [(CLOSED_2011, t) for t in GOOD_CLOSED_REPLIES]
    + [(OPEN_2102, t) for t in GOOD_OPEN_REPLIES],
)  # fmt: skip
def test_regression_grounded_replies_about_each_claim_pass(kwargs: dict, text: str) -> None:
    assert check_reply(text, verified=True, secret_tokens=SECRET_TOKENS, grounding=ctx(**kwargs)) == []


def test_regression_status_check_skipped_when_several_claims_are_in_scope() -> None:
    # A summary that also covers another claim states that claim's own status.
    text = "We also looked at your healthcare claim CL-2048, which is currently denied."
    both = ctx(allowed=("CL-2011", "CL-2048"), deadline=None, passed=False, case_status="closed")
    assert STATUS_CONTRADICTED not in codes(check_grounding(text, both))
    assert STATUS_CONTRADICTED in codes(check_grounding("Your claim was denied.", ctx(**CLOSED_2011)))


def test_regression_status_detail_names_only_the_status() -> None:
    result = check_grounding("Good news, your claim has been approved and paid.", ctx(**DENIED))
    assert [v.detail for v in result if v.code == STATUS_CONTRADICTED] == [
        "status or outcome wording conflicts with the evidence status denied"
    ]


@pytest.mark.parametrize("text", [
    "You'll be reimbursed $1,450.00 once you send the documents.",
    "Your claim will get approved.",
    "Your claim is likely to be approved.",
    "It's likely to be approved.",
    "Your claim will be reprocessed and paid within a week.",
    "You will receive $1,450.00 after the appeal.",
    "You'll get a refund once it's reviewed.",
    "You have 180 days from the denial to file an appeal.",
    "You have a right to appeal.",
    "You're eligible to appeal.",
    "Thanks, we've received your pathology report and your claim is now under review again.",
])  # fmt: skip
def test_regression_invented_outcomes_and_appeal_rights_are_promises(text: str) -> None:
    assert FORBIDDEN_PROMISE in codes(check_grounding(text, ctx(passed=False)))


@pytest.mark.parametrize("text", [
    "You will receive a confirmation email once the documents are uploaded.",
    "We have 5 business days to review your appeal documents.",
    "A representative can tell you whether you're eligible to appeal.",
    "I don't think we've received your documents yet.",
])  # fmt: skip
def test_regression_broadened_promise_patterns_leave_harmless_wording_alone(text: str) -> None:
    assert FORBIDDEN_PROMISE not in codes(check_grounding(text, ctx(passed=False)))


# ---------------------------------------------------------------------------
# Regression: intensifiers are not hedges (C29)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("text", [
    "I have no doubt your claim will be approved.",
    "There's no question it will be paid once the documents arrive.",
    "I only need the pathology report and then your claim will be approved.",
    "Without a doubt your claim will be approved.",
    "Once you upload only the pathology report your claim will be approved.",
    "Don't worry, your claim will be approved.",
    "No worries, it will be paid.",
])  # fmt: skip
def test_regression_intensified_promises_are_flagged(text: str) -> None:
    assert FORBIDDEN_PROMISE in codes(check_grounding(text, ctx()))


# ---------------------------------------------------------------------------
# Regression: invented document and appeal actions (C30)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("text", [
    "I've started an appeal for you.",
    "I have initiated the appeal on your behalf.",
    "I've forwarded your documents to the review team.",
    "I've attached the pathology report to your claim.",
    "Your appeal has been filed.",
    "Good news, your appeal has been filed.",
    "The reviewer has reopened your claim.",
    "Your claim has been reopened.",
    "I've opened an appeal on claim CL-2048 for you.",
    "I've lodged an appeal for you.",
    "We've received your pathology report and office note.",
    "Your documents have been submitted to the review team.",
    "Thanks. The pathology report has already been received.",
    "I've resent the paperwork.",
])  # fmt: skip
def test_regression_invented_document_and_appeal_actions_are_flagged(text: str) -> None:
    assert FORBIDDEN_PROMISE in codes(check_grounding(text, ctx()))


@pytest.mark.parametrize("text", [
    "I've opened a callback request for you.",
    "I've flagged this conversation for a human claims representative to follow up with you.",
    "Once your documents have been received, the review usually restarts.",
    "Your documents have not been received yet.",
    "I haven't forwarded anything on your behalf.",
    "I can't forward documents for you, but you can upload them through the member portal.",
    "Has your claim been reopened? I can't tell from here.",
    "If the claim still looks unchanged after a few business days, support can check whether the pathology "
    "report and the office note were attached correctly.",
])  # fmt: skip
def test_regression_action_wording_that_claims_nothing_is_allowed(text: str) -> None:
    assert FORBIDDEN_PROMISE not in codes(check_grounding(text, ctx()))


# ---------------------------------------------------------------------------
# Regression: email described as sent when it was not (C36)
# ---------------------------------------------------------------------------

FALSE_DELIVERY_CLAIMS = [
    "Yes, I have emailed you the summary, so it should be in your inbox now.",
    "Your summary has been sent to your email on file.",
    "Good news, the email went out and should be in your inbox shortly.",
    "It should be in your inbox now.",
    "I've emailed it over to you.",
    "The email was delivered to your address on file.",
    "I sent you the summary by email.",
    "I've sent the summary to your email.",
    "I've sent it to your email on file.",
    "Please check your inbox.",
    "If you check, it should already be in your inbox.",
]


@pytest.mark.parametrize("status", ["queued", "skipped", "failed", "delivery_unknown", "needs_human", "offered", "none"])
@pytest.mark.parametrize("text", FALSE_DELIVERY_CLAIMS)
def test_regression_sent_claim_is_flagged_unless_status_is_sent(text: str, status: str) -> None:
    assert codes(check_grounding(text, ctx(passed=False, email_status=status))) == [EMAIL_STATUS_MISSTATED]
    assert check_grounding(text, ctx(passed=False, email_status="sent")) == []
    assert check_grounding(text, ctx(passed=False)) == []  # no status supplied, no check
    assert codes(check_email_status(text, status)) == [EMAIL_STATUS_MISSTATED]


@pytest.mark.parametrize("text", [
    "I've saved the summary to the local demo outbox. This demo doesn't deliver real email, so nothing was sent to your inbox.",
    "No email was sent, since you chose to skip it.",
    "I'm sorry, the email couldn't be sent and nothing was delivered.",
    "I'm sorry, I can't confirm whether the email went through, so I've flagged it for a team member to check.",
    "Here's the summary I can email you. Would you like me to send it to m*******@email.com?",
    "Would you like me to send it to your inbox?",
    "If you say yes, it will be in your inbox shortly.",
    "Once it's sent, it should be in your inbox within a few minutes.",
    "I haven't emailed anything yet.",
    "Your claim was denied because it was sent without the pathology report.",
    "I've sent this to a human representative for review.",
])  # fmt: skip
def test_regression_truthful_email_wording_is_allowed(text: str) -> None:
    assert check_email_status(text, "queued") == []


def test_regression_email_detail_never_echoes_unknown_status_text() -> None:
    [violation] = check_email_status("I've emailed you the summary.", "Margaret Chen 1985")
    assert "Margaret" not in violation.detail and violation.detail.endswith("other")


# ---------------------------------------------------------------------------
# Regression: amount arithmetic and bare amounts (L4, L5)
# ---------------------------------------------------------------------------


def test_regression_differences_between_evidence_amounts_are_allowed_when_compared() -> None:
    closed = ctx(**CLOSED_2011)
    opened = ctx(**OPEN_2102)
    text = "You were reimbursed $780.00 of the $800.00 allowed, which is $20.00 less than the maximum."
    assert check_grounding(text, closed) == []
    assert check_grounding("The expected reimbursement of $3,200.00 is $300.00 below the allowed maximum of $3,500.00.", opened) == []
    # The same figure stated as a payment, or an invented gap, stays unsupported.
    assert codes(check_grounding("You were paid $20.00.", closed)) == [UNSUPPORTED_AMOUNT]
    assert codes(check_grounding("That is $25.00 less than the maximum.", closed)) == [UNSUPPORTED_AMOUNT]


@pytest.mark.parametrize("text", [
    "The allowed maximum is 1500.",
    "We will pay 2000 once approved.",
    "The net payment is 999.",
])  # fmt: skip
def test_regression_bare_amounts_after_money_words_are_checked(text: str) -> None:
    assert UNSUPPORTED_AMOUNT in codes(check_grounding(text, ctx(passed=False)))


@pytest.mark.parametrize("text", [
    "The allowed amount is 1450.",
    "The payment for CL-2048 is $0.00.",
    "The allowed amount for claim 2048 was $1,450.00.",
    "The payment in 2026 was $0.00.",
    "Payment usually arrives within 30 days.",
    "The allowed amount was $1,450.00 and nothing has been paid yet, so the net pay is $0.00.",
    "Your payment reference ends in digits 4472.",
])  # fmt: skip
def test_regression_bare_numbers_that_are_not_new_amounts_pass(text: str) -> None:
    assert UNSUPPORTED_AMOUNT not in codes(check_grounding(text, ctx(passed=False)))


# ---------------------------------------------------------------------------
# Every deterministic template passes grounding for every fixture claim and email status
# ---------------------------------------------------------------------------
