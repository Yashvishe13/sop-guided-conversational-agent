"""Unit tests for insurance_claims.claims.normalize (pure normalizers and grounding checks)."""

from __future__ import annotations

from datetime import date

import pytest

from insurance_claims.claims.normalize import (
    identity_value_grounded,
    name_tokens,
    names_match,
    normalize_case_id,
    normalize_dob,
    normalize_email,
    normalize_last4,
    normalize_name,
    normalize_phone,
    normalize_policy_number,
)

# ---------------------------------------------------------------------------
# Names
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Margaret Chen", "margaret chen"),
        ("  MARGARET   chen  ", "margaret chen"),
        ("José García", "jose garcia"),
        ("O'Brien-Smith", "o brien smith"),
        ("O’Brien", "o brien"),
        ("Chen, Margaret.", "chen margaret"),
        ("Margaret\tChen\n", "margaret chen"),
        ("Margaret Chen 1985", "margaret chen"),
        ("Zoë – Smith", "zoe smith"),
    ],
)
def test_normalize_name_forms(raw: str, expected: str) -> None:
    assert normalize_name(raw) == expected


@pytest.mark.parametrize("raw", ["", "   ", "1234", "!!!", "@#$"])
def test_normalize_name_empty_results_are_none(raw: str) -> None:
    assert normalize_name(raw) is None


@pytest.mark.parametrize("raw", [None, 42, b"Margaret Chen", ["Margaret"]])
def test_normalize_name_non_string_is_none(raw: object) -> None:
    assert normalize_name(raw) is None  # type: ignore[arg-type]


def test_normalize_name_rejects_oversized_input() -> None:
    assert normalize_name("a " * 5000) is None


def test_name_tokens_sorted() -> None:
    assert name_tokens("Margaret Chen") == ("chen", "margaret")
    assert name_tokens("") == ()


@pytest.mark.parametrize(
    ("provided", "canonical", "expected"),
    [
        ("Margaret Chen", "Margaret Chen", True),
        ("chen margaret", "Margaret Chen", True),
        ("MARGARET CHEN", "Margaret Chen", True),
        ("Margaret", "Margaret Chen", False),
        ("Margaret Chen Smith", "Margaret Chen", False),
        ("Margaret Chan", "Margaret Chen", False),
        ("Chen", "Chen", False),
        ("Ya Wen Li", "Ya Wen Li", True),
        ("Yawen Li", "Ya Wen Li", False),
        ("Li Li", "Li", False),
        ("Ann Ann", "Ann Ann", True),
        ("Ann Lee", "Ann Ann Lee", False),
    ],
)
def test_names_match(provided: str, canonical: str, expected: bool) -> None:
    assert names_match(provided, canonical) is expected


# ---------------------------------------------------------------------------
# Phones, emails, ID digits
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw",
    ["+1 650-521-2836", "(650) 521 2836", "6505212836", "+16505212836", "1-650-521-2836", "650.521.2836"],
)
def test_normalize_phone_accepts_us_formats(raw: str) -> None:
    assert normalize_phone(raw) == "6505212836"


@pytest.mark.parametrize("raw", ["650521283", "26505212836", "+44 20 7946 0958", "", "phone", "123"])
def test_normalize_phone_rejects_other_lengths(raw: str) -> None:
    assert normalize_phone(raw) is None


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Margaret@Email.com", "margaret@email.com"),
        ("  yawen.li@example.com ", "yawen.li@example.com"),
        ("a+b@sub.domain.org", "a+b@sub.domain.org"),
    ],
)
def test_normalize_email_valid(raw: str, expected: str) -> None:
    assert normalize_email(raw) == expected


@pytest.mark.parametrize("raw", ["margaret", "margaret@email", "@email.com", "a b@c.com", "a@@b.com", ""])
def test_normalize_email_invalid(raw: str) -> None:
    assert normalize_email(raw) is None


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("4472", "4472"), (" 44-72 ", "4472"), ("123-45-4472", "4472"), ("123454472", "4472"), ("xxx-xx-4472", "4472")],
)
def test_normalize_last4_valid(raw: str, expected: str) -> None:
    assert normalize_last4(raw) == expected


@pytest.mark.parametrize("raw", ["447", "44721", "12345678", "1234567890", "", "abcd"])
def test_normalize_last4_invalid(raw: str) -> None:
    assert normalize_last4(raw) is None


# ---------------------------------------------------------------------------
# Dates
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw",
    [
        "1985-03-15",
        "1985/03/15",
        "1985-3-15",
        "03/15/1985",
        "3/15/1985",
        "03-15-1985",
        "03.15.1985",
        "March 15, 1985",
        "March 15 1985",
        "march 15th, 1985",
        "Mar 15th 1985",
        "Mar. 15, 1985",
        "15 March 1985",
        "15th of March, 1985",
        "15 Mar 1985",
        "  March   15,   1985  ",
    ],
)
def test_normalize_dob_accepts_spellings(raw: str) -> None:
    assert normalize_dob(raw) == date(1985, 3, 15)


@pytest.mark.parametrize(
    "raw",
    [
        "03/15/85",  # two-digit year
        "85-03-15",
        "1985-02-30",  # impossible date
        "02/30/1985",
        "1985-13-01",
        "1899-12-31",  # out of range
        "2101-01-01",
        "Marchember 15, 1985",
        "15/03/1985",  # day-first numeric is not accepted (US order)
        "03/15-1985",  # mixed separators
        "yesterday",
        "",
    ],
)
def test_normalize_dob_rejects(raw: str) -> None:
    assert normalize_dob(raw) is None


def test_normalize_dob_boundaries() -> None:
    assert normalize_dob("1900-01-01") == date(1900, 1, 1)
    assert normalize_dob("2100-12-31") == date(2100, 12, 31)
    assert normalize_dob("February 29, 2024") == date(2024, 2, 29)
    assert normalize_dob("February 29, 2023") is None


# ---------------------------------------------------------------------------
# Locators
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("raw", ["pol 9921", "POL-9921", "pol9921", "Pol_9921", "9921", "policy 9921", "POL #9921"])
def test_normalize_policy_number(raw: str) -> None:
    assert normalize_policy_number(raw) == "POL-9921"


@pytest.mark.parametrize("raw", ["12", "123456789", "POL-", "CL-2048", "policy number nine", ""])
def test_normalize_policy_number_invalid(raw: str) -> None:
    assert normalize_policy_number(raw) is None


@pytest.mark.parametrize("raw", ["cl 2048", "CL2048", "cl-2048", "CL-2048", "claim 2048", "Case #2048"])
def test_normalize_case_id(raw: str) -> None:
    assert normalize_case_id(raw) == "CL-2048"


@pytest.mark.parametrize("raw", ["2048", "POL-2048", "CL-20", "CL-123456789", "", "cl-abc"])
def test_normalize_case_id_invalid(raw: str) -> None:
    assert normalize_case_id(raw) is None


# ---------------------------------------------------------------------------
# Grounding of model-extracted identity values
# ---------------------------------------------------------------------------

DEMO_UTTERANCE = (
    "I'm the policyholder. My name is Margaret Chen, policy POL-9921. I'm calling about my denied "
    "healthcare claim from January. DOB is 1985-03-15, SSN last four is 4472."
)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("full_name", "Margaret Chen"),
        ("full_name", "chen margaret"),
        ("dob", "1985-03-15"),
        ("id_last4", "4472"),
        ("policy_number", "POL-9921"),
    ],
)
def test_grounded_values_in_demo_utterance(field: str, value: str) -> None:
    assert identity_value_grounded(field, value, DEMO_UTTERANCE)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("full_name", "Margaret Smith"),
        ("dob", "1984-03-15"),  # hallucinated year
        ("dob", "1985-03-16"),  # hallucinated day
        ("id_last4", "4473"),
        ("phone", "6505212836"),
        ("email", "margaret@email.com"),
        ("policy_number", "POL-1044"),
    ],
)
def test_hallucinated_values_not_grounded(field: str, value: str) -> None:
    assert not identity_value_grounded(field, value, DEMO_UTTERANCE)


def test_dob_without_year_in_utterance_is_not_grounded() -> None:
    assert not identity_value_grounded("dob", "1985-03-15", "My birthday is March 15th")


@pytest.mark.parametrize(
    "utterance",
    ["born March 15, 1985", "dob 03/15/1985", "15 March 1985", "the 15th of March, 1985"],
)
def test_dob_grounded_in_various_spellings(utterance: str) -> None:
    assert identity_value_grounded("dob", "1985-03-15", utterance)


def test_dob_single_digit_day_accepts_padded_forms() -> None:
    assert identity_value_grounded("dob", "1989-12-03", "dob 12/03/1989")
    assert identity_value_grounded("dob", "1989-12-03", "December 3rd, 1989")
    assert not identity_value_grounded("dob", "1989-12-03", "December 13th, 1989")


@pytest.mark.parametrize(
    "utterance",
    [
        "call me at +1 650-521-2836",
        "my number is (650) 521 2836 thanks",
        "6505212836",
        "phone 650.521.2836 and ssn 4472",
        "ssn 4472 phone 650-521-2836",
        "16505212836",
    ],
)
def test_phone_grounded(utterance: str) -> None:
    assert identity_value_grounded("phone", "6505212836", utterance)


@pytest.mark.parametrize(
    "utterance",
    ["call 650 521 2837", "650-521", "my phone is 521 2836", "650 and then 5212836"],
)
def test_phone_not_grounded(utterance: str) -> None:
    assert not identity_value_grounded("phone", "6505212836", utterance)


@pytest.mark.parametrize(
    "utterance",
    ["last four 4472", "ssn 123-45-4472", "ssn 123454472", "4472."],
)
def test_last4_grounded(utterance: str) -> None:
    assert identity_value_grounded("id_last4", "4472", utterance)


@pytest.mark.parametrize("utterance", ["44721", "last four 447", "number 14472 9"])
def test_last4_not_grounded(utterance: str) -> None:
    assert not identity_value_grounded("id_last4", "4472", utterance)


def test_email_grounding_requires_whole_address() -> None:
    assert identity_value_grounded("email", "margaret@email.com", "email MARGARET@EMAIL.COM.")
    assert not identity_value_grounded("email", "margaret@email.com", "xmargaret@email.com")
    assert not identity_value_grounded("email", "margaret@email.com", "margaret@email.com.au")


def test_name_grounding_needs_every_token() -> None:
    assert identity_value_grounded("full_name", "Margaret Chen", "this is margaret, surname chen")
    assert not identity_value_grounded("full_name", "Margaret Chen", "this is margaret")


def test_grounding_rejects_unknown_field_and_bad_values() -> None:
    assert not identity_value_grounded("ssn", "4472", "4472")
    assert not identity_value_grounded("phone", "", "6505212836")
    assert not identity_value_grounded("dob", "not a date", "1985 15")
    assert not identity_value_grounded("full_name", "Margaret Chen", None)  # type: ignore[arg-type]


def test_grounding_rejects_oversized_utterance() -> None:
    assert not identity_value_grounded("id_last4", "4472", "4472 " + "x" * 10_000)


# ---------------------------------------------------------------------------
# Grounding formats (L0) and the DOB month (L1)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "utterance",
    [
        "The last four of my social are 4 4 7 2",
        "last four 4-4-7-2",
        "last four 44 72",
        "last four 4–4–7–2",
        "last four ４４７２",
    ],
)
def test_last4_grounded_when_spaced_or_hyphenated(utterance: str) -> None:
    assert identity_value_grounded("id_last4", "4472", utterance)


def test_slash_does_not_join_date_parts_into_a_last4() -> None:
    assert not identity_value_grounded("id_last4", "0315", "born 1985/03/15")


@pytest.mark.parametrize("utterance", ["650/521/2836", "650–521–2836", "650 — 521 — 2836"])
def test_phone_grounded_with_slash_or_unicode_dashes(utterance: str) -> None:
    assert identity_value_grounded("phone", "6505212836", utterance)


@pytest.mark.parametrize(
    "utterance",
    [
        "born 1985-05-15",
        "born in 1985, my claim was from January 15",
        "born on the 15th, 1985",
        "dob 05/15/1985",
        "May 15, 1985",
    ],
)
def test_dob_with_a_different_or_missing_month_is_not_grounded(utterance: str) -> None:
    assert not identity_value_grounded("dob", "1985-03-15", utterance)


@pytest.mark.parametrize(
    "utterance",
    ["dob 1985–03–15", "dob 15/03/1985", "dob 3 15 1985", "Mar 15th 1985", "1985/3/15"],
)
def test_dob_grounded_with_month_in_other_spellings(utterance: str) -> None:
    assert identity_value_grounded("dob", "1985-03-15", utterance)


def test_ambiguous_numeric_dob_reads_month_first() -> None:
    assert identity_value_grounded("dob", "1989-12-03", "dob 12/03/1989")
    assert not identity_value_grounded("dob", "1989-03-12", "dob 12/03/1989")
