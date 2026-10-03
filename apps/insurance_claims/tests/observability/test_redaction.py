"""Redaction: known fixture values, secrets, generic PII shapes, structure-aware rules, and safety properties."""

from __future__ import annotations

import hashlib
import json
import re
import threading
import time
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import pytest

from insurance_claims.config import APP_ROOT
from insurance_claims.domain.models import Claim, Policyholder, Representative
from insurance_claims.observability.redaction import (
    PII_PREFIX,
    SECRET_MARK,
    Redactor,
    build_fixture_redactor,
)

FIXTURES = APP_ROOT / "fixtures"
SEEDED_KEY = "sk-test-SEEDED-0123456789"
DENIAL_REASON = "the review file did not include the pathology report and the treating provider office note"
MARK_RE = re.compile(r"\[PII:h:[0-9a-f]{10}\]")


def _load(name: str, model: type) -> list:
    return [model(**row) for row in json.loads((FIXTURES / name).read_text(encoding="utf-8"))]


@pytest.fixture(scope="module")
def bundle() -> SimpleNamespace:
    """Duck-typed stand-in for fixtures.FixtureBundle (only the attributes the redactor reads)."""
    return SimpleNamespace(
        policyholders=tuple(_load("policyholders.json", Policyholder)),
        claims=tuple(_load("claims.json", Claim)),
        representatives=tuple(_load("representatives.json", Representative)),
    )


@pytest.fixture(scope="module")
def redactor(bundle: SimpleNamespace) -> Redactor:
    return build_fixture_redactor(bundle, secrets=[SEEDED_KEY, None, ""], salt="unit-test-salt")


def _assert_absent(text: str, *needles: str) -> None:
    lowered = text.lower()
    for needle in needles:
        assert needle.lower() not in lowered, f"leaked {needle!r} in {text!r}"


# ---------------------------------------------------------------------------
# fingerprint
# ---------------------------------------------------------------------------


def test_fingerprint_is_salted_sha256_prefix() -> None:
    plain = Redactor()
    assert plain.fingerprint("4472") == "h:" + hashlib.sha256(b"4472").hexdigest()[:10]
    salted = Redactor(salt="pepper")
    assert salted.fingerprint("4472") == "h:" + hashlib.sha256(b"pepper4472").hexdigest()[:10]
    assert salted.fingerprint("4472") != plain.fingerprint("4472")


def test_fixture_redactor_salt_is_random_by_default(bundle: SimpleNamespace) -> None:
    first = build_fixture_redactor(bundle, secrets=[])
    second = build_fixture_redactor(bundle, secrets=[])
    assert first.redact_text("4472") != second.redact_text("4472")
    assert MARK_RE.fullmatch(first.redact_text("4472"))


# ---------------------------------------------------------------------------
# Known fixture values
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "Margaret Chen",
        "margaret chen",
        "MARGARET CHEN",
        "Chen, Margaret",
        "Chen-Margaret",
        "Ya Wen Li",
        "Yaven Li",
        "Ava Lopez",
        "Ma Tian",
        "David Chen",
    ],
)
def test_names_and_aliases_are_redacted_case_insensitively(redactor: Redactor, text: str) -> None:
    out = redactor.redact_text(f"caller said {text} today")
    assert out.startswith("caller said ") and out.endswith(" today")
    assert MARK_RE.search(out)
    for token in re.findall(r"[A-Za-z]{3,}", text):
        _assert_absent(out, token)


def test_name_tokens_of_three_or_more_letters_are_redacted_alone(redactor: Redactor) -> None:
    out = redactor.redact_text("Margaret called. Lopez too. Also Tian and Wen.")
    _assert_absent(out, "margaret", "lopez", "tian", "wen")


def test_name_tokens_respect_letter_boundaries(redactor: Redactor) -> None:
    assert redactor.redact_text("kitchen and chenille and avatar") == "kitchen and chenille and avatar"
    assert "chen" not in redactor.redact_text("chen_x chen1").lower()


def test_short_name_tokens_alone_are_not_redacted(redactor: Redactor) -> None:
    # Two-letter tokens ("Ma", "Li") are too common to redact alone; the full name is still caught.
    assert redactor.redact_text("Ma said hi to Li") == "Ma said hi to Li"
    _assert_absent(redactor.redact_text("this is Ma Tian"), "ma tian", "tian")


@pytest.mark.parametrize(
    "text",
    [
        "+16505212836",
        "6505212836",
        "16505212836",
        "(650) 521-2836",
        "650-521-2836",
        "650.521.2836",
        "+1 650 521 2836",
        "1-650-521-2836",
        "+1 (650) 521-2836",
    ],
)
def test_phone_formats_are_redacted(redactor: Redactor, text: str) -> None:
    out = redactor.redact_text(f"call {text} now")
    assert re.fullmatch(r"call \[PII:h:[0-9a-f]{10}\] now", out), out


def test_all_phone_formats_share_one_fingerprint(redactor: Redactor) -> None:
    marks = {redactor.redact_text(t) for t in ("+16505212836", "(650) 521-2836", "650.521.2836")}
    assert len(marks) == 1


@pytest.mark.parametrize(
    "text",
    [
        "1985-03-15",
        "1985/03/15",
        "03/15/1985",
        "3/15/1985",
        "15.03.1985",
        "March 15, 1985",
        "March 15 1985",
        "Mar 15th, 1985",
        "mar. 15 1985",
        "15 March 1985",
        "15th of March, 1985",
    ],
)
def test_dob_forms_are_redacted(redactor: Redactor, text: str) -> None:
    out = redactor.redact_text(f"born {text}.")
    _assert_absent(out, "1985", "march")
    assert MARK_RE.search(out)


def test_dob_forms_share_the_iso_fingerprint(redactor: Redactor) -> None:
    expected = f"{PII_PREFIX}{redactor.fingerprint('1985-03-15')}]"
    assert redactor.redact_text("1985-03-15") == expected
    assert redactor.redact_text("March 15, 1985") == expected


def test_emails_and_aliases_are_redacted(redactor: Redactor) -> None:
    out = redactor.redact_text("mail margaret@email.com or YAWEN.LI@EXAMPLE.COM or yawen.li@gmail.com")
    _assert_absent(out, "margaret", "email.com", "yawen", "example.com", "gmail")


def test_id_digits_redacted_only_as_standalone_groups(redactor: Redactor) -> None:
    out = redactor.redact_text("ssn ends 4472, id 6688; not 14472 or 44721 or CL-4472x")
    assert "14472" in out and "44721" in out
    _assert_absent(out.replace("14472", "").replace("44721", ""), "4472", "6688")


def test_policy_numbers_are_redacted_in_common_spellings(redactor: Redactor) -> None:
    out = redactor.redact_text("policy POL-9921, pol 9921, POL9921, pol_1044")
    _assert_absent(out, "9921", "1044")


def test_claim_summary_and_denial_reason_are_redacted(redactor: Redactor) -> None:
    wrapped = DENIAL_REASON.replace(" pathology ", "\n  pathology ").upper()
    out = redactor.redact_text(f"Reason: {wrapped}. Summary: Healthcare claim denied due to missing pathology report and office note")
    _assert_absent(out, "review file", "treating provider", "denied due to missing")
    assert out.startswith("Reason: [PII:h:")


def test_non_pii_survives(redactor: Redactor) -> None:
    text = (
        "CL-2048 CL-3001 phase VERIFY_ID -> PROCESS_CASE; fields full_name dob phone email id_last4 policy_number; "
        "party P9; created 2026-01-12 deadline 2026-03-18; amounts 1450.00 780.00 0.00; matched 3 of 5; "
        "start 2026-10-03T16:17:00.123Z; uuid 123e4567-e89b-12d3-a456-426614174000; "
        "sha 9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08; pathology report; status denied"
    )
    assert redactor.redact_text(text) == text


# ---------------------------------------------------------------------------
# Secrets and generic backstops
# ---------------------------------------------------------------------------


def test_registered_secrets_become_redacted_marker_anywhere(redactor: Redactor) -> None:
    out = redactor.redact_text(f"key={SEEDED_KEY} and x{SEEDED_KEY}y")
    assert SEEDED_KEY not in out and out.count(SECRET_MARK) == 2


@pytest.mark.parametrize(
    ("text", "leak"),
    [
        ("Authorization: Bearer abcdefghijklmnop1234", "abcdefghijklmnop1234"),
        ("OPENAI_API_KEY=sk-proj-AbCdEf0123456789", "sk-proj-AbCdEf0123456789"),
        ("api_key: hunter2hunter2", "hunter2hunter2"),
        ("password=correct-horse", "correct-horse"),
        ("token sk_live_abcdefghij123", "sk_live_abcdefghij123"),
    ],
)
def test_generic_credentials_are_redacted(text: str, leak: str) -> None:
    out = Redactor().redact_text(text)
    assert leak not in out and SECRET_MARK in out


def test_short_or_empty_secrets_are_ignored() -> None:
    redactor = Redactor(secrets=["", "ab", None])  # type: ignore[list-item]
    assert redactor.redact_text("ab abc") == "ab abc"


@pytest.mark.parametrize(
    ("text", "leak"),
    [
        ("write to someone.else+tag@corp.example.org please", "someone.else"),
        ("call 415-555-0199 today", "555-0199"),
        ("call (415) 555 0199 today", "555 0199"),
        ("intl +44 20 7946 0958 ok", "7946"),
        ("ssn 123-45-6789", "6789"),
        ("ssn 123456789", "123456789"),
        ("My name is Zed Quill and I need help", "Quill"),
        ("This is Priya Raman calling", "Raman"),
        ("date of birth is 1990-01-02", "1990-01-02"),
        ("DOB: January 2, 1990", "1990"),
        ("born on 02/01/1990", "1990"),
        ("SSN last four is 1234", "1234"),
        ("last 4 digits 9876", "9876"),
    ],
)
def test_generic_pii_backstops_catch_unregistered_values(text: str, leak: str) -> None:
    out = Redactor().redact_text(text)
    assert leak not in out, out
    assert MARK_RE.search(out)


def test_generic_rules_do_not_eat_ordinary_numbers() -> None:
    text = "turn 12 of 80, 1450.00 dollars, 2026-01-12 2026-03-18, version 2, took 3218.4 ms, I am Frustrated? no: i am fine"
    out = Redactor().redact_text(text)
    for keep in ("turn 12 of 80", "1450.00", "2026-01-12 2026-03-18", "3218.4", "i am fine"):
        assert keep in out


def test_redaction_is_idempotent(redactor: Redactor) -> None:
    text = f"My name is Margaret Chen, DOB 1985-03-15, ssn 4472, {SEEDED_KEY}, phone 650-521-2836, bob@x.io"
    once = redactor.redact_text(text)
    assert redactor.redact_text(once) == once
    assert "[PII:h:[PII" not in once


def test_markers_with_digit_only_fingerprints_are_not_re_redacted() -> None:
    assert Redactor().redact_text("[PII:h:6505212836]") == "[PII:h:6505212836]"


# ---------------------------------------------------------------------------
# Structure-aware redact()
# ---------------------------------------------------------------------------


def test_redact_recurses_and_keeps_keys(redactor: Redactor) -> None:
    data = {
        "case_id": "CL-2048",
        "notes": ["Margaret Chen called", ("tuple", "margaret@email.com")],
        "nested": {"deep": {"deeper": "phone +16505212836"}},
        "count": 3,
        "ok": True,
        "ratio": 0.5,
        "none": None,
    }
    out = redactor.redact(data)
    assert set(out) == set(data)
    assert out["case_id"] == "CL-2048" and out["count"] == 3 and out["ok"] is True and out["ratio"] == 0.5
    assert isinstance(out["notes"][1], tuple)
    _assert_absent(json.dumps(out), "margaret", "6505212836")


def test_identity_keys_are_fingerprinted_whole_even_for_unknown_values() -> None:
    out = Redactor(salt="s").redact(
        {
            "full_name": "Unknown Person",
            "dob": "1999-01-01",
            "phone": "555 0100",
            "email": "nobody at nowhere",
            "id_last4": 1234,
            "represented_person": "someone",
            "identity": {"policy_number": "XYZ-1"},
        }
    )
    assert all(MARK_RE.fullmatch(v) for k, v in out.items() if k != "identity"), out
    assert MARK_RE.fullmatch(out["identity"]["policy_number"])


def test_identity_keys_keep_status_codes() -> None:
    out = Redactor().redact({"email": {"status": "offered"}, "phone": "missing", "dob": "not_provided"})
    assert out == {"email": {"status": "offered"}, "phone": "missing", "dob": "not_provided"}


def test_field_value_pairs_are_redacted() -> None:
    out = Redactor().redact([{"field": "dob", "value": "2001-02-03"}, {"field": "phase", "value": "VERIFY_ID"}])
    assert MARK_RE.fullmatch(out[0]["value"]) and out[1]["value"] == "VERIFY_ID"


def test_content_keys_are_fingerprinted() -> None:
    out = Redactor().redact(
        {"utterance": "hello there", "content": "anything else", "body_text": "email body", "name": "turn",
         "draft": "rejected", "text": 3, "reply": ["Hi there", "ok"]}
    )  # fmt: skip
    assert MARK_RE.fullmatch(out["utterance"]) and MARK_RE.fullmatch(out["content"]) and MARK_RE.fullmatch(out["body_text"])
    assert out["name"] == "turn"  # span names are not content
    assert out["draft"] == "rejected" and out["text"] == 3  # status codes and counts carry no caller text
    assert MARK_RE.fullmatch(out["reply"][0]) and out["reply"][1] == "ok"


def test_credential_keys_are_redacted() -> None:
    out = Redactor().redact(
        {"api_key": "abc", "OPENAI_API_KEY": "zzz", "smtp_password": "pw", "csrf_token": "t", "tokens": {"input": 5}, "secret_hash": "ab12"}
    )
    assert out["api_key"] == out["OPENAI_API_KEY"] == out["smtp_password"] == out["csrf_token"] == SECRET_MARK
    assert out["tokens"] == {"input": 5} and out["secret_hash"] == "ab12"


def test_redact_handles_models_dataclasses_dates_and_objects(bundle: SimpleNamespace, redactor: Redactor) -> None:
    @dataclass
    class Card:
        holder: str
        born: date

    class Opaque:
        def __str__(self) -> str:
            return "opaque margaret@email.com"

    out = redactor.redact({"model": bundle.policyholders[0], "card": Card("Margaret Chen", date(1985, 3, 15)), "obj": Opaque()})
    text = json.dumps(out)
    _assert_absent(text, "margaret", "4472", "1985", "6505212836", "POL-9921")
    assert out["model"]["party_id"] == "P9"


def test_large_ints_are_checked_small_ints_kept(redactor: Redactor) -> None:
    out = redactor.redact({"phone_int": 6505212836, "ssn_int": 123456789, "tokens": 4472, "big": 10**12 + 7})
    assert MARK_RE.fullmatch(out["phone_int"]) and MARK_RE.fullmatch(out["ssn_int"])
    assert out["tokens"] == 4472  # a 4-digit count is far more likely than an ID; keys carry intent
    assert out["big"] == 10**12 + 7


def test_redact_survives_cycles_and_broken_objects() -> None:
    loop: dict = {"name": "x"}
    loop["self"] = loop

    class Broken:
        def __str__(self) -> str:
            raise RuntimeError("no")

        def model_dump(self, **_: object) -> dict:
            raise RuntimeError("no")

    out = Redactor().redact({"loop": loop, "broken": Broken(), "set": {"b", "a"}, (1, 2): "tuple key"})
    assert out["loop"]["self"] == "[cycle]"
    assert out["broken"] == "<unprintable Broken>"
    assert out["set"] == ["a", "b"] and out["(1, 2)"] == "tuple key"


# ---------------------------------------------------------------------------
# Runtime additions, inference, threading, performance
# ---------------------------------------------------------------------------


def test_add_values_infers_kinds() -> None:
    redactor = Redactor()
    redactor.add_values(["Jane Q Public", "jane@corp.io", "+1 415 555 0100", "1970-07-04", "1234", "POL-555", "secret claim words"])
    out = redactor.redact_text("Jane Public jane@corp.io 415.555.0100 July 4, 1970 #1234 pol 555 the secret  claim words here")
    _assert_absent(out, "jane", "public", "corp", "0100", "1970", "1234", "555", "secret claim")


def test_add_values_ignores_empty_short_and_none() -> None:
    redactor = Redactor()
    redactor.add_values(["", "a", "ab", None, "  "])  # type: ignore[list-item]
    assert redactor.redact_text("a ab abc") == "a ab abc"


def test_add_values_rejects_unknown_kind() -> None:
    with pytest.raises(ValueError):
        Redactor().add_values(["x"], kind="bogus")  # type: ignore[arg-type]


def test_accent_folded_names_are_matched() -> None:
    redactor = Redactor()
    redactor.add_values(["José Núñez"], kind="name")
    _assert_absent(redactor.redact_text("José Núñez and Jose Nunez and NÚÑEZ"), "jos", "nunez", "núñez")


def test_add_values_is_thread_safe() -> None:
    redactor = Redactor()
    errors: list[BaseException] = []
    names = [f"Person{chr(65 + i)} Example{chr(65 + i)}" for i in range(20)]

    def writer(index: int) -> None:
        try:
            redactor.add_values([names[index]], kind="name")
        except BaseException as exc:  # pragma: no cover - failure path
            errors.append(exc)

    def reader() -> None:
        try:
            for _ in range(50):
                redactor.redact_text(" ".join(names))
        except BaseException as exc:  # pragma: no cover - failure path
            errors.append(exc)

    threads = [threading.Thread(target=writer, args=(i,)) for i in range(20)] + [threading.Thread(target=reader) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert not errors
    _assert_absent(redactor.redact_text(" ".join(names)), *[n.split()[1] for n in names])


@pytest.mark.parametrize(
    "text",
    [
        "x" * 50_000,
        "a@" * 25_000,
        "1 " * 25_000,
        "dob" + " :" * 25_000,
        "Margaret " * 5_000,
        "4472 " * 10_000,
        "My name is " * 4_000,
    ],
    ids=["run", "at", "digits", "dob-ctx", "names", "ids", "name-ctx"],
)
def test_adversarial_inputs_stay_fast(redactor: Redactor, text: str) -> None:
    started = time.perf_counter()
    redactor.redact_text(text)
    assert time.perf_counter() - started < 2.0


def test_build_fixture_redactor_accepts_partial_bundles() -> None:
    holder = SimpleNamespace(name="Solo Person", dob="2000-01-02", phone=None, email="", id_last4="0042")
    redactor = build_fixture_redactor(SimpleNamespace(policyholders=[holder]), secrets=None, salt="")  # type: ignore[arg-type]
    _assert_absent(redactor.redact_text("Solo Person 2000-01-02 0042"), "solo", "2000", "0042")
    assert build_fixture_redactor(object(), secrets=[]).redact_text("plain") == "plain"


def test_fixture_file_values_all_redacted(bundle: SimpleNamespace, redactor: Redactor) -> None:
    raw = json.dumps([p.model_dump(mode="json") for p in bundle.policyholders])
    out = redactor.redact_text(raw)
    for holder in bundle.policyholders:
        values = [holder.name, holder.email, holder.id_last4, holder.dob.isoformat(), holder.policy_number]
        values += [re.sub(r"\D", "", holder.phone)[-10:]] + holder.email_aliases + holder.name_aliases
        _assert_absent(out, *values)
    assert '"party_id": "P9"' in out


def test_redacted_output_has_no_raw_secret_from_settings_style_names() -> None:
    redactor = Redactor(secrets=["gAAAAA-fernet-key_with=padding=="])
    assert "fernet" not in redactor.redact_text("key gAAAAA-fernet-key_with=padding== used")


def test_module_constants() -> None:
    assert SECRET_MARK == "[REDACTED]" and PII_PREFIX == "[PII:"
    assert Path(FIXTURES).is_dir()
