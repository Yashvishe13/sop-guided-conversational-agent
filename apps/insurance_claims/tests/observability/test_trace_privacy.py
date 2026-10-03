"""Trace content and privacy across a full conversation (integration level)."""

from __future__ import annotations

import json
import re

from insurance_claims.observability import tracing
from tests.conftest import DEMO_UTTERANCE, SEEDED_KEY

SEEDED_PII = (
    "Margaret Chen",
    "margaret chen",
    "margaret@email.com",
    "+16505212836",
    "6505212836",
    "650-521-2836",
    "1985-03-15",
    "March 15, 1985",
    "4472",
    "missing pathology report and office note",
    "the review file did not include the pathology report",
    SEEDED_KEY,
)


def _run(h):
    h.say("I'm calling about my denied healthcare claim from January.")
    h.say("My name is Margaret Chen, my phone is 650-521-2836.")
    h.say("Date of birth March 15, 1985 and SSN last four 4472.")
    h.say("What documents do I need?")
    h.say("that's all")
    h.act("email_send")


def test_traces_are_nested_redacted_and_explain_decisions(harness, settings):
    h = harness()
    _run(h)
    files = sorted(settings.trace_dir.glob("*.json"))
    assert files, "each turn must write a trace"
    blob = "\n".join(p.read_text() for p in files)
    for secret in SEEDED_PII:
        if secret.isdigit():  # digit-bounded, so hex fingerprints that happen to contain the digits do not count
            assert not re.search(rf"(?<![0-9a-f]){secret}(?![0-9a-f])", blob), f"trace leaked {secret!r}"
        else:
            assert secret not in blob, f"trace leaked {secret!r}"
    assert "CL-2048" in blob  # source claim IDs are kept for debugging

    turns = [json.loads(p.read_text()) for p in files]
    turn_traces = [t for t in turns if t["name"] == "turn"]
    assert len(turn_traces) == 6
    names = {n["name"] for t in turn_traces for n in tracing.walk(t)}
    for expected in (
        "agent",
        "tool.verify_identity",
        "verification_gate",
        "phase_transition",
        "tool.select_claim",
        "tool.get_claim_details",
        "email_dispatch",
    ):
        assert expected in names, f"missing span {expected}"
    gate_outcomes = [n["output"] for t in turn_traces for n in tracing.walk(t) if n["name"] == "verification_gate"]
    assert any(o and o.get("status") == "verified" for o in gate_outcomes)
    # model call spans carry token accounting but no raw prompt or output text
    summary = tracing.summarize(turn_traces[2])
    assert summary["tokens"]["total"] >= 0
    for t in turn_traces:
        for node in tracing.walk(t):
            if node.get("type") == "llm":
                assert "instructions" not in json.dumps(node.get("input") or "")


def test_trace_files_are_private(harness, settings):
    h = harness()
    h.say(DEMO_UTTERANCE)
    for path in settings.trace_dir.glob("*.json"):
        assert path.stat().st_mode & 0o077 == 0
