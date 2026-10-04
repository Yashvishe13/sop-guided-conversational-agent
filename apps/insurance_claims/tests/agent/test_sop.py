"""The SOP file (sop.toml) is the single definition of the workflow, and the code follows it."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from insurance_claims.agent.sop import ENFORCEMENT_POINTS, SopError, SopViolation, advance, load_sop
from insurance_claims.agent.tools import SCHEMAS
from insurance_claims.config import APP_ROOT
from insurance_claims.domain.models import PHASE_ORDER, Phase
from insurance_claims.domain.state import EmailOffer, SessionState
from tests.agent.test_agent import IDENTITY, Scripted, call, say

SOP_PATH = APP_ROOT / "sop.toml"
NOW = datetime(2026, 10, 3, 15, 0, tzinfo=UTC)


@pytest.fixture(scope="module")
def sop():
    return load_sop(SOP_PATH, known_tools=SCHEMAS)


def state(phase: Phase, email: str = "none") -> SessionState:
    s = SessionState(session_id="s", created_at=NOW, last_activity_at=NOW, phase=phase)
    s.email = EmailOffer(status=email)  # type: ignore[arg-type]
    return s


# ---------------------------------------------------------------------- the shipped SOP


def test_shipped_sop_loads_with_the_four_phases_in_order(sop):
    assert tuple(p.phase for p in sop.phases) == PHASE_ORDER
    assert sop.version and len(sop.sha256) == 64


@pytest.mark.parametrize(
    ("phase", "email", "expected"),
    [
        (Phase.VERIFY_ID, "none", {"verify_identity", "request_human"}),
        (Phase.RESOLVE_INTENT, "none", {"list_my_claims", "select_claim", "request_human"}),
        (
            Phase.PROCESS_CASE,
            "none",
            {"list_my_claims", "select_claim", "get_claim_details", "get_document_guidance", "get_followup_guidance",
             "record_document_status", "offer_email_summary", "request_human"},
        ),
        (
            Phase.POST_PROCESS,
            "offered",
            {"list_my_claims", "select_claim", "get_claim_details", "get_document_guidance", "get_followup_guidance",
             "record_email_decision", "request_human"},
        ),
        (
            Phase.POST_PROCESS,
            "skipped",
            {"list_my_claims", "select_claim", "get_claim_details", "get_document_guidance", "get_followup_guidance",
             "offer_email_summary", "request_human"},
        ),
        (
            Phase.POST_PROCESS,
            "queued",
            {"list_my_claims", "select_claim", "get_claim_details", "get_document_guidance", "get_followup_guidance", "request_human"},
        ),
    ],
)  # fmt: skip
def test_tool_menus_come_from_the_sop(sop, phase, email, expected):
    assert set(sop.tool_names(state(phase, email))) == expected


def test_claim_tools_never_exist_before_verification(sop):
    claim_tools = {"list_my_claims", "select_claim", "get_claim_details", "get_document_guidance", "get_followup_guidance"}
    assert not claim_tools & set(sop.tool_names(state(Phase.VERIFY_ID)))


@pytest.mark.parametrize(
    ("phase", "event", "to"),
    [
        (Phase.VERIFY_ID, "identity_verified", Phase.RESOLVE_INTENT),
        (Phase.RESOLVE_INTENT, "claim_selected", Phase.PROCESS_CASE),
        (Phase.PROCESS_CASE, "email_summary_offered", Phase.POST_PROCESS),
        (Phase.POST_PROCESS, "case_question_after_wrap_up", Phase.PROCESS_CASE),
        (Phase.PROCESS_CASE, "caller_changed", Phase.VERIFY_ID),
        (Phase.POST_PROCESS, "verification_expired", Phase.VERIFY_ID),
    ],
)
def test_allowed_transitions(sop, phase, event, to):
    assert sop.next_phase(phase, event) == to


@pytest.mark.parametrize(
    ("phase", "event"),
    [
        (Phase.VERIFY_ID, "claim_selected"),  # no skipping verification
        (Phase.VERIFY_ID, "email_summary_offered"),
        (Phase.RESOLVE_INTENT, "email_summary_offered"),  # no summary before a claim is discussed
        (Phase.PROCESS_CASE, "identity_verified"),
        (Phase.VERIFY_ID, "made_up_event"),
    ],
)
def test_transitions_the_sop_does_not_list_are_violations(sop, phase, event):
    with pytest.raises(SopViolation):
        sop.next_phase(phase, event)


def test_advance_records_the_transition(sop):
    s = state(Phase.VERIFY_ID)
    advance(sop, s, "identity_verified", NOW)
    assert s.phase == Phase.RESOLVE_INTENT and s.phase_log[-1].reason == "identity_verified"


def test_every_rule_names_a_real_enforcement_point_and_every_point_is_used(sop):
    named = {r.enforced_by for r in sop.always_strict} | {r.enforced_by for p in sop.phases for r in p.strict}
    assert named == set(ENFORCEMENT_POINTS)


def test_markdown_view_covers_every_phase_rule_and_memory_item(sop):
    text = sop.render_markdown()
    assert all(f"## {p.phase.value}" in text for p in sop.phases)
    assert all(r.rule in text for p in sop.phases for r in p.strict)
    assert all(m.what in text for m in sop.memory)


def test_the_running_app_uses_the_sop_file(harness):
    h = harness()
    health = h.runtime.health()
    assert health["checks"]["sop"]["version"] == h.runtime.sop.version


def test_a_transition_the_sop_does_not_allow_aborts_the_turn_and_saves_nothing(harness):
    from dataclasses import replace

    model = Scripted(call("verify_identity", **IDENTITY), say("You're verified."))
    h = harness(model=model)
    agent = h.runtime.agent
    agent.sop = replace(agent.sop, transitions=tuple(t for t in agent.sop.transitions if t.event != "identity_verified"))
    with pytest.raises(SopViolation):
        h.say("My name is Margaret Chen, DOB 1985-03-15, SSN last four 4472.")
    assert not h.state.verified and h.state.phase == Phase.VERIFY_ID


# ---------------------------------------------------------------------- validation of a broken SOP file

BROKEN = [
    ('to = "RESOLVE_INTENT"', 'to = "APPROVED"', "unknown phase 'APPROVED'"),
    ('tools = ["verify_identity"]', 'tools = ["verify_identity", "pay_claim"]', "unknown tool 'pay_claim'"),
    ('record_email_decision = "email_offer_open"', 'record_email_decision = "when_convenient"', "unknown condition 'when_convenient'"),
    ('offer_email_summary = "no_open_offer_or_send"', 'verify_identity = "email_offer_open"', "not in the phase's tools"),
    ('enforced_by = "claims.party_scoped"', 'enforced_by = "trust_the_model"', "unknown enforcement point 'trust_the_model'"),
    ('enforced_by = "claims.party_scoped"', 'enforced_by = "reply.grounding"', "not named by any rule in the SOP: claims.party_scoped"),
    ('phases = ["VERIFY_ID", "RESOLVE_INTENT", "PROCESS_CASE", "POST_PROCESS"]', 'phases = ["RESOLVE_INTENT", "VERIFY_ID", "PROCESS_CASE", "POST_PROCESS"]', "workflow.phases must be"),
    ('event = "identity_verified"', 'event = "claim_selected"', "listed once"),
    ('goal = "Confirm the caller', 'aim = "Confirm the caller', "missing 'goal'"),
]  # fmt: skip


@pytest.mark.parametrize(("old", "new", "message"), BROKEN, ids=[b[2][:30] for b in BROKEN])
def test_inconsistent_sop_files_are_refused(tmp_path: Path, old: str, new: str, message: str):
    text = SOP_PATH.read_text(encoding="utf-8")
    assert old in text
    path = tmp_path / "sop.toml"
    path.write_text(text.replace(old, new, 1), encoding="utf-8")
    with pytest.raises(SopError, match=message):
        load_sop(path, known_tools=SCHEMAS)


def test_unreachable_phase_is_refused(tmp_path: Path):
    text = SOP_PATH.read_text(encoding="utf-8").replace(
        'event = "identity_verified"\nfrom = ["VERIFY_ID"]', 'event = "identity_verified"\nfrom = ["POST_PROCESS"]'
    )
    path = tmp_path / "sop.toml"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(SopError, match="unreachable"):
        load_sop(path, known_tools=SCHEMAS)


def test_missing_or_malformed_file_is_refused(tmp_path: Path):
    with pytest.raises(SopError, match="not readable"):
        load_sop(tmp_path / "nope.toml", known_tools=SCHEMAS)
    (tmp_path / "bad.toml").write_text("version = ", encoding="utf-8")
    with pytest.raises(SopError, match="Malformed"):
        load_sop(tmp_path / "bad.toml", known_tools=SCHEMAS)
