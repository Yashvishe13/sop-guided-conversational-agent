"""The SOP (standard operating procedure) as data: loaded from ``sop.toml`` and enforced by code.

``sop.toml`` is the single definition of the claims workflow. This module loads it, validates it,
and answers the two questions the agent loop asks on every step:

* ``tool_names(state)``: which tools the agent may call now (the phase's menu, the tools
  available in every phase, and per-tool conditions such as "only while an email offer is open");
* ``next_phase(phase, event)``: where a reported event may take the conversation. Code changes the
  phase only through this, so a transition the SOP does not list raises ``SopViolation``.

Each strict rule in the file names the code that enforces it (``enforced_by``). The names must be
keys of ``ENFORCEMENT_POINTS`` below, and every enforcement point must be named by some rule, so the
file and the code cannot drift apart silently. ``render_markdown`` gives a readable Markdown view.

Run ``python -m insurance_claims.agent.sop [path]`` to print the SOP as Markdown.
"""

from __future__ import annotations

import hashlib
import sys
import tomllib
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from insurance_claims.domain.models import PHASE_ORDER, Phase
from insurance_claims.domain.state import PhaseTransition, SessionState
from insurance_claims.observability import tracing


class SopError(RuntimeError):
    """``sop.toml`` is missing, malformed, or inconsistent with the code (startup must fail)."""


class SopViolation(RuntimeError):
    """Code tried a phase change that the SOP does not allow (a programming error)."""


# Where each strict rule is enforced. Keys are the names sop.toml uses in "enforced_by".
ENFORCEMENT_POINTS: Mapping[str, str] = {
    "guard.caller_review": "agent/loop.py ClaimsAgent._apply_caller_review, verdict from agent/guard.py review_caller",
    "guard.reply_review": "agent/reply_guard.py ReplyGuard.check (code checks, then guard judge_reply)",
    "reply.required_human_offer": "agent/loop.py human_offer_due + agent/reply_guard.py required_in_reply + fallback offer",
    "tools.phase_menu": "agent/tools.py ToolExecutor._dispatch re-checks Sop.tool_names on every call",
    "access.withdraw_on_new_speaker": "agent/loop.py ClaimsAgent._withdraw_access",
    "verification.idle_expiry": "agent/loop.py ClaimsAgent._expire_if_needed",
    "reply.pre_verification_leak": "agent/guardrails.py check_pre_verification_leak + guard disclosed_before_verification",
    "verify_identity.speaker_confirmed": "agent/tools.py ToolExecutor._verify_identity (speaker must be account_holder)",
    "verify_identity.grounded_values": "agent/tools.py ToolExecutor._verify_identity + claims/normalize.py identity_value_grounded",
    "verify_identity.three_fields": "agent/tools.py ToolExecutor._verify_identity (REQUIRED_PII)",
    "verify_identity.unique_match": "claims/verification.py apply_identity_proposals",
    "verify_identity.lockout": "claims/verification.py (per session) + agent/tools.py _party_locked (across sessions)",
    "caller_review.refusal_count": "agent/loop.py _apply_caller_review + agent/tools.py _request_human gate",
    "reply.missed_verification": "agent/reply_guard.py ReplyGuard._skipped_verification",
    "claims.party_scoped": "claims/repository.py ClaimRepository.get_for_party / list_for_party",
    "reply.grounding": "agent/guardrails.py check_grounding + guard unsupported_fact / promise_or_invented_action",
    "guard.document_status": "agent/tools.py ToolExecutor._record_document_status + guard judge_document",
    "guard.summary_check": "agent/tools.py ToolExecutor._offer_email_summary + guard judge_summary",
    "reply.email_offer_before_goodbye": "agent/reply_guard.py ReplyGuard._ends_without_email_offer",
    "guard.consent_check": "agent/tools.py ToolExecutor._record_email_decision + guard judge_consent; agent/loop.py _handle_button",
    "email.consent_rules": "agent/tools.py _record_email_decision (not in the offer's turn); agent/loop.py PendingEmail to the on-file address",
    "email.truthful_status": "agent/reply_guard.py (no 'sent' claims) + agent/loop.py complete_email delivery notices",
}

# Conditions a tool may need inside a phase (sop.toml [phases.X.conditions]).
CONDITIONS: Mapping[str, Callable[[SessionState], bool]] = {
    "email_offer_open": lambda s: s.email.status == "offered",
    # Re-offer only after a skip or a failed or blocked send, never over an open offer or a send in progress.
    "no_open_offer_or_send": lambda s: s.email.status not in ("offered", "consented", "dispatching", "sent", "queued"),
}


@dataclass(frozen=True)
class Rule:
    rule: str
    enforced_by: str


@dataclass(frozen=True)
class PhaseSpec:
    phase: Phase
    goal: str
    tools: tuple[str, ...]
    conditions: Mapping[str, str]
    strict: tuple[Rule, ...]
    flexible: tuple[str, ...]


@dataclass(frozen=True)
class Transition:
    event: str
    from_phases: tuple[Phase, ...]
    to: Phase
    trigger: str


@dataclass(frozen=True)
class MemoryItem:
    what: str
    captured: str
    used: str
    cleared: str


@dataclass(frozen=True)
class Sop:
    version: str
    sha256: str
    phases: tuple[PhaseSpec, ...]
    always_available: tuple[str, ...]
    always_strict: tuple[Rule, ...]
    always_flexible: tuple[str, ...]
    transitions: tuple[Transition, ...]
    memory: tuple[MemoryItem, ...]

    def spec(self, phase: Phase) -> PhaseSpec:
        return next(p for p in self.phases if p.phase == phase)

    def tool_names(self, state: SessionState) -> tuple[str, ...]:
        """The tools the agent may call in this state: the phase menu (minus unmet conditions) plus the always-available ones."""
        spec = self.spec(state.phase)
        allowed = [t for t in spec.tools if t not in spec.conditions or CONDITIONS[spec.conditions[t]](state)]
        return (*allowed, *(t for t in self.always_available if t not in allowed))

    def next_phase(self, phase: Phase, event: str) -> Phase:
        """The phase ``event`` leads to from ``phase``; raises SopViolation if the SOP does not allow it."""
        for t in self.transitions:
            if t.event == event and phase in t.from_phases:
                return t.to
        raise SopViolation(f"the SOP has no transition for event {event!r} in {phase.value}")

    def render_markdown(self) -> str:
        return _render(self)


def advance(sop: Sop, state: SessionState, event: str, now: datetime) -> None:
    """The only way code changes the phase: along a transition the SOP lists for this event."""
    to = sop.next_phase(state.phase, event)
    if state.phase == to:
        return
    tracing.event("phase_transition", from_phase=state.phase.value, to_phase=to.value, reason=event)
    state.phase_log.append(PhaseTransition(turn_index=state.turn_index, from_phase=state.phase, to_phase=to, reason=event, at=now))
    state.phase_log = state.phase_log[-50:]
    state.phase = to


# ---------------------------------------------------------------------- loading and validation


def load_sop(path: Path, *, known_tools: Iterable[str]) -> Sop:
    """Load ``sop.toml`` and check it against the code. Raises SopError naming the first problem."""
    path = Path(path)
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise SopError(f"SOP file not readable: {path} ({exc.strerror})") from None
    try:
        doc = tomllib.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
        raise SopError(f"Malformed SOP file: {exc}") from None
    tools = frozenset(known_tools)

    version = _text(doc, "version")
    workflow = _table(doc, "workflow")
    order = tuple(_phase(name, "workflow.phases") for name in _list(workflow, "phases", "workflow"))
    if order != PHASE_ORDER:
        raise SopError(f"workflow.phases must be {[p.value for p in PHASE_ORDER]} (adding or reordering phases needs code changes)")
    always = tuple(_tool(t, tools, "workflow.always_available") for t in _list(workflow, "always_available", "workflow"))
    always_strict = tuple(_rule(r, "workflow.always_strict") for r in _list(workflow, "always_strict", "workflow"))
    always_flexible = tuple(_str(f, "workflow.always_flexible") for f in _list(workflow, "always_flexible", "workflow"))

    phase_tables = _table(doc, "phases")
    unknown = sorted(set(phase_tables) - {p.value for p in order})
    if unknown:
        raise SopError(f"Unknown phases in [phases]: {', '.join(unknown)}")
    phases = tuple(_phase_spec(p, phase_tables, tools) for p in order)

    transitions = tuple(_transition(t, i) for i, t in enumerate(_list(doc, "transitions", "file")))
    memory = tuple(_memory(m, i) for i, m in enumerate(_list(doc, "memory", "file")))
    sop = Sop(version, hashlib.sha256(raw).hexdigest(), phases, always, always_strict, always_flexible, transitions, memory)
    _check_consistency(sop)
    return sop


def _check_consistency(sop: Sop) -> None:
    events = [t.event for t in sop.transitions]
    duplicates = sorted({e for e in events if events.count(e) > 1})
    if duplicates:
        raise SopError(f"Each transition event must be listed once: {', '.join(duplicates)}")
    reachable, frontier = {sop.phases[0].phase}, [sop.phases[0].phase]
    while frontier:
        here = frontier.pop()
        for t in sop.transitions:
            if here in t.from_phases and t.to not in reachable:
                reachable.add(t.to)
                frontier.append(t.to)
    unreachable = [p.phase.value for p in sop.phases if p.phase not in reachable]
    if unreachable:
        raise SopError(f"Phases unreachable from {sop.phases[0].phase.value}: {', '.join(unreachable)}")
    named = {r.enforced_by for r in sop.always_strict} | {r.enforced_by for p in sop.phases for r in p.strict}
    unused = sorted(set(ENFORCEMENT_POINTS) - named)
    if unused:
        raise SopError(f"Enforcement points not named by any rule in the SOP: {', '.join(unused)}")


def _phase_spec(phase: Phase, tables: Mapping[str, Any], tools: frozenset[str]) -> PhaseSpec:
    where = f"phases.{phase.value}"
    if phase.value not in tables:
        raise SopError(f"Missing [{where}]")
    t = tables[phase.value]
    if not isinstance(t, dict):
        raise SopError(f"[{where}] must be a table")
    menu = tuple(_tool(name, tools, f"{where}.tools") for name in _list(t, "tools", where))
    conditions = t.get("conditions", {})
    if not isinstance(conditions, dict):
        raise SopError(f"{where}.conditions must be a table")
    for tool, condition in conditions.items():
        if tool not in menu:
            raise SopError(f"{where}.conditions names {tool!r}, which is not in the phase's tools")
        if condition not in CONDITIONS:
            raise SopError(f"{where}.conditions: unknown condition {condition!r} (known: {', '.join(CONDITIONS)})")
    return PhaseSpec(
        phase=phase,
        goal=_text(t, "goal", where),
        tools=menu,
        conditions=dict(conditions),
        strict=tuple(_rule(r, f"{where}.strict") for r in _list(t, "strict", where)),
        flexible=tuple(_str(f, f"{where}.flexible") for f in _list(t, "flexible", where)),
    )


def _transition(t: Any, i: int) -> Transition:
    where = f"transitions[{i}]"
    if not isinstance(t, dict):
        raise SopError(f"{where} must be a table")
    sources = tuple(_phase(p, f"{where}.from") for p in _list(t, "from", where))
    if not sources:
        raise SopError(f"{where}.from must list at least one phase")
    return Transition(_text(t, "event", where), sources, _phase(_text(t, "to", where), f"{where}.to"), _text(t, "trigger", where))


def _memory(m: Any, i: int) -> MemoryItem:
    where = f"memory[{i}]"
    if not isinstance(m, dict):
        raise SopError(f"{where} must be a table")
    return MemoryItem(*(_text(m, k, where) for k in ("what", "captured", "used", "cleared")))


def _rule(r: Any, where: str) -> Rule:
    if not isinstance(r, dict):
        raise SopError(f"{where} entries must be tables with rule and enforced_by")
    point = _text(r, "enforced_by", where)
    if point not in ENFORCEMENT_POINTS:
        raise SopError(f"{where}: unknown enforcement point {point!r}")
    return Rule(_text(r, "rule", where), point)


def _phase(name: Any, where: str) -> Phase:
    try:
        return Phase(name)
    except ValueError:
        raise SopError(f"{where}: unknown phase {name!r}") from None


def _tool(name: Any, tools: frozenset[str], where: str) -> str:
    if name not in tools:
        raise SopError(f"{where}: unknown tool {name!r}")
    return str(name)


def _table(container: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = container.get(key)
    if not isinstance(value, dict):
        raise SopError(f"Missing table [{key}]")
    return value


def _list(container: Mapping[str, Any], key: str, where: str) -> list[Any]:
    value = container.get(key)
    if not isinstance(value, list):
        raise SopError(f"{where}.{key} must be a list")
    return value


def _str(value: Any, where: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SopError(f"{where} entries must be non-empty strings")
    return value.strip()


def _text(container: Mapping[str, Any], key: str, where: str = "file") -> str:
    if key not in container:
        raise SopError(f"{where} is missing {key!r}")
    return _str(container[key], f"{where}.{key}")


# ---------------------------------------------------------------------- Markdown


def _render(sop: Sop) -> str:
    lines = [
        "# Claims support SOP",
        "",
        f"Generated from `sop.toml` (version `{sop.version}`); do not edit by hand.",
        "Print it with `python -m insurance_claims.agent.sop`.",
        "",
        "Workflow: " + " → ".join(p.phase.value for p in sop.phases) + ".",
        "",
        "## In every phase",
        "",
        f"Tools always available: {', '.join(f'`{t}`' for t in sop.always_available)}.",
        "",
        "| Strict (enforced by code or the guard) | Enforced by |",
        "| --- | --- |",
        *(f"| {r.rule} | {ENFORCEMENT_POINTS[r.enforced_by]} |" for r in sop.always_strict),
        "",
        "Left to the model:",
        "",
        *(f"* {f}" for f in sop.always_flexible),
    ]
    for p in sop.phases:
        menu = []
        for t in p.tools:
            menu.append(f"`{t}` (only when {p.conditions[t].replace('_', ' ')})" if t in p.conditions else f"`{t}`")
        exits = [t for t in sop.transitions if p.phase in t.from_phases]
        lines += [
            "",
            f"## {p.phase.value}",
            "",
            f"Goal: {p.goal}",
            "",
            f"Tools: {', '.join(menu)}.",
            "",
            "| Strict (enforced by code or the guard) | Enforced by |",
            "| --- | --- |",
            *(f"| {r.rule} | {ENFORCEMENT_POINTS[r.enforced_by]} |" for r in p.strict),
            "",
            "Left to the model:",
            "",
            *(f"* {f}" for f in p.flexible),
            "",
            "Leaves to: " + "; ".join(f"{t.to.value} when {t.trigger} (`{t.event}`)" for t in exits) + ".",
        ]
    lines += ["", "## Memory across phases", "", "| What | Captured | Used | Cleared |", "| --- | --- | --- | --- |"]
    lines += [f"| {m.what} | {m.captured} | {m.used} | {m.cleared} |" for m in sop.memory]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    from insurance_claims.agent.tools import SCHEMAS
    from insurance_claims.config import Settings

    args = sys.argv[1:] if argv is None else argv
    # The same file the server uses: SOP_PATH if set (the Docker image sets it), else the checkout's sop.toml.
    sop = load_sop(Path(args[0]) if args else Settings.from_env().sop_path, known_tools=SCHEMAS)
    sys.stdout.write(sop.render_markdown())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
