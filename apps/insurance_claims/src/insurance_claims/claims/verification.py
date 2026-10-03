"""Deterministic identity verification (the VERIFY_ID gate).

A caller is verified only when at least ``REQUIRED_MATCHES`` distinct PII
fields match exactly one policyholder and no provided field (including a
policy number) conflicts with that policyholder. The model can only propose
values; this module decides. The outcome never says which field failed or
whether a named policyholder exists: it exposes counts of *provided* fields.

Only field names and the party IDs each latest value matched are kept in
state; raw identity values are never stored or logged.

Failure counting: a failing snapshot is counted once. The snapshot signature
covers ``field_matches`` *and* a hash of any value proposed this turn that
matched nobody. Without the second part every wrong guess for a field (all
matching nobody, ``[]``) would produce the same snapshot, so guessing the last
four digits value by value would never count toward the lockout.

Stale values: a value that matched nobody normally keeps blocking (it conflicts
with every party). Once it has been charged as a failed attempt, a turn that adds
new identity values may drop it, but only while at least ``REQUIRED_MATCHES``
fields remain. That keeps every later non-verifying turn a counted failure, so
dropping a stale value never turns a guess into a free, uncounted probe.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime
from typing import Literal

from insurance_claims.claims.normalize import (
    name_tokens,
    normalize_dob,
    normalize_email,
    normalize_last4,
    normalize_phone,
    normalize_policy_number,
)
from insurance_claims.claims.repository import PolicyholderDirectory
from insurance_claims.domain.models import (
    ALL_IDENTITY_KEYS,
    PII_FIELDS,
    POLICY_NUMBER,
    REQUIRED_MATCHES,
)
from insurance_claims.domain.state import VerificationState

__all__ = ["OutcomeStatus", "VerificationOutcome", "apply_identity_proposals"]

OutcomeStatus = Literal[
    "verified",
    "already_verified",
    "pending",
    "mismatch",
    "ambiguous",
    "locked",
    "caller_changed",
    "representative_blocked",
]


def _canonical_name(raw: str) -> str | None:
    tokens = name_tokens(raw)
    return " ".join(tokens) if len(tokens) >= 2 else None


def _canonical_dob(raw: str) -> str | None:
    parsed = normalize_dob(raw)
    return parsed.isoformat() if parsed else None


_CANONICAL: dict[str, Callable[[str], str | None]] = {
    "full_name": _canonical_name,
    "dob": _canonical_dob,
    "phone": normalize_phone,
    "email": normalize_email,
    "id_last4": normalize_last4,
    POLICY_NUMBER: normalize_policy_number,
}


@dataclass(frozen=True)
class VerificationOutcome:
    status: OutcomeStatus
    party_id: str | None
    newly_provided: tuple[str, ...]
    provided_pii_count: int
    attempt_counted: bool
    reason: str


@dataclass(frozen=True)
class _Turn:
    """This turn's accepted proposals: raw values for lookup, canonical values for signatures."""

    raw: dict[str, str]
    canonical: dict[str, str]

    @property
    def fields(self) -> tuple[str, ...]:
        return tuple(self.raw)


def _accept(proposals: Mapping[str, str]) -> _Turn:
    """Restrict proposals to identity keys whose values normalize (canonical key order)."""
    raw: dict[str, str] = {}
    canonical: dict[str, str] = {}
    if not isinstance(proposals, Mapping):
        return _Turn(raw, canonical)
    for field in ALL_IDENTITY_KEYS:
        value = proposals.get(field)
        if not isinstance(value, str) or not value.strip():
            continue
        normalized = _CANONICAL[field](value)
        if normalized is not None:
            raw[field] = value
            canonical[field] = normalized
    return _Turn(raw, canonical)


def _provided_pii_count(field_matches: Mapping[str, list[str]]) -> int:
    return sum(1 for field in PII_FIELDS if field in field_matches)


def _qualifying_parties(field_matches: Mapping[str, list[str]]) -> list[str]:
    """Parties with >= REQUIRED_MATCHES matched PII fields and no conflicting field."""
    candidates = sorted({party for parties in field_matches.values() for party in parties})
    qualifying: list[str] = []
    for party in candidates:
        matched = sum(1 for field in PII_FIELDS if party in field_matches.get(field, ()))
        conflicts = any(party not in parties for parties in field_matches.values())
        if matched >= REQUIRED_MATCHES and not conflicts:
            qualifying.append(party)
    return qualifying


def _snapshot_signature(field_matches: Mapping[str, list[str]], unmatched_values: Mapping[str, str]) -> str:
    payload = json.dumps(
        {
            "matches": sorted((field, sorted(parties)) for field, parties in field_matches.items()),
            "unmatched": sorted(unmatched_values.items()),
        },
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _outcome(
    status: OutcomeStatus,
    state: VerificationState,
    turn: _Turn,
    reason: str,
    *,
    party_id: str | None = None,
    attempt_counted: bool = False,
) -> VerificationOutcome:
    return VerificationOutcome(
        status=status,
        party_id=party_id,
        newly_provided=turn.fields,
        provided_pii_count=_provided_pii_count(state.field_matches),
        attempt_counted=attempt_counted,
        reason=reason,
    )


def _handle_verified(
    base: VerificationState, turn: _Turn, turn_matches: dict[str, list[str]]
) -> tuple[VerificationState, VerificationOutcome]:
    """A verified session keeps access only while every new value still matches the same party."""
    party_id = base.party_id
    if any(party_id not in parties for parties in turn_matches.values()):
        fresh = VerificationState(
            field_matches=dict(turn_matches),
            failed_attempts=base.failed_attempts,
            # The new snapshot has not been charged yet (see ``_merge_matches``).
            last_failed_signature=None,
            caller_role=base.caller_role,
            representative_declared=base.representative_declared,
        )
        return fresh, _outcome("caller_changed", fresh, turn, "new_identity_differs")
    updated = base.model_copy(update={"field_matches": {**base.field_matches, **turn_matches}}, deep=True)
    return updated, _outcome("already_verified", updated, turn, "consistent_with_verified", party_id=party_id)


def _handle_failure(
    state: VerificationState,
    turn: _Turn,
    turn_matches: dict[str, list[str]],
    *,
    snapshot_changed: bool,
    max_failures: int,
) -> tuple[VerificationState, VerificationOutcome]:
    """Count a failing snapshot once; lock when the failure budget is spent."""
    unmatched = {field: turn.canonical[field] for field, parties in turn_matches.items() if not parties}
    signature = _snapshot_signature(state.field_matches, unmatched)
    counted = (snapshot_changed or bool(unmatched)) and signature != state.last_failed_signature
    failed = state.failed_attempts + 1 if counted else state.failed_attempts
    last_signature = signature if counted else state.last_failed_signature
    if failed >= max_failures:
        locked = state.model_copy(
            update={
                "status": "locked",
                "party_id": None,
                "verified_at": None,
                "failed_attempts": failed,
                "last_failed_signature": last_signature,
            }
        )
        return locked, _outcome("locked", locked, turn, "max_failures_reached", attempt_counted=counted)
    updated = state.model_copy(update={"failed_attempts": failed, "last_failed_signature": last_signature})
    reason = "failed_attempt_counted" if counted else "failed_attempt_repeated"
    return updated, _outcome("mismatch", updated, turn, reason, attempt_counted=counted)


def _merge_matches(base: VerificationState, turn_matches: dict[str, list[str]]) -> dict[str, list[str]]:
    """Latest value per field wins; charged values that matched nobody may give way to new ones."""
    merged = {**base.field_matches, **turn_matches}
    charged = base.failed_attempts > 0 and base.last_failed_signature is not None
    if not (charged and turn_matches):
        return merged
    pruned = {**{field: parties for field, parties in base.field_matches.items() if parties}, **turn_matches}
    # Guard: never drop below REQUIRED_MATCHES, so a miss stays a counted failure, never ``pending``.
    return pruned if _provided_pii_count(pruned) >= REQUIRED_MATCHES else merged


def apply_identity_proposals(
    vstate: VerificationState,
    proposals: Mapping[str, str],
    directory: PolicyholderDirectory,
    *,
    max_failures: int,
    now: datetime,
) -> tuple[VerificationState, VerificationOutcome]:
    """Fold this turn's proposed identity values into a new verification state.

    Never mutates ``vstate``. Keys outside the identity vocabulary (for example a
    model-proposed ``verified`` or ``party_id``) and values that do not normalize
    are ignored.
    """
    base = vstate.model_copy(deep=True)
    turn = _accept(proposals)
    if base.status == "locked":
        return base, _outcome("locked", base, _Turn({}, {}), "locked")

    turn_matches = {field: sorted(directory.parties_matching(field, value)) for field, value in turn.raw.items()}
    if base.status == "verified" and base.party_id:
        return _handle_verified(base, turn, turn_matches)

    merged = _merge_matches(base, turn_matches)
    state = base.model_copy(update={"status": "unverified", "party_id": None, "verified_at": None, "field_matches": merged})
    qualifying = _qualifying_parties(merged)

    if state.representative_declared:
        return state, _outcome("representative_blocked", state, turn, "representative_declared")
    if len(qualifying) == 1:
        verified = state.model_copy(update={"status": "verified", "party_id": qualifying[0], "verified_at": now})
        return verified, _outcome("verified", verified, turn, "unique_match", party_id=qualifying[0])
    if len(qualifying) > 1:
        return state, _outcome("ambiguous", state, turn, "multiple_parties_match")
    if _provided_pii_count(merged) >= REQUIRED_MATCHES:
        return _handle_failure(
            state,
            turn,
            turn_matches,
            snapshot_changed=merged != base.field_matches,
            max_failures=max_failures,
        )
    return state, _outcome("pending", state, turn, "insufficient_fields")
