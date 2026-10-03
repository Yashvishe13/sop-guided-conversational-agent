"""Load and validate the fixture files at startup.

Whole-file problems (missing file, invalid JSON, wrong top-level type, an
unusable guideline or schema document) raise :class:`FixtureError` so the
server fails fast with a clear message. A single malformed row in a list file
is quarantined instead: it is skipped and recorded as a :class:`FixtureIssue`
holding only the file name, row index, and an error code (never raw values).

Fixture text is data, never instructions.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

from insurance_claims.domain.models import (
    Claim,
    ClaimSchemaDoc,
    DocumentGuidelines,
    FixtureIssue,
    FollowupRule,
    Policyholder,
    Representative,
)

__all__ = [
    "CLAIMS_FILE",
    "GUIDELINES_FILE",
    "POLICYHOLDERS_FILE",
    "REPRESENTATIVES_FILE",
    "SCHEMA_FILE",
    "FixtureBundle",
    "FixtureError",
    "load_fixtures",
]

logger = logging.getLogger(__name__)

POLICYHOLDERS_FILE = "policyholders.json"
CLAIMS_FILE = "claims.json"
REPRESENTATIVES_FILE = "representatives.json"
GUIDELINES_FILE = "required_document_guideline.json"
SCHEMA_FILE = "claim_schema.json"

MAX_FIXTURE_BYTES = 5 * 1024 * 1024
"""A fixture file larger than this is rejected (protects startup from a runaway file)."""

_ModelT = TypeVar("_ModelT", bound=BaseModel)


class FixtureError(RuntimeError):
    """A fixture file is missing or unusable; the server must not start on it."""


@dataclass(frozen=True)
class FixtureBundle:
    policyholders: tuple[Policyholder, ...]
    claims: tuple[Claim, ...]
    representatives: tuple[Representative, ...]
    guidelines: DocumentGuidelines
    claim_schema: ClaimSchemaDoc
    issues: tuple[FixtureIssue, ...]


# ---------------------------------------------------------------------------
# File reading
# ---------------------------------------------------------------------------


def _read_json(path: Path) -> Any:
    """Parse one JSON file; every failure becomes a ``FixtureError`` naming the file only."""
    name = path.name
    if not path.is_file():
        raise FixtureError(f"Fixture file not found: {name}")
    try:
        size = path.stat().st_size
        if size > MAX_FIXTURE_BYTES:
            raise FixtureError(f"Fixture file too large: {name}")
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise FixtureError(f"Fixture file is not valid UTF-8: {name}") from exc
    except OSError as exc:
        raise FixtureError(f"Fixture file cannot be read: {name}") from exc
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        # Position only: the parser message never echoes file content.
        raise FixtureError(f"Fixture file is not valid JSON: {name} (line {exc.lineno}, column {exc.colno})") from exc


def _read_list(directory: Path, name: str) -> list[Any]:
    data = _read_json(directory / name)
    if not isinstance(data, list):
        raise FixtureError(f"Fixture file must contain a JSON array: {name}")
    return data


def _read_object(directory: Path, name: str) -> dict[str, Any]:
    data = _read_json(directory / name)
    if not isinstance(data, dict):
        raise FixtureError(f"Fixture file must contain a JSON object: {name}")
    return data


# ---------------------------------------------------------------------------
# Row validation and quarantine
# ---------------------------------------------------------------------------


def _error_code(exc: ValidationError) -> str:
    """``invalid:<field>:<error type>`` for the first error (field names and types only, no values)."""
    first = exc.errors(include_url=False, include_input=False, include_context=False)[0]
    location = ".".join(str(part) for part in first.get("loc", ())) or "row"
    return f"invalid:{location}:{first.get('type', 'error')}"


class _Quarantine:
    """Collects quarantined rows as PII-free ``FixtureIssue`` records."""

    def __init__(self) -> None:
        self.issues: list[FixtureIssue] = []

    def add(self, file: str, index: int, error: str) -> None:
        self.issues.append(FixtureIssue(file=file, index=index, error=error))

    def validate_rows(self, file: str, rows: list[Any], model: type[_ModelT]) -> list[tuple[int, _ModelT]]:
        valid: list[tuple[int, _ModelT]] = []
        for index, row in enumerate(rows):
            if not isinstance(row, dict):
                self.add(file, index, "not_an_object")
                continue
            try:
                valid.append((index, model.model_validate(row)))
            except ValidationError as exc:
                self.add(file, index, _error_code(exc))
        return valid


def _load_policyholders(directory: Path, quarantine: _Quarantine) -> tuple[Policyholder, ...]:
    rows = _read_list(directory, POLICYHOLDERS_FILE)
    kept: list[Policyholder] = []
    seen: set[str] = set()
    for index, holder in quarantine.validate_rows(POLICYHOLDERS_FILE, rows, Policyholder):
        if holder.party_id in seen:
            quarantine.add(POLICYHOLDERS_FILE, index, "duplicate_party_id")
            continue
        seen.add(holder.party_id)
        kept.append(holder)
    return tuple(kept)


def _load_claims(directory: Path, party_ids: set[str], quarantine: _Quarantine) -> tuple[Claim, ...]:
    rows = _read_list(directory, CLAIMS_FILE)
    kept: list[Claim] = []
    seen: set[str] = set()
    for index, claim in quarantine.validate_rows(CLAIMS_FILE, rows, Claim):
        if claim.case_id in seen:
            quarantine.add(CLAIMS_FILE, index, "duplicate_case_id")
        elif claim.party_id not in party_ids:
            quarantine.add(CLAIMS_FILE, index, "unknown_party_id")
        else:
            seen.add(claim.case_id)
            kept.append(claim)
    return tuple(kept)


def _load_representatives(directory: Path, party_ids: set[str], quarantine: _Quarantine) -> tuple[Representative, ...]:
    rows = _read_list(directory, REPRESENTATIVES_FILE)
    kept: list[Representative] = []
    for index, rep in quarantine.validate_rows(REPRESENTATIVES_FILE, rows, Representative):
        if rep.buyer_party_id not in party_ids:
            quarantine.add(REPRESENTATIVES_FILE, index, "unknown_buyer_party_id")
            continue
        kept.append(rep)
    return tuple(kept)


def _load_guidelines(directory: Path, quarantine: _Quarantine) -> DocumentGuidelines:
    data = _read_object(directory, GUIDELINES_FILE)
    rules = data.get("claim_followup_guidance")
    if isinstance(rules, list):
        valid = quarantine.validate_rows(GUIDELINES_FILE, rules, FollowupRule)
        data = {**data, "claim_followup_guidance": [rule.model_dump() for _, rule in valid]}
    try:
        return DocumentGuidelines.model_validate(data)
    except ValidationError as exc:
        raise FixtureError(f"Fixture file failed validation: {GUIDELINES_FILE} ({_error_code(exc)})") from exc


def _load_schema(directory: Path) -> ClaimSchemaDoc:
    data = _read_object(directory, SCHEMA_FILE)
    try:
        return ClaimSchemaDoc.model_validate(data)
    except ValidationError as exc:
        raise FixtureError(f"Fixture file failed validation: {SCHEMA_FILE} ({_error_code(exc)})") from exc


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_fixtures(fixtures_dir: Path) -> FixtureBundle:
    """Load every required fixture file, quarantining malformed rows."""
    directory = Path(fixtures_dir)
    if not directory.is_dir():
        raise FixtureError("Fixture directory not found")
    quarantine = _Quarantine()
    policyholders = _load_policyholders(directory, quarantine)
    party_ids = {holder.party_id for holder in policyholders}
    claims = _load_claims(directory, party_ids, quarantine)
    representatives = _load_representatives(directory, party_ids, quarantine)
    guidelines = _load_guidelines(directory, quarantine)
    claim_schema = _load_schema(directory)
    if quarantine.issues:
        logger.warning("Quarantined %d fixture row(s)", len(quarantine.issues))
    return FixtureBundle(
        policyholders=policyholders,
        claims=claims,
        representatives=representatives,
        guidelines=guidelines,
        claim_schema=claim_schema,
        issues=tuple(quarantine.issues),
    )
