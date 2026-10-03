"""Read-only, party-scoped access to fixture data.

* :class:`PolicyholderDirectory` answers "which parties does this raw value
  match?" for one identity field at a time. It never says why a value failed.
* :class:`ClaimRepository` requires an authenticated ``party_id`` and filters
  before returning any row. A case owned by someone else is reported exactly
  like a case that does not exist.
* :class:`RepresentativeDirectory` exposes the relationship fixture, which is
  never proof of claim access.
* :class:`GuidelineRepository` resolves document guidance and follow-up rules.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable, Sequence
from typing import Literal

from pydantic import BaseModel, ConfigDict

from insurance_claims.claims.normalize import (
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
from insurance_claims.domain.models import (
    PII_FIELDS,
    POLICY_NUMBER,
    Claim,
    DocumentGuidelines,
    FollowupRule,
    Policyholder,
    Representative,
)

__all__ = [
    "AuthorizationError",
    "ClaimRepository",
    "GuidelineRepository",
    "LookupResult",
    "PolicyholderDirectory",
    "RepresentativeDirectory",
    "natural_join",
]

_SUPPORTED_FIELDS = frozenset((*PII_FIELDS, POLICY_NUMBER))
_ALTERNATIVE_DEFAULT_KEY = "default"
_PLACEHOLDER_RE = re.compile(r"\{([A-Za-z_][A-Za-z0-9_]{0,63})\}")
_WORD_RE = re.compile(r"[a-z0-9]+")
_ARTICLES = ("the ", "a ", "an ", "your ", "any ")


class AuthorizationError(PermissionError):
    """A party-scoped read was attempted without an authenticated party."""


def _require_party(party_id: str | None) -> str:
    if not isinstance(party_id, str) or not party_id.strip():
        raise AuthorizationError("an authenticated party_id is required")
    return party_id


# ---------------------------------------------------------------------------
# Policyholders
# ---------------------------------------------------------------------------


def _dob_key(raw: str) -> str | None:
    parsed = normalize_dob(raw)
    return parsed.isoformat() if parsed else None


def _name_key(raw: str) -> str | None:
    tokens = name_tokens(raw)
    return " ".join(tokens) if len(tokens) >= 2 else None


_NAME_PREFIXES = frozenset({"mr", "mrs", "ms", "miss", "mx", "dr", "prof"})
_NAME_SUFFIXES = frozenset({"jr", "sr", "ii", "iii", "iv"})


def _name_fallback_keys(raw: str) -> list[str]:
    """Lookup keys for a stated name with a leading title, a trailing suffix, or one middle word.

    Only used when the exact key matches nobody. The first and last remaining words
    always stay, so a single given name or surname can never match on its own.
    """
    words = (normalize_name(raw) or "").split()
    while words and words[0] in _NAME_PREFIXES:
        words = words[1:]
    while words and words[-1] in _NAME_SUFFIXES:
        words = words[:-1]
    if len(words) < 2:
        return []
    keys = [" ".join(sorted(words))]
    keys += [" ".join(sorted(words[:i] + words[i + 1 :])) for i in range(1, len(words) - 1)]
    return keys


_KEYERS: dict[str, Callable[[str], str | None]] = {
    "full_name": _name_key,
    "dob": _dob_key,
    "phone": normalize_phone,
    "email": normalize_email,
    "id_last4": normalize_last4,
    POLICY_NUMBER: normalize_policy_number,
}


def _holder_values(holder: Policyholder, field: str) -> list[str]:
    """Every raw value on record for one field (canonical plus aliases)."""
    if field == "full_name":
        return [holder.name, *holder.name_aliases]
    if field == "phone":
        return [holder.phone, *holder.phone_aliases]
    if field == "email":
        return [holder.email, *holder.email_aliases]
    if field == "dob":
        return [holder.dob.isoformat()]
    if field == "id_last4":
        return [holder.id_last4]
    return [holder.policy_number]


class PolicyholderDirectory:
    """Per-field indexes from normalized value to party IDs."""

    def __init__(self, policyholders: Iterable[Policyholder]) -> None:
        self._holders: dict[str, Policyholder] = {}
        self._index: dict[str, dict[str, set[str]]] = {field: {} for field in _SUPPORTED_FIELDS}
        for holder in policyholders:
            if holder.party_id in self._holders:
                continue  # first row wins, matching fixture quarantine order
            self._holders[holder.party_id] = holder
            for field in _SUPPORTED_FIELDS:
                for raw in _holder_values(holder, field):
                    key = _KEYERS[field](raw)
                    if key is not None:
                        self._index[field].setdefault(key, set()).add(holder.party_id)

    def parties_matching(self, field: str, value: str) -> frozenset[str]:
        """Party IDs whose value (or alias) for ``field`` equals the raw ``value`` after normalization."""
        if field not in _SUPPORTED_FIELDS:
            raise ValueError("unsupported identity field")
        if not isinstance(value, str):
            return frozenset()
        key = _KEYERS[field](value)
        if key is None:
            return frozenset()
        exact = self._index[field].get(key)
        if exact or field != "full_name":
            return frozenset(exact or ())
        return frozenset(party for k in _name_fallback_keys(value) for party in self._index[field].get(k, ()))

    def _holder(self, party_id: str) -> Policyholder:
        holder = self._holders.get(_require_party(party_id))
        if holder is None:
            raise AuthorizationError("unknown party")
        return holder

    def contact_email(self, party_id: str) -> str:
        """Canonical email on record (only for sending to a verified caller)."""
        return self._holder(party_id).email

    def first_name(self, party_id: str) -> str:
        """First word of the canonical name, for a friendly greeting after verification."""
        return self._holder(party_id).name.split()[0]


# ---------------------------------------------------------------------------
# Claims
# ---------------------------------------------------------------------------


class LookupResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    status: Literal["found", "not_found", "ambiguous"]
    claims: list[Claim]


def _newest_first(claims: Iterable[Claim]) -> list[Claim]:
    return sorted(claims, key=lambda claim: (claim.created_at, claim.case_id), reverse=True)


class ClaimRepository:
    """Claims indexed by case ID; every answer is filtered to one authenticated party."""

    def __init__(self, claims: Iterable[Claim]) -> None:
        self._by_case: dict[str, Claim] = {}
        for claim in claims:
            self._by_case.setdefault(claim.case_id, claim)

    def list_for_party(self, party_id: str) -> list[Claim]:
        """The party's claims, newest first."""
        owner = _require_party(party_id)
        return _newest_first(claim for claim in self._by_case.values() if claim.party_id == owner)

    def get_for_party(self, party_id: str, case_id: str) -> LookupResult:
        """One owned claim, or ``not_found`` (also when the case belongs to another party)."""
        owner = _require_party(party_id)
        normalized = normalize_case_id(case_id) if isinstance(case_id, str) else None
        claim = self._by_case.get(normalized) if normalized else None
        if claim is None or claim.party_id != owner:
            return LookupResult(status="not_found", claims=[])
        return LookupResult(status="found", claims=[claim])

    def all_claims_unscoped(self) -> tuple[Claim, ...]:
        """Every claim, unfiltered. ONLY for building leak-detection token sets; never for answers."""
        return tuple(self._by_case.values())


# ---------------------------------------------------------------------------
# Representatives
# ---------------------------------------------------------------------------


class RepresentativeDirectory:
    """Known representative relationships (a relationship is not an authorization)."""

    def __init__(self, reps: Iterable[Representative]) -> None:
        self._reps: tuple[Representative, ...] = tuple(reps)

    def find(self, rep_name: str | None, represented_name: str | None) -> Representative | None:
        """First relationship matching every name given; ``None`` when no name is given."""
        if not rep_name and not represented_name:
            return None
        for rep in self._reps:
            if rep_name and not names_match(rep_name, rep.rep_name):
                continue
            if represented_name and not names_match(represented_name, rep.buyer_name):
                continue
            return rep
        return None


# ---------------------------------------------------------------------------
# Guidelines
# ---------------------------------------------------------------------------


def _words(text: str) -> frozenset[str]:
    return frozenset(_WORD_RE.findall(text.lower()))


def _collapse(text: str) -> str:
    return " ".join(text.lower().split())


def _with_article(item: str) -> str:
    cleaned = " ".join(item.split())
    return cleaned if cleaned.lower().startswith(_ARTICLES) else f"the {cleaned}"


def natural_join(items: Sequence[str]) -> str:
    """``the pathology report and the office note`` style join; empty -> ``the requested documents``."""
    phrases = [_with_article(item) for item in items if isinstance(item, str) and item.strip()]
    if not phrases:
        return "the requested documents"
    if len(phrases) == 1:
        return phrases[0]
    if len(phrases) == 2:
        return f"{phrases[0]} and {phrases[1]}"
    return f"{', '.join(phrases[:-1])}, and {phrases[-1]}"


class GuidelineRepository:
    """Document guidance, follow-up rules, and settings from the guideline fixture."""

    def __init__(self, guidelines: DocumentGuidelines) -> None:
        self._g = guidelines
        names = set(guidelines.document_guidance) | set(guidelines.document_alternative_guidance)
        names.discard(_ALTERNATIVE_DEFAULT_KEY)
        self._document_names: tuple[str, ...] = tuple(sorted(names))

    # --- documents -------------------------------------------------------
    def resolve_document_name(self, requested: str) -> str | None:
        """Exact (case-insensitive) guideline key, else the unique key whose words cover the request."""
        if not isinstance(requested, str) or not requested.strip():
            return None
        wanted = _collapse(requested)
        for name in self._document_names:
            if _collapse(name) == wanted:
                return name
        wanted_words = _words(requested)
        if not wanted_words:
            return None
        candidates = [name for name in self._document_names if wanted_words <= _words(name)]
        return candidates[0] if len(candidates) == 1 else None

    def document_guidance(self, requested: str) -> dict | None:
        """``{"document", "guideline_name", "guidance", "alternative"}`` or ``None`` when unresolved."""
        name = self.resolve_document_name(requested)
        if name is None:
            return None
        guidance = self._g.document_guidance.get(name)
        alternative = self._g.document_alternative_guidance.get(name)
        return {
            "document": requested.strip(),
            "guideline_name": name,
            "guidance": guidance.en if guidance else self.default_guidance(),
            "alternative": alternative.en if alternative else self.default_alternative(),
        }

    def case_type_guidance(self, case_type: str) -> str | None:
        if not isinstance(case_type, str):
            return None
        entry = self._g.case_type_guidance.get(case_type.strip().lower())
        return entry.en if entry else None

    def default_guidance(self) -> str:
        return self._g.default_guidance.en

    def default_alternative(self) -> str | None:
        entry = self._g.document_alternative_guidance.get(_ALTERNATIVE_DEFAULT_KEY)
        return entry.en if entry else None

    def setting(self, key: str) -> str | None:
        entry = self._g.claim_followup_settings.get(key)
        return entry.en if entry else None

    # --- follow-ups -------------------------------------------------------
    def render_followup(self, rule: FollowupRule, claim: Claim) -> str:
        """Fill ``{case_id}``, ``{documents}`` and setting placeholders; unknown placeholders are dropped.

        Uses plain substitution (never ``str.format``) so fixture text cannot reach attributes.
        """
        values = {
            "case_id": claim.case_id,
            "documents": natural_join(claim.documents_needed),
        }

        def fill(match: re.Match[str]) -> str:
            key = match.group(1)
            if key in values:
                return values[key]
            return self.setting(key) or ""

        rendered = _PLACEHOLDER_RE.sub(fill, rule.en)
        return " ".join(rendered.split())

    def fallback_followup(self) -> str:
        return self._g.claim_followup_fallback.en
