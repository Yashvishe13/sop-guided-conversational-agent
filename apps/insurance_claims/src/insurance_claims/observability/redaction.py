"""PII and secret redaction applied to trace trees before they are serialized.

A :class:`Redactor` holds two kinds of sensitive material:

* **Known values** (fixture names, aliases, emails, phones, dates of birth,
  ID digits, policy numbers, claim ``summary``/``denial_reason`` text, plus
  anything added at runtime with :meth:`Redactor.add_values`). Each one is
  replaced by ``[PII:h:<10 hex>]``, a salted fingerprint, so a trace can
  still show that the same value appeared twice without revealing it.
* **Secrets** (API keys, SMTP passwords, encryption keys). They are replaced
  by ``[REDACTED]`` and never fingerprinted.

On top of that, generic backstop patterns catch values nobody registered:
email addresses, ``sk-`` style keys, bearer tokens, ``api_key=...`` style
credentials, phone-number shaped digit runs, 9-digit SSN shapes, and
contextual phrases ("my name is ...", "DOB is ...", "SSN last four is ...").

:meth:`Redactor.redact` walks dicts, lists, tuples, sets, Pydantic models and
dataclasses. Dict keys are kept, but values under identity keys (``full_name``,
``dob``, ``phone``, ``email``, ``id_last4`` ...) and free-text keys
(``utterance``, ``text``, ``content``, ``body_text`` ...) are fingerprinted
whole, and values under credential keys (``api_key``, ``password`` ...) become
``[REDACTED]``. Case IDs such as ``CL-2048``, phases, field names, party IDs,
amounts, and counts are not PII and survive.

Performance: known values are matched in one combined pass that only runs
when the text contains one of their trigger tokens (a name token, the last
digits of a phone or ID, a birth year, ...). Generic patterns run on every
string. Standard library only; this module never logs the values it handles.
"""

from __future__ import annotations

import dataclasses
import hashlib
import re
import secrets as _secrets
import threading
import unicodedata
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from datetime import date, datetime
from enum import Enum
from typing import Any, Literal

__all__ = [
    "PII_PREFIX",
    "SECRET_MARK",
    "Redactor",
    "ValueKind",
    "build_fixture_redactor",
]

SECRET_MARK = "[REDACTED]"
PII_PREFIX = "[PII:"

ValueKind = Literal["text", "name", "email", "phone", "dob", "id_digits", "policy_number"]
_VALUE_KINDS: frozenset[str] = frozenset({"text", "name", "email", "phone", "dob", "id_digits", "policy_number"})

_MIN_VALUE_LEN = 3
_MIN_SECRET_LEN = 4
_MIN_NAME_TOKEN_LEN = 3
_MAX_DEPTH = 64
_LARGE_INT = 10**8
"""Integers with nine or more digits are checked as if they were strings (phones, SSNs)."""

# Order inside the known-value pattern: a lower number is tried first at a given position.
_PRIORITY: dict[str, int] = {
    "secret": 0,
    "text": 1,
    "email": 2,
    "phone": 3,
    "dob": 4,
    "policy_number": 5,
    "name": 6,
    "id_digits": 7,
    "name_token": 8,
}

# ---------------------------------------------------------------------------
# Pattern building blocks
# ---------------------------------------------------------------------------

_LETTER_BEFORE = r"(?<![^\W\d_])"
_LETTER_AFTER = r"(?![^\W\d_])"
_ORDINAL = r"(?:st|nd|rd|th)?"
_MONTH_NAMES: tuple[tuple[str, str], ...] = (
    ("january", "jan"),
    ("february", "feb"),
    ("march", "mar"),
    ("april", "apr"),
    ("may", "may"),
    ("june", "jun"),
    ("july", "jul"),
    ("august", "aug"),
    ("september", "sept?"),
    ("october", "oct"),
    ("november", "nov"),
    ("december", "dec"),
)
_ANY_MONTH = "|".join(f"{full}|{abbr}" for full, abbr in _MONTH_NAMES)
_GENERIC_DATE = (
    r"(?:\d{4}[-/.]\d{1,2}[-/.]\d{1,2}"
    r"|\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4}"
    rf"|(?:{_ANY_MONTH})\.?\s+\d{{1,2}}{_ORDINAL},?\s+\d{{4}}"
    rf"|\d{{1,2}}{_ORDINAL}\s+(?:of\s+)?(?:{_ANY_MONTH})\.?,?\s+\d{{4}})"
)

_MARKER = r"\[(?:PII:h:[0-9a-f]{10}|REDACTED)\]"
_MARKER_FULL_RE = re.compile(rf"(?:{_MARKER}|\s)+", re.IGNORECASE)
_RUN_RE = re.compile(r"[^\W\d_]+|\d+")
"""Letter runs and digit runs: the tokens that trigger the known-value pass."""


@dataclass(frozen=True)
class _Rule:
    kind: str
    """``secret``, ``pii``, or ``marker``."""
    canonical: str | None
    """Fingerprint source for known values; ``None`` derives it from the matched text."""
    inner: str | None = None
    """For contextual rules: the named group holding the value (the rest of the match is kept)."""


# Generic backstops: (group name, kind, regex source, inner group or None).
_GENERIC_RULES: tuple[tuple[str, str, str, str | None], ...] = (
    (
        "ctx_secret",
        "secret",
        r"\b(?:api[_\-\s]?key|access[_\-]?token|auth[_\-]?token|secret|password|passwd|pwd)"
        r"\s*[=:]\s*[\"']?(?P<ctx_secret_v>[^\s\"'&,;]{4,})",
        "ctx_secret_v",
    ),
    ("bearer", "secret", r"\bbearer\s+[A-Za-z0-9\-._~+/]{12,}=*", None),
    ("sk_key", "secret", r"(?<![A-Za-z0-9])sk[-_][A-Za-z0-9_\-]{8,}", None),
    (
        "ctx_name",
        "pii",
        r"(?-i:\b(?:[Mm]y\s+name\s+is|[Nn]ame\s+is|[Tt]his\s+is|I\s+am|I'm|[Nn]ame:)\s+)"
        r"(?P<ctx_name_v>(?-i:[A-Z][a-z'\-]+(?:\s+[A-Z][a-z'\-]+){0,3}))",
        "ctx_name_v",
    ),
    (
        "ctx_dob",
        "pii",
        r"\b(?:dob|d\.o\.b\.?|date\s+of\s+birth|birth\s*date|birthday|born(?:\s+on)?)"
        rf"(?:[\s:=,'\"\-]|is\b|was\b|of\b){{0,12}}(?P<ctx_dob_v>{_GENERIC_DATE})(?!\d)",
        "ctx_dob_v",
    ),
    (
        "ctx_ssn",
        "pii",
        r"\b(?:ssn|social(?:\s+security)?(?:\s+number)?|last\s+(?:four|4)(?:\s+digits)?|national\s+id|id\s+number)"
        r"[^\d\n]{0,25}?(?P<ctx_ssn_v>\d{3}[\s\-]?\d{2}[\s\-]?\d{4}|\d{4})(?!\d)",
        "ctx_ssn_v",
    ),
    (
        "email",
        "pii",
        r"(?<![A-Za-z0-9._%+\-])[A-Za-z0-9._%+\-]{1,64}@(?:[A-Za-z0-9\-]{1,63}\.){1,8}[A-Za-z]{2,24}(?![A-Za-z0-9\-])",
        None,
    ),
    ("intl_phone", "pii", r"(?<![\w+])\+(?:\d[\s.\-()]{0,2}){7,14}\d(?!\d)", None),
    (
        "nanp_phone",
        "pii",
        r"(?<![\w+])(?:\+?1[\s.\-]{0,2})?\(?\d{3}\)?[\s.\-]{0,2}\d{3}[\s.\-]{0,2}\d{4}(?![\w])",
        None,
    ),
    ("ssn", "pii", r"(?<![\w+])\d{3}[\s\-]?\d{2}[\s\-]?\d{4}(?![\w])", None),
)


def _compile_generic() -> tuple[re.Pattern[str], dict[str, _Rule]]:
    groups: dict[str, _Rule] = {"marker": _Rule("marker", None)}
    parts = [f"(?P<marker>{_MARKER})"]
    for name, kind, source, inner in _GENERIC_RULES:
        groups[name] = _Rule(kind, None, inner)
        parts.append(f"(?P<{name}>{source})")
    return re.compile("|".join(parts), re.IGNORECASE), groups


_GENERIC_PATTERN, _GENERIC_GROUPS = _compile_generic()

# ---------------------------------------------------------------------------
# Key-based rules (structure-aware redaction in ``redact``)
# ---------------------------------------------------------------------------

_NAME_KEYS = frozenset(
    {
        "full_name", "first_name", "last_name", "rep_name", "buyer_name", "caller_name",
        "customer_name", "policyholder_name", "represented_name", "represented_person", "name_aliases",
    }
)  # fmt: skip
_IDENTITY_KEYS = frozenset(
    {
        "dob", "date_of_birth", "birth_date", "birthdate", "phone", "phone_number", "phone_aliases",
        "email", "email_address", "email_aliases", "requested_email", "to_address", "recipient",
        "recipient_address", "id_last4", "ssn", "ssn_last4", "last4", "national_id", "address",
        "street_address", "policy_number",
    }
)  # fmt: skip
_CONTENT_KEYS = frozenset(
    {
        "utterance", "user_text", "user_message", "message_text", "text", "content", "body",
        "body_text", "email_body", "summary_text", "reply", "reply_text", "draft", "draft_text",
        "output_text", "prompt", "instructions", "transcript", "denial_reason", "refusal",
    }
)  # fmt: skip
_SECRET_KEY_RE = re.compile(
    r"(?:^|[_\-])(?:api[_\-]?key|apikey|password|passwd|secret|client[_\-]?secret|authorization|"
    r"auth[_\-]?token|access[_\-]?token|refresh[_\-]?token|csrf[_\-]?token|session[_\-]?secret|"
    r"private[_\-]?key|encryption[_\-]?key|cookie|set[_\-]?cookie)$"
)
_PLACEHOLDER_RE = re.compile(r"\[(?:omitted|cycle|depth-limit)\]|<\d+ bytes>|<(?:unserializable|unprintable) \w+>")
"""Placeholders written by the tracer itself; they carry no caller data."""
_CODE_RE = re.compile(r"[a-z]+(?:_[a-z]+)*")
"""Lowercase snake_case values (``missing``, ``not_provided``) under identity keys are status codes, kept."""

_EMAIL_VALUE_RE = re.compile(r"[^@\s]+@[^@\s]+\.[^@\s]+")
_ISO_DATE_RE = re.compile(r"\d{4}-\d{2}-\d{2}")
_PHONEISH_RE = re.compile(r"[\d\s().+\-]+")
_POLICY_VALUE_RE = re.compile(r"([A-Za-z]{1,6})[\s\-_#]*(\d{3,12})")
_NAME_TOKEN_RE = re.compile(r"[^\W\d_]+")
_NAMEISH_RE = re.compile(r"[^\W\d_][^\W\d_'’.\-]*(?:\s+[^\W\d_][^\W\d_'’.\-]*){0,4}")
_DATE_FORMATS = (
    "%Y-%m-%d", "%Y/%m/%d", "%Y.%m.%d", "%m/%d/%Y", "%m-%d-%Y", "%m.%d.%Y",
    "%B %d, %Y", "%B %d %Y", "%b %d, %Y", "%b %d %Y", "%d %B %Y", "%d %b %Y", "%d %B, %Y", "%d %b, %Y",
)  # fmt: skip


@dataclass(frozen=True)
class _Entry:
    priority: int
    source: str
    rule: _Rule
    words: tuple[str, ...] = ()
    """Lowercase letter runs, one of which must be present for this entry to match."""
    digits: tuple[str, ...] = ()
    """Digit strings, one of which must end some digit run for this entry to match."""
    literal: str | None = None
    """Secrets: the lowercase literal checked by substring instead of runs."""


@dataclass
class _Compiled:
    pattern: re.Pattern[str] | None
    groups: dict[str, _Rule]
    words: frozenset[str] = frozenset()
    digits: dict[int, frozenset[str]] = field(default_factory=dict)
    literals: tuple[str, ...] = ()

    def triggered(self, text: str) -> bool:
        """Cheap pre-check: can any known value occur in ``text``?"""
        if self.pattern is None:
            return False
        lowered = text.lower()
        if any(lit in lowered for lit in self.literals):
            return True
        words, digits = self.words, self.digits
        for run in _RUN_RE.findall(lowered):
            if run[0].isdigit():
                if any(run[-size:] in options for size, options in digits.items() if len(run) >= size):
                    return True
            elif run in words:
                return True
        return False


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _fold_accents(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))


def _runs(text: str) -> list[str]:
    return _RUN_RE.findall(text.lower())


def _longest_run(text: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Trigger for a literal-ish value: its longest letter run or digit run."""
    runs = _runs(text)
    if not runs:
        return (), ()
    best = max(runs, key=len)
    return ((), (best,)) if best.isdigit() else ((best,), ())


def _parse_date(text: str) -> date | None:
    cleaned = re.sub(r"(\d)(?:st|nd|rd|th)\b", r"\1", text.strip(), flags=re.IGNORECASE)
    cleaned = re.sub(r"\bof\s+", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"(?<=[A-Za-z])\.", "", " ".join(cleaned.split()))
    if not re.search(r"\d{4}", cleaned):
        return None
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(cleaned, fmt).date()
        except ValueError:
            continue
    return None


def _canonical(text: str) -> str:
    """Stable fingerprint source so different spellings of one value share a fingerprint."""
    collapsed = " ".join(text.split()).casefold()
    parsed = _parse_date(collapsed)
    if parsed is not None:
        return parsed.isoformat()
    if _PHONEISH_RE.fullmatch(collapsed):
        digits = re.sub(r"\D", "", collapsed)
        if len(digits) == 11 and digits.startswith("1"):
            digits = digits[1:]
        return digits
    return collapsed


def _flex_text(value: str) -> str:
    """Literal text, case-insensitive, with any whitespace run matching any whitespace run."""
    body = r"\s+".join(re.escape(part) for part in value.split())
    if value[:1].isalnum():
        body = r"(?<!\w)" + body
    if value[-1:].isalnum():
        body += r"(?!\w)"
    return body


def _name_entries(value: str) -> list[_Entry]:
    """Full-name patterns (in order, and reversed for two tokens) plus letter-bounded tokens >= 3 chars."""
    canonical = " ".join(value.split()).casefold()
    entries: list[_Entry] = []
    for variant in {value, _fold_accents(value)}:
        tokens = _NAME_TOKEN_RE.findall(variant)
        if not tokens:
            continue
        lowered = [tok.lower() for tok in tokens]
        longest = (max(lowered, key=len),)
        joiner = r"[\s\-'’.,]+"
        orders = [tokens] + ([list(reversed(tokens))] if len(tokens) == 2 else [])
        for order in orders:
            body = joiner.join(re.escape(tok) for tok in order)
            source = f"{_LETTER_BEFORE}{body}{_LETTER_AFTER}"
            entries.append(_Entry(_PRIORITY["name"], source, _Rule("pii", canonical), words=longest))
        for tok, low in zip(tokens, lowered, strict=True):
            if len(tok) >= _MIN_NAME_TOKEN_LEN:
                source = f"{_LETTER_BEFORE}{re.escape(tok)}{_LETTER_AFTER}"
                entries.append(_Entry(_PRIORITY["name_token"], source, _Rule("pii", None), words=(low,)))
    return entries


def _phone_entry(value: str) -> _Entry | None:
    digits = re.sub(r"\D", "", value)
    if len(digits) == 11 and digits.startswith("1"):
        digits = digits[1:]
    if len(digits) == 10:
        area, mid, last = digits[:3], digits[3:6], digits[6:]
        sep = r"[\s.\-]{0,3}"
        source = rf"(?<!\d)(?:\+?1{sep})?\(?{area}\)?{sep}{mid}{sep}{last}(?!\d)"
        return _Entry(_PRIORITY["phone"], source, _Rule("pii", digits), digits=(last,))
    if 7 <= len(digits) <= 15:
        source = r"(?<!\d)\+?" + r"[\s.\-()]{0,3}".join(digits) + r"(?!\d)"
        return _Entry(_PRIORITY["phone"], source, _Rule("pii", digits), digits=(digits[-1],))
    return None


def _dob_entry(value: date) -> _Entry:
    year = str(value.year)
    month = f"0?{value.month}" if value.month < 10 else str(value.month)
    day = f"0?{value.day}" if value.day < 10 else str(value.day)
    full, abbr = _MONTH_NAMES[value.month - 1]
    names = f"(?:{full}|{abbr})\\.?"
    forms = (
        rf"{year}[-/.]{month}[-/.]{day}",
        rf"{month}[-/.]{day}[-/.]{year}",
        rf"{day}[-/.]{month}[-/.]{year}",
        rf"{names}\s+{day}{_ORDINAL},?\s+{year}",
        rf"{day}{_ORDINAL}\s+(?:of\s+)?{names},?\s+{year}",
    )
    source = r"(?<!\d)(?:" + "|".join(forms) + r")(?!\d)"
    return _Entry(_PRIORITY["dob"], source, _Rule("pii", value.isoformat()), digits=(year,))


def _id_entries(digits: str) -> list[_Entry]:
    tail = digits[-4:]
    entries = [_Entry(_PRIORITY["id_digits"], rf"(?<!\d){digits}(?!\d)", _Rule("pii", digits), digits=(tail,))]
    if len(digits) == 9:
        grouped = rf"(?<!\d){digits[:3]}[\s\-]?{digits[3:5]}[\s\-]?{digits[5:]}(?!\d)"
        entries.append(_Entry(_PRIORITY["id_digits"], grouped, _Rule("pii", digits), digits=(tail,)))
        entries.append(_Entry(_PRIORITY["id_digits"], rf"(?<!\d){tail}(?!\d)", _Rule("pii", tail), digits=(tail,)))
    return entries


def _infer_kind(value: str) -> str:
    if _EMAIL_VALUE_RE.fullmatch(value):
        return "email"
    if _ISO_DATE_RE.fullmatch(value) and _parse_date(value) is not None:
        return "dob"
    if _PHONEISH_RE.fullmatch(value):
        digits = re.sub(r"\D", "", value)
        if len(digits) in (4, 9) and "+" not in value:
            return "id_digits"
        if 7 <= len(digits) <= 15:
            return "phone"
        return "text"
    if _POLICY_VALUE_RE.fullmatch(value) and not value.isalpha():
        return "policy_number"
    if _NAMEISH_RE.fullmatch(value) and len(value) <= 80:
        return "name"
    return "text"


# ---------------------------------------------------------------------------
# Redactor
# ---------------------------------------------------------------------------


class Redactor:
    """Replace known sensitive values, secrets, and PII-shaped strings. Thread-safe."""

    def __init__(self, sensitive_values: Iterable[str] = (), *, secrets: Iterable[str] = (), salt: str = "") -> None:
        self._salt = salt
        self._lock = threading.RLock()
        self._entries: dict[str, _Entry] = {}
        self._compiled: _Compiled | None = None
        self.add_values(sensitive_values)
        self.add_secrets(secrets)

    # ------------------------------------------------------------- registry
    def add_values(self, values: Iterable[str], *, kind: ValueKind | None = None) -> None:
        """Register sensitive values (thread-safe). ``kind`` is inferred per value when omitted."""
        if values is None:
            return
        if isinstance(values, (str, bytes)):
            values = [values]  # type: ignore[list-item]
        if kind is not None and kind not in _VALUE_KINDS:
            raise ValueError(f"unknown value kind: {kind}")
        new: list[_Entry] = []
        for raw in values:
            new.extend(self._entries_for(raw, kind))
        self._register(new)

    def add_secrets(self, secrets: Iterable[str | None]) -> None:
        """Register secrets (API keys, passwords). They become ``[REDACTED]`` wherever they appear."""
        if secrets is None:
            return
        if isinstance(secrets, str):
            secrets = [secrets]
        new = []
        for raw in secrets:
            if not isinstance(raw, str) or len(raw.strip()) < _MIN_SECRET_LEN:
                continue
            value = raw.strip()
            new.append(_Entry(_PRIORITY["secret"], re.escape(value), _Rule("secret", None), literal=value.lower()))
        self._register(new)

    def _register(self, entries: list[_Entry]) -> None:
        if not entries:
            return
        with self._lock:
            changed = False
            for entry in entries:
                existing = self._entries.get(entry.source)
                if existing is None or entry.priority < existing.priority:
                    self._entries[entry.source] = entry
                    changed = True
            if changed:
                self._compiled = None

    def _entries_for(self, raw: object, kind: str | None) -> list[_Entry]:
        if raw is None or isinstance(raw, bool):
            return []
        if isinstance(raw, date) and not isinstance(raw, datetime):
            return [_dob_entry(raw)]
        value = " ".join(str(raw).split())
        if len(value) < _MIN_VALUE_LEN and not (kind == "id_digits" and value.isdigit()):
            return []
        resolved = kind or _infer_kind(value)
        if resolved == "dob":
            parsed = _parse_date(value)
            return [_dob_entry(parsed)] if parsed else self._text_entries(value)
        if resolved == "phone":
            entry = _phone_entry(value)
            return [entry] if entry else self._text_entries(value)
        if resolved == "id_digits":
            digits = re.sub(r"\D", "", value)
            return _id_entries(digits) if digits else self._text_entries(value)
        if resolved == "policy_number":
            return self._policy_entries(value)
        if resolved == "email":
            source = rf"(?<![A-Za-z0-9._%+\-]){re.escape(value)}(?![A-Za-z0-9\-])"
            email_words, email_digits = _longest_run(value)
            return [_Entry(_PRIORITY["email"], source, _Rule("pii", value.casefold()), email_words, email_digits)]
        if resolved == "name":
            return _name_entries(value)
        return self._text_entries(value)

    def _policy_entries(self, value: str) -> list[_Entry]:
        match = _POLICY_VALUE_RE.fullmatch(value)
        if match is None:
            return self._text_entries(value)
        prefix, digits = match.groups()
        source = rf"(?<![A-Za-z0-9]){re.escape(prefix)}[\s\-_#]*{digits}(?!\d)"
        rule = _Rule("pii", f"{prefix}-{digits}".casefold())
        return [_Entry(_PRIORITY["policy_number"], source, rule, digits=(digits[-4:],))]

    @staticmethod
    def _text_entries(value: str) -> list[_Entry]:
        words, digits = _longest_run(value)
        if not words and not digits:
            return [_Entry(_PRIORITY["text"], re.escape(value), _Rule("pii", value.casefold()), literal=value.lower())]
        return [_Entry(_PRIORITY["text"], _flex_text(value), _Rule("pii", value.casefold()), words, digits)]

    def _known(self) -> _Compiled:
        with self._lock:
            if self._compiled is None:
                self._compiled = self._build()
            return self._compiled

    def _build(self) -> _Compiled:
        groups: dict[str, _Rule] = {"marker": _Rule("marker", None)}
        if not self._entries:
            return _Compiled(None, groups)
        parts = [f"(?P<marker>{_MARKER})"]
        words: set[str] = set()
        digits: dict[int, set[str]] = {}
        literals: list[str] = []
        ordered = sorted(self._entries.values(), key=lambda e: (e.priority, -len(e.source), e.source))
        for index, entry in enumerate(ordered):
            name = f"v{index}"
            groups[name] = entry.rule
            parts.append(f"(?P<{name}>{entry.source})")
            words.update(entry.words)
            for digit in entry.digits:
                digits.setdefault(len(digit), set()).add(digit)
            if entry.literal:
                literals.append(entry.literal)
        return _Compiled(
            re.compile("|".join(parts), re.IGNORECASE),
            groups,
            frozenset(words),
            {size: frozenset(options) for size, options in digits.items()},
            tuple(literals),
        )

    # ------------------------------------------------------------ redaction
    def fingerprint(self, value: str) -> str:
        """``"h:"`` plus the first 10 hex chars of ``sha256(salt + value)``."""
        digest = hashlib.sha256((self._salt + str(value)).encode("utf-8")).hexdigest()
        return "h:" + digest[:10]

    def _mark(self, canonical: str) -> str:
        return f"{PII_PREFIX}{self.fingerprint(canonical)}]"

    def redact_text(self, text: str) -> str:
        """Return ``text`` with every known value, secret, and PII-shaped substring replaced."""
        if not isinstance(text, str):
            text = _safe_str(text)
        if not text:
            return text
        return self._redact_with(text, self._known())

    def _redact_with(self, text: str, known: _Compiled) -> str:
        if known.pattern is not None and known.triggered(text):
            text = known.pattern.sub(lambda m: self._replace(m, known.groups), text)
        return _GENERIC_PATTERN.sub(lambda m: self._replace(m, _GENERIC_GROUPS), text)

    def _replace(self, match: re.Match[str], groups: dict[str, _Rule]) -> str:
        rule = groups.get(match.lastgroup or "")
        if rule is None or rule.kind == "marker":
            return match.group(0)
        if rule.inner is not None:
            start, end = match.span(rule.inner)
            if start < 0:
                return match.group(0)
            whole, base = match.group(0), match.start()
            mark = SECRET_MARK if rule.kind == "secret" else self._mark(_canonical(match.group(rule.inner)))
            return whole[: start - base] + mark + whole[end - base :]
        if rule.kind == "secret":
            return SECRET_MARK
        return self._mark(rule.canonical or _canonical(match.group(0)))

    def redact(self, obj: Any) -> Any:
        """Recursively redact ``obj``. Dict keys are kept; the result is JSON-friendly."""
        return self._walk(obj, self._known(), 0, set())

    def _walk(self, obj: Any, known: _Compiled, depth: int, active: set[int]) -> Any:
        if obj is None or isinstance(obj, (bool, float)):
            return obj
        if isinstance(obj, int):
            return self._redact_int(obj, known)
        if isinstance(obj, str):
            return self._redact_with(obj, known) if obj else obj
        if depth >= _MAX_DEPTH:
            return "[depth-limit]"
        if id(obj) in active:
            return "[cycle]"
        active.add(id(obj))
        try:
            return self._walk_container(obj, known, depth, active)
        finally:
            active.discard(id(obj))

    def _walk_container(self, obj: Any, known: _Compiled, depth: int, active: set[int]) -> Any:
        if isinstance(obj, Mapping):
            out: dict[Any, Any] = {}
            for key, value in list(obj.items()):
                safe_key = key if isinstance(key, (str, int, float, bool)) or key is None else _safe_str(key)
                out[safe_key] = self._walk_field(safe_key, value, obj, known, depth, active)
            return out
        if isinstance(obj, (list, tuple)):
            items = [self._walk(item, known, depth + 1, active) for item in obj]
            return tuple(items) if isinstance(obj, tuple) else items
        if isinstance(obj, (set, frozenset)):
            return [self._walk(item, known, depth + 1, active) for item in sorted(obj, key=repr)]
        converted = _convert(obj)
        if converted is _UNCONVERTED:
            return self._redact_with(_safe_str(obj), known)
        return self._walk(converted, known, depth + 1, active)

    def _walk_field(self, key: Any, value: Any, parent: Mapping[Any, Any], known: _Compiled, depth: int, active: set[int]) -> Any:
        lowered = key.casefold() if isinstance(key, str) else ""
        if lowered and _SECRET_KEY_RE.search(lowered):
            return self._secret_value(value, known, depth, active)
        if lowered in _CONTENT_KEYS:
            return self._whole_value(value, known, depth, active, allow_codes=True, numbers=False)
        if lowered in _NAME_KEYS:
            return self._whole_value(value, known, depth, active, allow_codes=False)
        if lowered in _IDENTITY_KEYS:
            return self._whole_value(value, known, depth, active, allow_codes=True)
        if lowered == "value" and str(parent.get("field", "")).casefold() in _NAME_KEYS | _IDENTITY_KEYS:
            return self._whole_value(value, known, depth, active, allow_codes=True)
        return self._walk(value, known, depth + 1, active)

    def _secret_value(self, value: Any, known: _Compiled, depth: int, active: set[int]) -> Any:
        if value is None or isinstance(value, bool) or value == "":
            return value
        if isinstance(value, (str, int, float, bytes, bytearray)):
            return SECRET_MARK
        return self._walk(value, known, depth + 1, active)

    def _whole_value(self, value: Any, known: _Compiled, depth: int, active: set[int], *, allow_codes: bool, numbers: bool = True) -> Any:
        """Fingerprint a value whose key says it is PII or free text (numbers too, unless ``numbers`` is False)."""
        if value is None or isinstance(value, (bool, float)) or value == "":
            return value
        if isinstance(value, (date, datetime)):
            value = value.isoformat()
        if isinstance(value, int):
            if not numbers:
                return self._redact_int(value, known)
            value = str(value)
        if isinstance(value, str):
            if _MARKER_FULL_RE.fullmatch(value) or _PLACEHOLDER_RE.fullmatch(value):
                return value
            if allow_codes and _CODE_RE.fullmatch(value):
                return value
            return self._mark(_canonical(value))
        if isinstance(value, (list, tuple)) and all(isinstance(v, (str, int)) or v is None for v in value):
            items = [self._whole_value(v, known, depth, active, allow_codes=allow_codes, numbers=numbers) for v in value]
            return tuple(items) if isinstance(value, tuple) else items
        return self._walk(value, known, depth + 1, active)

    def _redact_int(self, value: int, known: _Compiled) -> int | str:
        if abs(value) < _LARGE_INT:
            return value
        text = str(value)
        cleaned = self._redact_with(text, known)
        return value if cleaned == text else cleaned


_UNCONVERTED = object()


def _convert(obj: Any) -> Any:
    """Turn models, dataclasses, enums, dates and bytes into plain data, or ``_UNCONVERTED``."""
    try:
        if isinstance(obj, Enum):
            return obj.value
        if isinstance(obj, (date, datetime)):
            return obj.isoformat()
        if isinstance(obj, (bytes, bytearray)):
            return f"<{len(obj)} bytes>"
        dump = getattr(obj, "model_dump", None)
        if callable(dump) and not isinstance(obj, type):
            return dump(mode="json")
        if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
            return {f.name: getattr(obj, f.name, None) for f in dataclasses.fields(obj)}
    except Exception:
        return _UNCONVERTED
    return _UNCONVERTED


def _safe_str(obj: Any) -> str:
    try:
        return str(obj)
    except Exception:
        return f"<unprintable {obj.__class__.__name__}>"


# ---------------------------------------------------------------------------
# Fixture redactor
# ---------------------------------------------------------------------------


def build_fixture_redactor(bundle: Any, *, secrets: Iterable[str | None], salt: str | None = None) -> Redactor:
    """Redactor seeded with every identity value and claim text in a fixture bundle.

    ``bundle`` is duck-typed: it needs ``policyholders`` and ``claims`` sequences
    (and optionally ``representatives``). ``secrets`` may contain ``None`` entries
    (unset settings), which are skipped. ``salt`` defaults to a random per-process
    value so short values such as ID digits cannot be recovered from fingerprints.
    """
    chosen_salt = _secrets.token_hex(16) if salt is None else salt
    redactor = Redactor(secrets=[s for s in (secrets or ()) if s], salt=chosen_salt)
    for holder in getattr(bundle, "policyholders", None) or ():
        _add_policyholder(redactor, holder)
    for claim in getattr(bundle, "claims", None) or ():
        texts = [getattr(claim, "summary", None), getattr(claim, "denial_reason", None)]
        redactor.add_values([t for t in texts if isinstance(t, str) and t.strip()], kind="text")
    for rep in getattr(bundle, "representatives", None) or ():
        names = [getattr(rep, "rep_name", None), getattr(rep, "buyer_name", None)]
        redactor.add_values([n for n in names if isinstance(n, str)], kind="name")
    return redactor


def _add_policyholder(redactor: Redactor, holder: Any) -> None:
    def values(*attrs: str) -> list[str]:
        out: list[str] = []
        for attr in attrs:
            raw = getattr(holder, attr, None)
            items = raw if isinstance(raw, (list, tuple)) else [raw]
            out.extend(str(item) for item in items if item not in (None, ""))
        return out

    redactor.add_values(values("name", "name_aliases"), kind="name")
    redactor.add_values(values("email", "email_aliases"), kind="email")
    redactor.add_values(values("phone", "phone_aliases"), kind="phone")
    redactor.add_values(values("id_last4"), kind="id_digits")
    redactor.add_values(values("policy_number"), kind="policy_number")
    dob = getattr(holder, "dob", None)
    if isinstance(dob, date):
        redactor.add_values([dob.isoformat()], kind="dob")
    elif isinstance(dob, str) and dob.strip():
        redactor.add_values([dob], kind="dob")
