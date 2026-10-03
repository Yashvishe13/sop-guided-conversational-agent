"""Pure normalizers for identity values, locators, and grounding cross-checks.

Every function here is deterministic and side-effect free. Values are never
logged: callers pass raw caller input and get back a canonical comparison form
(or ``None`` when the input is not a valid value of that kind).
"""

from __future__ import annotations

import re
import unicodedata
from datetime import date

__all__ = [
    "MONTHS",
    "identity_value_grounded",
    "name_tokens",
    "names_match",
    "normalize_case_id",
    "normalize_dob",
    "normalize_email",
    "normalize_last4",
    "normalize_name",
    "normalize_phone",
    "normalize_policy_number",
]

MAX_INPUT_CHARS = 4000
"""Inputs longer than this are rejected outright (bounded work on hostile input)."""

MIN_YEAR = 1900
MAX_YEAR = 2100

MONTHS: dict[str, int] = {
    "january": 1, "jan": 1,
    "february": 2, "feb": 2,
    "march": 3, "mar": 3,
    "april": 4, "apr": 4,
    "may": 5,
    "june": 6, "jun": 6,
    "july": 7, "jul": 7,
    "august": 8, "aug": 8,
    "september": 9, "sep": 9, "sept": 9,
    "october": 10, "oct": 10,
    "november": 11, "nov": 11,
    "december": 12, "dec": 12,
}  # fmt: skip

# Straight, curly, and modifier-letter apostrophes plus backtick.
_APOSTROPHES = frozenset({"'", "`", chr(0x2018), chr(0x2019), chr(0x02BC)})
_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
_ORDINAL = r"(?:st|nd|rd|th)?"
_ISO_RE = re.compile(r"^(\d{4})([-/])(\d{1,2})\2(\d{1,2})$")
_US_RE = re.compile(r"^(\d{1,2})([-/.])(\d{1,2})\2(\d{4})$")
_MDY_RE = re.compile(rf"^([a-z]+)\.?\s+(\d{{1,2}}){_ORDINAL},?\s+(\d{{4}})$")
_DMY_RE = re.compile(rf"^(\d{{1,2}}){_ORDINAL}\s+(?:of\s+)?([a-z]+)\.?,?\s+(\d{{4}})$")
_POLICY_RE = re.compile(r"^(?:pol(?:icy)?)?[\s\-_#.:]*(\d{3,8})$")
_CASE_RE = re.compile(r"^(?:cl|claim|case)[\s\-_#.:]*(\d{3,8})$")
_DIGIT_GROUP_RE = re.compile(r"\d+")
_RUN_SEPARATORS = frozenset(" \t-.()+")
"""Characters that may sit between digit groups of one spoken/typed number (plus any Unicode dash)."""
_PHONE_EXTRA_SEPARATORS = frozenset("/")
_MAX_SEPARATOR_GAP = 3
# Spanish month names (callers may answer in Spanish); used for parsing and grounding only.
MONTHS.update({"enero": 1, "febrero": 2, "marzo": 3, "abril": 4, "mayo": 5, "junio": 6, "julio": 7, "agosto": 8,
               "septiembre": 9, "setiembre": 9, "octubre": 10, "noviembre": 11, "diciembre": 12})  # fmt: skip
_NUMERIC_DATE_RE = re.compile(r"(?<!\d)(\d{1,4})\s*([-/.]|\s)\s*(\d{1,2})\s*\2\s*(\d{1,4})(?!\d)")
_WORD_RE = re.compile(r"[a-z]+")


def _usable(raw: object) -> str | None:
    """Return ``raw`` stripped when it is a non-empty, bounded string."""
    if not isinstance(raw, str):
        return None
    text = raw.strip()
    if not text or len(text) > MAX_INPUT_CHARS:
        return None
    return text


def _digits(text: str) -> str:
    return "".join(ch for ch in text if "0" <= ch <= "9")


# ---------------------------------------------------------------------------
# Names
# ---------------------------------------------------------------------------


def _name_form(text: str) -> str:
    """NFKD, drop accents, casefold, keep letters; dashes, apostrophes and whitespace become spaces."""
    kept: list[str] = []
    for ch in unicodedata.normalize("NFKD", text):
        if unicodedata.combining(ch):
            continue
        if ch.isalpha():
            kept.append(ch.casefold())
        elif ch.isspace() or ch in _APOSTROPHES or unicodedata.category(ch) == "Pd":
            kept.append(" ")
    return " ".join("".join(kept).split())


def normalize_name(raw: str) -> str | None:
    """Accent-free, lowercase, punctuation-free name form; ``None`` when nothing is left."""
    text = _usable(raw)
    if text is None:
        return None
    return _name_form(text) or None


def name_tokens(raw: str) -> tuple[str, ...]:
    """Sorted tokens of :func:`normalize_name` (empty when the name is unusable)."""
    normalized = normalize_name(raw)
    return tuple(sorted(normalized.split())) if normalized else ()


def names_match(provided: str, canonical: str) -> bool:
    """Order-insensitive token multiset equality; both names need at least two tokens."""
    left = name_tokens(provided)
    right = name_tokens(canonical)
    return len(left) >= 2 and len(right) >= 2 and left == right


# ---------------------------------------------------------------------------
# Contact details and ID digits
# ---------------------------------------------------------------------------


def normalize_phone(raw: str) -> str | None:
    """Exactly ten digits (a leading US country code ``1`` is dropped), else ``None``."""
    text = _usable(raw)
    if text is None:
        return None
    digits = _digits(text)
    if len(digits) == 11 and digits.startswith("1"):
        digits = digits[1:]
    return digits if len(digits) == 10 else None


def normalize_email(raw: str) -> str | None:
    """Stripped, lowercased address with a basic ``local@domain.tld`` shape, else ``None``."""
    text = _usable(raw)
    if text is None:
        return None
    lowered = text.lower()
    return lowered if _EMAIL_RE.fullmatch(lowered) else None


def normalize_last4(raw: str) -> str | None:
    """Four digits as given, or the last four of a nine-digit SSN; otherwise ``None``."""
    text = _usable(raw)
    if text is None:
        return None
    digits = _digits(text)
    if len(digits) == 4:
        return digits
    if len(digits) == 9:
        return digits[-4:]
    return None


# ---------------------------------------------------------------------------
# Dates
# ---------------------------------------------------------------------------


def _safe_date(year: int, month: int, day: int) -> date | None:
    if not MIN_YEAR <= year <= MAX_YEAR:
        return None
    try:
        return date(year, month, day)
    except ValueError:
        return None


def normalize_dob(raw: str) -> date | None:
    """Parse the accepted date spellings (US order for numeric dates); ``None`` otherwise."""
    text = _usable(raw)
    if text is None:
        return None
    text = " ".join(text.lower().split())
    if match := _ISO_RE.fullmatch(text):
        return _safe_date(int(match[1]), int(match[3]), int(match[4]))
    if match := _US_RE.fullmatch(text):
        return _safe_date(int(match[4]), int(match[1]), int(match[3]))
    if match := _MDY_RE.fullmatch(text):
        month = MONTHS.get(match[1])
        return _safe_date(int(match[3]), month, int(match[2])) if month else None
    if match := _DMY_RE.fullmatch(text):
        month = MONTHS.get(match[2])
        return _safe_date(int(match[3]), month, int(match[1])) if month else None
    return None


# ---------------------------------------------------------------------------
# Locators
# ---------------------------------------------------------------------------


def normalize_policy_number(raw: str) -> str | None:
    """``pol 9921`` / ``POL-9921`` / ``pol9921`` / bare ``9921`` -> ``POL-9921``."""
    text = _usable(raw)
    if text is None:
        return None
    match = _POLICY_RE.fullmatch(" ".join(text.lower().split()))
    return f"POL-{match[1]}" if match else None


def normalize_case_id(raw: str) -> str | None:
    """``cl 2048`` / ``CL2048`` / ``cl-2048`` / ``claim 2048`` -> ``CL-2048``; bare digits -> ``None``."""
    text = _usable(raw)
    if text is None:
        return None
    match = _CASE_RE.fullmatch(" ".join(text.lower().split()))
    return f"CL-{match[1]}" if match else None


# ---------------------------------------------------------------------------
# Grounding: was a model-extracted value actually stated in this utterance?
# ---------------------------------------------------------------------------


def _is_dash(ch: str) -> bool:
    return unicodedata.category(ch) == "Pd"


def _digit_runs(utterance: str, extra_separators: frozenset[str] = frozenset()) -> list[list[str]]:
    """Digit groups, merged into runs when separated only by a short gap of phone-style separators."""
    runs: list[list[str]] = []
    last_end: int | None = None
    for match in _DIGIT_GROUP_RE.finditer(utterance):
        gap = utterance[last_end : match.start()] if last_end is not None else ""
        joinable = 0 < len(gap) <= _MAX_SEPARATOR_GAP and all(ch in _RUN_SEPARATORS or ch in extra_separators or _is_dash(ch) for ch in gap)
        if joinable and runs:
            runs[-1].append(match.group())
        else:
            runs.append([match.group()])
        last_end = match.end()
    return runs


def _phone_grounded(phone: str, utterance: str) -> bool:
    for groups in _digit_runs(utterance, _PHONE_EXTRA_SEPARATORS):
        for end in range(len(groups)):
            tail = ""
            for start in range(end, -1, -1):
                tail = groups[start] + tail
                if len(tail) >= len(phone):
                    break
            if tail.endswith(phone):
                return True
    return False


def _last4_grounded(last4: str, utterance: str) -> bool:
    for groups in _digit_runs(utterance):
        if last4 in groups:
            return True
        joined = "".join(groups)
        # ``4 4 7 2`` / ``44-72`` (joined == last4) or the tail of a longer number such as a full SSN.
        if len(joined) >= 4 and joined.endswith(last4):
            return True
    return False


def _standalone_number(number: str, utterance: str) -> bool:
    """``number`` appears as a whole digit group (an ordinal suffix such as 15th is fine)."""
    return re.search(rf"(?<!\d){re.escape(number)}(?!\d)", utterance) is not None


def _numeric_dates(text: str) -> set[date]:
    """Dates written with digits only: ISO (y-m-d), US (m-d-y), and day-first only when unambiguous."""
    found: set[date] = set()
    for match in _NUMERIC_DATE_RE.finditer(text):
        first, middle, last = match[1], int(match[3]), match[4]
        candidates: list[date | None] = []
        if len(first) == 4 and len(last) <= 2:
            candidates.append(_safe_date(int(first), middle, int(last)))
        elif len(last) in (2, 4) and len(first) <= 2:
            # two-digit years ("3/15/85") are read as both centuries; grounding only asks "was it said"
            years = [int(last)] if len(last) == 4 else [1900 + int(last), 2000 + int(last)]
            lead = int(first)
            for year in years:
                candidates.append(_safe_date(year, lead, middle))
                if lead > 12:  # cannot be a month, so day-first is the only reading
                    candidates.append(_safe_date(year, middle, lead))
        found.update(parsed for parsed in candidates if parsed is not None)
    return found


def _dob_grounded(value: date, utterance: str) -> bool:
    """Year, month and day must all be visible: as one numeric date, or a month name with day and year."""
    text = "".join("-" if _is_dash(ch) else ch for ch in utterance.lower())
    if value in _numeric_dates(text):
        return True
    if not _standalone_number(f"{value.year:04d}", text):
        return False
    if not any(MONTHS.get(word) == value.month for word in _WORD_RE.findall(text)):
        return False
    day_forms = {str(value.day), f"{value.day:02d}"}
    return any(_standalone_number(form, text) for form in day_forms)


def _email_grounded(email: str, utterance: str) -> bool:
    pattern = rf"(?<![\w.%+\-]){re.escape(email)}(?![\w\-]|\.\w)"
    return re.search(pattern, utterance.lower()) is not None


def _name_grounded(value: str, utterance: str) -> bool:
    wanted = name_tokens(value)
    if not wanted:
        return False
    available = set(_name_form(utterance).split())
    return all(token in available for token in wanted)


def identity_value_grounded(field: str, normalized_value: str, utterance: str) -> bool:
    """True when ``normalized_value`` for ``field`` is visibly present in ``utterance``.

    Guards against a model inventing identity values. The value is re-normalized
    here, so raw and normalized forms are both accepted. Unknown fields -> False.
    """
    if not isinstance(utterance, str) or len(utterance) > MAX_INPUT_CHARS:
        return False
    if _usable(normalized_value) is None:
        return False
    utterance = unicodedata.normalize("NFKC", utterance)  # full-width digits and similar forms
    if field == "full_name":
        return _name_grounded(normalized_value, utterance)
    if field == "email":
        email = normalize_email(normalized_value)
        return email is not None and _email_grounded(email, utterance)
    if field == "phone":
        phone = normalize_phone(normalized_value)
        return phone is not None and _phone_grounded(phone, utterance)
    if field == "id_last4":
        last4 = normalize_last4(normalized_value)
        return last4 is not None and _last4_grounded(last4, utterance)
    if field == "dob":
        dob = normalize_dob(normalized_value)
        return dob is not None and _dob_grounded(dob, utterance)
    if field == "policy_number":
        policy = normalize_policy_number(normalized_value)
        return policy is not None and _standalone_number(policy.removeprefix("POL-"), utterance)
    return False
