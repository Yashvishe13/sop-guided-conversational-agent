"""Deterministic checks on model-drafted, caller-facing text.

The model drafts wording; these checks decide whether a draft may be shown:

* style: no em dashes and no colons in chat replies or email summaries
  (``check_style``), with a markup cleaner and a mechanical fixer;
* pre-verification leaks: no claim data before the identity gate
  (``check_pre_verification_leak``);
* grounding: case IDs, amounts, and dates must come from the authorized
  evidence, no promises or claimed actions, a passed deadline is never
  described as still open, a stated claim status or outcome matches the
  evidence status, and the email summary is never described as sent unless
  it was (``check_grounding``);
* internal references: no phase names, party IDs, or prompt vocabulary
  (``check_internal_reference``).

``agent/reply_guard.py`` applies these to each draft with the session's context.
A ``Violation`` carries a machine code and a detail meant for traces. Details
never contain PII or the offending amounts or dates; they hold counts, fixed
labels, and claim IDs (which are not PII).
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from decimal import Decimal, InvalidOperation

from insurance_claims.claims.evidence import GroundingTokens

__all__ = [
    "COLONS",
    "DEADLINE_MISSTATED",
    "EMAIL_STATUS_MISSTATED",
    "EM_DASHES",
    "FORBIDDEN_PROMISE",
    "GROUNDING_CODES",
    "INTERNAL_REFERENCE",
    "LEAK_CLAIM_DATA",
    "STATUS_CONTRADICTED",
    "STYLE_CODES",
    "STYLE_COLON",
    "STYLE_EM_DASH",
    "UNSUPPORTED_AMOUNT",
    "UNSUPPORTED_CASE_ID",
    "UNSUPPORTED_DATE",
    "GroundingContext",
    "Violation",
    "check_email_status",
    "check_grounding",
    "check_internal_reference",
    "check_pre_verification_leak",
    "check_style",
    "mechanical_style_fix",
    "sanitize_markup",
]

# ---------------------------------------------------------------------------
# Violation codes
# ---------------------------------------------------------------------------

STYLE_EM_DASH = "style_em_dash"
STYLE_COLON = "style_colon"
LEAK_CLAIM_DATA = "leak_claim_data"
UNSUPPORTED_CASE_ID = "unsupported_case_id"
UNSUPPORTED_AMOUNT = "unsupported_amount"
UNSUPPORTED_DATE = "unsupported_date"
FORBIDDEN_PROMISE = "forbidden_promise"
DEADLINE_MISSTATED = "deadline_misstated"
STATUS_CONTRADICTED = "status_contradicted"
EMAIL_STATUS_MISSTATED = "email_status_misstated"
INTERNAL_REFERENCE = "internal_reference"

STYLE_CODES: frozenset[str] = frozenset({STYLE_EM_DASH, STYLE_COLON})
"""Violations that ``mechanical_style_fix`` can always repair."""
GROUNDING_CODES: frozenset[str] = frozenset(
    {
        UNSUPPORTED_CASE_ID,
        UNSUPPORTED_AMOUNT,
        UNSUPPORTED_DATE,
        FORBIDDEN_PROMISE,
        DEADLINE_MISSTATED,
        STATUS_CONTRADICTED,
        EMAIL_STATUS_MISSTATED,
    }
)


@dataclass(frozen=True)
class Violation:
    code: str
    detail: str  # never contains PII


@dataclass(frozen=True)
class GroundingContext:
    tokens: GroundingTokens
    allowed_case_ids: frozenset[str]
    today: date
    deadline: date | None
    deadline_passed: bool
    case_status: str | None = None
    """The selected claim's status ("denied", "open", "closed", ...); checked only while one case ID is allowed."""
    email_status: str | None = None
    """The session's email summary status; None skips the email check, anything but "sent" forbids a sent claim."""


# ---------------------------------------------------------------------------
# Character classes
# ---------------------------------------------------------------------------

EM_DASHES = "\u2014\u2015\u2e3a\u2e3b\ufe58"
"""Em dash, horizontal bar, two- and three-em dashes, small em dash."""
COLONS = ":\uff1a\ufe55\ufe13\ua789\u2236\u02d0"
"""ASCII colon plus fullwidth, small, vertical, modifier-letter, ratio, and triangular colons."""

_SENTINEL = "\ue000"  # private-use marker for "a removed dash or colon was here"
_APOSTROPHES = str.maketrans({"\u2019": "'", "\u2018": "'", "\u02bc": "'"})
_HYPHENS = str.maketrans({c: "-" for c in "\u2010\u2011\u2012\u2013\u2212"})

# ---------------------------------------------------------------------------
# Markup and style
# ---------------------------------------------------------------------------

_MD_FENCE = re.compile(r"(?m)^[ \t]*```[^\n]*\n?")
_MD_LINK = re.compile(r"!?\[([^\[\]\n]+)\]\([^()\n]*\)")
_MD_BOLD = re.compile(r"(\*\*|__)(?=\S)([^\n]+?)(?<=\S)\1")
_MD_STAR = re.compile(r"(?<![\w*])\*(?=[^\s*])([^*\n]+?)(?<=[^\s*])\*(?![\w*])")
_MD_UNDERSCORE = re.compile(r"(?<![\w_])_(?=[^\s_])([^_\n]+?)(?<=[^\s_])_(?![\w_])")
_MD_RULE = re.compile(r"(?m)^[ \t]*(?:[-*_][ \t]*){3,}$")
_MD_HEADING = re.compile(r"(?m)^[ \t]{0,3}#{1,6}(?:[ \t]+|$)")
_MD_QUOTE = re.compile(r"(?m)^[ \t]*>[ \t]?")
_MD_BULLET = re.compile(r"(?m)^[ \t]*(?:[-*+\u2022\u25aa\u25cf\u25e6\u2023\u2013\u2014]|\d{1,3}[.)])[ \t]+")


def sanitize_markup(text: str) -> str:
    """Remove markdown emphasis, code ticks, headings, quotes, and bullets; collapse spaces."""
    out = text.replace("\r\n", "\n").replace("\r", "\n").replace("\u00a0", " ")
    out = _MD_FENCE.sub("", out)
    out = out.replace("`", "")
    out = _MD_LINK.sub(r"\1", out)
    out = _MD_BOLD.sub(r"\2", out)
    out = out.replace("**", "").replace("__", "")
    out = _MD_STAR.sub(r"\1", out)
    out = _MD_UNDERSCORE.sub(r"\1", out)
    out = _MD_RULE.sub("", out)
    out = _MD_HEADING.sub("", out)
    out = _MD_QUOTE.sub("", out)
    out = _MD_BULLET.sub("", out)
    return _tidy_whitespace(out)


def check_style(text: str) -> list[Violation]:
    """Flag em dashes (``style_em_dash``) and colons, including fullwidth ones (``style_colon``)."""
    violations: list[Violation] = []
    dashes = sum(text.count(c) for c in EM_DASHES)
    if dashes:
        violations.append(Violation(STYLE_EM_DASH, f"{dashes} em dash character(s)"))
    colons = sum(text.count(c) for c in COLONS)
    if colons:
        violations.append(Violation(STYLE_COLON, f"{colons} colon character(s)"))
    return violations


_TIME = re.compile(rf"(?<![\d{COLONS}])(\d{{1,2}})[{COLONS}]([0-5]\d)(?![\d{COLONS}])")
# The lookbehind lets only the start of a blank run try to match, which keeps long blank runs linear.
_DASH_OR_COLON_RUN = re.compile(rf"(?:(?<![ \t])[ \t]+)?[{EM_DASHES}{COLONS}](?:[ \t]*[{EM_DASHES}{COLONS}])*[ \t]*")
_S = _SENTINEL


def mechanical_style_fix(text: str) -> str:
    """Replace em dashes and colons with commas (or periods at line ends) without dropping words."""
    out = text.replace(_SENTINEL, "")
    out = _TIME.sub(_time_without_colon, out)
    out = _DASH_OR_COLON_RUN.sub(_SENTINEL, out)
    out = re.sub(rf"(?m)^([ \t]*){_S}", r"\1", out)  # leading marker: drop
    out = re.sub(rf"([.!?,;\u2026]){_S}", r"\1 ", out)  # after punctuation: drop
    out = re.sub(rf"{_S}([.!?,;\u2026])", r"\1", out)  # before punctuation: drop
    out = re.sub(rf"(?m){_S}$", ".", out)  # line end: full stop
    out = re.sub(rf"{_S}(?=[)\]])", "", out)  # before a closing bracket
    out = out.replace(_SENTINEL, ", ")
    return _tidy_whitespace(out)


def _time_without_colon(match: re.Match[str]) -> str:
    hours, minutes = match.group(1), match.group(2)
    return hours if minutes == "00" else f"{hours}.{minutes}"


def _tidy_whitespace(text: str) -> str:
    """Collapse runs of spaces, trim line ends, and cap blank lines."""
    out = re.sub(r"[ \t]{2,}", " ", text)
    out = re.sub(r"(?m)^[ \t]+|[ \t]+$", "", out)
    out = re.sub(r"\n{3,}", "\n\n", out)
    return out.strip()


# ---------------------------------------------------------------------------
# Shared parsing: case IDs, amounts, dates, sentences
# ---------------------------------------------------------------------------

_MONTH_NUMBERS: dict[str, int] = {
    "jan": 1, "january": 1, "feb": 2, "february": 2, "mar": 3, "march": 3,
    "apr": 4, "april": 4, "may": 5, "jun": 6, "june": 6, "jul": 7, "july": 7,
    "aug": 8, "august": 8, "sep": 9, "sept": 9, "september": 9, "oct": 10, "october": 10,
    "nov": 11, "november": 11, "dec": 12, "december": 12,
}  # fmt: skip
_MONTH_NAMES = (
    "", "january", "february", "march", "april", "may", "june",
    "july", "august", "september", "october", "november", "december",
)  # fmt: skip
_MONTH = (
    r"(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|june?|july?|aug(?:ust)?"
    r"|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)"
)
_ORD = r"(?:st|nd|rd|th)?"

_CASE_ID = re.compile(r"(?<![A-Za-z0-9])CL[ \t_\-\u2010-\u2013]?(\d{3,8})(?!\d)", re.IGNORECASE)

_DATE_ISO = re.compile(r"(?<![\d.])(?P<year>\d{4})[-/.](?P<month>\d{1,2})[-/.](?P<day>\d{1,2})(?![\d])")
_DATE_NUMERIC = re.compile(r"(?<![\d.,/-])(?P<a>\d{1,2})(?P<sep>[/.-])(?P<b>\d{1,2})(?P=sep)(?P<year>\d{4}|\d{2})(?![\d/-]|\.\d)")
_DATE_MONTH_FIRST = re.compile(
    rf"\b(?P<mon>{_MONTH})\.?\s+(?P<day>\d{{1,2}}){_ORD}\b(?:,?\s+(?P<year>\d{{4}})\b)?",
    re.IGNORECASE,
)
_DATE_DAY_FIRST = re.compile(
    rf"\b(?P<day>\d{{1,2}}){_ORD}\s+(?:of\s+)?(?P<mon>{_MONTH})\b\.?(?:,?\s+(?P<year>\d{{4}})\b)?",
    re.IGNORECASE,
)

_NUMBER = re.compile(r"(?<![\w.,])(?P<int>\d{1,3}(?:,\d{3})+|\d+)(?P<frac>\.\d+)?(?!\w|[.,]\d)")
_CURRENCY_BEFORE = re.compile(r"(?:\$|\bUSD|\bUS\$)\s?$", re.IGNORECASE)
_CURRENCY_AFTER = re.compile(r"^\s*(?:dollars?|usd|bucks)\b", re.IGNORECASE)
_TIME_AFTER = re.compile(r"^\s*(?:a\.?m\.?|p\.?m\.?|o'clock)(?![a-z])", re.IGNORECASE)


@dataclass(frozen=True)
class _DateMention:
    start: int
    end: int
    year: int | None
    month: int
    day: int

    def as_date(self) -> date | None:
        if self.year is None:
            return None
        try:
            return date(self.year, self.month, self.day)
        except ValueError:
            return None

    def is_valid(self) -> bool:
        try:
            date(self.year if self.year is not None else 2000, self.month, self.day)
        except ValueError:
            return False
        return True


def _case_ids(text: str) -> set[str]:
    """Canonical "CL-<digits>" IDs mentioned in the text ("cl 2048" and "CL2048" included)."""
    return {f"CL-{m.group(1)}" for m in _CASE_ID.finditer(text)}


def _canonical_case_id(value: str) -> str:
    found = _case_ids(value)
    return found.pop() if len(found) == 1 else value.strip().upper()


def _amounts(text: str) -> list[Decimal]:
    """Monetary amounts: currency-marked numbers, thousands-grouped numbers, N.NN decimals,
    and plain whole numbers right after a money word ("the allowed maximum is 1500")."""
    found: list[Decimal] = []
    date_spans: list[tuple[int, int]] | None = None
    for match in _NUMBER.finditer(text):
        integer, frac = match.group("int"), match.group("frac") or ""
        before = text[max(0, match.start() - 8) : match.start()]
        after = text[match.end() : match.end() + 12]
        currency = bool(_CURRENCY_BEFORE.search(before) or _CURRENCY_AFTER.match(after))
        if not currency and not _looks_like_bare_amount(integer, frac, after):
            if frac or "," in integer or len(integer) < 2:
                continue
            if date_spans is None:
                date_spans = [(d.start, d.end) for d in _date_mentions(text)]
            if not _money_word_before(text, match, date_spans):
                continue
        try:
            found.append(Decimal(integer.replace(",", "") + frac))
        except InvalidOperation:  # pragma: no cover - the regex only admits digits
            continue
    return found


def _looks_like_bare_amount(integer: str, frac: str, after: str) -> bool:
    """Without a currency marker, only "1,450", "1,450.00", or "1450.00" (not "3.30 pm") count."""
    two_decimals = len(frac) == 3  # frac includes the dot
    if "," in integer:
        return not frac or two_decimals
    return two_decimals and not _TIME_AFTER.match(after)


_MONEY_WORD = re.compile(
    r"\b(?:amounts?|max(?:imum)?|minimum|reimburs\w*|pay|pays|paid|paying|payments?|payouts?|fees?|refunds?|"
    r"costs?|owed?|owing|balance|total|deductible|copay|net|sum|price|charged?|billed|allowed|covered|"
    r"coverage|benefits?|compensation|settlement|settled|receive)\b",
    re.IGNORECASE,
)
_NOT_MONEY_BEFORE = re.compile(
    r"(?:\bCL[ \t_-]?|#|\b(?:claim|case|policy|number|no|id|reference|ref|ssn|digits?|phone|page|step|item|"
    r"version|line|unit|zip|code|extension|ext)\s+)$",
    re.IGNORECASE,
)
_NOT_MONEY_AFTER = re.compile(
    r"^\s*(?:%|percent|days?|weeks?|months?|years?|hours?|minutes?|business|calendar|working|pages?|documents?|"
    r"files?|digits?|times?|claims?|items?|photos?|pictures?|visits?|people|members?|a\.?m\b|p\.?m\b|[-/])",
    re.IGNORECASE,
)
_YEAR_LEAD = re.compile(
    rf"\b(?:in|since|of|from|for|by|until|till|before|after|during|year|early|late|mid|this|last|next|{_MONTH})\s+$",
    re.IGNORECASE,
)


def _money_word_before(text: str, match: re.Match[str], date_spans: Sequence[tuple[int, int]]) -> bool:
    """A plain whole number counts as money only right after a money word, and never as a date, ID, or count."""
    start, end = match.start(), match.end()
    if any(s <= start < e for s, e in date_spans):
        return False
    lead = text[max(0, start - 40) : start]
    if _NOT_MONEY_BEFORE.search(lead) or _NOT_MONEY_AFTER.match(text[end : end + 12]):
        return False
    value = int(match.group("int"))
    if 1900 <= value <= 2099 and len(match.group("int")) == 4 and _YEAR_LEAD.search(lead):
        return False
    clause = re.split(r"[.!?;\n]", lead)[-1]
    return bool(_MONEY_WORD.search(" ".join(clause.split()[-4:])))


def _date_mentions(text: str) -> list[_DateMention]:
    """Every date-like mention, earliest pattern wins on overlap."""
    mentions: list[_DateMention] = []
    taken: list[tuple[int, int]] = []

    def claim(start: int, end: int) -> bool:
        if any(start < t_end and t_start < end for t_start, t_end in taken):
            return False
        taken.append((start, end))
        return True

    for m in _DATE_ISO.finditer(text):
        if claim(m.start(), m.end()):
            mentions.append(_DateMention(m.start(), m.end(), int(m["year"]), int(m["month"]), int(m["day"])))
    for m in _DATE_NUMERIC.finditer(text):
        year = int(m["year"])
        year = year + 2000 if year < 100 else year
        a, b = int(m["a"]), int(m["b"])
        month, day = (b, a) if a > 12 >= b else (a, b)
        if claim(m.start(), m.end()):
            mentions.append(_DateMention(m.start(), m.end(), year, month, day))
    for pattern in (_DATE_MONTH_FIRST, _DATE_DAY_FIRST):
        for m in pattern.finditer(text):
            month_word = m["mon"].lower().rstrip(".")
            year = int(m["year"]) if m["year"] else None
            if pattern is _DATE_DAY_FIRST and month_word == "may" and year is None:
                continue  # "2 may apply" is not a date
            if claim(m.start(), m.end()):
                mentions.append(_DateMention(m.start(), m.end(), year, _MONTH_NUMBERS[month_word], int(m["day"])))
    mentions.sort(key=lambda d: d.start)
    return mentions


def _sentences(text: str) -> list[str]:
    """Split prose into sentences without breaking amounts or abbreviated months."""
    flat = re.sub(rf"\b({_MONTH})\.", r"\1", text, flags=re.IGNORECASE)
    parts = re.split(r"(?<=[.!?])\s+|\n+", flat)
    return [p.strip() for p in parts if p and p.strip()]


# ---------------------------------------------------------------------------
# Pre-verification leak detection
# ---------------------------------------------------------------------------


def check_pre_verification_leak(text: str, secret_tokens: Sequence[str]) -> list[Violation]:
    """Flag any protected claim token in the reply (case-insensitive, format-tolerant)."""
    haystack = _leak_haystack(text)
    tokens = {_leak_normalize(t) for t in secret_tokens if isinstance(t, str)}
    hits = sum(1 for token in tokens if len(token) >= 2 and _contains_bounded(haystack, token))
    if not hits:
        return []
    return [Violation(LEAK_CLAIM_DATA, f"{hits} protected claim token(s) in a pre-verification reply")]


def _leak_normalize(value: str) -> str:
    """Lowercase and canonicalize formats so "CL 2048", "$1,450", "Mar 18th" match their tokens."""
    out = unicodedata.normalize("NFKC", value).translate(_HYPHENS).translate(_APOSTROPHES).lower()
    out = re.sub(r"(\d)(?:st|nd|rd|th)\b", r"\1", out)
    out = re.sub(r"(?<=\d),(?=\d{3}(?!\d))", "", out)
    out = re.sub(rf"\b({_MONTH})\.?(?=\s*\d)", lambda m: _MONTH_NAMES[_MONTH_NUMBERS[m.group(1)]], out)
    out = re.sub(rf"\b(\d{{1,2}})\s+(?:of\s+)?({_MONTH})\b", _month_first, out)
    out = _CASE_ID.sub(lambda m: f"cl-{m.group(1)}", out)
    return re.sub(r"\s+", " ", out).strip()


def _month_first(match: re.Match[str]) -> str:
    """Rewrite "18 march" as "march 18", but leave "2 may apply" alone."""
    day, month = match.group(1), match.group(2)
    if month == "may":
        return match.group(0)
    return f"{_MONTH_NAMES[_MONTH_NUMBERS[month]]} {day}"


def _leak_haystack(text: str) -> str:
    """Normalized text plus canonical spellings of every date it mentions."""
    normalized = _leak_normalize(text)
    extra: list[str] = []
    for mention in _date_mentions(normalized):
        if not mention.is_valid():
            continue
        month_name = _MONTH_NAMES[mention.month]
        extra.append(f"{month_name} {mention.day}")
        parsed = mention.as_date()
        if parsed is not None:
            extra.append(parsed.isoformat())
            extra.append(f"{month_name} {mention.day}, {mention.year}")
    return "\n".join([normalized, *extra])


def _contains_bounded(haystack: str, token: str) -> bool:
    """Substring match that does not start or end inside a word or number."""
    start = 0
    while (index := haystack.find(token, start)) != -1:
        end = index + len(token)
        left_ok = not token[0].isalnum() or index == 0 or not haystack[index - 1].isalnum()
        right_ok = not token[-1].isalnum() or end == len(haystack) or not haystack[end].isalnum()
        if left_ok and right_ok:
            return True
        start = index + 1
    return False


# ---------------------------------------------------------------------------
# Grounding
# ---------------------------------------------------------------------------

_NEGATORS = re.compile(
    r"\b(?:not|no|never|cannot|can't|won't|unable|without|isn't|aren't|wasn't|don't|doesn't|"
    r"couldn't|nor|neither)\b|n't\b",
    re.IGNORECASE,
)
_HEDGES = re.compile(
    r"\b(?:whether|unclear|uncertain|depends|depending|"
    r"only\s+(?:\w+\s+){0,3}?(?:can|could|may|might)|"
    r"can't (?:say|tell|promise|predict)|cannot (?:say|tell|promise|predict)|"
    r"no way to know|not sure|don't know|do not know)\b",
    re.IGNORECASE,
)
# Negative words that strengthen rather than soften ("I have no doubt it will be approved").
_INTENSIFIERS = re.compile(
    r"\b(?:no|not\s+(?:a|any)|without(?:\s+(?:a|any))?|beyond(?:\s+(?:a|any))?)\s+(?:shadow\s+of\s+a\s+)?"
    r"(?:doubts?|questions?)\b"
    r"|\bno\s+(?:worries|problem|need\s+to\s+worry)\b|\b(?:don't|do\s+not|not\s+to)\s+worry\b|\bnot\s+a\s+problem\b",
    re.IGNORECASE,
)
_CONDITIONALS = re.compile(
    r"\b(?:if|once|when|whenever|after|unless|until|till|whether|before|assuming|suppose|supposing|"
    r"in\s+case|as\s+soon\s+as)\b",
    re.IGNORECASE,
)
_NOTHING = re.compile(r"\b(?:nothing|none|nobody|anything|yet)\b", re.IGNORECASE)

_I_WE = r"\b(?:I|we)"
_I_WE_DID = rf"{_I_WE}(?:\s+have|'ve)?(?:\s+(?:just|already|now|also))?"
_ACTION_PAST = r"(?:submitted|uploaded|filed|faxed|mailed|resubmitted|re-submitted)"
# Past-tense verbs that count as an invented action only when their object is paperwork or an appeal.
_HANDLED_PAST = r"(?:sent|resent|re-sent|forwarded|attached|started|initiated|opened|reopened|re-opened|lodged)"
_PAPERWORK = r"(?:documents?|files?|records?|forms?|paperwork|appeals?|reports?|notes?)"
_NOT_A_HANDOFF = r"(?!\s+(?:a|an|your|the)\s+(?:handoff\s+|callback\s+)?(?:request|ticket|referral))"
_OUTCOME = r"(?:approved|paid|overturned|reversed|accepted|reimbursed)"

# How a promise pattern may be excused: never, when negated or hedged in its own clause,
# or (for statements of fact) also when its clause is conditional ("Once your documents have been received").
_HARD, _SOFT, _ASSERT = "hard", "soft", "assert"

_PROMISES: tuple[tuple[str, re.Pattern[str], str], ...] = (
    ("guarantee", re.compile(r"\bguarantee(?:d|s|ing)?\b", re.IGNORECASE), _SOFT),
    ("promise", re.compile(rf"{_I_WE}\s+promise\b", re.IGNORECASE), _SOFT),
    (
        "outcome",
        re.compile(
            r"(?:\b(?:will|should|going\s+to|gonna|likely\s+to|expected\s+to|sure\s+to|bound\s+to|set\s+to)|'ll)\s+"
            rf"(?:\w+\s+){{0,2}}?(?:be|get)\s+(?:\w+\s+and\s+)?{_OUTCOME}\b"
            r"|\byou(?:'ll|\s+will|\s+should)\s+(?:\w+\s+)?(?:receive|get)\s+"
            r"(?:\$|a\s+(?:payment|refund|reimbursement|check)\b"
            r"|(?:your|the)\s+(?:money|payment|reimbursement|refund)\b)",
            re.IGNORECASE,
        ),
        _SOFT,
    ),
    ("certainty", re.compile(r"\bwill\s+definitely\b|\b100\s*(?:%|percent\b)", re.IGNORECASE), _SOFT),
    ("approval", re.compile(r"\bapprove\s+your\s+claim\b", re.IGNORECASE), _SOFT),
    (
        "claimed_action",
        re.compile(
            rf"{_I_WE_DID}(?:\s+(?:gone|went)\s+ahead\s+and)?\s+{_ACTION_PAST}\b{_NOT_A_HANDOFF}"
            rf"|{_I_WE_DID}\s+{_HANDLED_PAST}\s+(?:\w+\s+){{0,2}}?{_PAPERWORK}\b",
            re.IGNORECASE,
        ),
        _HARD,
    ),
    (
        "claimed_action",
        re.compile(
            rf"\b{_PAPERWORK}\s+(?:has|have)(?:\s+(?:now|already|just|also))?\s+been\s+"
            rf"(?:{_ACTION_PAST}|{_HANDLED_PAST}|received)\b"
            r"|\b(?:has|have)(?:\s+(?:now|already|just|also))?\s+(?:been\s+)?re-?opened\b",
            re.IGNORECASE,
        ),
        _ASSERT,
    ),
    (
        "claimed_receipt",
        re.compile(rf"{_I_WE_DID}\s+received\s+(?:\w+\s+){{0,2}}?{_PAPERWORK}\b", re.IGNORECASE),
        _ASSERT,
    ),
    (
        "promised_action",
        re.compile(
            rf"{_I_WE}(?:\s+will|'ll)\s+(?:go\s+ahead\s+and\s+)?"
            rf"(?:submit|upload|file|fax|mail|resubmit)\b{_NOT_A_HANDOFF}"
            rf"|{_I_WE}(?:\s+will|'ll)\s+send\s+(?:\w+\s+){{0,2}}?{_PAPERWORK}\b"
            r"|\blet\s+me\s+(?:submit|upload|file)\b|\bI\s+can\s+(?:submit|upload|file)\b",
            re.IGNORECASE,
        ),
        _HARD,
    ),
    ("entitlement", re.compile(r"\byou(?:\s+are|'re)\s+(?:fully\s+)?entitled\b", re.IGNORECASE), _SOFT),
    (
        "appeal_right",
        re.compile(
            r"\byou(?:\s+still)?\s+(?:have|'ve\s+got|'ve)\s+(?:a|the)\s+right\s+to\s+(?:an?\s+)?appeal"
            r"|\byou(?:\s+are|'re)\s+(?:still\s+|also\s+|fully\s+)?(?:eligible|able)\s+(?:to|for)\s+"
            r"(?:file\s+|submit\s+)?(?:an?\s+)?appeal"
            r"|\b(?:have|get|given|allowed)\s+(?:\w+\s+){1,2}?(?:days?|weeks?|months?)\b[^.!?\n]{0,40}?"
            r"\bto\s+(?:file\s+|submit\s+|request\s+|start\s+)?(?:an?\s+|your\s+|the\s+)?appeal",
            re.IGNORECASE,
        ),
        _SOFT,
    ),
)

_DEADLINE_PHRASE = re.compile(
    r"\bappeals?\s+deadline\b"
    r"|\bdeadline\s+(?:to|for)\s+(?:file\s+|filing\s+|submit\s+|submitting\s+)?(?:an?\s+|your\s+|the\s+)?appeal"
    r"|\b(?:last|final)\s+day\s+(?:to|for)\s+(?:file\s+|filing\s+|submit\s+)?(?:an?\s+|your\s+|the\s+)?appeal"
    r"|\bappeals?\s+(?:window|period|time\s+limit)\b"
    r"|\btime\s+limit\s+(?:to|for)\s+(?:file\s+)?(?:an?\s+|your\s+)?appeal",
    re.IGNORECASE,
)
# A year-less "3/18" is only read as the deadline, never as a general date (it could be a fraction).
_SHORT_DATE = re.compile(r"(?<![\d/.,-])(?P<a>\d{1,2})/(?P<b>\d{1,2})(?![\d/]|\.\d)")
_FUTURE_WORDING = re.compile(
    r"\b(?:until|till|by|before|still\s+have|upcoming|coming\s+up|you\s+can\s+still|still\s+time|not\s+too\s+late|"
    r"is\s+due|are\s+due|still\s+open|is\s+(?:today|tomorrow)|soon|away|next\s+(?:week|month)|plenty\s+of\s+time|"
    r"(?:days?|weeks?|months?|time)\s+(?:left|remaining)|remaining\s+(?:days?|weeks?|months?|time))\b",
    re.IGNORECASE,
)
# "gone"/"went" cover "has gone by", "came and went", and "has come and gone". "is over" counts only at a
# clause end, so "you still have over two weeks" is not read as past.
_PAST_WORDING = re.compile(
    r"\b(?:passed|expired|was|were|had|ended|already|lapsed|past|missed|elapsed|closed|no\s+longer|anymore|"
    r"any\s+longer|gone|went|behind\s+(?:us|you)|"
    r"(?:is|are)\s+(?:now\s+)?over(?=\s*(?:[.,;!?]|$)|\s+(?:so|and|but|now)\b))\b",
    re.IGNORECASE,
)
_SEGMENT_BREAK = re.compile(r"[,;:()]|\b(?:and|but)\b", re.IGNORECASE)
_STILL_TIME = re.compile(
    r"\bstill\s+(?:have|has|got)\s+(?:some\s+|plenty\s+of\s+|enough\s+)?time\b|\bstill\s+(?:some\s+)?time\b"
    r"|\bnot\s+too\s+late\b|\bplenty\s+of\s+time\b|\b(?:time|days?|weeks?|months?)\s+(?:left|remaining)\b"
    r"|\bstill\s+(?:open|upcoming|ahead)\b",
    re.IGNORECASE,
)
_TIME_FOR_SOMETHING_ELSE = re.compile(
    r"\s+(?:to|for)\s+(?!(?:file\s+|submit\s+|start\s+|request\s+)?(?:an?\s+|your\s+|the\s+)?appeal)\w",
    re.IGNORECASE,
)
_APPEAL_OPEN = re.compile(
    r"\bstill\s+(?:(?:have|has|got|plenty|of|enough|some|time|be|being|able|possible|eligible|allowed|open|within|"
    r"in|the|an?|your|to|file|submit|request|start|make|lodge|for|can|could|may)\s+){0,6}appeal"
    r"|\b(?:can|could|may)\s+(?:also\s+|still\s+){0,2}"
    r"(?:file\s+|submit\s+|request\s+|start\s+|make\s+|lodge\s+|pursue\s+)?"
    r"(?:an?\s+|your\s+|the\s+)?appeal\b"
    r"|\b(?:until|till)\b[^.!?;\n]{0,40}?\bto\s+(?:file\s+|submit\s+)?(?:an?\s+|your\s+)?appeal"
    r"|\bnot\s+too\s+late\s+(?:for\s+(?:an?\s+)?|to\s+(?:file\s+|submit\s+)?(?:an?\s+)?)appeal"
    r"|\bappeals?\s+(?:window|period)\s+(?:is\s+)?(?:still\s+)?open\b",
    re.IGNORECASE,
)

# Status and outcome wording, compared with the selected claim's evidence status.
_DONE = r"(?:approved|paid|settled|overturned|reversed|reimbursed)"
_CLAIM_SUBJECT = r"\b(?:claim|case|CL[ -]?\d+)"
_STATUS_POSITIVE = re.compile(
    rf"\b(?:has|have|had)\s+(?:now\s+|already\s+|just\s+|finally\s+|also\s+)?been\s+(?:\w+\s+and\s+)?{_DONE}\b"
    rf"|(?:\b(?:is|are|was|were|got|gets)|'s|'re)\s+(?:now\s+|already\s+|fully\s+|finally\s+|also\s+)?{_DONE}\b"
    r"|\bpayment\s+(?:has\s+been|was|is\s+being|is)\s+(?:issued|sent|made|released|processed|scheduled)\b"
    r"|\b(?:I|we|they)(?:\s+have|'ve)?\s+(?:now\s+|already\s+|just\s+)?"
    r"(?:approved|paid|reimbursed|overturned|reversed)\b",
    re.IGNORECASE,
)
_STATUS_REVIEW = re.compile(
    r"(?:\b(?:is|are)|'s)\s+(?:now\s+|currently\s+|back\s+|again\s+){0,2}(?:under|in)\s+(?:review|reconsideration)\b"
    r"|(?:\b(?:is|are)|'s)\s+(?:now\s+|currently\s+|again\s+)?being\s+"
    r"(?:reprocessed|reviewed|reconsidered|re-?reviewed|re-?evaluated|reopened)\b"
    r"|\b(?:has|have)\s+(?:now\s+|already\s+|just\s+)?been\s+(?:reopened|re-opened|reprocessed|reconsidered)\b",
    re.IGNORECASE,
)
_STATUS_OPEN = re.compile(
    rf"{_CLAIM_SUBJECT}(?:\s+(?:is|remains)|'s)\s+(?:still\s+|now\s+|currently\s+){{0,2}}"
    r"(?:open|pending|in\s+progress|active)\b",
    re.IGNORECASE,
)
_STATUS_CLOSED = re.compile(
    rf"{_CLAIM_SUBJECT}(?:\s+(?:is|was|has\s+been|remains)|'s)\s+(?:now\s+|already\s+|currently\s+){{0,2}}closed\b",
    re.IGNORECASE,
)
_STATUS_DENIED = re.compile(
    r"(?:\b(?:is|are|was|were|got|has\s+been|have\s+been)|'s)\s+(?:now\s+|already\s+|currently\s+){0,2}denied\b",
    re.IGNORECASE,
)
_STATUS_CONFLICTS: dict[str, tuple[re.Pattern[str], ...]] = {
    "denied": (_STATUS_POSITIVE, _STATUS_REVIEW, _STATUS_OPEN, _STATUS_CLOSED),
    "open": (_STATUS_POSITIVE, _STATUS_DENIED, _STATUS_CLOSED),
    "pending": (_STATUS_POSITIVE, _STATUS_DENIED, _STATUS_CLOSED),
    "closed": (_STATUS_DENIED, _STATUS_REVIEW, _STATUS_OPEN),
}
# "about why the claim was denied" reports a topic, it does not state the status.
_EMBEDDED_QUESTION = re.compile(
    r"\b(?:about|discussed|asked|ask|asking|explain|explained|wondering|wonder|know|understand|see)"
    r"\s+why\s+(?:\w+\s+){0,3}$",
    re.IGNORECASE,
)
_ZERO_OR_YET = re.compile(r"\s*(?:\$\s?)?0(?:\.0+)?(?![\d.,]\d)|\s+(?:yet|nothing)\b", re.IGNORECASE)

# Claims that the email summary was sent or delivered.
_EMAIL_THING = r"(?:summary|summaries|email|e-mail|recap|copy)"
_EMAIL_SENT_CLAIM = re.compile(
    rf"{_I_WE_DID}\s+(?:sent|re-?sent|e-?mailed|forwarded|delivered|mailed)\s+"
    rf"(?:you\s+)?(?:\w+\s+){{0,2}}?{_EMAIL_THING}\b"
    rf"|{_I_WE_DID}\s+(?:sent|re-?sent|forwarded|delivered|mailed)\s+"
    r"(?:it|that|this|everything)\s+(?:over\s+|along\s+)?"
    r"(?:to\s+you\b|to\s+your\s+(?:email|e-mail|inbox|address)|by\s+e-?mail|via\s+e-?mail)"
    rf"|{_I_WE_DID}\s+e-?mailed\b"
    rf"|\b{_EMAIL_THING}(?:\s+(?:has|have|had)(?:\s+(?:now|already|just))?\s+been|'s(?:\s+(?:now|already|just))?\s+been"
    r"|\s+(?:was|is|got)(?:\s+(?:now|already|just|successfully))?)\s+(?:sent|e-?mailed|delivered|dispatched)\b"
    rf"|\b{_EMAIL_THING}\s+(?:has\s+)?(?:went|gone)\s+(?:out|through)\b"
    rf"|\b{_EMAIL_THING}(?:\s+is|'s)\s+on\s+(?:its|the)\s+way\b"
    r"|\b(?:be|is|it's|'s|are|now|already|sitting|waiting)\s+(?:\w+\s+)?in\s+your\s+inbox\b"
    r"|\b(?:arrived?|arrives|landed|lands|showed\s+up|shows\s+up|went|gone|delivered|reached|hit|hits)\s+(?:\w+\s+)?"
    r"(?:in\s+|into\s+|to\s+)?your\s+inbox\b"
    r"|\bcheck\s+your\s+inbox\b",
    re.IGNORECASE,
)
_LEADING_CONDITIONAL = re.compile(r"\s*(?:if|once|when|whenever|after|as\s+soon\s+as|assuming|unless)\b", re.IGNORECASE)
_PAST_DELIVERY = re.compile(
    r"\b(?:sent|re-?sent|e-?mailed|forwarded|delivered|mailed|dispatched|went|gone|arrived|landed|showed|"
    r"reached|already)\b",
    re.IGNORECASE,
)

# Comparison words that let a reply state the gap between two evidence amounts ("$20.00 less than").
_COMPARISON = re.compile(
    r"\b(?:less|more|lower|higher|below|above|short|shy|under|over|difference|differs?|gap|remaining|remainder|"
    r"left|than|minus|reduced|reduction|fewer)\b",
    re.IGNORECASE,
)


def check_grounding(text: str, ctx: GroundingContext) -> list[Violation]:
    """Check case IDs, amounts, dates, promises, deadline, status, and email wording against the evidence."""
    prose = unicodedata.normalize("NFKC", text).translate(_APOSTROPHES).translate(_HYPHENS)
    violations: list[Violation] = []
    checks = (
        _check_case_ids, _check_amounts, _check_dates, _check_promises, _check_deadline, _check_status, _check_email,
    )  # fmt: skip
    for check in checks:
        violation = check(prose, ctx)
        if violation is not None:
            violations.append(violation)
    return violations


def check_email_status(text: str, email_status: str | None) -> list[Violation]:
    """Flag a reply that says the summary was sent or is in the inbox when the status is not "sent".

    ``email_status=None`` means the caller did not supply a status, so nothing is checked.
    """
    if email_status is None or email_status == "sent":
        return []
    prose = unicodedata.normalize("NFKC", text).translate(_APOSTROPHES).translate(_HYPHENS)
    for match in _EMAIL_SENT_CLAIM.finditer(prose):
        if _is_softened(prose, match, conditional=True):
            continue
        if _NOTHING.search(prose[_clause_start(prose, match.start()) : match.start()]):
            continue  # "nothing was sent to your inbox"
        sentence_start = max(prose.rfind(c, 0, match.start()) for c in ".!?\n") + 1
        if _LEADING_CONDITIONAL.match(prose, sentence_start) and not _PAST_DELIVERY.search(match.group(0)):
            continue  # "If you say yes, it will be in your inbox shortly."
        detail = f"reply says the email was sent but its status is {_status_label(email_status)}"
        return [Violation(EMAIL_STATUS_MISSTATED, detail)]
    return []


def _status_label(value: str) -> str:
    """Status values are fixed enum words; anything else is reported as a fixed label, never echoed."""
    cleaned = value.strip().lower()
    return cleaned if re.fullmatch(r"[a-z_]{1,24}", cleaned) else "other"


def _check_case_ids(text: str, ctx: GroundingContext) -> Violation | None:
    allowed = {_canonical_case_id(c) for c in ctx.allowed_case_ids}
    unsupported = sorted(_case_ids(text) - allowed)
    if not unsupported:
        return None
    return Violation(UNSUPPORTED_CASE_ID, "claim ID(s) not in the evidence " + " ".join(unsupported))


def _check_amounts(text: str, ctx: GroundingContext) -> Violation | None:
    """Every amount must be an evidence amount; a sentence that compares amounts may also state
    the difference between two evidence amounts ("$780.00 of the $800.00 allowed, $20.00 less")."""
    allowed = {a for a in ctx.tokens.amounts if a.is_finite()}
    differences: set[Decimal] | None = None
    unsupported: set[Decimal] = set()
    for sentence in _sentences(text):
        for amount in _amounts(sentence):
            if amount in allowed:
                continue
            if _COMPARISON.search(sentence):
                if differences is None:
                    positive = [a for a in allowed if a > 0]
                    differences = {abs(a - b) for a in positive for b in positive if a != b}
                if amount in differences:
                    continue
            unsupported.add(amount)
    if not unsupported:
        return None
    return Violation(UNSUPPORTED_AMOUNT, f"{len(unsupported)} amount(s) not found in the evidence")


def _check_dates(text: str, ctx: GroundingContext) -> Violation | None:
    allowed = set(ctx.tokens.dates) | {ctx.today}
    month_days = {(d.month, d.day) for d in allowed}
    bad = 0
    for mention in _date_mentions(text):
        if not mention.is_valid():
            bad += 1
        elif mention.year is None:
            bad += (mention.month, mention.day) not in month_days
        else:
            bad += mention.as_date() not in allowed
    if not bad:
        return None
    return Violation(UNSUPPORTED_DATE, f"{bad} date(s) not found in the evidence")


def _check_promises(text: str, ctx: GroundingContext) -> Violation | None:
    labels = sorted(
        {
            label
            for label, pattern, mode in _PROMISES
            for match in pattern.finditer(text)
            if mode == _HARD or not _is_softened(text, match, conditional=mode == _ASSERT)
        }
    )
    if not labels:
        return None
    return Violation(FORBIDDEN_PROMISE, "forbidden wording " + " ".join(labels))


def _clause_start(text: str, index: int) -> int:
    return max(text.rfind(c, 0, index) for c in ".!?;,\n") + 1


def _is_softened(text: str, match: re.Match[str], *, conditional: bool = False) -> bool:
    """True when the phrase is negated or hedged inside its own clause ("I can't guarantee").

    Intensifiers such as "no doubt" do not count as negation. With ``conditional``, a clause
    that opens with if, once, when, and the like also counts ("Once your documents have been received").
    """
    if _NEGATORS.search(match.group(0)):
        return True
    before = _INTENSIFIERS.sub(" ", text[_clause_start(text, match.start()) : match.start()])
    window = " ".join(before.split()[-6:])
    if _NEGATORS.search(window) or _HEDGES.search(before):
        return True
    return conditional and bool(_CONDITIONALS.search(before))


def _check_deadline(text: str, ctx: GroundingContext) -> Violation | None:
    """A passed deadline must never read as open, current, or still appealable."""
    if not ctx.deadline_passed:
        return None
    misstated = Violation(DEADLINE_MISSTATED, "a passed deadline is described as still open")
    previous_mentions_deadline = False
    for sentence in _sentences(text):
        mentions = _deadline_mentions(sentence, ctx.deadline)
        if mentions and (_deadline_reads_open(sentence, mentions) or _claims_time_left(sentence)):
            return misstated
        if previous_mentions_deadline and _claims_time_left(sentence):
            return misstated  # "The appeal deadline was March 18. You still have time."
        previous_mentions_deadline = bool(mentions)
    if _appeal_reads_open(text):
        return misstated
    return None


def _deadline_mentions(sentence: str, deadline: date | None) -> list[tuple[int, bool]]:
    """Start offsets of deadline mentions in the sentence, each flagged True when it is the date itself."""
    found = [(m.start(), False) for m in _DEADLINE_PHRASE.finditer(sentence)]
    if deadline is None:
        return found
    target = (deadline.month, deadline.day)
    for mention in _date_mentions(sentence):
        if (mention.month, mention.day) == target and mention.year in (None, deadline.year):
            found.append((mention.start, True))
    for m in _SHORT_DATE.finditer(sentence):
        a, b = int(m["a"]), int(m["b"])
        if target in ((a, b), (b, a)):
            found.append((m.start(), True))
    return found


def _deadline_reads_open(sentence: str, mentions: Sequence[tuple[int, bool]]) -> bool:
    """Open wording about the deadline, or the deadline date stated without saying it is past.

    Each mention is judged from the start of its own clause to the end of the sentence, so an
    unrelated "was" earlier in the sentence ("Since it was denied, you have until March 18")
    does not hide the misstatement, and "The appeal deadline is March 18, 2026." is caught.
    """
    if _reads_as_open(sentence):
        return True
    for start, is_date in mentions:
        tail = sentence[_segment_start(sentence, start) :]
        if _PAST_WORDING.search(tail):
            continue
        if is_date or _FUTURE_WORDING.search(tail):
            return True
    return False


def _segment_start(sentence: str, index: int) -> int:
    last = 0
    for m in _SEGMENT_BREAK.finditer(sentence, 0, index):
        last = m.end()
    return last


def _reads_as_open(text: str) -> bool:
    return bool(_FUTURE_WORDING.search(text)) and not _PAST_WORDING.search(text)


def _claims_time_left(sentence: str) -> bool:
    """ "You still have time" or "it's not too late", unless negated, hedged, past, or about another task."""
    for match in _STILL_TIME.finditer(sentence):
        if _TIME_FOR_SOMETHING_ELSE.match(sentence, match.end()):
            continue  # "you still have time to upload the documents"
        before = sentence[_clause_start(sentence, match.start()) : match.start()]
        if _open_claim_excused(before, match.group(0)):
            continue
        return True
    return False


def _appeal_reads_open(text: str) -> bool:
    """An appeal described as still possible ("You can still file an appeal", "until 3/18 to appeal")."""
    for match in _APPEAL_OPEN.finditer(text):
        before = text[_clause_start(text, match.start()) : match.start()]
        if _open_claim_excused(before, match.group(0)):
            continue
        return True
    return False


def _open_claim_excused(before: str, matched: str) -> bool:
    """Negated, hedged, conditional, or past ("You had until March 18 to appeal") in its own clause."""
    before = _INTENSIFIERS.sub(" ", before)
    if _NEGATORS.search(before) or _HEDGES.search(before) or _CONDITIONALS.search(before):
        return True
    return bool(_PAST_WORDING.search(before + " " + matched))


def _check_status(text: str, ctx: GroundingContext) -> Violation | None:
    """A stated status or outcome must not contradict the selected claim's evidence status.

    One status describes one claim, so the check is skipped when several case IDs are in scope
    (a summary that also covers another claim may state that claim's own status).
    """
    status = (ctx.case_status or "").strip().lower()
    if len({_canonical_case_id(c) for c in ctx.allowed_case_ids}) > 1:
        return None
    for pattern in _STATUS_CONFLICTS.get(status, ()):
        for match in pattern.finditer(text):
            if _is_softened(text, match, conditional=True):
                continue
            if _NOTHING.search(text[_clause_start(text, match.start()) : match.start()]):
                continue  # "nothing has been paid so far"
            if _ZERO_OR_YET.match(text, match.end()):
                continue  # "you were paid $0.00", "it has been paid yet"
            if _EMBEDDED_QUESTION.search(text, max(0, match.start() - 60), match.start()):
                continue  # "We talked about why the claim was denied"
            detail = f"status or outcome wording conflicts with the evidence status {status}"
            return Violation(STATUS_CONTRADICTED, detail)
    return None


def _check_email(text: str, ctx: GroundingContext) -> Violation | None:
    found = check_email_status(text, ctx.email_status)
    return found[0] if found else None


# ---------------------------------------------------------------------------
# Internal references
# ---------------------------------------------------------------------------

_INTERNAL = re.compile(
    r"\b(?:VERIFY_ID|RESOLVE_INTENT|PROCESS_CASE|POST_PROCESS|party_id)\b"
    r"|\bparty\s+id\b|\bsystem\s+prompt\b|\bdeveloper\s+message\b",
    re.IGNORECASE,
)
_PARTY_REF = re.compile(r"(?<![\w-])P\d{1,6}\b")


def check_internal_reference(text: str) -> list[Violation]:
    """Phase names, party IDs, and prompt vocabulary must never reach the caller."""
    hits = len(_INTERNAL.findall(text)) + len(_PARTY_REF.findall(text))
    if not hits:
        return []
    return [Violation(INTERNAL_REFERENCE, f"{hits} internal identifier(s) in the reply")]
