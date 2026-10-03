"""Nested JSON trace spans, redacted before they touch disk (agent-tracing skill, PII-adapted).

Usage::

    tracing.configure(trace_dir=settings.trace_dir, enabled=settings.traces_enabled,
                      redactor=build_fixture_redactor(bundle, secrets=[settings.openai_api_key]))

    with tracing.span("turn", type="task", metadata={"phase": "VERIFY_ID"}):
        tracing.event("verification_gate", status="pending", provided=2)
        tracing.annotate(stop_reason="awaiting_identity")

Spans nest through a ``ContextVar`` (``reset(token)``, never ``set(None)``), so
siblings keep the right parent and asyncio tasks get their own branch. A span
is a plain dict::

    {"name", "type", "start", "duration_ms", "input", "output", "metadata", "tokens"?, "error"?, "children"}

Values are snapshotted into plain JSON data when captured (later mutation of
the caller's objects does not rewrite history). When a root span closes,
exactly one file is written: the tree is redacted with the configured
:class:`~insurance_claims.observability.redaction.Redactor`, strings over 4000 chars are
truncated, the JSON is serialized, every JSON string is scrubbed once more as
a safety net, and the file is written to a temp file and atomically replaced
(mode 0600). Tracing never raises into the caller: write errors are swallowed
and counted in ``write_failures``. Until :func:`configure` enables it, nothing
is written (spans still work in memory).

:func:`wrap_openai` records metadata only (model, limits, tool names, schema
name, a sha256 of the instructions), token usage, status, output item types and
function-call names. Raw prompts, inputs, and outputs are never stored.

:func:`summarize` reads these keys: ``phase_before`` (input/metadata/output, where a
span started), ``from_phase``/``to_phase`` (a transition), ``phase``/``phase_after``
(where a span ended; the root's value is the final phase), and ``stop_reason``
(distinct values, in order). Do not record the same usage twice: ``llm``
spans from :func:`wrap_openai` already carry tokens, so only call
:func:`add_tokens` for model calls that bypass the wrapped client.
"""

from __future__ import annotations

import contextvars
import dataclasses
import functools
import hashlib
import inspect
import json
import logging
import math
import os
import re
import secrets
import tempfile
import threading
import time
from collections import deque
from collections.abc import Iterable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, date, datetime
from datetime import time as dt_time
from decimal import Decimal
from enum import Enum
from pathlib import Path, PurePath
from typing import Any, Callable, TypeVar

from insurance_claims.observability.redaction import Redactor

__all__ = [
    "MAX_STRING_CHARS",
    "TRUNCATION_MARK",
    "add_tokens",
    "annotate",
    "configure",
    "current_span",
    "event",
    "is_enabled",
    "set_output",
    "span",
    "summarize",
    "traced",
    "walk",
    "wrap_openai",
    "write_failures",
]

_log = logging.getLogger(__name__)

MAX_STRING_CHARS = 4000
"""Strings longer than this are cut in the written file."""
TRUNCATION_MARK = "...[truncated]"
CAPTURE_MAX_STRING = 65_536
"""Strings are cut to this length when captured (bounds memory before the write-time cut)."""
CAPTURE_MAX_ITEMS = 2000
"""Containers keep at most this many items when captured."""
MAX_CHILDREN = 1000
"""A span keeps at most this many children; extra ones are counted in ``dropped_children``."""
_WRITE_MAX_ITEMS = 10 * CAPTURE_MAX_ITEMS
"""Write-time cap for containers assigned to a span directly (captured ones are already capped)."""
_MAX_DEPTH = 32
"""Nesting kept when a value is captured."""
_WRITE_MAX_DEPTH = 64
"""Nesting kept for the whole tree at write time (spans plus their captured values)."""
_OMITTED = "[omitted]"
_TRACED_FLAG = "__insurance_claims_traced__"
_SAFE_ERROR_ATTR = "_insurance_claims_trace_error"
"""Set on provider exceptions so every enclosing span records the sanitized description."""

write_failures = 0
"""Number of trace files that could not be written (read it as ``tracing.write_failures``)."""

F = TypeVar("F", bound=Callable[..., Any])


@dataclass(frozen=True)
class _Config:
    trace_dir: Path
    enabled: bool
    redactor: Redactor


_CONFIG = _Config(trace_dir=Path("traces"), enabled=False, redactor=Redactor())
_STATE_LOCK = threading.Lock()
_current: contextvars.ContextVar[dict[str, Any] | None] = contextvars.ContextVar("insurance_claims_current_span", default=None)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


def configure(*, trace_dir: Path, enabled: bool, redactor: Redactor | None = None) -> None:
    """Set where traces go, whether they are written, and how they are redacted.

    Without a ``redactor`` only the generic patterns (emails, keys, phone and SSN
    shapes, contextual PII phrases) apply; pass ``build_fixture_redactor(...)``
    in the application. The directory is created lazily on the first write.
    """
    global _CONFIG
    with _STATE_LOCK:
        _CONFIG = _Config(Path(trace_dir), bool(enabled), redactor if redactor is not None else Redactor())


def is_enabled() -> bool:
    """True when root spans are written to disk."""
    return _CONFIG.enabled


# ---------------------------------------------------------------------------
# Snapshots: plain JSON data captured at record time
# ---------------------------------------------------------------------------


def _snapshot(value: Any, *, max_items: int = CAPTURE_MAX_ITEMS, max_depth: int = _MAX_DEPTH) -> Any:
    """Copy ``value`` into JSON-safe data with size limits. Never raises."""
    try:
        return _snap(value, max_depth, set(), max_items)
    except Exception:
        return "<unserializable value>"


def _snap(value: Any, budget: int, active: set[int], max_items: int) -> Any:
    """``budget`` is the remaining nesting depth."""
    if value is None or isinstance(value, (bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if isinstance(value, str):
        return value if len(value) <= CAPTURE_MAX_STRING else value[:CAPTURE_MAX_STRING] + TRUNCATION_MARK
    if isinstance(value, Enum):
        return _snap(value.value, budget, active, max_items)
    if isinstance(value, (datetime, date, dt_time)):
        return value.isoformat()
    if isinstance(value, (Decimal, PurePath)):
        return str(value)
    if isinstance(value, (bytes, bytearray, memoryview)):
        return f"<{len(value)} bytes>"
    if isinstance(value, BaseException):
        return _describe_error(value)
    if budget <= 0:
        return "[depth-limit]"
    if id(value) in active:
        return "[cycle]"
    active.add(id(value))
    try:
        return _snap_container(value, budget - 1, active, max_items)
    except Exception:
        return f"<unserializable {value.__class__.__name__}>"
    finally:
        active.discard(id(value))


def _sorted_safely(items: Iterable[Any]) -> list[Any]:
    try:
        return sorted(items, key=repr)
    except Exception:
        return list(items)


def _snap_container(value: Any, budget: int, active: set[int], max_items: int) -> Any:
    if isinstance(value, Mapping):
        out: dict[str, Any] = {}
        for index, (key, item) in enumerate(list(value.items())):
            if index >= max_items:
                out["[more]"] = f"{len(value) - index} more items"
                break
            out[_key(key)] = _snap(item, budget, active, max_items)
        return out
    if isinstance(value, (list, tuple, set, frozenset, deque)):
        items = _sorted_safely(value) if isinstance(value, (set, frozenset)) else list(value)
        out_list = [_snap(item, budget, active, max_items) for item in items[:max_items]]
        if len(items) > max_items:
            out_list.append(f"[{len(items) - max_items} more items]")
        return out_list
    dump = getattr(value, "model_dump", None)
    if callable(dump) and not isinstance(value, type):
        return _snap(dump(mode="json"), budget, active, max_items)
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        fields = {f.name: getattr(value, f.name, None) for f in dataclasses.fields(value)}
        return _snap(fields, budget, active, max_items)
    text = str(value)
    return text if len(text) <= CAPTURE_MAX_STRING else text[:CAPTURE_MAX_STRING] + TRUNCATION_MARK


def _key(key: Any) -> str:
    if isinstance(key, str):
        return key
    if isinstance(key, Enum):
        return str(key.value)
    try:
        return str(key)
    except Exception:
        return f"<{key.__class__.__name__}>"


def _describe_error(exc: BaseException) -> str:
    safe = getattr(exc, _SAFE_ERROR_ATTR, None)
    if isinstance(safe, str):
        return safe
    try:
        message = str(exc)
    except Exception:
        message = ""
    name = exc.__class__.__name__
    text = f"{name}: {message}" if message else name
    return text if len(text) <= MAX_STRING_CHARS else text[:MAX_STRING_CHARS] + TRUNCATION_MARK


# ---------------------------------------------------------------------------
# Spans
# ---------------------------------------------------------------------------


def _now_iso() -> str:
    now = datetime.now(UTC)
    return now.strftime("%Y-%m-%dT%H:%M:%S.") + f"{now.microsecond // 1000:03d}Z"


def _new_node(name: str, type: str, input: Any, metadata: Mapping[str, Any] | None) -> dict[str, Any]:
    meta = _snapshot(dict(metadata)) if metadata else {}
    return {
        "name": str(name),
        "type": str(type),
        "start": _now_iso(),
        "duration_ms": None,
        "input": _snapshot(input),
        "output": None,
        "metadata": meta if isinstance(meta, dict) else {"value": meta},
        "children": [],
    }


def _attach(parent: dict[str, Any], node: dict[str, Any]) -> None:
    children = parent.get("children")
    if not isinstance(children, list):
        children = []
        parent["children"] = children
    if len(children) >= MAX_CHILDREN:
        parent["dropped_children"] = int(parent.get("dropped_children") or 0) + 1
        return
    children.append(node)


def _restore(token: contextvars.Token[Any], parent: dict[str, Any] | None) -> None:
    try:
        _current.reset(token)
    except (ValueError, RuntimeError):
        # Closed in a different context than it was opened in (e.g. a generator moved
        # across tasks): fall back to the parent so later siblings still nest correctly.
        _current.set(parent)


@contextmanager
def span(name: str, type: str = "function", input: Any = None, metadata: dict | None = None) -> Iterator[dict]:
    """Open a span; it nests under the current span, or becomes a root that writes one file."""
    node = _new_node(name, type, input, metadata)
    parent = _current.get()
    if parent is not None:
        _attach(parent, node)
    token = _current.set(node)
    started = time.perf_counter()
    try:
        yield node
    except BaseException as exc:
        if not isinstance(exc, GeneratorExit) and not node.get("error"):
            node["error"] = _describe_error(exc)
        raise
    finally:
        _restore(token, parent)
        node["duration_ms"] = round((time.perf_counter() - started) * 1000, 3)
        if parent is None:
            _write(node)


def _bind_input(
    signature: inspect.Signature | None, args: tuple, kwargs: dict, skip: frozenset[str], omit: frozenset[str]
) -> dict[str, Any]:
    """Call arguments by parameter name (``self``/``cls`` dropped, excluded names omitted)."""
    if signature is None:
        return {"args": list(args), "kwargs": dict(kwargs)}
    try:
        bound = signature.bind(*args, **kwargs)
        bound.apply_defaults()
    except TypeError:
        return {"bind_error": "arguments do not match the signature"}
    return {key: (_OMITTED if key in omit else value) for key, value in bound.arguments.items() if key not in skip}


def _signature_of(fn: Callable[..., Any]) -> inspect.Signature | None:
    try:
        return inspect.signature(fn)
    except (TypeError, ValueError):
        return None


def traced(
    func: Callable[..., Any] | None = None,
    *,
    type: str = "function",
    name: str | None = None,
    exclude: Iterable[str] = (),
    capture_output: bool = True,
) -> Any:
    """Decorate a sync or async function so each call is a span.

    Arguments are bound by name into ``input`` (``self``/``cls`` skipped; names in
    ``exclude`` recorded as ``"[omitted]"``). The return value becomes ``output``
    unless the function already called :func:`set_output` or ``capture_output`` is False.
    """
    omit = frozenset(exclude)

    def decorate(fn: F) -> F:
        span_name = str(name or getattr(fn, "__name__", "call"))
        signature = _signature_of(fn)
        skip = frozenset({"self", "cls"})

        if inspect.iscoroutinefunction(fn):

            @functools.wraps(fn)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                with span(span_name, type=type, input=_bind_input(signature, args, kwargs, skip, omit)) as node:
                    result = await fn(*args, **kwargs)
                    _record_return(node, result, capture_output)
                    return result

            return async_wrapper  # type: ignore[return-value]

        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            with span(span_name, type=type, input=_bind_input(signature, args, kwargs, skip, omit)) as node:
                result = fn(*args, **kwargs)
                _record_return(node, result, capture_output)
                return result

        return wrapper  # type: ignore[return-value]

    if isinstance(func, str):  # tolerate @traced("name")
        name, func = func, None
    return decorate(func) if func is not None else decorate


def _record_return(node: dict[str, Any], result: Any, capture_output: bool) -> None:
    if not capture_output:
        if node.get("output") is None:
            node["output"] = _OMITTED
        return
    if node.get("output") is None:
        node["output"] = _snapshot(result)


def current_span() -> dict | None:
    """The innermost open span in this context, or None."""
    return _current.get()


def annotate(**fields: Any) -> None:
    """Merge ``fields`` into the current span's ``metadata`` (no-op outside a span)."""
    node = _current.get()
    if node is None:
        return
    metadata = node.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}
        node["metadata"] = metadata
    for key, value in fields.items():
        metadata[key] = _snapshot(value)


def set_output(value: Any) -> None:
    """Set the current span's ``output`` (no-op outside a span)."""
    node = _current.get()
    if node is not None:
        node["output"] = _snapshot(value)


def event(name: str, type: str = "guard", **data: Any) -> None:
    """Record a zero-duration child span whose ``output`` is ``data``.

    Outside any span the event is its own root and is written as a small file.
    """
    node = _new_node(name, type, None, None)
    node["duration_ms"] = 0.0
    node["output"] = _snapshot(data)
    parent = _current.get()
    if parent is None:
        _write(node)
    else:
        _attach(parent, node)


def _as_count(value: Any) -> int:
    if isinstance(value, bool):
        return 0
    if isinstance(value, int):
        return max(value, 0)
    if isinstance(value, float) and math.isfinite(value):
        return max(int(value), 0)
    return 0


def add_tokens(input: int, output: int, total: int) -> None:
    """Accumulate token usage into the current span's ``tokens`` (no-op outside a span)."""
    node = _current.get()
    if node is None:
        return
    tokens = node.get("tokens")
    if not isinstance(tokens, dict):
        tokens = {"input": 0, "output": 0, "total": 0}
        node["tokens"] = tokens
    for key, value in (("input", input), ("output", output), ("total", total)):
        tokens[key] = _as_count(tokens.get(key)) + _as_count(value)


# ---------------------------------------------------------------------------
# OpenAI wrapper (Responses API): metadata only, never content
# ---------------------------------------------------------------------------


def _get(obj: Any, key: str) -> Any:
    if obj is None:
        return None
    if isinstance(obj, Mapping):
        return obj.get(key)
    return getattr(obj, key, None)


def _short(value: Any, limit: int = 64) -> str | None:
    if value is None:
        return None
    text = value if isinstance(value, str) else str(value)
    return text[:limit]


def _sha256_of(value: Any) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        value = json.dumps(_snapshot(value), sort_keys=True, default=str)
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _tool_names(tools: Any) -> list[str]:
    names: list[str] = []
    if not isinstance(tools, (list, tuple)):
        return names
    for tool in tools[:64]:
        label = _get(tool, "name") or _get(_get(tool, "function"), "name") or _get(tool, "type")
        if label:
            names.append(_short(label) or "")
    return names


def _reasoning_meta(reasoning: Any) -> dict[str, str] | None:
    if reasoning is None:
        return None
    out = {key: _short(_get(reasoning, key)) for key in ("effort", "summary")}
    return {key: value for key, value in out.items() if value is not None}


def _tool_choice_meta(choice: Any) -> str | None:
    if choice is None or isinstance(choice, str):
        return choice
    name = _get(choice, "name") or _get(_get(choice, "function"), "name")
    return _short(f"{_get(choice, 'type')}:{name}" if name else _get(choice, "type"))


def _input_shape(value: Any) -> dict[str, Any]:
    if value is None:
        return {"input_items": 0}
    if isinstance(value, str):
        return {"input_items": 1, "input_item_types": {"text": 1}}
    if not isinstance(value, (list, tuple)):
        return {"input_items": 1}
    counts: dict[str, int] = {}
    for item in value:
        kind = _get(item, "type") or ("message" if _get(item, "role") else "unknown")
        label = _short(kind, 40) or "unknown"
        counts[label] = counts.get(label, 0) + 1
    return {"input_items": len(value), "input_item_types": counts}


def _request_metadata(kwargs: Mapping[str, Any]) -> dict[str, Any]:
    meta: dict[str, Any] = {
        "model": _short(kwargs.get("model")),
        "max_output_tokens": kwargs.get("max_output_tokens") if isinstance(kwargs.get("max_output_tokens"), int) else None,
        "reasoning": _reasoning_meta(kwargs.get("reasoning")),
        "tool_names": _tool_names(kwargs.get("tools")),
        "prompt_sha256": _sha256_of(kwargs.get("instructions")),
        "tool_choice": _tool_choice_meta(kwargs.get("tool_choice")),
        "stream": bool(kwargs.get("stream")),
    }
    for flag in ("parallel_tool_calls", "store"):
        if isinstance(kwargs.get(flag), bool):
            meta[flag] = kwargs[flag]
    meta.update(_input_shape(kwargs.get("input")))
    return meta


def _record_response(node: dict[str, Any], response: Any) -> None:
    usage = _get(response, "usage")
    if usage is not None:
        tokens = {key: _as_count(_get(usage, f"{key}_tokens")) for key in ("input", "output", "total")}
        if not tokens["total"]:
            tokens["total"] = tokens["input"] + tokens["output"]
        node["tokens"] = tokens
        metadata = node.setdefault("metadata", {})
        metadata["reasoning_tokens"] = _as_count(_get(_get(usage, "output_tokens_details"), "reasoning_tokens"))
        metadata["cached_tokens"] = _as_count(_get(_get(usage, "input_tokens_details"), "cached_tokens"))
    items = _get(response, "output")
    items = list(items) if isinstance(items, (list, tuple)) else []
    item_types = [_short(_get(item, "type"), 40) or "unknown" for item in items]
    calls = [_short(_get(item, "name")) or "" for item in items if _get(item, "type") == "function_call"]
    parts = [part for item in items if _get(item, "type") == "message" for part in (_get(item, "content") or [])]
    text_chars = sum(len(_get(part, "text") or "") for part in parts if _get(part, "type") == "output_text")
    output: dict[str, Any] = {
        "status": _short(_get(response, "status"), 40),
        "item_types": item_types,
        "function_calls": calls,
        "has_refusal": any(_get(part, "type") == "refusal" for part in parts),
        "text_chars": text_chars,
    }
    reason = _get(_get(response, "incomplete_details"), "reason")
    if reason is not None:
        output["incomplete_reason"] = _short(reason, 40)
    node["output"] = output
    model = _get(response, "model")
    if isinstance(model, str):
        node.setdefault("metadata", {})["response_model"] = _short(model)


def _safe_api_error(exc: BaseException) -> str:
    """Exception class and status only: provider messages can echo request content."""
    status = getattr(exc, "status_code", None)
    code = getattr(exc, "code", None)
    parts = [exc.__class__.__name__]
    if isinstance(status, int):
        parts.append(f"status={status}")
    if isinstance(code, str) and re.fullmatch(r"[a-z0-9_.\-]{1,64}", code):
        parts.append(f"code={code}")
    return " ".join(parts)


def _mark_api_error(node: dict[str, Any], exc: BaseException) -> None:
    safe = _safe_api_error(exc)
    node["error"] = safe
    try:
        setattr(exc, _SAFE_ERROR_ATTR, safe)
    except Exception:
        pass


def _llm_span_start(span_name: str, kwargs: Mapping[str, Any]) -> Any:
    try:
        metadata = _request_metadata(kwargs)
    except Exception:
        metadata = {"metadata_error": True}
    return span(span_name, type="llm", input=None, metadata=metadata)


def _after_call(node: dict[str, Any], response: Any, kwargs: Mapping[str, Any]) -> None:
    try:
        if kwargs.get("stream"):
            node["output"] = {"status": "stream_not_captured"}
        else:
            _record_response(node, response)
    except Exception:
        node["output"] = {"status": "unrecorded"}


def _wrap_method(original: Callable[..., Any], span_name: str) -> Callable[..., Any]:
    if inspect.iscoroutinefunction(original):

        @functools.wraps(original)
        async def async_call(*args: Any, **kwargs: Any) -> Any:
            with _llm_span_start(span_name, kwargs) as node:
                try:
                    response = await original(*args, **kwargs)
                except BaseException as exc:
                    _mark_api_error(node, exc)
                    raise
                _after_call(node, response, kwargs)
                return response

        wrapped: Callable[..., Any] = async_call
    else:

        @functools.wraps(original)
        def call(*args: Any, **kwargs: Any) -> Any:
            with _llm_span_start(span_name, kwargs) as node:
                try:
                    response = original(*args, **kwargs)
                except BaseException as exc:
                    _mark_api_error(node, exc)
                    raise
                _after_call(node, response, kwargs)
                return response

        wrapped = call
    setattr(wrapped, _TRACED_FLAG, True)
    return wrapped


def wrap_openai(client: Any) -> Any:
    """Trace ``client.responses.create`` as ``llm`` spans.

    Idempotent: wrapping an already wrapped client changes nothing.
    """
    responses = getattr(client, "responses", None)
    if responses is None:
        return client
    original = getattr(responses, "create", None)
    if original is None or not callable(original) or getattr(original, _TRACED_FLAG, False):
        return client
    try:
        responses.create = _wrap_method(original, "responses.create")
    except (AttributeError, TypeError):
        _log.warning("could not wrap responses.create for tracing")
    return client


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------


def _truncate(value: Any) -> Any:
    if isinstance(value, str):
        return value if len(value) <= MAX_STRING_CHARS else value[:MAX_STRING_CHARS] + TRUNCATION_MARK
    if isinstance(value, dict):
        return {key: _truncate(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_truncate(item) for item in value]
    return value


def _strings(value: Any, out: set[str]) -> set[str]:
    if isinstance(value, str):
        out.add(value)
    elif isinstance(value, dict):
        for item in value.values():
            _strings(item, out)
    elif isinstance(value, list):
        for item in value:
            _strings(item, out)
    return out


_JSON_STRING = re.compile(r'"(?:[^"\\]|\\.)*"')


def _scrub_serialized(text: str, redactor: Redactor, already_clean: set[str]) -> str:
    """Final safety net: redact every JSON string token (keys included), keeping the JSON valid."""

    def replace(match: re.Match[str]) -> str:
        raw = match.group(0)
        value = json.loads(raw) if "\\" in raw else raw[1:-1]
        if value in already_clean:
            return raw
        cleaned = redactor.redact_text(value)
        return raw if cleaned == value else json.dumps(cleaned, ensure_ascii=False)

    return _JSON_STRING.sub(replace, text)


def _fallback(obj: Any) -> Any:
    dump = getattr(obj, "model_dump", None)
    if callable(dump):
        try:
            return dump(mode="json", exclude_none=True)
        except Exception:
            pass
    return str(obj)


def _file_name(node: Mapping[str, Any], redactor: Redactor) -> str:
    stamp = str(node.get("start") or _now_iso()).replace(":", "-")
    label = re.sub(r"[^A-Za-z0-9_.\-]+", "_", redactor.redact_text(str(node.get("name") or "trace")))
    label = label.strip("._")[:64] or "trace"
    return f"{stamp}_{label}_{secrets.token_hex(3)}.json"


_NODE_ORDER = ("name", "type", "start", "duration_ms", "input", "output", "metadata", "tokens", "error", "dropped_children")


def _ordered(node: Mapping[str, Any], depth: int = 0) -> dict[str, Any]:
    """Copy a span tree with keys in the documented order and ``children`` last."""
    out = {key: node[key] for key in _NODE_ORDER if key in node}
    out.update({key: value for key, value in node.items() if key not in out and key != "children"})
    children = node.get("children")
    if depth >= _WRITE_MAX_DEPTH or not isinstance(children, list):
        out["children"] = []
    else:
        out["children"] = [_ordered(child, depth + 1) if isinstance(child, Mapping) else child for child in children]
    return out


def _render(node: dict[str, Any], redactor: Redactor) -> str:
    plain = _snapshot(_ordered(node), max_items=_WRITE_MAX_ITEMS, max_depth=_WRITE_MAX_DEPTH)
    redacted = _truncate(redactor.redact(plain))
    text = json.dumps(redacted, indent=2, ensure_ascii=False, default=_fallback)
    text = _scrub_serialized(text, redactor, _strings(redacted, set()))
    json.loads(text)  # never write a file that is not valid JSON
    return text


def _atomic_write(directory: Path, file_name: str, text: str) -> Path:
    existed = directory.is_dir()
    directory.mkdir(parents=True, exist_ok=True)
    if not existed:
        try:
            os.chmod(directory, 0o700)
        except OSError:
            pass
    fd, tmp_name = tempfile.mkstemp(prefix=".trace-", suffix=".tmp", dir=directory)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
        os.chmod(tmp_name, 0o600)
        target = directory / file_name
        os.replace(tmp_name, target)
        return target
    except BaseException:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def _write(node: dict[str, Any]) -> Path | None:
    """Redact, truncate, serialize and atomically write one root span. Never raises."""
    global write_failures
    config = _CONFIG
    if not config.enabled:
        return None
    try:
        text = _render(node, config.redactor)
        return _atomic_write(config.trace_dir, _file_name(node, config.redactor), text)
    except Exception as exc:
        with _STATE_LOCK:
            write_failures += 1
        _log.warning("trace write failed (%s)", exc.__class__.__name__)
        return None


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------


def walk(node: dict) -> Iterator[dict]:
    """Yield ``node`` and every descendant span in pre-order (file order)."""
    stack: list[Any] = [node]
    while stack:
        current = stack.pop()
        if not isinstance(current, dict):
            continue
        yield current
        children = current.get("children")
        if isinstance(children, list):
            stack.extend(reversed(children))


def _dicts_of(node: Mapping[str, Any], *, with_input: bool = False) -> list[Mapping[str, Any]]:
    parts = [node.get("input")] if with_input else []
    parts += [node.get("metadata"), node.get("output")]
    return [part for part in parts if isinstance(part, Mapping)]


def _text(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None


def summarize(trace: dict) -> dict:
    """Tool path, guard decisions, token totals, LLM call count, stop reasons, phases, errors."""
    summary: dict[str, Any] = {
        "tool_path": [],
        "guards": [],
        "tokens": {"input": 0, "output": 0, "total": 0},
        "llm_calls": 0,
        "stop_reasons": [],
        "phases": [],
        "errors": [],
    }
    if not isinstance(trace, dict):
        return summary
    error_index: dict[str, int] = {}
    for node in walk(trace):
        _summarize_node(node, summary, error_index)
    summary["phases"] = _phase_sequence(trace)
    return summary


def _summarize_node(node: Mapping[str, Any], summary: dict[str, Any], error_index: dict[str, int]) -> None:
    kind, name = node.get("type"), node.get("name")
    if kind == "tool":
        summary["tool_path"].append(name)
    elif kind == "guard":
        summary["guards"].append({"name": name, "output": node.get("output")})
    elif kind == "llm":
        summary["llm_calls"] += 1
    tokens = node.get("tokens")
    if isinstance(tokens, Mapping):
        for key in ("input", "output", "total"):
            summary["tokens"][key] += _as_count(tokens.get(key))
    for source in _dicts_of(node):
        reason = _text(source.get("stop_reason"))
        if reason and reason not in summary["stop_reasons"]:
            summary["stop_reasons"].append(reason)
    error = node.get("error")
    if isinstance(error, str) and error:
        entry = {"name": name, "error": error}
        if error in error_index:  # same exception seen higher up: keep the innermost span
            summary["errors"][error_index[error]] = entry
        else:
            error_index[error] = len(summary["errors"])
            summary["errors"].append(entry)


def _phase_sequence(trace: Mapping[str, Any]) -> list[str]:
    """Phases visited: the starting phase, every ``from_phase -> to_phase`` transition, then the final phase.

    ``phase_before`` (input, metadata or output) marks where a span started; ``phase`` and
    ``phase_after`` (metadata or output) mark where it ended, with the root's value taken as final.
    """
    start: str | None = None
    transitions: list[tuple[str, str]] = []
    ends: list[str] = []
    for node in walk(dict(trace)):
        for source in _dicts_of(node, with_input=True):
            start = start or _text(source.get("phase_before"))
        for source in _dicts_of(node):
            origin, target = _text(source.get("from_phase")), _text(source.get("to_phase"))
            if origin and target:
                transitions.append((origin, target))
            ends += [v for v in (_text(source.get("phase")), _text(source.get("phase_after"))) if v]
    sequence: list[str] = []

    def add(value: str | None) -> None:
        if value and (not sequence or sequence[-1] != value):
            sequence.append(value)

    add(start or (transitions[0][0] if transitions else None))
    for origin, target in transitions:
        add(origin)
        add(target)
    root_end = [v for src in _dicts_of(trace) for v in (_text(src.get("phase_after")), _text(src.get("phase"))) if v]
    add(root_end[0] if root_end else (ends[-1] if ends else None))
    return sequence
