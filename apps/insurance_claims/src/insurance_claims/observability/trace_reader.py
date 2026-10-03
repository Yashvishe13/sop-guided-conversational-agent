"""Print what happened in saved traces: phases, stop reasons, guards, tool path, LLM calls, tokens, errors.

    python -m insurance_claims.observability.trace_reader traces/            # every trace in the directory
    python -m insurance_claims.observability.trace_reader traces/ --last 3   # the three newest
    python -m insurance_claims.observability.trace_reader one_trace.json --json

Trace files are already redacted when written; output is passed through the
generic redaction patterns once more in case the reader is pointed at other
JSON files. Exit codes: 0 success, 1 some files unreadable, 2 bad arguments
or missing path.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence, TextIO

from insurance_claims.observability.redaction import Redactor
from insurance_claims.observability.tracing import summarize

__all__ = ["format_summary", "load_trace", "main", "trace_files"]

_GUARD_OUTPUT_CHARS = 200
_GENERIC = Redactor()


class TraceReadError(Exception):
    """A trace file could not be read or is not a span tree."""


def trace_files(path: Path, last: int | None = None) -> list[Path]:
    """Trace files under ``path`` (a file or a directory), oldest first; temp files skipped."""
    if path.is_file():
        return [path]
    # Names start with a millisecond timestamp; break same-millisecond ties by write order.
    files = sorted(
        (p for p in path.glob("*.json") if p.is_file() and not p.name.startswith(".")),
        key=lambda p: (p.name[:24], p.stat().st_mtime_ns, p.name),
    )
    return files[-last:] if last else files


def load_trace(path: Path) -> dict[str, Any]:
    """Load one trace file; raise :class:`TraceReadError` when it is not a JSON span tree."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, ValueError) as exc:
        raise TraceReadError(exc.__class__.__name__) from exc
    if not isinstance(data, dict) or "name" not in data:
        raise TraceReadError("not a span tree")
    return data


def _compact(value: Any) -> str:
    text = json.dumps(value, sort_keys=True, ensure_ascii=False, default=str)
    return text if len(text) <= _GUARD_OUTPUT_CHARS else text[:_GUARD_OUTPUT_CHARS] + "..."


def format_summary(name: str, trace: dict[str, Any], summary: dict[str, Any]) -> list[str]:
    """Human-readable lines for one trace."""
    tokens = summary["tokens"]
    lines = [
        f"== {name}",
        f"   root: {trace.get('name')} ({trace.get('type')})  start {trace.get('start')}  duration {trace.get('duration_ms')} ms",
        f"   phases: {' -> '.join(summary['phases']) or 'none recorded'}",
        f"   stop reasons: {', '.join(summary['stop_reasons']) or 'none recorded'}",
        f"   guards ({len(summary['guards'])}):",
    ]
    lines += [f"     - {g['name']}: {_compact(g['output'])}" for g in summary["guards"]] or ["     (none)"]
    lines += [
        f"   tool path: {' -> '.join(str(t) for t in summary['tool_path']) or '(no tool calls)'}",
        f"   llm calls: {summary['llm_calls']}",
        f"   tokens: input {tokens['input']}, output {tokens['output']}, total {tokens['total']}",
        f"   errors ({len(summary['errors'])}):",
    ]
    lines += [f"     - {e['name']}: {e['error']}" for e in summary["errors"]] or ["     (none)"]
    return lines


def _totals(summaries: list[dict[str, Any]]) -> str:
    tokens = {key: sum(s["tokens"][key] for s in summaries) for key in ("input", "output", "total")}
    llm = sum(s["llm_calls"] for s in summaries)
    errored = sum(1 for s in summaries if s["errors"])
    return (
        f"== totals: {len(summaries)} traces, {llm} llm calls, tokens input {tokens['input']} "
        f"output {tokens['output']} total {tokens['total']}, {errored} with errors"
    )


def _emit(line: str, stream: TextIO) -> None:
    print(_GENERIC.redact_text(line), file=stream)


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="python -m insurance_claims.observability.trace_reader", description=__doc__.splitlines()[0])
    parser.add_argument("path", type=Path, help="a trace file or a directory of trace files")
    parser.add_argument("--last", type=_positive_int, default=None, help="only the N newest traces in a directory")
    parser.add_argument("--json", action="store_true", help="print one JSON summary per line")
    return parser.parse_args(argv)


def _positive_int(raw: str) -> int:
    try:
        value = int(raw)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a positive integer") from exc
    if value < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return value


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point; returns the process exit code."""
    try:
        args = _parse_args(argv)
    except SystemExit as exc:  # argparse exits on --help and on bad arguments
        return int(exc.code) if isinstance(exc.code, int) else 2
    if not args.path.exists():
        print(f"no such file or directory: {args.path}", file=sys.stderr)
        return 2
    files = trace_files(args.path, args.last)
    if not files:
        print(f"no trace files in {args.path}")
        return 0
    summaries: list[dict[str, Any]] = []
    unreadable = 0
    for path in files:
        try:
            trace = load_trace(path)
        except TraceReadError as exc:
            unreadable += 1
            print(f"skipped {path.name}: unreadable trace ({exc})", file=sys.stderr)
            continue
        summary = summarize(trace)
        summaries.append(summary)
        if args.json:
            _emit(json.dumps({"file": path.name, **summary}, ensure_ascii=False, default=str), sys.stdout)
        else:
            for line in format_summary(path.name, trace, summary):
                _emit(line, sys.stdout)
    if not args.json and len(summaries) > 1:
        _emit(_totals(summaries), sys.stdout)
    return 1 if unreadable else 0


if __name__ == "__main__":
    raise SystemExit(main())
