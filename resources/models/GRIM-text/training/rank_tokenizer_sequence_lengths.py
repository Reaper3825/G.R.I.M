#!/usr/bin/env python3
"""Rank concept block rendered text lengths for tokenizer/Viterbi capacity checks.

This replaces the one-off terminal paste that generated
``training/logs/tokenizer_sequence_length_rankings.txt``.  It supports bounded
line ranges and prints periodic progress heartbeats so long runs show whether
they are still advancing.
"""

from __future__ import annotations

import argparse
import heapq
import json
import math
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


SPACE = 0x20
SPIECE_LEN = 3
I64_MIN = -(2**63)
I64_MAX = 2**63 - 1

# Mirrors the raw numeric detector behavior used by the earlier diagnostic:
# - floats require a dot or exponent
# - ints reject a following '.' or an exponent marker with a digit/sign payload
# - there is intentionally no word-boundary requirement
NUMERIC_RE = re.compile(
    rb"[+-]?(?:(?:\d+\.\d*)|(?:\.\d+)|(?:\d+(?:\.\d*)?[eE][+-]?\d+)|(?:\.\d+[eE][+-]?\d+))"
    rb"|[+-]?\d+(?!\.)(?!(?:[eE](?:[+-]|\d)))"
)
FLOAT_MARKERS = (b".", b"e", b"E")
WHITESPACE_RE = re.compile(r"\s+")


@dataclass(frozen=True)
class CurriculumSelection:
    name: str | None
    format_as_concept: bool | None
    selected_ids: set[str] | None
    plaintext_ids: set[str]
    concept_ids: set[str]


@dataclass(frozen=True)
class Metrics:
    raw_bytes: int
    norm_bytes: int
    spaces: int
    longest_segment: int
    numeric_atoms: int
    unparseable: int


class Progress:
    def __init__(self, total_target_lines: int | None, line_interval: int, seconds_interval: float) -> None:
        if line_interval <= 0:
            raise ValueError("progress line interval must be positive")
        if seconds_interval <= 0:
            raise ValueError("progress seconds interval must be positive")
        self.total_target_lines = total_target_lines
        self.line_interval = line_interval
        self.seconds_interval = seconds_interval
        self.started = time.monotonic()
        self.last_emit = self.started
        self.last_line = 0

    def maybe_emit(self, line_no: int, accepted: int, skipped: int, forced: bool = False) -> None:
        now = time.monotonic()
        line_delta = line_no - self.last_line
        elapsed_since_emit = now - self.last_emit
        if not forced and line_delta < self.line_interval and elapsed_since_emit < self.seconds_interval:
            return

        elapsed = max(now - self.started, 1e-9)
        rate = accepted / elapsed
        scanned_rate = line_no / elapsed
        pct = ""
        eta = ""
        if self.total_target_lines is not None and self.total_target_lines > 0:
            done = min(line_no, self.total_target_lines)
            pct = f" {100.0 * done / self.total_target_lines:6.2f}%"
            remaining = max(self.total_target_lines - done, 0)
            if scanned_rate > 0:
                eta = f" eta={format_duration(remaining / scanned_rate)}"

        print(
            f"[progress] scanned={line_no:,}{pct} accepted={accepted:,} skipped={skipped:,} "
            f"accepted_rate={rate:,.1f}/s scanned_rate={scanned_rate:,.1f}/s elapsed={format_duration(elapsed)}{eta}",
            file=sys.stderr,
            flush=True,
        )
        self.last_emit = now
        self.last_line = line_no


def format_duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, rem = divmod(seconds, 3600)
    minutes, sec = divmod(rem, 60)
    if hours:
        return f"{hours:d}h{minutes:02d}m{sec:02d}s"
    return f"{minutes:d}m{sec:02d}s" if minutes else f"{sec:d}s"


def default_concept_path() -> Path:
    return Path(__file__).resolve().parent / "data" / "concept_blocks.jsonl"


def default_registry_path() -> Path:
    return Path(__file__).resolve().parent / "data" / "curriculum_registry.json"


def default_output_path(start_line: int, end_line: int | None, mode: str) -> Path:
    suffix_end = str(end_line) if end_line is not None else "end"
    suffix_mode = "both" if mode == "both" else mode
    return Path(__file__).resolve().parent / "logs" / f"tokenizer_sequence_length_rankings_{suffix_mode}_{start_line}_{suffix_end}.txt"


def load_curriculum_selection(registry_path: Path, curriculum_name: str | None) -> CurriculumSelection:
    if curriculum_name is None:
        return CurriculumSelection(
            name=None,
            format_as_concept=None,
            selected_ids=None,
            plaintext_ids=set(),
            concept_ids=set(),
        )

    with registry_path.open("r", encoding="utf-8") as f:
        reg = json.load(f)

    for cur in reg.get("curriculums", []):
        if cur.get("name") != curriculum_name:
            continue

        format_as_concept = bool(cur.get("format_as_concept", True))
        concept_ids = {x for x in cur.get("concept_block_ids", []) if isinstance(x, str)}
        plaintext_ids = {x for x in cur.get("plaintext_block_ids", []) if isinstance(x, str)}
        if not format_as_concept:
            plaintext_ids |= concept_ids
            concept_ids = set()
        return CurriculumSelection(
            name=curriculum_name,
            format_as_concept=format_as_concept,
            selected_ids=concept_ids | plaintext_ids,
            plaintext_ids=plaintext_ids,
            concept_ids=concept_ids,
        )

    raise RuntimeError(f"curriculum not found: {curriculum_name}")


def render_plain(j: dict[str, Any]) -> str:
    parts: list[str] = []
    q = j.get("prompt")
    if isinstance(q, str) and q:
        parts.extend((q, "\n"))

    expl = None
    if isinstance(j.get("explanation"), list):
        expl = j.get("explanation")
    elif isinstance(j.get("intermediates"), list):
        expl = j.get("intermediates")
    if expl is not None:
        for s in expl:
            if isinstance(s, str):
                parts.extend((s, "\n"))

    a = j.get("answer")
    if isinstance(a, str) and a:
        parts.extend((a, "\n"))
    return "".join(parts)


def parse_int_ok(raw: bytes) -> bool:
    try:
        value = int(raw)
    except ValueError:
        return False
    return I64_MIN <= value <= I64_MAX


def parse_float_ok(raw: bytes) -> bool:
    try:
        value = float(raw)
    except ValueError:
        return False
    return math.isfinite(value)


def normalized_len(bs: bytes, start: int, end: int) -> int:
    return sum(SPIECE_LEN if b == SPACE else 1 for b in bs[start:end])


def analyze_text(text: str) -> Metrics:
    bs = text.encode("utf-8")
    raw_bytes = len(bs)
    spaces = bs.count(SPACE)
    norm_bytes = SPIECE_LEN + raw_bytes + (SPIECE_LEN - 1) * spaces

    spans: list[tuple[int, int]] = []
    numeric_atoms = 0
    unparseable = 0

    for match in NUMERIC_RE.finditer(bs):
        start, end = match.span()
        raw = match.group(0)
        is_float = any(marker in raw for marker in FLOAT_MARKERS)
        ok = parse_float_ok(raw) if is_float else parse_int_ok(raw)

        if ok:
            spans.append((start, end))
            numeric_atoms += 1
        else:
            unparseable += 1

    longest_segment = 0
    scan_byte = 0
    scan_norm = SPIECE_LEN
    previous_atom_end_norm = 0
    for start, end in spans:
        start_norm = scan_norm + normalized_len(bs, scan_byte, start)
        segment_len = start_norm - previous_atom_end_norm
        if segment_len > longest_segment:
            longest_segment = segment_len
        end_norm = start_norm + normalized_len(bs, start, end)
        scan_byte = end
        scan_norm = end_norm
        previous_atom_end_norm = end_norm

    tail_len = norm_bytes - previous_atom_end_norm
    if tail_len > longest_segment:
        longest_segment = tail_len

    return Metrics(
        raw_bytes=raw_bytes,
        norm_bytes=norm_bytes,
        spaces=spaces,
        longest_segment=longest_segment,
        numeric_atoms=numeric_atoms,
        unparseable=unparseable,
    )


def raw_excerpt(text: str, max_chars: int = 160) -> str:
    t = WHITESPACE_RE.sub(" ", text).strip()
    return t if len(t) <= max_chars else f"{t[:max_chars]}…"


def make_record(j: dict[str, Any], cid: str, line_no: int, metrics: Metrics, text: str) -> dict[str, Any]:
    prompt_value = j.get("prompt")
    prompt = prompt_value[:120] if isinstance(prompt_value, str) else ""
    return {
        "id": cid,
        "line": line_no,
        "raw_bytes": metrics.raw_bytes,
        "spaces": metrics.spaces,
        "norm_bytes": metrics.norm_bytes,
        "longest_segment": metrics.longest_segment,
        "numeric_atoms": metrics.numeric_atoms,
        "unparseable": metrics.unparseable,
        "excerpt": raw_excerpt(text),
        "prompt": prompt,
    }


def heap_wants_largest(heap: list[tuple[int, int, dict[str, Any]]], metric: int, top_n: int) -> bool:
    return len(heap) < top_n or metric > heap[0][0]


def heap_wants_shortest(heap: list[tuple[int, int, dict[str, Any]]], metric: int, top_n: int) -> bool:
    return len(heap) < top_n or metric < -heap[0][0]


def push_largest(heap: list[tuple[int, int, dict[str, Any]]], metric: int, line_no: int, record: dict[str, Any], top_n: int) -> None:
    heapq.heappush(heap, (metric, line_no, record))
    if len(heap) > top_n:
        heapq.heappop(heap)


def push_shortest(heap: list[tuple[int, int, dict[str, Any]]], metric: int, line_no: int, record: dict[str, Any], top_n: int) -> None:
    heapq.heappush(heap, (-metric, -line_no, record))
    if len(heap) > top_n:
        heapq.heappop(heap)


def iter_lines(path: Path, start_line: int, end_line: int | None) -> Iterable[tuple[int, str]]:
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if line_no < start_line:
                continue
            if end_line is not None and line_no > end_line:
                break
            yield line_no, line


def dump_largest(title: str, heap: list[tuple[int, int, dict[str, Any]]], key: str) -> list[str]:
    lines = ["", title, "rank\tmetric\tline\tid\traw_bytes\tnorm_bytes\tspaces\tnumeric_atoms\tunparseable\texcerpt"]
    rows = sorted((x[2] for x in heap), key=lambda r: (r[key], r["line"]), reverse=True)
    for i, r in enumerate(rows, 1):
        excerpt = r["excerpt"].replace("\t", " ").replace("\n", " ")
        lines.append(
            f"{i}\t{r[key]}\t{r['line']}\t{r['id']}\t{r['raw_bytes']}\t{r['norm_bytes']}\t"
            f"{r['spaces']}\t{r['numeric_atoms']}\t{r['unparseable']}\t{excerpt}"
        )
    return lines


def dump_shortest(title: str, heap: list[tuple[int, int, dict[str, Any]]], key: str) -> list[str]:
    lines = ["", title, "rank\tmetric\tline\tid\traw_bytes\tnorm_bytes\tspaces\tnumeric_atoms\tunparseable\texcerpt"]
    rows = sorted((x[2] for x in heap), key=lambda r: (r[key], r["line"]))
    for i, r in enumerate(rows, 1):
        excerpt = r["excerpt"].replace("\t", " ").replace("\n", " ")
        lines.append(
            f"{i}\t{r[key]}\t{r['line']}\t{r['id']}\t{r['raw_bytes']}\t{r['norm_bytes']}\t"
            f"{r['spaces']}\t{r['numeric_atoms']}\t{r['unparseable']}\t{excerpt}"
        )
    return lines


def build_report(
    args: argparse.Namespace,
    selection: CurriculumSelection,
    elapsed: float,
    total_seen: int,
    accepted: int,
    skipped_by_filter: int,
    blank_rows: int,
    parse_errors: int,
    unparseable_total: int,
    longest_heaps: dict[str, list[tuple[int, int, dict[str, Any]]]],
    shortest_heaps: dict[str, list[tuple[int, int, dict[str, Any]]]],
) -> str:
    selected_count = "all"
    if selection.selected_ids is not None:
        selected_count = str(len(selection.selected_ids))

    lines = [
        "GRIM-text tokenizer sequence length ranking",
        f"concept_blocks={args.concept_blocks}",
        f"line_range={args.start_line}-{args.end_line if args.end_line is not None else 'end'}",
        f"curriculum={selection.name if selection.name is not None else '(none)'}",
        f"format_as_concept={selection.format_as_concept if selection.format_as_concept is not None else '(not filtered)'}",
        f"selected_ids={selected_count} plaintext_ids={len(selection.plaintext_ids)} concept_ids={len(selection.concept_ids)}",
        f"scanned_nonblank_rows={total_seen} accepted_rows={accepted} skipped_by_filter={skipped_by_filter} blank_rows={blank_rows}",
        f"parse_errors={parse_errors} unparseable_numeric_spans={unparseable_total}",
        f"elapsed={format_duration(elapsed)} accepted_rate={accepted / max(elapsed, 1e-9):.2f}/s",
        "measurement_notes=raw/plaintext render from question+explanation/intermediates+answer; "
        "norm_bytes = raw_utf8_bytes + 3 + 2*ASCII_spaces; longest_segment splits normalized text around numeric atoms",
    ]

    if args.mode in ("longest", "both"):
        lines.extend(
            dump_largest(
                "TOP_LONGEST_NORMALIZED_NON_ATOM_VITERBI_SEGMENTS",
                longest_heaps["longest_segment"],
                "longest_segment",
            )
        )
        lines.extend(dump_largest("TOP_LONGEST_NORMALIZED_FULL_RENDERED_TEXTS", longest_heaps["norm_bytes"], "norm_bytes"))
        lines.extend(dump_largest("TOP_LONGEST_RAW_RENDERED_TEXTS", longest_heaps["raw_bytes"], "raw_bytes"))

    if args.mode in ("shortest", "both"):
        lines.extend(
            dump_shortest(
                "TOP_SHORTEST_NORMALIZED_NON_ATOM_VITERBI_SEGMENTS",
                shortest_heaps["longest_segment"],
                "longest_segment",
            )
        )
        lines.extend(dump_shortest("TOP_SHORTEST_NORMALIZED_FULL_RENDERED_TEXTS", shortest_heaps["norm_bytes"], "norm_bytes"))
        lines.extend(dump_shortest("TOP_SHORTEST_RAW_RENDERED_TEXTS", shortest_heaps["raw_bytes"], "raw_bytes"))

    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--concept-blocks", type=Path, default=default_concept_path(), help="Path to concept_blocks.jsonl")
    parser.add_argument("--registry", type=Path, default=default_registry_path(), help="Path to curriculum_registry.json")
    parser.add_argument("--curriculum", default=None, help="Optional curriculum name used to filter concept block IDs")
    parser.add_argument("--start-line", type=int, default=1, help="First JSONL line to scan, inclusive")
    parser.add_argument("--end-line", type=int, default=None, help="Last JSONL line to scan, inclusive")
    parser.add_argument("--top-n", type=int, default=30, help="Number of records in each ranking table")
    parser.add_argument("--mode", choices=("longest", "shortest", "both"), default="longest", help="Which ranking tables to emit")
    parser.add_argument("--output", type=Path, default=None, help="Report output path")
    parser.add_argument("--progress-lines", type=int, default=10_000, help="Emit progress after this many scanned lines")
    parser.add_argument("--progress-seconds", type=float, default=15.0, help="Emit progress after this many seconds")
    parser.add_argument("--fail-on-json-error", action="store_true", help="Abort instead of counting malformed JSON rows")
    args = parser.parse_args()

    if args.start_line < 1:
        raise ValueError("--start-line must be >= 1")
    if args.end_line is not None and args.end_line < args.start_line:
        raise ValueError("--end-line must be >= --start-line")
    if args.top_n <= 0:
        raise ValueError("--top-n must be positive")
    if args.output is None:
        args.output = default_output_path(args.start_line, args.end_line, args.mode)
    return args


def main() -> int:
    args = parse_args()
    args.concept_blocks = args.concept_blocks.resolve()
    args.registry = args.registry.resolve()
    args.output = args.output.resolve()

    if not args.concept_blocks.exists():
        raise FileNotFoundError(args.concept_blocks)
    if args.curriculum is not None and not args.registry.exists():
        raise FileNotFoundError(args.registry)

    selection = load_curriculum_selection(args.registry, args.curriculum)
    target_lines = None
    if args.end_line is not None:
        target_lines = args.end_line - args.start_line + 1
    progress = Progress(target_lines, args.progress_lines, args.progress_seconds)

    longest_heaps: dict[str, list[tuple[int, int, dict[str, Any]]]] = {
        "longest_segment": [],
        "norm_bytes": [],
        "raw_bytes": [],
    }
    shortest_heaps: dict[str, list[tuple[int, int, dict[str, Any]]]] = {
        "longest_segment": [],
        "norm_bytes": [],
        "raw_bytes": [],
    }

    total_seen = 0
    accepted = 0
    skipped_by_filter = 0
    blank_rows = 0
    parse_errors = 0
    unparseable_total = 0
    last_line_no = args.start_line - 1
    started = time.monotonic()

    range_end_label = str(args.end_line) if args.end_line is not None else "end"
    curriculum_label = "(none)"
    if args.curriculum is not None:
        curriculum_label = args.curriculum
    print(
        f"[start] concept_blocks={args.concept_blocks} range={args.start_line}-{range_end_label} "
        f"curriculum={curriculum_label} mode={args.mode} top_n={args.top_n}",
        file=sys.stderr,
        flush=True,
    )

    for line_no, line in iter_lines(args.concept_blocks, args.start_line, args.end_line):
        last_line_no = line_no - args.start_line + 1
        if not line.strip():
            blank_rows += 1
            progress.maybe_emit(last_line_no, accepted, skipped_by_filter)
            continue

        total_seen += 1
        try:
            j = json.loads(line)
        except json.JSONDecodeError:
            parse_errors += 1
            if args.fail_on_json_error:
                raise
            progress.maybe_emit(last_line_no, accepted, skipped_by_filter)
            continue

        cid_value = j.get("id", "")
        cid = cid_value if isinstance(cid_value, str) else ""
        if selection.selected_ids is not None and cid not in selection.selected_ids:
            skipped_by_filter += 1
            progress.maybe_emit(last_line_no, accepted, skipped_by_filter)
            continue

        text = render_plain(j)
        metrics = analyze_text(text)
        accepted += 1
        unparseable_total += metrics.unparseable

        record: dict[str, Any] | None = None

        def get_record() -> dict[str, Any]:
            nonlocal record
            if record is None:
                record = make_record(j, cid, line_no, metrics, text)
            return record

        if args.mode in ("longest", "both"):
            for key, metric in (
                ("longest_segment", metrics.longest_segment),
                ("norm_bytes", metrics.norm_bytes),
                ("raw_bytes", metrics.raw_bytes),
            ):
                heap = longest_heaps[key]
                if heap_wants_largest(heap, metric, args.top_n):
                    push_largest(heap, metric, line_no, get_record(), args.top_n)

        if args.mode in ("shortest", "both"):
            for key, metric in (
                ("longest_segment", metrics.longest_segment),
                ("norm_bytes", metrics.norm_bytes),
                ("raw_bytes", metrics.raw_bytes),
            ):
                heap = shortest_heaps[key]
                if heap_wants_shortest(heap, metric, args.top_n):
                    push_shortest(heap, metric, line_no, get_record(), args.top_n)

        progress.maybe_emit(last_line_no, accepted, skipped_by_filter)

    elapsed = time.monotonic() - started
    progress.maybe_emit(last_line_no, accepted, skipped_by_filter, forced=True)

    report = build_report(
        args,
        selection,
        elapsed,
        total_seen,
        accepted,
        skipped_by_filter,
        blank_rows,
        parse_errors,
        unparseable_total,
        longest_heaps,
        shortest_heaps,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report, encoding="utf-8")
    print(f"[done] wrote {args.output} accepted={accepted:,} elapsed={format_duration(elapsed)}", file=sys.stderr, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())