#!/usr/bin/env python3
"""Rank GRIM-text tokenizer corpus lengths for concept_blocks.jsonl line range 1..500185."""

from __future__ import annotations

import json
import math
import re
from heapq import heappop, heappush
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "resources" / "models" / "GRIM-text" / "training" / "data"
LOG_DIR = ROOT / "resources" / "models" / "GRIM-text" / "training" / "logs"
CONCEPT_PATH = DATA_DIR / "concept_blocks.jsonl"
OUT_PATH = LOG_DIR / "tokenizer_sequence_length_rankings_lines_1_500185.txt"
START_LINE = 1
END_LINE = 500185
TOP_N = 30
I64_MIN = -(2**63)
I64_MAX = 2**63 - 1
NUMBER_CANDIDATE_RE = re.compile(r"[+\-]?(?:\d+(?:\.\d*)?(?:[eE][+\-]?\d+)?|\.\d+(?:[eE][+\-]?\d+)?|\d+)")
SPACE_RE = re.compile(r"\s+")


def first_string(j: dict, keys: tuple[str, ...]) -> str:
    for key in keys:
        value = j.get(key)
        if isinstance(value, str) and value:
            return value
    return ""


def render_plain_text(j: dict) -> str:
    """Mirror DataLoader plaintext rendering, with mild source-schema tolerance."""
    parts: list[str] = []

    question = first_string(j, ("question", "prompt", "input", "title"))
    if question:
        parts.append(question + "\n")

    explanation = None
    if isinstance(j.get("explanation"), list):
        explanation = j.get("explanation")
    elif isinstance(j.get("intermediates"), list):
        explanation = j.get("intermediates")
    elif isinstance(j.get("content"), str):
        parts.append(j["content"] + "\n")
    elif isinstance(j.get("text"), str):
        parts.append(j["text"] + "\n")

    if explanation is not None:
        parts.extend(item + "\n" for item in explanation if isinstance(item, str))

    answer = first_string(j, ("answer", "output", "completion"))
    if answer:
        parts.append(answer + "\n")

    return "".join(parts)


def normalized_len_from_utf8(raw: bytes) -> int:
    return len(raw) + 3 + 2 * raw.count(b" ")


def char_to_byte_offsets(text: str) -> list[int]:
    offsets = [0] * (len(text) + 1)
    pos = 0
    for i, ch in enumerate(text):
        offsets[i] = pos
        pos += len(ch.encode("utf-8"))
    offsets[len(text)] = pos
    return offsets


def parse_numeric_ok(raw: str) -> bool:
    if not raw or any(ch.isspace() for ch in raw):
        return False
    if "." in raw or "e" in raw or "E" in raw:
        try:
            return math.isfinite(float(raw))
        except Exception:
            return False
    try:
        value = int(raw, 10)
    except Exception:
        return False
    return I64_MIN <= value <= I64_MAX


def numeric_atom_spans(text: str) -> list[tuple[int, int]]:
    matches: list[tuple[int, int, str]] = []
    for match in NUMBER_CANDIDATE_RE.finditer(text):
        raw = match.group(0)
        end = match.end()
        if raw.isdigit() or (raw[:1] in "+-" and raw[1:].isdigit()):
            if end < len(text) and text[end] == ".":
                continue
            if (
                end < len(text)
                and text[end] in "eE"
                and end + 1 < len(text)
                and (text[end + 1].isdigit() or text[end + 1] in "+-")
            ):
                continue
        if parse_numeric_ok(raw):
            matches.append((match.start(), match.end(), raw))
    if not matches:
        return []
    offsets = char_to_byte_offsets(text)
    return [(offsets[start], offsets[end]) for start, end, _ in matches]


def normalized_offsets_for_spans(raw: bytes, spans: Iterable[tuple[int, int]]) -> list[tuple[int, int]]:
    prefix_spaces = [0] * (len(raw) + 1)
    count = 0
    for i, b in enumerate(raw):
        prefix_spaces[i] = count
        if b == 0x20:
            count += 1
    prefix_spaces[len(raw)] = count

    def norm_offset(byte_pos: int) -> int:
        return 3 + byte_pos + 2 * prefix_spaces[byte_pos]

    return [(norm_offset(start), norm_offset(end)) for start, end in spans]


def longest_normalized_non_atom_segment(text: str, raw: bytes) -> tuple[int, int]:
    norm_len = normalized_len_from_utf8(raw)
    spans = numeric_atom_spans(text)
    if not spans:
        return norm_len, 0
    max_segment = 0
    pos = 0
    for start, end in normalized_offsets_for_spans(raw, spans):
        if start > pos:
            max_segment = max(max_segment, start - pos)
        pos = end
    if pos < norm_len:
        max_segment = max(max_segment, norm_len - pos)
    return max_segment, len(spans)


def excerpt(text: str, max_chars: int = 150) -> str:
    compact = SPACE_RE.sub(" ", text).strip()
    if not compact:
        return "(empty rendered text)"
    return compact[:max_chars] + ("…" if len(compact) > max_chars else "")


def push_largest(heap: list, key: str, rec: dict) -> None:
    heappush(heap, (rec[key], rec["line"], rec))
    if len(heap) > TOP_N:
        heappop(heap)


def push_smallest(heap: list, key: str, rec: dict) -> None:
    heappush(heap, (-rec[key], -rec["line"], rec))
    if len(heap) > TOP_N:
        heappop(heap)


def emit_table(lines: list[str], title: str, heap: list, key: str, reverse: bool) -> None:
    lines.extend([
        "",
        title,
        "rank\tmetric\tline\tid\traw_bytes\tnorm_bytes\tspaces\tnumeric_atoms\texcerpt",
    ])
    rows = sorted((item[2] for item in heap), key=lambda rec: (rec[key], rec["line"]), reverse=reverse)
    lines.extend(
        f"{rank}\t{rec[key]}\t{rec['line']}\t{rec['id']}\t{rec['raw_bytes']}\t"
        f"{rec['norm_bytes']}\t{rec['spaces']}\t{rec['numeric_atoms']}\t{rec['excerpt']}"
        for rank, rec in enumerate(rows, 1)
    )


def main() -> None:
    max_seg_heap: list = []
    max_norm_heap: list = []
    max_raw_heap: list = []
    min_seg_heap: list = []
    min_norm_heap: list = []
    min_raw_heap: list = []
    total_rows = 0
    rendered_empty = 0
    parsed_failures = 0

    with CONCEPT_PATH.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if line_no < START_LINE:
                continue
            if line_no > END_LINE:
                break
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except Exception:
                parsed_failures += 1
                continue
            total_rows += 1
            text = render_plain_text(row)
            raw = text.encode("utf-8")
            if not text:
                rendered_empty += 1
            raw_bytes = len(raw)
            norm_bytes = normalized_len_from_utf8(raw)
            longest_segment, numeric_atoms = longest_normalized_non_atom_segment(text, raw)
            rec = {
                "line": line_no,
                "id": row.get("id", ""),
                "raw_bytes": raw_bytes,
                "norm_bytes": norm_bytes,
                "spaces": raw.count(b" "),
                "numeric_atoms": numeric_atoms,
                "longest_segment": longest_segment,
                "excerpt": excerpt(text),
            }
            push_largest(max_seg_heap, "longest_segment", rec)
            push_largest(max_norm_heap, "norm_bytes", rec)
            push_largest(max_raw_heap, "raw_bytes", rec)
            push_smallest(min_seg_heap, "longest_segment", rec)
            push_smallest(min_norm_heap, "norm_bytes", rec)
            push_smallest(min_raw_heap, "raw_bytes", rec)

    lines: list[str] = [
        "GRIM-text tokenizer sequence length ranking for concept_blocks line range",
        f"concept_blocks={CONCEPT_PATH}",
        f"line_range={START_LINE}..{END_LINE}",
        "filter=NONE (all JSONL rows in range)",
        f"processed_rows={total_rows} parse_failures={parsed_failures} empty_rendered_rows={rendered_empty}",
        "measurement_notes=plaintext-like render from question/prompt/input/title + explanation/intermediates/content/text + answer/output/completion; norm_bytes = raw_utf8_bytes + 3 + 2*ASCII_spaces; longest_segment splits normalized text around numeric atoms",
    ]

    emit_table(lines, "TOP_LONGEST_NORMALIZED_NON_ATOM_VITERBI_SEGMENTS", max_seg_heap, "longest_segment", True)
    emit_table(lines, "TOP_LONGEST_NORMALIZED_FULL_RENDERED_TEXTS", max_norm_heap, "norm_bytes", True)
    emit_table(lines, "TOP_LONGEST_RAW_RENDERED_TEXTS", max_raw_heap, "raw_bytes", True)
    emit_table(lines, "TOP_SHORTEST_NORMALIZED_NON_ATOM_VITERBI_SEGMENTS", min_seg_heap, "longest_segment", False)
    emit_table(lines, "TOP_SHORTEST_NORMALIZED_FULL_RENDERED_TEXTS", min_norm_heap, "norm_bytes", False)
    emit_table(lines, "TOP_SHORTEST_RAW_RENDERED_TEXTS", min_raw_heap, "raw_bytes", False)

    OUT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
