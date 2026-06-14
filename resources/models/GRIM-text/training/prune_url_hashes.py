#!/usr/bin/env python3
"""Soft-prune long hash-like substrings inside URLs in JSON/JSONL corpora.

This is meant to sanitize tokenizer-training text without dropping whole rows.
It walks string values recursively, finds URL spans, and rewrites long hex-like
hash payloads inside those URLs to a stable placeholder.

Examples:
  https://secure.gravatar.com/avatar/415c5057418c11b23487a6f88e60b765?s=96
  -> https://secure.gravatar.com/avatar/URLHASH?s=96
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


URL_RE = re.compile(r"https?://[^\s\"'<>]+")


@dataclass
class RewriteEvent:
    line_no: int
    json_path: str
    original_url: str
    rewritten_url: str


@dataclass
class RewriteStats:
    total_lines: int = 0
    changed_lines: int = 0
    parse_errors: int = 0
    strings_scanned: int = 0
    urls_scanned: int = 0
    urls_changed: int = 0
    hash_replacements: int = 0
    events: list[RewriteEvent] = field(default_factory=list)


def compile_hash_pattern(min_hex_length: int) -> re.Pattern[str]:
    return re.compile(rf"(?<![A-Za-z0-9])[A-Fa-f0-9]{{{min_hex_length},}}(?![A-Za-z0-9])")


def rewrite_url(url: str, hash_re: re.Pattern[str], placeholder: str) -> tuple[str, int]:
    replacements = 0

    def repl(match: re.Match[str]) -> str:
        nonlocal replacements
        replacements += 1
        return placeholder

    rewritten = hash_re.sub(repl, url)
    return rewritten, replacements


def rewrite_string(
    value: str,
    hash_re: re.Pattern[str],
    placeholder: str,
    stats: RewriteStats,
) -> tuple[str, int, list[tuple[str, str]]]:
    stats.strings_scanned += 1
    total_replacements = 0
    changed_urls: list[tuple[str, str]] = []

    def repl(match: re.Match[str]) -> str:
        nonlocal total_replacements
        url = match.group(0)
        stats.urls_scanned += 1
        rewritten, replacements = rewrite_url(url, hash_re, placeholder)
        if replacements > 0:
            stats.urls_changed += 1
            stats.hash_replacements += replacements
            total_replacements += replacements
            changed_urls.append((url, rewritten))
        return rewritten

    rewritten_value = URL_RE.sub(repl, value)
    return rewritten_value, total_replacements, changed_urls


def walk_and_rewrite(
    node: Any,
    path: str,
    line_no: int,
    hash_re: re.Pattern[str],
    placeholder: str,
    stats: RewriteStats,
    max_examples: int,
) -> tuple[Any, bool]:
    changed = False

    if isinstance(node, dict):
        out: dict[str, Any] = {}
        for key, value in node.items():
            child_path = f"{path}.{key}" if path else key
            rewritten, child_changed = walk_and_rewrite(
                value, child_path, line_no, hash_re, placeholder, stats, max_examples
            )
            out[key] = rewritten
            changed = changed or child_changed
        return out, changed

    if isinstance(node, list):
        out_list: list[Any] = []
        for idx, value in enumerate(node):
            child_path = f"{path}[{idx}]"
            rewritten, child_changed = walk_and_rewrite(
                value, child_path, line_no, hash_re, placeholder, stats, max_examples
            )
            out_list.append(rewritten)
            changed = changed or child_changed
        return out_list, changed

    if isinstance(node, str):
        rewritten, replacements, changed_urls = rewrite_string(node, hash_re, placeholder, stats)
        if replacements > 0:
            changed = True
            if len(stats.events) < max_examples:
                for original_url, rewritten_url in changed_urls:
                    if len(stats.events) >= max_examples:
                        break
                    stats.events.append(
                        RewriteEvent(
                            line_no=line_no,
                            json_path=path or "<root>",
                            original_url=original_url,
                            rewritten_url=rewritten_url,
                        )
                    )
        return rewritten, changed

    return node, False


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            yield line_no, line


def process_jsonl(
    input_path: Path,
    output_path: Path | None,
    in_place: bool,
    hash_re: re.Pattern[str],
    placeholder: str,
    max_examples: int,
) -> RewriteStats:
    stats = RewriteStats()

    if in_place:
        fd, tmp_name = tempfile.mkstemp(prefix=input_path.name + ".", suffix=".tmp", dir=str(input_path.parent))
        os.close(fd)
        destination = Path(tmp_name)
    elif output_path is not None:
        destination = output_path
    else:
        destination = None

    writer = None
    try:
        if destination is not None:
            writer = destination.open("w", encoding="utf-8", newline="\n")

        for line_no, line in iter_jsonl(input_path):
            stats.total_lines += 1
            if not line.strip():
                if writer is not None:
                    writer.write(line)
                continue

            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                stats.parse_errors += 1
                if writer is not None:
                    writer.write(line)
                continue

            rewritten, changed = walk_and_rewrite(
                obj,
                path="",
                line_no=line_no,
                hash_re=hash_re,
                placeholder=placeholder,
                stats=stats,
                max_examples=max_examples,
            )
            if changed:
                stats.changed_lines += 1

            if writer is not None:
                writer.write(json.dumps(rewritten, ensure_ascii=False) + "\n")

    finally:
        if writer is not None:
            writer.close()

    if in_place and destination is not None:
        destination.replace(input_path)

    return stats


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Soft-prune long hash-like substrings inside URLs in JSONL corpora."
    )
    parser.add_argument(
        "input",
        type=Path,
        help="Input JSONL file, e.g. training/data/concept_blocks.jsonl",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Write rewritten JSONL here. Omit with --dry-run, or use --in-place.",
    )
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="Rewrite the input file atomically in place.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Scan and report only; do not write output.",
    )
    parser.add_argument(
        "--min-hex-length",
        type=int,
        default=16,
        help="Minimum contiguous hex length to treat as a hash inside a URL (default: 16).",
    )
    parser.add_argument(
        "--placeholder",
        default="URLHASH",
        help="Replacement token inserted for detected URL hashes (default: URLHASH).",
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=10,
        help="How many example rewrites to print (default: 10).",
    )
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.min_hex_length <= 0:
        raise SystemExit("--min-hex-length must be positive")
    if args.in_place and args.output is not None:
        raise SystemExit("use either --output or --in-place, not both")
    if not args.in_place and not args.dry_run and args.output is None:
        raise SystemExit("provide --output, or use --in-place, or use --dry-run")
    if not args.input.exists():
        raise SystemExit(f"input file does not exist: {args.input}")
    if not args.input.is_file():
        raise SystemExit(f"input path is not a file: {args.input}")


def print_summary(stats: RewriteStats) -> None:
    print(f"lines_scanned={stats.total_lines}")
    print(f"changed_lines={stats.changed_lines}")
    print(f"parse_errors={stats.parse_errors}")
    print(f"strings_scanned={stats.strings_scanned}")
    print(f"urls_scanned={stats.urls_scanned}")
    print(f"urls_changed={stats.urls_changed}")
    print(f"hash_replacements={stats.hash_replacements}")
    if stats.events:
        print("examples:")
        for event in stats.events:
            print(f"  line={event.line_no} path={event.json_path}")
            print(f"    before={event.original_url}")
            print(f"    after ={event.rewritten_url}")


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()
    validate_args(args)

    hash_re = compile_hash_pattern(args.min_hex_length)
    stats = process_jsonl(
        input_path=args.input,
        output_path=args.output,
        in_place=args.in_place,
        hash_re=hash_re,
        placeholder=args.placeholder,
        max_examples=args.max_examples,
    )
    print_summary(stats)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
