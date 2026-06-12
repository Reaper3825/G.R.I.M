#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = (
    ROOT_DIR
    / "resources"
    / "models"
    / "GRIM-text"
    / "training"
    / "data"
    / "concept_blocks.jsonl"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Clip contiguous digit spans of length 16 or more down to 16 digits "
            "inside concept_blocks.jsonl, skipping id fields."
        )
    )
    parser.add_argument(
        "input",
        nargs="?",
        default=str(DEFAULT_INPUT),
        help=f"JSONL file to rewrite (default: {DEFAULT_INPUT})",
    )
    parser.add_argument(
        "--min-digits",
        type=int,
        default=16,
        help="Minimum contiguous digit length to clip (default: 16)",
    )
    parser.add_argument(
        "--clip-length",
        type=int,
        default=16,
        help="Length to keep from each matching digit span (default: 16)",
    )
    parser.add_argument(
        "--max-lines",
        type=int,
        default=0,
        help="Stop after processing this many non-empty JSONL lines (0 = no limit)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview how many changes would be made without rewriting the file",
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Do not create a .bak copy before rewriting in place",
    )
    return parser


def is_id_key(key: str) -> bool:
    lowered = key.lower()
    return lowered == "id" or lowered.endswith("_id")


def apply_clipping(text: str, min_digits: int, clip_length: int) -> tuple[str, int]:
    out: list[str] = []
    changed = 0
    i = 0
    text_length = len(text)

    while i < text_length:
        if not text[i].isdigit():
            out.append(text[i])
            i += 1
            continue

        j = i
        while j < text_length and text[j].isdigit():
            j += 1

        run = text[i:j]
        if len(run) >= min_digits and len(run) > clip_length:
            out.append(run[:clip_length])
            changed += 1
        else:
            out.append(run)
        i = j

    return "".join(out), changed


def transform_value(
    value: object,
    *,
    min_digits: int,
    clip_length: int,
) -> tuple[object, int]:
    if isinstance(value, dict):
        total_changes = 0
        rewritten: dict[object, object] = {}
        for key, nested in value.items():
            if is_id_key(str(key)):
                rewritten[key] = nested
                continue
            rewritten_nested, nested_changes = transform_value(
                nested,
                min_digits=min_digits,
                clip_length=clip_length,
            )
            rewritten[key] = rewritten_nested
            total_changes += nested_changes
        return rewritten, total_changes

    if isinstance(value, list):
        total_changes = 0
        rewritten_list: list[object] = []
        for nested in value:
            rewritten_nested, nested_changes = transform_value(
                nested,
                min_digits=min_digits,
                clip_length=clip_length,
            )
            rewritten_list.append(rewritten_nested)
            total_changes += nested_changes
        return rewritten_list, total_changes

    if isinstance(value, str):
        return apply_clipping(value, min_digits, clip_length)

    return value, 0


def main() -> int:
    args = build_parser().parse_args()
    if args.min_digits < 1:
        print("--min-digits must be at least 1", file=sys.stderr)
        return 1
    if args.clip_length < 1:
        print("--clip-length must be at least 1", file=sys.stderr)
        return 1
    if args.clip_length > args.min_digits:
        print("--clip-length cannot be greater than --min-digits", file=sys.stderr)
        return 1

    input_path = Path(args.input).resolve()
    if not input_path.is_file():
        print(f"File not found: {input_path}", file=sys.stderr)
        return 1

    lines_processed = 0
    invalid_lines = 0
    changed_lines = 0
    total_replacements = 0

    temp_output_path: Path | None = None
    temp_handle = None
    if not args.dry_run:
        temp_handle = tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=str(input_path.parent),
            prefix=f"{input_path.name}.",
            suffix=".tmp",
            delete=False,
        )
        temp_output_path = Path(temp_handle.name)

    try:
        with input_path.open("r", encoding="utf-8") as source:
            for line_number, raw_line in enumerate(source, start=1):
                stripped = raw_line.strip()
                if not stripped:
                    if temp_handle is not None:
                        temp_handle.write(raw_line)
                    continue

                if args.max_lines and lines_processed >= args.max_lines:
                    if temp_handle is not None:
                        temp_handle.write(raw_line)
                    continue

                lines_processed += 1

                try:
                    payload = json.loads(stripped)
                except json.JSONDecodeError as exc:
                    invalid_lines += 1
                    print(
                        f"line {line_number}: invalid JSON ({exc.msg})",
                        file=sys.stderr,
                    )
                    if temp_handle is not None:
                        temp_handle.write(raw_line)
                    continue

                rewritten_payload, replacements = transform_value(
                    payload,
                    min_digits=args.min_digits,
                    clip_length=args.clip_length,
                )

                if replacements:
                    changed_lines += 1
                    total_replacements += replacements

                if temp_handle is not None:
                    temp_handle.write(
                        json.dumps(rewritten_payload, ensure_ascii=False) + "\n"
                    )
    finally:
        if temp_handle is not None:
            temp_handle.close()

    if args.dry_run:
        print(
            (
                f"Scanned {lines_processed} lines, would update {changed_lines} lines, "
                f"clip {total_replacements} digit spans, and saw {invalid_lines} invalid JSON lines."
            ),
            file=sys.stderr,
        )
        return 0

    assert temp_output_path is not None

    if not args.no_backup:
        backup_path = input_path.with_suffix(input_path.suffix + ".bak")
        shutil.copy2(input_path, backup_path)
        print(f"Backup written to {backup_path}", file=sys.stderr)

    temp_output_path.replace(input_path)
    print(
        (
            f"Updated {input_path}: processed {lines_processed} lines, changed {changed_lines} lines, "
            f"clipped {total_replacements} digit spans, invalid JSON lines {invalid_lines}."
        ),
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
