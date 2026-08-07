#!/usr/bin/env python3
"""Atomically migrate ConceptBlock JSONL records from question to prompt."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_PATH = (
    ROOT_DIR
    / "resources"
    / "models"
    / "GRIM-text"
    / "training"
    / "data"
    / "concept_blocks.jsonl"
)


def migrate_entry(entry: dict[str, object], source: str) -> tuple[dict[str, object], bool]:
    has_prompt = "prompt" in entry
    has_question = "question" in entry
    if not has_prompt and not has_question:
        raise ValueError(f"{source}: missing prompt")
    if has_prompt and has_question and entry["prompt"] != entry["question"]:
        raise ValueError(f"{source}: conflicting prompt and question values")

    prompt = entry["prompt"] if has_prompt else entry["question"]
    if not isinstance(prompt, str):
        raise ValueError(f"{source}: prompt must be a string")

    migrated: dict[str, object] = {}
    inserted_prompt = False
    for key, value in entry.items():
        if key == "question":
            if not inserted_prompt:
                migrated["prompt"] = prompt
                inserted_prompt = True
            continue
        if key == "prompt":
            if not inserted_prompt:
                migrated["prompt"] = prompt
                inserted_prompt = True
            continue
        migrated[key] = value
    return migrated, has_question


def migrate_file(path: Path) -> tuple[int, int]:
    if not path.is_file():
        raise FileNotFoundError(f"concept block JSONL not found: {path}")

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    total = 0
    changed = 0
    try:
        with path.open("r", encoding="utf-8") as source, temporary.open(
            "w", encoding="utf-8", newline="\n"
        ) as destination:
            for line_number, line in enumerate(source, start=1):
                if not line.strip():
                    raise ValueError(f"{path}:{line_number}: blank JSONL row")
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(f"{path}:{line_number}: {error}") from error
                if not isinstance(entry, dict):
                    raise ValueError(f"{path}:{line_number}: row must be a JSON object")
                migrated, row_changed = migrate_entry(entry, f"{path}:{line_number}")
                destination.write(
                    json.dumps(migrated, ensure_ascii=False, separators=(",", ":")) + "\n"
                )
                total += 1
                changed += int(row_changed)
                if total % 50_000 == 0:
                    print(f"Validated {total} rows", flush=True)
            destination.flush()
            os.fsync(destination.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return total, changed


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path, nargs="?", default=DEFAULT_PATH)
    args = parser.parse_args()
    total, changed = migrate_file(args.path.resolve())
    print(f"Migrated {changed} of {total} ConceptBlock rows to prompt", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())