#!/usr/bin/env python3
"""Export migrated JSONL plus a prose-evidence archive; never alter the source.

Accepts ConceptBlock JSONL or an existing GRCB .fb using the installed flatc.
No prose is interpreted as an executable condition. Requirements without
structured evidence remain unverified until an author supplies conditions.
"""
from __future__ import annotations

import argparse
import copy
import json
import subprocess
import tempfile
from pathlib import Path

from build_concept_blocks_flatbuffer import default_flatc
from goal_evidence import validate_goal


def migrate_entry(source):
    entry = copy.deepcopy(source)
    archive = []
    goal = entry.get("goal")
    if goal is None:
        return entry, archive
    if not isinstance(goal, dict):
        raise ValueError("goal must be an object")
    if "legacy_constraints" in goal:
        if "constraints" in goal:
            raise ValueError("Mixed legacy and structured constraints")
        goal["constraints"] = goal.pop("legacy_constraints")
    criteria = goal.get("success_criteria", [])
    if not isinstance(criteria, list):
        raise ValueError("success_criteria must be an array")
    for index, criterion in enumerate(criteria):
        if not isinstance(criterion, dict):
            raise ValueError("criterion must be an object")
        if "legacy_evidence" in criterion:
            legacy = criterion.pop("legacy_evidence")
            if not isinstance(legacy, str):
                raise ValueError("legacy_evidence must be a string")
            if legacy:
                archive.append({"id": entry.get("id"), "path": f"/goal/success_criteria/{index}/evidence", "text": legacy})
        if isinstance(criterion.get("evidence"), str):
            prose = criterion.pop("evidence")
            if prose:
                archive.append({"id": entry.get("id"), "path": f"/goal/success_criteria/{index}/evidence", "text": prose})
    constraints = goal.get("constraints", [])
    if not isinstance(constraints, list):
        raise ValueError("constraints must be an array")
    goal["constraints"] = [({"constraint": item} if isinstance(item, str) else item)
                           for item in constraints]
    validate_goal(goal)
    return entry, archive


def migrate_file(source, output, archive, schema=None, flatc=None):
    source, output, archive = map(Path, (source, output, archive))
    paths = [path.resolve() for path in (source, output, archive)]
    if len(set(paths)) != 3 or output.exists() or archive.exists():
        raise ValueError("Source, output, and archive must be distinct; outputs must not already exist")
    repo = Path(__file__).resolve().parent.parent
    with tempfile.TemporaryDirectory(prefix="grim_goal_migration_") as folder:
        folder = Path(folder)
        if source.suffix.lower() == ".fb":
            subprocess.run([str(flatc or default_flatc(repo)), "--json", "--strict-json",
                            "-o", str(folder), str(schema or repo / "DataCollection/concept_block.fbs"),
                            "--", str(source.resolve())], check=True, capture_output=True, text=True)
            envelope = json.loads((folder / (source.stem + ".json")).read_text(encoding="utf-8"))
            if envelope.get("schema_version", 1) not in (1, 2, 3):
                raise ValueError("Unsupported dataset schema version")
            entries = enumerate(envelope.get("blocks", []), 1)
            return _write_entries(entries, output, archive)
        with source.open(encoding="utf-8") as stream:
            entries = ((number, json.loads(line)) for number, line in enumerate(stream, 1) if line.strip())
            return _write_entries(entries, output, archive)


def _write_entries(entries, output, archive):
    # Stage everything before publishing. Exclusive creation protects existing
    # files; an archive is published first so prose cannot be lost on failure.
    with tempfile.TemporaryFile(mode="w+", encoding="utf-8") as staged, \
            tempfile.TemporaryFile(mode="w+", encoding="utf-8") as archived:
        count = prose_count = 0
        for number, source in entries:
            try:
                entry, records = migrate_entry(source)
            except (ValueError, TypeError, KeyError) as error:
                raise ValueError(f"Entry {number}: {error}") from error
            staged.write(json.dumps(entry, ensure_ascii=False) + "\n")
            for record in records:
                record["entry_number"] = number
                archived.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
            prose_count += len(records)
        import shutil
        archived.seek(0)
        with archive.open("x", encoding="utf-8", newline="\n") as destination:
            shutil.copyfileobj(archived, destination)
        staged.seek(0)
        with output.open("x", encoding="utf-8", newline="\n") as destination:
            shutil.copyfileobj(staged, destination)
        return count, prose_count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path, help="New migrated JSONL path")
    parser.add_argument("--archive", required=True, type=Path, help="New archive path for legacy prose")
    parser.add_argument("--flatc", type=Path)
    args = parser.parse_args()
    count, prose = migrate_file(args.input, args.output, args.archive, flatc=args.flatc)
    print(f"Exported {count} blocks; archived {prose} prose evidence values. Source unchanged.")


if __name__ == "__main__":
    main()
