#!/usr/bin/env python3
"""
One-time migration: add "arg_slots" to every execution step in concept_blocks.jsonl
that doesn't already have it.

Uses the same value-based resolution logic that the C++ builder used to have,
but writes the result as canonical arg_slots so the builder no longer needs
the fallback.

Steps that are ambiguous (multiple valid slot pairs) are reported and SKIPPED
— those entries need manual curation.

Usage:
    python3 scripts/migrate_add_arg_slots.py \
        --input  resources/models/GRIM-text/training/data/concept_blocks.jsonl \
        --output resources/models/GRIM-text/training/data/concept_blocks_migrated.jsonl
"""
from __future__ import annotations

import argparse
import json
import math
import sys


OPS = {
    "add": lambda a, b: a + b,
    "+":   lambda a, b: a + b,
    "sub": lambda a, b: a - b,
    "-":   lambda a, b: a - b,
    "mul": lambda a, b: a * b,
    "*":   lambda a, b: a * b,
    "div": lambda a, b: a / b if b != 0 else float("inf"),
    "/":   lambda a, b: a / b if b != 0 else float("inf"),
}


def result_matches(computed: float, expected: float) -> bool:
    if math.isnan(computed) or math.isinf(computed):
        return False
    if expected == 0.0:
        return abs(computed) < 1e-9
    return abs(computed - expected) / max(abs(expected), 1e-12) < 1e-6


def resolve_arg_slots(
    slot_values: list[float],
    args: list[float],
    op_str: str,
    expected: float,
) -> list[int] | None:
    """Return [slot1, slot2] or None if ambiguous/unresolvable."""
    normalized_op = op_str.strip().lower()
    op_fn = OPS.get(normalized_op)
    if op_fn is None:
        return None

    candidates_1 = [i for i, v in enumerate(slot_values) if v == args[0]]
    candidates_2 = [i for i, v in enumerate(slot_values) if v == args[1]]

    if not candidates_1 or not candidates_2:
        return None

    valid = []
    for c1 in candidates_1:
        for c2 in candidates_2:
            computed = op_fn(slot_values[c1], slot_values[c2])
            if result_matches(computed, expected):
                valid.append((c1, c2))

    if len(valid) == 1:
        return list(valid[0])

    if valid:
        distinct_pairs = sorted({(c1, c2) for c1, c2 in valid if c1 != c2})
        if distinct_pairs:
            return list(distinct_pairs[0])
        return list(sorted(set(valid))[0])

    return None  # 0 or >1 matches


def migrate_entry(entry: dict) -> tuple[dict, list[str]]:
    """Add arg_slots to execution steps. Returns (entry, warnings)."""
    warnings: list[str] = []

    if "execution" not in entry or not isinstance(entry["execution"], list):
        return entry, warnings
    if "state_0" not in entry or "atoms" not in entry.get("state_0", {}):
        return entry, warnings

    atoms = entry["state_0"]["atoms"]
    if not isinstance(atoms, list) or not all(isinstance(a, (int, float)) for a in atoms):
        return entry, warnings

    slot_values = list(atoms)

    for step_idx, step in enumerate(entry["execution"]):
        if not isinstance(step, dict):
            continue

        # Already has arg_slots — skip
        if "arg_slots" in step and isinstance(step["arg_slots"], list) and len(step["arg_slots"]) >= 2:
            slot_values.append(step.get("result", 0.0))
            continue

        args = step.get("args", [])
        op_str = step.get("op", "")
        expected = step.get("result", 0.0)

        if len(args) < 2:
            warnings.append(f"step {step_idx}: <2 args")
            slot_values.append(expected)
            continue

        resolved = resolve_arg_slots(slot_values, args, op_str, expected)
        if resolved is not None:
            step["arg_slots"] = resolved
        else:
            warnings.append(
                f"step {step_idx}: ambiguous or unresolvable "
                f"(op={op_str}, args={args}, expected={expected}, "
                f"slot_values={slot_values})"
            )

        slot_values.append(expected)

    return entry, warnings


def main() -> int:
    ap = argparse.ArgumentParser(description="Add arg_slots to concept_blocks.jsonl")
    ap.add_argument("--input", required=True, help="Input JSONL path")
    ap.add_argument("--output", required=True, help="Output JSONL path")
    args = ap.parse_args()

    total = 0
    migrated = 0
    already_ok = 0
    no_execution = 0
    problem_entries: list[tuple[str, list[str]]] = []

    with open(args.input, encoding="utf-8") as fin, \
         open(args.output, "w", encoding="utf-8") as fout:
        for line_no, line in enumerate(fin, 1):
            line = line.strip()
            if not line:
                continue
            total += 1
            try:
                entry = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"Line {line_no}: JSON parse error: {e}", file=sys.stderr)
                fout.write(line + "\n")
                continue

            if "execution" not in entry or not entry["execution"]:
                no_execution += 1
                fout.write(json.dumps(entry, ensure_ascii=False) + "\n")
                continue

            entry, warnings = migrate_entry(entry)
            if warnings:
                eid = entry.get("id", f"line_{line_no}")
                problem_entries.append((eid, warnings))
            else:
                # Check if all steps now have arg_slots
                all_have = all(
                    "arg_slots" in s and len(s.get("arg_slots", [])) >= 2
                    for s in entry.get("execution", [])
                    if isinstance(s, dict)
                )
                if all_have:
                    migrated += 1
                else:
                    already_ok += 1

            fout.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"Total entries: {total}")
    print(f"No execution (chat/instruct): {no_execution}")
    print(f"Migrated (arg_slots added): {migrated}")
    print(f"Already OK: {already_ok}")
    print(f"Problem entries (ambiguous): {len(problem_entries)}")

    if problem_entries:
        print("\n--- Problem entries (need manual curation) ---")
        for eid, warns in problem_entries[:50]:
            print(f"  {eid}:")
            for w in warns:
                print(f"    {w}")
        if len(problem_entries) > 50:
            print(f"  ... and {len(problem_entries) - 50} more")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
