#!/usr/bin/env python3
"""Generate the deterministic GRIM-text execution-control v1 curriculum.

The generator is append-only and idempotent:
  * existing concept blocks are never rewritten;
  * missing v1 blocks are appended to concept_blocks.jsonl;
  * curriculum_registry.json is replaced atomically after merging the three
    curriculum definitions.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any, Iterable


OPS = ("add", "sub", "mul", "div")
SYMBOLS = {"add": "+", "sub": "-", "mul": "*", "div": "/"}
SPLIT_CONFIG = {
    "train": {
        "number_range": (2, 50),
        "exec_templates": (
            "Let {bindings}. Calculate {expression}.",
            "Given {bindings}, evaluate {expression}.",
            "Use {bindings}. What is the value of {expression}?",
            "With {bindings}, solve {expression}.",
        ),
        "noop_templates": (
            "Ticket {number} is assigned to {name}. Who is assigned to the ticket?",
            "The label on box {number} is {label}. What label is on the box?",
            "Repeat the identifier {number} exactly.",
            "Room {number} is described as {adjective}. Which word describes the room?",
            "Record {number} belongs to category {category}. Which category is listed?",
        ),
    },
    "validation": {
        "number_range": (51, 99),
        "exec_templates": (
            "Assign {bindings}; determine {expression}.",
            "For the values {bindings}, find {expression}.",
        ),
        "noop_templates": (
            "Archive entry {number} names {name} as reviewer. Who is the reviewer?",
            "Copy code {number} without altering it.",
            "Shelf {number} carries the tag {label}. What tag does it carry?",
        ),
    },
    "test": {
        "number_range": (101, 199),
        "exec_templates": (
            "Suppose {bindings}. Work out {expression}.",
            "When {bindings}, compute the expression {expression}.",
        ),
        "noop_templates": (
            "Case file {number} is owned by {name}. Who owns the case file?",
            "Echo reference number {number} exactly as written.",
            "Container {number} has the marker {label}. What marker is shown?",
        ),
    },
}

NAMES = ("Avery", "Blake", "Casey", "Devon", "Emery", "Finley", "Gray", "Harper")
LABELS = ("blue", "green", "amber", "violet", "silver", "orange", "white", "black")
ADJECTIVES = ("quiet", "bright", "sealed", "empty", "reserved", "open")
CATEGORIES = ("alpha", "beta", "gamma", "delta", "archive", "priority")


def format_number(value: float) -> str:
    if value == math.floor(value) and abs(value) < 1e12:
        return str(int(value))
    return format(value, ".12g")


def apply_op(op: str, lhs: float, rhs: float) -> float:
    if op == "add":
        return lhs + rhs
    if op == "sub":
        return lhs - rhs
    if op == "mul":
        return lhs * rhs
    if op == "div":
        if rhs == 0.0:
            raise ZeroDivisionError
        return lhs / rhs
    raise ValueError(f"unknown operation: {op}")


def program_shape(step_count: int) -> tuple[int, list[tuple[int, int]]]:
    if step_count == 1:
        return 2, [(0, 1)]
    if step_count == 2:
        return 3, [(0, 1), (3, 2)]
    if step_count == 3:
        return 4, [(0, 1), (2, 3), (4, 5)]
    if step_count == 4:
        # Four atoms + four result slots exactly fill the configured 8 slots.
        # The fourth operation reuses symbolic variable `a`; the numeric value
        # appears only once in the prompt, so inference bootstrap alignment is
        # unchanged.
        return 4, [(0, 1), (2, 3), (4, 5), (6, 0)]
    raise ValueError(f"unsupported step count: {step_count}")


def symbolic_expression(step_count: int, ops: list[str]) -> str:
    sym = [SYMBOLS[op] for op in ops]
    if step_count == 1:
        return f"(a {sym[0]} b)"
    if step_count == 2:
        return f"((a {sym[0]} b) {sym[1]} c)"
    base = f"((a {sym[0]} b) {sym[2]} (c {sym[1]} d))"
    if step_count == 3:
        return base
    return f"({base} {sym[3]} a)"


def make_execute_entry(split: str, index: int, rng: random.Random) -> dict[str, Any]:
    cfg = SPLIT_CONFIG[split]
    step_count = index % 4 + 1
    atom_count, arg_slots = program_shape(step_count)
    op_cycle = index // 4
    ops = [OPS[(op_cycle + step * 3) % len(OPS)] for step in range(step_count)]
    low, high = cfg["number_range"]

    for _ in range(1000):
        atoms = [float(rng.randint(low, high)) for _ in range(atom_count)]
        slots = list(atoms)
        steps: list[dict[str, Any]] = []
        explanations: list[str] = []
        try:
            for step_index, ((lhs_slot, rhs_slot), op) in enumerate(zip(arg_slots, ops)):
                lhs = slots[lhs_slot]
                rhs = slots[rhs_slot]
                result = apply_op(op, lhs, rhs)
                if not math.isfinite(result) or abs(result) > 1_000_000.0:
                    raise ArithmeticError
                steps.append({
                    "op": op,
                    "args": [lhs, rhs],
                    "arg_slots": [lhs_slot, rhs_slot],
                    "result": result,
                })
                explanations.append(
                    f"Step {step_index + 1}: {format_number(lhs)} {SYMBOLS[op]} "
                    f"{format_number(rhs)} = {format_number(result)}."
                )
                slots.append(result)
        except (ArithmeticError, ZeroDivisionError):
            continue
        break
    else:
        raise RuntimeError(f"failed to generate stable {split} execute row {index}")

    variables = ("a", "b", "c", "d")[:atom_count]
    bindings = ", ".join(
        f"{name}={format_number(value)}" for name, value in zip(variables, atoms)
    )
    expression = symbolic_expression(step_count, ops)
    template = cfg["exec_templates"][index % len(cfg["exec_templates"])]
    question = template.format(bindings=bindings, expression=expression)
    final_result = steps[-1]["result"]
    block_id = f"ecv1_{split}_execute_{index:06d}"
    return {
        "id": block_id,
        "name": f"Execution Control v1 {split} EXECUTE {index:06d}",
        "prompt": question,
        "intermediates": explanations,
        "explanation": explanations,
        "answer": format_number(final_result),
        "execution_gate_target": "execute",
        "state_0": {"type": "arithmetic", "atoms": atoms},
        "execution": steps,
        "state_1": {"result": final_result},
        "intermediate_count": len(explanations),
        "step_index": list(range(len(explanations))),
        "format_type": "derivation",
        "timestamp": 0,
    }


def make_noop_entry(split: str, index: int, rng: random.Random) -> dict[str, Any]:
    cfg = SPLIT_CONFIG[split]
    low, high = cfg["number_range"]
    number = rng.randint(low * 1000, high * 1000 + 999)
    name = NAMES[index % len(NAMES)]
    label = LABELS[(index * 3) % len(LABELS)]
    adjective = ADJECTIVES[(index * 5) % len(ADJECTIVES)]
    category = CATEGORIES[(index * 7) % len(CATEGORIES)]
    template_index = index % len(cfg["noop_templates"])
    question = cfg["noop_templates"][template_index].format(
        number=number,
        name=name,
        label=label,
        adjective=adjective,
        category=category,
    )

    lowered = question.lower()
    if "who" in lowered:
        answer = name
    elif "label" in lowered or "tag" in lowered or "marker" in lowered:
        answer = label
    elif "describes" in lowered:
        answer = adjective
    elif "category" in lowered:
        answer = category
    else:
        answer = str(number)

    block_id = f"ecv1_{split}_noop_{index:06d}"
    return {
        "id": block_id,
        "name": f"Execution Control v1 {split} NOOP {index:06d}",
        "prompt": question,
        "intermediates": [],
        "explanation": [],
        "answer": answer,
        "execution_gate_target": "noop",
        "intermediate_count": 0,
        "step_index": [],
        "format_type": "qa",
        "timestamp": 0,
    }


def validate_entry(entry: dict[str, Any], max_slots: int = 8) -> None:
    target = entry["execution_gate_target"]
    if not entry.get("prompt"):
        raise ValueError(f"{entry['id']}: missing question")
    if target == "noop":
        if "state_0" in entry or "execution" in entry:
            raise ValueError(f"{entry['id']}: NOOP row carries execution data")
        return
    if target != "execute":
        raise ValueError(f"{entry['id']}: invalid gate target {target}")

    atoms = entry["state_0"]["atoms"]
    slots = list(atoms)
    for step_index, step in enumerate(entry["execution"]):
        if len(step["args"]) != 2 or len(step["arg_slots"]) != 2:
            raise ValueError(f"{entry['id']}: malformed step {step_index}")
        lhs_slot, rhs_slot = step["arg_slots"]
        if lhs_slot >= len(slots) or rhs_slot >= len(slots):
            raise ValueError(f"{entry['id']}: out-of-range step {step_index}")
        lhs, rhs = slots[lhs_slot], slots[rhs_slot]
        if step["args"] != [lhs, rhs]:
            raise ValueError(f"{entry['id']}: args/slots disagree at step {step_index}")
        computed = apply_op(step["op"], lhs, rhs)
        expected = step["result"]
        if abs(computed - expected) > 1e-9 * max(1.0, abs(expected)):
            raise ValueError(f"{entry['id']}: incorrect result at step {step_index}")
        slots.append(expected)
    if len(slots) > max_slots:
        raise ValueError(f"{entry['id']}: requires {len(slots)} slots, max is {max_slots}")


def generated_entries(split: str, per_class: int, seed: int) -> Iterable[dict[str, Any]]:
    execute_rng = random.Random(seed + {"train": 11, "validation": 23, "test": 37}[split])
    noop_rng = random.Random(seed + {"train": 101, "validation": 211, "test": 307}[split])
    for index in range(per_class):
        yield make_execute_entry(split, index, execute_rng)
        yield make_noop_entry(split, index, noop_rng)


def find_existing_v1_ids(concept_path: Path) -> set[str]:
    existing: set[str] = set()
    if not concept_path.exists():
        return existing
    marker = '"id":"ecv1_'
    with concept_path.open("r", encoding="utf-8", errors="replace") as source:
        for line in source:
            if marker not in line:
                continue
            try:
                block_id = json.loads(line).get("id", "")
            except json.JSONDecodeError:
                continue
            if block_id.startswith("ecv1_"):
                existing.add(block_id)
    return existing


def append_missing_blocks(
    concept_path: Path,
    entries_by_split: dict[str, list[dict[str, Any]]],
    dry_run: bool,
) -> int:
    existing = find_existing_v1_ids(concept_path)
    missing = [
        entry
        for split_entries in entries_by_split.values()
        for entry in split_entries
        if entry["id"] not in existing
    ]
    if dry_run or not missing:
        return len(missing)

    concept_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", newline="\n", delete=False,
        dir=concept_path.parent, prefix="execution_control_v1_", suffix=".jsonl.tmp"
    ) as stage:
        stage_path = Path(stage.name)
        for entry in missing:
            stage.write(json.dumps(entry, ensure_ascii=False, separators=(",", ":")) + "\n")
        stage.flush()
        os.fsync(stage.fileno())

    try:
        with concept_path.open("ab+") as destination:
            destination.seek(0, os.SEEK_END)
            if destination.tell() > 0:
                destination.seek(-1, os.SEEK_END)
                if destination.read(1) != b"\n":
                    destination.seek(0, os.SEEK_END)
                    destination.write(b"\n")
            destination.seek(0, os.SEEK_END)
            with stage_path.open("rb") as source:
                shutil.copyfileobj(source, destination, length=1024 * 1024)
            destination.flush()
            os.fsync(destination.fileno())
    finally:
        stage_path.unlink(missing_ok=True)
    return len(missing)


def merge_curricula(
    registry_path: Path,
    entries_by_split: dict[str, list[dict[str, Any]]],
    dry_run: bool,
) -> None:
    if registry_path.exists():
        registry = json.loads(registry_path.read_text(encoding="utf-8"))
    else:
        registry = {"curriculums": []}
    curricula = registry.setdefault("curriculums", [])
    by_id = {item.get("id"): item for item in curricula}
    timestamp = int(time.time())

    for split, entries in entries_by_split.items():
        curriculum_id = f"curr_exec_control_v1_{split}"
        desired_ids = [entry["id"] for entry in entries]
        item = by_id.get(curriculum_id)
        if item is None:
            item = {
                "id": curriculum_id,
                "name": f"Execution Control v1 - {split.title()}",
                "timestamp": timestamp,
                "format_as_concept": True,
                "concept_block_ids": [],
            }
            curricula.append(item)
            by_id[curriculum_id] = item
        current_ids = item.setdefault("concept_block_ids", [])
        current_set = set(current_ids)
        current_ids.extend(block_id for block_id in desired_ids if block_id not in current_set)
        item["format_as_concept"] = True

    if dry_run:
        return
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = registry_path.with_suffix(registry_path.suffix + ".tmp")
    with temp_path.open("w", encoding="utf-8", newline="\n") as output:
        json.dump(registry, output, ensure_ascii=False, indent=2)
        output.write("\n")
        output.flush()
        os.fsync(output.fileno())
    os.replace(temp_path, registry_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("resources/models/GRIM-text/training/data"),
    )
    parser.add_argument("--train-per-class", type=int, default=20_000)
    parser.add_argument("--validation-per-class", type=int, default=2_000)
    parser.add_argument("--test-per-class", type=int, default=2_000)
    parser.add_argument("--seed", type=int, default=260715)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    counts = {
        "train": args.train_per_class,
        "validation": args.validation_per_class,
        "test": args.test_per_class,
    }
    if any(count <= 0 for count in counts.values()):
        raise ValueError("all per-class counts must be positive")

    entries_by_split: dict[str, list[dict[str, Any]]] = {}
    for split, count in counts.items():
        entries = list(generated_entries(split, count, args.seed))
        for entry in entries:
            validate_entry(entry)
        entries_by_split[split] = entries

    concept_path = args.data_dir / "concept_blocks.jsonl"
    registry_path = args.data_dir / "curriculum_registry.json"
    appended = append_missing_blocks(concept_path, entries_by_split, args.dry_run)
    merge_curricula(registry_path, entries_by_split, args.dry_run)

    total = sum(len(entries) for entries in entries_by_split.values())
    print(f"Execution Control v1: {total} deterministic blocks")
    for split, entries in entries_by_split.items():
        execute_count = sum(e["execution_gate_target"] == "execute" for e in entries)
        noop_count = len(entries) - execute_count
        print(f"  {split}: {len(entries)} blocks ({execute_count} EXECUTE, {noop_count} NOOP)")
    print(f"  {'would append' if args.dry_run else 'appended'}: {appended}")
    print(f"  concept blocks: {concept_path}")
    print(f"  registry: {registry_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
