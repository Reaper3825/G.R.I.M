#!/usr/bin/env python3
"""Generate and optionally merge the single-step arithmetic tool curriculum.

Every ConceptBlock teaches one symbolic Determine phase followed by one
grounded arithmetic expression inside an authored TOOL span. The tool result is
never included in the span, and Update remains deliberately empty for a later
state-update curriculum.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any, Iterable


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = ROOT_DIR / ".codex_tmp" / "single_step_arithmetic_tool_v1"
DEFAULT_COUNT = 60_000
DEFAULT_SEED = 7_913
ENTRY_PREFIX = "ssatv1_"
CURRICULUM_ID = "curr_single_step_arithmetic_tool_v1"
CURRICULUM_NAME = "Single-Step Arithmetic Tool v1"
COURSE_ID = "course_single_step_arithmetic_tool_v1"
COURSE_NAME = "Single-Step Arithmetic Tool Use"

MODES = ("direct", "contextual")
OPERATIONS = ("add", "sub", "mul", "div")
CATEGORIES = ("metric", "imperial", "count")
SYMBOLS = {"add": "+", "sub": "-", "mul": "*", "div": "/"}
OP_WORDS = {
    "add": "addition",
    "sub": "subtraction",
    "mul": "multiplication",
    "div": "division",
}


@dataclass(frozen=True)
class UnitSpec:
    unit: str
    slug: str
    subject: str


UNITS: dict[str, tuple[UnitSpec, ...]] = {
    "metric": (
        UnitSpec("millimeters", "millimeters", "measurement"),
        UnitSpec("centimeters", "centimeters", "measurement"),
        UnitSpec("meters", "meters", "cable"),
        UnitSpec("kilometers", "kilometers", "route"),
        UnitSpec("milligrams", "milligrams", "sample"),
        UnitSpec("grams", "grams", "shipment"),
        UnitSpec("kilograms", "kilograms", "shipment"),
        UnitSpec("milliliters", "milliliters", "tank"),
        UnitSpec("liters", "liters", "tank"),
    ),
    "imperial": (
        UnitSpec("inches", "inches", "board"),
        UnitSpec("feet", "feet", "rope"),
        UnitSpec("yards", "yards", "fabric roll"),
        UnitSpec("miles", "miles", "route"),
        UnitSpec("ounces", "ounces", "package"),
        UnitSpec("pounds", "pounds", "shipment"),
        UnitSpec("fluid ounces", "fluid_ounces", "container"),
        UnitSpec("gallons", "gallons", "tank"),
    ),
    "count": (
        UnitSpec("apples", "apples", "basket"),
        UnitSpec("places", "places", "travel guide"),
        UnitSpec("people", "people", "event"),
        UnitSpec("books", "books", "shelf"),
        UnitSpec("tickets", "tickets", "ticket office"),
        UnitSpec("chairs", "chairs", "hall"),
        UnitSpec("boxes", "boxes", "warehouse"),
        UnitSpec("packages", "packages", "depot"),
        UnitSpec("students", "students", "class"),
        UnitSpec("vehicles", "vehicles", "parking area"),
        UnitSpec("marbles", "marbles", "bag"),
        UnitSpec("tools", "tools", "workshop"),
    ),
}

DIRECT_TEMPLATES = {
    "add": (
        "Calculate the combined amount of {a} {unit} and {b} {unit}.",
        "Add {a} {unit} to {b} {unit}. What is the total?",
        "Find the sum of {a} {unit} and {b} {unit}.",
        "What do {a} {unit} plus {b} {unit} equal?",
    ),
    "sub": (
        "Calculate {a} {unit} minus {b} {unit}.",
        "Subtract {b} {unit} from {a} {unit}. What remains?",
        "Find the difference between {a} {unit} and {b} {unit}.",
        "What do {a} {unit} minus {b} {unit} equal?",
    ),
    "mul": (
        "Calculate {a} groups of {b} {unit} per group.",
        "Multiply {a} by {b} {unit}. What is the total amount?",
        "Find the total for {a} equal groups containing {b} {unit} each.",
        "What do {a} groups times {b} {unit} per group equal?",
    ),
    "div": (
        "Divide {a} {unit} into {b} equal groups. What is in each group?",
        "Calculate {a} {unit} divided by {b} groups.",
        "Find the amount per group when {a} {unit} is shared across {b} groups.",
        "What does {a} {unit} divided into {b} equal parts give per part?",
    ),
}

CONTEXT_TEMPLATES = {
    "add": (
        "A {subject} has {a} {unit}. Another {b} {unit} are added. How many {unit} are there altogether?",
        "During the first period, {a} {unit} were recorded. The next period added {b} {unit}. What is the combined amount?",
        "A project used {a} {unit} and then used another {b} {unit}. How many {unit} did it use in total?",
        "One allocation contains {a} {unit}, and a second contains {b} {unit}. How many {unit} do both allocations contain?",
    ),
    "sub": (
        "A {subject} starts with {a} {unit}. After {b} {unit} are used, how many {unit} remain?",
        "A recorded total is {a} {unit}, of which {b} {unit} are removed. What amount remains?",
        "A project has an allowance of {a} {unit} and consumes {b} {unit}. How many {unit} are left?",
        "There are {a} {unit} available. If {b} {unit} are taken away, how many remain?",
    ),
    "mul": (
        "A project needs {a} equal groups with {b} {unit} in each group. How many {unit} are needed altogether?",
        "Each of {a} batches contains {b} {unit}. What is the total number of {unit}?",
        "A plan repeats an allocation of {b} {unit} across {a} groups. How many {unit} does the plan allocate?",
        "There are {a} sections, each accounting for {b} {unit}. What is their combined amount?",
    ),
    "div": (
        "A total of {a} {unit} is divided into {b} equal groups. How many {unit} are in each group?",
        "A {subject} accounts for {a} {unit} across {b} equal sections. What is the amount per section?",
        "A project distributes {a} {unit} evenly among {b} groups. How many {unit} does each group receive?",
        "There are {a} {unit} to be shared equally across {b} allocations. How many {unit} belong to each allocation?",
    ),
}


def format_decimal(value: Decimal) -> str:
    rendered = format(value, "f")
    if "." in rendered:
        rendered = rendered.rstrip("0").rstrip(".")
    return rendered or "0"


def arithmetic_values(operation: str, occurrence: int, decimal_values: bool) -> tuple[Decimal, Decimal, Decimal]:
    first = 10 + (occurrence * 37) % 490
    second = 2 + (occurrence * 53) % 97
    # Coprime periods keep all 2,500 rows in each operation/mode/category
    # stratum prompt-unique without injecting artificial row identifiers.
    group_count = 2 + (occurrence * 11) % 97
    per_group = 2 + (occurrence * 29) % 499
    scale = Decimal(10) if decimal_values else Decimal(1)

    if operation == "add":
        lhs = Decimal(first) / scale
        rhs = Decimal(second) / scale
        return lhs, rhs, lhs + rhs
    if operation == "sub":
        result = Decimal(first) / scale
        rhs = Decimal(second) / scale
        return result + rhs, rhs, result
    if operation == "mul":
        lhs = Decimal(group_count)
        rhs = Decimal(per_group) / scale
        return lhs, rhs, lhs * rhs
    if operation == "div":
        rhs = Decimal(group_count)
        result = Decimal(per_group) / scale
        return rhs * result, rhs, result
    raise ValueError(f"unsupported operation {operation}")


def variable_names(operation: str, slug: str) -> tuple[str, str, str]:
    if operation == "add":
        return f"starting_{slug}", f"added_{slug}", f"total_{slug}"
    if operation == "sub":
        return f"available_{slug}", f"used_{slug}", f"remaining_{slug}"
    if operation == "mul":
        return "group_count", f"{slug}_per_group", f"total_{slug}"
    if operation == "div":
        return f"total_{slug}", "group_count", f"{slug}_per_group"
    raise ValueError(f"unsupported operation {operation}")


def known_bindings(operation: str, names: tuple[str, str, str], lhs: str, rhs: str, unit: str) -> list[str]:
    lhs_name, rhs_name, _ = names
    if operation in ("add", "sub"):
        return [f"{lhs_name} = {lhs} {unit}", f"{rhs_name} = {rhs} {unit}"]
    if operation == "mul":
        return [f"{lhs_name} = {lhs} groups", f"{rhs_name} = {rhs} {unit} per group"]
    return [f"{lhs_name} = {lhs} {unit}", f"{rhs_name} = {rhs} groups"]


def answer_text(operation: str, result: str, unit: str) -> str:
    if operation == "add":
        return f"There are {result} {unit} altogether."
    if operation == "sub":
        return f"There are {result} {unit} remaining."
    if operation == "mul":
        return f"The total is {result} {unit}."
    return f"Each group receives {result} {unit}."


def make_entry(index: int, seed: int = DEFAULT_SEED) -> tuple[dict[str, Any], dict[str, Any]]:
    del seed  # Reserved so future versions can add seeded wording without changing the CLI.
    stratum_count = len(MODES) * len(OPERATIONS) * len(CATEGORIES)
    stratum = index % stratum_count
    occurrence = index // stratum_count
    mode_index = stratum // (len(OPERATIONS) * len(CATEGORIES))
    remainder = stratum % (len(OPERATIONS) * len(CATEGORIES))
    operation_index = remainder // len(CATEGORIES)
    category_index = remainder % len(CATEGORIES)
    mode = MODES[mode_index]
    operation = OPERATIONS[operation_index]
    category = CATEGORIES[category_index]
    unit_specs = UNITS[category]
    spec = unit_specs[(occurrence * 7 + operation_index * 3) % len(unit_specs)]
    decimal_values = category != "count" and occurrence % 4 == 1
    lhs_value, rhs_value, result_value = arithmetic_values(
        operation, occurrence, decimal_values)
    lhs = format_decimal(lhs_value)
    rhs = format_decimal(rhs_value)
    result = format_decimal(result_value)
    symbol = SYMBOLS[operation]
    parenthesized = occurrence % 2 == 0
    expression = f"{lhs} {symbol} {rhs}"
    tool_expression = f"({expression})" if parenthesized else expression
    names = variable_names(operation, spec.slug)
    lhs_name, rhs_name, result_name = names
    templates = DIRECT_TEMPLATES if mode == "direct" else CONTEXT_TEMPLATES
    prompt = templates[operation][occurrence % len(templates[operation])].format(
        a=lhs, b=rhs, unit=spec.unit, subject=spec.subject)
    evidence = {
        "add": "The two stated quantities provide the addends for one addition.",
        "sub": "The stated available and used quantities provide the operands for one subtraction.",
        "mul": "The stated group count and amount per group provide the factors for one multiplication.",
        "div": "The stated total and group count provide the operands for one division.",
    }[operation]

    entry = {
        "id": f"{ENTRY_PREFIX}{index:06d}",
        "name": f"Single-step arithmetic tool: {mode} {OP_WORDS[operation]} {category}",
        "prompt": prompt,
        "knowns": known_bindings(operation, names, lhs, rhs, spec.unit),
        "unknowns": [result_name],
        "determine": f"{lhs_name} {symbol} {rhs_name} = {result_name}",
        "execute": f"<TOOL>{tool_expression}</TOOL>",
        "update": "",
        "answer": answer_text(operation, result, spec.unit),
        "goal": {
            "target_state": f"{result_name} is correctly determined",
            "success_criteria": [{
                "criterion": f"{result_name} is derived from the quantities stated in the problem",
                "evidence": evidence,
            }],
            "constraints": [
                "Use only the quantities stated in the problem.",
                f"Preserve the unit {spec.unit}.",
                "Use exactly one single-step arithmetic tool call.",
            ],
        },
        "format_type": "derivation",
        "source_sequence_id": f"synthetic_single_step_arithmetic_tool_v1:{mode}:{operation}:{category}",
        "timestamp": 0,
    }
    metadata = {
        "mode": mode,
        "operation": operation,
        "category": category,
        "unit": spec.unit,
        "parenthesized": parenthesized,
        "lhs": lhs_value,
        "rhs": rhs_value,
        "result": result_value,
    }
    return entry, metadata


def validate_entry(entry: dict[str, Any], metadata: dict[str, Any], seen_ids: set[str], seen_prompts: set[str]) -> None:
    expected_fields = {
        "id", "name", "prompt", "knowns", "unknowns", "determine", "execute",
        "update", "answer", "goal", "format_type", "source_sequence_id", "timestamp",
    }
    if set(entry) != expected_fields:
        raise ValueError(f"{entry.get('id', '<missing>')}: unexpected fields")
    block_id = entry["id"]
    if not isinstance(block_id, str) or not block_id.startswith(ENTRY_PREFIX):
        raise ValueError(f"invalid block id {block_id!r}")
    if block_id in seen_ids:
        raise ValueError(f"duplicate block id {block_id}")
    if not entry["prompt"] or entry["prompt"] in seen_prompts:
        raise ValueError(f"{block_id}: empty or duplicate prompt")
    if len(entry["knowns"]) != 2 or len(entry["unknowns"]) != 1:
        raise ValueError(f"{block_id}: expected two knowns and one unknown")
    if entry["update"] != "":
        raise ValueError(f"{block_id}: Update must remain empty")
    if entry["execute"].count("<TOOL>") != 1 or entry["execute"].count("</TOOL>") != 1:
        raise ValueError(f"{block_id}: expected exactly one TOOL span")
    payload = entry["execute"][len("<TOOL>"):-len("</TOOL>")]
    expected_expression = (
        f"{format_decimal(metadata['lhs'])} {SYMBOLS[metadata['operation']]} "
        f"{format_decimal(metadata['rhs'])}"
    )
    expected_payload = f"({expected_expression})" if metadata["parenthesized"] else expected_expression
    if payload != expected_payload or "=" in payload:
        raise ValueError(f"{block_id}: invalid TOOL payload {payload!r}")
    if metadata["operation"] == "add":
        computed = metadata["lhs"] + metadata["rhs"]
    elif metadata["operation"] == "sub":
        computed = metadata["lhs"] - metadata["rhs"]
    elif metadata["operation"] == "mul":
        computed = metadata["lhs"] * metadata["rhs"]
    else:
        if metadata["rhs"] == 0:
            raise ValueError(f"{block_id}: division by zero")
        computed = metadata["lhs"] / metadata["rhs"]
    if computed != metadata["result"]:
        raise ValueError(f"{block_id}: arithmetic result mismatch")
    result_text = format_decimal(metadata["result"])
    if result_text not in entry["answer"] or metadata["unit"] not in entry["answer"]:
        raise ValueError(f"{block_id}: answer does not preserve result and unit")
    goal = entry["goal"]
    if set(goal) != {"target_state", "success_criteria", "constraints"}:
        raise ValueError(f"{block_id}: malformed goal")
    if len(goal["success_criteria"]) != 1 or not goal["constraints"]:
        raise ValueError(f"{block_id}: incomplete goal")
    if entry["format_type"] != "derivation" or entry["timestamp"] != 0:
        raise ValueError(f"{block_id}: invalid format metadata")
    serialized = json.dumps(entry, ensure_ascii=False, separators=(",", ":"))
    for marker in ("<INT>", "<FLOAT>", "<STRING>", "<BOOL>", "<ENTITY>"):
        if marker in serialized:
            raise ValueError(f"{block_id}: unexpected atom marker {marker}")
    seen_ids.add(block_id)
    seen_prompts.add(entry["prompt"])


def atomic_text_path(destination: Path) -> tuple[Path, Any]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, stage_name = tempfile.mkstemp(
        prefix=destination.name + ".", suffix=".tmp", dir=destination.parent)
    return Path(stage_name), os.fdopen(descriptor, "w", encoding="utf-8", newline="\n")


def generated_registry(ids: list[str]) -> dict[str, Any]:
    return {
        "schema_version": 2,
        "curriculums": [{
            "id": CURRICULUM_ID,
            "name": CURRICULUM_NAME,
            "course_ids": [COURSE_ID],
            "concept_block_ids": ids,
            "plaintext_block_ids": [],
            "training_stage": "sft",
            "timestamp": 0,
            "randomize_course_order": False,
            "randomize_concept_block_order": False,
            "format_as_concept": True,
        }],
        "courses": [{
            "id": COURSE_ID,
            "name": COURSE_NAME,
            "concept_block_ids": ids,
        }],
    }


def write_json_atomic(path: Path, value: Any) -> None:
    stage, output = atomic_text_path(path)
    try:
        with output:
            json.dump(value, output, ensure_ascii=False, indent=2)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(stage, path)
    finally:
        stage.unlink(missing_ok=True)


def generate_dataset(output_dir: Path, count: int, seed: int) -> dict[str, Any]:
    if count != DEFAULT_COUNT:
        raise ValueError(f"this course requires exactly {DEFAULT_COUNT:,} entries")
    jsonl_path = output_dir / "concept_blocks.jsonl"
    registry_path = output_dir / "curriculum_registry.json"
    manifest_path = output_dir / "single_step_arithmetic_tool_v1_manifest.json"
    seen_ids: set[str] = set()
    seen_prompts: set[str] = set()
    family_counts: collections.Counter[str] = collections.Counter()
    unit_counts: collections.Counter[str] = collections.Counter()
    parenthesized_count = 0
    ids: list[str] = []
    digest = hashlib.sha256()

    stage, output = atomic_text_path(jsonl_path)
    try:
        with output:
            for index in range(count):
                entry, metadata = make_entry(index, seed)
                validate_entry(entry, metadata, seen_ids, seen_prompts)
                serialized = json.dumps(entry, ensure_ascii=False, separators=(",", ":"))
                encoded_line = (serialized + "\n").encode("utf-8")
                output.write(encoded_line.decode("utf-8"))
                digest.update(encoded_line)
                ids.append(entry["id"])
                family_counts[
                    f"{metadata['mode']}:{metadata['operation']}:{metadata['category']}"
                ] += 1
                unit_counts[metadata["unit"]] += 1
                parenthesized_count += int(metadata["parenthesized"])
            output.flush()
            os.fsync(output.fileno())
        os.replace(stage, jsonl_path)
    finally:
        stage.unlink(missing_ok=True)

    expected_family_count = count // (len(MODES) * len(OPERATIONS) * len(CATEGORIES))
    if len(family_counts) != 24 or set(family_counts.values()) != {expected_family_count}:
        raise ValueError(f"family balance failed: {dict(family_counts)}")
    if parenthesized_count != count // 2:
        raise ValueError(f"parenthesis balance failed: {parenthesized_count}")

    write_json_atomic(registry_path, generated_registry(ids))
    manifest = {
        "dataset_id": "single_step_arithmetic_tool_v1",
        "curriculum_id": CURRICULUM_ID,
        "course_id": COURSE_ID,
        "generator": "scripts/generate_single_step_arithmetic_tool_curriculum.py",
        "seed": seed,
        "entry_count": count,
        "mode_counts": {
            mode: sum(value for family, value in family_counts.items() if family.startswith(mode + ":"))
            for mode in MODES
        },
        "operation_counts": {
            operation: sum(value for family, value in family_counts.items() if f":{operation}:" in family)
            for operation in OPERATIONS
        },
        "unit_category_counts": {
            category: sum(value for family, value in family_counts.items() if family.endswith(":" + category))
            for category in CATEGORIES
        },
        "family_counts": dict(sorted(family_counts.items())),
        "unit_counts": dict(sorted(unit_counts.items())),
        "parenthesized_tool_calls": parenthesized_count,
        "non_parenthesized_tool_calls": count - parenthesized_count,
        "update_policy": "the update field is intentionally empty in every entry",
        "tool_payload_policy": "one grounded arithmetic expression with no result inside one TOOL span",
        "sha256_concept_blocks_jsonl": digest.hexdigest(),
    }
    write_json_atomic(manifest_path, manifest)
    return manifest


def replace_one(items: list[dict[str, Any]], replacement: dict[str, Any], item_id: str, label: str) -> None:
    matches = [index for index, item in enumerate(items) if item.get("id") == item_id]
    if len(matches) > 1:
        raise ValueError(f"duplicate {label} id {item_id}")
    if matches:
        items[matches[0]] = replacement
    else:
        items.append(replacement)


def merge_into_existing_dataset(source_dir: Path, data_dir: Path) -> tuple[int, int]:
    source_jsonl = source_dir / "concept_blocks.jsonl"
    source_registry_path = source_dir / "curriculum_registry.json"
    target_jsonl = data_dir / "concept_blocks.jsonl"
    target_registry_path = data_dir / "curriculum_registry.json"
    for path in (source_jsonl, source_registry_path, target_jsonl, target_registry_path):
        if not path.is_file():
            raise FileNotFoundError(f"required merge input is missing: {path}")

    source_registry = json.loads(source_registry_path.read_text(encoding="utf-8"))
    target_registry = json.loads(target_registry_path.read_text(encoding="utf-8"))
    if source_registry.get("schema_version") != 2 or target_registry.get("schema_version") != 2:
        raise ValueError("both curriculum registries must use schema_version 2")
    source_curricula = source_registry.get("curriculums", [])
    source_courses = source_registry.get("courses", [])
    if len(source_curricula) != 1 or len(source_courses) != 1:
        raise ValueError("generated registry must contain one curriculum and one course")
    curricula = target_registry.get("curriculums")
    courses = target_registry.get("courses")
    if not isinstance(curricula, list) or not isinstance(courses, list):
        raise ValueError("target registry must contain curriculum and course arrays")
    replace_one(curricula, source_curricula[0], CURRICULUM_ID, "curriculum")
    replace_one(courses, source_courses[0], COURSE_ID, "course")

    registry_stage, registry_output = atomic_text_path(target_registry_path)
    jsonl_stage, jsonl_output = atomic_text_path(target_jsonl)
    kept = 0
    removed = 0
    owned_marker = f'"id":"{ENTRY_PREFIX}'.encode("ascii")
    try:
        with registry_output:
            json.dump(target_registry, registry_output, ensure_ascii=False, indent=2)
            registry_output.write("\n")
            registry_output.flush()
            os.fsync(registry_output.fileno())

        jsonl_output.close()
        with target_jsonl.open("rb") as existing, jsonl_stage.open("wb") as output:
            last_line_had_newline = True
            for line_number, line in enumerate(existing, 1):
                if not line.strip():
                    raise ValueError(f"{target_jsonl}:{line_number}: blank JSONL row")
                if owned_marker in line:
                    row = json.loads(line)
                    block_id = row.get("id")
                    if not isinstance(block_id, str) or not block_id.startswith(ENTRY_PREFIX):
                        raise ValueError(f"{target_jsonl}:{line_number}: invalid owned row")
                    removed += 1
                    continue
                output.write(line)
                last_line_had_newline = line.endswith(b"\n")
                kept += 1
            if kept and not last_line_had_newline:
                output.write(b"\n")
            with source_jsonl.open("rb") as generated:
                while chunk := generated.read(8 * 1024 * 1024):
                    output.write(chunk)
            output.flush()
            os.fsync(output.fileno())

        os.replace(jsonl_stage, target_jsonl)
        os.replace(registry_stage, target_registry_path)
    finally:
        if not jsonl_output.closed:
            jsonl_output.close()
        jsonl_stage.unlink(missing_ok=True)
        registry_stage.unlink(missing_ok=True)
    return removed, kept


def sample_entries(indices: Iterable[int], seed: int = DEFAULT_SEED) -> list[dict[str, Any]]:
    return [make_entry(index, seed)[0] for index in indices]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--count", type=int, default=DEFAULT_COUNT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--merge-data-dir",
        type=Path,
        help="replace owned rows and register the course in an existing data directory",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    manifest = generate_dataset(output_dir, args.count, args.seed)
    print(f"{CURRICULUM_NAME}: {manifest['entry_count']:,} validated ConceptBlocks")
    print(f"  modes: {manifest['mode_counts']}")
    print(f"  operations: {manifest['operation_counts']}")
    print(f"  unit categories: {manifest['unit_category_counts']}")
    print(f"  parenthesized tool calls: {manifest['parenthesized_tool_calls']:,}")
    print(f"  output: {output_dir}")
    if args.merge_data_dir is not None:
        removed, kept = merge_into_existing_dataset(output_dir, args.merge_data_dir.resolve())
        print(
            f"  merged into: {args.merge_data_dir.resolve()} "
            f"(replaced {removed:,} prior owned rows; preserved {kept:,} unrelated rows)"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
