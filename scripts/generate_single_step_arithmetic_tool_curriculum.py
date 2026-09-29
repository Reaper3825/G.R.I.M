#!/usr/bin/env python3
"""Generate and optionally merge the single-step arithmetic tool curriculum.

Every ConceptBlock teaches a natural-language Determine and local Define followed by one
variable-based arithmetic expression inside an authored TOOL span. The
postfix pointer binds the pending tool result to the unknown, and the answer
references that same variable. Update remains deliberately empty for a later
state-update curriculum.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import re
import tempfile
from functools import lru_cache
from decimal import Decimal
from pathlib import Path
from typing import Any, Iterable

try:
    from .single_step_arithmetic_scenarios import STORIES, RELATIONSHIPS, TARGETS, make_case, render_case
except ImportError:  # Direct script execution and unittest discovery.
    from single_step_arithmetic_scenarios import STORIES, RELATIONSHIPS, TARGETS, make_case, render_case


ROOT_DIR = Path(__file__).resolve().parents[1]
MANUAL_VOCAB_PATH = ROOT_DIR / "resources/models/GRIM-text/training/manual_vocab.json"
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

def format_decimal(value: Decimal) -> str:
    rendered = format(value, "f")
    if "." in rendered:
        rendered = rendered.rstrip("0").rstrip(".")
    return rendered or "0"


def variable_reference(name: str) -> str:
    return f"${{{name}}}"


def arithmetic_values(operation: str, occurrence: int, decimal_values: bool) -> tuple[Decimal, Decimal, Decimal]:
    first = 10 + (occurrence * 37) % 490
    second = 2 + (occurrence * 53) % 97
    # Coprime periods keep all 1,000 rows in each direct operation/category
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
        return f"base_{slug}", f"quantity_{slug}", f"total_{slug}"
    if operation == "sub":
        return f"base_{slug}", f"change_{slug}", f"difference_{slug}"
    if operation == "mul":
        return "group_count", f"quantity_{slug}", f"total_{slug}"
    if operation == "div":
        return f"total_{slug}", "group_count", f"quantity_{slug}"
    raise ValueError(f"unsupported operation {operation}")


def local_definitions(names: tuple[str, str, str], lhs: str, rhs: str) -> str:
    lhs_name, rhs_name, result_name = names
    return "\n".join((
        f"{variable_reference(lhs_name)} -> {lhs};",
        f"{variable_reference(rhs_name)} -> {rhs};",
        f"{variable_reference(result_name)};",
    ))


DETERMINE = {
    "add": "Add the two stated quantities to find their sum.",
    "sub": "Compute the stated subtraction to find the difference.",
    "mul": "Multiply the number of groups by the quantity in each group to find the total amount.",
    "div": "Divide the total quantity by the number of equal groups to find the amount per group.",
}


def answer_text(operation: str, result: str, unit: str) -> str:
    if operation == "add":
        return f"There are {result} {unit} altogether."
    if operation == "sub":
        return f"The difference is {result} {unit}."
    if operation == "mul":
        return f"The total is {result} {unit}."
    return f"Each group receives {result} {unit}."


def make_entry(index: int, seed: int = DEFAULT_SEED) -> tuple[dict[str, Any], dict[str, Any]]:
    if not 0 <= index < DEFAULT_COUNT:
        raise ValueError(f"entry index out of range: {index}")
    cycle, slot = divmod(index, 5)
    if slot == 0:
        mode = "direct"
        occurrence, stratum = divmod(cycle, 12)
        operation = OPERATIONS[stratum // 3]
        category = CATEGORIES[stratum % 3]
        story = STORIES[category][occurrence % len(STORIES[category])]
        lhs_value, rhs_value, result_value = arithmetic_values(
            operation, occurrence + seed, category != "count" and occurrence % 4 == 1)
        names = variable_names(operation, story.suffix)
        prompt = DIRECT_TEMPLATES[operation][(occurrence // len(STORIES[category])) % 4].format(
            a=format_decimal(lhs_value), b=format_decimal(rhs_value), unit=story.unit)
        result_ref = variable_reference(names[2])
        answer = answer_text(operation, result_ref, story.unit)
        metadata = dict(operation=operation, category=category, unit=story.unit,
                        input_unit=story.unit, names=names, lhs=lhs_value, rhs=rhs_value,
                        result=result_value, prompt=prompt, determine=DETERMINE[operation],
                        answer=answer, relationship="direct", target_role=operation,
                        wording=(occurrence // len(STORIES[category])) % 4,
                        question_first=False, reverse_facts=False)
    else:
        mode = "contextual"
        occurrence, stratum = divmod(cycle * 4 + slot - 1, 12)
        relationship = RELATIONSHIPS[stratum // 3]
        target = stratum % 3
        metadata = render_case(make_case(relationship, occurrence, seed), target)
        metadata["case_id"] = f"{relationship}:{occurrence}"
    metadata["mode"] = mode
    metadata["parenthesized"] = index % 2 == 0
    operation, category = metadata["operation"], metadata["category"]
    names = metadata["names"]
    lhs_name, rhs_name, result_name = names
    expression = (
        f"{variable_reference(lhs_name)} {SYMBOLS[operation]} "
        f"{variable_reference(rhs_name)}"
    )
    tool_expression = f"({expression})" if metadata["parenthesized"] else expression
    result_reference = variable_reference(result_name)

    entry = {
        "id": f"{ENTRY_PREFIX}{index:06d}",
        "name": f"Single-step arithmetic tool: {mode} {metadata['relationship']} {metadata['target_role']} {OP_WORDS[operation]} {category}",
        "prompt": metadata["prompt"],
        "knowns": [],
        "unknowns": [],
        "determine": metadata["determine"],
        "define": local_definitions(names, format_decimal(metadata["lhs"]), format_decimal(metadata["rhs"])),
        "execute": f"<TOOL>{tool_expression}</TOOL> -> {result_reference}",
        "update": "",
        "answer": metadata["answer"],
        "goal": {
            "target_state": "The quantity requested in the question is correctly reported.",
            "success_criteria": [{
                "criterion": "The response answers the question using the stated quantities.",
                "evidence": "",
            }],
            "constraints": [
                "Use only the quantities stated in the problem.",
                "Report the quantity in the unit requested by the question.",
                "Use exactly one single-step arithmetic tool call.",
            ],
        },
        "format_type": "derivation",
        "source_sequence_id": f"synthetic_single_step_arithmetic_tool_v1_roles:{mode}:{metadata.get('case_id', str(index))}",
        "timestamp": 0,
    }
    return entry, metadata


@lru_cache(maxsize=1)
def manual_variable_stems() -> frozenset[str]:
    tokens = json.loads(MANUAL_VOCAB_PATH.read_text(encoding="utf-8"))["tokens"]
    return frozenset(token[:-1] for token in tokens if re.fullmatch(r"[a-z]+_", token))


def validate_variable_names(names: tuple[str, str, str]) -> None:
    stems = manual_variable_stems()
    if len(set(names)) != 3:
        raise ValueError("local names must be distinct")
    for name in names:
        parts = name.split("_")
        if len(parts) != 2 or any(part not in stems for part in parts):
            raise ValueError(f"variable must pair two manual-vocabulary stems: {name}")


def validate_entry(entry: dict[str, Any], metadata: dict[str, Any], seen_ids: set[str], seen_prompts: set[str]) -> None:
    expected_fields = {
        "id", "name", "prompt", "knowns", "unknowns", "determine", "define", "execute",
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
    if entry["knowns"] != [] or entry["unknowns"] != []:
        raise ValueError(f"{block_id}: persisted state must be empty")
    if entry["define"] != local_definitions(
            metadata["names"], format_decimal(metadata["lhs"]), format_decimal(metadata["rhs"])):
        raise ValueError(f"{block_id}: invalid local definitions")
    validate_variable_names(metadata["names"])
    if entry["determine"] != metadata["determine"] or re.search(r"\d|\$\{|_", entry["determine"]):
        raise ValueError(f"{block_id}: invalid natural-language determine")
    if entry["prompt"] != metadata["prompt"] or entry["answer"] != metadata["answer"]:
        raise ValueError(f"{block_id}: prompt/answer role mismatch")
    if entry["update"] != "":
        raise ValueError(f"{block_id}: Update must remain empty")
    if entry["execute"].count("<TOOL>") != 1 or entry["execute"].count("</TOOL>") != 1:
        raise ValueError(f"{block_id}: expected exactly one TOOL span")
    close_index = entry["execute"].find("</TOOL>")
    payload = entry["execute"][len("<TOOL>"):close_index]
    lhs_name, rhs_name, result_name = metadata["names"]
    expected_expression = (
        f"{variable_reference(lhs_name)} {SYMBOLS[metadata['operation']]} "
        f"{variable_reference(rhs_name)}"
    )
    expected_payload = f"({expected_expression})" if metadata["parenthesized"] else expected_expression
    expected_execute = (
        f"<TOOL>{expected_payload}</TOOL> -> "
        f"{variable_reference(result_name)}"
    )
    if entry["execute"] != expected_execute or payload != expected_payload or "=" in payload:
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
    result_reference = variable_reference(result_name)
    if result_reference not in entry["answer"] or metadata["unit"] not in entry["answer"]:
        raise ValueError(f"{block_id}: answer does not preserve result reference and unit")
    if format_decimal(metadata["result"]) in entry["answer"]:
        raise ValueError(f"{block_id}: answer leaks the resolved arithmetic value")
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
    vocab_bytes = MANUAL_VOCAB_PATH.read_bytes()
    vocab = json.loads(vocab_bytes)["tokens"]
    manual_variable_stems.cache_clear()
    pieces: set[str] = set()
    jsonl_path = output_dir / "concept_blocks.jsonl"
    registry_path = output_dir / "curriculum_registry.json"
    manifest_path = output_dir / "single_step_arithmetic_tool_v1_manifest.json"
    seen_ids: set[str] = set()
    seen_prompts: set[str] = set()
    family_counts: collections.Counter[str] = collections.Counter()
    unit_counts: collections.Counter[str] = collections.Counter()
    operation_counts: collections.Counter[str] = collections.Counter()
    role_counts: collections.Counter[str] = collections.Counter()
    wording_counts: collections.Counter[str] = collections.Counter()
    shape_counts: collections.Counter[str] = collections.Counter()
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
                    f"{metadata['mode']}:{metadata['relationship']}:{metadata['target_role']}:{metadata['category']}"
                ] += 1
                unit_counts[metadata["input_unit"]] += 1
                operation_counts[metadata["operation"]] += 1
                role_counts[f"{metadata['relationship']}:{metadata['target_role']}"] += 1
                wording_counts[f"{metadata['relationship']}:{metadata['wording']}"] += 1
                shape_counts[f"question_first:{metadata['question_first']}"] += 1
                shape_counts[f"reverse_facts:{metadata['reverse_facts']}"] += 1
                pieces.update(part + "_" for name in metadata["names"] for part in name.split("_"))
                parenthesized_count += int(metadata["parenthesized"])
            expected_roles = {f"direct:{op}": 3_000 for op in OPERATIONS}
            expected_roles.update({f"{rel}:{role}": 4_000 for rel in RELATIONSHIPS for role in TARGETS[rel]})
            if dict(role_counts) != expected_roles:
                raise ValueError(f"role balance failed: {dict(role_counts)}")
            if parenthesized_count != count // 2:
                raise ValueError(f"parenthesis balance failed: {parenthesized_count}")
            if missing := pieces - vocab.keys():
                raise ValueError(f"missing manual-vocabulary pieces: {sorted(missing)}")
            output.flush()
            os.fsync(output.fileno())
        os.replace(stage, jsonl_path)
    finally:
        stage.unlink(missing_ok=True)

    write_json_atomic(registry_path, generated_registry(ids))
    manifest = {
        "dataset_id": "single_step_arithmetic_tool_v1",
        "revision": "semantic_roles_v2",
        "curriculum_id": CURRICULUM_ID,
        "course_id": COURSE_ID,
        "generator": "scripts/generate_single_step_arithmetic_tool_curriculum.py",
        "seed": seed,
        "entry_count": count,
        "definition_policy": "two ${variable} -> value; bindings and one ${variable}; declaration; persisted knowns/unknowns empty",
        "manual_vocab": str(MANUAL_VOCAB_PATH.relative_to(ROOT_DIR)),
        "manual_vocab_sha256": hashlib.sha256(vocab_bytes).hexdigest(),
        "variable_name_policy": "exactly two stems, each backed by an existing manual-vocabulary stem_ entry",
        "variable_pieces": sorted(pieces),
        "mode_counts": {
            mode: sum(value for family, value in family_counts.items() if family.startswith(mode + ":"))
            for mode in MODES
        },
        "operation_counts": dict(sorted(operation_counts.items())),
        "role_counts": dict(sorted(role_counts.items())),
        "wording_counts": dict(sorted(wording_counts.items())),
        "prompt_shape_counts": dict(sorted(shape_counts.items())),
        "unit_category_counts": {
            category: sum(value for family, value in family_counts.items() if family.endswith(":" + category))
            for category in CATEGORIES
        },
        "family_counts": dict(sorted(family_counts.items())),
        "unit_counts": dict(sorted(unit_counts.items())),
        "parenthesized_tool_calls": parenthesized_count,
        "non_parenthesized_tool_calls": count - parenthesized_count,
        "update_policy": "the update field is intentionally empty in every entry",
        "tool_payload_policy": (
            "one variable-based arithmetic expression inside one TOOL span, "
            "followed by one ${variable} result pointer"
        ),
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
    if source_dir.resolve() == data_dir.resolve():
        raise ValueError("generation and merge directories must be different")
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
                row = json.loads(line)
                block_id = row.get("id")
                if not isinstance(block_id, str):
                    raise ValueError(f"{target_jsonl}:{line_number}: missing row id")
                if block_id.startswith(ENTRY_PREFIX):
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
