#!/usr/bin/env python3
"""Generate the counterfactual GRIM-text Execution Policy v2 curriculum.

The curriculum teaches the EXECUTE/NOOP gate whether the current prompt needs
the arithmetic Execution Block.  Every generated scenario is a matched pair:
both rows share the same numeric context, while one question requires a
supported computation and the other asks for an explicitly stated value.

The script owns one curriculum only.  Train/validation sampling remains the
training orchestrator's responsibility.
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import os
import random
import re
import shutil
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable


CURRICULUM_ID = "curr_execution_policy_v2"
CURRICULUM_NAME = "Execution Policy v2"
ENTRY_PREFIX = "epv2_"
LEGACY_CURRICULUM_IDS = {
    "curr_exec_control_v1_train",
    "curr_exec_control_v1_validation",
    "curr_exec_control_v1_test",
}
OPS = {"add", "sub", "mul", "div"}
NUMBER_RE = re.compile(r"(?<![A-Za-z0-9_])[-+]?\d+(?:\.\d+)?")
WORD_RE = re.compile(r"[a-z]+(?:'[a-z]+)?")
CONTROL_CUE_WORDS = {"calculate", "compute", "evaluate", "solve"}
MAX_LEXICAL_LABEL_GAP = 0.20

PROMPT_LAYOUTS: tuple[tuple[str, bool], ...] = (
    ("{context} {request}", False),
    ("Report: {context} {request}", False),
    ("Details: {context} Question: {request}", False),
    ("Question: {request} Details: {context}", True),
    ("Task: {request} Supporting record: {context}", True),
    ("Record: {context} Requested value: {request}", False),
    ("Requested value: {request} Record: {context}", True),
    ("Read this entry: {context} Then respond: {request}", False),
    ("Consider this question: {request} Here is the relevant entry: {context}", True),
)

NAMES = (
    "Avery", "Blake", "Casey", "Devon", "Emery", "Finley", "Gray", "Harper",
    "Indigo", "Jordan", "Kai", "Logan", "Morgan", "Nico", "Oakley", "Parker",
)
PLACES = (
    "Harbor Depot", "North Annex", "Cedar Market", "Riverside Center",
    "Orchard Hall", "Summit Store", "Willow Station", "Granite Warehouse",
)
ITEMS = (
    "crates", "notebooks", "bottles", "tiles", "packets", "lamps",
    "cables", "folders", "bolts", "samples", "cartons", "badges",
)


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
            raise ZeroDivisionError("execution program divides by zero")
        return lhs / rhs
    raise ValueError(f"unknown operation: {op}")


def build_program(
    atoms: list[float], specs: list[tuple[str, int, int]]
) -> tuple[list[dict[str, Any]], list[str], float]:
    slots = list(atoms)
    steps: list[dict[str, Any]] = []
    explanations: list[str] = []
    for step_index, (op, lhs_slot, rhs_slot) in enumerate(specs):
        if op not in OPS:
            raise ValueError(f"unsupported operation: {op}")
        if lhs_slot < 0 or lhs_slot >= len(slots):
            raise ValueError(f"lhs slot {lhs_slot} is unavailable at step {step_index}")
        if rhs_slot < 0 or rhs_slot >= len(slots):
            raise ValueError(f"rhs slot {rhs_slot} is unavailable at step {step_index}")
        lhs = slots[lhs_slot]
        rhs = slots[rhs_slot]
        result = apply_op(op, lhs, rhs)
        if not math.isfinite(result) or abs(result) > 1_000_000.0:
            raise ValueError(f"unstable execution result at step {step_index}: {result}")
        steps.append({
            "op": op,
            "args": [lhs, rhs],
            "arg_slots": [lhs_slot, rhs_slot],
            "result": result,
        })
        symbol = {"add": "+", "sub": "-", "mul": "*", "div": "/"}[op]
        explanations.append(
            f"Step {step_index + 1}: {format_number(lhs)} {symbol} "
            f"{format_number(rhs)} = {format_number(result)}."
        )
        slots.append(result)
    if not steps:
        raise ValueError("EXECUTE rows require at least one execution step")
    return steps, explanations, slots[-1]


@dataclass(frozen=True)
class Scenario:
    family: str
    context: str
    execute_request: str
    noop_request: str
    atoms: list[float]
    program: list[tuple[str, int, int]]
    noop_answer: float


@dataclass(frozen=True)
class PolicyPair:
    pair_id: str
    family: str
    prompt_layout: int
    execute: dict[str, Any]
    noop: dict[str, Any]


def choose_common(rng: random.Random) -> tuple[str, str, str]:
    return rng.choice(NAMES), rng.choice(PLACES), rng.choice(ITEMS)


def scenario_inventory_add(rng: random.Random) -> Scenario:
    name, place, item = choose_common(rng)
    initial = rng.randint(12, 480)
    received = rng.randint(3, 190)
    context = f"At {place}, {name} counted {initial} {item} before {received} more arrived."
    return Scenario(
        "inventory_add", context,
        f"After the arrival, how many {item} were there?",
        f"Before the arrival, how many {item} had {name} counted?",
        [float(initial), float(received)], [("add", 0, 1)], float(initial))


def scenario_inventory_sub(rng: random.Random) -> Scenario:
    name, place, item = choose_common(rng)
    removed = rng.randint(3, 160)
    initial = rng.randint(removed + 5, removed + 480)
    context = f"At {place}, {name} started with {initial} {item} and sent out {removed} of them."
    return Scenario(
        "inventory_sub", context,
        f"After the shipment, how many {item} remained?",
        f"In the shipment, how many {item} were sent out?",
        [float(initial), float(removed)], [("sub", 0, 1)], float(removed))


def scenario_equal_groups(rng: random.Random) -> Scenario:
    name, place, item = choose_common(rng)
    groups = rng.randint(2, 28)
    each = rng.randint(3, 54)
    context = f"At {place}, {name} arranged {groups} shelves with {each} {item} on every shelf."
    return Scenario(
        "equal_groups", context,
        f"Across all the shelves, how many {item} were arranged?",
        "Across all the shelves, how many shelves were used?",
        [float(groups), float(each)], [("mul", 0, 1)], float(groups))


def scenario_equal_share(rng: random.Random) -> Scenario:
    name, place, item = choose_common(rng)
    containers = rng.randint(2, 24)
    each = rng.randint(3, 48)
    total = containers * each
    context = f"At {place}, {name} divided {total} {item} equally among {containers} bins."
    return Scenario(
        "equal_share", context,
        f"Across all the bins, how many {item} went into each bin?",
        f"Across all the bins, how many bins received {item}?",
        [float(total), float(containers)], [("div", 0, 1)], float(containers))


def scenario_difference(rng: random.Random) -> Scenario:
    name, place, item = choose_common(rng)
    smaller = rng.randint(8, 260)
    larger = rng.randint(smaller + 2, smaller + 230)
    context = (
        f"At {place}, {name} recorded {larger} {item} in the north section "
        f"and {smaller} in the south section."
    )
    return Scenario(
        "difference", context,
        "Between the north and south sections, how many more were recorded in the north?",
        "Between the north and south sections, how many were recorded in the south?",
        [float(larger), float(smaller)], [("sub", 0, 1)], float(smaller))


def scenario_two_step_stock(rng: random.Random) -> Scenario:
    name, place, item = choose_common(rng)
    initial = rng.randint(20, 350)
    received = rng.randint(5, 140)
    shipped = rng.randint(3, initial + received - 2)
    context = (
        f"At {place}, {name} began with {initial} {item}, received {received} more, "
        f"and later shipped {shipped}."
    )
    return Scenario(
        "two_step_stock", context,
        f"Across both changes, how many {item} remained?",
        f"Across both changes, how many {item} were received?",
        [float(initial), float(received), float(shipped)],
        [("add", 0, 1), ("sub", 3, 2)], float(received))


def scenario_two_group_total(rng: random.Random) -> Scenario:
    name, place, item = choose_common(rng)
    first_groups = rng.randint(2, 14)
    first_each = rng.randint(3, 25)
    second_groups = rng.randint(2, 14)
    second_each = rng.randint(3, 25)
    context = (
        f"At {place}, {name} packed {first_groups} cases with {first_each} {item} each "
        f"and {second_groups} cases with {second_each} each."
    )
    return Scenario(
        "two_group_total", context,
        f"Across both groups, how many {item} were packed?",
        "Across both groups, how many cases were in the second group?",
        [float(first_groups), float(first_each), float(second_groups), float(second_each)],
        [("mul", 0, 1), ("mul", 2, 3), ("add", 4, 5)], float(second_groups))


def scenario_price_with_fee(rng: random.Random) -> Scenario:
    name, place, item = choose_common(rng)
    count = rng.randint(2, 35)
    unit_price = rng.randint(2, 45)
    fee = rng.randint(1, 30)
    context = (
        f"At {place}, {name} ordered {count} {item} at {unit_price} credits each "
        f"and paid a {fee}-credit handling fee."
    )
    return Scenario(
        "price_with_fee", context,
        "Including the handling fee, what was the complete charge?",
        "Within the complete charge, what was the handling fee?",
        [float(count), float(unit_price), float(fee)],
        [("mul", 0, 1), ("add", 3, 2)], float(fee))


def scenario_share_with_bonus(rng: random.Random) -> Scenario:
    name, place, item = choose_common(rng)
    recipients = rng.randint(2, 18)
    base_each = rng.randint(3, 42)
    total = recipients * base_each
    bonus = rng.randint(1, 20)
    context = (
        f"At {place}, {name} shared {total} {item} equally among {recipients} teams, "
        f"then gave every team {bonus} extra."
    )
    return Scenario(
        "share_with_bonus", context,
        f"In all, how many {item} did each team receive?",
        f"In all, how many extra {item} did each team receive?",
        [float(total), float(recipients), float(bonus)],
        [("div", 0, 1), ("add", 3, 2)], float(bonus))


def scenario_average_pair(rng: random.Random) -> Scenario:
    name, place, item = choose_common(rng)
    first = rng.randint(10, 300)
    second = rng.randint(10, 300)
    context = (
        f"At {place}, {name} logged 2 readings for the {item}: "
        f"the first was {first} and the second was {second}."
    )
    return Scenario(
        "average_pair", context,
        "Across the two readings, what was the average reading?",
        "Across the two readings, what was the first reading?",
        [2.0, float(first), float(second)],
        [("add", 1, 2), ("div", 3, 0)], float(first))


def scenario_rate_duration(rng: random.Random) -> Scenario:
    name, place, item = choose_common(rng)
    rate = rng.randint(3, 65)
    duration = rng.randint(2, 16)
    context = (
        f"At {place}, {name}'s machine processed {rate} {item} per hour for {duration} hours."
    )
    return Scenario(
        "rate_duration", context,
        f"How many {item} did the machine process during that time?",
        f"How many {item} did the machine process per hour?",
        [float(rate), float(duration)], [("mul", 0, 1)], float(rate))


def scenario_identifier_distractor(rng: random.Random) -> Scenario:
    name, place, item = choose_common(rng)
    record_id = rng.randint(1000, 9999)
    first = rng.randint(5, 220)
    second = rng.randint(3, 180)
    context = (
        f"At {place}, record {record_id} says {name} logged {first} {item} in the morning "
        f"and {second} in the afternoon."
    )
    return Scenario(
        "identifier_distractor", context,
        f"How many {item} were logged across both periods?",
        "Across both periods, what record number identifies this entry?",
        [float(record_id), float(first), float(second)],
        [("add", 1, 2)], float(record_id))


SCENARIO_BUILDERS: tuple[Callable[[random.Random], Scenario], ...] = (
    scenario_inventory_add,
    scenario_inventory_sub,
    scenario_equal_groups,
    scenario_equal_share,
    scenario_difference,
    scenario_two_step_stock,
    scenario_two_group_total,
    scenario_price_with_fee,
    scenario_share_with_bonus,
    scenario_average_pair,
    scenario_rate_duration,
    scenario_identifier_distractor,
)


def make_base_entry(entry_id: str, name: str, question: str, answer: float) -> dict[str, Any]:
    return {
        "id": entry_id,
        "name": name,
        "question": question,
        "intermediates": [],
        "explanation": [],
        "answer": format_number(answer),
        "intermediate_count": 0,
        "step_index": [],
        "format_type": "qa",
        "timestamp": 0,
    }


def make_pair(pair_index: int, rng: random.Random) -> PolicyPair:
    builder = SCENARIO_BUILDERS[pair_index % len(SCENARIO_BUILDERS)]
    scenario = builder(rng)
    pair_id = f"{ENTRY_PREFIX}{scenario.family}_{pair_index:06d}"
    prompt_layout = rng.randrange(len(PROMPT_LAYOUTS))
    layout, _ = PROMPT_LAYOUTS[prompt_layout]
    execute_question = layout.format(
        context=scenario.context, request=scenario.execute_request)
    noop_question = layout.format(
        context=scenario.context, request=scenario.noop_request)

    steps, explanations, execute_answer = build_program(scenario.atoms, scenario.program)
    execute = make_base_entry(
        f"{pair_id}_execute",
        f"Execution Policy v2 {scenario.family} {pair_index:06d} EXECUTE",
        execute_question,
        execute_answer,
    )
    execute.update({
        "execution_gate_target": "execute",
        "state_0": {"type": "arithmetic", "atoms": scenario.atoms},
        "execution": steps,
        "state_1": {"result": execute_answer},
        "intermediates": explanations,
        "explanation": explanations,
        "intermediate_count": len(explanations),
        "step_index": list(range(len(explanations))),
    })

    noop = make_base_entry(
        f"{pair_id}_noop",
        f"Execution Policy v2 {scenario.family} {pair_index:06d} NOOP",
        noop_question,
        scenario.noop_answer,
    )
    noop["execution_gate_target"] = "noop"
    return PolicyPair(pair_id, scenario.family, prompt_layout, execute, noop)


def validate_entry(entry: dict[str, Any], max_slots: int = 8, max_steps: int = 4) -> None:
    target = entry.get("execution_gate_target")
    if target not in {"execute", "noop"}:
        raise ValueError(f"{entry.get('id')}: invalid gate target {target!r}")
    if not entry.get("question") or not entry.get("answer"):
        raise ValueError(f"{entry.get('id')}: question and answer are required")
    if entry.get("format_type") != "qa":
        raise ValueError(f"{entry['id']}: all policy rows must use format_type=qa")
    cue_words = CONTROL_CUE_WORDS.intersection(WORD_RE.findall(entry["question"].lower()))
    if cue_words:
        raise ValueError(f"{entry['id']}: leaked control cue words {sorted(cue_words)}")

    if target == "noop":
        if "state_0" in entry or "execution" in entry or "state_1" in entry:
            raise ValueError(f"{entry['id']}: NOOP row carries execution data")
        return

    atoms = entry.get("state_0", {}).get("atoms", [])
    steps = entry.get("execution", [])
    if not atoms or not steps:
        raise ValueError(f"{entry['id']}: EXECUTE row requires atoms and steps")
    if len(atoms) + len(steps) > max_slots:
        raise ValueError(
            f"{entry['id']}: requires {len(atoms) + len(steps)} slots, max is {max_slots}")
    if len(steps) > max_steps:
        raise ValueError(f"{entry['id']}: has {len(steps)} steps, max is {max_steps}")

    slots = [float(value) for value in atoms]
    for step_index, step in enumerate(steps):
        if step.get("op") not in OPS:
            raise ValueError(f"{entry['id']}: invalid op at step {step_index}")
        arg_slots = step.get("arg_slots", [])
        args = step.get("args", [])
        if len(arg_slots) != 2 or len(args) != 2:
            raise ValueError(f"{entry['id']}: malformed args at step {step_index}")
        lhs_slot, rhs_slot = arg_slots
        if lhs_slot < 0 or lhs_slot >= len(slots) or rhs_slot < 0 or rhs_slot >= len(slots):
            raise ValueError(f"{entry['id']}: unavailable arg slot at step {step_index}")
        lhs, rhs = slots[lhs_slot], slots[rhs_slot]
        if args != [lhs, rhs]:
            raise ValueError(f"{entry['id']}: args disagree with slots at step {step_index}")
        expected = apply_op(step["op"], lhs, rhs)
        actual = float(step.get("result"))
        if abs(expected - actual) > 1e-9 * max(1.0, abs(expected)):
            raise ValueError(f"{entry['id']}: incorrect result at step {step_index}")
        slots.append(actual)
    answer = float(entry["answer"])
    if abs(answer - slots[-1]) > 1e-9 * max(1.0, abs(slots[-1])):
        raise ValueError(f"{entry['id']}: answer does not match terminal execution result")


def lexical_audit(entries: Iterable[dict[str, Any]]) -> list[tuple[str, float, int, int]]:
    counts: dict[str, collections.Counter[str]] = {
        "execute": collections.Counter(),
        "noop": collections.Counter(),
    }
    totals = collections.Counter()
    for entry in entries:
        target = entry["execution_gate_target"]
        words = set(WORD_RE.findall(entry["question"].lower()))
        counts[target].update(words)
        totals[target] += 1

    minimum_support = max(20, min(totals.values()) // 100)
    audit: list[tuple[str, float, int, int]] = []
    for word in counts["execute"].keys() | counts["noop"].keys():
        execute_count = counts["execute"][word]
        noop_count = counts["noop"][word]
        if execute_count + noop_count < minimum_support:
            continue
        execute_rate = execute_count / totals["execute"]
        noop_rate = noop_count / totals["noop"]
        audit.append((word, abs(execute_rate - noop_rate), execute_count, noop_count))
    audit.sort(key=lambda item: (-item[1], item[0]))
    return audit[:20]


def validate_pairs(pairs: list[PolicyPair]) -> dict[str, Any]:
    if not pairs:
        raise ValueError("at least one counterfactual pair is required")
    ids: set[str] = set()
    questions: set[str] = set()
    family_counts: dict[str, collections.Counter[str]] = collections.defaultdict(collections.Counter)
    lengths: dict[str, list[int]] = {"execute": [], "noop": []}
    numeric_counts: dict[str, collections.Counter[int]] = {
        "execute": collections.Counter(), "noop": collections.Counter()}
    first_token_counts: collections.Counter[str] = collections.Counter()
    layout_counts: collections.Counter[int] = collections.Counter()
    common_prefix_ratios: list[float] = []

    all_entries: list[dict[str, Any]] = []
    for pair in pairs:
        execute_words = WORD_RE.findall(pair.execute["question"].lower())
        noop_words = WORD_RE.findall(pair.noop["question"].lower())
        if not execute_words or not noop_words:
            raise ValueError(f"{pair.pair_id}: prompt has no lexical tokens")
        if execute_words[0] != noop_words[0]:
            raise ValueError(f"{pair.pair_id}: counterfactual first-token mismatch")
        first_token_counts[execute_words[0]] += 1
        layout_counts[pair.prompt_layout] += 1
        common_prefix = 0
        for execute_word, noop_word in zip(execute_words, noop_words):
            if execute_word != noop_word:
                break
            common_prefix += 1
        common_prefix_ratios.append(
            common_prefix / max(len(execute_words), len(noop_words)))

        for target, entry in (("execute", pair.execute), ("noop", pair.noop)):
            validate_entry(entry)
            if entry["execution_gate_target"] != target:
                raise ValueError(f"{pair.pair_id}: target mismatch")
            if entry["id"] in ids:
                raise ValueError(f"duplicate id: {entry['id']}")
            if entry["question"] in questions:
                raise ValueError(f"duplicate question: {entry['question']}")
            ids.add(entry["id"])
            questions.add(entry["question"])
            family_counts[pair.family][target] += 1
            lengths[target].append(len(entry["question"]))
            numeric_counts[target][len(NUMBER_RE.findall(entry["question"]))] += 1
            all_entries.append(entry)

        execute_numbers = NUMBER_RE.findall(pair.execute["question"])
        noop_numbers = NUMBER_RE.findall(pair.noop["question"])
        if execute_numbers != noop_numbers:
            raise ValueError(
                f"{pair.pair_id}: counterfactual rows expose different numeric contexts")

    for family, counts in family_counts.items():
        if counts["execute"] != counts["noop"]:
            raise ValueError(f"{family}: unbalanced counterfactual labels")
    if numeric_counts["execute"] != numeric_counts["noop"]:
        raise ValueError("numeric-count distributions differ between labels")

    mean_lengths = {
        target: sum(values) / len(values) for target, values in lengths.items()
    }
    relative_length_gap = abs(mean_lengths["execute"] - mean_lengths["noop"]) / max(mean_lengths.values())
    if relative_length_gap > 0.20:
        raise ValueError(f"question length gap is too large: {relative_length_gap:.3f}")

    lexical_gaps = lexical_audit(all_entries)
    if lexical_gaps and lexical_gaps[0][1] > MAX_LEXICAL_LABEL_GAP:
        word, gap, execute_count, noop_count = lexical_gaps[0]
        raise ValueError(
            f"lexical label shortcut is too strong: {word!r} gap={gap:.3f} "
            f"execute={execute_count} noop={noop_count}")

    mean_common_prefix_ratio = sum(common_prefix_ratios) / len(common_prefix_ratios)
    request_first_count = sum(
        count for layout_index, count in layout_counts.items()
        if PROMPT_LAYOUTS[layout_index][1])
    request_first_fraction = request_first_count / len(pairs)
    if len(pairs) >= 100:
        if len(first_token_counts) < len(PROMPT_LAYOUTS) - 2:
            raise ValueError(
                f"insufficient first-token diversity: {dict(first_token_counts)}")
        if max(first_token_counts.values()) / len(pairs) > 0.25:
            raise ValueError(
                f"first-token template concentration is too high: {dict(first_token_counts)}")
        if not 0.25 <= request_first_fraction <= 0.75:
            raise ValueError(
                f"request position is imbalanced: front fraction={request_first_fraction:.3f}")
        if mean_common_prefix_ratio > 0.65:
            raise ValueError(
                f"counterfactual common-prefix ratio is too high: "
                f"{mean_common_prefix_ratio:.3f}")

    return {
        "pairs": len(pairs),
        "entries": len(all_entries),
        "families": {family: dict(counts) for family, counts in sorted(family_counts.items())},
        "mean_question_length": mean_lengths,
        "numeric_count_distribution": {
            target: dict(sorted(counts.items())) for target, counts in numeric_counts.items()},
        "first_token_distribution": dict(sorted(first_token_counts.items())),
        "request_first_fraction": request_first_fraction,
        "mean_common_prefix_ratio": mean_common_prefix_ratio,
        "top_lexical_gaps": lexical_gaps,
    }


def generate_pairs(pair_count: int, seed: int) -> list[PolicyPair]:
    if pair_count <= 0:
        raise ValueError("pair_count must be positive")
    rng = random.Random(seed)
    pairs: list[PolicyPair] = []
    questions: set[str] = set()
    for index in range(pair_count):
        for _ in range(1000):
            pair = make_pair(index, rng)
            if (pair.execute["question"] not in questions
                    and pair.noop["question"] not in questions):
                pairs.append(pair)
                questions.add(pair.execute["question"])
                questions.add(pair.noop["question"])
                break
        else:
            raise RuntimeError(
                f"failed to generate a unique counterfactual pair at index {index}")
    # Ordering is deliberately not part of the split contract.  The trainer
    # owns randomized sampling; this shuffle only prevents family runs in the
    # persisted curriculum and remains reproducible for audits.
    rng.shuffle(pairs)
    return pairs


def flattened_entries(pairs: Iterable[PolicyPair], seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed ^ 0x5A17C0DE)
    entries: list[dict[str, Any]] = []
    for pair in pairs:
        variants = [pair.execute, pair.noop]
        rng.shuffle(variants)
        entries.extend(variants)
    return entries


def scan_existing_entries(path: Path) -> tuple[set[str], dict[str, dict[str, Any]]]:
    ids: set[str] = set()
    owned: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return ids, owned
    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            entry_id = entry.get("id")
            if not isinstance(entry_id, str) or not entry_id:
                raise ValueError(f"{path}:{line_number}: missing string id")
            if entry_id in ids:
                raise ValueError(f"{path}:{line_number}: duplicate id {entry_id}")
            ids.add(entry_id)
            if entry_id.startswith(ENTRY_PREFIX):
                owned[entry_id] = entry
    return ids, owned


def append_missing_entries(path: Path, entries: list[dict[str, Any]], dry_run: bool) -> int:
    existing_ids, existing_owned = scan_existing_entries(path)
    expected = {entry["id"]: entry for entry in entries}
    unexpected_owned = sorted(existing_owned.keys() - expected.keys())
    if unexpected_owned:
        raise ValueError(
            f"{path}: found {len(unexpected_owned)} unexpected {ENTRY_PREFIX} entries; "
            "use a new curriculum version instead of silently rewriting owned data")
    for entry_id, existing in existing_owned.items():
        if existing != expected[entry_id]:
            raise ValueError(
                f"{path}: existing generated entry {entry_id} differs from this generator; "
                "use a new curriculum version")

    missing = [entry for entry in entries if entry["id"] not in existing_ids]
    if dry_run or not missing:
        return len(missing)

    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", newline="\n", delete=False,
        dir=path.parent, prefix="execution_policy_v2_", suffix=".jsonl.tmp",
    ) as stage:
        stage_path = Path(stage.name)
        for entry in missing:
            stage.write(json.dumps(entry, ensure_ascii=False, separators=(",", ":")) + "\n")
        stage.flush()
        os.fsync(stage.fileno())

    try:
        with path.open("ab+") as destination:
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


def update_registry(
    path: Path, entries: list[dict[str, Any]], dry_run: bool, retire_v1: bool
) -> None:
    if path.exists():
        registry = json.loads(path.read_text(encoding="utf-8"))
    else:
        registry = {"curriculums": []}
    curricula = registry.setdefault("curriculums", [])
    if retire_v1:
        curricula[:] = [item for item in curricula if item.get("id") not in LEGACY_CURRICULUM_IDS]

    item = next((candidate for candidate in curricula if candidate.get("id") == CURRICULUM_ID), None)
    if item is None:
        item = {"id": CURRICULUM_ID, "timestamp": int(time.time())}
        curricula.append(item)
    item.update({
        "name": CURRICULUM_NAME,
        "format_as_concept": True,
        "concept_block_ids": [entry["id"] for entry in entries],
    })

    if dry_run:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as output:
        json.dump(registry, output, ensure_ascii=False, indent=2)
        output.write("\n")
        output.flush()
        os.fsync(output.fileno())
    os.replace(temporary, path)


def print_report(report: dict[str, Any]) -> None:
    print(
        f"Execution Policy v2: {report['pairs']} counterfactual pairs, "
        f"{report['entries']} rows, {len(report['families'])} families")
    print("  mean question length: "
          f"EXECUTE={report['mean_question_length']['execute']:.1f}, "
          f"NOOP={report['mean_question_length']['noop']:.1f}")
    print(f"  numeric-count distribution: {report['numeric_count_distribution']}")
    print(f"  first-token distribution: {report['first_token_distribution']}")
    print(
        "  request position: "
        f"{report['request_first_fraction']:.3f} front, "
        f"{1.0 - report['request_first_fraction']:.3f} trailing")
    print(
        "  mean counterfactual common-prefix ratio: "
        f"{report['mean_common_prefix_ratio']:.3f}")
    print("  top lexical label gaps (audit only):")
    for word, gap, execute_count, noop_count in report["top_lexical_gaps"][:10]:
        print(
            f"    {word:<16} gap={gap:.3f} "
            f"execute={execute_count} noop={noop_count}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir", type=Path,
        default=Path("resources/models/GRIM-text/training/data"))
    parser.add_argument("--pair-count", type=int, default=24_000)
    parser.add_argument("--seed", type=int, default=260722)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--keep-v1-curricula", action="store_true",
        help="Keep the three obsolete Execution Control v1 registry entries")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    pairs = generate_pairs(args.pair_count, args.seed)
    report = validate_pairs(pairs)
    entries = flattened_entries(pairs, args.seed)
    print_report(report)

    concept_path = args.data_dir / "concept_blocks.jsonl"
    registry_path = args.data_dir / "curriculum_registry.json"
    appended = append_missing_entries(concept_path, entries, args.dry_run)
    update_registry(
        registry_path, entries, args.dry_run,
        retire_v1=not args.keep_v1_curricula)
    print(f"  {'would append' if args.dry_run else 'appended'}: {appended}")
    print(f"  curriculum: {CURRICULUM_NAME} ({len(entries)} rows)")
    print(f"  concept blocks: {concept_path}")
    print(f"  registry: {registry_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
