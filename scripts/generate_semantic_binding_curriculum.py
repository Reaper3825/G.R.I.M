#!/usr/bin/env python3
"""Generate and optionally merge an atom-free semantic-binding curriculum.

Generated rows use top-level ``knowns`` and ``unknowns`` while ``goal`` remains
a separate, fully populated object. Derived targets use ``<UNRESOLVED>`` while
directly supplied bindings intentionally omit an unknown entry. Use
``--merge-data-dir`` to replace the generator-owned rows in an existing corpus
and update its existing curriculum registry.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = (
    ROOT_DIR / ".codex_tmp" / "semantic_binding_v1"
)
DEFAULT_COUNT = 60_000
DEFAULT_SEED = 4_271
CURRICULUM_ID = "curr_semantic_binding_v1"
CURRICULUM_NAME = "Semantic Binding v1"
ENTRY_PREFIX = "sbv1_"
UNRESOLVED = "<UNRESOLVED>"

ASSIGNMENT_RE = re.compile(r"^[a-z][a-z0-9_]* = .+$")
ATOM_MARKERS = (
    "<INT>", "</INT>", "<FLOAT>", "</FLOAT>",
    "<STRING>", "</STRING>", "<BOOL>", "</BOOL>",
    "<ENTITY>", "</ENTITY>",
)


NAMES = (
    "ava", "liam", "maya", "noah", "iris", "theo", "nora", "elias",
    "zoe", "milo", "lena", "owen", "ruby", "felix", "clara", "jonah",
    "mina", "lucas", "esme", "caleb", "talia", "silas", "freya", "roman",
    "anika", "ezra", "dahlia", "leon", "sasha", "marcus", "ines", "kai",
    "selene", "hugo", "nadia", "orion", "amira", "jude", "lyra", "devon",
    "celeste", "asher", "mara", "dorian", "vivian", "rowan", "elena", "samir",
    "bianca", "julian",
)

RELATIONS = (
    "friend", "brother", "sister", "parent", "cousin",
    "mentor", "manager", "neighbor", "teammate", "coworker",
)

INVERSE_RELATIONS = (
    ("parent", "child"),
    ("mentor", "mentee"),
    ("manager", "direct_report"),
    ("teacher", "student"),
    ("landlord", "tenant"),
)

CONTAINERS = (
    "box", "crate", "drawer", "backpack", "cabinet", "basket", "locker",
    "toolbox", "suitcase", "envelope", "jar", "bin", "chest", "pouch",
    "folder", "cupboard", "carton", "satchel", "case", "trunk",
)

OBJECTS = (
    "rubber ball", "brass key", "blue notebook", "silver compass",
    "wooden puzzle", "red scarf", "glass marble", "paper map",
    "small hammer", "green cable", "ceramic mug", "cotton glove",
    "travel adapter", "memory card", "paint brush", "measuring tape",
    "flashlight", "music box", "photo album", "garden trowel",
    "wool blanket", "metal whistle", "pocket mirror", "recipe card",
    "toy train", "orange ribbon", "water bottle", "safety goggles",
    "charging cable", "folding ruler", "ink stamp", "door handle",
    "seed packet", "coffee filter", "bike light", "canvas strap",
    "spare button", "model airplane", "chess piece", "rubber eraser",
)

LOCATIONS = (
    "north workshop", "south archive", "main office", "west studio",
    "east laboratory", "upper pantry", "lower storeroom", "garden shed",
    "front hallway", "rear classroom",
)

GPU_MODELS = (
    "Aster X1", "Boreal V2", "Cinder Pro", "Delta Arc", "Ember Max",
    "Fjord S4", "Garnet One", "Helix Prime", "Ion Nova", "Jade Ultra",
    "Kestrel 8", "Lumen Core", "Meteor Z", "Nimbus XT", "Onyx Plus",
    "Pulse 7", "Quartz Neo", "Raptor Q", "Solace 12", "Titan Edge",
    "Umbra GX", "Vector Mini", "Willow AI", "Xenon Forge", "Yarrow RTX",
    "Zenith Duo", "Aurora Compute", "Beacon ML", "Cascade Render",
    "Drift Matrix", "Equinox Tensor", "Flux Studio",
)

DESTINATIONS = (
    "Harbor Station", "Maple Library", "Cedar Clinic", "North Terminal",
    "River Museum", "Orchard Market", "Granite Hall", "Sunset Theater",
    "Pine Research Center", "Lakeview Hotel", "Central Depot", "Elm Campus",
    "Beacon Tower", "Willow Park", "Atlas Workshop", "Juniper Gallery",
    "Silver Arena", "Meadow School", "Summit Lodge", "Oak Conference Center",
)

ADJECTIVES = (
    "amber", "brisk", "calm", "delta", "eager", "frost", "gold", "hazel",
    "indigo", "jade", "kind", "lunar", "maple", "north", "opal", "prime",
    "quiet", "rapid", "solar", "tidal", "urban", "vivid", "west", "young",
    "zen", "bright", "clear", "deep", "early", "fresh", "grand", "swift",
)

NOUNS = (
    "anchor", "bridge", "canyon", "dawn", "ember", "forest", "garden",
    "harbor", "island", "junction", "kernel", "lantern", "meadow", "network",
    "orbit", "portal", "quartz", "river", "summit", "trail", "uplink",
    "valley", "willow", "yard", "zenith", "archive", "beacon", "circuit",
    "domain", "engine", "foundry", "gateway",
)


@dataclass(frozen=True)
class BindingExample:
    family: str
    prompt: str
    knowns: list[str]
    unknowns: list[str]
    answer: str
    target: str
    evidence_basis: str


def rotate(value: int, size: int, seed: int, salt: int) -> int:
    return (value * 104_729 + seed * 97 + salt * 7_919) % size


def pick(values: tuple[str, ...], value: int, seed: int, salt: int) -> str:
    return values[rotate(value, len(values), seed, salt)]


def derived_example(
    family: str,
    prompt: str,
    knowns: list[str],
    target: str,
    value: str,
    evidence_basis: str,
) -> BindingExample:
    return BindingExample(
        family=family,
        prompt=prompt,
        knowns=knowns,
        unknowns=[f"{target} = {UNRESOLVED}"],
        answer=f"{target} = {value}",
        target=target,
        evidence_basis=evidence_basis,
    )


def direct_example(
    family: str,
    prompt: str,
    binding: str,
    target: str,
) -> BindingExample:
    return BindingExample(
        family=family,
        prompt=prompt,
        knowns=[binding],
        unknowns=[],
        answer=binding,
        target=target,
        evidence_basis="the requested binding is explicitly present in the known facts",
    )


def volume_used(occurrence: int, seed: int) -> BindingExample:
    capacity = 120 + occurrence
    remaining = 84 if occurrence == 0 else 10 + ((occurrence * 53 + seed) % (capacity - 19))
    used = capacity - remaining
    vessel = ("tank", "reservoir", "cistern", "container")[occurrence % 4]
    prompt = (
        f"A {vessel} has a capacity of {capacity} liters and now has "
        f"{remaining} liters remaining. Bind the number of liters used."
    )
    return derived_example(
        "volume_used", prompt,
        [f"liters_capacity = {capacity}", f"liters_remaining = {remaining}"],
        "liters_used", str(used),
        "subtracting liters_remaining from liters_capacity resolves the requested unit binding",
    )


def distance_remaining(occurrence: int, seed: int) -> BindingExample:
    total = 100 + occurrence
    traveled = 45 if occurrence == 0 else 5 + ((occurrence * 67 + seed) % (total - 9))
    remaining = total - traveled
    route = ("delivery route", "rail journey", "cycling course", "survey path")[occurrence % 4]
    prompt = (
        f"The {route} is {total} kilometers long. After {traveled} kilometers "
        "have been traveled, bind the remaining distance."
    )
    return derived_example(
        "distance_remaining", prompt,
        [f"distance_total = {total}", f"distance_traveled = {traveled}"],
        "distance_remaining", str(remaining),
        "subtracting distance_traveled from distance_total resolves the requested unit binding",
    )


def final_temperature(occurrence: int, seed: int) -> BindingExample:
    current = 72 if occurrence == 0 else 45 + ((occurrence * 17 + seed) % 46)
    magnitude = 2 + ((occurrence * 29 + seed) % 19)
    warming = ((occurrence + seed) % 2) == 0
    change = magnitude if warming else -magnitude
    final = current + change
    place = pick(LOCATIONS, occurrence, seed, 3)
    sensor = f"sensor_{pick(ADJECTIVES, occurrence, seed, 5)}_{occurrence:04d}"
    direction = "rises" if warming else "falls"
    prompt = (
        f"At the {place}, {sensor} reports a current temperature of {current} degrees. It "
        f"{direction} by {magnitude} degrees. Bind the final temperature."
    )
    return derived_example(
        "final_temperature", prompt,
        [f"temperature_sensor = {sensor}", f"temperature_current = {current}", f"temperature_change = {change}"],
        "final_temperature", str(final),
        "adding temperature_change to temperature_current resolves the requested unit binding",
    )


def storage_free(occurrence: int, seed: int) -> BindingExample:
    capacity = 512 + 2 * occurrence
    used = 16 + 16 * ((occurrence * 23 + seed) % ((capacity // 16) - 1))
    free = capacity - used
    device = ("workstation", "server", "archive drive", "render node")[occurrence % 4]
    prompt = (
        f"A {device} has {capacity} gigabytes of storage and {used} gigabytes "
        "are occupied. Bind the free storage capacity."
    )
    return derived_example(
        "storage_free", prompt,
        [f"storage_capacity_gb = {capacity}", f"storage_used_gb = {used}"],
        "storage_free_gb", str(free),
        "subtracting storage_used_gb from storage_capacity_gb resolves the requested unit binding",
    )


def inventory_remaining(occurrence: int, seed: int) -> BindingExample:
    stocked = 100 + occurrence
    sold = 1 + ((occurrence * 71 + seed) % (stocked - 1))
    remaining = stocked - sold
    item = ("filters", "bearings", "cables", "labels", "fasteners")[occurrence % 5]
    prompt = (
        f"The warehouse stocked {stocked} {item} and distributed {sold}. "
        "Bind the number of inventory units remaining."
    )
    return derived_example(
        "inventory_remaining", prompt,
        [f"inventory_stocked = {stocked}", f"inventory_distributed = {sold}"],
        "inventory_remaining", str(remaining),
        "subtracting inventory_distributed from inventory_stocked resolves the requested binding",
    )


def package_total(occurrence: int, seed: int) -> BindingExample:
    packages = 2 + occurrence
    per_package = 3 + ((occurrence * 31 + seed) % 48)
    total = packages * per_package
    item = ("washers", "cards", "clips", "tiles", "sensors")[occurrence % 5]
    prompt = (
        f"There are {packages} packages with {per_package} {item} in each one. "
        "Bind the total item count."
    )
    return derived_example(
        "package_total", prompt,
        [f"package_count = {packages}", f"items_per_package = {per_package}"],
        "items_total", str(total),
        "multiplying package_count by items_per_package resolves the requested binding",
    )


def direct_relationship(occurrence: int, seed: int) -> BindingExample:
    relation = RELATIONS[occurrence % len(RELATIONS)]
    subject = NAMES[(occurrence // len(RELATIONS)) % len(NAMES)]
    subject_index = NAMES.index(subject)
    other_index = (
        occurrence // (len(RELATIONS) * len(NAMES)) + 13 + seed
    ) % (len(NAMES) - 1)
    if other_index >= subject_index:
        other_index += 1
    other = NAMES[other_index]
    target = f"{subject}_{relation}"
    binding = f"{target} = {other}"
    prompt = f"The records state that {other.title()} is {subject.title()}'s {relation}. Who is {subject.title()}'s {relation}?"
    return direct_example("direct_relationship", prompt, binding, target)


def inverse_relationship(occurrence: int, seed: int) -> BindingExample:
    forward, inverse = INVERSE_RELATIONS[occurrence % len(INVERSE_RELATIONS)]
    subject = NAMES[(occurrence // len(INVERSE_RELATIONS)) % len(NAMES)]
    subject_index = NAMES.index(subject)
    other_index = (
        occurrence // (len(INVERSE_RELATIONS) * len(NAMES)) + 19 + seed
    ) % (len(NAMES) - 1)
    if other_index >= subject_index:
        other_index += 1
    other = NAMES[other_index]
    known = f"{subject}_{forward} = {other}"
    target = f"{other}_{inverse}"
    prompt = (
        f"{other.title()} is listed as {subject.title()}'s {forward}. "
        f"Bind {other.title()}'s corresponding {inverse}."
    )
    return derived_example(
        "inverse_relationship", prompt, [known], target, subject,
        f"inverting the supplied {forward} relationship yields the corresponding {inverse} binding",
    )


def direct_container(occurrence: int, seed: int) -> BindingExample:
    container = CONTAINERS[occurrence % len(CONTAINERS)]
    obj = OBJECTS[(occurrence // len(CONTAINERS)) % len(OBJECTS)]
    location = LOCATIONS[(occurrence // (len(CONTAINERS) * len(OBJECTS))) % len(LOCATIONS)]
    target = f"{container}_hold"
    binding = f"{target} = {obj}"
    prompt = f"In the {location}, the {container} holds a {obj}. What does the {container} hold?"
    return direct_example("direct_container", prompt, binding, target)


def nested_container(occurrence: int, seed: int) -> BindingExample:
    outer = CONTAINERS[occurrence % len(CONTAINERS)]
    outer_index = CONTAINERS.index(outer)
    inner_index = (occurrence // len(CONTAINERS)) % (len(CONTAINERS) - 1)
    if inner_index >= outer_index:
        inner_index += 1
    inner = CONTAINERS[inner_index]
    obj = OBJECTS[(occurrence // (len(CONTAINERS) * (len(CONTAINERS) - 1))) % len(OBJECTS)]
    location = pick(LOCATIONS, occurrence, seed, 11)
    target = f"{outer}_indirect_hold"
    prompt = (
        f"At the {location}, the {outer} holds the {inner}, and the {inner} "
        f"holds a {obj}. Bind what the {outer} indirectly holds."
    )
    return derived_example(
        "nested_container", prompt,
        [f"{outer}_hold = {inner}", f"{inner}_hold = {obj}"],
        target, obj,
        "following the two supplied containment bindings resolves the indirectly held object",
    )


def direct_attribute(occurrence: int, seed: int) -> BindingExample:
    subtype = occurrence % 5
    local = occurrence // 5
    if subtype == 0:
        adjective = ADJECTIVES[local % len(ADJECTIVES)]
        noun = NOUNS[(local // len(ADJECTIVES)) % len(NOUNS)]
        region = LOCATIONS[(local // (len(ADJECTIVES) * len(NOUNS))) % len(LOCATIONS)]
        region_key = region.replace(" ", "_")
        target = "account_enabled" if local == 0 else f"account_{adjective}_{noun}_{region_key}_enabled"
        value = "true" if local == 0 or ((local + seed) % 2 == 0) else "false"
        prompt = (
            f"The {adjective} {noun} account in the {region} has enabled set "
            f"to {value}. Bind its enabled state."
        )
        return direct_example("direct_attribute", prompt, f"{target} = {value}", target)
    if subtype == 1:
        folder = ("models", "checkpoints", "adapters", "artifacts")[local % 4]
        filename = f"{pick(ADJECTIVES, local, seed, 13)}_{local:04d}.bin"
        value = '"C:/GRIM/model.bin"' if local == 0 else f'"C:/GRIM/{folder}/{filename}"'
        target = "required_file"
        prompt = f"The runtime manifest identifies {value} as the required file. Bind the required file path."
        return derived_example(
            "direct_attribute", prompt, [f"file_path = {value}"], target, value,
            "the manifest states that file_path is the required file",
        )
    if subtype == 2:
        user = NAMES[local % len(NAMES)]
        destination = DESTINATIONS[(local // len(NAMES)) % len(DESTINATIONS)]
        origin = LOCATIONS[(local // (len(NAMES) * len(DESTINATIONS))) % len(LOCATIONS)]
        target = "user_destination" if local == 0 else f"{user}_destination"
        prompt = (
            f"Before leaving the {origin}, {user.title()} selected {destination} "
            f"as the destination. Bind {user.title()}'s destination."
        )
        return direct_example("direct_attribute", prompt, f"{target} = {destination}", target)
    if subtype == 3:
        adjective = ADJECTIVES[local % len(ADJECTIVES)]
        noun = NOUNS[(local // len(ADJECTIVES)) % len(NOUNS)]
        region = LOCATIONS[(local // (len(ADJECTIVES) * len(NOUNS))) % len(LOCATIONS)]
        region_key = region.replace(" ", "_")
        target = "task_complete" if local == 0 else f"task_{adjective}_{noun}_{region_key}_complete"
        value = "true" if local == 0 or ((local + seed) % 3 != 0) else "false"
        prompt = (
            f"The {adjective} {noun} task in the {region} records completion as "
            f"{value}. Bind its completion state."
        )
        return direct_example("direct_attribute", prompt, f"{target} = {value}", target)
    gpu = GPU_MODELS[local % len(GPU_MODELS)]
    location = LOCATIONS[(local // len(GPU_MODELS)) % len(LOCATIONS)]
    vram = 32 if local == 0 else 8 * (1 + ((local * 7 + seed) % 12))
    asset = f"{gpu.lower().replace(' ', '_')}_{local:04d}"
    target = "gpu_vram" if local == 0 else f"{asset}_vram"
    prompt = (
        f"The {gpu} asset {asset} installed in the {location} has {vram} "
        "gigabytes of VRAM. Bind its VRAM capacity."
    )
    return direct_example("direct_attribute", prompt, f"{target} = {vram}", target)


def fastest_gpu(occurrence: int, seed: int) -> BindingExample:
    first = GPU_MODELS[occurrence % len(GPU_MODELS)]
    second = GPU_MODELS[(occurrence // len(GPU_MODELS) + 9) % len(GPU_MODELS)]
    third = GPU_MODELS[(occurrence // (len(GPU_MODELS) * len(GPU_MODELS)) + 21) % len(GPU_MODELS)]
    candidates = []
    for candidate in (first, second, third):
        if candidate not in candidates:
            candidates.append(candidate)
    while len(candidates) < 3:
        candidates.append(GPU_MODELS[(GPU_MODELS.index(candidates[-1]) + 1) % len(GPU_MODELS)])
    base = 70 + ((occurrence * 43 + seed) % 930)
    scores = [base + 3, base + 11 + (occurrence % 7), base + 7 + (occurrence % 3)]
    winner_index = max(range(3), key=lambda index: scores[index])
    knowns = [
        f"{candidates[index].lower().replace(' ', '_')}_benchmark_score = {scores[index]}"
        for index in range(3)
    ]
    prompt = (
        f"Benchmark scores are {candidates[0]} at {scores[0]}, {candidates[1]} at "
        f"{scores[1]}, and {candidates[2]} at {scores[2]}. Bind the fastest GPU."
    )
    return derived_example(
        "fastest_gpu", prompt, knowns, "fastest_gpu", candidates[winner_index],
        "selecting the GPU with the greatest supplied benchmark score resolves the comparison binding",
    )


BUILDERS: tuple[Callable[[int, int], BindingExample], ...] = (
    volume_used,
    distance_remaining,
    final_temperature,
    storage_free,
    inventory_remaining,
    package_total,
    direct_relationship,
    inverse_relationship,
    direct_container,
    nested_container,
    direct_attribute,
    fastest_gpu,
)


GOAL_PROFILES: dict[str, tuple[str, str, str]] = {
    "volume_used": (
        "The consumed volume has been derived from the stated capacity and remainder.",
        "The response contains one consumed-volume assignment.",
        "Capacity and remaining volume establish a complete subtraction relationship.",
    ),
    "distance_remaining": (
        "The unfinished portion of the journey has been derived from the stated measurements.",
        "The response contains one remaining-distance assignment.",
        "Total distance and completed distance establish a complete subtraction relationship.",
    ),
    "final_temperature": (
        "The post-change measurement has been determined from the initial reading and stated change.",
        "The response contains one resulting-temperature assignment.",
        "The initial reading and signed change establish the resulting measurement.",
    ),
    "storage_free": (
        "The unoccupied storage capacity has been derived from the supplied device measurements.",
        "The response contains one free-capacity assignment.",
        "Total capacity and occupied capacity establish a complete subtraction relationship.",
    ),
    "inventory_remaining": (
        "The inventory left after distribution has been determined.",
        "The response contains one remaining-inventory assignment.",
        "The stocked and distributed counts establish the remaining count.",
    ),
    "package_total": (
        "The aggregate item count across the packages has been determined.",
        "The response contains one aggregate-count assignment.",
        "Package count and per-package count establish a multiplication relationship.",
    ),
    "direct_relationship": (
        "The person occupying the relationship requested by the prompt has been identified.",
        "The response contains one person-to-relationship assignment.",
        "The requested relationship is explicitly stated among the supplied facts.",
    ),
    "inverse_relationship": (
        "The inverse role implied by the stated interpersonal relationship has been established.",
        "The response contains one inverse-role assignment.",
        "The directed relationship in the prompt determines its corresponding inverse role.",
    ),
    "direct_container": (
        "The object directly held by the requested container has been identified.",
        "The response contains one direct-containment assignment.",
        "The requested containment fact is explicitly supplied by the example.",
    ),
    "nested_container": (
        "The object reached through the stated containment chain has been identified.",
        "The response contains one indirect-containment assignment.",
        "The two directed containment facts form a complete two-hop chain.",
    ),
    "direct_attribute": (
        "The property or resource requested by the prompt has been resolved from the supplied record.",
        "The response contains one property or resource assignment.",
        "The supplied record provides the fact or alias relationship needed for resolution.",
    ),
    "fastest_gpu": (
        "The highest-ranked candidate under the supplied comparison has been selected.",
        "The response contains one candidate-selection assignment.",
        "The supplied scores provide a complete ordering criterion for the candidates.",
    ),
}


def make_goal(example: BindingExample) -> dict[str, Any]:
    target_state, criterion, evidence = GOAL_PROFILES[example.family]
    return {
        "target_state": target_state,
        "success_criteria": [
            {
                "criterion": criterion,
                "evidence": evidence,
            }
        ],
        "constraints": [
            "Use only facts supplied by the example.",
            "Preserve the direction and scope of relationships described in the prompt.",
            "Express the resolved result as one concise assignment without additional claims.",
        ],
    }


def make_entry(index: int, seed: int) -> dict[str, Any]:
    family_index = index % len(BUILDERS)
    occurrence = index // len(BUILDERS)
    example = BUILDERS[family_index](occurrence, seed)
    block_id = f"{ENTRY_PREFIX}{index:06d}"
    return {
        "id": block_id,
        "name": f"Semantic binding: {example.family.replace('_', ' ')}",
        "prompt": example.prompt,
        "knowns": example.knowns,
        "unknowns": example.unknowns,
        "answer": example.answer,
        "goal": make_goal(example),
        "format_type": "qa",
        "source_sequence_id": f"synthetic_semantic_binding_v1:{example.family}",
        "timestamp": 0,
    }


def validate_entry(entry: dict[str, Any], seen_ids: set[str], seen_prompts: set[str]) -> None:
    expected_fields = {
        "id", "name", "prompt", "knowns", "unknowns", "answer", "goal",
        "format_type", "source_sequence_id", "timestamp",
    }
    if set(entry) != expected_fields:
        raise ValueError(f"{entry.get('id', '<missing>')}: unexpected fields")
    block_id = entry["id"]
    if not isinstance(block_id, str) or not block_id.startswith(ENTRY_PREFIX):
        raise ValueError(f"invalid id: {block_id!r}")
    if block_id in seen_ids:
        raise ValueError(f"duplicate id: {block_id}")
    prompt = entry["prompt"]
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError(f"{block_id}: prompt is empty")
    if prompt in seen_prompts:
        raise ValueError(f"{block_id}: duplicate prompt")

    knowns = entry["knowns"]
    unknowns = entry["unknowns"]
    if not isinstance(knowns, list) or not knowns:
        raise ValueError(f"{block_id}: knowns must be a non-empty list")
    if not isinstance(unknowns, list):
        raise ValueError(f"{block_id}: unknowns must be a list")
    for field, bindings in (("knowns", knowns), ("unknowns", unknowns)):
        for binding in bindings:
            if not isinstance(binding, str) or not ASSIGNMENT_RE.fullmatch(binding):
                raise ValueError(f"{block_id}: invalid {field} binding {binding!r}")
    for binding in unknowns:
        if not binding.endswith(f" = {UNRESOLVED}"):
            raise ValueError(f"{block_id}: unknown binding must use {UNRESOLVED}")

    answer = entry["answer"]
    if not isinstance(answer, str) or not ASSIGNMENT_RE.fullmatch(answer):
        raise ValueError(f"{block_id}: answer is not one semantic binding")
    answer_target = answer.split(" = ", 1)[0]
    unresolved_targets = {binding.split(" = ", 1)[0] for binding in unknowns}
    if unknowns:
        if unresolved_targets != {answer_target}:
            raise ValueError(f"{block_id}: answer does not resolve the sole unknown")
    elif answer not in knowns:
        raise ValueError(f"{block_id}: direct answer must be present in knowns")

    goal = entry["goal"]
    if set(goal) != {"target_state", "success_criteria", "constraints"}:
        raise ValueError(f"{block_id}: goal object is not fully populated")
    if not isinstance(goal["target_state"], str) or not goal["target_state"].strip():
        raise ValueError(f"{block_id}: goal target_state is empty")
    if not isinstance(goal["success_criteria"], list) or not goal["success_criteria"]:
        raise ValueError(f"{block_id}: goal success_criteria is empty")
    for criterion in goal["success_criteria"]:
        if set(criterion) != {"criterion", "evidence"}:
            raise ValueError(f"{block_id}: malformed success criterion")
        if not all(isinstance(criterion[key], str) and criterion[key].strip()
                   for key in ("criterion", "evidence")):
            raise ValueError(f"{block_id}: empty success criterion field")
    if not isinstance(goal["constraints"], list) or not goal["constraints"]:
        raise ValueError(f"{block_id}: goal constraints are empty")
    goal_text = json.dumps(goal, ensure_ascii=False, separators=(",", ":"))
    if answer_target in goal_text or answer in goal_text or UNRESOLVED in goal_text:
        raise ValueError(f"{block_id}: goal leaks target or answer binding data")
    for binding in knowns + unknowns:
        if binding in goal_text:
            raise ValueError(f"{block_id}: goal leaks a supplied semantic binding")

    encoded = json.dumps(entry, ensure_ascii=False, separators=(",", ":"))
    for marker in ATOM_MARKERS:
        if marker in encoded:
            raise ValueError(f"{block_id}: authored atom marker {marker} is forbidden")
    for forbidden in ("execution", "state_0", "state_1", "raw"):
        if forbidden in entry:
            raise ValueError(f"{block_id}: atom/execution field {forbidden} is forbidden")

    seen_ids.add(block_id)
    seen_prompts.add(prompt)


def atomic_text_path(destination: Path) -> tuple[Path, Any]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, stage_name = tempfile.mkstemp(
        prefix=destination.name + ".", suffix=".tmp", dir=destination.parent)
    return Path(stage_name), os.fdopen(descriptor, "w", encoding="utf-8", newline="\n")


def generate_dataset(output_dir: Path, count: int, seed: int) -> dict[str, Any]:
    if count <= 0:
        raise ValueError("count must be positive")
    jsonl_path = output_dir / "concept_blocks.jsonl"
    registry_path = output_dir / "curriculum_registry.json"
    manifest_path = output_dir / "semantic_binding_manifest.json"
    seen_ids: set[str] = set()
    seen_prompts: set[str] = set()
    family_counts: collections.Counter[str] = collections.Counter()
    unknown_rows = 0
    known_only_rows = 0
    ids: list[str] = []
    digest = hashlib.sha256()

    stage_path, output = atomic_text_path(jsonl_path)
    try:
        with output:
            for index in range(count):
                entry = make_entry(index, seed)
                validate_entry(entry, seen_ids, seen_prompts)
                serialized = json.dumps(entry, ensure_ascii=False, separators=(",", ":"))
                line = (serialized + "\n").encode("utf-8")
                output.write(line.decode("utf-8"))
                digest.update(line)
                ids.append(entry["id"])
                family = entry["source_sequence_id"].rsplit(":", 1)[-1]
                family_counts[family] += 1
                if entry["unknowns"]:
                    unknown_rows += 1
                else:
                    known_only_rows += 1
            output.flush()
            os.fsync(output.fileno())
        os.replace(stage_path, jsonl_path)
    finally:
        stage_path.unlink(missing_ok=True)

    registry = {
        "curriculums": [
            {
                "id": CURRICULUM_ID,
                "name": CURRICULUM_NAME,
                "concept_block_ids": ids,
                "plaintext_block_ids": [],
                "training_stage": "sft",
                "format_as_concept": True,
                "timestamp": 0,
            }
        ]
    }
    registry_stage, registry_output = atomic_text_path(registry_path)
    try:
        with registry_output:
            json.dump(registry, registry_output, ensure_ascii=False, indent=2)
            registry_output.write("\n")
            registry_output.flush()
            os.fsync(registry_output.fileno())
        os.replace(registry_stage, registry_path)
    finally:
        registry_stage.unlink(missing_ok=True)

    manifest = {
        "dataset_id": "semantic_binding_v1",
        "curriculum_id": CURRICULUM_ID,
        "generator": "scripts/generate_semantic_binding_curriculum.py",
        "seed": seed,
        "entry_count": count,
        "family_counts": dict(sorted(family_counts.items())),
        "rows_with_unknowns": unknown_rows,
        "known_only_rows": known_only_rows,
        "unresolved_marker": UNRESOLVED,
        "authored_atom_markers": 0,
        "goal_policy": "target_state, success_criteria with evidence, and constraints are all populated",
        "sha256_concept_blocks_jsonl": digest.hexdigest(),
    }
    manifest_stage, manifest_output = atomic_text_path(manifest_path)
    try:
        with manifest_output:
            json.dump(manifest, manifest_output, ensure_ascii=False, indent=2)
            manifest_output.write("\n")
            manifest_output.flush()
            os.fsync(manifest_output.fileno())
        os.replace(manifest_stage, manifest_path)
    finally:
        manifest_stage.unlink(missing_ok=True)
    return manifest


def merge_into_existing_dataset(source_dir: Path, data_dir: Path) -> tuple[int, int]:
    """Replace this generator's owned rows in an existing native data directory."""
    source_jsonl = source_dir / "concept_blocks.jsonl"
    source_registry = source_dir / "curriculum_registry.json"
    target_jsonl = data_dir / "concept_blocks.jsonl"
    target_registry = data_dir / "curriculum_registry.json"
    for path in (source_jsonl, source_registry, target_jsonl, target_registry):
        if not path.is_file():
            raise FileNotFoundError(f"required merge input is missing: {path}")

    source_curricula = json.loads(source_registry.read_text(encoding="utf-8"))["curriculums"]
    if len(source_curricula) != 1 or source_curricula[0].get("id") != CURRICULUM_ID:
        raise ValueError("generated curriculum registry has an unexpected shape")
    source_curriculum = source_curricula[0]

    registry = json.loads(target_registry.read_text(encoding="utf-8"))
    curricula = registry.get("curriculums")
    if not isinstance(curricula, list):
        raise ValueError("target curriculum registry must contain a curriculums array")
    matching_indices = [
        index for index, item in enumerate(curricula)
        if item.get("id") == CURRICULUM_ID
    ]
    if len(matching_indices) > 1:
        raise ValueError(f"target registry has duplicate curriculum id {CURRICULUM_ID}")
    if matching_indices:
        curricula[matching_indices[0]] = source_curriculum
    else:
        curricula.append(source_curriculum)

    registry_stage, registry_output = atomic_text_path(target_registry)
    jsonl_stage, jsonl_output = atomic_text_path(target_jsonl)
    kept = 0
    removed = 0
    owned_marker = ENTRY_PREFIX.encode("ascii")
    try:
        with registry_output:
            json.dump(registry, registry_output, ensure_ascii=False, indent=2)
            registry_output.write("\n")
            registry_output.flush()
            os.fsync(registry_output.fileno())

        # Reopen the text stage in binary mode so existing JSONL bytes are
        # preserved exactly and the generated rows can be appended verbatim.
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
                        raise ValueError(
                            f"{target_jsonl}:{line_number}: {ENTRY_PREFIX} appears outside an owned id")
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
        os.replace(registry_stage, target_registry)
    finally:
        if not jsonl_output.closed:
            jsonl_output.close()
        jsonl_stage.unlink(missing_ok=True)
        registry_stage.unlink(missing_ok=True)
    return removed, kept


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--count", type=int, default=DEFAULT_COUNT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--merge-data-dir",
        type=Path,
        help="replace owned sbv1 rows and register the curriculum in this existing data directory",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = generate_dataset(args.output_dir.resolve(), args.count, args.seed)
    print(f"{CURRICULUM_NAME}: {manifest['entry_count']:,} validated ConceptBlocks")
    print(f"  families: {manifest['family_counts']}")
    print(f"  rows with unknowns: {manifest['rows_with_unknowns']:,}")
    print(f"  known-only rows: {manifest['known_only_rows']:,}")
    print(f"  output: {args.output_dir.resolve()}")
    if args.merge_data_dir is not None:
        removed, kept = merge_into_existing_dataset(
            args.output_dir.resolve(), args.merge_data_dir.resolve())
        print(
            f"  merged into: {args.merge_data_dir.resolve()} "
            f"(replaced {removed:,} prior owned rows; preserved {kept:,} unrelated rows)"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
