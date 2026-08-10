#!/usr/bin/env python3
"""Generate generic single-step success-criterion/evidence SFT examples."""

from __future__ import annotations

import argparse
import collections
from pathlib import Path
from typing import Any

import generate_intent_action_curriculum as source
import generate_intent_action_target_state_curriculum as target_source
from intent_action_curriculum_content import BUILDERS, generate_content
from success_criterion_curriculum_content import CONTRACTS, criterion_for, evidence_for
from vocab_playground import build_trie, load_vocab_bin, tokenize

CURRICULUM_ID = "curr_generic_success_criteria_v1"
CURRICULUM_NAME = "Generic Success Criteria v1"
ENTRY_PREFIX = "gscv1_"
ROW_KINDS = ("criterion", "evidence", "pair")


def make_entries(index: int, seed: int = source.DEFAULT_SEED) -> list[dict[str, Any]]:
    target_entry = target_source.make_entry(index, seed)
    content = generate_content(index, seed)
    occurrence = index // len(BUILDERS)
    criterion = criterion_for(content.family, occurrence)
    evidence = evidence_for(content.family, occurrence)
    prompt = target_entry["prompt"]
    target_state = target_entry["goal"]["target_state"]
    base_goal = {"target_state": target_state}

    return [
        {
            "id": f"{ENTRY_PREFIX}{index:05d}_criterion",
            "prompt": prompt,
            "answer": criterion,
            "goal": base_goal,
        },
        {
            "id": f"{ENTRY_PREFIX}{index:05d}_evidence",
            "prompt": prompt,
            "answer": evidence,
            "goal": {
                "target_state": target_state,
                "success_criteria": [{"criterion": criterion}],
            },
        },
        {
            "id": f"{ENTRY_PREFIX}{index:05d}_pair",
            "prompt": prompt,
            "answer": criterion + "\n" + evidence,
            "goal": base_goal,
        },
    ]


def generate_entries(
    count: int,
    seed: int = source.DEFAULT_SEED,
) -> list[dict[str, Any]]:
    if count <= 0:
        raise ValueError("count must be positive")
    return [entry for index in range(count) for entry in make_entries(index, seed)]


def render_for_validation(entry: dict[str, Any]) -> str:
    output = entry["prompt"] + "\n\n"
    goal = entry["goal"]
    output += goal["target_state"] + "\n\n"
    criteria = goal.get("success_criteria", [])
    for pair_index, pair in enumerate(criteria):
        output += pair["criterion"]
        evidence = pair.get("evidence")
        if evidence:
            output += "\n" + evidence
        if pair_index + 1 < len(criteria):
            output += "\n\n"
    if criteria:
        output += "\n\n"
    return output + entry["answer"] + "\n"


def validate_entries(entries: list[dict[str, Any]], vocab_path: Path) -> dict[str, Any]:
    vocab = load_vocab_bin(vocab_path)
    trie = build_trie(vocab)
    ids: set[str] = set()
    row_counts: collections.Counter[str] = collections.Counter()
    criterion_lengths: set[int] = set()
    maximum_tokens = 0
    maximum_id = ""
    expected_fields = {"id", "prompt", "answer", "goal"}

    for entry in entries:
        block_id = entry.get("id")
        if not isinstance(block_id, str) or not block_id.startswith(ENTRY_PREFIX):
            raise ValueError(f"invalid generated id: {block_id!r}")
        if block_id in ids:
            raise ValueError(f"duplicate generated id: {block_id}")
        if set(entry) != expected_fields:
            raise ValueError(f"{block_id}: unexpected fields")
        if not isinstance(entry["prompt"], str) or not entry["prompt"].strip():
            raise ValueError(f"{block_id}: prompt is empty")
        goal = entry["goal"]
        if not isinstance(goal, dict) or not isinstance(goal.get("target_state"), str):
            raise ValueError(f"{block_id}: target_state is missing")
        if not goal["target_state"].strip():
            raise ValueError(f"{block_id}: target_state is empty")

        answer = entry["answer"]
        suffix = block_id.rsplit("_", 1)[-1]
        if suffix not in ROW_KINDS:
            raise ValueError(f"{block_id}: unknown row kind")
        row_counts[suffix] += 1
        criteria = goal.get("success_criteria", [])
        if suffix == "criterion":
            if not answer.strip() or "\n" in answer or criteria:
                raise ValueError(f"{block_id}: invalid criterion-generation row")
            criterion_lengths.add(len(answer.split()))
        elif suffix == "evidence":
            if not answer.strip() or "\n" in answer or len(criteria) != 1:
                raise ValueError(f"{block_id}: invalid evidence-generation row")
            if set(criteria[0]) != {"criterion"}:
                raise ValueError(f"{block_id}: evidence row must omit prior evidence")
            criterion_lengths.add(len(criteria[0]["criterion"].split()))
        else:
            answer_parts = answer.splitlines()
            if len(answer_parts) != 2 or not all(part.strip() for part in answer_parts) or criteria:
                raise ValueError(f"{block_id}: invalid pair-generation row")
            criterion_lengths.add(len(answer_parts[0].split()))

        rendered = render_for_validation(entry)
        bounded_count = (
            len(tokenize(rendered, vocab, trie, detect_numbers=True).tokens)
            + source.BOUNDARY_TOKEN_BUDGET)
        if bounded_count > source.MAX_SEQUENCE_TOKENS:
            raise ValueError(
                f"{block_id}: canonical sequence uses {bounded_count} tokens, "
                f"exceeding {source.MAX_SEQUENCE_TOKENS}")
        if bounded_count > maximum_tokens:
            maximum_tokens = bounded_count
            maximum_id = block_id
        ids.add(block_id)

    expected_per_kind = len(entries) // len(ROW_KINDS)
    if any(row_counts[kind] != expected_per_kind for kind in ROW_KINDS):
        raise ValueError(f"unbalanced row kinds: {dict(row_counts)}")
    if len(criterion_lengths) < 3:
        raise ValueError("criterion wording does not vary in length")

    return {
        "entries": len(entries),
        "source_examples": expected_per_kind,
        "families": len(CONTRACTS),
        "row_kinds": dict(sorted(row_counts.items())),
        "criterion_lengths": sorted(criterion_lengths),
        "maximum_sequence_tokens": maximum_tokens,
        "maximum_sequence_id": maximum_id,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=source.DATA_DIR)
    parser.add_argument("--count", type=int, default=source.DEFAULT_COUNT)
    parser.add_argument("--seed", type=int, default=source.DEFAULT_SEED)
    parser.add_argument("--vocab", type=Path, default=source.DATA_DIR / "vocab.bin")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    entries = generate_entries(args.count, args.seed)
    report = validate_entries(entries, args.vocab)
    concept_path = args.data_dir / "concept_blocks.jsonl"
    registry_path = args.data_dir / "curriculum_registry.json"
    removed, kept = source.replace_owned_entries(
        concept_path, entries, args.dry_run, entry_prefix=ENTRY_PREFIX)
    source.update_registry(
        registry_path,
        entries,
        args.dry_run,
        curriculum_id=CURRICULUM_ID,
        curriculum_name=CURRICULUM_NAME,
    )

    action = "would replace" if args.dry_run else "replaced"
    print(
        f"{CURRICULUM_NAME}: {report['entries']} validated rows from "
        f"{report['source_examples']} source examples across {report['families']} families")
    print(f"  {action}: {removed} prior {ENTRY_PREFIX} rows; preserved: {kept} unrelated rows")
    print(f"  row kinds: {report['row_kinds']}")
    print(f"  criterion word lengths: {report['criterion_lengths']}")
    print(
        f"  maximum sequence: {report['maximum_sequence_tokens']} tokens "
        f"including BOS/EOS ({report['maximum_sequence_id']})")
    print(f"  concept blocks: {concept_path}")
    print(f"  registry: {registry_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
