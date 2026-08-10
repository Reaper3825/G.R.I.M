#!/usr/bin/env python3
"""Copy Intent Action v1 answers into structured goal target-state spans."""

from __future__ import annotations

import argparse
import collections
import re
from pathlib import Path
from typing import Any

import generate_intent_action_curriculum as source
from intent_action_curriculum_content import BUILDERS
from vocab_playground import build_trie, load_vocab_bin, tokenize

CURRICULUM_ID = "curr_intent_action_target_state_v1"
CURRICULUM_NAME = "Intent Action Target-State v1"
ENTRY_PREFIX = "iatsv1_"
WORD_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")


def make_entry(index: int, seed: int = source.DEFAULT_SEED) -> dict[str, Any]:
    original = source.make_entry(index, seed)
    return {
        "id": f"{ENTRY_PREFIX}{index:05d}",
        "prompt": original["prompt"],
        "answer": "",
        "goal": {"target_state": original["answer"]},
    }


def generate_entries(
    count: int,
    seed: int = source.DEFAULT_SEED,
) -> list[dict[str, Any]]:
    if count <= 0:
        raise ValueError("count must be positive")
    return [make_entry(index, seed) for index in range(count)]


def validate_entries(entries: list[dict[str, Any]], vocab_path: Path) -> dict[str, Any]:
    vocab = load_vocab_bin(vocab_path)
    trie = build_trie(vocab)
    ids: set[str] = set()
    prompts: set[str] = set()
    target_states: collections.Counter[str] = collections.Counter()
    target_length_counts: collections.Counter[str] = collections.Counter()
    maximum_tokens = 0
    maximum_id = ""
    expected_keys = {"id", "prompt", "answer", "goal"}

    for entry in entries:
        block_id = entry.get("id")
        prompt = entry.get("prompt")
        answer = entry.get("answer")
        goal = entry.get("goal")
        target_state = goal.get("target_state") if isinstance(goal, dict) else None

        if not isinstance(block_id, str) or not block_id.startswith(ENTRY_PREFIX):
            raise ValueError(f"invalid generated id: {block_id!r}")
        if block_id in ids:
            raise ValueError(f"duplicate generated id: {block_id}")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError(f"{block_id}: prompt is empty")
        if prompt in prompts:
            raise ValueError(f"duplicate generated prompt: {prompt}")
        if answer != "":
            raise ValueError(f"{block_id}: answer must be empty after target-state move")
        if set(entry) != expected_keys:
            raise ValueError(
                f"{block_id}: fields must be exactly {sorted(expected_keys)}, "
                f"found {sorted(entry)}")
        if not isinstance(target_state, str) or not target_state.strip():
            raise ValueError(f"{block_id}: goal.target_state is empty")
        if set(goal) != {"target_state"}:
            raise ValueError(f"{block_id}: goal must contain only target_state")
        if any(character.isdigit() for character in target_state):
            raise ValueError(f"{block_id}: target_state contains a number: {target_state!r}")

        # Concept-mode rendering is prompt, blank separator, then target state.
        rendered = f"{prompt}\n\n{target_state}\n"
        token_count = len(tokenize(rendered, vocab, trie, detect_numbers=True).tokens)
        bounded_count = token_count + source.BOUNDARY_TOKEN_BUDGET
        if bounded_count > source.MAX_SEQUENCE_TOKENS:
            raise ValueError(
                f"{block_id}: canonical sequence uses {bounded_count} tokens, "
                f"exceeding {source.MAX_SEQUENCE_TOKENS}")
        if bounded_count > maximum_tokens:
            maximum_tokens = bounded_count
            maximum_id = block_id

        ids.add(block_id)
        prompts.add(prompt)
        target_states[target_state] += 1
        target_words = len(WORD_RE.findall(target_state))
        if target_words <= 3:
            target_length_counts["short"] += 1
        elif target_words <= 7:
            target_length_counts["medium"] += 1
        else:
            target_length_counts["long"] += 1

    missing_lengths = {"short", "medium", "long"} - target_length_counts.keys()
    if missing_lengths:
        raise ValueError(
            f"target-state length variation is missing: {sorted(missing_lengths)}")

    return {
        "entries": len(entries),
        "families": len(BUILDERS),
        "unique_target_states": len(target_states),
        "target_state_lengths": dict(sorted(target_length_counts.items())),
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
        concept_path,
        entries,
        args.dry_run,
        entry_prefix=ENTRY_PREFIX,
    )
    source.update_registry(
        registry_path,
        entries,
        args.dry_run,
        curriculum_id=CURRICULUM_ID,
        curriculum_name=CURRICULUM_NAME,
    )

    action = "would replace" if args.dry_run else "replaced"
    print(
        f"{CURRICULUM_NAME}: {report['entries']} validated blocks "
        f"across {report['families']} families")
    print(f"  {action}: {removed} prior {ENTRY_PREFIX} rows; preserved: {kept} unrelated rows")
    print(
        f"  target states: {report['unique_target_states']} unique; "
        f"lengths: {report['target_state_lengths']}")
    print(
        f"  maximum sequence: {report['maximum_sequence_tokens']} tokens "
        f"including BOS/EOS ({report['maximum_sequence_id']})")
    print(f"  concept blocks: {concept_path}")
    print(f"  registry: {registry_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
