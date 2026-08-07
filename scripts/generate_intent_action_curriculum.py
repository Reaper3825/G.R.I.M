#!/usr/bin/env python3
"""Generate the broad, general-purpose GRIM-text Intent Action v1 curriculum."""

from __future__ import annotations

import argparse
import collections
import json
import os
import re
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from vocab_playground import build_trie, load_vocab_bin, tokenize  # noqa: E402
from intent_action_curriculum_content import BUILDERS, generate_content  # noqa: E402

DATA_DIR = ROOT_DIR / "resources" / "models" / "GRIM-text" / "training" / "data"
CURRICULUM_ID = "curr_intent_action_v1"
CURRICULUM_NAME = "Intent Action v1"
ENTRY_PREFIX = "iav1_"
DEFAULT_COUNT = 10_000
DEFAULT_SEED = 1_780_605
MAX_SEQUENCE_TOKENS = 1024
BOUNDARY_TOKEN_BUDGET = 2
WORD_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")


def make_entry(index: int, seed: int = DEFAULT_SEED) -> dict[str, Any]:
    content = generate_content(index, seed)
    block_id = f"{ENTRY_PREFIX}{index:05d}"
    return {
        "id": block_id,
        "prompt": content.prompt,
        "answer": content.answer,
        "goal": {"target_state": ""},
    }


def generate_entries(count: int, seed: int = DEFAULT_SEED) -> list[dict[str, Any]]:
    if count <= 0:
        raise ValueError("count must be positive")
    return [make_entry(index, seed) for index in range(count)]


def validate_entries(entries: list[dict[str, Any]], vocab_path: Path) -> dict[str, Any]:
    vocab = load_vocab_bin(vocab_path)
    trie = build_trie(vocab)
    ids: set[str] = set()
    prompts: set[str] = set()
    answer_counts: collections.Counter[str] = collections.Counter()
    answer_length_counts: collections.Counter[str] = collections.Counter()
    first_token_counts: collections.Counter[str] = collections.Counter()
    max_tokens = 0
    max_id = ""
    expected_keys = {"id", "prompt", "answer", "goal"}

    for entry in entries:
        block_id = entry.get("id")
        prompt = entry.get("prompt")
        answer = entry.get("answer")
        if not isinstance(block_id, str) or not block_id.startswith(ENTRY_PREFIX):
            raise ValueError(f"invalid generated id: {block_id!r}")
        if block_id in ids:
            raise ValueError(f"duplicate generated id: {block_id}")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError(f"{block_id}: prompt is empty")
        if prompt in prompts:
            raise ValueError(f"duplicate generated prompt: {prompt}")
        if not isinstance(answer, str) or not answer.strip():
            raise ValueError(f"{block_id}: answer is empty")
        if set(entry) != expected_keys:
            raise ValueError(
                f"{block_id}: fields must be exactly {sorted(expected_keys)}, "
                f"found {sorted(entry)}")
        if any(character.isdigit() for character in answer):
            raise ValueError(f"{block_id}: answer contains a number: {answer!r}")
        if entry.get("goal") != {"target_state": ""}:
            raise ValueError(f"{block_id}: goal must contain an empty target_state")

        rendered = f"{prompt}\n{answer}\n"
        token_count = len(tokenize(rendered, vocab, trie, detect_numbers=True).tokens)
        bounded_count = token_count + BOUNDARY_TOKEN_BUDGET
        if bounded_count > MAX_SEQUENCE_TOKENS:
            raise ValueError(
                f"{block_id}: canonical sequence uses {bounded_count} tokens, "
                f"exceeding {MAX_SEQUENCE_TOKENS}"
            )
        if bounded_count > max_tokens:
            max_tokens = bounded_count
            max_id = block_id

        words = WORD_RE.findall(prompt.lower())
        if not words:
            raise ValueError(f"{block_id}: prompt has no lexical tokens")
        ids.add(block_id)
        prompts.add(prompt)
        answer_counts[answer] += 1
        answer_words = len(WORD_RE.findall(answer))
        if answer_words <= 3:
            answer_length_counts["short"] += 1
        elif answer_words <= 7:
            answer_length_counts["medium"] += 1
        else:
            answer_length_counts["long"] += 1
        first_token_counts[words[0]] += 1

    missing_lengths = {"short", "medium", "long"} - answer_length_counts.keys()
    if missing_lengths:
        raise ValueError(f"answer-length variation is missing: {sorted(missing_lengths)}")
    if len(entries) >= 100:
        top_answer, top_answer_count = answer_counts.most_common(1)[0]
        if top_answer_count / len(entries) > 0.02:
            raise ValueError(
                f"exact-answer concentration is too high: {top_answer!r} "
                f"appears {top_answer_count}/{len(entries)} times")
        top_token, top_token_count = first_token_counts.most_common(1)[0]
        if top_token_count / len(entries) > 0.20:
            raise ValueError(
                f"prompt first-token concentration is too high: {top_token!r} "
                f"appears {top_token_count}/{len(entries)} times")

    return {
        "entries": len(entries),
        "families": len(BUILDERS),
        "unique_answers": len(answer_counts),
        "top_exact_answer_count": answer_counts.most_common(1)[0][1],
        "answer_lengths": dict(sorted(answer_length_counts.items())),
        "first_tokens": len(first_token_counts),
        "maximum_sequence_tokens": max_tokens,
        "maximum_sequence_id": max_id,
    }


def replace_owned_entries(path: Path, entries: list[dict[str, Any]], dry_run: bool) -> tuple[int, int]:
    if not path.exists():
        raise FileNotFoundError(f"concept block dataset not found: {path}")
    kept = 0
    removed = 0
    seen_owned_ids: set[str] = set()
    owned_marker = ENTRY_PREFIX.encode("ascii")
    if dry_run:
        with path.open("rb") as source:
            for line_number, line in enumerate(source, start=1):
                if not line.strip():
                    raise ValueError(f"{path}:{line_number}: blank JSONL row")
                if owned_marker in line:
                    row = json.loads(line)
                    block_id = row.get("id")
                    if not isinstance(block_id, str) or not block_id.startswith(ENTRY_PREFIX):
                        raise ValueError(
                            f"{path}:{line_number}: {ENTRY_PREFIX} appears outside an owned id")
                    if block_id in seen_owned_ids:
                        raise ValueError(f"{path}:{line_number}: duplicate owned id {block_id}")
                    seen_owned_ids.add(block_id)
                    removed += 1
                else:
                    kept += 1
        return removed, kept

    fd, stage_name = tempfile.mkstemp(prefix="intent_action.", suffix=".jsonl.tmp", dir=path.parent)
    os.close(fd)
    stage_path = Path(stage_name)
    try:
        with path.open("rb") as source, stage_path.open("wb") as destination:
            for line_number, line in enumerate(source, start=1):
                if not line.strip():
                    raise ValueError(f"{path}:{line_number}: blank JSONL row")
                if owned_marker in line:
                    row = json.loads(line)
                    block_id = row.get("id")
                    if not isinstance(block_id, str) or not block_id.startswith(ENTRY_PREFIX):
                        raise ValueError(
                            f"{path}:{line_number}: {ENTRY_PREFIX} appears outside an owned id")
                    if block_id in seen_owned_ids:
                        raise ValueError(f"{path}:{line_number}: duplicate owned id {block_id}")
                    seen_owned_ids.add(block_id)
                    removed += 1
                    continue
                destination.write(line)
                if not line.endswith(b"\n"):
                    destination.write(b"\n")
                kept += 1
            for entry in entries:
                serialized = json.dumps(
                    entry, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
                destination.write(serialized + b"\n")
            destination.flush()
            os.fsync(destination.fileno())
        if not dry_run:
            os.replace(stage_path, path)
    finally:
        stage_path.unlink(missing_ok=True)
    return removed, kept


def update_registry(path: Path, entries: list[dict[str, Any]], dry_run: bool) -> None:
    if not path.exists():
        raise FileNotFoundError(f"curriculum registry not found: {path}")
    registry = json.loads(path.read_text(encoding="utf-8"))
    curricula = registry.get("curriculums")
    if not isinstance(curricula, list):
        raise ValueError("curriculum registry must contain a curriculums array")
    matches = [item for item in curricula if item.get("id") == CURRICULUM_ID]
    if len(matches) > 1:
        raise ValueError(f"curriculum registry contains duplicate id {CURRICULUM_ID}")
    curriculum = matches[0] if matches else {"id": CURRICULUM_ID, "timestamp": int(time.time())}
    if not matches:
        curricula.append(curriculum)
    curriculum.update({
        "name": CURRICULUM_NAME,
        "format_as_concept": True,
        "concept_block_ids": [entry["id"] for entry in entries],
    })
    if dry_run:
        return
    fd, stage_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    os.close(fd)
    stage_path = Path(stage_name)
    try:
        with stage_path.open("w", encoding="utf-8", newline="\n") as output:
            json.dump(registry, output, ensure_ascii=False, indent=2)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(stage_path, path)
    finally:
        stage_path.unlink(missing_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--count", type=int, default=DEFAULT_COUNT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--vocab", type=Path, default=DATA_DIR / "vocab.bin")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    entries = generate_entries(args.count, args.seed)
    report = validate_entries(entries, args.vocab)
    concept_path = args.data_dir / "concept_blocks.jsonl"
    registry_path = args.data_dir / "curriculum_registry.json"
    removed, kept = replace_owned_entries(concept_path, entries, args.dry_run)
    update_registry(registry_path, entries, args.dry_run)

    action = "would replace" if args.dry_run else "replaced"
    print(f"{CURRICULUM_NAME}: {report['entries']} validated blocks across {report['families']} families")
    print(f"  {action}: {removed} prior {ENTRY_PREFIX} rows; preserved: {kept} unrelated rows")
    print(
        f"  answer diversity: {report['unique_answers']} unique; "
        f"most common exact answer count={report['top_exact_answer_count']}"
    )
    print(f"  answer lengths: {report['answer_lengths']}; distinct prompt starts: {report['first_tokens']}")
    print(
        f"  maximum sequence: {report['maximum_sequence_tokens']} tokens "
        f"including BOS/EOS ({report['maximum_sequence_id']})"
    )
    print(f"  concept blocks: {concept_path}")
    print(f"  registry: {registry_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
