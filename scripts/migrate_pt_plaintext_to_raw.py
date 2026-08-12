#!/usr/bin/env python3
"""Move Pre-Trainingv1 records into ConceptBlock.raw and unified membership."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "resources/models/GRIM-text/training/data"
REGISTRY_PATH = DATA_DIR / "curriculum_registry.json"
MANIFEST_PATH = DATA_DIR / "Pre-Trainingv1.json"
CONCEPT_BLOCKS_PATH = DATA_DIR / "concept_blocks.jsonl"
CURRICULUM_NAME = "Pre-Trainingv1"


def raw_text_from_legacy_block(block: dict) -> str:
    if isinstance(block.get("raw"), str) and block["raw"]:
        return block["raw"]

    parts: list[str] = []
    prompt = block.get("prompt")
    if isinstance(prompt, str) and prompt:
        parts.append(prompt)

    explanation = block.get("explanation")
    if not isinstance(explanation, list):
        explanation = block.get("intermediates")
    if isinstance(explanation, list):
        parts.extend(item for item in explanation if isinstance(item, str))

    answer = block.get("answer")
    if isinstance(answer, str) and answer:
        parts.append(answer)
    return "\n".join(parts)


def migrate_block(block: dict) -> dict:
    raw = raw_text_from_legacy_block(block)
    if not raw:
        raise ValueError(f"ConceptBlock {block.get('id', '<missing>')} has no text")

    block["format_type"] = "raw"
    block["raw"] = raw
    block["prompt"] = ""
    block["answer"] = ""
    block["intermediates"] = []
    block["explanation"] = []
    block["execution"] = []
    block["intermediate_count"] = 0
    block["step_index"] = []
    block.pop("goal", None)
    return block


def load_registry() -> tuple[dict, dict, list[str]]:
    registry = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    curriculum = next(
        (item for item in registry.get("curriculums", [])
         if item.get("name") == CURRICULUM_NAME),
        None,
    )
    if curriculum is None:
        raise RuntimeError(f"Curriculum {CURRICULUM_NAME!r} was not found")

    legacy_ids = curriculum.get("plaintext_block_ids", [])
    concept_ids = curriculum.get("concept_block_ids", [])
    if not isinstance(legacy_ids, list) or not isinstance(concept_ids, list):
        raise RuntimeError("Pre-Trainingv1 membership fields must be arrays")

    ordered_ids = list(dict.fromkeys([*concept_ids, *legacy_ids]))
    if not ordered_ids:
        raise RuntimeError("Pre-Trainingv1 has no member IDs")
    return registry, curriculum, ordered_ids


def rewrite_jsonl(member_ids: list[str], dry_run: bool) -> int:
    wanted = set(member_ids)
    found: set[str] = set()
    handle, temporary_name = tempfile.mkstemp(
        prefix="concept_blocks.raw.", suffix=".jsonl.tmp", dir=DATA_DIR)
    os.close(handle)
    temporary_path = Path(temporary_name)

    try:
        with CONCEPT_BLOCKS_PATH.open("r", encoding="utf-8") as source, \
             temporary_path.open("w", encoding="utf-8", newline="\n") as output:
            for line_number, line in enumerate(source, 1):
                if not line.strip():
                    continue
                block = json.loads(line)
                block_id = block.get("id")
                if block_id in wanted:
                    if block_id in found:
                        raise RuntimeError(f"Duplicate PT ConceptBlock ID {block_id}")
                    migrate_block(block)
                    found.add(block_id)
                output.write(json.dumps(block, ensure_ascii=False, separators=(",", ":")))
                output.write("\n")
                if line_number % 50000 == 0:
                    print(f"scanned={line_number} migrated={len(found)}", flush=True)

        missing = wanted - found
        if missing:
            sample = ", ".join(sorted(missing)[:10])
            raise RuntimeError(
                f"Refusing partial migration: {len(missing)} PT IDs were not found; {sample}")

        if dry_run:
            temporary_path.unlink()
        else:
            os.replace(temporary_path, CONCEPT_BLOCKS_PATH)
        return len(found)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise


def write_membership(registry: dict, curriculum: dict, member_ids: list[str]) -> None:
    curriculum["concept_block_ids"] = member_ids
    curriculum.pop("plaintext_block_ids", None)
    curriculum["training_stage"] = "pt"
    curriculum["format_as_concept"] = False

    temporary_registry = REGISTRY_PATH.with_suffix(".json.tmp")
    temporary_registry.write_text(
        json.dumps(registry, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temporary_registry, REGISTRY_PATH)

    temporary_manifest = MANIFEST_PATH.with_suffix(".json.tmp")
    temporary_manifest.write_text(
        json.dumps({
            "concept_block_ids": member_ids,
            "format_as_concept": False,
            "training_stage": "pt",
        }, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary_manifest, MANIFEST_PATH)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    registry, curriculum, member_ids = load_registry()
    migrated = rewrite_jsonl(member_ids, args.dry_run)
    if not args.dry_run:
        write_membership(registry, curriculum, member_ids)
    print(
        f"{'validated' if args.dry_run else 'migrated'} {migrated} raw PT ConceptBlocks",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
