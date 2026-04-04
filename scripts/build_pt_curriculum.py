#!/usr/bin/env python3
"""
Pull FineWeb-Edu documents and create Pre-Trainingv1 curriculum for GRIM-text.

Streams from HuggingFace, filters for quality/length, converts to concept block
format with plaintext rendering (no execution payload).
"""

import datasets
from contextlib import suppress
import json
import hashlib
import time
import sys
import os

DATA_DIR = os.path.join(os.path.dirname(__file__),
    "..", "resources", "models", "GRIM-text", "training", "data")
DATA_DIR = os.path.abspath(DATA_DIR)

CONCEPT_BLOCKS = os.path.join(DATA_DIR, "concept_blocks.jsonl")
CURRICULUM_REGISTRY = os.path.join(DATA_DIR, "curriculum_registry.json")
CURRICULUM_NAME = "Pre-Trainingv1"

# FineWeb-Edu token_count is GPT-2 tokens (~4 chars/token).
# We want docs that fill ~2048 UnigramByte tokens well.
# Conservative: 400-3000 GPT-2 tokens ≈ 1600-12000 chars.
MIN_TOKENS = 400
MAX_TOKENS = 3000
MIN_SCORE = 3.0   # Educational quality >= 3/5
TARGET_COUNT = 50000


def split_plaintext_document(text: str):
    """Preserve document order while mapping it into ConceptBlock plaintext fields.

    For PT rows we want the rendered plaintext path to reconstruct the original
    document as:

        question + "\n" + intermediates... + "\n" + answer

    So we split by non-empty lines, using the first line as the leading prompt
    or title, middle lines as the body, and the final line as the trailing
    conclusion. Single-line documents are stored entirely in `answer` so the
    rendered text still contains the full document.
    """
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return "", [], ""
    if len(lines) == 1:
        return "", [], lines[0]
    if len(lines) == 2:
        return lines[0], [], lines[1]
    return lines[0], lines[1:-1], lines[-1]


def build_pt_block(index: int, block_id: str, text: str):
    question, intermediates, answer = split_plaintext_document(text)
    step_index = list(range(len(intermediates)))
    return {
        "answer": answer,
        "explanation": intermediates,
        "format_type": "chain_of_thought",
        "id": block_id,
        "intermediate_count": len(intermediates),
        "intermediates": intermediates,
        "name": f"fineweb_edu_{index:06d}",
        "question": question,
        "step_index": step_index,
        "timestamp": 0,
    }

def main():
    print("Loading FineWeb-Edu (streaming, sample-10BT)...")
    ds = datasets.load_dataset(
        "HuggingFaceFW/fineweb-edu",
        name="sample-10BT",
        split="train",
        streaming=True,
    )

    # Load existing concept block IDs to avoid duplicates
    existing_ids = set()
    with open(CONCEPT_BLOCKS) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            with suppress(Exception):
                j = json.loads(line)
                existing_ids.add(j.get("id", ""))
    print(f"Existing concept blocks: {len(existing_ids)}")

    # Stream and filter
    collected = []
    scanned = 0
    skipped_quality = 0
    skipped_length = 0
    skipped_lang = 0
    t0 = time.time()

    for sample in ds:
        scanned += 1

        if sample.get("language") != "en" or sample.get("language_score", 0) < 0.9:
            skipped_lang += 1
            continue

        score = sample.get("score", 0)
        if score < MIN_SCORE:
            skipped_quality += 1
            continue

        tc = sample.get("token_count", 0)
        if tc < MIN_TOKENS or tc > MAX_TOKENS:
            skipped_length += 1
            continue

        text = sample.get("text", "").strip()
        if not text or len(text) < 200:
            continue

        content_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]
        block_id = f"cb_pt_{content_hash}"

        if block_id in existing_ids:
            continue

        collected.append(build_pt_block(len(collected), block_id, text))
        existing_ids.add(block_id)

        if len(collected) % 5000 == 0:
            elapsed = time.time() - t0
            print(f"  Collected {len(collected)}/{TARGET_COUNT} "
                  f"(scanned {scanned}, {elapsed:.1f}s, "
                  f"skip: quality={skipped_quality} length={skipped_length} lang={skipped_lang})")

        if len(collected) >= TARGET_COUNT:
            break

    elapsed = time.time() - t0
    print(f"\nDone: collected {len(collected)} documents in {elapsed:.1f}s")
    print(f"Scanned {scanned} total, skipped: "
          f"quality={skipped_quality}, length={skipped_length}, lang={skipped_lang}")

    if not collected:
        print("ERROR: No documents collected!")
        sys.exit(1)

    # Append to concept_blocks.jsonl
    print(f"Appending {len(collected)} blocks to concept_blocks.jsonl...")
    with open(CONCEPT_BLOCKS, "a") as f:
        for block in collected:
            f.write(json.dumps(block, ensure_ascii=False, separators=(",", ":")) + "\n")

    pt_block_ids = [b["id"] for b in collected]

    # Update curriculum registry
    print("Updating curriculum registry...")
    with open(CURRICULUM_REGISTRY) as f:
        registry = json.load(f)

    found_idx = next(
        (i for i, curr in enumerate(registry["curriculums"])
         if curr["name"] == CURRICULUM_NAME),
        None,
    )

    if found_idx is not None:
        registry["curriculums"][found_idx]["concept_block_ids"] = pt_block_ids
        registry["curriculums"][found_idx]["format_as_concept"] = False
        curr_id = registry["curriculums"][found_idx]["id"]
        print(f"Updated '{CURRICULUM_NAME}' (id={curr_id}) with {len(pt_block_ids)} blocks")
    else:
        curr_id = f"curr_pt_{hashlib.sha256(CURRICULUM_NAME.encode()).hexdigest()[:16]}"
        registry["curriculums"].append({
            "id": curr_id,
            "name": CURRICULUM_NAME,
            "concept_block_ids": pt_block_ids,
            "format_as_concept": False,
            "timestamp": int(time.time()),
        })
        print(f"Created '{CURRICULUM_NAME}' (id={curr_id}) with {len(pt_block_ids)} blocks")

    with open(CURRICULUM_REGISTRY, "w") as f:
        json.dump(registry, f, indent=2, ensure_ascii=False)
        f.write("\n")

    # Create the named curriculum manifest: Pre-Trainingv1.json
    manifest = {
        "concept_block_ids": [],
        "plaintext_block_ids": pt_block_ids,
    }
    manifest_path = os.path.join(DATA_DIR, f"{CURRICULUM_NAME}.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, ensure_ascii=False)
        f.write("\n")
    print(f"Wrote curriculum manifest: {manifest_path}")
    print(f"  plaintext_block_ids: {len(pt_block_ids)}")
    print("  concept_block_ids: 0 (pure PT, no execution blocks)")

    # Show sample
    print(f"\nSample document (first 300 chars):")
    print(collected[0]["question"][:300])
    print("...")

    print(f"\nDone! Set ai_config.json training.current_curriculum to '{CURRICULUM_NAME}' to use.")


if __name__ == "__main__":
    main()
