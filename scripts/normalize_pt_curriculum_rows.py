#!/usr/bin/env python3
"""Normalize existing cb_pt_* concept block rows to the project house format.

This rewrites only PT rows so they:
- serialize in the same compact key order as the existing concept block corpus
- map document text into question/intermediates/answer in document order
- preserve ids, names, and overall plaintext rendering semantics
"""

from __future__ import annotations

import json
import os
import tempfile

DATA_DIR = os.path.join(
    os.path.dirname(__file__),
    "..", "resources", "models", "GRIM-text", "training", "data",
)
DATA_DIR = os.path.abspath(DATA_DIR)
CONCEPT_BLOCKS = os.path.join(DATA_DIR, "concept_blocks.jsonl")
PT_PREFIX = "cb_pt_"


def split_plaintext_document(text: str):
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return "", [], ""
    if len(lines) == 1:
        return "", [], lines[0]
    if len(lines) == 2:
        return lines[0], [], lines[1]
    return lines[0], lines[1:-1], lines[-1]


def reconstruct_plaintext(row: dict) -> str:
    parts = []
    question = row.get("prompt", "")
    if isinstance(question, str) and question.strip():
        parts.append(question.strip())

    explanation = row.get("explanation")
    if not isinstance(explanation, list):
        explanation = row.get("intermediates", [])
    if isinstance(explanation, list):
        parts.extend(
            item.strip()
            for item in explanation
            if isinstance(item, str) and item.strip()
        )

    answer = row.get("answer", "")
    if isinstance(answer, str) and answer.strip():
        parts.append(answer.strip())

    return "\n".join(parts)


def normalize_pt_row(row: dict) -> dict:
    block_id = row["id"]
    name = row.get("name", block_id)
    full_text = reconstruct_plaintext(row)
    question, intermediates, answer = split_plaintext_document(full_text)
    step_index = list(range(len(intermediates)))
    return {
        "answer": answer,
        "explanation": intermediates,
        "format_type": row.get("format_type", "chain_of_thought"),
        "id": block_id,
        "intermediate_count": len(intermediates),
        "intermediates": intermediates,
        "name": name,
        "prompt": question,
        "step_index": step_index,
        "timestamp": row.get("timestamp", 0),
    }


def main():
    fd, tmp_path = tempfile.mkstemp(prefix="concept_blocks.", suffix=".jsonl.tmp", dir=DATA_DIR)
    os.close(fd)

    total = 0
    rewritten = 0

    with open(CONCEPT_BLOCKS, "r", encoding="utf-8") as src, \
         open(tmp_path, "w", encoding="utf-8") as dst:
        for line in src:
            total += 1
            stripped = line.strip()
            if not stripped:
                dst.write("\n")
                continue

            row = json.loads(stripped)
            block_id = row.get("id", "")
            if isinstance(block_id, str) and block_id.startswith(PT_PREFIX):
                normalized = normalize_pt_row(row)
                dst.write(json.dumps(normalized, ensure_ascii=False, separators=(",", ":")) + "\n")
                rewritten += 1
            else:
                dst.write(line if line.endswith("\n") else line + "\n")

    os.replace(tmp_path, CONCEPT_BLOCKS)
    print(f"Rewrote {rewritten} PT rows out of {total} total lines in {CONCEPT_BLOCKS}")


if __name__ == "__main__":
    main()
