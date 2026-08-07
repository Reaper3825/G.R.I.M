#!/usr/bin/env python3
"""
Stream HuggingFaceH4/ultrachat_200k and append ConceptBlock-shaped JSONL lines.

Each user -> assistant turn becomes one block:
    prompt    = user content
  answer    = assistant content
  intermediates = []  (plain instruct/chat; no forced CoT)

Usage:
  export HF_HOME=/path/in/repo/.cache/huggingface   # recommended (writable cache)
  python3 scripts/export_ultrachat_to_concept_blocks.py \\
      --append resources/models/GRIM-text/training/data/concept_blocks.jsonl \\
      --max-blocks 250000

  --max-blocks 0  means no cap (expect ~600k blocks from train_sft; large file).

Dataset: https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k
License: CC BY-NC-4.0 (see dataset card) — verify before commercial use.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from typing import Iterator, Tuple


def iter_user_assistant_pairs(messages: list) -> Iterator[Tuple[str, str]]:
    last_user: str | None = None
    for m in messages:
        role = m.get("role")
        content = (m.get("content") or "").strip()
        if role == "user":
            last_user = content if content else None
        elif role == "assistant" and last_user:
            if content:
                yield last_user, content
            last_user = None


def max_existing_uc_index(path: str) -> int:
    pat = re.compile(r'"id"\s*:\s*"cb_uc_(\d+)"')
    mmax = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            m = pat.search(line)
            if m:
                mmax = max(mmax, int(m.group(1)))
    return mmax


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--append",
        required=True,
        help="Path to concept_blocks.jsonl to append to",
    )
    ap.add_argument(
        "--split",
        default="train_sft",
        help="UltraChat split (default train_sft)",
    )
    ap.add_argument(
        "--max-blocks",
        type=int,
        default=250_000,
        help="Stop after this many new blocks (0 = no limit)",
    )
    ap.add_argument(
        "--min-chars",
        type=int,
        default=8,
        help="Skip pairs where question or answer is shorter than this",
    )
    args = ap.parse_args()

    try:
        from datasets import load_dataset
    except ImportError:
        print(
            "Install dependencies: python3 -m venv .venv_ultrachat && "
            "source .venv_ultrachat/bin/activate && pip install datasets",
            file=sys.stderr,
        )
        return 1

    out_path = os.path.abspath(args.append)
    if not os.path.isfile(out_path):
        print(f"File not found: {out_path}", file=sys.stderr)
        return 1

    start_idx = max_existing_uc_index(out_path) + 1
    written = 0
    rows_seen = 0

    # Cache under repo if HF_HOME not set (avoids home-dir permission issues)
    if not os.environ.get("HF_HOME"):
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        os.environ.setdefault(
            "HF_HOME", os.path.join(repo_root, ".cache", "huggingface")
        )
        os.makedirs(os.environ["HF_HOME"], exist_ok=True)

    ds = load_dataset(
        "HuggingFaceH4/ultrachat_200k",
        split=args.split,
        streaming=True,
    )

    with open(out_path, "a", encoding="utf-8") as out:
        for row in ds:
            rows_seen += 1
            pid = row.get("prompt_id", "")
            msgs = row.get("messages") or []
            turn = 0
            for q, a in iter_user_assistant_pairs(msgs):
                if len(q) < args.min_chars or len(a) < args.min_chars:
                    continue
                turn += 1
                idx = start_idx + written
                block = {
                    "id": f"cb_uc_{idx:08d}",
                    "name": "UltraChat",
                    "prompt": q,
                    "intermediates": [],
                    "answer": a,
                    "format_type": "chain_of_thought",
                    "timestamp": 0,
                    "source_sequence_id": f"ultrachat:{pid}:turn{turn}",
                }
                out.write(json.dumps(block, ensure_ascii=False) + "\n")
                written += 1
                if args.max_blocks and written >= args.max_blocks:
                    print(
                        f"Done (hit --max-blocks={args.max_blocks}). "
                        f"Rows scanned: {rows_seen}. Appended: {written}.",
                        file=sys.stderr,
                    )
                    return 0

    print(
        f"Done (end of stream). Rows scanned: {rows_seen}. Appended: {written}.",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
