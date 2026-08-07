#!/usr/bin/env python3
"""Build concept_blocks.fb from concept_blocks.jsonl with vcpkg's flatc."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile
from pathlib import Path


def default_flatc(repo_root: Path) -> Path:
    executable = "flatc.exe" if os.name == "nt" else "flatc"
    triplet = "x64-windows" if os.name == "nt" else "x64-linux"
    return repo_root / "vcpkg_installed" / triplet / "tools" / "flatbuffers" / executable


def write_dataset_envelope(jsonl_path: Path, output_path: Path) -> int:
    count = 0
    with jsonl_path.open("r", encoding="utf-8") as source, output_path.open(
        "w", encoding="utf-8", newline="\n"
    ) as output:
        output.write('{"schema_version":2,"blocks":[')
        for line_number, line in enumerate(source, 1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                entry = json.loads(stripped)
            except json.JSONDecodeError as error:
                raise ValueError(f"{jsonl_path}:{line_number}: {error}") from error
            if not isinstance(entry.get("id"), str) or not entry["id"]:
                raise ValueError(f"{jsonl_path}:{line_number}: missing string id")
            if "question" in entry:
                raise ValueError(f"{jsonl_path}:{line_number}: legacy question field is forbidden")
            prompt = entry.get("prompt")
            if not isinstance(prompt, str) or not prompt:
                raise ValueError(f"{jsonl_path}:{line_number}: missing non-empty prompt")

            entry.pop("execution_gate_target", None)
            entry.pop("state_0", None)
            entry.pop("state_1", None)
            if count:
                output.write(",")
            # Preserve numeric JSON types while normalizing syntax for flatc.
            output.write(json.dumps(entry, ensure_ascii=False, separators=(",", ":")))
            count += 1
            if count % 50000 == 0:
                print(f"Envelope: {count} concept blocks", flush=True)
        output.write("]}")
        output.flush()
        os.fsync(output.fileno())
    return count


def build_flatbuffer(jsonl_path: Path, fb_path: Path, schema_path: Path, flatc: Path) -> int:
    if not flatc.is_file():
        raise FileNotFoundError(f"flatc not found: {flatc}")
    if not schema_path.is_file():
        raise FileNotFoundError(f"schema not found: {schema_path}")
    if not jsonl_path.is_file():
        raise FileNotFoundError(f"JSONL not found: {jsonl_path}")

    fb_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="concept_blocks_flatc_", dir=fb_path.parent) as temporary:
        temporary_path = Path(temporary)
        envelope_path = temporary_path / "concept_blocks.json"
        block_count = write_dataset_envelope(jsonl_path, envelope_path)
        print(f"Running {flatc} for {block_count} concept blocks...", flush=True)
        subprocess.run(
            [str(flatc), "--binary", "--unknown-json", "-o", str(temporary_path),
             str(schema_path), str(envelope_path)],
            check=True,
        )
        generated_path = temporary_path / "concept_blocks.fb"
        if not generated_path.is_file():
            fallback = temporary_path / "concept_blocks.bin"
            if fallback.is_file():
                generated_path = fallback
            else:
                raise RuntimeError("flatc did not produce concept_blocks.fb or concept_blocks.bin")
        with generated_path.open("rb") as generated:
            header = generated.read(8)
        if len(header) != 8 or header[4:8] != b"GRCB":
            raise RuntimeError("generated FlatBuffer is missing the GRCB file identifier")
        os.replace(generated_path, fb_path)
        print(
            f"Wrote {block_count} concept blocks to {fb_path} "
            f"({fb_path.stat().st_size} bytes)",
            flush=True,
        )
        return block_count


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--jsonl", type=Path,
        default=repo_root / "resources/models/GRIM-text/training/data/concept_blocks.jsonl")
    parser.add_argument(
        "--output", type=Path,
        default=repo_root / "resources/models/GRIM-text/training/data/concept_blocks.fb")
    parser.add_argument(
        "--schema", type=Path,
        default=repo_root / "DataCollection/concept_block.fbs")
    parser.add_argument("--flatc", type=Path, default=default_flatc(repo_root))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    build_flatbuffer(
        args.jsonl.resolve(), args.output.resolve(),
        args.schema.resolve(), args.flatc.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
