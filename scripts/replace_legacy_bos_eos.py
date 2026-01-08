#!/usr/bin/env python3
"""
Replace legacy BOS/EOS tokens with <s> and </s>.

Defaults to scanning resources/models/GRIM-text/training/data for .txt/.jsonl/.log.
"""

import argparse
import fnmatch
import os
import sys
import tempfile
from pathlib import Path


LEGACY_BOS = "<|startoftext|>"
LEGACY_EOS = "<|endoftext|>"
NEW_BOS = "<s>"
NEW_EOS = "</s>"


def iter_files(paths, include_patterns):
    for path in paths:
        p = Path(path)
        if p.is_dir():
            for root, _, files in os.walk(p):
                for name in files:
                    if any(fnmatch.fnmatch(name, pat) for pat in include_patterns):
                        yield Path(root) / name
        elif p.is_file():
            yield p


def replace_in_file(path, dry_run=False, backup=False, encoding="utf-8"):
    replaced_bos = 0
    replaced_eos = 0

    if dry_run:
        with path.open("r", encoding=encoding, errors="replace") as src:
            for line in src:
                replaced_bos += line.count(LEGACY_BOS)
                replaced_eos += line.count(LEGACY_EOS)
        return replaced_bos, replaced_eos, False

    tmp_path = None
    changed = False
    with path.open("r", encoding=encoding, errors="replace") as src:
        fd, tmp_name = tempfile.mkstemp(prefix=path.name, suffix=".tmp", dir=str(path.parent))
        tmp_path = Path(tmp_name)
        with os.fdopen(fd, "w", encoding=encoding, errors="replace") as dst:
            for line in src:
                if LEGACY_BOS in line or LEGACY_EOS in line:
                    replaced_bos += line.count(LEGACY_BOS)
                    replaced_eos += line.count(LEGACY_EOS)
                    line = line.replace(LEGACY_BOS, NEW_BOS).replace(LEGACY_EOS, NEW_EOS)
                    changed = True
                dst.write(line)

    if not changed:
        tmp_path.unlink(missing_ok=True)
        return replaced_bos, replaced_eos, False

    if backup:
        backup_path = path.with_suffix(path.suffix + ".bak")
        if backup_path.exists():
            backup_path.unlink()
        path.replace(backup_path)

    os.replace(tmp_path, path)
    return replaced_bos, replaced_eos, True


def main():
    parser = argparse.ArgumentParser(description="Replace legacy BOS/EOS tokens with <s> and </s>.")
    parser.add_argument(
        "paths",
        nargs="*",
        default=["resources/models/GRIM-text/training/data"],
        help="Files or directories to scan (default: training/data).",
    )
    parser.add_argument(
        "--include",
        nargs="*",
        default=["*.txt", "*.jsonl", "*.log"],
        help="Filename patterns to include when scanning directories.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Report changes without writing.")
    parser.add_argument("--backup", action="store_true", help="Keep a .bak copy of each modified file.")
    args = parser.parse_args()

    total_bos = 0
    total_eos = 0
    touched = 0
    scanned = 0

    for path in iter_files(args.paths, args.include):
        scanned += 1
        bos, eos, changed = replace_in_file(path, dry_run=args.dry_run, backup=args.backup)
        total_bos += bos
        total_eos += eos
        if changed:
            touched += 1
            print(f"[updated] {path}")
        elif bos or eos:
            print(f"[match]   {path}")

    print(f"Scanned: {scanned} files")
    print(f"Replaced BOS: {total_bos}")
    print(f"Replaced EOS: {total_eos}")
    if args.dry_run:
        print("Dry run only; no files modified.")
    else:
        print(f"Updated files: {touched}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
