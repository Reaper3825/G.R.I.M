#!/usr/bin/env python3
"""
Phase A of the HyperParameters refactor.

Move TrainingHyperparameters (struct + LogRecorderConfig + TapeLogConfig + all
loader/derivation/validation helpers) from control/ai_config_paths.hpp into
resources/models/GRIM-text/Shared/HyperParameters/HyperParameters_GPU.hpp.

After this script:
  - HyperParameters_GPU.hpp owns the struct definition and its loaders.
  - ai_config_paths.hpp #includes HyperParameters_GPU.hpp and only owns
    GrimTextPaths / DataCollectionConfig / TokenizerConfig / AiConfigSnapshot
    plus the snapshot loaders that aggregate them.
  - The previous include-cycle from HyperParameters_GPU.hpp -> ai_config_paths.hpp
    is removed, including the GRIM_CONFIG_AI_CONFIG_PATHS_HPP_INCLUDED guard.

Idempotency: this script is a one-shot move. Re-running it after success will
be a no-op because the source markers will be missing; it will print "nothing
to extract" and exit 1 so CI catches accidental re-runs.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path("/Users/austinwadkins/G.R.I.M")
AI_CONFIG = ROOT / "control" / "ai_config_paths.hpp"
HYPER_HPP = ROOT / "resources" / "models" / "GRIM-text" / "Shared" / "HyperParameters" / "HyperParameters_GPU.hpp"


def die(msg: str) -> None:
    print(f"ERROR: {msg}", file=sys.stderr)
    sys.exit(1)


def find_block(text: str, start_pat: re.Pattern, label: str, *, brace_open: str = "{") -> tuple[int, int]:
    """Find the byte range [start, end) of a balanced-brace block whose declaration
    matches start_pat. The block is terminated when brace depth returns to 0 after
    the first '{'. Includes a trailing semicolon if present (for struct definitions)
    and the immediately preceding doc comment if any."""
    m = start_pat.search(text)
    if not m:
        die(f"could not locate {label}")
    decl_start = m.start()

    # Walk back to include leading doc comment (/** ... */) and any blank lines.
    pos = decl_start
    # Skip back over whitespace / blank lines
    while pos > 0 and text[pos - 1] in " \t":
        pos -= 1
    if pos > 0 and text[pos - 1] == "\n":
        # Try to include preceding /** ... */ block
        # Find the previous non-blank line
        scan = pos - 1
        # Walk over blank lines
        while scan > 0 and text[scan] == "\n" and (scan == 0 or text[scan - 1] in "\n"):
            scan -= 1
        # Look for end of /** ... */
        # We do a simple regex search backwards from pos for "/**" ... "*/" immediately
        comment_end = text.rfind("*/", 0, decl_start)
        if comment_end != -1:
            # Confirm it's the comment immediately before (only whitespace/newlines between)
            between = text[comment_end + 2 : decl_start]
            if between.strip() == "":
                comment_start = text.rfind("/**", 0, comment_end)
                if comment_start != -1:
                    decl_start = comment_start

    # Find first '{' at or after match
    brace = text.find(brace_open, m.end() - 1)
    if brace == -1:
        die(f"no {{ after {label}")
    depth = 0
    i = brace
    while i < len(text):
        c = text[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                # Include trailing ';' if present (for struct foo { ... };)
                j = i + 1
                while j < len(text) and text[j] in " \t":
                    j += 1
                if j < len(text) and text[j] == ";":
                    j += 1
                # Include trailing newline
                if j < len(text) and text[j] == "\n":
                    j += 1
                return decl_start, j
        i += 1
    die(f"unterminated block for {label}")


def extract_and_remove(text: str, start_pat: re.Pattern, label: str) -> tuple[str, str]:
    """Returns (extracted_text, text_with_block_removed)."""
    s, e = find_block(text, start_pat, label)
    return text[s:e], text[:s] + text[e:]


def main() -> int:
    ai_text = AI_CONFIG.read_text()
    hyper_text = HYPER_HPP.read_text()

    extracted: list[str] = []
    new_ai = ai_text

    # Order matters for line-number stability: extract from the bottom up so that
    # earlier line numbers don't shift. We use unique regex anchors so order
    # within a single function doesn't matter, but we still go bottom-up to be
    # extra safe against accidental partial overlaps.

    # ── 1. TrainingHyperparameters struct (line ~160) ────────────────────────
    pat_struct = re.compile(r"\bstruct\s+TrainingHyperparameters\s*\{", re.MULTILINE)

    # ── 2. LogRecorderConfig struct ──────────────────────────────────────────
    pat_log = re.compile(r"\bstruct\s+LogRecorderConfig\s*\{", re.MULTILINE)

    # ── 3. TapeLogConfig struct ──────────────────────────────────────────────
    pat_tape = re.compile(r"\bstruct\s+TapeLogConfig\s*\{", re.MULTILINE)

    # ── 4. assignTrainingField template (line ~628) ──────────────────────────
    # template<typename FieldType>\ninline void assignTrainingField(...)
    pat_assign = re.compile(
        r"template\s*<\s*typename\s+FieldType\s*>\s*\n"
        r"inline\s+void\s+assignTrainingField\s*\(",
        re.MULTILINE,
    )

    # ── 5. setDefaultHyperparameters (line ~661) ─────────────────────────────
    pat_setdef = re.compile(
        r"inline\s+void\s+setDefaultHyperparameters\s*\(\s*TrainingHyperparameters\s*&", re.MULTILINE
    )

    # ── 6. validateTrainingConfigJson (line ~801) ────────────────────────────
    pat_validate = re.compile(
        r"inline\s+void\s+validateTrainingConfigJson\s*\(", re.MULTILINE
    )

    # ── 7. applyTrainingConfigObject (line ~894) ─────────────────────────────
    pat_apply = re.compile(
        r"inline\s+void\s+applyTrainingConfigObject\s*\(", re.MULTILINE
    )

    # ── 8. deriveComputedHyperparameters (line ~1455) ────────────────────────
    pat_derive_comp = re.compile(
        r"inline\s+void\s+deriveComputedHyperparameters\s*\(", re.MULTILINE
    )

    # ── 9. INNER deriveWarmupSteps (inside namespace detail, line ~1499) ────
    # Match the FIRST occurrence inside the file. We extract both occurrences
    # later via the public-facing one (which we KEEP in HyperParameters but
    # also need to delete from ai_config_paths).
    pat_derive_warm = re.compile(
        r"inline\s+void\s+deriveWarmupSteps\s*\(", re.MULTILINE
    )

    # ── 10. populateTrainingHyperparametersFromConfig (line ~1511) ──────────
    pat_populate = re.compile(
        r"inline\s+bool\s+populateTrainingHyperparametersFromConfig\s*\(", re.MULTILINE
    )

    # ── 11. loadTrainingHyperparameters (line ~1688) ────────────────────────
    pat_loadhp = re.compile(
        r"inline\s+bool\s+loadTrainingHyperparameters\s*\(", re.MULTILINE
    )

    # Extract bottom-up: highest line number first.
    moves = [
        ("loadTrainingHyperparameters", pat_loadhp),
        ("populateTrainingHyperparametersFromConfig", pat_populate),
        ("deriveWarmupSteps (inner, namespace detail)", pat_derive_warm),
        ("deriveWarmupSteps (outer, namespace Config)", pat_derive_warm),  # second occurrence
        ("deriveComputedHyperparameters", pat_derive_comp),
        ("applyTrainingConfigObject", pat_apply),
        ("validateTrainingConfigJson", pat_validate),
        ("setDefaultHyperparameters", pat_setdef),
        ("assignTrainingField", pat_assign),
        ("TrainingHyperparameters struct", pat_struct),
        ("TapeLogConfig struct", pat_tape),
        ("LogRecorderConfig struct", pat_log),
    ]

    # We need to be careful: pat_derive_warm matches twice. We extract both, but
    # only keep ONE in the HyperParameters file (they're identical). We do this
    # by tracking previously-extracted texts and deduping.
    seen_extracted_signatures: set[str] = set()
    moved_blocks: list[tuple[str, str]] = []  # (label, code)

    for label, pat in moves:
        try:
            block, new_ai = extract_and_remove(new_ai, pat, label)
        except SystemExit:
            print(f"  warn: {label} not found, skipping", file=sys.stderr)
            continue
        sig = block.strip()
        if sig in seen_extracted_signatures:
            print(f"  dedup: dropping duplicate {label}")
            continue
        seen_extracted_signatures.add(sig)
        moved_blocks.append((label, block))
        print(f"  extracted: {label}  ({len(block)} bytes)")

    if not moved_blocks:
        die("nothing to extract — script may have already been run")

    # Reverse so we re-emit in declaration order (top-down).
    moved_blocks.reverse()

    # ── Build the insertion text for HyperParameters_GPU.hpp ────────────────
    # We emit it inside `namespace GRIM { namespace Config { ... } }` so the
    # qualified name `GRIM::Config::TrainingHyperparameters` is preserved
    # exactly, with no shim.
    insertion_lines: list[str] = []
    insertion_lines.append("")
    insertion_lines.append("//======================================================//")
    insertion_lines.append("// TrainingHyperparameters + JSON loaders (Phase A move)")
    insertion_lines.append("//")
    insertion_lines.append("// Previously lived in control/ai_config_paths.hpp. Moved here so that")
    insertion_lines.append("// HyperParameters_GPU.hpp is the single source of truth and consumers")
    insertion_lines.append("// no longer need to include the JSON-reader header directly.")
    insertion_lines.append("//======================================================//")
    insertion_lines.append("")
    insertion_lines.append("#ifndef __CUDACC__")
    insertion_lines.append("")
    insertion_lines.append("namespace GRIM {")
    insertion_lines.append("namespace Config {")
    insertion_lines.append("")
    for label, block in moved_blocks:
        insertion_lines.append(f"// ── {label} ──")
        insertion_lines.append(block.rstrip())
        insertion_lines.append("")
    insertion_lines.append("namespace detail {")
    insertion_lines.append("// (helpers nested under detail in the original file are now top-level in")
    insertion_lines.append("//  Config; the empty detail namespace is preserved as a forward-compat")
    insertion_lines.append("//  marker — REMOVE if not needed by Phase B.)")
    insertion_lines.append("} // namespace detail")
    insertion_lines.append("")
    insertion_lines.append("} // namespace Config")
    insertion_lines.append("} // namespace GRIM")
    insertion_lines.append("")
    insertion_lines.append("#endif // __CUDACC__")
    insertion_lines.append("")

    insertion_text = "\n".join(insertion_lines)

    # ── Modify HyperParameters_GPU.hpp ──────────────────────────────────────
    # 1. Add necessary #includes if missing.
    needed_includes = [
        ("#include <fstream>", "<fstream>"),
        ("#include <filesystem>", "<filesystem>"),
        ("#include <map>", "<map>"),
        ("#include <set>", "<set>"),
        ("#include <vector>", "<vector>"),
        ("#include <nlohmann/json.hpp>", "<nlohmann/json.hpp>"),
    ]
    new_hyper = hyper_text
    # Find a good anchor: the existing #include <utility> line.
    anchor = "#include <utility>"
    if anchor not in new_hyper:
        die(f"could not find anchor '{anchor}' in HyperParameters_GPU.hpp")
    additions: list[str] = []
    for line, key in needed_includes:
        if key not in new_hyper:
            additions.append(line)
    if additions:
        new_hyper = new_hyper.replace(anchor, anchor + "\n" + "\n".join(additions), 1)

    # 2. Remove the existing #include of ai_config_paths.hpp (the cycle).
    cycle_block = (
        "#ifndef __CUDACC__\n"
        "// Regular C++ compilation - include the config header\n"
        "#include \"../../../../../control/ai_config_paths.hpp\"\n"
        "#endif\n"
    )
    if cycle_block in new_hyper:
        new_hyper = new_hyper.replace(cycle_block, "")
    else:
        # Try a more lenient match
        cycle_pat = re.compile(
            r"#ifndef\s+__CUDACC__\s*\n"
            r"//[^\n]*\n"
            r"#include\s+\"\.\./\.\./\.\./\.\./\.\./control/ai_config_paths\.hpp\"\s*\n"
            r"#endif\s*\n",
            re.MULTILINE,
        )
        if cycle_pat.search(new_hyper):
            new_hyper = cycle_pat.sub("", new_hyper)
        else:
            die("could not find ai_config_paths.hpp include cycle to remove")

    # 3. Remove the GRIM_CONFIG_AI_CONFIG_PATHS_HPP_INCLUDED guard pair.
    guard_open = "#ifdef GRIM_CONFIG_AI_CONFIG_PATHS_HPP_INCLUDED"
    guard_close = "#endif // GRIM_CONFIG_AI_CONFIG_PATHS_HPP_INCLUDED"
    if guard_open in new_hyper:
        new_hyper = new_hyper.replace(guard_open + "\n", "")
    if guard_close in new_hyper:
        new_hyper = new_hyper.replace(guard_close + "\n", "")
    if guard_close in new_hyper:
        die("multiple instances of guard close found, manual review needed")

    # 4. Insert the moved code block. We place it BEFORE the existing
    # `validateTrainingHyperparameters` function (which already operates on
    # GRIM::Config::TrainingHyperparameters) so the struct is in scope for it.
    insert_anchor = "//======================================================//\n// (A) Validation primitive."
    if insert_anchor not in new_hyper:
        die(f"could not find insert anchor in HyperParameters_GPU.hpp")
    new_hyper = new_hyper.replace(insert_anchor, insertion_text + insert_anchor, 1)

    # ── Modify ai_config_paths.hpp ──────────────────────────────────────────
    # 1. Remove the GRIM_CONFIG_AI_CONFIG_PATHS_HPP_INCLUDED define marker.
    marker_define = "// Include guard macro for detection by other headers\n#define GRIM_CONFIG_AI_CONFIG_PATHS_HPP_INCLUDED\n"
    if marker_define in new_ai:
        new_ai = new_ai.replace(marker_define, "")
    else:
        # Try just the define line.
        new_ai = re.sub(
            r"//\s*Include guard macro[^\n]*\n#define\s+GRIM_CONFIG_AI_CONFIG_PATHS_HPP_INCLUDED\s*\n",
            "",
            new_ai,
        )

    # 2. Add #include of HyperParameters_GPU.hpp after the system includes.
    # Anchor: the line `#include <vector>`
    ai_anchor = "#include <vector>"
    if ai_anchor not in new_ai:
        die(f"could not find #include <vector> anchor in ai_config_paths.hpp")
    extra_include = (
        ai_anchor
        + "\n\n"
        + "// TrainingHyperparameters and its loaders moved to HyperParameters_GPU.hpp\n"
        + "// (Phase A refactor: HyperParameters_GPU.hpp is the single source of truth.)\n"
        + "#include \"../resources/models/GRIM-text/Shared/HyperParameters/HyperParameters_GPU.hpp\"\n"
    )
    new_ai = new_ai.replace(ai_anchor, extra_include, 1)

    # ── Write files ─────────────────────────────────────────────────────────
    HYPER_HPP.write_text(new_hyper)
    AI_CONFIG.write_text(new_ai)

    print("\nDone.")
    print(f"  ai_config_paths.hpp:  {len(ai_text)} -> {len(new_ai)} bytes")
    print(f"  HyperParameters_GPU.hpp: {len(hyper_text)} -> {len(new_hyper)} bytes")
    print(f"  Moved {len(moved_blocks)} blocks.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
