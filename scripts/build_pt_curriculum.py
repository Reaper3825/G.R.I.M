#!/usr/bin/env python3
"""
Rebuild Pre-Trainingv1 from FineWeb-Edu with stronger quality controls.

This version does not depend on the `datasets` Python package. It streams rows
from Hugging Face's dataset rows HTTP API, filters aggressively for cleaner
English educational prose, and then replaces the current PT curriculum in place.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
import os
import re
import sys
import tempfile
import time
import unicodedata
from collections import Counter
from pathlib import Path
from urllib.parse import urlparse

import requests


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "resources" / "models" / "GRIM-text" / "training" / "data"

CONCEPT_BLOCKS_PATH = DATA_DIR / "concept_blocks.jsonl"
CURRICULUM_REGISTRY_PATH = DATA_DIR / "curriculum_registry.json"
CURRICULUM_NAME = "Pre-Trainingv1"
CURRICULUM_MANIFEST_PATH = DATA_DIR / f"{CURRICULUM_NAME}.json"
CURRICULUM_PROGRESS_PATH = DATA_DIR / f"{CURRICULUM_NAME}.progress.json"

HF_ROWS_API = "https://datasets-server.huggingface.co/rows"
HF_DATASET = "HuggingFaceFW/fineweb-edu"
HF_CONFIG = "sample-10BT"
HF_SPLIT = "train"

DEFAULT_BATCH_SIZE = 100
DEFAULT_REQUEST_TIMEOUT = 30
DEFAULT_RETRIES = 12
DEFAULT_TARGET_COUNT = 10000
DEFAULT_MIN_SCORE = 3.5
DEFAULT_MIN_LANG_SCORE = 0.93
DEFAULT_MIN_TOKENS = 500
DEFAULT_MAX_TOKENS = 2600
DEFAULT_MIN_CHARS = 1800
DEFAULT_MAX_CHARS = 14000
DEFAULT_MAX_SCANNED = 800000
DEFAULT_PROGRESS_EVERY = 1000
DEFAULT_PER_DOMAIN_CAP = 120
DEFAULT_REQUEST_DELAY = 0.35
LOAD_PROGRESS_EVERY_LINES = 50000
RESCAN_PROGRESS_EVERY_BLOCKS = 5000

BOILERPLATE_PATTERNS = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"\bcookie(s)?\b",
        r"\bprivacy policy\b",
        r"\bterms of use\b",
        r"\ball rights reserved\b",
        r"\bnewsletter\b",
        r"\bsubscribe\b",
        r"\bsign up\b",
        r"\blog in\b",
        r"\bskip to (main )?content\b",
        r"\bjavascript (is )?required\b",
        r"\bcomments? (are )?closed\b",
        r"\bclick here\b",
    )
]

SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
WHITESPACE_RE = re.compile(r"[ \t]+")
URL_RE = re.compile(r"https?://|www\.", re.IGNORECASE)
EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b")
DIGIT_RUN_RE = re.compile(r"\d+")
# Mirrors the tokenizer's numeric detector family more closely than a
# word-boundary regex. These matches are allowed to start inside a larger token.
TOKENIZER_NUMERIC_RE = re.compile(
    r"[+-]?(?:(?:\d+\.\d+)|(?:\.\d+)|(?:\d+(?:\.\d+)?[eE][+-]?\d+)|(?:\.\d+[eE][+-]?\d+))"
    r"|[+-]?\d+(?!\.)(?!(?:[eE](?:[+-]|\d)))"
)
I64_MIN = -(2 ** 63)
I64_MAX = 2 ** 63 - 1
# Mirror the tokenizer's float detector more closely than a word-boundary regex.
# The detector can recognize scientific-notation spans starting at any digit
# position, even when they are embedded inside a larger alphanumeric token
# (for example "abc8e467xyz"). Cleanup must reject those too.
SCIENTIFIC_NOTATION_RE = re.compile(
    r"(?:\d+(?:\.\d+)?|\.\d+)[eE][+-]?\d+"
)
ZERO_WIDTH_RE = re.compile(r"[\u200b-\u200f\u2060\ufeff]")
SOFT_HYPHEN_RE = re.compile(r"\u00ad")
HTML_TAG_RE = re.compile(r"</?[A-Za-z][^>\n]{0,200}>")
BROKEN_WORD_HYPHEN_RE = re.compile(r"[A-Za-z]-$")

SHORT_ARTIFACT_PATTERNS = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"^(share|print|email|copy link|download|subscribe|sign up|log in)\b",
        r"^(read more|related articles?|previous|next)\b",
        r"^(home|menu|search|contact us|about us)(\s*(/|>|[|:])\s*.+)+$",
        r"^(facebook|twitter|x|linkedin|reddit|pinterest|whatsapp)(\s*[|/]\s*.+)*$",
    )
]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Rebuild the Pre-Trainingv1 plaintext curriculum from FineWeb-Edu."
    )
    ap.add_argument("--target-count", type=int, default=0,
                    help="How many PT rows to collect. Default 0 = reuse current curriculum size, or 10000 if empty.")
    ap.add_argument("--start-offset", type=int, default=0,
                    help="FineWeb row offset to start scanning from.")
    ap.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE,
                    help="Rows to request per API call.")
    ap.add_argument("--max-scanned", type=int, default=DEFAULT_MAX_SCANNED,
                    help="Abort if this many rows are scanned without reaching target.")
    ap.add_argument("--min-score", type=float, default=DEFAULT_MIN_SCORE)
    ap.add_argument("--min-lang-score", type=float, default=DEFAULT_MIN_LANG_SCORE)
    ap.add_argument("--min-tokens", type=int, default=DEFAULT_MIN_TOKENS)
    ap.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    ap.add_argument("--min-chars", type=int, default=DEFAULT_MIN_CHARS)
    ap.add_argument("--max-chars", type=int, default=DEFAULT_MAX_CHARS)
    ap.add_argument("--max-digit-run", type=int, default=16,
                    help="Reject documents containing any contiguous digit span longer than this many digits.")
    ap.add_argument("--per-domain-cap", type=int, default=DEFAULT_PER_DOMAIN_CAP,
                    help="Maximum accepted documents per domain to avoid overconcentration.")
    ap.add_argument("--progress-every", type=int, default=DEFAULT_PROGRESS_EVERY)
    ap.add_argument("--commit-every", type=int, default=1000,
                    help="Persist a partial curriculum snapshot every N accepted PT rows during long runs.")
    ap.add_argument("--request-delay", type=float, default=DEFAULT_REQUEST_DELAY,
                    help="Sleep this many seconds after each successful API page fetch.")
    ap.add_argument("--clean", action="store_true",
                    help="Rescan existing PT entries with current local text/number filters and replace only the flagged rows.")
    ap.add_argument("--replace-all", action="store_true",
                    help="Discard existing PT rows and rebuild the curriculum from scratch.")
    ap.add_argument("--reset-progress", action="store_true",
                    help="Ignore and delete the saved progress checkpoint before collecting.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Collect and report only; do not rewrite local files.")
    return ap.parse_args()


def atomic_write_text(path: Path, text: str) -> None:
    fd, tmp_path_str = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=str(path.parent))
    os.close(fd)
    tmp_path = Path(tmp_path_str)
    try:
        tmp_path.write_text(text, encoding="utf-8")
        tmp_path.replace(path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise


def load_registry() -> dict:
    with CURRICULUM_REGISTRY_PATH.open("r", encoding="utf-8") as handle:
        registry = json.load(handle)
    if not isinstance(registry, dict) or not isinstance(registry.get("curriculums"), list):
        raise RuntimeError(f"{CURRICULUM_REGISTRY_PATH} is missing a top-level 'curriculums' array")
    return registry


def find_or_create_curriculum(registry: dict) -> dict:
    for curriculum in registry["curriculums"]:
        if curriculum.get("name") == CURRICULUM_NAME:
            return curriculum

    curriculum = {
        "id": f"curr_pt_{hashlib.sha256(CURRICULUM_NAME.encode('utf-8')).hexdigest()[:16]}",
        "name": CURRICULUM_NAME,
        "concept_block_ids": [],
        "plaintext_block_ids": [],
        "format_as_concept": False,
        "timestamp": int(time.time()),
    }
    registry["curriculums"].append(curriculum)
    return curriculum


def get_existing_plaintext_ids(curriculum: dict) -> list[str]:
    plaintext_ids = curriculum.get("plaintext_block_ids")
    if isinstance(plaintext_ids, list):
        return [item for item in plaintext_ids if isinstance(item, str) and item.strip()]

    concept_ids = curriculum.get("concept_block_ids")
    if isinstance(concept_ids, list):
        return [item for item in concept_ids if isinstance(item, str) and item.strip()]

    return []


def load_progress_checkpoint(reset_progress: bool) -> dict:
    if reset_progress:
        CURRICULUM_PROGRESS_PATH.unlink(missing_ok=True)
        return {}
    if not CURRICULUM_PROGRESS_PATH.exists():
        return {}
    try:
        with CURRICULUM_PROGRESS_PATH.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def save_progress_checkpoint(next_offset: int) -> None:
    atomic_write_text(
        CURRICULUM_PROGRESS_PATH,
        json.dumps({"next_offset": next_offset}, indent=2, ensure_ascii=False) + "\n",
    )


def determine_target_count(args: argparse.Namespace, old_ids: list[str]) -> int:
    if args.target_count > 0:
        return args.target_count
    if old_ids:
        return len(old_ids)
    return DEFAULT_TARGET_COUNT


def normalize_whitespace(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [WHITESPACE_RE.sub(" ", line).strip() for line in text.split("\n")]
    lines = [line for line in lines if line]
    return "\n".join(lines).strip()


def normalize_line(line: str) -> str:
    line = html.unescape(line)
    line = unicodedata.normalize("NFKC", line)
    line = ZERO_WIDTH_RE.sub("", line)
    line = SOFT_HYPHEN_RE.sub("", line)
    line = line.replace("\ufffd", "")
    line = HTML_TAG_RE.sub(" ", line)
    return WHITESPACE_RE.sub(" ", line).strip()


def is_short_artifact_line(line: str) -> bool:
    if not line or len(line) > 120:
        return False

    lowered = line.lower()
    if any(pattern.search(line) for pattern in BOILERPLATE_PATTERNS):
        return True
    if any(pattern.search(line) for pattern in SHORT_ARTIFACT_PATTERNS):
        return True
    if lowered.count(" / ") >= 2 or lowered.count(" > ") >= 2 or line.count("|") >= 2:
        return True
    return False


def should_merge_with_previous(previous: str, current: str) -> bool:
    if not previous or not current:
        return False
    if BROKEN_WORD_HYPHEN_RE.search(previous) and current[0].islower():
        return True
    if previous[-1] in ".!?)]}\"":
        return False
    if previous.endswith(":") and current[:1].isupper():
        return False
    if current[:1].islower():
        return True
    if previous.endswith((",", ";")):
        return True
    return False


def strip_repeated_short_artifacts(lines: list[str]) -> list[str]:
    counts = Counter(line.casefold() for line in lines if is_short_artifact_line(line))
    return [
        line for line in lines
        if not (is_short_artifact_line(line) and counts.get(line.casefold(), 0) > 1)
    ]


def clean_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    raw_lines = text.split("\n")

    cleaned_lines: list[str] = []
    previous_blank = False
    for raw_line in raw_lines:
        line = normalize_line(raw_line)
        if not line:
            if not previous_blank:
                cleaned_lines.append("")
            previous_blank = True
            continue

        if is_short_artifact_line(line):
            previous_blank = False
            continue

        cleaned_lines.append(line)
        previous_blank = False

    cleaned_lines = strip_repeated_short_artifacts(cleaned_lines)

    paragraphs: list[str] = []
    current = ""
    for line in cleaned_lines:
        if not line:
            if current:
                paragraphs.append(current)
                current = ""
            continue

        if not current:
            current = line
            continue

        if BROKEN_WORD_HYPHEN_RE.search(current) and line[:1].islower():
            current = current[:-1] + line
        elif should_merge_with_previous(current, line):
            current = f"{current} {line}"
        else:
            paragraphs.append(current)
            current = line

    if current:
        paragraphs.append(current)

    return "\n".join(paragraphs).strip()


def normalize_domain(url: str) -> str:
    try:
        host = urlparse(url).netloc.lower().strip()
    except Exception:
        return ""
    if host.startswith("www."):
        host = host[4:]
    return host


def paragraph_segments(text: str) -> list[str]:
    return [segment.strip() for segment in text.split("\n") if segment.strip()]


def split_plaintext_document(text: str) -> tuple[str, list[str], str]:
    segments = paragraph_segments(text)
    if not segments:
        return "", [], ""
    if len(segments) == 1:
        return "", [], segments[0]
    if len(segments) == 2:
        return segments[0], [], segments[1]
    return segments[0], segments[1:-1], segments[-1]


def build_house_block(*, block_id: str, name: str, question: str,
                      intermediates: list[str], answer: str, timestamp: int,
                      source_sequence_id: str = "") -> dict:
    clean_question = question.strip()
    clean_intermediates = [item.strip() for item in intermediates if item.strip()]
    clean_answer = answer.strip()
    if not clean_answer:
        raise RuntimeError("plaintext PT row requires a non-empty answer segment")

    block = {
        "answer": clean_answer,
        "explanation": clean_intermediates,
        "format_type": "chain_of_thought",
        "id": block_id,
        "intermediate_count": len(clean_intermediates),
        "intermediates": clean_intermediates,
        "name": name,
        "prompt": clean_question,
        "step_index": list(range(len(clean_intermediates))),
        "timestamp": timestamp,
    }
    if source_sequence_id.strip():
        block["source_sequence_id"] = source_sequence_id.strip()
    return block


def sentence_count(text: str) -> int:
    pieces = [piece.strip() for piece in SENTENCE_SPLIT_RE.split(text) if piece.strip()]
    return len(pieces)


def repeated_line_ratio(lines: list[str]) -> float:
    if not lines:
        return 0.0
    counts = Counter(lines)
    duplicates = sum(count - 1 for count in counts.values() if count > 1)
    return duplicates / len(lines)


def char_class_ratios(text: str) -> tuple[float, float, float]:
    total = max(len(text), 1)
    alpha = sum(ch.isalpha() for ch in text) / total
    digits = sum(ch.isdigit() for ch in text) / total
    upper = sum(ch.isupper() for ch in text) / max(sum(ch.isalpha() for ch in text), 1)
    return alpha, digits, upper


def longest_digit_run(text: str) -> int:
    longest = 0
    for match in DIGIT_RUN_RE.finditer(text):
        longest = max(longest, match.end() - match.start())
    return longest


def contains_scientific_notation(text: str) -> bool:
    return SCIENTIFIC_NOTATION_RE.search(text) is not None


def parse_int_atom_ok(raw: str) -> bool:
    try:
        value = int(raw)
    except ValueError:
        return False
    return I64_MIN <= value <= I64_MAX


def parse_float_atom_ok(raw: str) -> bool:
    try:
        value = float(raw)
    except ValueError:
        return False
    return math.isfinite(value)


def first_unparseable_numeric_atom(text: str) -> str | None:
    for match in TOKENIZER_NUMERIC_RE.finditer(text):
        raw = match.group(0)
        is_float = "." in raw or "e" in raw or "E" in raw
        ok = parse_float_atom_ok(raw) if is_float else parse_int_atom_ok(raw)
        if not ok:
            return raw
    return None


def screen_text_content(text: str, args: argparse.Namespace) -> tuple[bool, str]:
    if len(text) < args.min_chars or len(text) > args.max_chars:
        return False, "char_count"

    if args.max_digit_run > 0 and longest_digit_run(text) > args.max_digit_run:
        return False, "long_digit_span"

    if first_unparseable_numeric_atom(text) is not None:
        return False, "unparseable_numeric_atom"

    if contains_scientific_notation(text):
        return False, "scientific_notation"

    lines = text.split("\n")
    if len(lines) < 3:
        return False, "too_few_segments"
    if sentence_count(text) < 8:
        return False, "too_few_sentences"

    alpha_ratio, digit_ratio, upper_ratio = char_class_ratios(text)
    if alpha_ratio < 0.62:
        return False, "low_alpha_ratio"
    if digit_ratio > 0.18:
        return False, "high_digit_ratio"
    if upper_ratio > 0.22:
        return False, "high_upper_ratio"

    if repeated_line_ratio(lines) > 0.12:
        return False, "repeated_lines"

    url_hits = len(URL_RE.findall(text))
    if url_hits > 2:
        return False, "too_many_urls"
    if EMAIL_RE.search(text):
        return False, "email"

    short_lines = sum(1 for line in lines if len(line) < 40)
    if short_lines / len(lines) > 0.45:
        return False, "fragmented_layout"

    for pattern in BOILERPLATE_PATTERNS:
        if pattern.search(text):
            return False, "boilerplate"

    return True, "ok"


def block_plaintext(block: dict) -> str:
    parts: list[str] = []
    question = block.get("prompt", "")
    if isinstance(question, str) and question.strip():
        parts.append(question.strip())

    intermediates = block.get("intermediates", [])
    if not isinstance(intermediates, list):
        intermediates = block.get("explanation", [])
    if isinstance(intermediates, list):
        for item in intermediates:
            if isinstance(item, str) and item.strip():
                parts.append(item.strip())

    answer = block.get("answer", "")
    if isinstance(answer, str) and answer.strip():
        parts.append(answer.strip())

    return "\n".join(parts)


def looks_high_quality(text: str, row: dict, args: argparse.Namespace) -> tuple[bool, str]:
    if row.get("language") != "en":
        return False, "language"
    if float(row.get("language_score") or 0.0) < args.min_lang_score:
        return False, "lang_score"
    if float(row.get("score") or 0.0) < args.min_score:
        return False, "score"

    token_count = int(row.get("token_count") or 0)
    if token_count < args.min_tokens or token_count > args.max_tokens:
        return False, "token_count"
    return screen_text_content(text, args)


def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


def load_existing_pt_blocks(pt_ids: set[str]) -> tuple[list[dict], set[str], set[str], Counter[str]]:
    blocks: list[dict] = []
    normalized_hashes: set[str] = set()
    seen_urls: set[str] = set()
    domain_counts: Counter[str] = Counter()

    if not pt_ids:
        return blocks, normalized_hashes, seen_urls, domain_counts

    invalid_lines = 0
    with CONCEPT_BLOCKS_PATH.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if line_number % LOAD_PROGRESS_EVERY_LINES == 0:
                print(
                    f"Loading current PT rows: scanned {line_number:,} concept block lines, "
                    f"matched {len(blocks):,} PT rows so far...",
                    flush=True,
                )
            stripped = line.strip()
            if not stripped:
                continue
            try:
                row = json.loads(stripped)
            except json.JSONDecodeError as exc:
                invalid_lines += 1
                print(
                    f"WARNING: skipping invalid JSON in {CONCEPT_BLOCKS_PATH} at line "
                    f"{line_number}: {exc}",
                    file=sys.stderr,
                )
                continue

            block_id = row.get("id", "")
            if not isinstance(block_id, str) or block_id not in pt_ids:
                continue

            blocks.append(row)
            if block_id.startswith("cb_pt_") and len(block_id) > len("cb_pt_"):
                normalized_hashes.add(block_id[len("cb_pt_"):])

            source_sequence_id = row.get("source_sequence_id", "")
            if isinstance(source_sequence_id, str) and source_sequence_id:
                seen_urls.add(source_sequence_id)
                domain = normalize_domain(source_sequence_id)
                if domain:
                    domain_counts[domain] += 1

    if invalid_lines:
        print(
            f"WARNING: skipped {invalid_lines} invalid JSONL rows while loading existing PT blocks.",
            file=sys.stderr,
        )

    return blocks, normalized_hashes, seen_urls, domain_counts


def partition_existing_blocks_for_cleanup(
    blocks: list[dict],
    args: argparse.Namespace,
) -> tuple[list[dict], list[tuple[str, str]], set[str], set[str], set[str], Counter[str]]:
    kept_blocks: list[dict] = []
    flagged: list[tuple[str, str]] = []
    kept_hashes: set[str] = set()
    kept_urls: set[str] = set()
    flagged_ids: set[str] = set()
    kept_domain_counts: Counter[str] = Counter()

    total_blocks = len(blocks)
    for index, block in enumerate(blocks, start=1):
        if index % RESCAN_PROGRESS_EVERY_BLOCKS == 0:
            print(
                f"Rescanning current PT rows: checked {index:,}/{total_blocks:,}, "
                f"flagged {len(flagged):,}, kept {len(kept_blocks):,}...",
                flush=True,
            )
        text = block_plaintext(block)
        ok, reason = screen_text_content(text, args)
        block_id = block.get("id", "")
        if not ok:
            if isinstance(block_id, str) and block_id:
                flagged.append((block_id, reason))
                flagged_ids.add(block_id)
            continue

        kept_blocks.append(block)
        if isinstance(block_id, str) and block_id.startswith("cb_pt_") and len(block_id) > len("cb_pt_"):
            kept_hashes.add(block_id[len("cb_pt_"):])

        source_sequence_id = block.get("source_sequence_id", "")
        if isinstance(source_sequence_id, str) and source_sequence_id:
            kept_urls.add(source_sequence_id)
            domain = normalize_domain(source_sequence_id)
            if domain:
                kept_domain_counts[domain] += 1

    return kept_blocks, flagged, flagged_ids, kept_hashes, kept_urls, kept_domain_counts


def fetch_rows(offset: int, length: int, request_delay: float) -> list[dict]:
    params = {
        "dataset": HF_DATASET,
        "config": HF_CONFIG,
        "split": HF_SPLIT,
        "offset": offset,
        "length": length,
    }

    last_error: Exception | None = None
    for attempt in range(DEFAULT_RETRIES):
        try:
            response = requests.get(HF_ROWS_API, params=params, timeout=DEFAULT_REQUEST_TIMEOUT)
            if response.status_code == 429:
                retry_after_header = response.headers.get("Retry-After", "").strip()
                retry_after = 0.0
                if retry_after_header:
                    try:
                        retry_after = float(retry_after_header)
                    except ValueError:
                        retry_after = 0.0
                sleep_for = retry_after if retry_after > 0 else min(15.0 + attempt * 10.0, 120.0)
                print(
                    f"Rate limited at offset={offset}; sleeping {sleep_for:.1f}s "
                    f"before retry {attempt + 1}/{DEFAULT_RETRIES}...",
                    flush=True,
                )
                time.sleep(sleep_for)
                continue
            response.raise_for_status()
            payload = response.json()
            rows = payload.get("rows", [])
            if request_delay > 0:
                time.sleep(request_delay)
            return [row.get("row", {}) for row in rows if isinstance(row, dict)]
        except Exception as exc:
            last_error = exc
            time.sleep(min(2 ** attempt, 8))

    raise RuntimeError(f"Failed to fetch FineWeb rows at offset={offset}: {last_error}")


def collect_blocks(
    *,
    target_count: int,
    args: argparse.Namespace,
    existing_blocks: list[dict],
    existing_hashes: set[str],
    existing_urls: set[str],
    existing_domain_counts: Counter[str],
    start_offset: int,
    on_commit=None,
) -> tuple[list[dict], dict, int]:
    accepted: list[dict] = list(existing_blocks)
    domain_counts: Counter[str] = Counter(existing_domain_counts)
    normalized_hashes: set[str] = set(existing_hashes)
    seen_urls: set[str] = set(existing_urls)
    stats = Counter()
    started = time.time()
    offset = start_offset
    next_commit_at = 0
    if args.commit_every > 0:
        next_commit_at = ((len(accepted) // args.commit_every) + 1) * args.commit_every

    while len(accepted) < target_count and stats["scanned"] < args.max_scanned:
        rows = fetch_rows(offset, args.batch_size, args.request_delay)
        if not rows:
            break
        crossed_commit_boundary = False

        for row in rows:
            stats["scanned"] += 1
            raw_text = row.get("text", "")
            if not isinstance(raw_text, str):
                stats["bad_text_type"] += 1
                continue

            text = clean_text(raw_text)
            if not text:
                stats["empty"] += 1
                continue

            ok, reason = looks_high_quality(text, row, args)
            if not ok:
                stats[reason] += 1
                continue

            normalized_id = content_hash(text)
            if normalized_id in normalized_hashes:
                stats["duplicate_text"] += 1
                continue

            url = row.get("url", "")
            if not isinstance(url, str):
                url = ""
            if url and url in seen_urls:
                stats["duplicate_url"] += 1
                continue

            domain = normalize_domain(url)
            if domain and domain_counts[domain] >= args.per_domain_cap:
                stats["domain_cap"] += 1
                continue

            question, intermediates, answer = split_plaintext_document(text)
            if not answer:
                stats["bad_split"] += 1
                continue

            source_sequence_id = url or str(row.get("id", "")).strip()
            block = build_house_block(
                block_id=f"cb_pt_{normalized_id}",
                name=f"fineweb_edu_{len(accepted):06d}",
                question=question,
                intermediates=intermediates,
                answer=answer,
                timestamp=0,
                source_sequence_id=source_sequence_id,
            )

            accepted.append(block)
            normalized_hashes.add(normalized_id)
            if url:
                seen_urls.add(url)
            if domain:
                domain_counts[domain] += 1

            if next_commit_at > 0 and len(accepted) >= next_commit_at:
                crossed_commit_boundary = True
                while next_commit_at > 0 and len(accepted) >= next_commit_at:
                    next_commit_at += args.commit_every

            if args.progress_every > 0 and len(accepted) % args.progress_every == 0:
                elapsed = time.time() - started
                print(
                    f"Collected {len(accepted)}/{target_count} "
                    f"(scanned={stats['scanned']}, elapsed={elapsed:.1f}s, "
                    f"top skips: score={stats['score']}, token_count={stats['token_count']}, "
                    f"boilerplate={stats['boilerplate']}, domain_cap={stats['domain_cap']})"
                )

            if len(accepted) >= target_count:
                break
            if stats["scanned"] >= args.max_scanned:
                break

        offset += len(rows)
        save_progress_checkpoint(offset)
        if crossed_commit_boundary and on_commit is not None:
            on_commit(accepted, offset)

    return accepted, dict(stats), offset


def rewrite_concept_blocks(old_pt_ids: set[str], new_blocks: list[dict]) -> tuple[int, int]:
    fd, tmp_path_str = tempfile.mkstemp(
        prefix="concept_blocks.",
        suffix=".jsonl.tmp",
        dir=str(DATA_DIR),
    )
    os.close(fd)
    tmp_path = Path(tmp_path_str)

    kept_non_pt = 0
    removed_old_pt = 0
    skipped_invalid = 0

    try:
        with CONCEPT_BLOCKS_PATH.open("r", encoding="utf-8") as src, \
             tmp_path.open("w", encoding="utf-8") as dst:
            for line_number, line in enumerate(src, start=1):
                stripped = line.strip()
                if not stripped:
                    continue

                try:
                    row = json.loads(stripped)
                except json.JSONDecodeError as exc:
                    skipped_invalid += 1
                    print(
                        f"WARNING: dropping invalid JSON in {CONCEPT_BLOCKS_PATH} at line "
                        f"{line_number}: {exc}",
                        file=sys.stderr,
                    )
                    continue

                block_id = row.get("id", "")
                if isinstance(block_id, str) and block_id in old_pt_ids:
                    removed_old_pt += 1
                    continue

                dst.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
                kept_non_pt += 1

            for block in new_blocks:
                dst.write(json.dumps(block, ensure_ascii=False, separators=(",", ":")) + "\n")

        tmp_path.replace(CONCEPT_BLOCKS_PATH)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise

    if skipped_invalid:
        print(
            f"WARNING: dropped {skipped_invalid} invalid JSONL rows while rewriting concept blocks.",
            file=sys.stderr,
        )

    return kept_non_pt, removed_old_pt


def commit_curriculum_snapshot(
    registry: dict,
    curriculum: dict,
    replace_ids: set[str],
    blocks: list[dict],
    next_offset: int,
) -> set[str]:
    plaintext_ids = [block["id"] for block in blocks]
    kept_non_pt, removed_old_pt = rewrite_concept_blocks(replace_ids, blocks)
    write_registry_and_manifest(registry, curriculum, plaintext_ids)
    print(
        f"Committed snapshot: pt_rows={len(blocks)} removed_previous={removed_old_pt} "
        f"kept_non_pt={kept_non_pt} next_offset={next_offset}",
        flush=True,
    )
    return set(plaintext_ids)


def write_registry_and_manifest(registry: dict, curriculum: dict, plaintext_ids: list[str]) -> None:
    curriculum["concept_block_ids"] = []
    curriculum["plaintext_block_ids"] = plaintext_ids
    curriculum["format_as_concept"] = False
    curriculum["timestamp"] = int(time.time())

    atomic_write_text(
        CURRICULUM_REGISTRY_PATH,
        json.dumps(registry, indent=2, ensure_ascii=False) + "\n",
    )
    atomic_write_text(
        CURRICULUM_MANIFEST_PATH,
        json.dumps({
            "concept_block_ids": [],
            "plaintext_block_ids": plaintext_ids,
            "format_as_concept": False,
        }, ensure_ascii=False) + "\n",
    )


def print_sample(block: dict) -> None:
    preview = "\n".join(
        part for part in [
            block.get("prompt", ""),
            *block.get("intermediates", []),
            block.get("answer", ""),
        ]
        if isinstance(part, str) and part
    )
    print("\nSample PT plaintext preview:")
    print(preview[:800])
    if len(preview) > 800:
        print("...")


def print_flagged_entries(flagged: list[tuple[str, str]], limit: int = 25) -> None:
    if not flagged:
        print("  flagged entries   : 0")
        return

    print(f"  flagged entries   : {len(flagged)}")
    print(f"  showing first     : {min(limit, len(flagged))}")
    for block_id, reason in flagged[:limit]:
        print(f"    - {block_id} ({reason})")


def configure_standard_streams() -> None:
    # Keep redirected logs live in PowerShell / file redirection runs.
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        if stream is None or not hasattr(stream, "reconfigure"):
            continue
        try:
            stream.reconfigure(line_buffering=True)
        except Exception:
            pass


def main() -> None:
    configure_standard_streams()
    args = parse_args()

    print(
        f"Starting {CURRICULUM_NAME} rebuild script "
        f"(clean={args.clean}, replace_all={args.replace_all}, dry_run={args.dry_run})",
        flush=True,
    )

    if not CONCEPT_BLOCKS_PATH.exists():
        raise RuntimeError(f"Missing concept block corpus: {CONCEPT_BLOCKS_PATH}")
    if not CURRICULUM_REGISTRY_PATH.exists():
        raise RuntimeError(f"Missing curriculum registry: {CURRICULUM_REGISTRY_PATH}")

    registry = load_registry()
    curriculum = find_or_create_curriculum(registry)
    old_pt_ids = get_existing_plaintext_ids(curriculum)
    progress = load_progress_checkpoint(args.reset_progress)
    target_count = determine_target_count(args, old_pt_ids)
    existing_blocks: list[dict] = []
    existing_hashes: set[str] = set()
    existing_urls: set[str] = set()
    existing_domain_counts: Counter[str] = Counter()
    start_offset = args.start_offset

    if not args.replace_all and old_pt_ids:
        print(
            f"Loading existing PT rows from {CONCEPT_BLOCKS_PATH} "
            f"for {len(old_pt_ids):,} current PT ids...",
            flush=True,
        )
        existing_blocks, existing_hashes, existing_urls, existing_domain_counts = (
            load_existing_pt_blocks(set(old_pt_ids))
        )
        if progress.get("next_offset") is not None:
            try:
                start_offset = max(start_offset, int(progress["next_offset"]))
            except Exception:
                pass

    flagged_existing: list[tuple[str, str]] = []
    if args.clean:
        print(
            f"Rescanning {len(existing_blocks):,} loaded PT rows with current cleanup filters...",
            flush=True,
        )
        original_total = len(existing_blocks)
        (
            existing_blocks,
            flagged_existing,
            flagged_ids,
            existing_hashes,
            existing_urls,
            existing_domain_counts,
        ) = partition_existing_blocks_for_cleanup(existing_blocks, args)
        target_count = original_total
        print(f"Preparing curriculum: {CURRICULUM_NAME}")
        print("  mode              : clean replacement")
        print(f"  current PT rows   : {original_total}")
        print(f"  kept PT rows      : {len(existing_blocks)}")
        print_flagged_entries(flagged_existing)
        print(f"  target count      : {target_count}")
        print(f"  source            : {HF_DATASET}/{HF_CONFIG}:{HF_SPLIT}")
        print(f"  start offset      : {start_offset}")

        if not flagged_existing:
            print("No existing PT entries are flagged by the current filters.")
            return

        if args.dry_run:
            print("Dry run only; local files were not modified.")
            return

        replace_ids_for_commit = set(old_pt_ids)

        def on_clean_commit(blocks: list[dict], next_offset_value: int) -> None:
            nonlocal replace_ids_for_commit
            replace_ids_for_commit = commit_curriculum_snapshot(
                registry,
                curriculum,
                replace_ids_for_commit,
                blocks,
                next_offset_value,
            )

        collected, stats, next_offset = collect_blocks(
            target_count=target_count,
            args=args,
            existing_blocks=existing_blocks,
            existing_hashes=existing_hashes,
            existing_urls=existing_urls,
            existing_domain_counts=existing_domain_counts,
            start_offset=start_offset,
            on_commit=on_clean_commit,
        )
        elapsed = stats and stats.get("scanned", 0)
        print(f"\nCollection finished: accepted={len(collected)} scanned={elapsed} next_offset={next_offset}")

        if len(collected) < target_count:
            raise RuntimeError(
                f"Collected only {len(collected)} PT rows out of target {target_count}. "
                f"Increase --max-scanned or relax the quality gates."
            )

        plaintext_ids = [block["id"] for block in collected]
        kept_non_pt, removed_old_pt = rewrite_concept_blocks(replace_ids_for_commit, collected)
        write_registry_and_manifest(registry, curriculum, plaintext_ids)

        print("\nCurriculum cleanup complete.")
        print(f"  flagged replaced  : {len(flagged_existing)}")
        print(f"  kept non-PT rows  : {kept_non_pt}")
        print(f"  removed old PT    : {removed_old_pt}")
        print(f"  wrote PT rows     : {len(collected)}")
        print(f"  next resume offset: {next_offset}")
        print(f"  registry updated  : {CURRICULUM_REGISTRY_PATH}")
        print(f"  manifest updated  : {CURRICULUM_MANIFEST_PATH}")
        print_sample(collected[0])
        return

    current_count = len(existing_blocks)
    mode = "replace-all rebuild" if args.replace_all else "incremental top-up"
    print(f"Preparing curriculum: {CURRICULUM_NAME}")
    print(f"  mode              : {mode}")
    print(f"  current PT rows   : {current_count}")
    print(f"  target count      : {target_count}")
    print(f"  source            : {HF_DATASET}/{HF_CONFIG}:{HF_SPLIT}")
    print(f"  start offset      : {start_offset}")

    if not args.replace_all and current_count >= target_count:
        print("Existing PT set already meets or exceeds target; nothing to collect.")
        if args.dry_run and existing_blocks:
            print_sample(existing_blocks[0])
        return

    replace_ids_for_commit = set(old_pt_ids)

    def on_commit(blocks: list[dict], next_offset_value: int) -> None:
        nonlocal replace_ids_for_commit
        if args.dry_run:
            return
        replace_ids_for_commit = commit_curriculum_snapshot(
            registry,
            curriculum,
            replace_ids_for_commit,
            blocks,
            next_offset_value,
        )

    collected, stats, next_offset = collect_blocks(
        target_count=target_count,
        args=args,
        existing_blocks=[] if args.replace_all else existing_blocks,
        existing_hashes=set() if args.replace_all else existing_hashes,
        existing_urls=set() if args.replace_all else existing_urls,
        existing_domain_counts=Counter() if args.replace_all else existing_domain_counts,
        start_offset=start_offset,
        on_commit=on_commit,
    )
    elapsed = stats and stats.get("scanned", 0)
    print(f"\nCollection finished: accepted={len(collected)} scanned={elapsed} next_offset={next_offset}")

    if len(collected) < target_count:
        raise RuntimeError(
            f"Collected only {len(collected)} PT rows out of target {target_count}. "
            f"Increase --max-scanned or relax the quality gates."
        )

    if args.dry_run:
        print("Dry run only; local files were not modified.")
        print_sample(collected[0])
        return

    plaintext_ids = [block["id"] for block in collected]
    kept_non_pt, removed_old_pt = rewrite_concept_blocks(replace_ids_for_commit, collected)
    write_registry_and_manifest(registry, curriculum, plaintext_ids)

    print("\nCurriculum update complete.")
    print(f"  kept non-PT rows  : {kept_non_pt}")
    print(f"  removed old PT    : {removed_old_pt}")
    print(f"  wrote new PT rows : {len(collected)}")
    print(f"  next resume offset: {next_offset}")
    print(f"  registry updated  : {CURRICULUM_REGISTRY_PATH}")
    print(f"  manifest updated  : {CURRICULUM_MANIFEST_PATH}")
    print_sample(collected[0])


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        raise SystemExit(130)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
