#!/usr/bin/env python3
"""Inspect how GRIM-text tokenizes command-line text or JSONL records.

This mirrors the current UniByte runtime pipeline:
  1. Detect integer and float atoms in the original UTF-8 bytes.
  2. Normalize ASCII whitespace to the SentencePiece space marker.
  3. Select the highest-scoring learned-piece path with Viterbi.
  4. Emit one byte token for each uncovered UTF-8 byte.

The tokenizer itself does not add BOS, EOS, or padding tokens.

Examples:
    python vocab_playground.py "Hello, GRIM!"
    python vocab_playground.py --text "The answer is 42."
    python vocab_playground.py --jsonl data.jsonl
    python vocab_playground.py --jsonl data.jsonl --field messages.0.content
    python vocab_playground.py --jsonl - --json
"""

import argparse
import json
import math
import struct
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, TextIO


# -- Paths --------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent
VOCAB_BIN = REPO_ROOT / "resources/models/GRIM-text/training/data/vocab.bin"


# -- Token layout -------------------------------------------------------------

NUM_SPECIAL_TOKENS = 4
BYTE_TOKEN_OFFSET = NUM_SPECIAL_TOKENS
BYTE_VOCAB_SIZE = 256
ATOM_TOKEN_OFFSET = BYTE_TOKEN_OFFSET + BYTE_VOCAB_SIZE
ATOM_INT_ID = ATOM_TOKEN_OFFSET
ATOM_FLOAT_ID = ATOM_TOKEN_OFFSET + 1
UNIGRAM_TOKEN_START = ATOM_TOKEN_OFFSET + 2

SPECIAL_NAMES = {0: "<unk>", 1: "<pad>", 2: "<s>", 3: "</s>"}
ATOM_NAMES = {ATOM_INT_ID: "<INT>", ATOM_FLOAT_ID: "<FLOAT>"}
KTMG_VOCAB_VERSION = 4
KTMG_MAX_PIECE_LENGTH = 32
UNKNOWN_SCORE = -100.0
SPACE_MARKER_BYTES = "\u2581".encode("utf-8")


@dataclass(frozen=True)
class VocabPiece:
    token_id: int
    text: str
    encoded: bytes
    score: float


@dataclass(frozen=True)
class Vocab:
    pieces: tuple[VocabPiece, ...]
    by_id: dict[int, VocabPiece]
    token_space_size: int


@dataclass(frozen=True)
class AtomSpan:
    start: int
    end: int
    token_id: int
    text: str


@dataclass(frozen=True)
class TokenUse:
    token_id: int
    token_type: str
    piece: str
    score: float | None
    byte_value: int | None = None
    atom_text: str | None = None


@dataclass(frozen=True)
class TokenizationResult:
    text: str
    tokens: tuple[TokenUse, ...]


@dataclass
class TrieNode:
    children: dict[int, int]
    token_id: int | None = None
    score: float | None = None


def read_exact(stream, size: int, source: str) -> bytes:
    data = stream.read(size)
    if len(data) != size:
        raise EOFError(
            f"truncated read from {source}: expected {size} bytes, got {len(data)}"
        )
    return data


def read_u16(stream, source: str) -> int:
    return struct.unpack("<H", read_exact(stream, 2, source))[0]


def read_u32(stream, source: str) -> int:
    return struct.unpack("<I", read_exact(stream, 4, source))[0]


def read_i32(stream, source: str) -> int:
    return struct.unpack("<i", read_exact(stream, 4, source))[0]


def read_f32(stream, source: str) -> float:
    return struct.unpack("<f", read_exact(stream, 4, source))[0]


def load_vocab_bin(path: Path) -> Vocab:
    """Load and validate a current KTMG v4 vocabulary, including scores."""
    if not path.is_file():
        raise FileNotFoundError(f"vocab.bin not found: {path}")

    pieces: list[VocabPiece] = []
    seen_special_ids: set[int] = set()

    with path.open("rb") as stream:
        source = str(path)
        magic = read_exact(stream, 4, source)
        if magic != b"KTMG":
            raise ValueError(f"Bad vocab.bin magic: {magic!r}; expected KTMG")

        version = read_u16(stream, source)
        if version != KTMG_VOCAB_VERSION:
            raise ValueError(
                f"Unsupported vocab.bin version {version}; "
                f"expected KTMG v{KTMG_VOCAB_VERSION}"
            )

        read_u32(stream, source)  # checksum placeholder
        record_count = read_u32(stream, source)
        max_piece_length = read_u32(stream, source)
        read_exact(stream, 3, source)  # persisted flags
        token_space_size = read_u32(stream, source)

        if max_piece_length != KTMG_MAX_PIECE_LENGTH:
            raise ValueError(
                f"Unexpected KTMG max_piece_length={max_piece_length}; "
                f"expected {KTMG_MAX_PIECE_LENGTH}"
            )
        if record_count < NUM_SPECIAL_TOKENS:
            raise ValueError(
                f"KTMG record_count={record_count} is smaller than the "
                f"{NUM_SPECIAL_TOKENS} required special records"
            )

        for record_index in range(record_count):
            piece_length = read_u32(stream, source)
            if piece_length == 0 or piece_length > max_piece_length:
                raise ValueError(
                    f"KTMG record {record_index} has invalid piece_length={piece_length}"
                )
            encoded = read_exact(stream, piece_length, source)
            text = encoded.decode("utf-8", errors="strict")
            score = read_f32(stream, source)
            token_id = read_i32(stream, source)

            if token_id in SPECIAL_NAMES:
                if text != SPECIAL_NAMES[token_id]:
                    raise ValueError(
                        f"KTMG special record {record_index} has text={text!r} "
                        f"for token_id={token_id}; expected {SPECIAL_NAMES[token_id]!r}"
                    )
                if token_id in seen_special_ids:
                    raise ValueError(f"Duplicate KTMG special token_id={token_id}")
                seen_special_ids.add(token_id)
                continue

            expected_id = UNIGRAM_TOKEN_START + len(pieces)
            if token_id != expected_id:
                raise ValueError(
                    f"KTMG learned-piece ID mismatch at record {record_index}: "
                    f"stored={token_id} expected={expected_id}"
                )
            if not math.isfinite(score):
                raise ValueError(
                    f"KTMG record {record_index} has non-finite score={score}"
                )
            pieces.append(VocabPiece(token_id, text, encoded, score))

        trailing = stream.read(1)
        if trailing:
            raise ValueError(f"KTMG file has trailing bytes after {record_count} records")

    if seen_special_ids != set(SPECIAL_NAMES):
        raise ValueError(
            f"KTMG special metadata mismatch: seen={sorted(seen_special_ids)} "
            f"expected={sorted(SPECIAL_NAMES)}"
        )

    expected_size = UNIGRAM_TOKEN_START + len(pieces)
    if token_space_size != expected_size:
        raise ValueError(
            f"KTMG token-space mismatch: header={token_space_size} "
            f"computed={expected_size}"
        )

    return Vocab(tuple(pieces), {piece.token_id: piece for piece in pieces}, token_space_size)


def build_trie(vocab: Vocab) -> list[TrieNode]:
    trie = [TrieNode({})]
    for piece in vocab.pieces:
        node_index = 0
        for byte_value in piece.encoded:
            next_index = trie[node_index].children.get(byte_value)
            if next_index is None:
                next_index = len(trie)
                trie[node_index].children[byte_value] = next_index
                trie.append(TrieNode({}))
            node_index = next_index

        node = trie[node_index]
        if node.token_id is not None:
            raise ValueError(
                f"duplicate learned piece text {piece.text!r}: "
                f"token_ids={node.token_id},{piece.token_id}"
            )
        node.token_id = piece.token_id
        node.score = piece.score
    return trie


def float32(value: float) -> float:
    """Round an operation result to CUDA/C++ float precision."""
    return struct.unpack("<f", struct.pack("<f", value))[0]


def should_replace(
    candidate_score: float,
    candidate_start: int,
    candidate_token_id: int,
    candidate_fallback: bool,
    current_score: float,
    current_start: int,
    current_token_id: int,
    current_fallback: bool,
    end: int,
) -> bool:
    if candidate_score != current_score:
        return candidate_score > current_score
    if current_start < 0:
        return True
    if candidate_fallback != current_fallback:
        return not candidate_fallback
    candidate_span = end - candidate_start
    current_span = end - current_start
    if candidate_span != current_span:
        return candidate_span > current_span
    return candidate_token_id < current_token_id


def normalize_spaces(text: bytes, prepend_space: bool) -> bytes:
    output = bytearray(SPACE_MARKER_BYTES if prepend_space else b"")
    index = 0
    while index < len(text):
        if text[index:index + 2] == b"\r\n":
            output.extend(SPACE_MARKER_BYTES)
            index += 2
        elif text[index] in b" \t\n\r":
            output.extend(SPACE_MARKER_BYTES)
            index += 1
        else:
            output.append(text[index])
            index += 1
    return bytes(output)


def sign_is_operator_context(text: bytes, position: int) -> bool:
    if position == 0:
        return False
    previous = text[position - 1]
    return (
        ord("0") <= previous <= ord("9")
        or ord("a") <= previous <= ord("z")
        or ord("A") <= previous <= ord("Z")
        or previous in b".)]"
    )


def detect_integer_end(text: bytes, position: int) -> int | None:
    if position >= len(text):
        return None
    index = position
    if text[index] in b"+-":
        if sign_is_operator_context(text, index):
            return None
        if index + 1 >= len(text) or not ord("0") <= text[index + 1] <= ord("9"):
            return None
        index += 1
    if not ord("0") <= text[index] <= ord("9"):
        return None
    while index < len(text) and ord("0") <= text[index] <= ord("9"):
        index += 1
    if (
        index + 1 < len(text)
        and text[index] == ord(".")
        and ord("0") <= text[index + 1] <= ord("9")
    ):
        return None
    if index < len(text) and text[index] in b"eE" and index + 1 < len(text):
        if text[index + 1] in b"+-" or ord("0") <= text[index + 1] <= ord("9"):
            return None
    return index


def detect_float_end(text: bytes, position: int) -> int | None:
    if position >= len(text):
        return None
    index = position
    has_dot = False
    has_exponent = False
    has_digit = False
    if text[index] in b"+-":
        if sign_is_operator_context(text, index):
            return None
        index += 1
    while index < len(text) and ord("0") <= text[index] <= ord("9"):
        has_digit = True
        index += 1
    if (
        index + 1 < len(text)
        and text[index] == ord(".")
        and ord("0") <= text[index + 1] <= ord("9")
    ):
        has_dot = True
        index += 1
        while index < len(text) and ord("0") <= text[index] <= ord("9"):
            has_digit = True
            index += 1
    if index < len(text) and text[index] in b"eE":
        has_exponent = True
        index += 1
        if index < len(text) and text[index] in b"+-":
            index += 1
        exponent_start = index
        while index < len(text) and ord("0") <= text[index] <= ord("9"):
            index += 1
        if index == exponent_start:
            return None
    if not has_digit or (not has_dot and not has_exponent):
        return None
    return index


def detect_atom_spans(text: bytes) -> list[AtomSpan]:
    spans: list[AtomSpan] = []
    position = 0
    while position < len(text):
        float_end = detect_float_end(text, position)
        integer_end = detect_integer_end(text, position)
        if float_end is not None and (
            integer_end is None or float_end >= integer_end
        ):
            token_id = ATOM_FLOAT_ID
            end = float_end
        elif integer_end is not None:
            token_id = ATOM_INT_ID
            end = integer_end
        else:
            position += 1
            continue

        raw = text[position:end].decode("utf-8", errors="strict")
        spans.append(AtomSpan(position, end, token_id, raw))
        position = end
    return spans


def viterbi_segment(
    normalized: bytes,
    vocab: Vocab,
    trie: list[TrieNode],
) -> list[TokenUse]:
    if not normalized:
        return []

    length = len(normalized)
    scores = [float32(-1.0e30)] * (length + 1)
    previous = [-1] * (length + 1)
    token_ids = [0] * (length + 1)
    fallbacks = [False] * (length + 1)
    scores[0] = 0.0

    for position in range(length):
        if position != 0 and previous[position] < 0:
            continue
        node_index = 0
        for piece_length in range(1, KTMG_MAX_PIECE_LENGTH + 1):
            end = position + piece_length
            if end > length:
                break
            next_index = trie[node_index].children.get(normalized[end - 1])
            if next_index is None:
                break
            node_index = next_index
            node = trie[node_index]
            if node.token_id is None:
                continue
            if node.score is None:
                raise RuntimeError(f"trie token_id={node.token_id} has no score")
            candidate_score = float32(scores[position] + node.score)
            if should_replace(
                candidate_score,
                position,
                node.token_id,
                False,
                scores[end],
                previous[end],
                token_ids[end],
                fallbacks[end],
                end,
            ):
                scores[end] = candidate_score
                previous[end] = position
                token_ids[end] = node.token_id
                fallbacks[end] = False

        end = position + 1
        fallback_id = BYTE_TOKEN_OFFSET + normalized[position]
        fallback_score = float32(scores[position] + UNKNOWN_SCORE)
        if should_replace(
            fallback_score,
            position,
            fallback_id,
            True,
            scores[end],
            previous[end],
            token_ids[end],
            fallbacks[end],
            end,
        ):
            scores[end] = fallback_score
            previous[end] = position
            token_ids[end] = fallback_id
            fallbacks[end] = True

    if previous[length] < 0:
        raise RuntimeError("Viterbi failed to reach the end of normalized input")

    reversed_tokens: list[TokenUse] = []
    position = length
    while position > 0:
        start = previous[position]
        token_id = token_ids[position]
        if start < 0 or start >= position:
            raise RuntimeError(
                f"Viterbi has invalid backpointer start={start} end={position}"
            )
        if fallbacks[position]:
            byte_value = token_id - BYTE_TOKEN_OFFSET
            reversed_tokens.append(
                TokenUse(
                    token_id,
                    "BYTE",
                    bytes([byte_value]).decode("latin-1"),
                    UNKNOWN_SCORE,
                    byte_value=byte_value,
                )
            )
        else:
            piece = vocab.by_id.get(token_id)
            if piece is None:
                raise RuntimeError(f"Viterbi emitted unknown learned token_id={token_id}")
            reversed_tokens.append(
                TokenUse(token_id, "UNIGRAM", piece.text, piece.score)
            )
        position = start

    reversed_tokens.reverse()
    return reversed_tokens


def tokenize(
    text: str,
    vocab: Vocab,
    trie: list[TrieNode],
    detect_numbers: bool,
) -> TokenizationResult:
    source = text.encode("utf-8", errors="strict")
    if not source:
        return TokenizationResult(text, ())

    atom_spans = detect_atom_spans(source) if detect_numbers else []
    tokens: list[TokenUse] = []
    position = 0

    for atom in atom_spans:
        if atom.start > position:
            segment = source[position:atom.start]
            normalized = normalize_spaces(
                segment,
                prepend_space=(position == 0 and not tokens),
            )
            tokens.extend(viterbi_segment(normalized, vocab, trie))
        tokens.append(
            TokenUse(
                atom.token_id,
                "ATOM",
                ATOM_NAMES[atom.token_id],
                None,
                atom_text=atom.text,
            )
        )
        position = atom.end

    if position < len(source):
        normalized = normalize_spaces(
            source[position:],
            prepend_space=(position == 0 and not tokens),
        )
        tokens.extend(viterbi_segment(normalized, vocab, trie))

    return TokenizationResult(text, tuple(tokens))


def escape_display(text: str) -> str:
    return json.dumps(text, ensure_ascii=True)


def token_to_dict(index: int, token: TokenUse) -> dict:
    result = {
        "index": index,
        "id": token.token_id,
        "type": token.token_type,
        "piece": token.piece,
        "score": token.score,
    }
    if token.byte_value is not None:
        result["byte_hex"] = f"0x{token.byte_value:02X}"
    if token.atom_text is not None:
        result["atom_text"] = token.atom_text
    return result


def result_to_dict(label: str, result: TokenizationResult) -> dict:
    counts = Counter(token.token_type for token in result.tokens)
    return {
        "source": label,
        "text": result.text,
        "characters": len(result.text),
        "utf8_bytes": len(result.text.encode("utf-8")),
        "token_count": len(result.tokens),
        "type_counts": {
            "special": counts["SPECIAL"],
            "byte": counts["BYTE"],
            "atom": counts["ATOM"],
            "unigram": counts["UNIGRAM"],
        },
        "token_ids": [token.token_id for token in result.tokens],
        "tokens": [token_to_dict(index, token) for index, token in enumerate(result.tokens)],
    }


def print_text_report(label: str, result: TokenizationResult) -> None:
    report = result_to_dict(label, result)
    counts = report["type_counts"]
    print("=" * 80)
    print(f"Source       : {label}")
    print(f"Text         : {escape_display(result.text)}")
    print(f"Characters   : {report['characters']}")
    print(f"UTF-8 bytes  : {report['utf8_bytes']}")
    print(f"Tokens       : {report['token_count']}")
    print(
        "Token types  : "
        f"unigram={counts['unigram']} byte={counts['byte']} atom={counts['atom']}"
    )
    print(f"Token IDs    : {report['token_ids']}")
    print()
    print(f"{'INDEX':>5}  {'ID':>6}  {'TYPE':>7}  {'SCORE':>12}  TOKEN")
    print(f"{'-' * 5}  {'-' * 6}  {'-' * 7}  {'-' * 12}  {'-' * 30}")
    for index, token in enumerate(result.tokens):
        score = "-" if token.score is None else f"{token.score:.7g}"
        display = token.atom_text if token.atom_text is not None else token.piece
        if token.byte_value is not None:
            display = f"{escape_display(display)} (0x{token.byte_value:02X})"
        else:
            display = escape_display(display)
        print(
            f"{index:>5}  {token.token_id:>6}  {token.token_type:>7}  "
            f"{score:>12}  {display}"
        )
    print()


def extract_field(value, field: str, source: str) -> str:
    if isinstance(value, str):
        return value
    current = value
    for component in field.split("."):
        if isinstance(current, dict):
            if component not in current:
                raise ValueError(f"{source}: JSON field {field!r} does not exist")
            current = current[component]
        elif isinstance(current, list):
            try:
                index = int(component)
            except ValueError as exc:
                raise ValueError(
                    f"{source}: {component!r} is not a list index in field {field!r}"
                ) from exc
            if index < 0 or index >= len(current):
                raise ValueError(
                    f"{source}: list index {index} is out of range in field {field!r}"
                )
            current = current[index]
        else:
            raise ValueError(
                f"{source}: field {field!r} traverses non-container value {current!r}"
            )
    if not isinstance(current, str):
        raise ValueError(
            f"{source}: JSON field {field!r} must contain a string, "
            f"got {type(current).__name__}"
        )
    return current


def iter_jsonl(stream: TextIO, source_name: str, field: str) -> Iterator[tuple[str, str]]:
    for line_number, line in enumerate(stream, start=1):
        source = f"{source_name}:{line_number}"
        if not line.strip():
            raise ValueError(f"{source}: blank JSONL records are not allowed")
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{source}: invalid JSON: {exc.msg}") from exc
        yield source, extract_field(value, field, source)


def resolve_inputs(args) -> Iterator[tuple[str, str]]:
    direct_values = [value for value in (args.input_text, args.text) if value is not None]
    if args.jsonl is not None:
        if direct_values:
            raise ValueError("use either command-line text or --jsonl, not both")
        if args.jsonl == "-":
            yield from iter_jsonl(sys.stdin, "<stdin>", args.field)
            return
        path = Path(args.jsonl)
        if not path.is_file():
            raise FileNotFoundError(f"JSONL input not found: {path}")
        with path.open("r", encoding="utf-8", errors="strict", newline="") as stream:
            yield from iter_jsonl(stream, str(path), args.field)
        return
    if len(direct_values) != 1:
        raise ValueError("provide one positional text, --text TEXT, or --jsonl PATH")
    yield "command line", direct_values[0]


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Show exactly which GRIM-text tokens a text sequence uses"
    )
    parser.add_argument("input_text", nargs="?", help="Text to tokenize (quote spaces)")
    parser.add_argument("--text", help="Text to tokenize")
    parser.add_argument("--jsonl", help="JSONL path, or - to read JSONL from stdin")
    parser.add_argument(
        "--field",
        default="text",
        help="Dot-separated JSONL string field (default: text); JSON strings need no field",
    )
    parser.add_argument("--vocab", default=str(VOCAB_BIN), help="Path to KTMG v4 vocab.bin")
    parser.add_argument(
        "--no-atoms",
        action="store_true",
        help="Disable numeric atom detection and tokenize numbers as ordinary text",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Write one machine-readable JSON report per input record",
    )
    return parser


def main() -> int:
    parser = make_parser()
    args = parser.parse_args()
    try:
        vocab = load_vocab_bin(Path(args.vocab))
        trie = build_trie(vocab)
        emitted = 0
        for label, text in resolve_inputs(args):
            result = tokenize(text, vocab, trie, detect_numbers=not args.no_atoms)
            if args.json:
                print(json.dumps(result_to_dict(label, result), ensure_ascii=False))
            else:
                print_text_report(label, result)
            emitted += 1
        if emitted == 0:
            raise ValueError("input contained zero JSONL records")
    except (EOFError, FileNotFoundError, UnicodeError, ValueError, RuntimeError) as exc:
        parser.exit(2, f"error: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())