#!/usr/bin/env python3
"""
Decode token IDs from training_data.grmt using the current KTMG vocab.bin.

Token ID Layout (GrimTokenizer):
  [0-3]       = Special tokens: <unk>=0, <pad>=1, <s>=2, </s>=3
  [4-259]     = Byte fallback (byte value = token_id - 4)
    [260-262]   = Atom placeholders: <ATOM_NONE>, <INT>, <FLOAT>
    [263+]      = Unigram vocabulary pieces (from vocab.bin)

Current vocab.bin format is KTMG v4. The saved record count is the number of
serialized records (4 special-token metadata records + learned unigram pieces),
not the full token-space size. The token-space size is stored separately in the
header and must equal special + bytes + atoms + learned pieces.

Current training_data.grmt format is GRMT v12. Rows include atom side-channel
text and execution metadata after the token arrays, so this script reads/skips
the full row to keep the stream synchronized.

Usage:
    python decode_token_ids.py
    python decode_token_ids.py --seq 0 5          # Sequences 0 through 4
    python decode_token_ids.py --ids 277 512 36   # Decode specific token IDs
    python decode_token_ids.py --search hello      # Find sequences containing text
    python decode_token_ids.py --raw               # Show raw token IDs alongside text
    python decode_token_ids.py --stats             # Vocabulary and data statistics
"""

import struct
import argparse
import sys
from pathlib import Path
from collections import Counter

# ── Paths (repo-relative defaults) ───────────────────────────────────────────

REPO_ROOT = Path(__file__).resolve().parent
VOCAB_BIN = REPO_ROOT / "resources/models/GRIM-text/training/data/vocab.bin"
GRMT_FILE = REPO_ROOT / "resources/models/GRIM-text/training/data/training_data.grmt"

# ── Token layout constants ────────────────────────────────────────────────────

NUM_SPECIAL_TOKENS = 4
BYTE_TOKEN_OFFSET  = NUM_SPECIAL_TOKENS   # 4
BYTE_VOCAB_SIZE    = 256
ATOM_TOKEN_START   = BYTE_TOKEN_OFFSET + BYTE_VOCAB_SIZE  # 260

SPECIAL_NAMES = {0: "<unk>", 1: "<pad>", 2: "<s>", 3: "</s>"}

ATOM_TYPE_LABELS = {
    0:  "<ATOM_NONE>",
    1:  "<INT>",
    2:  "<FLOAT>",
}
NUM_ATOM_TYPES = 3  # AtomType::ATOM_ACTIVE_COUNT: NONE, INT, FLOAT

ATOM_TOKEN_END        = ATOM_TOKEN_START + NUM_ATOM_TYPES  # 263
UNIGRAM_TOKEN_START   = ATOM_TOKEN_END                     # 263
KTMG_VOCAB_VERSION    = 4
KTMG_MAX_PIECE_LENGTH = 32
GRMT_MAGIC            = 0x474D5254
GRMT_FORMAT_VERSION   = 12


# ── Binary helpers ───────────────────────────────────────────────────────────

def read_exact(f, size: int, source: str) -> bytes:
    data = f.read(size)
    if len(data) != size:
        raise EOFError(f"truncated read from {source}: expected {size} bytes, got {len(data)}")
    return data


def read_u8(f, source: str) -> int:
    return struct.unpack("<B", read_exact(f, 1, source))[0]


def read_u16(f, source: str) -> int:
    return struct.unpack("<H", read_exact(f, 2, source))[0]


def read_u32(f, source: str) -> int:
    return struct.unpack("<I", read_exact(f, 4, source))[0]


def read_i32(f, source: str) -> int:
    return struct.unpack("<i", read_exact(f, 4, source))[0]


def read_f32(f, source: str) -> float:
    return struct.unpack("<f", read_exact(f, 4, source))[0]


def read_i32_array(f, count: int, source: str) -> list:
    if count == 0:
        return []
    return list(struct.unpack(f"<{count}i", read_exact(f, 4 * count, source)))


def skip_exact(f, size: int, source: str):
    read_exact(f, size, source)


# ── vocab.bin loader (KTMG v4 format) ────────────────────────────────────────

def load_vocab_bin(path: Path) -> dict:
    # sourcery skip: extract-method
    """
    Load vocab.bin and build a complete token_id -> text mapping.

    Returns dict mapping every known token ID to its string representation.
    """
    if not path.exists():
        raise FileNotFoundError(f"vocab.bin not found: {path}")

    id_to_text = dict(SPECIAL_NAMES)

    # 2) Byte fallback tokens — emit the raw byte (matching C++ decode behavior).
    #    The C++ tokenizer does: result.push_back(static_cast<char>(tid - BYTE_TOKEN_OFFSET))
    #    so every byte token maps to its literal byte value.
    for b in range(BYTE_VOCAB_SIZE):
        tid = BYTE_TOKEN_OFFSET + b
        id_to_text[tid] = bytes([b]).decode("latin-1")  # 1:1 byte→char mapping

    # 3) Atom placeholder tokens
    for i in range(NUM_ATOM_TYPES):
        tid = ATOM_TOKEN_START + i
        id_to_text[tid] = ATOM_TYPE_LABELS.get(i, f"<ATOM{i}>")

    # 4) Special-token metadata + unigram pieces from vocab.bin.
    #    KTMG v4 persists special-token records plus learned pieces. It does
    #    NOT persist byte/atom records; those remain fixed by TokenLayout.hpp.
    with open(path, "rb") as f:
        source = str(path)
        magic = read_exact(f, 4, source)
        if magic != b"KTMG":
            raise ValueError(f"Bad vocab.bin magic: {magic!r}; expected KTMG")

        version = read_u16(f, source)
        if version != KTMG_VOCAB_VERSION:
            raise ValueError(
                f"Unsupported vocab.bin version {version}; expected KTMG v{KTMG_VOCAB_VERSION}. "
                f"Retrain/pull the current tokenizer artifact."
            )

        _checksum = read_u32(f, source)
        serialized_record_count = read_u32(f, source)
        max_length = read_u32(f, source)
        _flags = read_exact(f, 3, source)
        token_space_size = read_u32(f, source)

        if max_length != KTMG_MAX_PIECE_LENGTH:
            raise ValueError(f"Unexpected KTMG max_piece_length={max_length}; expected {KTMG_MAX_PIECE_LENGTH}")
        if serialized_record_count < NUM_SPECIAL_TOKENS:
            raise ValueError(
                f"KTMG serialized_record_count={serialized_record_count} is smaller than "
                f"NUM_SPECIAL_TOKENS={NUM_SPECIAL_TOKENS}"
            )

        seen_special_ids = set()
        learned_piece_count = 0
        for record_idx in range(serialized_record_count):
            piece_len = read_u32(f, source)
            if piece_len > max_length:
                raise ValueError(
                    f"KTMG record {record_idx} has piece_len={piece_len}, max_length={max_length}"
                )
            text = read_exact(f, piece_len, source).decode("utf-8", errors="strict")
            _score = read_f32(f, source)
            token_id = read_i32(f, source)

            if token_id in SPECIAL_NAMES:
                expected_text = SPECIAL_NAMES[token_id]
                if text != expected_text:
                    raise ValueError(
                        f"KTMG special record mismatch at record {record_idx}: "
                        f"token_id={token_id} text={text!r} expected={expected_text!r}"
                    )
                if token_id in seen_special_ids:
                    raise ValueError(f"KTMG duplicate special token record for token_id={token_id}")
                seen_special_ids.add(token_id)
                continue

            expected_id = UNIGRAM_TOKEN_START + learned_piece_count
            if token_id != expected_id:
                raise ValueError(
                    f"KTMG learned-piece token_id mismatch at record {record_idx}: "
                    f"stored={token_id} expected={expected_id}. Do not patch the vocab header; retrain/pull it."
                )

            id_to_text[token_id] = text
            learned_piece_count += 1

        if seen_special_ids != set(SPECIAL_NAMES):
            raise ValueError(
                f"KTMG special metadata set mismatch: seen={sorted(seen_special_ids)} "
                f"expected={sorted(SPECIAL_NAMES)}"
            )

        expected_token_space_size = UNIGRAM_TOKEN_START + learned_piece_count
        if token_space_size != expected_token_space_size:
            raise ValueError(
                f"KTMG token-space size mismatch: header={token_space_size} "
                f"computed={expected_token_space_size} "
                f"({NUM_SPECIAL_TOKENS} special + {BYTE_VOCAB_SIZE} bytes + "
                f"{NUM_ATOM_TYPES} atoms + {learned_piece_count} unigrams)"
            )

    return id_to_text


# ── training_data.grmt loader ────────────────────────────────────────────────

def read_grmt_header(path: Path) -> dict:
    """Read and validate the 16-byte current GRMT header."""
    with open(path, "rb") as f:
        magic, version, num_sequences, vocab_size = struct.unpack("<IIII", read_exact(f, 16, str(path)))

    if magic != GRMT_MAGIC:
        raise ValueError(f"Bad GRMT magic in {path}: actual=0x{magic:08X} expected=0x{GRMT_MAGIC:08X}")
    if version != GRMT_FORMAT_VERSION:
        raise ValueError(f"Unsupported GRMT version {version}; expected {GRMT_FORMAT_VERSION}")
    if num_sequences == 0:
        raise ValueError(f"GRMT header reports num_sequences=0: {path}")
    if vocab_size == 0:
        raise ValueError(f"GRMT header reports vocab_size=0: {path}")

    return {
        "magic": magic,
        "magic_hex": hex(magic),
        "version": version,
        "num_sequences": num_sequences,
        "vocab_size": vocab_size,
    }


def iter_grmt_sequences(path: Path):
    """Yield (index, token_id_list, atom_text_list) for each sequence in the GRMT file.

    GRMT v12 per-sequence layout (must read ALL fields to stay in sync):
      uint32         seq_len
      int32[seq_len] token_ids
      int32[seq_len] targets
      float[seq_len] numeric_values
      uint8[seq_len] atom_mask
      uint32[seq_len] atom_flags
      repeated seq_len times: uint16 atom_text_len + atom_text bytes
      uint8 execution_active
      int32[seq_len] token_exec_slots
      uint32 compiled_bootstrap_binding_count, then 12 bytes each
      uint32 teacher_step_count, then 20 bytes each
      uint32 slot_selection_target_count, then {uint8 kind, int32 slot_id} each
    """
    with open(path, "rb") as f:
        source = str(path)
        header = read_grmt_header(path)
        f.seek(16)
        num_sequences = header["num_sequences"]

        for idx in range(num_sequences):
            row_source = f"{source}#seq{idx}"
            seq_len = read_u32(f, row_source)
            if seq_len == 0:
                raise ValueError(f"GRMT sequence length is zero in {row_source}")
            if seq_len > 1_000_000:
                raise ValueError(f"GRMT sequence length is suspiciously large in {row_source}: {seq_len}")

            tokens = read_i32_array(f, seq_len, row_source)

            skip_exact(f, 4 * seq_len, row_source)       # targets (int32)
            skip_exact(f, 4 * seq_len, row_source)       # token_numeric_values (float32)
            skip_exact(f, 1 * seq_len, row_source)       # token_atom_mask (uint8)
            skip_exact(f, 4 * seq_len, row_source)       # token_atom_flags (uint32)

            atom_texts = []
            for token_index in range(seq_len):
                atom_text_len = read_u16(f, row_source)
                if atom_text_len > 0:
                    atom_text = read_exact(f, atom_text_len, row_source).decode("utf-8", errors="strict")
                else:
                    atom_text = ""
                atom_texts.append(atom_text)

                token_id = tokens[token_index]
                if atom_text and not (ATOM_TOKEN_START <= token_id < ATOM_TOKEN_END):
                    raise ValueError(
                        f"GRMT atom text exists for non-atom token in {row_source} "
                        f"index={token_index} token_id={token_id}"
                    )

            _execution_active = read_u8(f, row_source)
            skip_exact(f, 4 * seq_len, row_source)       # token_exec_slots (int32)

            cbb_count = read_u32(f, row_source)
            skip_exact(f, 12 * cbb_count, row_source)    # CompiledBootstrapBinding

            teacher_step_count = read_u32(f, row_source)
            skip_exact(f, 20 * teacher_step_count, row_source)  # TeacherStep

            slot_selection_target_count = read_u32(f, row_source)
            for _ in range(slot_selection_target_count):
                skip_exact(f, 1, row_source)             # SlotSelectionTargetKind uint8
                skip_exact(f, 4, row_source)             # slot_id int32

            yield idx, tokens, atom_texts


# ── Decode helpers ────────────────────────────────────────────────────────────

def decode_token(tid: int, vocab: dict) -> str:
    """Decode a single token ID to its text representation."""
    return vocab.get(tid, f"<UNK:{tid}>")


def decode_sequence(tokens: list, vocab: dict, atom_texts: list | None = None) -> str:
    """Decode a full sequence of token IDs to readable text.
    
    Merges consecutive byte-fallback tokens into actual UTF-8 characters
    instead of showing <BYTE 0xNN> tags.
    """
    # Two-pass: first collect raw pieces, merging byte runs into UTF-8
    output_parts = []
    byte_buffer = bytearray()
    
    def flush_bytes():
        nonlocal byte_buffer
        if byte_buffer:
            # Decode accumulated bytes as UTF-8
            output_parts.append(byte_buffer.decode("utf-8", errors="replace"))
            byte_buffer = bytearray()
    
    for token_index, tid in enumerate(tokens):
        # Check if this is a byte-fallback token (IDs 4-259)
        if BYTE_TOKEN_OFFSET <= tid < ATOM_TOKEN_START:
            byte_val = tid - BYTE_TOKEN_OFFSET
            if byte_val >= 0x80:  # Non-ASCII byte → accumulate for UTF-8
                byte_buffer.append(byte_val)
                continue
            else:
                flush_bytes()
                output_parts.append(chr(byte_val))
                continue
        
        flush_bytes()
        if ATOM_TOKEN_START <= tid < ATOM_TOKEN_END:
            atom_text = ""
            if atom_texts is not None and token_index < len(atom_texts):
                atom_text = atom_texts[token_index]
            output_parts.append(atom_text or vocab[tid])
            continue

        if tid in vocab:
            output_parts.append(vocab[tid])
        else:
            output_parts.append(f"<UNK:{tid}>")
    
    flush_bytes()
    text = "".join(output_parts)
    text = text.replace("\u2581", " ")  # ▁ (Unigram space marker) → space
    return text


def token_type_label(tid: int) -> str:
    """Return which region a token ID belongs to."""
    if tid < NUM_SPECIAL_TOKENS:
        return "SPECIAL"
    elif BYTE_TOKEN_OFFSET <= tid < ATOM_TOKEN_START:
        return "BYTE"
    elif ATOM_TOKEN_START <= tid < ATOM_TOKEN_END:
        return "ATOM"
    return "UNIGRAM"


def validate_grmt_vocab_pair(header: dict, vocab: dict, grmt: Path):
    """Fail loudly if the GRMT header and loaded vocab disagree on token-space size."""
    if header["vocab_size"] != len(vocab):
        raise ValueError(
            f"GRMT/vocab token-space mismatch for {grmt}: "
            f"grmt_header={header['vocab_size']} loaded_vocab={len(vocab)}. "
            f"Pull/rebuild vocab.bin and training_data.grmt as one artifact pair."
        )


def parse_cli_token_ids(raw_values: list[str]) -> list[int]:
    """Parse --ids values, accepting either whitespace- or comma-separated integers."""
    parsed_ids = []

    for raw_value in raw_values:
        for piece in raw_value.split(","):
            piece = piece.strip()
            if not piece:
                continue
            try:
                parsed_ids.append(int(piece))
            except ValueError as exc:
                raise ValueError(f"Invalid token ID {piece!r} in --ids; expected integers") from exc

    if not parsed_ids:
        raise ValueError("--ids requires at least one integer token ID")

    return parsed_ids


# ── CLI actions ───────────────────────────────────────────────────────────────

def cmd_decode_ids(args, vocab):
    """Decode a list of token IDs from the command line."""
    for tid in args.ids:
        text = decode_token(tid, vocab)
        region = token_type_label(tid)
        print(f"  {tid:>6d}  [{region:>7s}]  {text!r}")

    print("\nDecoded sequence:")
    print(f"  {decode_sequence(args.ids, vocab)!r}")


def wrap_text(text: str, width: int = 100) -> str:
    """Word-wrap text at the given width, preserving existing newlines."""
    lines = []
    for paragraph in text.split("\n"):
        if not paragraph.strip():
            lines.append("")
            continue
        words = paragraph.split(" ")
        current_line = ""
        for word in words:
            if not word:
                continue
            if current_line and len(current_line) + 1 + len(word) > width:
                lines.append(current_line)
                current_line = word
            elif current_line:
                current_line += f" {word}"
            else:
                current_line = word
        if current_line:
            lines.append(current_line)
    return "\n".join(lines)


def cmd_decode_sequences(args, vocab):
    """Decode sequences from training_data.grmt."""
    grmt = Path(args.grmt)
    if not grmt.exists():
        raise FileNotFoundError(f"GRMT file not found: {grmt}")

    header = read_grmt_header(grmt)
    validate_grmt_vocab_pair(header, vocab, grmt)
    total = header["num_sequences"]
    start = args.seq[0] if args.seq else 0
    end   = args.seq[1] if args.seq and len(args.seq) > 1 else start + 10
    end   = min(end, total)

    # Default: show first 500 chars per sequence unless --full
    max_chars = None if args.full else 500

    print(f"GRMT: {total} sequences, vocab_size={header['vocab_size']}")
    print(f"Showing sequences [{start}, {end}):\n")

    for idx, tokens, atom_texts in iter_grmt_sequences(grmt):
        if idx < start:
            continue
        if idx >= end:
            break

        text = decode_sequence(tokens, vocab, atom_texts)

        print(f"{'═' * 80}")
        print(f"  Sequence {idx}  |  {len(tokens)} tokens  |  {len(text)} chars")
        print(f"{'═' * 80}")
        if args.raw:
            for i, tid in enumerate(tokens):
                piece = atom_texts[i] if ATOM_TOKEN_START <= tid < ATOM_TOKEN_END and atom_texts[i] else decode_token(tid, vocab)
                region = token_type_label(tid)
                print(f"  [{i:>4d}] {tid:>6d} {region:>7s}  {piece!r}")
            print()

        display = text
        truncated = False
        if max_chars is not None and len(text) > max_chars:
            display = text[:max_chars]
            truncated = True

        print(wrap_text(display.strip()))
        if truncated:
            assert max_chars is not None
            print(f"\n  ... [{len(text) - max_chars} more chars, use --full to see all]")
        print()


def cmd_search(args, vocab):
    """Search sequences for a substring."""
    grmt = Path(args.grmt)
    if not grmt.exists():
        raise FileNotFoundError(f"GRMT file not found: {grmt}")

    header = read_grmt_header(grmt)
    validate_grmt_vocab_pair(header, vocab, grmt)

    query = args.search.lower()
    found = 0
    limit = args.limit

    for idx, tokens, atom_texts in iter_grmt_sequences(grmt):
        text = decode_sequence(tokens, vocab, atom_texts)
        if query in text.lower():
            print(f"{'═' * 80}")
            print(f"  Sequence {idx}  |  {len(tokens)} tokens  |  {len(text)} chars")
            print(f"{'═' * 80}")
            print(wrap_text(text.strip()))
            print()
            found += 1
            if found >= limit:
                print(f"(stopped after {limit} matches, use --limit N to show more)")
                break

    if found == 0:
        print(f"No sequences contain '{args.search}'")
    else:
        print(f"Found {found} matching sequence(s).")


def cmd_stats(args, vocab):
    """Print vocab and GRMT statistics."""
    grmt = Path(args.grmt)
    if not grmt.exists():
        raise FileNotFoundError(f"GRMT file not found: {grmt}")

    header = read_grmt_header(grmt)
    validate_grmt_vocab_pair(header, vocab, grmt)

    print("═══ Vocabulary ═══")
    n_special = NUM_SPECIAL_TOKENS
    n_byte    = BYTE_VOCAB_SIZE
    n_atom    = NUM_ATOM_TYPES
    n_unigram = len(vocab) - UNIGRAM_TOKEN_START
    print(f"  Total entries : {len(vocab)}")
    print(f"  Special       : {n_special}  (IDs 0-{NUM_SPECIAL_TOKENS-1})")
    print(f"  Byte fallback : {n_byte}  (IDs {BYTE_TOKEN_OFFSET}-{ATOM_TOKEN_START-1})")
    print(f"  Atom slots    : {n_atom}  (IDs {ATOM_TOKEN_START}-{ATOM_TOKEN_END-1})")
    print(f"  Unigram pieces: {n_unigram}  (IDs {UNIGRAM_TOKEN_START}+)")
    print()

    print("═══ Training Data (GRMT) ═══")
    print(f"  Magic       : {header['magic_hex']}")
    print(f"  Version     : {header['version']}")
    print(f"  Sequences   : {header['num_sequences']}")
    print(f"  Vocab size  : {header['vocab_size']}")
    print()

    # Scan sequences for statistics
    token_counter = Counter()
    total_tokens = 0
    seq_lengths = []
    unknown_ids = set()

    for idx, tokens, _atom_texts in iter_grmt_sequences(grmt):
        seq_lengths.append(len(tokens))
        total_tokens += len(tokens)
        for tid in tokens:
            token_counter[tid] += 1
            if tid not in vocab:
                unknown_ids.add(tid)

    print("═══ Sequence Statistics ═══")
    print(f"  Total tokens   : {total_tokens:,}")
    print(f"  Total sequences: {len(seq_lengths):,}")
    if seq_lengths:
        print(f"  Avg seq length : {sum(seq_lengths)/len(seq_lengths):.1f}")
        print(f"  Min seq length : {min(seq_lengths)}")
        print(f"  Max seq length : {max(seq_lengths)}")
    print()

    if unknown_ids:
        print(f"  ⚠ Unknown token IDs ({len(unknown_ids)}): {sorted(unknown_ids)[:20]}")
    else:
        print("  ✓ All token IDs map to known vocab entries")
    print()

    # Top 20 tokens
    print("═══ Top 20 Most Frequent Tokens ═══")
    for tid, count in token_counter.most_common(20):
        text = decode_token(tid, vocab)
        region = token_type_label(tid)
        pct = 100.0 * count / total_tokens
        print(f"  {tid:>6d} [{region:>7s}]  {count:>8,}  ({pct:5.2f}%)  {text!r}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Decode GRIM-text token IDs from training_data.grmt using vocab.bin"
    )
    parser.add_argument("--vocab", default=str(VOCAB_BIN),
                        help="Path to vocab.bin")
    parser.add_argument("--grmt", default=str(GRMT_FILE),
                        help="Path to training_data.grmt")
    parser.add_argument("--ids", nargs="+",
                        help="Decode token IDs (e.g. --ids 277 512 36 or --ids 277,512,36)")
    parser.add_argument("--seq", type=int, nargs="*",
                        help="Sequence range: --seq START END (default: first 10)")
    parser.add_argument("--search", type=str,
                        help="Search decoded text for a substring")
    parser.add_argument("--raw", action="store_true",
                        help="Show raw token IDs alongside decoded text")
    parser.add_argument("--stats", action="store_true",
                        help="Print vocabulary and training data statistics")
    parser.add_argument("--limit", type=int, default=20,
                        help="Max results for --search (default: 20)")
    parser.add_argument("--full", action="store_true",
                        help="Show full sequence text (default: truncated to 500 chars)")

    args = parser.parse_args()

    # Load vocabulary
    vocab_path = Path(args.vocab)
    print(f"Loading vocab: {vocab_path}")
    vocab = load_vocab_bin(vocab_path)
    print(f"Loaded {len(vocab)} token mappings\n")

    if args.ids:
        args.ids = parse_cli_token_ids(args.ids)
        cmd_decode_ids(args, vocab)
    elif args.search:
        cmd_search(args, vocab)
    elif args.stats:
        cmd_stats(args, vocab)
    else:
        cmd_decode_sequences(args, vocab)


if __name__ == "__main__":
    main()
