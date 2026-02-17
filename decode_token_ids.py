#!/usr/bin/env python3
"""
Decode token IDs from training_data.grmt using vocab.bin.

Token ID Layout (GrimTokenizer):
  [0-3]       = Special tokens: <unk>=0, <pad>=1, <s>=2, </s>=3
  [4-259]     = Byte fallback (byte value = token_id - 4)
  [260-276]   = Atom placeholders (17 atom types)
  [277+]      = Unigram vocabulary pieces (from vocab.bin)

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

# ── Paths (from ai_config.json) ──────────────────────────────────────────────

VOCAB_BIN  = Path(r"D:/G.R.I.M/resources/models/GRIM-text/training/data/vocab.bin")
GRMT_FILE  = Path(r"D:/G.R.I.M/resources/models/GRIM-text/training/data/training_data.grmt")

# ── Token layout constants ────────────────────────────────────────────────────

NUM_SPECIAL_TOKENS = 4
BYTE_TOKEN_OFFSET  = NUM_SPECIAL_TOKENS   # 4
BYTE_VOCAB_SIZE    = 256
ATOM_TOKEN_START   = BYTE_TOKEN_OFFSET + BYTE_VOCAB_SIZE  # 260

SPECIAL_NAMES = {0: "<unk>", 1: "<pad>", 2: "<s>", 3: "</s>"}

ATOM_TYPE_LABELS = {
    0:  "<ATOM_NONE>",
    1:  "<ATOM_END>",
    2:  "<INT>",
    3:  "<FLOAT>",
    4:  "<HEX>",
    5:  "<BIN>",
    6:  "<ID>",
    7:  "<STR>",
    8:  "<REGEX>",
    9:  "<URL>",
    10: "<EMAIL>",
    11: "<PATH>",
    12: "<DATE>",
    13: "<TIME>",
    14: "<IP>",
    15: "<EQUATION>",
    16: "<EXPR>",
}
NUM_ATOM_TYPES = 17  # ATOM_ACTIVE_COUNT

ATOM_TOKEN_END        = ATOM_TOKEN_START + NUM_ATOM_TYPES  # 277
UNIGRAM_TOKEN_START   = ATOM_TOKEN_END                     # 277


# ── vocab.bin loader (KTMG format) ───────────────────────────────────────────

def load_vocab_bin(path: Path) -> dict:
    """
    Load vocab.bin and build a complete token_id -> text mapping.

    Returns dict mapping every known token ID to its string representation.
    """
    if not path.exists():
        raise FileNotFoundError(f"vocab.bin not found: {path}")

    id_to_text = {}

    # 1) Special tokens
    for tid, name in SPECIAL_NAMES.items():
        id_to_text[tid] = name

    # 2) Byte fallback tokens
    for b in range(BYTE_VOCAB_SIZE):
        tid = BYTE_TOKEN_OFFSET + b
        if 32 <= b <= 126:
            id_to_text[tid] = chr(b)
        else:
            id_to_text[tid] = f"<BYTE 0x{b:02X}>"

    # 3) Atom placeholder tokens
    for i in range(NUM_ATOM_TYPES):
        tid = ATOM_TOKEN_START + i
        id_to_text[tid] = ATOM_TYPE_LABELS.get(i, f"<ATOM{i}>")

    # 4) Unigram pieces from vocab.bin
    with open(path, "rb") as f:
        magic = f.read(4)
        if magic not in (b"KTMG", b"GMTK", b"GRIM"):
            raise ValueError(f"Bad vocab.bin magic: {magic!r}")

        if magic in (b"KTMG", b"GMTK"):
            # Binary format: uint16 version, then header fields, then pieces
            version = struct.unpack("<H", f.read(2))[0]
            checksum      = struct.unpack("<I", f.read(4))[0]
            unigram_count = struct.unpack("<I", f.read(4))[0]  # config_vocab_size
            max_length    = struct.unpack("<I", f.read(4))[0]
            flags         = f.read(3)
            total_vocab   = struct.unpack("<I", f.read(4))[0]

            for i in range(unigram_count):
                piece_len = struct.unpack("<I", f.read(4))[0]
                text = f.read(piece_len).decode("utf-8", errors="replace")
                score = struct.unpack("<f", f.read(4))[0]

                if version >= 3:
                    token_id = struct.unpack("<I", f.read(4))[0]
                else:
                    token_id = UNIGRAM_TOKEN_START + i

                id_to_text[token_id] = text

        elif magic == b"GRIM":
            # Older format: uint32 version, uint32 vocab_size, then tokens
            version    = struct.unpack("<I", f.read(4))[0]
            vocab_size = struct.unpack("<I", f.read(4))[0]

            for i in range(vocab_size):
                token_len = struct.unpack("<I", f.read(4))[0]
                text = f.read(token_len).decode("utf-8", errors="replace")
                id_to_text[i] = text

    return id_to_text


# ── training_data.grmt loader ────────────────────────────────────────────────

def read_grmt_header(path: Path) -> dict:
    """Read the 16-byte GRMT header."""
    with open(path, "rb") as f:
        magic, version, num_sequences, vocab_size = struct.unpack("<IIII", f.read(16))
    return {
        "magic": magic,
        "magic_hex": hex(magic),
        "version": version,
        "num_sequences": num_sequences,
        "vocab_size": vocab_size,
    }


TEXT_FEATURE_DIM = 16  # kTextFeatureDim in C++ code


def iter_grmt_sequences(path: Path):
    """Yield (index, token_id_list) for each sequence in the GRMT file.
    
    GRMT v5 per-sequence layout (must read ALL fields to stay in sync):
      uint32         seq_len
      int32[seq_len] token_ids
      int32[seq_len] targets
      float[seq_len] numeric_values
      uint8[seq_len] numeric_mask
      uint16[seq_len * TEXT_FEATURE_DIM] text_features
      uint8[seq_len] text_feature_mask
    """
    with open(path, "rb") as f:
        _magic, _version, num_sequences, _vocab_size = struct.unpack("<IIII", f.read(16))

        for idx in range(num_sequences):
            raw = f.read(4)
            if not raw or len(raw) < 4:
                break
            seq_len = struct.unpack("<I", raw)[0]
            
            # Read token_ids (the field we care about)
            token_bytes = f.read(4 * seq_len)
            if len(token_bytes) < 4 * seq_len:
                break
            tokens = list(struct.unpack(f"<{seq_len}I", token_bytes))
            
            # Skip remaining per-sequence fields to keep file position in sync
            f.read(4 * seq_len)                          # targets (int32)
            f.read(4 * seq_len)                          # numeric_values (float32)
            f.read(1 * seq_len)                          # numeric_mask (uint8)
            f.read(2 * seq_len * TEXT_FEATURE_DIM)       # text_features (uint16)
            f.read(1 * seq_len)                          # text_feature_mask (uint8)
            
            yield idx, tokens


# ── Decode helpers ────────────────────────────────────────────────────────────

def decode_token(tid: int, vocab: dict) -> str:
    """Decode a single token ID to its text representation."""
    if tid in vocab:
        return vocab[tid]
    return f"<UNK:{tid}>"


def decode_sequence(tokens: list, vocab: dict) -> str:
    """Decode a full sequence of token IDs to readable text."""
    pieces = [decode_token(tid, vocab) for tid in tokens]
    text = "".join(pieces)
    text = text.replace("\u2581", " ")  # ▁ (Unigram space marker) → space
    return text


def token_type_label(tid: int) -> str:
    """Return which region a token ID belongs to."""
    if tid < NUM_SPECIAL_TOKENS:
        return "SPECIAL"
    if BYTE_TOKEN_OFFSET <= tid < ATOM_TOKEN_START:
        return "BYTE"
    if ATOM_TOKEN_START <= tid < ATOM_TOKEN_END:
        return "ATOM"
    return "UNIGRAM"


# ── CLI actions ───────────────────────────────────────────────────────────────

def cmd_decode_ids(args, vocab):
    """Decode a list of token IDs from the command line."""
    for tid in args.ids:
        text = decode_token(tid, vocab)
        region = token_type_label(tid)
        print(f"  {tid:>6d}  [{region:>7s}]  {text!r}")


def cmd_decode_sequences(args, vocab):
    """Decode sequences from training_data.grmt."""
    grmt = Path(args.grmt)
    if not grmt.exists():
        raise FileNotFoundError(f"GRMT file not found: {grmt}")

    header = read_grmt_header(grmt)
    total = header["num_sequences"]
    start = args.seq[0] if args.seq else 0
    end   = args.seq[1] if args.seq and len(args.seq) > 1 else start + 10
    end   = min(end, total)

    print(f"GRMT: {total} sequences, vocab_size={header['vocab_size']}")
    print(f"Showing sequences [{start}, {end}):\n")

    for idx, tokens in iter_grmt_sequences(grmt):
        if idx < start:
            continue
        if idx >= end:
            break

        text = decode_sequence(tokens, vocab)

        print(f"── Sequence {idx} ({len(tokens)} tokens) ──")
        if args.raw:
            for i, tid in enumerate(tokens):
                piece = decode_token(tid, vocab)
                region = token_type_label(tid)
                print(f"  [{i:>4d}] {tid:>6d} {region:>7s}  {piece!r}")
            print()
        print(text)
        print()


def cmd_search(args, vocab):
    """Search sequences for a substring."""
    grmt = Path(args.grmt)
    if not grmt.exists():
        raise FileNotFoundError(f"GRMT file not found: {grmt}")

    query = args.search.lower()
    found = 0
    limit = args.limit

    for idx, tokens in iter_grmt_sequences(grmt):
        text = decode_sequence(tokens, vocab)
        if query in text.lower():
            print(f"── Sequence {idx} ({len(tokens)} tokens) ──")
            print(text)
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

    print("═══ Vocabulary ═══")
    n_special = sum(1 for t in vocab if t < NUM_SPECIAL_TOKENS)
    n_byte    = sum(1 for t in vocab if BYTE_TOKEN_OFFSET <= t < ATOM_TOKEN_START)
    n_atom    = sum(1 for t in vocab if ATOM_TOKEN_START <= t < ATOM_TOKEN_END)
    n_unigram = sum(1 for t in vocab if t >= UNIGRAM_TOKEN_START)
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

    for idx, tokens in iter_grmt_sequences(grmt):
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
        print(f"  ✓ All token IDs map to known vocab entries")
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
    parser.add_argument("--ids", type=int, nargs="+",
                        help="Decode specific token IDs (e.g. --ids 277 512 36)")
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

    args = parser.parse_args()

    # Load vocabulary
    vocab_path = Path(args.vocab)
    print(f"Loading vocab: {vocab_path}")
    vocab = load_vocab_bin(vocab_path)
    print(f"Loaded {len(vocab)} token mappings\n")

    if args.ids:
        cmd_decode_ids(args, vocab)
    elif args.search:
        cmd_search(args, vocab)
    elif args.stats:
        cmd_stats(args, vocab)
    else:
        cmd_decode_sequences(args, vocab)


if __name__ == "__main__":
    main()
