#!/usr/bin/env python3
"""Verify: re-encode source text with current vocab, compare with GRMT tokens."""
import struct, json
from pathlib import Path

VOCAB_BIN = Path("resources/models/GRIM-text/training/data/vocab.bin")
GRMT_FILE = Path("resources/models/GRIM-text/training/data/training_data.grmt")
JSONL_FILE = Path("resources/models/GRIM-text/training/data/merged_verified_cache.jsonl")

UNIGRAM_OFFSET = 277
BYTE_OFFSET = 4
MAX_PIECE_LEN = 32

# Load vocab pieces
pieces = []
with open(VOCAB_BIN, "rb") as f:
    f.read(4)
    version = struct.unpack("<H", f.read(2))[0]
    f.read(4 + 4 + 4 + 3 + 4)
    for i in range(10000):
        plen = struct.unpack("<I", f.read(4))[0]
        text = f.read(plen).decode("utf-8", errors="replace")
        score = struct.unpack("<f", f.read(4))[0]
        stored_id = struct.unpack("<I", f.read(4))[0] if version >= 3 else None
        pieces.append((text, score, stored_id))

# Build trie for Viterbi (using sequential IDs like getPiece expects)
# and also build one with stored IDs
class TrieNode:
    __slots__ = ["children", "token_id", "score"]
    def __init__(self):
        self.children = {}
        self.token_id = -1
        self.score = 0.0

def build_trie(pieces_list, use_stored_id=False):
    root = TrieNode()
    for i, (text, score, sid) in enumerate(pieces_list):
        node = root
        for ch in text.encode("utf-8"):
            if ch not in node.children:
                node.children[ch] = TrieNode()
            node = node.children[ch]
        tid = sid if use_stored_id else (UNIGRAM_OFFSET + i)
        node.token_id = tid
        node.score = score
    return root

def viterbi_encode(text_bytes, trie_root):
    """Viterbi encoding - returns list of token IDs."""
    n = len(text_bytes)
    scores = [float("-inf")] * (n + 1)
    prev_pos = [-1] * (n + 1)
    prev_tid = [-1] * (n + 1)
    scores[0] = 0.0

    for pos in range(n):
        if scores[pos] == float("-inf"):
            continue
        # Try trie matches starting at pos
        node = trie_root
        for length in range(1, min(MAX_PIECE_LEN + 1, n - pos + 1)):
            byte = text_bytes[pos + length - 1]
            if byte not in node.children:
                break
            node = node.children[byte]
            if node.token_id >= 0:
                cand = scores[pos] + node.score
                if cand > scores[pos + length]:
                    scores[pos + length] = cand
                    prev_pos[pos + length] = pos
                    prev_tid[pos + length] = node.token_id

        # Byte fallback
        byte_score = scores[pos] + (-100.0)
        if byte_score > scores[pos + 1]:
            scores[pos + 1] = byte_score
            prev_pos[pos + 1] = pos
            prev_tid[pos + 1] = text_bytes[pos] + BYTE_OFFSET

    # Backtrack
    tokens = []
    pos = n
    while pos > 0:
        tokens.append(prev_tid[pos])
        pos = prev_pos[pos]
    tokens.reverse()
    return tokens

# Build tries
trie_seq = build_trie(pieces, use_stored_id=False)
trie_stored = build_trie(pieces, use_stored_id=True)

# Read source text 0
with open(JSONL_FILE) as f:
    source_text = json.loads(f.readline())["content"]

# HTML cleaning (simple)
import re
source_text = re.sub(r"<[^>]+>", "", source_text)
source_text = re.sub(r"&amp;", "&", source_text)
source_text = re.sub(r"&lt;", "<", source_text)
source_text = re.sub(r"&gt;", ">", source_text)
source_text = re.sub(r"&nbsp;", " ", source_text)
source_text = re.sub(r"\s+", " ", source_text).strip()

text_bytes = source_text.encode("utf-8")

# Encode with both tries
tokens_seq = viterbi_encode(text_bytes, trie_seq)
tokens_stored = viterbi_encode(text_bytes, trie_stored)

# Read GRMT tokens
with open(GRMT_FILE, "rb") as f:
    f.read(16)
    seq_len = struct.unpack("<I", f.read(4))[0]
    grmt_tokens = list(struct.unpack(f"<{seq_len}i", f.read(4 * seq_len)))

print(f"Source text (first 100): {repr(source_text[:100])}")
print(f"Re-encoded (sequential): {len(tokens_seq)} tokens, first 20: {tokens_seq[:20]}")
print(f"Re-encoded (stored_id):  {len(tokens_stored)} tokens, first 20: {tokens_stored[:20]}")
print(f"GRMT tokens:             {len(grmt_tokens)} tokens, first 20: {grmt_tokens[:20]}")

# Check matches
match_seq = tokens_seq[:20] == grmt_tokens[:20]
match_stored = tokens_stored[:20] == grmt_tokens[:20]
print(f"\nSequential match: {match_seq}")
print(f"Stored_id match:  {match_stored}")

# Token-by-token comparison
print("\nToken-by-token (first 20):")
for i in range(min(20, len(tokens_seq), len(grmt_tokens))):
    gs = tokens_seq[i] if i < len(tokens_seq) else "?"
    gst = tokens_stored[i] if i < len(tokens_stored) else "?"
    gt = grmt_tokens[i]
    match = "SEQ" if gs == gt else ("STORED" if gst == gt else "NEITHER")
    print(f"  [{i:2d}] seq={gs:5d} stored={gst:5d} grmt={gt:5d} -> {match}")
