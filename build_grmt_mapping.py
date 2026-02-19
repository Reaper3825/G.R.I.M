#!/usr/bin/env python3
"""Build empirical GRMT_token -> piece_text mapping by aligning source texts with GRMT."""
import struct, json, re, sys
from pathlib import Path

VOCAB_BIN = Path("resources/models/GRIM-text/training/data/vocab.bin")
GRMT_FILE = Path("resources/models/GRIM-text/training/data/training_data.grmt")
JSONL_FILE = Path("resources/models/GRIM-text/training/data/merged_verified_cache.jsonl")

UNIGRAM_OFFSET = 277
BYTE_OFFSET = 4
MAX_PIECE_LEN = 32

# ─── Load vocab ─────────────────────────────────────────────
pieces = []  # (text, score, stored_id)
with open(VOCAB_BIN, "rb") as f:
    f.read(4)
    version = struct.unpack("<H", f.read(2))[0]
    f.read(4 + 4 + 4 + 3 + 4)
    for i in range(10000):
        plen = struct.unpack("<I", f.read(4))[0]
        text = f.read(plen).decode("utf-8", errors="replace")
        score = struct.unpack("<f", f.read(4))[0]
        stored_id = struct.unpack("<I", f.read(4))[0] if version >= 3 else (UNIGRAM_OFFSET + i)
        pieces.append((text, score, stored_id))

seq_id_to_text = {}
for i, (text, score, sid) in enumerate(pieces):
    seq_id_to_text[UNIGRAM_OFFSET + i] = text

# ─── Build trie (sequential IDs) ────────────────────────────
class TrieNode:
    __slots__ = ["children", "token_id", "score"]
    def __init__(self):
        self.children = {}
        self.token_id = -1
        self.score = 0.0

root = TrieNode()
for i, (text, score, _) in enumerate(pieces):
    node = root
    for ch in text.encode("utf-8"):
        if ch not in node.children:
            node.children[ch] = TrieNode()
        node = node.children[ch]
    node.token_id = UNIGRAM_OFFSET + i
    node.score = score

# ─── Viterbi encoder ────────────────────────────────────────
def viterbi_encode(text_bytes):
    n = len(text_bytes)
    scores = [float("-inf")] * (n + 1)
    prev_pos = [-1] * (n + 1)
    prev_tid = [-1] * (n + 1)
    scores[0] = 0.0

    for pos in range(n):
        if scores[pos] == float("-inf"):
            continue
        node = root
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

        byte_score = scores[pos] + (-100.0)
        if byte_score > scores[pos + 1]:
            scores[pos + 1] = byte_score
            prev_pos[pos + 1] = pos
            prev_tid[pos + 1] = text_bytes[pos] + BYTE_OFFSET

    tokens = []
    pos = n
    while pos > 0:
        tokens.append(prev_tid[pos])
        pos = prev_pos[pos]
    tokens.reverse()
    return tokens

# ─── HTML cleaning (matching DataLoader.cu) ──────────────────
def clean_text(text):
    text = re.sub(r"<[^>]+>", "", text)
    text = text.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
    text = text.replace("&nbsp;", " ").replace("&quot;", '"').replace("&#39;", "'")
    text = re.sub(r"&#x([0-9a-fA-F]+);", lambda m: chr(int(m.group(1), 16)), text)
    text = re.sub(r"&#(\d+);", lambda m: chr(int(m.group(1))), text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# ─── Read GRMT sequences ────────────────────────────────────
grmt_sequences = []
with open(GRMT_FILE, "rb") as f:
    magic, ver, num_seq, vocab_size = struct.unpack("<IIII", f.read(16))
    for _ in range(num_seq):
        seq_len = struct.unpack("<I", f.read(4))[0]
        tids = list(struct.unpack(f"<{seq_len}i", f.read(4 * seq_len)))
        targets = list(struct.unpack(f"<{seq_len}i", f.read(4 * seq_len)))
        f.read(seq_len * 4)  # numeric_values
        f.read(seq_len)      # numeric_mask
        f.read(seq_len * 16 * 2)  # text_features (16 dims * FP16)
        f.read(seq_len)      # text_feature_mask
        grmt_sequences.append(tids)
print(f"Loaded {len(grmt_sequences)} GRMT sequences")

# ─── Read source texts ──────────────────────────────────────
source_texts = []
with open(JSONL_FILE) as f:
    for line in f:
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
            if "content" in obj:
                source_texts.append(clean_text(obj["content"]))
        except:
            continue
print(f"Loaded {len(source_texts)} source texts")

# ─── Build mapping by aligning ───────────────────────────────
grmt_to_seq = {}  # GRMT token_id -> sequential token_id
mapped_count = 0
mismatch_count = 0
length_mismatch_count = 0

num_to_check = min(len(source_texts), len(grmt_sequences), 200)

for idx in range(num_to_check):
    text_bytes = source_texts[idx].encode("utf-8")
    if len(text_bytes) < 20:
        continue

    seq_tokens = viterbi_encode(text_bytes)
    grmt_tokens = grmt_sequences[idx]

    if len(seq_tokens) != len(grmt_tokens):
        length_mismatch_count += 1
        continue

    for s, g in zip(seq_tokens, grmt_tokens):
        if s == g:
            continue  # Already matches (byte tokens, early pieces)
        if g in grmt_to_seq:
            if grmt_to_seq[g] != s:
                mismatch_count += 1
        else:
            grmt_to_seq[g] = s
            mapped_count += 1

print(f"\nAlignment results ({num_to_check} sequences):")
print(f"  New mappings discovered: {mapped_count}")
print(f"  Conflicting mappings: {mismatch_count}")
print(f"  Length mismatches (skipped): {length_mismatch_count}")

# ─── Verify: decode GRMT seq 0 using the mapping ────────────
def decode_token(tid):
    if tid < 4:
        return {0: "<unk>", 1: "<pad>", 2: "<s>", 3: "</s>"}.get(tid, f"<sp:{tid}>")
    if 4 <= tid < 260:
        b = tid - 4
        return chr(b) if 32 <= b <= 126 else bytes([b]).decode("utf-8", errors="replace")
    if 260 <= tid < 277:
        return f"<ATOM:{tid-260}>"
    # Unigram: try GRMT->seq mapping first, then direct sequential
    mapped_seq = grmt_to_seq.get(tid, tid)
    return seq_id_to_text.get(mapped_seq, f"<UNK:{tid}>")

decoded_text = "".join(decode_token(t) for t in grmt_sequences[0])
print(f"\nDecoded GRMT seq 0 (first 300 chars):")
print(decoded_text[:300])
print(f"\nSource text 0 (first 300 chars):")
print(source_texts[0][:300])

# ─── Coverage stats ─────────────────────────────────────────
all_grmt_unigram = set()
for seq in grmt_sequences:
    for t in seq:
        if t >= 277:
            all_grmt_unigram.add(t)

mapped_ids = set(grmt_to_seq.keys()) | set(range(277, 277 + len(pieces)))  # mapped + identity
coverage = len(all_grmt_unigram & mapped_ids) / max(1, len(all_grmt_unigram)) * 100
print(f"\nGRMT unigram tokens used: {len(all_grmt_unigram)}")
print(f"Coverage with mapping: {coverage:.1f}%")

# Save mapping
import pickle
mapping_file = Path("resources/models/GRIM-text/training/data/grmt_token_mapping.pkl")
with open(mapping_file, "wb") as f:
    pickle.dump(grmt_to_seq, f)
print(f"Saved mapping to {mapping_file}")
