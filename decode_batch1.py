"""Decode all sequences from FULL TOKEN DUMP in training log."""
import re, sys
from pathlib import Path
from decode_token_ids import load_vocab_bin, decode_sequence

log_path = Path(r"D:/G.R.I.M/resources/models/GRIM-text/training/logs/training_17713862417198392.log")
vocab_path = Path(r"D:/G.R.I.M/resources/models/GRIM-text/training/data/vocab.bin")

vocab = load_vocab_bin(vocab_path)

# Parse sequences from FULL TOKEN DUMP
text = log_path.read_text(encoding="utf-8")
dump_start = text.find("[BOUNDARY_DIAGNOSTIC] FULL TOKEN DUMP")
dump_end = text.find("[BOUNDARY_DIAGNOSTIC] ====", dump_start + 1)
dump = text[dump_start:dump_end]

# Find each seq block
seq_pattern = re.compile(r"seq\[(\d+)\] \((\d+) tokens\):\n(.*?)(?=\n  seq\[|\Z)", re.DOTALL)
for m in seq_pattern.finditer(dump):
    seq_id = int(m.group(1))
    if seq_id == 0:
        continue  # already decoded
    token_count = int(m.group(2))
    token_text = m.group(3).strip()
    # Extract all integers
    ids = [int(x) for x in re.findall(r'\d+', token_text)]
    # Strip padding (token 1)
    ids_no_pad = [i for i in ids if i != 1]
    decoded = decode_sequence(ids_no_pad, vocab)
    print(f"=== SEQ[{seq_id}] ({len(ids_no_pad)} content tokens, {len(ids)} total) ===")
    print(decoded)
    print()
