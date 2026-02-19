#!/usr/bin/env python3
"""Check stored token IDs in vocab.bin vs sequential index."""
import struct
from pathlib import Path

VOCAB_BIN = Path("resources/models/GRIM-text/training/data/vocab.bin")
GRMT_FILE = Path("resources/models/GRIM-text/training/data/training_data.grmt")

pieces = []
with open(VOCAB_BIN, "rb") as f:
    f.read(4)
    version = struct.unpack("<H", f.read(2))[0]
    f.read(4 + 4 + 4 + 3 + 4)
    for i in range(10000):
        plen = struct.unpack("<I", f.read(4))[0]
        text = f.read(plen).decode("utf-8", errors="replace")
        score = struct.unpack("<f", f.read(4))[0]
        stored_id = struct.unpack("<I", f.read(4))[0] if version >= 3 else (277 + i)
        pieces.append((text, score, stored_id))

stored_map = {}
collisions = 0
for i, (text, score, sid) in enumerate(pieces):
    if sid in stored_map:
        collisions += 1
    stored_map[sid] = text

print(f"Pieces: {len(pieces)}, Unique stored_ids: {len(stored_map)}, Collisions: {collisions}")

print("\nFirst 15 pieces:")
for i in range(15):
    t, s, sid = pieces[i]
    match = "OK" if sid == 277 + i else "MISMATCH"
    print(f"  [{i}] stored_id={sid} expected={277+i} {match}  text={repr(t)}")

# Find first mismatch
first_mm = None
for i, (t, s, sid) in enumerate(pieces):
    if sid != 277 + i:
        first_mm = i
        break

if first_mm is not None:
    print(f"\nFirst mismatch at index {first_mm}:")
    for j in range(max(0, first_mm - 2), min(len(pieces), first_mm + 8)):
        tt, ss, ssid = pieces[j]
        match = "OK" if ssid == 277 + j else "MISMATCH"
        print(f"  [{j}] stored_id={ssid} expected={277+j} {match}  text={repr(tt)}")
else:
    print("\nAll stored_ids match sequential!")

# Using stored_id mapping: what is GRMT token 690?
print(f"\nstored_id 690 -> {repr(stored_map.get(690, 'NOT FOUND'))}")

# What stored_id does piece[5] ('he') have?
t5, s5, sid5 = pieces[5]
print(f"Piece[5] (text={repr(t5)}): stored_id = {sid5}")

# Read GRMT first sequence and decode with STORED_ID mapping
with open(GRMT_FILE, "rb") as f:
    f.read(16)  # header
    seq_len = struct.unpack("<I", f.read(4))[0]
    tokens = list(struct.unpack(f"<{seq_len}i", f.read(4 * seq_len)))

print(f"\nGRMT seq 0 first 20 tokens decoded via STORED_ID mapping:")
for i, tid in enumerate(tokens[:20]):
    if tid < 4:
        text = f"<special:{tid}>"
    elif 4 <= tid < 260:
        b = tid - 4
        text = chr(b) if 32 <= b <= 126 else f"<byte:{b:#x}>"
    elif 260 <= tid < 277:
        text = f"<atom:{tid-260}>"
    else:
        text = stored_map.get(tid, f"<UNMAPPED:{tid}>")
    print(f"  [{i}] token {tid} -> {repr(text)}")
