#!/usr/bin/env python3
"""Analyze UnigramLM vocabulary distribution."""

import re
from collections import Counter

vocab_path = r"D:\G.R.I.M\resources\models\GRIM-text\training\data\vocab.txt"

tokens = []
scores = []

with open(vocab_path, 'r', encoding='utf-8') as f:
    for line in f:
        # Format: piece<TAB>score
        parts = line.rstrip('\n').rsplit('\t', 1)
        if len(parts) == 2:
            piece, score = parts[0], float(parts[1])
            tokens.append((piece, score))
            scores.append(score)

print("=" * 60)
print("UNIGRAMLM VOCABULARY DISTRIBUTION")
print("=" * 60)
print(f"Total tokens: {len(tokens)}")
print()

# Token length distribution
lengths = [len(t[0]) for t in tokens]
length_dist = Counter()
for l in lengths:
    if l == 1:
        length_dist["1 char"] += 1
    elif l <= 3:
        length_dist["2-3 chars"] += 1
    elif l <= 8:
        length_dist["4-8 chars"] += 1
    else:
        length_dist["9+ chars"] += 1

print("Token length distribution:")
for k in ["1 char", "2-3 chars", "4-8 chars", "9+ chars"]:
    pct = length_dist[k] / len(tokens) * 100
    print(f"  {k:12s}: {length_dist[k]:6d} ({pct:5.1f}%)")
print()

# Special tokens (angle brackets)
special = [t for t in tokens if t[0].startswith('<') and t[0].endswith('>')]
print(f"Special tokens ({len(special)}):")
for piece, score in special:
    print(f"  {piece}: score={score:.4f}")
print()

# Score distribution
print("Score distribution (log-prob):")
print(f"  Min:  {min(scores):.4f}")
print(f"  Max:  {max(scores):.4f}")
print(f"  Mean: {sum(scores)/len(scores):.4f}")
print()

# Token type analysis
whitespace = sum(1 for t in tokens if t[0].strip() == '' or t[0] == ' ')
starts_space = sum(1 for t in tokens if t[0].startswith(' '))
alpha_only = sum(1 for t in tokens if t[0].isalpha())
has_digit = sum(1 for t in tokens if any(c.isdigit() for c in t[0]))
punctuation = sum(1 for t in tokens if all(not c.isalnum() and not c.isspace() for c in t[0]) and len(t[0]) > 0)

print("Token type analysis:")
print(f"  Whitespace tokens:     {whitespace:6d}")
print(f"  Starts with space:     {starts_space:6d}")
print(f"  Pure alphabetic:       {alpha_only:6d}")
print(f"  Contains digits:       {has_digit:6d}")
print(f"  Pure punctuation:      {punctuation:6d}")
print()

# Top 30 most frequent (highest score = most frequent)
sorted_tokens = sorted(tokens, key=lambda x: x[1], reverse=True)
print("Top 30 most frequent tokens:")
for i, (piece, score) in enumerate(sorted_tokens[:30], 1):
    display = piece.replace('\n', '\\n').replace('\t', '\\t').replace('\r', '\\r')
    if display == ' ':
        display = '[SPACE]'
    elif display == '':
        display = '[EMPTY]'
    print(f"  {i:2d}. [{display:15s}] score={score:.4f}")
print()

# Bottom 30 (rarest)
print("Bottom 30 rarest tokens:")
for i, (piece, score) in enumerate(sorted_tokens[-30:], 1):
    display = piece.replace('\n', '\\n').replace('\t', '\\t').replace('\r', '\\r')
    if len(display) > 20:
        display = display[:17] + "..."
    print(f"  {i:2d}. [{display:20s}] score={score:.4f}")
print()

# Expected initial loss (random baseline)
import math
expected_loss = math.log(len(tokens))
print(f"Expected initial loss (random uniform): {expected_loss:.4f}")
print(f"Expected initial loss ln(50376): {math.log(50376):.4f}")
print(f"Actual initial loss from training: ~5.93")
print()

# Check for common subword patterns
print("Common subword prefix patterns:")
prefix_counts = Counter()
for t, _ in tokens:
    if len(t) >= 2:
        if t.startswith(' '):
            prefix_counts["_word (space-prefixed)"] += 1
        elif t[0].isupper():
            prefix_counts["Capitalized"] += 1
prefix_counts["Single chars"] = length_dist["1 char"]
for k, v in prefix_counts.most_common(10):
    print(f"  {k}: {v}")
