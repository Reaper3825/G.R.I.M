#!/usr/bin/env python3
import json

lines_50 = 0
lines_100 = 0
tokens_50 = 0
tokens_100 = 0

with open('resources/models/GRIM-text/training/data/merged_verified_cache.jsonl') as f:
    for i, line in enumerate(f):
        if i < 50:
            data = json.loads(line)
            tokens_50 += len(data.get('content', ''))
            lines_50 += 1
        if i < 100:
            data = json.loads(line)
            tokens_100 += len(data.get('content', ''))
            lines_100 += 1
        if i >= 100:
            break

print(f"First 50 lines: {lines_50} lines, {tokens_50} tokens")
print(f"First 100 lines: {lines_100} lines, {tokens_100} tokens")
print(f"Average tokens per line (50): {tokens_50/lines_50:.1f}")
print(f"Average tokens per line (100): {tokens_100/lines_100:.1f}")

# Calculate expected loss
batch_size = 4
seq_len = 128
vocab_size = 257
print(f"\nWith batch_size={batch_size}, seq_len={seq_len}, vocab_size={vocab_size}:")
print(f"Random baseline loss: {__import__('math').log(vocab_size):.4f}")
