#!/usr/bin/env python3
"""Scan vocab for garbage tokens"""

import re

path = r'D:\G.R.I.M\resources\models\GRIM-text\training\data\vocab.txt'

# Patterns that indicate garbage tokens
garbage_patterns = [
    # Single letter + punctuation (P.,  A;, etc.)
    re.compile(r'^[A-Za-z][.,;:!?\-]+$'),
    # Pure punctuation combos (more than 1 char)
    re.compile(r'^[.,;:!?\-\"\'\(\)\[\]\{\}]{2,}$'),
    # Punctuation-letter-punctuation
    re.compile(r'^[.,;:!?]+[A-Za-z][.,;:!?]+$'),
    # Space + punctuation
    re.compile(r'^\s+[.,;:!?]+$'),
    # Punctuation + space
    re.compile(r'^[.,;:!?]+\s+$'),
]

garbage = []
total = 0
with open(path, 'r', encoding='utf-8') as f:
    for line in f:
        parts = line.strip().split('\t')
        if len(parts) >= 2:
            total += 1
            token = parts[0]
            score = float(parts[1])
            
            for pattern in garbage_patterns:
                if pattern.match(token):
                    garbage.append((token, score))
                    break

print(f'Total vocab entries: {total}')
print(f'Potential garbage tokens: {len(garbage)}')
print()
if garbage:
    print('Garbage tokens found:')
    for t, s in sorted(garbage, key=lambda x: x[1], reverse=True)[:50]:
        print(f'  {repr(t):25} score={s:.4f}')
else:
    print('✅ No obvious garbage patterns detected!')

# Also check for the specific "P.," pattern
print('\n--- Searching for "P.," specifically ---')
with open(path, 'r', encoding='utf-8') as f:
    for line in f:
        if 'P.,' in line or 'P.,' in line.split('\t')[0]:
            print(f'  Found: {line.strip()}')
