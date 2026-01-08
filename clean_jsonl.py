#!/usr/bin/env python3
"""Clean corrupted entries from merged_verified_cache.jsonl"""

import json
import re

path = r'D:\G.R.I.M\resources\models\GRIM-text\training\data\merged_verified_cache.jsonl'
out_path = r'D:\G.R.I.M\resources\models\GRIM-text\training\data\merged_verified_cache_clean.jsonl'

# Read all lines
with open(path, 'r', encoding='utf-8') as f:
    lines = f.readlines()

print(f'Total lines: {len(lines)}')

fixed = 0
skipped = 0
good_lines = []

for i, line in enumerate(lines, 1):
    orig_line = line.rstrip()
    
    # Try parsing as-is first
    try:
        data = json.loads(orig_line)
        good_lines.append(json.dumps(data) + '\n')
        continue
    except json.JSONDecodeError:
        pass
    
    # Fix pattern: "}extra stuff
    # Valid JSON ends with } - check if there's garbage after
    brace_count = 0
    in_string = False
    escape_next = False
    end_pos = -1
    
    for j, c in enumerate(orig_line):
        if escape_next:
            escape_next = False
            continue
        if c == '\\':
            escape_next = True
            continue
        if c == '"' and not escape_next:
            in_string = not in_string
            continue
        if not in_string:
            if c == '{':
                brace_count += 1
            elif c == '}':
                brace_count -= 1
                if brace_count == 0:
                    end_pos = j
                    break
    
    if end_pos > 0 and end_pos < len(orig_line) - 1:
        # There's extra stuff after the JSON closes
        trimmed = orig_line[:end_pos + 1]
        try:
            data = json.loads(trimmed)
            good_lines.append(json.dumps(data) + '\n')
            fixed += 1
            print(f'  Fixed line {i}: trimmed {len(orig_line) - end_pos - 1} trailing chars')
            continue
        except json.JSONDecodeError:
            pass
    
    # Check for nested JSON object (line 714 pattern)
    if '{"text":' in orig_line[15:] or '{"content":' in orig_line[15:]:
        skipped += 1
        print(f'  Skipping line {i}: nested/malformed JSON object')
        continue
    
    skipped += 1
    print(f'  Skipping line {i}: could not parse or fix')

print(f'\nResults:')
print(f'  Good lines (unchanged): {len(good_lines) - fixed}')
print(f'  Fixed lines: {fixed}')
print(f'  Skipped lines: {skipped}')
print(f'  Total output: {len(good_lines)}')

# Backup original and write cleaned file
import shutil
backup_path = path + '.backup'
shutil.copy(path, backup_path)
print(f'\nBacked up original to: {backup_path}')

with open(out_path, 'w', encoding='utf-8') as f:
    f.writelines(good_lines)
print(f'Wrote cleaned file to: {out_path}')

# Verify the cleaned file
print('\nVerifying cleaned file...')
with open(out_path, 'r', encoding='utf-8') as f:
    valid = 0
    invalid = 0
    for line in f:
        try:
            json.loads(line)
            valid += 1
        except:
            invalid += 1
print(f'  Valid entries: {valid}')
print(f'  Invalid entries: {invalid}')

if invalid == 0:
    print('\n✅ All entries valid! Safe to replace original.')
    # Replace original with clean version
    shutil.copy(out_path, path)
    print(f'Replaced {path} with cleaned version')
else:
    print('\n❌ Some entries still invalid - manual review needed')
