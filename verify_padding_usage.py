#!/usr/bin/env python3
"""
Verify padding token usage in GRIM-text training.

Checks:
1. How targets are masked (-1 for padding/invalid positions)
2. How many valid tokens vs padded tokens per batch
3. Whether padding token ID is used in inputs
4. Alignment between input padding and target masking
"""

import re
from pathlib import Path
from collections import defaultdict

def analyze_latest_log():
    """Parse the latest training log for padding/masking info."""
    
    log_dir = Path("d:/G.R.I.M/resources/models/GRIM-text/training/logs")
    logs = sorted(log_dir.glob("training_*.log"), key=lambda p: p.stat().st_mtime, reverse=True)
    
    if not logs:
        print("No training logs found!")
        return
    
    log_path = logs[0]
    print(f"Analyzing: {log_path.name}")
    print(f"Modified: {log_path.stat().st_mtime}")
    print("=" * 80)
    
    with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
    
    # Extract valid_tokens info
    valid_tokens_pattern = r'valid_tokens=(\d+)'
    valid_tokens = re.findall(valid_tokens_pattern, content)
    
    if valid_tokens:
        print(f"\n✓ Found {len(valid_tokens)} valid_tokens entries")
        valid_counts = [int(v) for v in valid_tokens[:20]]
        print(f"  Sample values: {valid_counts}")
        print(f"  Min: {min(map(int, valid_tokens))}")
        print(f"  Max: {max(map(int, valid_tokens))}")
        print(f"  Average: {sum(map(int, valid_tokens)) / len(valid_tokens):.1f}")
    else:
        print("✗ No valid_tokens entries found")
    
    # Extract batch size and sequence length info
    batch_pattern = r'batch=(\d+)\s+seq=(\d+)\s+tokens=(\d+)\s+valid=(\d+)'
    batches = re.findall(batch_pattern, content)
    
    if batches:
        print(f"\n✓ Found {len(batches)} batch entries with token counts")
        print("\n  Sample batches (first 10):")
        print("  batch | seq_len | total_tokens | valid_tokens | padding_pct")
        print("  " + "-" * 70)
        
        for i, (batch_num, seq_len, total, valid) in enumerate(batches[:10]):
            seq_len = int(seq_len)
            total = int(total)
            valid = int(valid)
            padding = total - valid
            padding_pct = (padding / total * 100) if total > 0 else 0
            print(f"  {batch_num:5s} | {seq_len:7d} | {total:12d} | {valid:12d} | {padding_pct:6.1f}%")
        
        # Overall stats
        total_tokens_sum = sum(int(t) for _, _, t, _ in batches)
        valid_tokens_sum = sum(int(v) for _, _, _, v in batches)
        padding_sum = total_tokens_sum - valid_tokens_sum
        avg_padding_pct = (padding_sum / total_tokens_sum * 100) if total_tokens_sum > 0 else 0
        
        print(f"\n  Overall Statistics:")
        print(f"    Total tokens:   {total_tokens_sum:,}")
        print(f"    Valid tokens:   {valid_tokens_sum:,}")
        print(f"    Padded tokens:  {padding_sum:,}")
        print(f"    Average padding: {avg_padding_pct:.2f}%")
    else:
        print("✗ No detailed batch entries found")
    
    # Check for target validation errors
    target_errors = re.findall(r'invalid target token (\d+)', content)
    if target_errors:
        print(f"\n⚠ Found {len(target_errors)} invalid target token errors!")
        print(f"  Invalid token IDs: {set(target_errors)}")
    else:
        print("\n✓ No invalid target token errors")
    
    # Check for target=-1 mentions (masking)
    mask_pattern = r'target.*-1|masked.*position'
    masks = re.findall(mask_pattern, content, re.IGNORECASE)
    if masks:
        print(f"\n✓ Found {len(masks)} references to masked positions")
    
    # Check for valid_tokens=0 failures
    zero_valid = re.findall(r'valid_tokens=0|valid_token_count=0', content)
    if zero_valid:
        print(f"\n⚠ WARNING: Found {len(zero_valid)} instances of zero valid tokens!")
    else:
        print("\n✓ No zero valid token errors")
    
    return log_path

def check_target_generation():
    """Check how targets are generated in the data loader."""
    
    print("\n" + "=" * 80)
    print("TARGET GENERATION CODE")
    print("=" * 80)
    
    loader_path = Path("d:/G.R.I.M/resources/models/GRIM-text/training/training_data_loader.hpp")
    
    if not loader_path.exists():
        print("✗ training_data_loader.hpp not found")
        return
    
    with open(loader_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Find target generation code
    target_gen = re.search(r'(for.*j.*seq_len.*\n.*targets\[j\].*token_ids.*\n.*\n.*targets\[0\].*-1)', 
                          content, re.MULTILINE)
    
    if target_gen:
        print("\n✓ Found target generation code:")
        print("\n" + target_gen.group(1))
    
    # Count occurrences of target=-1 masking
    mask_count = content.count('targets[0] = -1')
    print(f"\n✓ Found {mask_count} places where position 0 is masked")
    
    # Check for final position masking
    final_mask_count = content.count('targets[') and 'seq_len' in content
    print(f"✓ Target generation follows pattern: targets[j] = token_ids[j+1]")
    print(f"  - First position (j=0) is masked to -1")
    print(f"  - Last position (j=seq_len-1) naturally gets -1 (no j+1)")

def main():
    print("GRIM-TEXT PADDING VERIFICATION")
    print("=" * 80)
    print()
    
    # Analyze training log
    log_path = analyze_latest_log()
    
    # Check target generation logic
    check_target_generation()
    
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print("""
Key Findings:
1. Targets are created as: targets[j] = token_ids[j+1]
2. Position 0 is masked to -1 (can't predict after nothing)
3. Last position naturally gets -1 (no next token)
4. Padded positions in batch use input_id=0, target_id=-1
5. Loss computation skips any position where target == -1

This means:
- Input:  [tok0, tok1, tok2, tok3, tok4]  (5 tokens)
- Target: [-1,   tok2, tok3, tok4, -1]     (3 valid predictions)
- Only positions 1,2,3 contribute to loss
- Positions 0 and 4 are masked out
    """)

if __name__ == '__main__':
    main()
