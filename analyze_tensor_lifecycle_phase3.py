"""Phase 3: Analyze the 'close consecutive free' patterns - are they real double frees?"""
import re
from collections import defaultdict, Counter

LOG_PATH = r"D:\G.R.I.M\resources\models\GRIM-text\training\logs\training_run.log"

def analyze():
    with open(LOG_PATH, 'r', encoding='utf-16-le', errors='replace') as f:
        lines = f.readlines()
    
    # ========== 1. Examine the gap=3 patterns (unnamed → qkv_split_Q_bsm) ==========
    # These are the most suspicious - same pointer freed as "unnamed" then "qkv_split_Q_bsm" 3 lines later
    print("========== GAP=3 PATTERN: unnamed → qkv_split_*_bsm ==========")
    
    # Take first instance and show full context
    gap3_examples = []
    free_events = []
    for i, line in enumerate(lines):
        m = re.search(r'\[Tensor::release\] cudaFree data=([0-9A-Fa-f]+) name=(.+)', line)
        if m:
            free_events.append((i, m.group(1).upper(), m.group(2).strip()))
    
    # Find consecutive same-pointer frees
    for idx in range(len(free_events) - 1):
        ln1, ptr1, name1 = free_events[idx]
        ln2, ptr2, name2 = free_events[idx+1]
        if ptr1 == ptr2 and (ln2 - ln1) <= 5:
            gap3_examples.append((ln1, ln2, ptr1, name1, name2))
    
    print(f"Total same-pointer consecutive frees within 5 lines: {len(gap3_examples)}")
    
    # Categorize by pattern
    pattern_counts = Counter((n1, n2) for _, _, _, n1, n2 in gap3_examples)
    print(f"\nPatterns:")
    for (n1, n2), count in pattern_counts.most_common():
        print(f"  {n1} → {n2}: {count}")
    
    # Show full context around first few gap=3 examples
    print(f"\n--- Context around FIRST 3 examples of unnamed→qkv_split_Q_bsm ---")
    shown = 0
    for ln1, ln2, ptr, name1, name2 in gap3_examples:
        if name1 == 'unnamed' and 'qkv_split' in name2:
            if shown >= 3:
                break
            print(f"\n  Example {shown+1}: PTR={ptr}")
            start = max(0, ln1 - 3)
            end = min(len(lines), ln2 + 3)
            for j in range(start, end):
                marker = ">>>" if j == ln1 or j == ln2 else "   "
                text = lines[j].strip()[:140]
                print(f"  {marker} L{j+1}: {text}")
            shown += 1
    
    # Show context around matmul_result → layer_scale_result
    print(f"\n--- Context around FIRST 3 examples of matmul_result→layer_scale_result ---")
    shown = 0
    for ln1, ln2, ptr, name1, name2 in gap3_examples:
        if name1 == 'matmul_result' and name2 == 'layer_scale_result':
            if shown >= 3:
                break
            print(f"\n  Example {shown+1}: PTR={ptr} gap={ln2-ln1}")
            start = max(0, ln1 - 2)
            end = min(len(lines), ln2 + 2)
            for j in range(start, end):
                marker = ">>>" if j == ln1 or j == ln2 else "   "
                text = lines[j].strip()[:150]
                print(f"  {marker} L{j+1}: {text}")
            shown += 1
    
    # ========== 2. Check the backward pass - are tensors freed BEFORE being used? ==========
    print(f"\n========== BACKWARD PASS FLOW ANALYSIS (last batch) ==========")
    
    # Find the last ComputeLossBatch STEP-J (end of loss computation, start of backward)
    last_stepj = None
    for i, line in enumerate(lines):
        if '[ComputeLossBatch] STEP-J:' in line:
            last_stepj = i
    
    if last_stepj:
        print(f"Last batch loss returned at L{last_stepj+1}")
        
        # Track the backward flow
        print(f"\nBackward flow (showing MATMUL-BWD/CHAIN/EMBED/release events):")
        event_count = 0
        for i in range(last_stepj, min(last_stepj + 800, len(lines))):
            line = lines[i].strip()
            if any(m in line for m in ['[MATMUL-BWD', '[MATMUL-CHAIN]', '[EmbeddingGradFn',
                                        '[Tensor::release]', 'DTOR', 'SplitQKV', 'ReshapeBHSD',
                                        '[RMS', 'ScaledDotProduct', 'FlashAttn', 'LayerScale',
                                        'BiasAdd', 'broadcast_add']):
                text = line[:160]
                print(f"  L{i+1}: {text}")
                event_count += 1
                if event_count > 200:
                    print(f"  ... (truncated)")
                    break
    
    # ========== 3. Track which grad_fn types are being freed vs recycled ==========
    print(f"\n========== GRAD_FN LIFECYCLE ==========")
    
    # Count how many unique grad_fn pointers we see
    gradfn_ptrs = defaultdict(list)  # ptr -> [(line, event_type, op_name)]
    for i, line in enumerate(lines):
        m = re.search(r'\[Tensor::release\] deleting grad_fn=([0-9A-Fa-f]+) op=(\S+)', line)
        if m:
            gradfn_ptrs[m.group(1).upper()].append((i+1, 'delete', m.group(2)))
    
    print(f"Total grad_fn pointers deleted: {len(gradfn_ptrs)}")
    multi_delete = {p: evts for p, evts in gradfn_ptrs.items() if len(evts) > 1}
    print(f"Grad_fn pointers deleted multiple times: {len(multi_delete)}")
    
    if multi_delete:
        print(f"\n  Examples of multi-deleted grad_fn ptrs:")
        for ptr, evts in list(multi_delete.items())[:5]:
            print(f"    PTR {ptr}:")
            for ln, etype, op in evts:
                print(f"      L{ln}: {etype} op={op}")
    
    # ========== 4. Check batch-to-batch tensor count consistency ==========
    print(f"\n========== BATCH TENSOR COUNTS ==========")
    
    # Find all ComputeLossBatch STEP-A positions
    batch_starts = []
    for i, line in enumerate(lines):
        if '[ComputeLossBatch] STEP-A' in line:
            batch_starts.append(i)
    
    if len(batch_starts) >= 2:
        print(f"Batches found: {len(batch_starts)}")
        
        for b_idx in range(min(3, len(batch_starts) - 1)):
            start = batch_starts[b_idx]
            end = batch_starts[b_idx + 1] if b_idx + 1 < len(batch_starts) else len(lines)
            
            batch_frees = 0
            batch_grad_fn_deletes = 0
            for i in range(start, end):
                if '[Tensor::release] cudaFree' in lines[i]:
                    batch_frees += 1
                if 'deleting grad_fn=' in lines[i]:
                    batch_grad_fn_deletes += 1
            
            print(f"  Batch {b_idx}: L{start+1}-L{end}: frees={batch_frees}, grad_fn_deletes={batch_grad_fn_deletes}")
        
        # Also check last batch
        if len(batch_starts) >= 2:
            start = batch_starts[-1]
            batch_frees = sum(1 for i in range(start, len(lines)) if '[Tensor::release] cudaFree' in lines[i])
            batch_grad_fn_deletes = sum(1 for i in range(start, len(lines)) if 'deleting grad_fn=' in lines[i])
            print(f"  Last batch: L{start+1}-EOF: frees={batch_frees}, grad_fn_deletes={batch_grad_fn_deletes}")
    
    # ========== 5. Verify matmul backward grad accumulation correctness ==========
    print(f"\n========== MATMUL BACKWARD GRAD ACCUMULATION ==========")
    # Check: LM_HEAD backward should call applyLmHeadGradCorrections
    # Then chain to center_rows → EmbeddingGradFn (PCGrad)
    
    lm_head_bwd = []
    for i, line in enumerate(lines):
        if '[MATMUL-BWD-IN]' in line and 'LM_HEAD' in line:
            lm_head_bwd.append(i+1)
    
    print(f"LM_HEAD backward calls: {len(lm_head_bwd)}")
    
    # For each LM_HEAD backward, verify the chain
    chains_ok = 0
    chains_bad = 0
    for start_line_num in lm_head_bwd[-3:]:  # Check last 3
        start_idx = start_line_num - 1
        found_corrections = False
        found_center_rows = False
        found_embedding = False
        
        for j in range(start_idx, min(start_idx + 80, len(lines))):
            line = lines[j]
            if 'applyLmHeadGradCorrections' in line:
                found_corrections = True
            if 'center_rows' in line:
                found_center_rows = True
            if 'EmbeddingGradFn' in line:
                found_embedding = True
            if '[MATMUL-BWD-IN]' in line and j > start_idx + 1:
                break  # Next matmul backward
        
        if found_corrections and found_center_rows and found_embedding:
            chains_ok += 1
        else:
            chains_bad += 1
            print(f"  BROKEN chain at L{start_line_num}: corrections={found_corrections} center_rows={found_center_rows} embedding={found_embedding}")
    
    if chains_ok > 0:
        print(f"  ✓ Last {chains_ok} LM_HEAD backward chains verified correct")
    if chains_bad > 0:
        print(f"  *** {chains_bad} BROKEN chains ***")

if __name__ == '__main__':
    analyze()
