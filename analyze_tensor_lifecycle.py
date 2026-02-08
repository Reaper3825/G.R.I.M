"""Analyze tensor lifecycle in training_run.log - check for memory issues."""
import re
import sys
from collections import defaultdict, Counter

LOG_PATH = r"D:\G.R.I.M\resources\models\GRIM-text\training\logs\training_run.log"

def analyze():
    with open(LOG_PATH, 'r', encoding='utf-16-le', errors='replace') as f:
        lines = f.readlines()
    
    print(f"Total lines: {len(lines)}")
    
    # ========== 1. Count key markers ==========
    markers = {
        '[Tensor::release] cudaFree': 0,
        '[Tensor::release] deleting grad_fn': 0,
        '[NLLLossGradFn] DTOR': 0,
        '[LogSoftmaxGradFn] DTOR': 0,
        '[MATMUL-BWD-IN]': 0,
        '[MATMUL-BWD-TO-A]': 0,
        '[MATMUL-CHAIN]': 0,
        '[EmbeddingGradFn::apply]': 0,
        '[BIAS-ADD-FWD]': 0,
        '[FA-FWD': 0,
        '[ComputeLossBatch]': 0,
        '[log_softmax]': 0,
        '[unified_loss]': 0,
    }
    
    for line in lines:
        for m in markers:
            if m in line:
                markers[m] += 1
    
    print("\n========== MARKER COUNTS ==========")
    for m, c in sorted(markers.items(), key=lambda x: -x[1]):
        print(f"  {m}: {c}")
    
    # ========== 2. Track cudaFree'd pointers and their names ==========
    # Pattern: [Tensor::release] cudaFree data=XXXX name=YYY
    freed_ptrs = []
    freed_by_name = defaultdict(list)
    
    for i, line in enumerate(lines):
        m = re.search(r'\[Tensor::release\] cudaFree data=([0-9A-Fa-f]+) name=(.+)', line)
        if m:
            ptr = m.group(1).upper()
            name = m.group(2).strip()
            freed_ptrs.append((ptr, name, i+1))
            freed_by_name[name].append(ptr)
    
    print(f"\n========== TENSOR FREES ==========")
    print(f"Total cudaFree calls: {len(freed_ptrs)}")
    
    # Check for double frees (same pointer freed twice)
    ptr_free_counts = Counter(ptr for ptr, name, line in freed_ptrs)
    double_frees = {ptr: count for ptr, count in ptr_free_counts.items() if count > 1}
    
    if double_frees:
        print(f"\n  *** POTENTIAL DOUBLE FREES ({len(double_frees)} ptrs): ***")
        for ptr, count in sorted(double_frees.items(), key=lambda x: -x[1])[:20]:
            occurrences = [(name, ln) for p, name, ln in freed_ptrs if p == ptr]
            print(f"    PTR {ptr} freed {count}x:")
            for name, ln in occurrences:
                print(f"      line {ln}: name={name}")
    else:
        print("  No double frees detected (GOOD)")
    
    # Name frequency
    name_counts = Counter(name for ptr, name, ln in freed_ptrs)
    print(f"\n  Freed tensor names (top 20):")
    for name, count in name_counts.most_common(20):
        print(f"    {name}: {count}")
    
    # ========== 3. Track grad_fn deletions ==========
    # Pattern: [Tensor::release] deleting grad_fn=XXXX op=YYY
    grad_fn_deletes = []
    for i, line in enumerate(lines):
        m = re.search(r'\[Tensor::release\] deleting grad_fn=([0-9A-Fa-f]+) op=(\S+)', line)
        if m:
            ptr = m.group(1).upper()
            op = m.group(2)
            grad_fn_deletes.append((ptr, op, i+1))
    
    print(f"\n========== GRAD_FN DELETIONS ==========")
    print(f"Total grad_fn deletions: {len(grad_fn_deletes)}")
    grad_fn_op_counts = Counter(op for _, op, _ in grad_fn_deletes)
    for op, count in grad_fn_op_counts.most_common():
        print(f"  {op}: {count}")
    
    # ========== 4. NLLLossGradFn DTOR analysis ==========
    print(f"\n========== NLLLossGradFn DTOR ANALYSIS ==========")
    nll_dtors = []
    for i, line in enumerate(lines):
        if '[NLLLossGradFn] DTOR' in line:
            nll_dtors.append((i+1, line.strip()))
    
    print(f"Total NLLLossGradFn DTOR events: {len(nll_dtors)}")
    
    # Group DTOR events into complete sequences
    dtor_sequences = []
    current_seq = []
    for ln, text in nll_dtors:
        if 'DTOR ENTER' in text:
            if current_seq:
                dtor_sequences.append(current_seq)
            current_seq = [(ln, text)]
        else:
            current_seq.append((ln, text))
    if current_seq:
        dtor_sequences.append(current_seq)
    
    print(f"Complete DTOR sequences: {len(dtor_sequences)}")
    
    # Check each sequence for proper cleanup
    for seq_idx, seq in enumerate(dtor_sequences[:5]):  # First 5
        print(f"\n  Sequence {seq_idx+1}:")
        for ln, text in seq:
            # Extract key details
            short = text
            if len(short) > 120:
                short = short[:120] + "..."
            print(f"    L{ln}: {short}")
    
    # ========== 5. LogSoftmaxGradFn DTOR analysis ==========
    print(f"\n========== LogSoftmaxGradFn DTOR ANALYSIS ==========")
    lsm_dtors = []
    for i, line in enumerate(lines):
        if '[LogSoftmaxGradFn] DTOR' in line:
            lsm_dtors.append((i+1, line.strip()))
    
    print(f"Total LogSoftmaxGradFn DTOR events: {len(lsm_dtors)}")
    for ln, text in lsm_dtors[:10]:
        short = text if len(text) <= 120 else text[:120] + "..."
        print(f"  L{ln}: {short}")
    
    # ========== 6. Forward-Backward lifecycle per batch ==========
    # Find batch boundaries using ComputeLossBatch
    print(f"\n========== FORWARD-BACKWARD LIFECYCLE ==========")
    
    # Find all ComputeLossBatch STEP-A (start of loss computation)
    loss_starts = []
    for i, line in enumerate(lines):
        if '[ComputeLossBatch] STEP-A' in line:
            loss_starts.append(i+1)
    
    print(f"Loss computations found: {len(loss_starts)}")
    
    # ========== 7. MATMUL-CHAIN autograd analysis ==========
    print(f"\n========== MATMUL-CHAIN AUTOGRAD ANALYSIS ==========")
    chain_events = []
    for i, line in enumerate(lines):
        if '[MATMUL-CHAIN]' in line or '[MATMUL-BWD-TO-A]' in line or '[MATMUL-BWD-IN]' in line:
            chain_events.append((i+1, line.strip()))
    
    print(f"Total MATMUL events: {len(chain_events)}")
    
    # Check the last batch's backward pass in detail
    # Find the LAST ComputeLossBatch STEP-A
    if loss_starts:
        last_loss = loss_starts[-1]
        print(f"\nLast batch loss at line {last_loss}")
        
        # Get all events after this point
        backward_events = []
        for i, line in enumerate(lines[last_loss-1:], start=last_loss):
            stripped = line.strip()
            if any(marker in stripped for marker in [
                '[MATMUL-BWD', '[MATMUL-CHAIN]', '[EmbeddingGradFn',
                '[Tensor::release]', '[NLLLossGradFn]', '[LogSoftmaxGradFn]',
                '[ComputeLossBatch]', '[log_softmax]', '[unified_loss]',
                'DTOR', 'grad_fn', 'cudaFree'
            ]):
                backward_events.append((i, stripped))
        
        print(f"Events after last loss computation: {len(backward_events)}")
        print("\nLast batch lifecycle (first 100 events):")
        for ln, text in backward_events[:100]:
            short = text if len(text) <= 140 else text[:140] + "..."
            print(f"  L{ln}: {short}")
    
    # ========== 8. EmbeddingGradFn analysis ==========
    print(f"\n========== EMBEDDING GRAD FN ANALYSIS ==========")
    emb_events = []
    for i, line in enumerate(lines):
        if '[EmbeddingGradFn' in line:
            emb_events.append((i+1, line.strip()))
    
    print(f"Total EmbeddingGradFn events: {len(emb_events)}")
    
    # Look for last PCGrad application
    pcgrad_events = [e for e in emb_events if 'PCGrad' in e[1] or 'PHASE7' in e[1] or 'PHASE8' in e[1]]
    print(f"PCGrad-related events: {len(pcgrad_events)}")
    for ln, text in pcgrad_events[-10:]:
        short = text if len(text) <= 140 else text[:140] + "..."
        print(f"  L{ln}: {short}")
    
    # ========== 9. Check for "unnamed" tensors being freed (potential leaks) ==========
    print(f"\n========== UNNAMED TENSOR ANALYSIS ==========")
    unnamed_frees = [(ptr, ln) for ptr, name, ln in freed_ptrs if name == 'unnamed']
    named_frees = [(ptr, name, ln) for ptr, name, ln in freed_ptrs if name != 'unnamed']
    print(f"Named tensor frees: {len(named_frees)}")
    print(f"Unnamed tensor frees: {len(unnamed_frees)}")
    print(f"Ratio unnamed/total: {len(unnamed_frees)/max(1,len(freed_ptrs))*100:.1f}%")
    
    # ========== 10. Check BIAS-ADD-FWD call numbers for sequential order ==========
    print(f"\n========== BIAS-ADD-FWD SEQUENCE CHECK ==========")
    bias_calls = []
    for i, line in enumerate(lines):
        m = re.search(r'\[BIAS-ADD-FWD\] call=(\d+)', line)
        if m:
            bias_calls.append((int(m.group(1)), i+1))
    
    if bias_calls:
        print(f"Total BIAS-ADD-FWD calls: {len(bias_calls)}")
        print(f"Call range: {bias_calls[0][0]} to {bias_calls[-1][0]}")
        
        # Check for gaps or out-of-order
        expected = bias_calls[0][0]
        gaps = []
        for call_num, ln in bias_calls:
            if call_num != expected:
                gaps.append((expected, call_num, ln))
            expected = call_num + 1
        
        if gaps:
            print(f"  Gaps/jumps detected: {len(gaps)}")
            for exp, actual, ln in gaps[:5]:
                print(f"    Expected call={exp}, got call={actual} at line {ln}")
        else:
            print("  Sequential order: OK (no gaps)")
    
    # ========== 11. Verify paired alloc-free for named tensors ==========
    print(f"\n========== PAIRED ALLOC-FREE CHECK (by pointer in single batch) ==========")
    # Find the first and last batch boundaries more precisely
    # Look for patterns where same pointer is allocated (appears in operation output) then freed
    
    # For the last batch, track pointers
    if loss_starts and len(loss_starts) >= 2:
        second_last = loss_starts[-2]
        last = loss_starts[-1]
        batch_lines = lines[second_last-1:last-1]
        
        batch_frees = {}
        batch_allocs = {}
        for i, line in enumerate(batch_lines):
            m = re.search(r'\[Tensor::release\] cudaFree data=([0-9A-Fa-f]+) name=(.+)', line)
            if m:
                ptr = m.group(1).upper()
                name = m.group(2).strip()
                if ptr not in batch_frees:
                    batch_frees[ptr] = []
                batch_frees[ptr].append(name)
        
        print(f"  Pointers freed in second-to-last batch: {len(batch_frees)}")
        multi_free = {p: names for p, names in batch_frees.items() if len(names) > 1}
        print(f"  Pointers freed multiple times in same batch: {len(multi_free)}")
        for ptr, names in list(multi_free.items())[:10]:
            print(f"    {ptr}: {names}")

    # ========== 12. Verify vtable_ptr consistency ==========
    print(f"\n========== VTABLE POINTER CONSISTENCY ==========")
    vtable_ptrs = []
    for i, line in enumerate(lines):
        m = re.search(r'vtable_ptr=([0-9A-Fa-f]+)', line)
        if m:
            vtable_ptrs.append((m.group(1).upper(), i+1))
    
    vtable_unique = Counter(ptr for ptr, ln in vtable_ptrs)
    print(f"Total vtable references: {len(vtable_ptrs)}")
    print(f"Unique vtable pointers: {len(vtable_unique)}")
    for ptr, count in vtable_unique.most_common():
        print(f"  {ptr}: {count} references")

if __name__ == '__main__':
    analyze()
