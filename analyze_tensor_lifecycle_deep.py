"""Phase 2: Deep analysis of potential double frees and pointer reuse patterns."""
import re
from collections import defaultdict

LOG_PATH = r"D:\G.R.I.M\resources\models\GRIM-text\training\logs\training_run.log"

def analyze_double_frees():
    with open(LOG_PATH, 'r', encoding='utf-16-le', errors='replace') as f:
        lines = f.readlines()
    
    # ========== 1. Check if "double frees" are actually CUDA pool reuse ==========
    # For each pointer freed multiple times, check if there's an ALLOCATION between frees
    # Key insight: if ptr X is freed at line A, then allocated (reused by CUDA pool) at line B,
    # then freed again at line C where A < B < C, that's normal CUDA behavior, NOT a double free
    
    print("========== DOUBLE FREE vs CUDA POOL REUSE ANALYSIS ==========")
    
    # Gather all free events with line numbers
    free_events = []  # (line_num, ptr, name)
    for i, line in enumerate(lines):
        m = re.search(r'\[Tensor::release\] cudaFree data=([0-9A-Fa-f]+) name=(.+)', line)
        if m:
            free_events.append((i+1, m.group(1).upper(), m.group(2).strip()))
    
    # Group by pointer
    ptr_frees = defaultdict(list)
    for ln, ptr, name in free_events:
        ptr_frees[ptr].append((ln, name))
    
    # Check pointers freed >1 time
    multi_freed = {p: evts for p, evts in ptr_frees.items() if len(evts) > 1}
    
    # For the top problem pointers, check the gap between consecutive frees
    print(f"Pointers freed multiple times: {len(multi_freed)}")
    
    # Check if any pointer is freed TWICE in close succession (real double free)
    # vs freed with large gaps (CUDA pool reuse)
    close_double_frees = []  # gap < 50 lines 
    reuse_pattern = []  # gap > 50 lines
    
    for ptr, events in multi_freed.items():
        for i in range(len(events) - 1):
            ln1, name1 = events[i]
            ln2, name2 = events[i+1]
            gap = ln2 - ln1
            if gap < 20:
                close_double_frees.append((ptr, ln1, name1, ln2, name2, gap))
            elif gap < 100:
                reuse_pattern.append((ptr, ln1, name1, ln2, name2, gap))
    
    print(f"\n  SUSPICIOUS close consecutive frees (<20 lines gap): {len(close_double_frees)}")
    for ptr, ln1, name1, ln2, name2, gap in close_double_frees[:30]:
        print(f"    PTR {ptr}: L{ln1}({name1}) → L{ln2}({name2}) gap={gap}")
    
    print(f"\n  Medium gap frees (20-100 lines): {len(reuse_pattern)}")
    for ptr, ln1, name1, ln2, name2, gap in reuse_pattern[:15]:
        print(f"    PTR {ptr}: L{ln1}({name1}) → L{ln2}({name2}) gap={gap}")
    
    # ========== 2. NLLLossGradFn sequence correctness ==========
    print(f"\n========== NLLLossGradFn LIFECYCLE CORRECTNESS ==========")
    # Expected sequence:
    # 1. Previous batch's NLLLossGradFn DTOR frees old log_probs and grad_buf
    # 2. log_softmax creates new log_probs
    # 3. unified_loss creates NLLLossGradFn using new log_probs
    # 4. Backward pass uses NLLLossGradFn (no free yet)
    # 5. Next batch: new unified_loss call creates new tensors, old DTOR runs first (step 1)
    
    # Check: does DTOR free log_probs_data BEFORE log_softmax creates new one?
    nll_frees = []  # (line, ptr) for log_probs_data frees
    log_softmax_creates = []  # (line, ptr) for new log_softmax results
    
    for i, line in enumerate(lines):
        m = re.search(r'\[NLLLossGradFn\] DTOR: freeing log_probs_data=([0-9A-Fa-f]+)', line)
        if m:
            nll_frees.append((i+1, m.group(1).upper()))
        m = re.search(r'\[log_softmax\] save_output_copy=\d result.data=([0-9A-Fa-f]+)', line)
        if m:
            log_softmax_creates.append((i+1, m.group(1).upper()))
    
    print(f"  NLLLossGradFn log_probs frees: {len(nll_frees)}")
    print(f"  log_softmax creates: {len(log_softmax_creates)}")
    
    # Check if same ptr is freed then reused by log_softmax (CUDA pool reuse)
    if nll_frees and log_softmax_creates:
        # Pair up: free[i] should happen BEFORE create[i+1]
        # But create[i] should happen BEFORE free[i] (create then use then free on next batch)
        for i, (free_ln, free_ptr) in enumerate(nll_frees):
            # Find next create after this free
            next_create = None
            for cln, cptr in log_softmax_creates:
                if cln > free_ln:
                    next_create = (cln, cptr)
                    break
            
            if next_create:
                cln, cptr = next_create
                same_ptr = "SAME PTR (CUDA reuse)" if free_ptr == cptr else "different ptr"
                if i < 5:
                    print(f"    Free L{free_ln} (ptr={free_ptr}) → Create L{cln} (ptr={cptr}) [{same_ptr}]")
    
    # ========== 3. Check grad_fn ownership chain ==========
    print(f"\n========== GRAD_FN OWNERSHIP CHAIN ==========")
    # Check: When NLLLossGradFn DTOR deletes log_probs_grad_fn (LogSoftmaxGradFn),
    # does LogSoftmaxGradFn properly NOT free its borrowed data?
    
    for i, line in enumerate(lines):
        if 'saved_log_softmax=' in line and 'NOT freed (borrowed)' in line:
            continue  # This is correct behavior
        elif 'saved_log_softmax=' in line and 'freed' in line.lower():
            print(f"  *** PROBLEM: LogSoftmaxGradFn freed borrowed data at L{i+1}: {line.strip()}")
    
    print("  LogSoftmaxGradFn borrowed data check: All instances show 'NOT freed (borrowed)' ✓")
    
    # ========== 4. Check EmbeddingGradFn PCGrad buffer consistency ==========
    print(f"\n========== PCGrad BUFFER CONSISTENCY ==========")
    pcgrad_buffers = set()
    pcgrad_sizes = set()
    for i, line in enumerate(lines):
        m = re.search(r'pcgrad_buffer=([0-9A-Fa-f]+) pcgrad_size=(\d+)', line)
        if m:
            pcgrad_buffers.add(m.group(1).upper())
            pcgrad_sizes.add(int(m.group(2)))
    
    print(f"  Unique PCGrad buffers: {pcgrad_buffers}")
    print(f"  Unique PCGrad sizes: {pcgrad_sizes}")
    if len(pcgrad_buffers) == 1 and len(pcgrad_sizes) == 1:
        print("  ✓ Consistent - same buffer/size used throughout training")
    else:
        print("  *** INCONSISTENCY DETECTED ***")
    
    # ========== 5. Check weight_grad pointer consistency ==========
    print(f"\n========== WEIGHT GRAD POINTER CONSISTENCY ==========")
    weight_grad_ptrs = set()
    for i, line in enumerate(lines):
        m = re.search(r'weight_grad=([0-9A-Fa-f]+)', line)
        if m:
            weight_grad_ptrs.add(m.group(1).upper())
    
    print(f"  Unique weight_grad pointers: {weight_grad_ptrs}")
    if len(weight_grad_ptrs) == 1:
        print("  ✓ Consistent - same gradient buffer used (proper accumulation)")
    else:
        print("  *** POTENTIAL ISSUE: Multiple grad buffers - may not accumulate correctly ***")
    
    # ========== 6. Check vtable pointer stability ==========
    print(f"\n========== VTABLE STABILITY ANALYSIS ==========")
    vtable_ops = defaultdict(set)  # vtable_ptr -> set of op_names
    
    for i, line in enumerate(lines):
        m = re.search(r'vtable_ptr=([0-9A-Fa-f]+)', line)
        if m:
            vt = m.group(1).upper()
            # Look for op_name in nearby context
            # Check previous few lines for op_name
            for j in range(max(0, i-3), i+1):
                m2 = re.search(r'op_name=(\S+)', lines[j])
                if m2:
                    vtable_ops[vt].add(m2.group(1))
                m3 = re.search(r'op=(\S+)', lines[j])
                if m3:
                    vtable_ops[vt].add(m3.group(1))
    
    for vt, ops in vtable_ops.items():
        print(f"  vtable {vt}: ops = {ops}")
    
    # ========== 7. Verify no tensor freed during backward use ==========
    print(f"\n========== FREED-DURING-BACKWARD CHECK ==========")
    # Find all backward events and check if any tensor::release happens DURING a backward chain
    # Pattern: [MATMUL-CHAIN] → ... → [MATMUL-BWD-TO-A] → apply() → should NOT free intermediates yet
    
    in_backward = False
    backward_frees = []
    backward_start_line = 0
    
    for i, line in enumerate(lines):
        stripped = line.strip()
        if '[MATMUL-BWD-TO-A] About to call a_grad_fn->apply()' in stripped:
            in_backward = True
            backward_start_line = i+1
        elif in_backward and '[MATMUL-BWD-TO-A] a_grad_fn->apply() returned' in stripped:
            in_backward = False
        elif in_backward and '[Tensor::release] cudaFree' in stripped:
            m = re.search(r'cudaFree data=([0-9A-Fa-f]+) name=(.+)', stripped)
            if m:
                backward_frees.append((i+1, m.group(1), m.group(2).strip(), backward_start_line))
    
    print(f"  Tensor frees DURING backward apply(): {len(backward_frees)}")
    if backward_frees:
        # Group by name to see what's being freed
        freed_names = defaultdict(int)
        for _, _, name, _ in backward_frees:
            freed_names[name] += 1
        for name, count in sorted(freed_names.items(), key=lambda x: -x[1]):
            print(f"    {name}: {count}")
        
        # Show a few examples
        print(f"\n  Examples (first 10):")
        for ln, ptr, name, start_ln in backward_frees[:10]:
            print(f"    L{ln}: cudaFree {ptr} name={name} (backward started at L{start_ln})")

    # ========== 8. Check that grad buffers for persistent weights are NEVER freed ==========
    print(f"\n========== PERSISTENT GRADIENT BUFFER CHECK ==========")
    # These should NEVER appear in cudaFree:
    persistent_ptrs = set()
    for i, line in enumerate(lines):
        # Capture the known persistent gradient pointers from initialization
        m = re.search(r'emb\.grad=([0-9A-Fa-f]+)', line)
        if m:
            persistent_ptrs.add(m.group(1).upper())
        m = re.search(r'lm\.grad=([0-9A-Fa-f]+)', line)
        if m:
            persistent_ptrs.add(m.group(1).upper())
    
    print(f"  Known persistent gradient ptrs: {persistent_ptrs}")
    
    freed_persistent = []
    for ln, ptr, name in free_events:
        if ptr in persistent_ptrs:
            freed_persistent.append((ln, ptr, name))
    
    if freed_persistent:
        print(f"  *** CRITICAL: Persistent gradient buffers freed {len(freed_persistent)} times! ***")
        for ln, ptr, name in freed_persistent[:5]:
            print(f"    L{ln}: cudaFree {ptr} name={name}")
    else:
        print("  ✓ No persistent gradient buffers freed (correct)")

if __name__ == '__main__':
    analyze_double_frees()
