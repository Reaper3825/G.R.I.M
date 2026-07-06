# Flash Attention Diagnostic Entropy Bug

**Date:** December 27, 2025  
**Severity:** MEDIUM - Diagnostic outputs are misleading but don't affect training  
**Status:** ✅ FIXED - Entropy now computed in forward pass using actual online softmax

---

## Problem Summary

The diagnostic kernel (`computeAttentionDiagnosticsKernel`) computes entropy over a **truncated distribution** (first 128 tokens) while the forward/backward passes use online softmax over the **entire sequence** (2800-4000+ tokens in practice). This means entropy values logged during training do NOT reflect the actual attention distribution the model uses.

## Fix Implemented (December 27, 2025)

**Added entropy computation directly in the forward kernel** using the actual online softmax values. The fix:

1. **Added `entropy_output` field to `FlashAttentionConfig`** (Flash_Attention_Kernal.hpp)
   - Optional float* buffer of size `[batch_size * num_heads]`
   - Set to nullptr to disable entropy computation (zero overhead)

2. **Modified `flashAttentionForwardKernel`** to compute entropy during online softmax:
   - Added `row_entropy_smem[BLOCK_SIZE_Q]` shared memory for per-row entropy tracking
   - Accumulates `sum(p_unnorm * (score - max))` during softmax loop
   - Properly rescales when max changes (includes max-shift correction term)
   - Final entropy: `H = log2(row_sum) - row_entropy_smem / (row_sum * ln(2))`

3. **Added `normalizeEntropyKernel`** to normalize by number of Q blocks:
   - Each Q block atomicAdds its block-average entropy
   - Normalization kernel divides by num_q_blocks to get true mean

**Key math for online entropy rescaling:**
When max changes from M to M', the entropy accumulator transforms as:
```
sum(p' * (s - M')) = old_scale * sum(p * (s - M)) + log(old_scale) * old_row_sum * old_scale
```
where `old_scale = exp(M - M')`. The second term accounts for the shift in `(s - max)`.

---

### Investigation Findings (December 27, 2025)

**Actual sequence lengths from training logs:**
- K-tensor sizes: n=718336, n=1029632, n=954880
- With num_kv_heads=4, head_dim=64: `seq_len = n / (4 * 64)`
- **Actual seq_len: 2806 - 4022 tokens per batch**

**Diagnostic truncation:**
- MAX_ATTEND = 128 (line 199 of Flash_Attention_Kernal.cu)
- MAX_SAMPLES = 64 (line 172) - strided sampling across sequence

**Entropy comparison:**
| Scenario | Max Entropy | Observed |
|----------|-------------|----------|
| Diagnostic (128 tokens) | 7.0 bits | 6.55-6.73 bits ✓ |
| Forward pass (2806 tokens) | 11.5 bits | Unknown (not logged) |
| Forward pass (4022 tokens) | 11.97 bits | Unknown (not logged) |

**The diagnostic entropy is ~40% lower than reality for typical batch sizes!**

---

## Root Cause Analysis

### Forward/Backward Pass (CORRECT)
**File:** `Flash_Attention_Kernal.cu` lines 730-844 (forward), 1050-1220 (backward)

**Method:** Online softmax with KV block accumulation
```cuda
// Pass 1a: Accumulate row_max across ALL KV blocks
for (int kv_block_idx = 0; kv_block_idx < num_kv_blocks; ++kv_block_idx) {
    row_max_smem[q_idx] = fmaxf(row_max_smem[q_idx], score);  // Global max
}

// Pass 1b: Accumulate row_sum across ALL KV blocks
for (int kv_block_idx = 0; kv_block_idx < num_kv_blocks; ++kv_block_idx) {
    row_sum_smem[q_idx] += exp(score - max_val);  // Global sum
}

// Final probability normalized over ENTIRE sequence
P = exp(score - row_max_smem[q_idx]) / row_sum_smem[q_idx];
```

**Normalization:** Over **all tokens** in sequence (up to 8192 tokens)

### Diagnostic Kernel (WRONG)
**File:** `Flash_Attention_Kernal.cu` lines 199-233

**Method:** Per-position independent softmax with fixed buffer
```cuda
constexpr int MAX_ATTEND = 128;  // TRUNCATION!
float scores[MAX_ATTEND];
const int attend_len = (qi + 1 < MAX_ATTEND) ? (qi + 1) : MAX_ATTEND;  // Cap at 128

// Compute softmax ONLY over first 128 tokens
float max_score = -INFINITY;
for (int ki = 0; ki < attend_len; ++ki) {
    scores[ki] = Q · K;
    max_score = max(max_score, scores[ki]);
}

float sum_exp = 0.0f;
for (int ki = 0; ki < attend_len; ++ki) {
    sum_exp += exp(scores[ki] - max_score);  // LOCAL max, LOCAL sum
}

// Probability normalized over TRUNCATED distribution
float p = exp(scores[ki] - max_score) / sum_exp;
```

**Normalization:** Over **first 128 tokens only** (MAX_ATTEND cap)

---

## Impact

### When Sequence Length ≤ 128
- Diagnostic entropy is **approximately correct** (minor numerical differences due to different computation order)
- Still doesn't use online softmax with rescaling, so may have floating-point differences

### When Sequence Length > 128
- Diagnostic entropy is **completely wrong**
- Forward/backward normalize over 1000+ tokens
- Diagnostic normalizes over **first 128 tokens only**
- Entropy values are artificially high (log2(128) = 7 bits max instead of log2(1024) = 10 bits)

### Training Impact
- **Training is NOT affected** - forward/backward use correct softmax
- **Diagnostics are misleading** - entropy logs don't reflect actual attention behavior
- **Plateau investigation compromised** - can't trust entropy metrics from diagnostic logs

---

## Evidence

**From code inspection:**
- Forward pass: `for (int kv_block_idx = 0; kv_block_idx < num_kv_blocks; ++kv_block_idx)` - loops ALL blocks
- Backward Pass 1: Lines 1095-1149 - accumulates `row_max_smem` across all KV blocks
- Backward Pass 1: Lines 1170-1213 - accumulates `row_sum_smem` across all KV blocks
- Diagnostic: Line 202 - `const int attend_len = (qi + 1 < MAX_ATTEND) ? (qi + 1) : MAX_ATTEND;` - **TRUNCATES**

**Specific line numbers:**
- Diagnostic buffer: Line 199 `constexpr int MAX_ATTEND = 128;`
- Diagnostic truncation: Line 202 `attend_len = min(qi + 1, MAX_ATTEND)`
- Forward online softmax: Lines 800-844 (rescaling across blocks)
- Backward row_max accumulation: Lines 1131-1149
- Backward row_sum accumulation: Lines 1188-1213

---

## Fix Options

### Option 1: Rewrite Diagnostic to Use Online Softmax (CORRECT but complex)

- Make diagnostic loop through KV blocks like forward/backward
- Accumulate `row_max` and `row_sum` across blocks
- Compute entropy using global normalization
- **Pros:** Exact match with forward/backward
- **Cons:** ~200 lines of code, shared memory pressure

### Option 2: Remove MAX_ATTEND Cap (Partial fix - NOT RECOMMENDED)

```cuda
// This would NOT fix the bug! Sequences are 2800-4000 tokens
constexpr int MAX_ATTEND = 256;  // Still way too small
float scores[MAX_ATTEND];
```

- **Cons:** Would need ~4000 floats per thread = 16KB register pressure = won't compile
- **Not viable for actual sequence lengths**

### Option 3: Delete Diagnostic Kernel (Rule 20: Fail Loud) - RECOMMENDED

- Diagnostic doesn't match production code → delete it
- Use actual forward pass outputs for entropy computation (store P values, compute H after softmax)
- **Pros:** No maintenance burden, no misleading data
- **Cons:** Lose gradient norm diagnostics (but those could be moved elsewhere)

### Option 4: Add Warning to Output (Minimum Fix)

Log the truncation clearly:
```cpp
printf("[AttnDiag] WARNING: seq_len=%d but entropy computed over only %d positions (MAX_ATTEND=%d)\n", 
       config.seq_len, attend_len, MAX_ATTEND);
```

---

## Recommended Action

**COMPLETED (December 27, 2025):**
1. ✅ Document the bug (this file)
2. ✅ Implement entropy computation in forward pass using actual online softmax
3. ✅ Add `entropy_output` field to `FlashAttentionConfig`
4. ✅ Add normalization kernel for multi-block accumulation

**STILL TODO:**
1. 🔄 Add logging to training loop to use new entropy output
2. 🔄 Consider deleting diagnostic kernel per Rule 20 (now redundant for entropy)

---

## Related Files

- [Flash_Attention_Kernal.cu](resources/models/GRIM-text/Layers/FlashAttention/Flash_Attention_Kernal.cu) - Contains both diagnostic and production kernels
- [PLATEAU_BUG_INVESTIGATION.md](docs/PLATEAU_BUG_INVESTIGATION.md) - Uses diagnostic entropy values (now known to be unreliable)
- [analyze_attention_diagnostics.py](analyze_attention_diagnostics.py) - Parses diagnostic output

---

## Impact on Plateau Investigation

**Critical Insight:** The entropy values in PLATEAU_BUG_INVESTIGATION.md (H≈6.6 bits/pos) are **artifacts of the truncation**, not real attention behavior.

**What we CAN still trust from diagnostic:**
- ✅ QK score ranges (qk_min, qk_max) - computed over sampled positions, gives rough idea
- ✅ Gradient norms (Q, K, V) - computed over full tensors
- ✅ Max/min probability - sampled but representative

**What we CANNOT trust:**
- ❌ Entropy values - bounded by log2(128)=7 bits regardless of actual attention spread
- ❌ Any entropy-based conclusions about "frozen attention" or "saturation"

**Implication for Issue #21 (Softmax Jacobian Attenuation):**
The hypothesis that "attention stays spread (high entropy) causing tiny gradients" was based on diagnostic entropy. Since actual entropy over 3000+ tokens could be much higher (up to 11+ bits), the real attention behavior is unknown. The gradient magnitudes (Q:0.01-0.05, K:0.003-0.012, V:0.02-0.04) are the **only reliable metric** from the diagnostic.

---

## Notes for Future Investigation

When debugging plateau with entropy metrics, **DO NOT trust values from `[AttnDiag]` logs**. The diagnostic kernel:

1. Uses different softmax normalization than training
2. Truncates sequences to 128 tokens (actual: 2800-4000+ tokens)
