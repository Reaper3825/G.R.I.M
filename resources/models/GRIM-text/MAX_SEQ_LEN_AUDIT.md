# max_seq_len Usage Audit Report
**Date:** February 6, 2026  
**Auditor:** GitHub Copilot  
**Total Files Scanned:** 200+ matches in GRIM-text codebase

## Executive Summary

**FINDING:** `max_seq_len` is being used correctly in 95% of cases. The parameter serves its intended purpose as a **memory management constant** for buffer allocation. However, there are **TWO CRITICAL MISUSES** in positional bias computation that could cause bugs if runtime sequences exceed the configured max.

**Severity:** Medium (non-critical but architecturally incorrect)

---

## Parameter Definition & Purpose

```cpp
// From HyperParameters_GPU.hpp
struct ModelArchitecture {
    int max_seq_len = DEFAULT_MAX_SEQ_LEN;  // 2048
    // ...
};
```

**INTENDED USE:** Memory allocation upper bound for:
- GPU buffer sizing (embeddings, activations, KV cache)
- Batch construction limits (prevent OOM)
- Tensor shape validation

**NOT FOR:** Runtime algorithmic decisions that depend on actual sequence length

---

## Category 1: CORRECT USES (Memory Management) ✅

### 1.1 Buffer Allocation
**Files:** 15+ locations  
**Examples:**
```cpp
// TrainingState_GPU.hpp - Position embedding buffer
size_t pos_emb_size = max_seq_len * d_model * sizeof(float);

// Phase2_TrainingLoop.cu - Batch construction
size_t required_tokens = batch_inputs.size() * max_seq_len;

// ComputeLossBatch.cu - Padded input buffer sizing
const size_t total_tokens = result.batch_size * result.max_seq_len;
```
**Status:** ✅ CORRECT - These allocate WORST-CASE memory upfront

### 1.2 Capacity Validation
**Files:** Phase1_Startup.cu, ComputeLossBatch.cu, Phase2_TrainingLoop.cu  
**Examples:**
```cpp
// Reject sequences that exceed allocated capacity
if (static_cast<int>(seq.token_ids.size()) > max_seq_len) {
    throw std::runtime_error("Sequence exceeds max_seq_len");
}

// Batch overflow detection
if (max_seq_len > static_cast<size_t>(ts.max_cached_seq_len)) {
    fprintf(stderr, "*** max_seq_len=%zu > max_cached=%d ***\n", 
            max_seq_len, ts.max_cached_seq_len);
}
```
**Status:** ✅ CORRECT - Prevents buffer overruns

### 1.3 Configuration & Serialization
**Files:** Phase1_Startup.cu, Serialization_GPU.cu, grim_model_serialization.cu  
**Examples:**
```cpp
// Phase1_Startup.cu - Copy config value
config.architecture.max_seq_len = config.hyperparameters.max_seq_len;

// Serialization_GPU.cu - Save to FlatBuffer
static_cast<uint32_t>(cfg.max_seq_len)
```
**Status:** ✅ CORRECT - Propagating config parameter

---

## Category 2: DIAGNOSTIC/LOGGING USES ✅

**Files:** 20+ locations  
**Purpose:** Boundary detection, overflow warnings, debug messages

**Examples:**
```cpp
// Phase2_TrainingLoop.cu
diag << "[BOUNDARY_DIAGNOSTIC] *** REACHED model.max_seq_len=" 
     << model_cfg.max_seq_len << " ***\n";

// TensorContract_GPU.cu
// ISSUE #76: Detect max_seq_len boundary - assume typical max_seq_len values
```

**Status:** ✅ ACCEPTABLE - Logging/diagnostics don't affect runtime behavior

---

## Category 3: **MISUSE** - Positional Bias Computation ⚠️

### 3.1 ALiBi Slope Computation (CRITICAL)
**File:** `Shared/PBM/PositionalBiasMethod.cu`  
**Line:** 56-106  

**Current Code:**
```cpp
bool computeAlibiSlopes(const PBMConfig& config, ...) {
    const int d_max = config.max_seq_len;  // ❌ WRONG!
    
    // Slope formula uses max_seq_len as context length
    const float m_min = target_bias / static_cast<float>(d_max);
    
    // This means: If you train with max_seq_len=1024 but run inference 
    // at 2048 tokens, your ALiBi slopes will be HALF what they should be!
}
```

**Problem:**
- ALiBi slope computation is CACHED and only recomputed when `max_seq_len` changes
- If actual runtime sequence length exceeds `max_seq_len`, slopes are too weak
- The doc comment even warns about this: *"WARNING: If you set max_seq_len=2048 but run inference at 8192 tokens, your weakest heads will be too weak"*

**Why This is Wrong:**
- `max_seq_len` is a MEMORY LIMIT, not a context length specification
- ALiBi slopes should be calibrated for the ACTUAL working context length (could be 4k, 8k, 16k)
- Current implementation conflates "max allocated sequence" with "target context length"

**Recommended Fix:**
```cpp
struct PBMConfig {
    int num_heads = 0;
    int rotary_dim = 0;
    int target_context_length = 0;  // NEW: Separate from max_seq_len
    float alibi_slope_exponent = 0.0f;
    float alibi_max_bias = 0.0f;
};

bool computeAlibiSlopes(const PBMConfig& config, ...) {
    // Use target_context_length for slope computation
    const int d_max = config.target_context_length;
    
    // Slopes now scale correctly regardless of max_seq_len
    const float m_min = target_bias / static_cast<float>(d_max);
}
```

**Impact:** Medium  
**Affected Components:** Training, Inference (both use cached slopes)  
**Workaround:** Always set `max_seq_len` >= actual longest sequence in dataset

---

### 3.2 RoPE Frequency Scaling (MODERATE)
**File:** `Shared/PBM/PositionalBiasMethod.cu`  
**Line:** 192-226  

**Current Code:**
```cpp
bool computeRopeFreqs(const PBMConfig& config, ...) {
    // NTK-aware context scaling
    if (config.max_seq_len > BASE_SEQ_LEN && config.rotary_dim > 2) {
        const float ctx_ratio = 
            static_cast<float>(config.max_seq_len) / 
            static_cast<float>(BASE_SEQ_LEN);  // ❌ Uses max_seq_len
        
        const float exponent = 
            static_cast<float>(config.rotary_dim) / 
            static_cast<float>(config.rotary_dim - 2);
        
        scaled_theta = config.rope_theta * std::pow(ctx_ratio, exponent);
    }
}
```

**Problem:**
- Same issue as ALiBi: Uses `max_seq_len` as proxy for "target context length"
- If you train with `max_seq_len=1024` for memory savings but want 4k context, RoPE frequencies won't scale correctly

**Recommended Fix:**
```cpp
// Use target_context_length instead of max_seq_len
if (config.target_context_length > BASE_SEQ_LEN && config.rotary_dim > 2) {
    const float ctx_ratio = 
        static_cast<float>(config.target_context_length) / 
        static_cast<float>(BASE_SEQ_LEN);
    // ...
}
```

**Impact:** Moderate  
**Risk:** Lower than ALiBi (RoPE is more robust to slight misconfigurations)

---

## Category 4: CORRECT ALGORITHMIC USES ✅

### 4.1 Sinusoidal Position Embeddings
**File:** `training/Autograd/AutogradTraining.cu`  
**Line:** 96-135  

**Status:** ✅ CORRECT - Uses **actual** `seq_len`, NOT `max_seq_len`
```cpp
__global__ void addSinusoidalPositionEmbeddingsKernel(
    float* embeddings,
    int total_tokens,
    int d_model,
    int seq_len,  // ✅ Uses actual sequence length
    float scale
) {
    const int pos = token_idx % seq_len;  // ✅ Per-token position
    // ... sine/cosine computation ...
}
```

### 4.2 Position Embedding Lookup
**File:** `training/Autograd/AutogradTraining.cu`  
**Line:** 401-420  

**Status:** ✅ CORRECT - Buffer is `[max_seq_len, d_model]` but lookup uses actual positions
```cpp
// Buffer allocated for worst-case
ts->position_embedding_weights.shape = 
    TensorShape::make_BSM(cfg->max_seq_len, cfg->d_model);

// But lookup uses ACTUAL position IDs generated per sequence
Tensor pos_emb_output = autograd::embedding(
    ts->position_embedding_weights,
    d_position_ids,  // ✅ Actual positions [0..seq_len-1]
    total_tokens,
    // ...
);
```

---

## Summary Statistics

| Category | Count | Status |
|----------|-------|--------|
| Memory Allocation | 15+ | ✅ Correct |
| Capacity Validation | 8+ | ✅ Correct |
| Config/Serialization | 12+ | ✅ Correct |
| Diagnostic Logging | 20+ | ✅ Acceptable |
| **ALiBi Slope Computation** | **1** | ⚠️ **MISUSE** |
| **RoPE Frequency Scaling** | **1** | ⚠️ **MISUSE** |
| Algorithmic (Correct) | 5+ | ✅ Correct |

---

## Recommendations

### HIGH PRIORITY
1. **Add `target_context_length` config parameter**
   - Separate from `max_seq_len` (memory limit)
   - Add to `ai_config.json` under `architecture` section
   - Default: Same as `max_seq_len` (backward compatible)

2. **Refactor `PBMConfig` structure**
   ```cpp
   struct PBMConfig {
       int num_heads;
       int rotary_dim;
       int target_context_length;  // NEW: For positional bias scaling
       // max_seq_len removed - should not be in this struct
   };
   ```

3. **Update ALiBi/RoPE initialization**
   - Pass `target_context_length` instead of `max_seq_len`
   - Add validation: `target_context_length <= max_seq_len`
   - Update documentation to clarify separation

### MEDIUM PRIORITY
4. **Add runtime assertions**
   ```cpp
   // In forward pass, validate sequences don't exceed capacity
   assert(actual_seq_len <= max_seq_len && 
          "Sequence exceeds max_seq_len buffer capacity");
   
   // In ALiBi/RoPE, warn if extrapolating beyond target
   if (actual_seq_len > target_context_length) {
       std::cerr << "WARNING: Sequence length " << actual_seq_len 
                 << " exceeds target_context_length " 
                 << target_context_length << ". Positional encodings will extrapolate.\n";
   }
   ```

5. **Documentation updates**
   - Add section to `copilot-instructions.md` clarifying `max_seq_len` purpose
   - Update `MAX_SEQ_LEN_AUDIT.md` (this doc) periodically
   - Add comments to `HyperParameters_GPU.hpp` explaining the distinction

### LOW PRIORITY
6. **Consider renaming for clarity**
   - `max_seq_len` → `max_allocated_seq_len` (more explicit)
   - Or add typedef: `using BufferCapacity = int;`

---

## Testing Recommendations

1. **Test Case: Buffer Overflow Detection**
   ```cpp
   // Feed sequence with length > max_seq_len
   // Should reject with clear error (currently does ✅)
   ```

2. **Test Case: ALiBi Extrapolation**
   ```cpp
   // Train with target_context_length=1024
   // Run inference with seq_len=2048
   // Verify slopes still produce reasonable attention patterns
   ```

3. **Test Case: Memory Limit Independence**
   ```cpp
   // Set max_seq_len=512 (memory limit)
   // Set target_context_length=2048 (working context)
   // Verify ALiBi/RoPE scale for 2048, but batches respect 512 limit
   ```

---

## Conclusion

The GRIM-text codebase uses `max_seq_len` correctly in the vast majority of cases (95%+). The two misuses in positional bias computation are **architecturally incorrect** but currently work due to:

1. Training data sequences are typically <= `max_seq_len`
2. Inference hasn't pushed beyond the configured limit
3. The mismatch is documented in code comments (known issue)

**Action Required:** Add `target_context_length` parameter to decouple memory management from positional encoding configuration. This will enable:
- Training with smaller batches (lower `max_seq_len` for memory savings)
- While still supporting longer context lengths (higher `target_context_length`)
- Without retraining positional bias caches

**Priority:** Medium (not urgent, but should be addressed before extending context length support)

---

## Files Requiring Changes

1. `Shared/HyperParameters/HyperParameters_GPU.hpp`
   - Add `target_context_length` to `ModelArchitecture`
   - Update JSON parser

2. `Shared/PBM/PositionalBiasMethod.hpp`
   - Change `PBMConfig.max_seq_len` → `PBMConfig.target_context_length`

3. `Shared/PBM/PositionalBiasMethod.cu`
   - Use `target_context_length` in `computeAlibiSlopes()`
   - Use `target_context_length` in `computeRopeFreqs()`

4. `training/Phases/Phase1_Startup.cu`
   - Pass `target_context_length` (or `max_seq_len` as fallback) to PBM init

5. `ai_config.json`
   - Add `target_context_length` field to architecture section
   - Document default behavior

6. `copilot-instructions.md`
   - Add entry to "Common Pitfalls" section (#48)

---

**End of Audit Report**
