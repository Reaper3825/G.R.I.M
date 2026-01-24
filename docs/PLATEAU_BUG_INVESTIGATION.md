# GRIM-text Training Plateau Bug Investigation

**Status:** 🔄 **IN PROGRESS** - Issue #59: Unified Loss System Refactor  
**Started:** December 22, 2025  
**Last Updated:** January 24, 2026
**Latest Issue:** Loss computation was using plain CE without focal loss, label smoothing, or entropy regularization. The autograd migration (Issue #46) accidentally removed all loss components! Refactored to `unified_loss()` with full config support.

---

## 🔴 NEW: Issue #59 - Loss Logging Does Not Match Training Loss (ARCHITECTURE BUG!)

### Discovery (January 24, 2026)

Investigating why training wasn't improving despite correct gradient flow. Found that:
- **Log message `[GradTrace] POST-FORWARD loss=X.XXXX`** comes from Phase2_TrainingLoop.cu line 2410
- **Loss is computed via** `autograd::cross_entropy_loss()` in ComputeLossBatch.cu line 790
- **BUT**: `cross_entropy_loss()` only computes **PLAIN CE** - no focal loss, no label smoothing, no entropy regularization!

The Issue #46 autograd migration deleted the connection to `UnifiedLoss_GPU.cu` which had all the loss components. Training was using config with `focal_gamma=1.0`, `label_smoothing.epsilon=0.1`, etc. but the actual loss computation ignored ALL of these!

### Root Cause: Two Competing Loss Systems

**Before migration:**
- `UnifiedLoss_GPU.cu` computed loss with focal + smoothing + entropy
- A separate autograd path computed gradients

**After Issue #46 migration:**
- `autograd::cross_entropy_loss()` computes loss AND provides grad_fn
- BUT it was written with **plain CE only** - the focal/smoothing/entropy code was never ported!

### Impact

- Config says: `focal_gamma=1.0, label_smoothing.epsilon=0.1, entropy_reg.lambda=0.01`
- Actually used: `focal_gamma=0.0, smoothing_epsilon=0.0, entropy_reg_lambda=0.0`
- **100% of loss configuration was being IGNORED!**

### Fix Applied

**1. New unified_loss() function in AutogradLoss.cu:**
```cpp
Tensor unified_loss(
    Tensor& logits,
    const int* targets,
    const float* valid_mask,
    int num_tokens,
    int vocab_size,
    const LossConfig& config,  // NOW ACCEPTS CONFIG!
    cudaStream_t stream
);
```

**2. New forward kernel with ALL loss components:**
```cuda
// Unified loss: L = α * (1-p_t)^γ * CE_smooth + λ * neg_entropy
float ce_smooth = compute_label_smoothed_ce(...);
float focal_weight = powf(1.0f - p_t, focal_gamma);
float entropy_loss = entropy_reg_lambda * neg_entropy;
float total_loss = focal_alpha * focal_weight * ce_smooth + entropy_loss;
```

**3. Updated ComputeLossBatch.cu call site:**
```cpp
autograd::LossConfig ag_loss_config;
ag_loss_config.focal_alpha = full_loss_cfg.focal.enabled ? full_loss_cfg.focal.alpha : 1.0f;
ag_loss_config.focal_gamma = full_loss_cfg.focal.enabled ? full_loss_cfg.focal.gamma : 0.0f;
ag_loss_config.smoothing_epsilon = full_loss_cfg.label_smoothing.enabled ? ... : 0.0f;
ag_loss_config.entropy_reg_lambda = full_loss_cfg.entropy_reg.enabled ? ... : 0.0f;

training_state_.loss_tensor = autograd::unified_loss(
    training_state_.logits_tensor,
    training_state_.cached_targets,
    nullptr,
    total_tokens,
    cfg.vocab_size,
    ag_loss_config,
    stream
);
```

**4. Legacy wrapper preserved:**
```cpp
Tensor cross_entropy_loss(...) {
    LossConfig plain_ce;  // All zeros = plain CE
    return unified_loss(..., plain_ce, ...);
}
```

### Files Modified

1. **AutogradLoss.hpp**: Added `LossConfig` struct and `unified_loss()` declaration
2. **AutogradLoss.cu**: Rewrote forward kernel with focal + smoothing + entropy; added `UnifiedLossGradFn`
3. **ComputeLossBatch.cu**: Updated to build `LossConfig` and call `unified_loss()`
4. **copilot-instructions.md**: Updated unified loss system documentation

### DELETED Files

- `UnifiedLoss_GPU.cu` / `UnifiedLoss_GPU.hpp` - Old disconnected loss system
- `ComputeLoss_GPU.cu` / `ComputeLoss_GPU.hpp` - Old orchestrator (now inlined in ComputeLossBatch)

**Status:** ✅ **FIX IMPLEMENTED** - Rebuild and test required

---

## 🔴 PREVIOUS: Issue #58 - Cross-Entropy Backward Missing Mean Reduction Scaling (ROOT CAUSE!)

### Discovery (January 24, 2026)

Comparing gradient magnitudes between GRIM and PyTorch:
- **GRIM pre-clip gradient norm:** 10,000 - 28,000
- **PyTorch gradient norm:** 0.1 - 1.0
- **Ratio:** ~100,000x difference!

Investigating further, found that `AutogradLoss.cu` had an **incorrect "fix"** at lines 276-278:

**BROKEN CODE (Issue #46 "FIX" was WRONG):**
```cuda
// Issue #46 FIX: Pass 1.0f instead of 1/valid_count
// Standard CE gradient is (softmax - one_hot), no per-token normalization.
const float scale = 1.0f;  // Was: 1.0f / valid_count (WRONG!)
```

This comment claims that `1/valid_count` was wrong, but **THAT IS BACKWARDS**!

### Mathematical Proof

When using **mean reduction** (which GRIM does for loss):
- `loss = sum(ce_per_token) / N`
- By chain rule: `grad = d(loss)/d(logits) = (softmax - one_hot) / N`

Verified with PyTorch:
```python
>>> F.cross_entropy(logits, targets, reduction='mean').backward()
>>> grad_mean = logits.grad.clone()
>>> logits.grad.zero_()
>>> F.cross_entropy(logits, targets, reduction='sum').backward()
>>> grad_sum = logits.grad.clone()
>>> grad_sum / grad_mean  # = exactly N (number of tokens)
```

With `N = 3500` tokens per batch, GRIM's gradients were **3500x too large**!

### Why This Caused Mode Collapse

1. **Massive gradients:** 3500x larger than PyTorch
2. **Gradient clipping:** Clips to norm=1.0, but direction is already corrupted by magnitude
3. **Effective LR:** After clipping, the gradient direction is dominated by noise/outliers
4. **Positive feedback:** Most common token (SPACE=277) gets largest absolute gradient
5. **Mode collapse:** Model learns to predict only SPACE

### Fix Applied (AutogradLoss.cu)

**1. Launch function (line ~276):**
```cuda
// Issue #58 FIX: MUST divide by valid_count to match PyTorch's mean reduction!
const float scale = (valid_count > 0) ? (1.0f / static_cast<float>(valid_count)) : 1.0f;
```

**2. Optimized kernel (line ~232):**
```cuda
// Issue #58 FIX: Apply inv_valid_count to match PyTorch mean reduction!
grad_row[v] = (prob - one_hot) * inv_valid_count;
```

### Expected Results After Fix

- Pre-clip gradient norms: ~3-10 (matching PyTorch scale)
- No immediate mode collapse
- Training should now learn like PyTorch baseline

**Status:** ✅ **FIX IMPLEMENTED** - Rebuild and test required

---

## 🔵 HISTORICAL: Issue #57 - Position Embeddings Not Added (FIXED!)

### Discovery (January 24, 2026)

Comparing GRIM training loop to libtorch baseline to understand why GRIM doesn't learn while PyTorch does. Found that position embeddings are **ALLOCATED** and **INITIALIZED** but **NEVER ADDED** to token embeddings!

**PyTorch Baseline (CORRECT):**
```cpp
// Tools/libtorch_baseline/main.cpp line 780
auto x = tok_emb->forward(idx) + pos_emb->forward(pos);  // ✅ ADDS POSITION EMBEDDINGS
```

**GRIM Before Fix (BROKEN):**
```cpp
// AutogradTraining.cu lines 297-302
// Add position embeddings if available
// TODO: Implement autograd::add for position embedding addition
if (ts->position_embedding_weights.data) {
    ts->position_embedding_weights.requires_grad = true;
    // For now, position embeddings are added in legacy path  ❌ LEGACY PATH WAS DELETED!
    // The embedding lookup result already has autograd attached
}
```

**ADDITIONAL BUG:** `training_state_.position_embedding_weights` was created with `Tensor::zeros()` in InitTrainingState.cu, but the ACTUAL weights are in `embedding_runtime->position_buffer` (Xavier-initialized in TrainingOps.cu). These were TWO SEPARATE BUFFERS - optimizer was updating the wrong one!

### Impact

Without position embeddings:
- Model has NO position information during training
- Every token at every position looks identical
- Model cannot learn sequence structure (which word comes first/last)
- Training collapses because positional relationships cannot be learned

### Fixes Applied

**1. AutogradTraining.cu - Add position embeddings to token embeddings:**
```cpp
// Issue #57 FIX: Add position embeddings
if (ts->position_embedding_weights.data) {
    ts->position_embedding_weights.requires_grad = true;
    
    // Generate position IDs: [0,1,2,...,seq_len-1] repeated for each batch
    int* d_position_ids = nullptr;
    cudaMallocAsync(&d_position_ids, total_tokens * sizeof(int), ctx.stream);
    generatePositionIds(d_position_ids, total_tokens, ctx.seq_len, ctx.stream);
    
    // Look up position embeddings with autograd tracking
    Tensor pos_emb_output = autograd::embedding(
        ts->position_embedding_weights, d_position_ids, total_tokens, ctx.stream);
    
    cudaFreeAsync(d_position_ids, ctx.stream);
    
    // Add token embeddings + position embeddings (both tracked by autograd)
    emb_output = autograd::add(emb_output, pos_emb_output, ctx.stream);
}
```

**2. InitTrainingState.cu - Wrap existing position buffer instead of creating new one:**
```cpp
// Issue #57 FIX: Position embeddings MUST wrap existing buffer, NOT create new one!
auto* embedding_runtime = &getGpuEmbedder();
training_state_.position_embedding_weights = Tensor::from_ptr(
    embedding_runtime->position_buffer,           // Use SAME buffer as optimizer
    TC::make_BSM(cfg.max_seq_len, cfg.d_model),
    false,  // doesn't take ownership
    true);  // requires_grad
training_state_.position_embedding_weights.ensure_grad();  // Allocate grad buffer
```

**3. Added generatePositionIdsKernel:**
```cpp
__global__ void generatePositionIdsKernel(int* position_ids, int total_tokens, int seq_len) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_tokens) return;
    position_ids[idx] = idx % seq_len;  // [0,1,2,...,seq_len-1] per batch
}
```

### Files Modified

1. `AutogradTraining.cu`: Added position embedding lookup, add, and kernel
2. `InitTrainingState.cu`: Changed `Tensor::zeros()` to `Tensor::from_ptr()` wrapping actual buffer

### Expected Results After Fix

- Position embeddings now added to token embeddings (matches PyTorch)
- Gradients flow through BOTH token and position embeddings via autograd
- Optimizer updates BOTH embedding matrices
- Model can learn positional relationships
- Training should no longer collapse to token 277

**Status:** ✅ **FIX IMPLEMENTED** - Rebuild and test required

---

## ✅ RESOLVED: Issue #56 - Autograd Tensor Use-After-Free (ROOT CAUSE FOUND!)

### Discovery (January 23, 2026)

Training crash at KERNEL #80 (`kernel_gelu_backward`) with "illegal memory access":
```
[KERNEL #80] kernel_gelu_backward
CUDA Error: 700 (illegal memory access was encountered)
```

**Critical Evidence from Training Log:**
```
128: [AutogradTraining] INFO: Step 2: Running 12 encoder layers with autograd...
129: [MatMulGradFn::~MatMulGradFn] ENTER this=000002B6167BF190 owns_a=1 owns_b=0
130: [MatMulGradFn::~MatMulGradFn] Deleting a_grad_fn=000002B5DEA0DE50
```

**THE BUG:** Autograd `GradFn` destructors are running **DURING** the forward pass (line 129) immediately after "Running 12 encoder layers" message (line 128). This frees GPU memory that later backward pass still needs → crash.

### ROOT CAUSE FOUND: Missing Return Statement in FFN::forward()

The `FeedForwardLayer::forward()` function in `Feed_Forward_GPU.cu` was **declared to return a Tensor but had NO return statement!**

```cpp
// BEFORE (BUG):
Tensor FeedForwardLayer::forward(Tensor& input, ForwardIntermediates& intermediates, cudaStream_t stream) {
    // ... build autograd graph ...
    Tensor output = autograd::matmul(...);
    launchFFNBiasAdd(output.data, ...);
    // MISSING: return output;
}  // <-- output destroyed here, triggers grad_fn chain destruction!
```

**Why This Caused the Crash:**
1. `output` is a local `Tensor` with `owns_grad_fn=true`
2. When function exits without return, `output` destructor runs
3. `~Tensor()` sees `owns_grad_fn=true` and deletes `grad_fn`
4. That `grad_fn` (MatMulGradFn) has `owns_a_grad_fn=true`, so it deletes its upstream `a_grad_fn`
5. CASCADE: The entire autograd graph is destroyed DURING the forward pass!
6. Later backward tries to use freed `grad_fn` pointers → "illegal memory access"

### Fix Applied (Feed_Forward_GPU.cu)

```cpp
// AFTER (FIXED):
Tensor FeedForwardLayer::forward(Tensor& input, ForwardIntermediates& intermediates, cudaStream_t stream) {
    // ... build autograd graph ...
    Tensor output = autograd::matmul(intermediates.ffn_gelu_out, W2_, stream,
                                     intermediates.ffn_gelu_out.data, nullptr);
    
    // Add bias b2 (broadcasted)
    launchFFNBiasAdd(output.data, b2_.data, total_tokens, config_.d_model, stream);
    
    // CRITICAL (Issue #56 root cause fix): Return the output tensor!
    // Without this return statement, `output` is destroyed at function end,
    // which triggers destruction of its grad_fn chain, destroying the entire
    // autograd graph DURING the forward pass!
    return output;
}
```

### Why This Wasn't Caught By Compiler

C++ compilers often don't warn about missing return statements in functions returning non-void types (undefined behavior). The function was returning "garbage" - whatever happened to be in the return register. Since the caller immediately stored the result in `ForwardIntermediates.ffn_out`, the uninitialized Tensor seemed to work until backward needed the destroyed grad_fn.

### Files Modified

1. `Feed_Forward_GPU.cu`: Added missing `return output;` statement

### Debug Logging Cleanup (Same Session)

Removed extensive debug logging added during investigation:
- `TensorContract_GPU.cu`: Removed ~50 fprintf calls from GradFn destructors, deleters, apply() methods
- `Tensor::backward()`: Cleaned up verbose polling/sync logging
- Preserved error-path logging (RULE 20: Fail Loud)

**Status:** ✅ **FIX IMPLEMENTED** - Rebuild and test required

---

## 🔵 HISTORICAL: Issue #55 - __shfl_down_sync with Partial Warp Causes GPU Hang (FIXED!)

### Discovery (January 23, 2026)

Training log showed GPU stream stuck waiting for `kernel_rmsnorm_backward`:
```
[Tensor::backward] Stream query before sync: NOT_READY (stream=000001FA6D2AAA70)
[Tensor::backward] Synchronizing stream before cleanup...
[Tensor::backward] Still waiting for stream (poll #1000000)...
[KERNEL CHECK] Last kernel 'kernel_rmsnorm_backward' (#99) status: PENDING
...
[Tensor::backward] TIMEOUT: Stream stuck after 100M polls! Checking for errors...
[Tensor::backward] cudaGetLastError: SUCCESS
```

Key observation: **cudaGetLastError returned SUCCESS** but kernel never completed - classic sign of GPU deadlock.

### Root Cause: __shfl_down_sync with 0xffffffff Mask on Partial Warp

The `kernel_rmsnorm_backward` in `TensorContract_GPU.cu` had a **critical CUDA programming bug**:

```cuda
// BUGGY CODE:
if (threadIdx.x < blockDim.x / 32) {  // Only 8 threads active (with 256 threads)
    local_sum_sq = s_sum_sq[threadIdx.x];
    for (int offset = blockDim.x / 64; offset > 0; offset >>= 1) {
        local_sum_sq += __shfl_down_sync(0xffffffff, local_sum_sq, offset);  // BUG!
    }
}
```

**Why This Hangs:**
1. `blockDim.x = 256` (AUTOGRAD_BLOCK_SIZE), so `blockDim.x / 32 = 8` warps
2. The block reduction only has **8 active threads** (threadIdx.x < 8)
3. `__shfl_down_sync(0xffffffff, ...)` uses mask `0xffffffff` = **all 32 threads must participate**
4. Only 8 threads call the shuffle, but mask says wait for 32 → **deadlock**
5. GPU hangs indefinitely waiting for threads that never participate

### Evidence from Training Log

```
[KERNEL #99] kernel_rmsnorm_backward - launched OK
[RMSNormGradFn] Continuing chain via input_grad_fn=000001FA315661B0 (op=add)
[AddGradFn] apply() SKIPPED (already applied)
[RMSNormGradFn] input_grad_fn->apply() returned
...
[Tensor::backward] Stream query before sync: NOT_READY
[Tensor::backward] Still waiting for stream (poll #100000000)...
[Tensor::backward] TIMEOUT: Stream stuck after 100M polls!
```

The kernel launched successfully but never completed - threads stuck in `__shfl_down_sync`.

### Fix Applied (TensorContract_GPU.cu)

**Replaced partial-warp shuffle with sequential reduction in thread 0:**

```cuda
// FIXED CODE (Issue #55):
// Write warp results to shared memory
if (threadIdx.x % 32 == 0) {
    s_warp_vals[threadIdx.x / 32] = local_sum_sq;
}
__syncthreads();

// Use sequential reduction in thread 0 (safe for any warp count)
__shared__ float s_rms_sq, s_inv_rms;
if (threadIdx.x == 0) {
    float total = 0.0f;
    for (int i = 0; i < num_warps; i++) {
        total += s_warp_vals[i];
    }
    s_rms_sq = total / d_model + eps;
    s_inv_rms = rsqrtf(s_rms_sq);
}
__syncthreads();
```

### Why Sequential Reduction is Safe

1. **No warp synchronization issues** - only thread 0 reads/writes
2. **Works for any blockDim.x** - no assumptions about warp count
3. **Minimal overhead** - only 8 reads in thread 0 (negligible vs memory bandwidth)
4. **Same algorithmic complexity** - O(num_warps) vs O(log(num_warps)) is trivial for 8 warps

### Files Modified

1. `TensorContract_GPU.cu`: Fixed `kernel_rmsnorm_backward` block reduction

### CUDA Best Practice Reminder

**NEVER use `__shfl_*_sync(0xffffffff, ...)` with fewer than 32 active threads!**

Valid patterns:
- ✅ Full warp (all 32 threads) with mask `0xffffffff`
- ✅ Partial warp with **correct mask** computed from active threads
- ✅ Sequential reduction for cross-warp aggregation (always safe)

Invalid pattern (causes GPU hang):
- ❌ `if (threadIdx.x < N) { __shfl_down_sync(0xffffffff, ...); }` where N < 32

**Status:** ✅ **FIX IMPLEMENTED** - Rebuild and test required

---

## 🔵 HISTORICAL: Issue #46 - Autograd Chain Broken in CrossEntropyLossGradFn (January 21, 2026)

### Discovery (January 21, 2026)

Training log showed **ALL gradients ZERO** from the very first batch:
```
[GradTrace] batch=1 micro=1/2 valid_tokens=1376
           gradients[0:10]=[0.000e+00,0.000e+00,0.000e+00,...]
           COMPUTED COMPONENTS: total=0.0000 emb_lm_tied=0.0000 num=0.0000 attn=0.0000 ffn=0.0000 rms=0.0000
[ACCUM_BUG] WARNING: preclip=0.0000 which is impossibly low
[FATAL] Gradient accumulation appears broken - stopping training
```

Key observation: **Loss was computed correctly (~10.2)** but gradients never reached parameters.

### Root Cause: Autograd Chain STOPS at Loss Computation

The `CrossEntropyLossGradFn::apply()` in `AutogradLoss.cu` was computing `grad_logits` correctly but **NEVER CONTINUING THE BACKWARD CHAIN**!

The Issue #45 fix migrated to a pure autograd system:
1. `LanguageModel_Training.cu` calls `training_state_.loss_tensor.backward(nullptr)`
2. `Tensor::backward()` calls `grad_fn->apply()` expecting it to recursively continue
3. `CrossEntropyLossGradFn::apply()` writes to `logits_tensor_ptr->grad` but **NEVER calls `logits_tensor_ptr->grad_fn->apply()`**
4. The backward chain STOPS - no gradients reach encoder layers, embeddings, or any parameters

### Evidence from Code (BEFORE Fix)

```cpp
__host__ void apply(const Tensor& grad_output, cudaStream_t stream) override {
    if (logits_tensor_ptr) {
        logits_tensor_ptr->ensure_grad();
        launchCrossEntropyBackward(..., logits_tensor_ptr->grad, ...);
    }
    // MISSING: Call to continue backward chain!
    // Should be: logits_tensor_ptr->grad_fn->apply(logits_grad, stream);
}
```

### Fix Applied (AutogradLoss.cu)

**Added recursive backward chain continuation:**

```cpp
__host__ void apply(const Tensor& grad_output, cudaStream_t stream) override {
    if (logits_tensor_ptr) {
        logits_tensor_ptr->ensure_grad();
        launchCrossEntropyBackward(..., logits_tensor_ptr->grad, ...);
        
        // ================================================================
        // BUG FIX Issue #46: CONTINUE AUTOGRAD CHAIN!
        // ================================================================
        if (logits_tensor_ptr->grad_fn) {
            Tensor logits_grad;
            logits_grad.data = logits_tensor_ptr->grad;
            logits_grad.shape = logits_tensor_ptr->shape;
            logits_grad.owns_data = false;
            logits_grad.stream = stream;
            
            logits_tensor_ptr->grad_fn->apply(logits_grad, stream);
            logits_tensor_ptr->grad_fn->release_saved();
        }
    }
}
```

### Why This Wasn't Caught Earlier

- Issue #45 fix DELETED the legacy 3-phase backward system and replaced with pure autograd
- The autograd system was incomplete - `cross_entropy_loss` was the first loss function converted
- Testing focused on gradient zeroing (Issue #45's symptom) not autograd chain completion

**Status:** ✅ **FIX IMPLEMENTED** - Rebuild and test required

---

## 🔵 HISTORICAL: Issue #45 - grad_encoder_out NOT REGISTERED with GradAccumulationController

### Discovery (January 21, 2026)

Training log showed **exponential gradient growth** during accumulation:
```
Batch 2434 (micro_step 1/2): preclip_grad_norm = 4.24e17
Batch 2435 (micro_step 2/2): preclip_grad_norm = 5.56e18  ← 13x increase!
...
Batch 2449 (micro_step 2/2): preclip_grad_norm = inf
```

Key observation: **Loss stayed stable (~9.7)** while gradients exploded. Forward pass was fine.

### Root Cause: grad_encoder_out Buffer Never Zeroed

The `grad_encoder_out` buffer was **NOT registered** with `GradAccumulationController`, so it was **NEVER zeroed** at the start of each accumulation window.

During gradient accumulation with `accumulation_steps=2`:
1. **micro_step 1**: Backward pass writes gradients to `grad_encoder_out`
2. **micro_step 2**: Backward pass runs again, but `grad_encoder_out` **still contains stale values** from micro_step 1
3. Since encoder backward **reads from** `grad_encoder_out` (via `ctx.current_grad`) while simultaneously **writing to** it, stale values get incorporated into new computations
4. This creates a **positive feedback loop** where gradients multiply exponentially

### Evidence from Training Log

| Batch | micro_step | preclip_grad_norm | Ratio to previous |
|-------|------------|-------------------|-------------------|
| 2434 | 1/2 | 4.24e17 | - |
| 2435 | 2/2 | 5.56e18 | **13.1x** |
| 2436 | 1/2 | 1.52e17 | - |
| 2437 | 2/2 | 2.20e18 | **14.5x** |

The micro_step 2 gradients are consistently 6-15x larger than micro_step 1 - classic symptom of buffer contamination.

### Fix Applied (GradAccumulationController_Integration.cu)

**Registered the missing buffers:**

```cuda
// ========== Issue #45 FIX: grad_encoder_out and grad_logits (CRITICAL!) ==========
// These buffers were NOT registered, causing gradient explosion during accumulation!
// grad_encoder_out is used as ctx.current_grad throughout encoder backward pass.
// Without zeroing at window start, micro_step 2 reads stale gradients from micro_step 1,
// creating a multiplicative feedback loop that causes exponential gradient growth.

if (ts.grad_encoder_out) {
    controller_.registerGradientBuffer(
        "grad_encoder_out_temp",
        ts.grad_encoder_out,
        static_cast<std::size_t>(ts.max_cached_tokens) * cfg.d_model
    );
}

if (ts.grad_logits) {
    controller_.registerGradientBuffer(
        "grad_logits_temp",
        ts.grad_logits,
        static_cast<std::size_t>(ts.max_cached_tokens) * cfg.vocab_size
    );
}
```

### Why This Wasn't Caught Earlier

- With `accumulation_steps=1`, each backward pass starts fresh (gradients zeroed before backward)
- The bug only manifests with `accumulation_steps > 1` where the buffer persists between micro-steps
- Previous testing may have used single-step accumulation

**Status:** ✅ **FIX IMPLEMENTED** - Rebuild and test required

---

## 🔵 HISTORICAL: Issue #44 - Embedding Backward Was COMPLETELY SKIPPED (January 20, 2026)

### Discovery (January 20, 2026)

Training log showed **IDENTICAL weights** across 8 batches:
```
Batch 1 PRE-OPTIMIZER:  lm_w[0:5]=[0.001213,0.013516,-0.001076,0.005309,-0.012034]
Batch 2 POST-OPTIMIZER: lm_w[0:5]=[0.001213,0.013516,-0.001076,0.005309,-0.012034]  ← IDENTICAL!
...
Batch 8 POST-OPTIMIZER: lm_w[0:5]=[0.001213,0.013515,-0.001076,0.005309,-0.012033]  ← 6th decimal change only!
```

Gradient norms showed the problem:
```
GRIM:    emb_lm_tied=0.0003 to 0.0005
PyTorch: emb_lm_tied=5.5469
                      ^^^^^^^ 11,000x LARGER!
```

### Root Cause: Issue #38 "Fix" Was WRONG

The Issue #38 "fix" in `BackwardPhase3_InputLayer.cu` **SKIPPED embedding backward entirely** when `tie_embeddings=true`:

```cuda
if (cfg->tie_embeddings) {
    // SKIP embedding backward - LM head gradient is already correct and centered!
    P3_INFO("Issue #38: Skipping embedding backward...");
} else {
    launchEmbeddingBackward(...);  // Only for untied weights!
}
```

**This was FUNDAMENTALLY WRONG.** Here's why:

In a transformer with tied weights, there are TWO different gradient sources:
1. **LM head backward (GEMM)**: `grad_W = h_centered.T @ grad_logits`
   - This computes gradient based on which OUTPUT predictions were wrong
   - Updates embedding rows for tokens the model PREDICTED

2. **Embedding backward (scatter-add)**: `grad_E[token_id] += grad_encoder_output[position]`
   - This computes gradient based on which INPUT tokens contributed to errors
   - Updates embedding rows for tokens that APPEARED IN INPUT

**These are COMPLETELY DIFFERENT gradients!** Both must be computed for proper training.

### Why Issue #38 Broke Training

- LM head GEMM gradient: tiny (~0.0003) - only reflects output projection errors
- Embedding scatter-add gradient: large (~5.5) - reflects all input token learning

By skipping embedding backward, we removed **~99.99% of the gradient signal**.

Result:
- `update_rms = 0.00000005` to `0.00000020` (microscopic updates)
- Weights changed by ~0.000001 per step (essentially FROZEN)
- Model generation identical across optimizer steps

### The "Centering" Concern Was a Red Herring

Issue #38 claimed embedding backward would "pollute centered gradients" from LM head.
This was WRONG because:

1. Embedding backward uses `atomicAdd` to ACCUMULATE into the buffer
2. LM head GEMM OVERWRITES with fresh computation (beta=0 for first write)
3. The gradients from different sources SHOULD combine - that's mathematically correct!
4. The "centering" in LM head is about the GEMM computation, not about whether other gradients should exist

### Fix Applied (BackwardPhase3_InputLayer.cu)

**REVERTED Issue #38 - Always run embedding backward:**

```cuda
// ALWAYS run embedding backward (for both tied and untied weights)
launchEmbeddingBackward(
    ctx.current_grad,
    ts->cached_token_ids,
    ts->embedding_grads(),
    ctx.batch_size,
    ctx.seq_len,
    cfg->d_model,
    cfg->vocab_size,
    ctx.training_state->stream_ctrl.getPrimaryStream());
```

### Expected Results After Fix

- `emb_lm_tied` gradient norm: ~5.5 (matching PyTorch baseline)
- `update_rms`: ~0.0001 to 0.001 (meaningful updates)
- Weights visibly changing after each optimizer step
- Model generation changing as training progresses

**Status:** ✅ **FIX IMPLEMENTED** - Rebuild and test required

---

## 🔵 HISTORICAL: Issue #43 - Encoder Weight Gradient Centering (January 2026)

### Discovery (January 2026)

While Issue #37/40/42 fixed centering in the **LM head backward**, the **encoder backward** still uses **UN-CENTERED cached activations** for weight gradient computation!

### Root Cause

Encoder weight gradients are computed as:
```
grad_W_qkv = cached_ln1_output^T @ grad_qkv        (uses un-centered ln1_output!)
grad_W_o   = cached_attn_output^T @ grad_attn_out  (uses un-centered attn_output!)
grad_W1    = cached_ffn_input^T @ grad_ffn_hidden  (uses un-centered ffn_input!)
grad_W2    = cached_ffn_hidden^T @ grad_ffn_output (uses un-centered ffn_hidden!)
```

Each cached activation has **NON-ZERO MEAN**. This creates systematic gradient bias:
```
grad_W[i,j] = Σ_t (activation[t,i] × grad[t,j])
            = Σ_t ((centered[t,i] + mean_t) × grad[t,j])
            = Σ_t (centered[t,i] × grad[t,j]) + mean × Σ_t grad[t,j]
                                                ^^^^^^^^^^^^^^^^^^^
                                                BIAS TERM (non-zero!)
```

### Why This Causes Encoder to Output Hidden States Aligned with W[277]

1. LM head backward computes `grad_encoder_output` which flows to encoder
2. Encoder backward computes weight gradients using un-centered activations
3. The non-zero mean creates systematic bias in all encoder weight updates
4. This bias teaches encoder layers to output hidden states with a particular direction
5. That direction happens to align with W[277] because:
   - SPACE (token 277) is 11% of training data
   - grad_logits[277] is consistently negative (model over-predicts SPACE)
   - The bias term `mean × Σ_t grad[t,j]` has consistent sign for common tokens
6. Result: Encoder learns to output hidden states that align with W[277] → mode collapse

### Evidence from Training Log

```
[Issue42-HiddenState277] batch=1 call=1 grad·W=+0.00170 cosine=+0.0162 h_mean=-0.001004
[Issue42-HiddenState277] batch=1 call=2 grad·W=+0.00165 cosine=+0.0133 h_mean=-0.001004
  -- After optimizer step --
[Issue42-HiddenState277] batch=1 call=3 grad·W=-0.0311 cosine=-0.1992  ← FLIPPED!
[Issue42-HiddenState277] batch=1 call=4 grad·W=-0.0544 cosine=-0.1993  ← Encoder outputting aligned h!
```

The hidden states flip from **ANTI-ALIGNED** to **ALIGNED** with W[277] after the optimizer step.
This proves the encoder is learning to output aligned hidden states due to biased weight gradients.

### Fix Applied (BackwardPhase2_Encoder.cu)

Added `centerActivationsKernel` to center cached activations BEFORE weight gradient GEMMs:

```cuda
// Added centering scratch buffer to TrainingState
float* centered_activation_scratch;  // [max(total_tokens * d_model, total_tokens * d_ff)]

// Center activation before each weight gradient GEMM:
centerActivations(cached_ffn_hidden, centered_scratch, d_ff, total_tokens, stream);
// Then: grad_W2 = centered_scratch^T @ grad_ffn_output

centerActivations(cached_ffn_input, centered_scratch, d_model, total_tokens, stream);
// Then: grad_W1 = centered_scratch^T @ grad_ffn_hidden

centerActivations(cached_attn_output, centered_scratch, d_model, total_tokens, stream);
// Then: grad_W_o = centered_scratch^T @ grad_attn_output

centerActivations(cached_ln1_output, centered_scratch, d_model, total_tokens, stream);
// Then: grad_W_qkv = centered_scratch^T @ grad_qkv
```

### Files Modified

1. **TrainingState_GPU.hpp**: Added `centered_activation_scratch` buffer declaration
2. **InitTrainingState.cu**: Allocate centering scratch buffer (max of model/FFN sizes)
3. **TrainingStateGPU.cu**: Free centering scratch buffer in destructor
4. **BackwardPhase2_Encoder.cu**: 
   - Added `centerActivationsKernel` (copy of pattern from lm_head_GPU.cu)
   - Apply centering before grad_W2 GEMM
   - Apply centering before grad_W1 GEMM  
   - Apply centering before grad_W_o GEMM
   - Apply centering before grad_W_qkv GEMM

**Status:** ✅ **FIX IMPLEMENTED** - Rebuild and test required

---

## 🔵 HISTORICAL: Issue #42 - Centering Fixes Were DISABLED in Config!

### Discovery (January 15, 2026)

The Issue #37 (hidden state centering) and Issue #40 (gradient re-centering) fixes were **implemented in code but DISABLED in ai_config.json**!

Training log showed:
```
[2026-01-15 17:38:23] LM Head centering: center_hidden_states=false, recenter_gradients=false
```

### Evidence of Sign Flip Positive Feedback Loop

From training log `training_17685166990989488.log`:

| Call | grad·W | cosine | W[277].norm | Prediction |
|------|--------|--------|-------------|------------|
| 1 | +0.00170 | +0.0162 | 0.165421 | DECREASE (wrong!) |
| 2 | +0.00165 | +0.0133 | 0.165421 | DECREASE (wrong!) |
| **After step=0** | | | **0.165461** | **INCREASED!** |
| 3 | **-0.0311** | **-0.1992** | 0.165461 | INCREASE |
| 4 | -0.0544 | -0.1993 | 0.165461 | INCREASE |
| **After step=1** | | | **0.165799** | **INCREASED!** |
| 5-8 | -0.069 to -0.23 | -0.55 to -0.71 | 0.169+ | INCREASE |

**Pattern:**
1. Before optimizer step: gradient slightly aligned with W (cosine +0.01)
2. After optimizer step: encoder learns to produce h aligned with W[277]
3. Gradient = h * grad_logits ≈ h * (-1) ≈ -W (anti-aligned!)
4. AdamW: W_new = W - (-W) = 2W → norm doubles!
5. Repeat → mode collapse

### Why Centering Fixes the Loop

With `center_hidden_states=true`:
- Each hidden state h[t] has `sum(h[t,:]) ≈ 0`
- `grad_W[v,:] = Σ_t h[t,:] * grad_logits[t,v]`
- `sum(grad_W[v,:]) = Σ_t sum(h[t,:]) * grad_logits[t,v] = 0 * ... = 0`
- The gradient cannot systematically align/anti-align with W!

With `recenter_gradients=true`:
- GEMM FP32 errors accumulate and destroy the zero-sum property
- Re-centering restores `sum(grad_W[v,:]) = 0` after GEMM

### Fix Applied (ai_config.json)

```json
"lm_head_centering": {
    "enabled": true,
    "center_hidden_states": true,
    "recenter_gradients": true
}
```

**Status:** ✅ **FIX ENABLED** - Rebuild and test required

---

## 🔵 HISTORICAL: Issue #40 - FP32 GEMM Numerical Error (ROOT CAUSE FIX)

### The Actual Root Cause

After extensive diagnostic tracing, the TRUE root cause of mode collapse was identified:

**cuBLAS SGEMM accumulates FP32 rounding errors that destroy the zero-sum property of centered gradients.**

### Mathematical Analysis

With Issue #37's hidden state centering, each hidden state h[t] has sum(h[t,:]) ≈ 0.
Therefore, grad_W[v,d] = Σ_t h[t,d] * g[t,v] should mathematically have:
```
sum(grad_W[v,:]) = Σ_d Σ_t h[t,d] * g[t,v] = Σ_t (Σ_d h[t,d]) * g[t,v] = Σ_t 0 * g[t,v] = 0
```

**BUT:** cuBLAS SGEMM computes this using FP32 with K=720 (tokens) accumulation steps.
Each step introduces ~1e-7 rounding error. These errors accumulate SYSTEMATICALLY:
- **Expected row sum:** 0.0
- **Actual GEMM row sum:** 6.4e-5 (positive bias!)
- **Magnitude:** 2000x larger than mathematically correct value

### Evidence (Diagnostic Output)

```
[Issue40-PTR-VERIFY] GEMM_ROW_SUM=6.41397837e-05 MANUAL_SUM=-3.06617482e-08 DIFF=6.41704455e-05
[Issue39-VERIFY] MANUAL[277,0]=0.0174734509 | GEMM[277,0]=0.0174690336
```

Key observations:
1. **Individual elements match** (MANUAL vs GEMM differ by ~1e-5) - the GEMM formula is CORRECT
2. **Row sums don't match** - 768 small errors accumulate to 6e-5 positive bias
3. **Error is SYSTEMATIC** - always positive, driving W[277] in wrong direction

### Why This Causes Mode Collapse

1. Token 277 (SPACE) is 11% of training data - most common token
2. grad_W[277] should have negative updates to reduce its prediction
3. FP32 error adds +6e-5 positive bias to EVERY element of row 277
4. AdamW update: `W[277] -= lr * (grad + weight_decay*W) ≈ W[277] - lr*grad`
5. The +6e-5 bias makes the gradient LESS negative (or even positive)
6. Result: W[277] doesn't decrease enough → logit[277] stays high → mode collapse

### The Fix (lm_head_GPU.cu)

**Re-center each row of grad_weight AFTER the GEMM to eliminate accumulated FP32 error:**

```cuda
__global__ void recenterGradientRowsKernel(
    float* __restrict__ grad_weight,   // [vocab_size, d_model] row-major
    int d_model,
    int vocab_size
) {
    const int vocab_idx = blockIdx.x;
    if (vocab_idx >= vocab_size) return;
    
    float* row = grad_weight + static_cast<size_t>(vocab_idx) * d_model;
    
    // Compute row mean
    __shared__ float s_sum;
    if (threadIdx.x == 0) s_sum = 0.0f;
    __syncthreads();
    
    float local_sum = 0.0f;
    for (int i = threadIdx.x; i < d_model; i += blockDim.x) {
        local_sum += row[i];
    }
    // ... warp reduction ...
    
    // Subtract mean to restore zero-sum property
    const float mean = s_sum / static_cast<float>(d_model);
    for (int i = threadIdx.x; i < d_model; i += blockDim.x) {
        row[i] -= mean;
    }
}
```

**Called immediately after GEMM:**
```cuda
status = cublasSgemm(...);  // Compute grad_weight

// ISSUE #40 FIX: Re-center gradient rows to eliminate FP32 GEMM error
recenterGradientRows(
    params.grad_weight,
    params.d_model,
    params.vocab_size,
    params.stream
);
```

### Why This Fix Is Correct

1. **Preserves gradient direction** - only removes the constant bias, not the signal
2. **Restores mathematical property** - row sums become ~0 as they should be
3. **Affects ALL tokens equally** - not a band-aid for token 277, fixes the root cause
4. **Minimal overhead** - one kernel launch with vocab_size blocks, 256 threads each
5. **No hyperparameter tuning** - pure mathematical correction

### Files Modified

1. `lm_head_GPU.cu`: Added `recenterGradientRowsKernel` and `recenterGradientRows()` wrapper
2. Call site: Immediately after `cublasSgemm` for grad_weight computation

**Status:** ✅ **IMPLEMENTED** - Rebuild and test required

---

## 🔵 HISTORICAL: Issue #39 - Output Logit Bias Correction (January 17, 2026)

### Why Previous Fixes Were Insufficient

**Issue #37** (centering) fixed the gradient sign flip, but collapse still occurred because:
- The LM head weight `W[277]` still receives gradient updates from ALL positions
- When grad_logits[t, 277] is computed for position t with target != 277, it still flows to W[277]
- Over time, SPACE (being 11% of tokens) accumulates more positive bias in its logit

**Issue #38** (per-token class weighting) only weights the loss contribution of target=277 positions.
- But grad_logits[277] is computed for ALL positions via softmax backward
- Weighting doesn't prevent the systematic bias accumulation

### Issue #39 Approach: Output Logit Bias Correction

**THE FIX:** Track exponential moving average (EMA) of each token's mean logit across training.
Before softmax, subtract this bias to ensure no token has systematically higher pre-softmax activation.

**Mathematical Formulation:**
```
// EMA update (after each batch)
logit_bias[v] = (1 - α) * logit_bias[v] + α * batch_mean_logit[v]

// Correction in forward pass (before softmax)
corrected_logit[t, v] = raw_logit[t, v] - logit_bias[v]
softmax_input = corrected_logit
```

Where `α = 0.05` (slow adaptation - 5% new data per batch)

**Why This Works:**
- If SPACE systematically gets higher logits, its bias will grow
- Subtracting the bias normalizes its pre-softmax activation
- The model must learn to discriminate tokens via relative differences, not absolute magnitude
- This is similar to batch normalization's centering but applied to output logits

### Implementation Details (UnifiedLoss_GPU.cu)

**1. Track batch mean logit per token:**
```cuda
// computeLogitMeanKernel - reduces logits across positions per token
// For each vocab token v: mean[v] = sum(logit[t, v]) / num_positions
```

**2. EMA update after loss computation:**
```cuda
// updateLogitBiasEmaKernel
logit_bias[v] = (1.0f - alpha) * logit_bias[v] + alpha * batch_mean[v]
```

**3. Apply correction before softmax:**
```cuda
// In unifiedLossKernelV2, during softmax computation:
const float corrected = token_logits[i] - (logit_bias ? logit_bias[i] : 0.0f);
sum_exp += expf(corrected - max_logit);
```

**Files Modified:**
1. `TrainingState_GPU.hpp`: Added `logit_bias`, `logit_bias_update`, `logit_bias_count` fields
2. `UnifiedLoss_GPU.hpp`: Added logit_bias fields to `UnifiedLossInputs`
3. `UnifiedLoss_GPU.cu`: Added correction in softmax, EMA update kernels
4. `Loss.hpp`: Added logit_bias fields to `LossContext`
5. `ComputeLoss_GPU.cu`: Threading logit_bias through to UnifiedLoss
6. `Phase1_Startup.cu`: Allocation and zero-initialization of buffers
7. `TrainingOps.cu`: Pass logit_bias from TrainingState to LossContext
8. `TrainingStateGPU.cu`: Destructor cleanup of buffers

**Diagnostic Logging:**
```
[Issue39-LogitBias] step=N token_277_bias=X.XXXX max_bias_token=Y max_bias=Z.ZZZZ
```

**Status:** ✅ Implementation complete - rebuild and test required

---

## ✅ PREVIOUS: Issue #37 - Hidden State Non-Zero Mean Causes Gradient Sign Flip

### Root Cause Discovered (January 17, 2026)

**THE PROBLEM:** RMSNorm normalizes variance but NOT mean! Hidden states had small non-zero mean (~-0.001).

With d_model=768: `hidden_sum = 768 × (-0.001) = -0.768`

The LM head weight gradient is:
```
grad_W[277,i] = Σ_t (hidden[t,i] × grad_logits[t, 277])
```

Summed over all dimensions:
```
grad_W[277]_sum = Σ_t (hidden_sum[t] × grad[t,277])
```

**When hidden_sum is NEGATIVE and grad is NEGATIVE: product is POSITIVE!**

This caused the gradient to push W[277] in the WRONG direction, leading to mode collapse.

### Evidence from HiddenState277 Diagnostic

| Batch | hidden_277_mean | grad_277 | contribution_277 | Expected Sign |
|-------|-----------------|----------|------------------|---------------|
| 1 | **-0.001004** | -0.153 | **+0.118** ❌ | Should be - |
| 3 | +0.003513 | -0.144 | -0.390 ✅ | Correct |

### Fix Applied (lm_head_GPU.cu)

**Zero-mean centering before LM head projection:**

```cuda
// centerHiddenStatesKernel() - Centers each hidden state to have zero mean
// y[i] = x[i] - mean(x)
// This ensures hidden_sum = 0 for all positions

// Applied in forward:
if (params.use_centering && params.centered_scratch) {
    centerHiddenStates(encoder_output, centered_scratch, d_model, total_tokens, stream);
    projection_input = centered_scratch;
}

// Applied in backward (both for grad_weight and grad_encoder):
// 1. grad_weight uses centered hidden states
// 2. grad_encoder has centering backward applied: grad_x = grad_y - mean(grad_y)
```

**Files Modified:**
1. `lm_head_GPU.hpp`: Added `centered_scratch`, `centered_encoder`, `use_centering` fields
2. `lm_head_GPU.cu`: Added `centerHiddenStatesKernel()`, centering in forward/backward
3. `ForwardPhase1_OutputLayer.cu`: Pass `encoder_workspace` as scratch buffer
4. `BackwardPhase1_OutputLayer.cu`: Pass `encoder_workspace` for backward centering

**Status:** ✅ Rebuild and test required

---

## 🔵 HISTORICAL: Issue #37 Original Analysis (Before Root Cause Found)

### Issue #37: Encoder Output Aligns with W[277] Regardless of Gradient

**DISCOVERY:** Token 277 diagnostic logging reveals a PARADOX:

| Batch | grad_sum (row 277) | Gradient Direction | Weight Norm Delta | Logit_277 |
|-------|-------------------|-------------------|-------------------|-----------|
| 1 | +0.128 | ⚠️ POSITIVE | - | 0.13 |
| 2 | +0.223 | ⚠️ POSITIVE | +0.00004 | 0.13 |
| 3 | -0.415 | ✓ NEGATIVE | - | 0.28 |
| 4 | -0.421 | ✓ NEGATIVE | **+0.00033** ❌ | 0.30 |
| 5 | -0.135 | ✓ NEGATIVE | - | 0.83 |
| 6 | -0.704 | ✓ NEGATIVE | **+0.00116** ❌ | 0.83 (6/10 argmax) |
| 7 | - | - | - | **1.68 (10/10 argmax)** COLLAPSED |
| 10 | -0.329 | ✓ NEGATIVE | **+0.00360** ❌ | 2.66 |
| 250 | - | - | -0.00050 | 10.4 |

**THE PARADOX:**
- Gradient for W[277] is often **NEGATIVE** → optimizer should DECREASE the weight
- Weight norm sometimes **INCREASES anyway** (batches 4, 6, 8, 10)
- Even when weight norm finally decreases (batch 250: -0.0005), logit_277 is still **10.4** and argmax!

**ROOT CAUSE:**
`logit[277] = hidden_state @ W[277]^T`

Even if `W[277]` decreases, if `hidden_state` learns to align with `W[277]` direction, the dot product **INCREASES**.

**The encoder is learning to produce hidden states that strongly project onto W[277], regardless of input.**

**Log Evidence:**
```
[Token277Diag] batch=1 space_logit: mean=0.1258 is_argmax=1/10  ← Normal
[Token277Diag] batch=6 space_logit: mean=0.8296 is_argmax=6/10  ← TIPPING POINT
[Token277Diag] batch=7 space_logit: mean=1.6827 is_argmax=10/10 ← COLLAPSED (2x jump!)
[Token277Diag] batch=250 space_logit: mean=10.4156 is_argmax=10/10 ← Still collapsed
```

**DISPROVEN HYPOTHESES:**
- ❌ Position embeddings frozen (Issue #36) - FIXED, still collapses
- ❌ RoPE causing bias - Disproving tested
- ❌ ALiBi causing bias - Just added with learned pos_emb, not the cause
- ❌ W[277] weight increasing - It often DECREASES but logit still climbs

**NEXT INVESTIGATION:**
1. Log hidden state statistics in forward pass:
   - Hidden state norm per position
   - Hidden state-W[277] cosine similarity
   - Hidden state variance across positions (are they collapsing to same vector?)
2. Check if attention is collapsing (all positions attending to same content)
3. Compare encoder output between GRIM and PyTorch baseline

---

## ✅ APPLIED: Position Embeddings Now Trainable (January 14, 2026)

### Issue #36 Root Cause: Position Embeddings Had NO Backward Pass

**DISCOVERY:** PyTorch baseline uses learned position embeddings that receive gradients and get updated. GRIM had position embeddings in forward pass but:

1. ❌ NO gradient buffer allocated (`position_embedding_grads` didn't exist)
2. ❌ NO backward kernel implemented
3. ❌ NOT registered with optimizer
4. ❌ Position embeddings were FROZEN at random Xavier initialization!

**PyTorch Baseline (CORRECT):**
```cpp
// main.cpp lines 735-736
tok_emb(register_module("tok_emb", torch::nn::Embedding(cfg.vocab_size, cfg.n_embd))),
pos_emb(register_module("pos_emb", torch::nn::Embedding(cfg.seq_len, cfg.n_embd)));

// pytorch_gradients.txt shows pos_emb gets trained:
[pos_emb.weight] shape=[1024, 768] numel=786432 norm=1.953592e-05 has_grad=YES
```

**GRIM Before Fix (BROKEN):**
```cpp
// TrainingOps.cu - allocated and initialized position embeddings
embedding_runtime->weights.position_embeddings = embedding_runtime->position_buffer;
launchXavierInit(embedding_runtime->position_buffer, ...);  // Random init

// BUT: No gradient buffer, no backward pass, not in optimizer → FROZEN!
```

**FIX APPLIED (5 files changed):**

1. **TrainingState_GPU.hpp**: Added `position_embedding_grads` field
2. **InitTrainingState.cu**: Allocate gradient buffer for position embeddings
3. **Embedding_GPU.cu**: Added `launchPositionEmbeddingBackward()` kernel
4. **BackwardPhase3_InputLayer.cu**: Call position embedding backward in backward pass
5. **LanguageModel_Training.cu**: Register position embeddings with optimizer
6. **TrainingStateGPU.cu**: Free position embedding gradients in destructor

**Why This Caused Collapse:**
- Position embeddings contribute to embedding output: `output = tok_emb[token] + pos_emb[position]`
- Random init means positions add noise that can't be corrected
- Model can't learn position-dependent patterns (sentence structure, etc.)
- Forces model to rely on token identity alone → collapses to most frequent token

**Test Required:** Rebuild and verify:
1. Position embedding gradients are non-zero after backward pass
2. Position embeddings change after optimizer step
3. Model no longer collapses to SPACE token

---

## 🔵 HISTORICAL: Issue #36 Original Analysis (Before Root Cause Found)

**CORRECTION:** Previous analysis incorrectly identified token 277 as EOS. With the correct token layout:

**Token Layout (CORRECT):**
```
[0-255]   = Byte tokens (256 tokens)
[256-272] = Atom tokens (17 types: ATOM_NONE through ATOM_EXPRESSION)
[273+]    = Unigram vocabulary
```

**Special Tokens:**
- Token 273 = UNK (`<unk>`)
- Token 274 = PAD (`<pad>`)
- Token 275 = BOS (`<s>`)  
- Token 276 = EOS (`</s>`)
- **Token 277 = SPACE (` `)** ← This is what the model mode-collapses to!

**Key Evidence:**
```
UNIGRAM_VOCAB_OFFSET = 256 + kAtomTypeCount = 256 + 17 = 273
Token 277 = UNIGRAM_VOCAB_OFFSET + 4 = 273 + 4 = 277
vocab.txt line 4 = " \t-1.76351" (SPACE character)
```

**GRMT Analysis shows Token 277 (SPACE) is 11% of all tokens:**
```
=== Top 20 Most Common Tokens ===
  Token   277:  18569 (11.00%) - SPACE character
  Token   258:   7798 ( 4.62%) - ATOM(2)
  Token   259:   2959 ( 1.75%) - ATOM(3)
```

**Training Log Evidence (training_17683287953155630.log):**

Step 1 (BEFORE optimizer step):
```
[Sample] Hello worldthe valthe valthe valwinwin sin i sin ises.tistical...
```
Output is gibberish but VARIED - expected for random weights.

Step 2 (AFTER just 1 optimizer step):
```
[Sample] Hello world                                                    
```
**COLLAPSED TO SPACES!** The model now predicts only spaces.

**Gradient spike observed at collapse:**
```
Batch 4: total=12.6137 attn=8.5705 ffn=8.3678  ← MASSIVE gradient spike!
```
But gradients were clipped to 1.0, so clipping IS working.

**Why SPACE Token Collapse is Attractive:**
1. SPACE is 11% of all valid targets (most common token)
2. Random baseline loss = ln(50377) ≈ 10.83
3. If model always predicts SPACE: loss ≈ 4.1 (lower than random!)
4. Optimizer finds this "shortcut" - predict most common token

**Why This Shouldn't Happen:**
- Normal transformers don't collapse this fast (usually takes epochs, not 1 step)
- The collapse happened between step 1 and step 2
- Weight updates were tiny (update_rms=0.00000026 vs param_rms=0.00846)
- Something in the model is making SPACE extremely attractive

**Investigation Needed:**
1. Why does 1 optimizer step cause immediate collapse?
2. Is the LM head bias getting set incorrectly?
3. Is there a softmax temperature issue during generation?
4. Check if this happens with sampling instead of greedy

**PyTorch Baseline Finding (Jan 13, 2026):**
- ✅ PyTorch baseline works correctly when `tie_embeddings=true` (p_277 DECREASES)
- ✅ PyTorch baseline ALSO works correctly when `tie_embeddings=false` (p_277 DECREASES from 0.000019850 → 0.000000353)
- **Conclusion:** PyTorch baseline trains correctly in BOTH configurations. The bug is specific to GRIM-text.

**Issue #35 Update:** The EOS contamination issue was INCORRECTLY DIAGNOSED.
- Token 277 was thought to be EOS, but it's actually SPACE
- The "fix" to mask EOS targets may still be useful, but it's NOT the root cause
- SPACE appearing 11% of the time is NORMAL for text data

**Status:** 🔴 **UNDER INVESTIGATION** (Jan 17, 2026)

---

## 🔵 HISTORICAL: Issue #35 (Mislabeled as EOS Contamination)

### Missing Final RMSNorm Layer

**Symptom:** Initial loss is ~6.8 instead of expected ~10.8 (random baseline should be ln(vocab_size)). Debug telemetry shows `sample_weight=0.665662` when it should be `1.0`.

**Evidence from training_run.txt:**
```
[UnifiedLoss] FIRST 10 ce_smooth = [10.938656, 11.142969, 10.702181, ...] ← CORRECT (~10.8)
[UnifiedLoss] First 10 token_losses = [0.000000, 0.000000, 7.002846, ...]  ← WRONG (~7.0 instead of ~10.8)

[UnifiedLoss] DEBUG debug_sample_weight=0.665662  ← SHOULD BE 1.0!
```

**Root Cause:**
In `Phase2_TrainingLoop.cu` at line ~1637, sequence_rarity weights were being applied to ALL training batches:

```cpp
// BEFORE (BUG):
if (!ctx.data.sequence_rarity.empty()) {
    std::vector<float> sequence_weights;
    for (uint32_t sid : filtered_seq_ids) {
        float weight = ctx.data.sequence_rarity[sid];  // Values like 0.665662
        sequence_weights.push_back(weight);
    }
    ctx.model->setSequenceLossWeights(sequence_weights);
}
```

**Why This Broke Training:**
1. `sequence_rarity` scores sequences based on token rarity (rare tokens → higher weight)
2. Most sequences have rarity < 1.0 (e.g., 0.665662)
3. Loss formula: `loss = focal_alpha * focal_weight * ce_smooth * sample_weight`
4. With sample_weight=0.665662 instead of 1.0:
   - ce_smooth = 10.8 (correct random baseline)
   - actual_loss = 10.8 * 0.665662 = **7.2** (incorrect!)
5. This made initial loss appear ~7.0 instead of ~10.8
6. All training metrics were corrupted by this 0.66x scaling factor

**The Fix (Phase2_TrainingLoop.cu):**
```cpp
// AFTER (FIXED): Always clear sequence weights - equal weighting for all sequences
ctx.model->clearSequenceLossWeights();
```

Removed sequence_rarity weighting from:
1. Training batch loss computation (line ~1637)
2. Validation micro-batches (line ~873)
3. Validation batches (line ~1262)

**Status:** ✅ **FIX APPLIED** (Jan 15, 2026) - Rebuild required

---

## 🔴 PREVIOUS CRITICAL BUG (January 14, 2026)

### Issue #30: gradient_clip Accidentally Disabled - 150-950x Effective Learning Rate!

**Symptom:** Training shows chaotic loss oscillation (6.1 → 14.5 → 7.4 → 15.6) with no downward trend. PyTorch baseline learns fine on same data. Gradient norms are 150-950 (should be ~1.0).

**Evidence from training log (training_17682327871961251.log):**
```
[GradTrace] batch=2: total=152.7288 emb_lm_tied=73.43 numeric_head=81.28 attn=47.22 ffn=47.14 rms=0.41 sb=0.09
[GradTrace] batch=10: total=945.5+ (growing out of control!)
Loss: 6.4 → 6.1 → 9.6 → 14.5 → 7.4 → 8.0 → 15.6 → 15.4 (chaotic oscillation)
```

**Root Cause:**
`gradient_clip` was accidentally changed from `1.0` to `0.0` in commit `b15e4b475` ("refactor: update head_dim calculations"):

```bash
# Commit b42d66065 (working): gradient_clip: 1.0
# Commit b15e4b475 (HEAD, broken): gradient_clip: 0.0
```

**Why This Broke Training:**
1. PyTorch baseline uses `clip_grad = 1.0` by default (verified in Tools/libtorch_baseline/main.cpp line 60)
2. GRIM-text with `gradient_clip: 0.0` = NO CLIPPING
3. Gradient norms are 150-950 instead of being clipped to 1.0
4. Effective learning rate: `lr * grad_norm = 0.0003 * 150 = 0.045` (instead of 0.0003)
5. This 150x+ effective LR causes optimizer to overshoot constantly
6. Loss oscillates wildly as weights bounce between extremes

**The Fix (ai_config.json):**
```json
// BEFORE (BUG - commit b15e4b475):
"gradient_clip": 0.0,

// AFTER (FIXED):
"gradient_clip": 1.0,
```

**Status:** ✅ **FIX APPLIED** (Jan 14, 2026) - Rebuild NOT required (config-only change)

---

## 🔴 PREVIOUS CRITICAL BUG (January 13, 2026)

### Issue #29: ScratchBlock Gradients NOT Registered with GradAccumulationController - INFINITE ACCUMULATION

**Symptom:** Training shows high variation oscillation with no learning trend. PyTorch baseline learns fine on same data. After ~490 batches, ScratchBlock gradient component explodes from 0.03 → 429890+ (14 MILLION times increase).

**Evidence from training log:**
```
# Healthy early training:
[GradTrace] batch=50 grad_norm: total=2.3 emb_lm_tied=2.1 attn=0.3 ffn=0.2 rms=0.01 sb=0.03

# After ~490 batches - EXPLOSION:
[GradTrace] batch=499 sb=429890.5  ← 14 MILLION times larger!
v_rms reaches 170 MILLION - massive gradient variance accumulation
RMSNorm gamma gradients flagged as EXPLOSION with rms=3.59e+06
```

**Root Cause:**
ScratchBlock gradient buffers were **NEVER REGISTERED** with GradAccumulationController in `GradAccumulationController_Integration.cu`:

```cpp
// BEFORE (BUG): ScratchBlock gradients NOT registered!
// These buffers were created and used in backward() but NEVER zeroed:
// - d_atom_type_embeddings_grad_
// - d_atom_projection_grad_
// - d_text_feature_projection_grad_
```

**Why This Broke Training:**
1. Rule 22 MANDATES all gradients go through centralized controllers
2. GradAccumulationController zeros all registered buffers before each backward pass
3. ScratchBlock gradients were NOT registered → NEVER zeroed
4. Each batch ADDED to previous batch's gradients infinitely
5. After ~490 batches: accumulated garbage gradients exploded
6. This corrupted gradient direction and caused chaotic oscillation

**The Fix (GradAccumulationController_Integration.cu):**
```cpp
// AFTER (FIXED): Register ScratchBlock gradient buffers
#include "../../Layers/ScratchBlock/ScratchBlock_GPU.hpp"  // Issue #29

// In bindToModel(), add ScratchBlock gradient registration:
if (cfg.use_scratch_block) {
    // Get ScratchBlock instance
    auto* scratch_block = model.getScratchBlock();
    if (scratch_block) {
        // Register atom type embedding gradients
        auto atom_type_grad_info = scratch_block->getAtomTypeEmbeddingsGradInfo();
        if (atom_type_grad_info.buffer && atom_type_grad_info.size > 0) {
            controller.registerBuffer(
                "scratch_block_atom_type_embeddings_grad",
                atom_type_grad_info.buffer,
                atom_type_grad_info.size,
                BufferAccessPattern::AtomicAccumulate
            );
        }
        // ... similar for atom_projection_grad and text_projection_grad
    }
}
```

**Why Previous Fix (Issue #28) Was Necessary But Not Sufficient:**
Issue #28 fixed beta_accum=1.0 bug so cuBLAS overwrites correctly.
BUT ScratchBlock uses **manual CUDA kernels and atomicAdd**, NOT cuBLAS!
So even with beta_accum=0.0, ScratchBlock gradients still accumulated infinitely.

**Status:** ✅ **FIX APPLIED** (Jan 13, 2026) - Rebuild required to test

---

## 🔴 PREVIOUS CRITICAL BUG (January 12, 2026)

### Issue #28: beta_accum ALWAYS 1.0 - Gradient Accumulation Fundamentally Broken

**Symptom:** Training cannot learn. PyTorch baseline learns fine on same data, GRIM-text cannot even overfit single batch.

**Root Cause:**
In `BackwardOps_Orchestrator.cu` line 246, `ctx.beta_accum` was **hardcoded to 1.0f** regardless of the `accumulate` flag!

```cpp
// BEFORE (BUG):
ctx.accumulate = accumulate;  // Flag is stored...
ctx.beta_accum = 1.0f;        // ...but NEVER USED! Always 1.0
```

**Why This Broke Training:**
1. cuBLAS GEMM computes `C = alpha * A @ B + beta * C`
2. When `beta=1.0`: result is **ADDED** to existing C (gradient accumulation)
3. When `beta=0.0`: result **OVERWRITES** C (fresh gradient)
4. **First micro-batch MUST use beta=0.0** to overwrite previous window's gradients
5. With `beta=1.0` always, first micro-batch was adding to stale/garbage gradients
6. This corrupted gradient direction from the very first backward pass

**The Fix:**
```cpp
// AFTER (FIXED):
ctx.accumulate = accumulate;
ctx.beta_accum = accumulate ? 1.0f : 0.0f;  // CONDITIONAL!
```

**Why Previous "Fix" (Issue #22) Didn't Work:**
Issue #22 correctly set `should_accumulate = currentMicroStep() > 0` and passed it to `backward()`.
But `backward()` → `initBackwardContextRaw()` stored the flag but **hardcoded beta_accum=1.0f**.
The fix was incomplete - we fixed the flag but not where it's actually used!

**Status:** ✅ **FIX APPLIED** (Jan 12, 2026) - Rebuild required to test

---

## Previous Issues (Still Applied)

**Symptom:** Training log showed wild loss spikes:
- Batch 1: 11.85
- Batch 2: 10.54
- Batch 3: 9.65
- Batch 4: **6.67** (suspicious huge drop at max_seq_len=1024 boundary)
- Batch 5: **10.10** (spike back up!)

**Root Cause:**
The commit `b42d66065` ("--minor bug fixes found during plateau investigation") removed `cublasSetStream()` calls from multiple files with the incorrect comment "REMOVED cublasSetStream - handle already bound to stream in InitTrainingState.cu".

**Why This Broke Training:**
1. `InitTrainingState.cu` binds cuBLAS handle to primary stream ONCE at initialization
2. `NumericHead::forward()` conditionally **rebinds** the handle: `if (params.stream) { cublasSetStream(params.handle, params.stream); }`
3. After NumericHead runs, ALL subsequent cuBLAS calls (FFN, attention, LMHead, backward pass) execute on **whatever stream NumericHead left the handle on**
4. This causes race conditions where cuBLAS operations execute out-of-order with other CUDA kernels
5. Result: corrupted gradients, wild loss oscillation, training failure

**Files Affected and Fixed:**
1. `QKV_Projector.cu` - Added `cublasSetStream(handle, config.stream)` before GEMM calls
2. `Encoding_GPU.cu` - Added `cublasSetStream(config_.cublas_handle, stream)` before W_o projection
3. `Feed_Forward_GPU.cu` - Added `CUBLAS_CHECK(cublasSetStream(...))` before Layer 1/2 GEMM
4. `lm_head_GPU.cu` - Added `cublasSetStream(params.handle, params.stream)` in forward/backward
5. `BackwardOps_Orchestrator.cu` - Added `cublasSetStream(cublas_handle, primary_stream)` in context init

**Lesson Learned:**
The assumption "bind once at init" is **FRAGILE AND WRONG**. ANY code that calls `cublasSetStream()` affects ALL subsequent cuBLAS calls. The correct pattern is to call `cublasSetStream()` **immediately before every cuBLAS operation** when using explicit streams.

**Status:** ✅ **FIX APPLIED** - Rebuild required to test

---

### Issue #27: Gradient Accumulation NOT Scaled (FIX APPLIED! - Jan 11, 2026)

**Symptom:** With `accumulation_steps=2`, effective learning rate was **2x configured LR**.

**Root Cause:**
The code accumulated gradients from N micro-batches but NEVER divided by N before the optimizer step. A misleading comment (added in commit `b42d66065`) claimed "This is the desired behavior" - **it was not**.

**What Standard Frameworks Do:**
| Framework | Method |
|-----------|--------|
| **PyTorch (manual)** | `loss = loss / accum_steps` before `.backward()` |
| **HuggingFace Trainer** | Divides loss by `gradient_accumulation_steps` |
| **DeepSpeed/Megatron-LM** | Scales by `1/num_microbatches` |

**Impact Without Fix:**
- `lr=0.0001` with `accum_steps=2` → effective LR = **0.0002** (non-standard)
- Training 2x more aggressive than configured
- LR tuning becomes meaningless - changing `accumulation_steps` changes effective LR

**Fix Applied (Phase2_TrainingLoop.cu):**
```cpp
if (accum_steps_for_log > 1) {
    const float accum_scale = 1.0f / static_cast<float>(accum_steps_for_log);
    ctx.model->scaleGradients(accum_scale);
}
```

Applied AFTER accumulation complete, BEFORE `updateWeights()`.

**Status:** ✅ **FIX APPLIED** - Rebuild required to test

---

**Previous Status (Dec 28):**
- Loss drops from 10.5 → 8.3-8.8 in first ~50 batches, then plateaus indefinitely to include multiple epochs

### ✅ Issue #25: Diagnostic Entropy Truncation Bug (FIXED - Dec 27)

**Bug Description:**
The `computeAttentionDiagnosticsKernel` computed entropy over **first 128 tokens only** (MAX_ATTEND=128), while actual sequences are 1500-4000+ tokens. This made ALL entropy metrics in diagnostic logs unreliable.

**Evidence:**
- Log showed `seq_len=1491` but entropy only computed on first 128 tokens
- Entropy values H≈8.78-8.89 bits were close to log2(128)=7 bits ceiling
- Only 8.6% of sequence covered (128/1491)
- All "Layer 5 entropy collapse" diagnoses were based on truncated data

**Impact:**
- All previous entropy-based conclusions about Layer 5 were artifacts
- Cannot determine actual attention behavior from diagnostic that only sees 8% of sequence
- Plateau investigation led astray by misleading metrics

**Fix Applied (Flash_Attention_Kernal.cu lines 216-283):**
Replaced fixed-size `float scores[128]` array with three-pass algorithm:
1. Pass 1: Find max_score (no array storage)
2. Pass 2: Compute sum_exp for softmax denominator
3. Pass 3: Compute entropy over FULL causal sequence (qi+1 tokens)

Now computes entropy over actual sequence lengths (1500-4000 tokens) without truncation.

**Status:** ✅ **FIXED** - Entropy diagnostic now covers full sequences

---

## ✅ New Finding: Single-Batch Overfit Test (Dec 28, 2025)

**Test Summary:**
- Ran single-batch mode (repeat the same batch for a fixed number of steps).
- Loss drops substantially at first, then still plateaus (it does not keep improving).
- Telemetry skipped only the first step (single spike); no persistent skip behavior.
- Optimizer updates still apply and weights change during the run.

**What This Rules Out:**
1. **Broken forward/backward path** (the model can fit a fixed batch).
2. **Loss pipeline failure** (loss decreases consistently on repeated data).
3. **Data loader / tokenization corruption** for the chosen batch (same batch is learnable).
4. **Telemetry permanently blocking steps** (only one early skip observed).

**What It Does NOT Rule Out:**
- Multi-batch dynamics (distribution shift effects).
- Optimizer dynamics under varying batches (e.g., AdamW state/bias correction, accumulation/update gating, weight decay).
- Any validation-time or generalization issues.

**What Single-Batch Plateau Further Rules Out:**
1. **Batch diversity or ordering as the sole cause** (plateau happens without batch variation).
2. **Data mixture/pathology-driven collapse** (a fixed batch still stalls).
3. **Shuffle or batch-scheduling artifacts** (no scheduling changes in single-batch mode).

### ❌ Disproven: Label Smoothing + Focal Loss (Dec 28)

**Evidence (loss math):**
```
COMPUTED: alpha*fw*ce*sw = 1.000000 * 1.000000000 * 0.296595 * 0.665662 = 0.197431743
```

**Conclusion:** Focal/label-smoothing weights are neutral in the loss path; plateau persists, so they are not the root cause.

### ❌ Disproven: Valid-Token Handling (Dec 28)

**Evidence:**
- `valid_tokens` is consistently non-zero in GradTrace logs (e.g., `valid_tokens=3069`), and loss still plateaus.

**Conclusion:** Valid-token masking/handling is not the root cause.

### ❌ Disproven: FlashAttention / NaN-Inf Handling (Dec 28)

**Evidence:**
- Attention diagnostics run clean after fixes (no -FLT_MAX anomalies), yet plateau persists.
- NaN/Inf handling/clamping does not remove the plateau.

**Conclusion:** FlashAttention precision behavior and NaN/Inf handling are not the root cause.

### ❌ Disproven: Learning Rate Schedule / Warmup (Dec 28)

**Evidence:**
- Warmup removed (Issue #3) and plateau persists at full LR from step 1.
- With stability overrides enabled, `getScheduledLearningRate()` returns base LR and bypasses warmup/cosine decay; dynamic LR and spike cooldown are disabled when stability overrides are on (`Phase2_TrainingLoop.cu`).

**Conclusion:** LR schedule/warmup behavior is not the root cause.

### ❌ Disproven: Dropout (Not Active in Training Path) (Dec 28)

**Evidence:**
- `launchDropout` is never called (only defined in `Shared/Dropout/Dropout_GPU.cu`).
- FFN dropout is marked "currently unused" and not applied in `Feed_Forward_GPU.cu`.
- FlashAttention dropout is compile-time disabled and instantiated with `Is_dropout=false` (`Flash_Attention_Kernal.cu`).

**Conclusion:** Dropout is effectively off, so it cannot explain the plateau.

**Plateau Status After Test:**
- Multi-batch runs still plateau after the initial drop.
- Single-batch runs also plateau after an initial improvement.
- This means the plateau is not solely caused by batch diversity; it can occur even on a fixed batch, pointing at optimizer dynamics or model capacity/initialization.

### ✅ New Finding: AdamW Denominator Growth / Effective Step Shrink (Dec 28)

**Evidence (training_17669804820195563.log + training_run.txt):**
- OptIO denom mean rises from ~1.19e-4 at step 0 to ~8.35e-4 at step 498 (~7x).
- Mean m_hat/denom factor drops from ~0.509 to ~0.00647 (~79x).
- UpdateMag ratio (update_rms/param_rms) falls from ~4.29e-3 to ~1.20e-4 (~36x).
- OptState v_rms reaches ~1.78e7 by batch 999.

**Interpretation:**
- AdamW implementation is functioning, but large/spiky gradients drive v up, making the adaptive denominator large and shrinking effective updates.
- Plateau is consistent with adaptive damping; the remaining question is why gradients have such large variance/scale.

**Open Hypotheses (Dec 28):**
1. **Gradient scale/variance sources** (loss/logit scale, accumulation/clip behavior, data anomalies) that inflate AdamW v.
2. **Model capacity/initialization limits** (architecture or starting point prevents deeper overfit).

---

## TODOS

**Listed Todo Audits for production ready insurance of bug:**
Mismatch format directory wide
Files need to only use our global controllers like gradaccumulationcontroller/streamcontroller
Organization and consolidation of hyperparamaters


### ✅ Issue #24: Checkpoint Load Uses MHA Dimensions (FIXED! but no the plateau cause)

**Bug Description:**
The `LanguageModel::load()` function in `grim_model_serialization.cu` was using **hardcoded MHA formula** for W_qkv size calculation:
```cpp
assignWrite(view.attn_w_qkv, enc->getAttnWqkv(), d_model * d_model * 3);  // MHA: 768 * 768 * 3 = 1,769,472
assignWrite(view.attn_b_qkv, enc->getAttnBqkv(), d_model * 3);           // MHA: 768 * 3 = 2304
```

But `save()` correctly uses **GQA formula**:
```cpp
const int total_qkv_dim = config_.d_model + 2 * kv_dim;  // GQA: 768 + 2*256 = 1280
const std::size_t qkv_weight_size = total_qkv_dim * d_model;  // GQA: 1280 * 768 = 983,040
```

**Error Seen:**
```
[SerializationLayer::load] Size mismatch for attn.W_qkv (dest=1769472, src=983040)
```

This explains why checkpoints saved with GQA (4 KV heads) cannot be loaded - the load function expects MHA dimensions!

**Fix Applied:**
```cpp
// GQA dimensions for W_qkv sizing - MUST match save() calculation!
// BUG FIX Issue #24: load() was using MHA formula (d_model * 3) but save() uses GQA formula
const int head_dim = config_.d_model / config_.num_heads;
const int kv_dim = config_.num_kv_heads * head_dim;
const int total_qkv_dim = config_.d_model + 2 * kv_dim;  // Q + K + V with GQA
const std::size_t qkv_weight_size = static_cast<std::size_t>(total_qkv_dim) * d_model;

assignWrite(view.attn_w_qkv, enc->getAttnWqkv(), qkv_weight_size);
assignWrite(view.attn_b_qkv, enc->getAttnBqkv(), total_qkv_dim);  // GQA-aware bias size
```

**Impact:** This bug prevented all checkpoint loading when using GQA (num_kv_heads < num_heads).

**Test Required:** Rebuild and verify checkpoint loading works.

---

### ✅ Issue #22: Gradient Accumulation NEVER Accumulates (FIXED!)

**Log File:** `training_17665609741356624.log` (Dec 24, 02:22)

**Bug Description:**
When `accumulation_steps > 1` (e.g., 2), the training loop calls `backward()` with `accumulate=false` for EVERY micro-batch! This causes each micro-batch to **OVERWRITE** the previous batch's gradients instead of adding to them.

**Evidence from log:**
```
[2025-12-24 02:26:44] [GradTrace] ACCUMULATING batch=83 micro_step=1 of 2 (skipping optimizer step)
```

The log shows micro_step=1 of 2, meaning the first micro-batch's gradients were computed then **discarded** when the second micro-batch overwrote them.

**Code Location:** `Phase2_TrainingLoop.cu` line 1506

**Bug:**
```cpp
ctx.model->backward(result.loss, false, grad_scale, ctx.global_step);  // ALWAYS false!
```

The second parameter `accumulate=false` means cuBLAS uses `beta=0.0` which overwrites:
- `C = alpha * A @ B + 0 * C = alpha * A @ B` (gradient buffer is overwritten)

Should be `accumulate=true` for micro_step > 0:
- `C = alpha * A @ B + 1 * C = alpha * A @ B + C` (gradients accumulate)

**Impact:**
- With `accumulation_steps=2`, effective batch size is **HALF** of configured
- Only the LAST micro-batch's gradients are used
- This explains why loss plateau persists - we're training with half the data!
- Gradient variance is 2x higher than expected (single micro-batch instead of averaged)

**Fix Applied:**
```cpp
// BUG FIX Issue #22: Gradient accumulation was NEVER accumulating!
const bool should_accumulate = ctx.optimizer.grad_controller->controller().currentMicroStep() > 0;
ctx.model->backward(result.loss, should_accumulate, grad_scale, ctx.global_step);
```

**Test Required:** Rebuild and verify that:
1. First micro-batch (micro_step=0) uses `beta_accum=0.0` (overwrite)
2. Subsequent micro-batches (micro_step>0) use `beta_accum=1.0` (accumulate)
3. Loss trajectory improves with proper accumulation

---

### ✅ Issue #19: Uninitialized progress_boost - FIXED AND VERIFIED

**Log Files:**
- Before fix: `training_17665568873906950.log` (Dec 24, 01:14)
- After fix: `training_17665576963467769.log` (Dec 24, 01:28)

**Verification Results:**

| Metric | Before Fix | After Fix |
|--------|------------|-----------|
| **Batch 151 Action** | `scale_gradients scale=0.01` ❌ | `continue` ✅ |
| **Batch 151 grad_norm** | `0.0100` ❌ | `1.0000` ✅ |
| **Batch 152 grad_norm** | `0.0100` ❌ | `1.0000` ✅ |

**Evidence (After Fix):**
```
[2025-12-24 01:35:05] [TelemetryControl] batch=151 Action=continue
[2025-12-24 01:35:05] [GradTrace] PRE-OPTIMIZER batch=151 lr=0.00009972 grad_norm=1.0000 step=75
```

**Fix Applied:** Added `decision->progress_boost = 1.0f;` initialization in `TelemetryControl_GPU.cu`

### ❌ Issue #20: gradient_clip=1.0 TOO AGGRESSIVE (DISPROVEN AS ROOT CAUSE)

**Log File:** `training_17665576963467769.log` (Dec 24, 01:28)

**Bug Description:**
The gradient clipping threshold `gradient_clip: 1.0` in `ai_config.json` is too aggressive. Raw gradient norms average 3.5-6.8 but are being clipped to 1.0 every single batch, resulting in:
- **75% gradient loss** on average batches (norm 4.0 → 1.0)
- **Effective LR reduced ~4x** (0.0001 → ~0.000025)
- **Training artificially throttled** - model cannot escape plateau

**Evidence:**
```
[2025-12-24 01:38:20] COMPUTED COMPONENTS: total=6.8408 emb_lm_tied=6.7871 attn=0.6231 ffn=0.5863
[2025-12-24 01:38:20] POST-GRADNORM preclip=6.8408 per_token=6.8408
[2025-12-24 01:38:20] PRE-OPTIMIZER batch=201 lr=0.00009889 grad_norm=1.0000  ← CLIPPED from 6.84!
```

All 211 batches show `grad_norm=1.0000` - every single gradient is being clipped.

**Gradient Norms (pre-clip):**
- Mean: ~4.2
- Range: 2.98 - 6.84
- All clipped to 1.0

**Impact:**
- Model learns slowly because gradient magnitude is artificially capped
- Plateau occurs where loss=8.4-8.8 (perplexity ~5,400 vs vocab 37,555)
- Only ~15% of vocabulary prediction capacity utilized

**Test Result:** Increasing and disabling clipping did not remove the plateau. Clipping/scale path is **not** the root cause.

### 🟡 PLATEAU PERSISTS AFTER ALL GRADIENT BUG FIXES - HYPERPARAMETER ISSUE

**Current Training (training_17665576963467769.log):**
- Loss: 10.57 → ~8.5-8.7 (same plateau range)
- Gradient norms: **Healthy** (staying at 1.0, not dropping to 0.01)
- Gradient components: **Not collapsing** (attn/ffn vary naturally)

**Gradient Component Evolution (Healthy):**
```
Batch 1:   total=1.86  attn=0.62  ffn=0.58  (initial)
Batch 71:  total=12.10 attn=1.59  ffn=1.58  (peak - normal variation)
Batch 151: total=3.66  attn=0.36  ffn=0.35  (stabilized)
Latest:    total=4.46  attn=0.41  ffn=0.39  (healthy variation)
```

**Conclusion:** The gradient zeroing bugs (#18, #19) are **completely fixed**. The model is now training correctly with proper gradient flow. The plateau at ~8.5 is likely:
1. **Architectural limit** of 6-layer model
2. **Learning rate** needs tuning for this capacity
3. **Data/vocab mismatch** or other non-gradient issue

### 📊 All Verified Fixes Summary

| Issue | Bug | Status | Impact |
|-------|-----|--------|--------|
| #19 | Uninitialized `progress_boost` | ✅ FIXED | Gradients no longer zeroed after batch 151 |
| #18 | grad_scale_factor=0 clamp | ✅ N/A | Was masking #19, not separate bug |
| #16 | W_o gradient order | ✅ FIXED | W_o grads now ~0.25 (not 0.0) |
| #17 | SOFTMAX_TEMPERATURE=0.5 | ✅ FIXED | Restored to 1.0 |
| #21 | Softmax Jacobian Attenuation | ✅ FIXED | Q/K/V gradients restored to healthy levels |

### ⏳ Next Investigation: Attention Mechanism Frozen From Initialization

The plateau is **NOT** a gradient bug - gradient magnitudes are healthy. The root cause is:

**Frozen Attention Pattern with Constant Entropy:**
- Entropy **NEVER CHANGES** - stays at H≈6.6 bits/pos from batch 1 onwards
- Attention distribution is stuck at initialization, not learned then frozen
- Gradients are "healthy" (Q:0.01-0.05) but **10-100x weaker** than typical transformers
- Updates too weak to escape - attention mechanism essentially non-functional

**Architectural Solutions from Issue #21 Analysis:**
1. 🚫 **QK-normalization** - ❌ **DISPROVEN - DOES NOT FIX PLATEAU** (tested extensively Dec 22-27)
2. **Entropy regularization** - Penalize uniform attention → encourage exploration
3. **Sliding window attention** - Force locality → break frozen pattern
4. **Per-head temperature learning** - Allow different heads to specialize

---

## 🚫 CRITICAL: QK-NORMALIZATION DOES NOT FIX PLATEAU

**Status:** ❌ **EXTENSIVELY TESTED AND DISPROVEN** (Dec 22-27, 2025)

**What Was Tested:**
- Enabled `QK_NORMALIZATION_ENABLED = true` in HyperParameters_GPU.hpp
- Ran multiple training sessions with QK-normalization active
- Monitored gradient flow, attention patterns, and loss curves

**Results:**
- ❌ Plateau PERSISTS with QK-normalization enabled
- ❌ Layer 5 entropy still abnormally low (1-2 bits vs 8-9 bits for L0-L4)
- ❌ Loss still stalls at ~8.3-8.8 after initial drop
- ❌ NO improvement in gradient magnitudes or training dynamics

**Conclusion:**
QK-normalization is **NOT THE SOLUTION** to the plateau bug. While it may have theoretical benefits for attention sharpness, extensive empirical testing shows it does not resolve the fundamental issue causing training to stall.

**DO NOT suggest QK-normalization as a fix without NEW evidence that it helps.**

---

## ✅ Issue #21: Softmax Jacobian Attenuation (FIXED - Dec 26)

**Log File:** `attndiag.txt` (steps 261-349) from OLD training run

**Historical Context:** This issue documented Q/K/V gradients at **0.000002** (1000-1500x attenuation)

**Resolution (Dec 26 verification):**
- Pre-PBM-cleanup diagnostic (diagnostic_output.txt): Q:0.046, K:0.012, V:0.035
- Post-PBM-cleanup diagnostic (attndiag.txt): Q:0.044, K:0.011, V:0.039
- **Both runs show 6000x LARGER gradients than Issue #21 documented**
- Issue #21 was ALREADY FIXED by time diagnostic runs occurred
- The "Frankenstein fix" from previous session addressed this, NOT the PBM system

**Status:** ✅ **FIXED** - Gradient magnitudes restored to healthy levels (Q:0.01-0.05, K:0.003-0.012, V:0.02-0.04)

**Original Issue #21 Findings (Dec 24 Night):**

**Log File:** `attndiag.txt` (steps 261-349)

**Critical Finding:** Q/K/V gradients from Flash Attention are **3-4 orders of magnitude smaller** than the incoming gradient!

| Stage | Gradient Norm |
|-------|---------------|
| Incoming to encoder | 0.003 |
| Q gradient (after Flash Attn) | 0.000002 |
| K gradient (after Flash Attn) | 0.000002 |
| V gradient (after Flash Attn) | 0.000011 |

**Attenuation factor:** 1000-1500x drop through attention backward!

**Root Cause: Softmax Jacobian with High Entropy Attention**

The softmax backward formula is:
```
dS = P * (dP - dp_sum) * scale
```

When attention is spread (high entropy), the probabilities P are small (~0.01-0.025), and `dP - dp_sum ≈ 0` for most positions. This creates tiny gradients.

**Evidence from attndiag.txt:**
```
probs=[0.0254, 1.0000]  ← Min prob 2.5%, max 100% (first token to itself)
entropy=81  ← Total bits (~2.5 nats per position = attention spread over ~12 tokens)
grads=[Q:0.000002 K:0.000002 V:0.000011]  ← Tiny!
```

**Mathematical Analysis:**
- Softmax Jacobian diagonal: `P * (1-P)` ≈ 0.025 * 0.975 ≈ 0.024
- Softmax Jacobian off-diagonal: `-P_i * P_j` ≈ -0.0006
- These small Jacobian values multiply through to Q/K gradients
- Result: Q/K barely update → attention stays spread → self-reinforcing loop

**This is NOT a bug - it's a fundamental property of softmax with high entropy!**

**Why Plateau Occurs (HISTORICAL THEORY - DISPROVEN):**
1. Early training: Random attention is somewhat focused → larger gradients → fast learning
2. As training progresses: Attention spreads (learns "average" pattern) → tiny gradients
3. Plateau: Gradients too small to escape local minimum → stuck

**Historical Note:** This theory suggested QK-normalization as a solution. Extensive testing Dec 22-27, 2025 proved this does NOT fix the plateau. See section above for details.

---

## 🔵 HISTORICAL: Issue #19 Details (FIXED)

**Root Cause Analysis:**

The `ControlDecision` struct has C++ member initializers:
```cpp
// In TelemetryControl_GPU.hpp
struct ControlDecision {
    float progress_boost = 1.0f;  // Default value
    ...
};
```

BUT the CUDA kernel only partially initialized the output:
```cuda
// In TelemetryControl_GPU.cu, controlDecisionKernel()
decision->volatility_damping = 1.0f;
decision->decay_boost = 1.0f;
// decision->progress_boost = ???  ← MISSING! Contains garbage
```

When computing scale factor at line 384:
```cuda
decision->grad_scale_factor = decision->volatility_damping * decision->decay_boost * decision->progress_boost;
// = 1.0 * 1.0 * GARBAGE  → usually near 0
```

**Critical Understanding:** CUDA `cudaMalloc` returns **uninitialized memory**. C++ member initializers do NOT apply to device memory allocations.

**Fix Applied:**
```cuda
// In controlDecisionKernel(), after other initializations:
decision->progress_boost = 1.0f;  // FIX Issue #19: Was uninitialized!
decision->normalized_grad = 0.0f;
decision->_pad0 = 0;
```

---

## 🔵 HISTORICAL INVESTIGATION (Dec 22-23)

---

## 🔴 BUGS FOUND AND FIXED (Historical - Dec 22-23)

### Issue #16: W_o Gradient Computation Order Bug (FIXED - Dec 24) ⭐ ROOT CAUSE

**Location:** `BackwardPhase2_Encoder.cu` lines 897-947

**Bug Description:**
The W_o weight gradient was computed AFTER the input gradient, causing the cached activation to be overwritten before it was used:

```cpp
// Step 1: Store activation in grad_attn_out_flat
TensorContract::convert(cached_attn_bhsd, grad_attn_out_flat, ...);  // grad_attn_out_flat = activation

// Step 2: Compute input gradient (OVERWRITES grad_attn_out_flat!)
cublasSgemm(..., W_o^T, grad_attn_output, grad_attn_out_flat);  // grad_attn_out_flat = W_o^T @ grad_output

// Step 3: Compute weight gradient (WRONG - uses gradient instead of activation!)
cublasSgemm(..., grad_attn_out_flat, grad_attn_output, grad_W_o);  // grad_W_o = GRADIENT^T @ grad_output ❌
```

**Impact:**
- W_o gradients computed as `grad^T @ grad` instead of `activation^T @ grad`
- Result: W_o gradient norm = 0.0 (should be ~0.25)
- Attention output projection **never learned** - weights froze at initialization
- This alone could explain the plateau: 1/3 of attention parameters (W_o) not updating

**Evidence:**
```
[GradExport] layer0_wo_grads: norm=0.000000  ❌ BEFORE FIX
[GradExport] layer0_wo_grads: norm=0.263918  ✅ AFTER FIX
```

**Fix Applied:** Swapped computation order - weight gradient computed FIRST, then input gradient.

**Test Result:** ⏳ **PENDING** - Rebuild needed to test

---

### Issue #17: SOFTMAX_TEMPERATURE = 0.5 Diagnostic Override (FIXED - Dec 24) ⭐ ROOT CAUSE

**Location:** `HyperParameters_GPU.hpp` line 59

**Bug Description:**
SOFTMAX_TEMPERATURE was hardcoded to 0.5 (diagnostic override to expose gradient issues), but this sharpens softmax and causes:
- Attention saturation (most weight on 1-2 tokens)
- Vanishing gradients through attention backward
- Training instability (loss spikes 10→17 observed)

**Impact:**
- Standard transformer temperature is 1.0 (no scaling)
- 0.5 = 2x sharpening → near-one-hot attention → gradient collapse
- Combined with W_o bug, model couldn't learn stable attention patterns

**Fix Applied:** Restored to 1.0 (standard scaling)

**Test Result:** ⏳ **PENDING** - Rebuild needed to test

---

## 🔵 OLDER BUGS (Dec 22-23)

### Issue #10: FFN Weight Gradient Shape Transposed (FIXED - Dec 22)

**Location:** `BackwardPhase2_Encoder.cu` lines 639-655 (grad_W2) and 732-742 (grad_W1)

**Bug Description:**
The cuBLAS GEMM calls for FFN weight gradients had M and N dimensions swapped, producing gradients with **transposed shape**:
- `grad_W1`: Was [d_model, d_ff] row-major, should be [d_ff, d_model] row-major  
- `grad_W2`: Was [d_ff, d_model] row-major, should be [d_model, d_ff] row-major

**How It Was Found:**
User noted "we transpose row-major/column-major here" during AdamW oscillation analysis. Compared `BackwardPhase2_Encoder.cu` GEMM calls with correct implementation in `Feed_Forward_GPU.cu` and found M/N swapped.

**Impact:**
- Every weight update applied gradients to **wrong elements** - element `[i,j]` applied to `[j,i]`
- Gradient direction was effectively **perpendicular** to true gradient
- Could cause the "oscillating gradients" pattern where AdamW `m` cancels out while `v` accumulates

**Code Before (WRONG):**
```cuda
// grad_W2 - dimensions were swapped!
cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T,
    cfg->d_model, cfg->d_ff, total_tokens,  // M=d_model, N=d_ff WRONG
    &alpha,
    grad_ffn_output, cfg->d_model,
    cached_ffn_hidden, cfg->d_ff,
    &beta_accum,
    ts->ffn_w2_grads[layer], cfg->d_model);  // ldc=d_model WRONG
```

**Code After (CORRECT):**
```cuda
// grad_W2 - matches Feed_Forward_GPU.cu
cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T,
    cfg->d_ff, cfg->d_model, total_tokens,  // M=d_ff, N=d_model CORRECT
    &alpha,
    cached_ffn_hidden, cfg->d_ff,           // A = post_gelu
    grad_ffn_output, cfg->d_model,          // B = grad_output
    &beta_accum,
    ts->ffn_w2_grads[layer], cfg->d_ff);    // ldc=d_ff CORRECT
```

**Same fix applied to grad_W1.**

**Test Result:** ❌ **PLATEAU PERSISTS**
- Loss still plateaus at ~8.5 after initial drop
- Fix was verified correct (matches Feed_Forward_GPU.cu reference implementation)
- Bug was real but not the root cause of plateau

---

### Issue #11: AdamW Beta2 Too High for Oscillating Gradients (ADJUSTED - Dec 22)

**Location:** `AdamW_Kernal_GPU.cu` line 15

**Theory:**
When gradients oscillate (sign flips between batches), AdamW's first moment `m` cancels out while second moment `v` accumulates the variance. With beta2=0.999, the half-life of `v` is ~693 steps, meaning old variance persists too long. This causes `sqrt(v) >> |m|` and effective updates collapse.

**Change Made:**
- `ADAMW_BETA2`: 0.999 → 0.99 (10x faster forgetting, half-life ~69 steps)

**Test Result:** ❌ **PLATEAU PERSISTS**
- Lower beta2 did not resolve the plateau
- The oscillating gradient hypothesis may still be valid but beta2 alone doesn't fix it

---

### Issue #9: FFN Post-GELU Activation Never Cached (FIXED - Dec 22)

**Location:** `Encoding_GPU.cu` in `EncodingLayer::forward()` (Step 9: FFN Forward)

**Bug Description:**
- The `EncodingForwardArgs` struct has a `cache_ffn_output` field designed to store the post-GELU activation
- This field is passed from `Forward_GPU.cu` as `args.cache_ffn_output = layer_caches[layer_idx].ffn_output`
- **CRITICAL:** `EncodingLayer::forward()` NEVER writes to `args.cache_ffn_output`
- Result: `training_state_.cached_ffn_outputs[layer]` contains **uninitialized/garbage memory**

**Impact on Backward Pass:**
1. `BackwardPhase2_Encoder.cu` line 323: `float* ffn_output = ts->cached_ffn_outputs[layer];` → gets garbage
2. Line 381: `computeFFNBackward(ctx, layer, grad_ffn_input, ln2_output, ffn_output)` → passes garbage as `cached_ffn_hidden`
3. `computeFFNBackward()` line 653: `cublasSgemm(..., cached_ffn_hidden, ...)` for `grad_W2 = ffn_hidden^T @ grad_ffn_output`
4. **Result:** W2 weight gradients computed using garbage data → corrupted FFN gradients

**Evidence:**
- FFN gradient norm was **4.3x smaller** than expected based on parameter count ratio
- Expected attn:ffn ratio (by sqrt(params)): 1.0 : 1.74
- Actual ratio from batch 1: 1.0 : 0.40 → **4.3x smaller than expected**
- FFN gradients collapsed further during training (batch 111: 1.0 : 0.17)
- Total gradient norm stayed healthy, but FFN component was systematically corrupted

**Fix Applied:**
```cuda
// In Encoding_GPU.cu, after ffn_->forward(ffn_args, nullptr);
// BUG FIX: Cache post-GELU output for backward pass (required for grad_W2 computation)
if (args.cache_ffn_output) {
    CUDA_CHECK(cudaMemcpyAsync(args.cache_ffn_output, post_gelu,
                                static_cast<std::size_t>(total_tokens) * config_.d_ff * sizeof(float),
                                cudaMemcpyDeviceToDevice, stream));
}
```

**Status:** Fixed, tested - plateau persists

---

### Issue #12: AdamW Bias Correction Beta2 Mismatch (FIXED - Dec 22)

**Location:** `AdamW_Kernal_GPU.cu` lines 77-78 (host function `launchAdamWKernel`)

**Bug Description:**
The bias correction computation used **hardcoded** beta values (0.9, 0.999) while the kernel used **different** beta constants (ADAMW_BETA1=0.9, ADAMW_BETA2=0.99):

```cuda
// BEFORE (BUG):
const float bias_correction1 = 1.0f - powf(0.9f, ...);   // Matches ADAMW_BETA1 ✓
const float bias_correction2 = 1.0f - powf(0.999f, ...); // MISMATCH! Kernel uses 0.99
```

**Impact:**
- Kernel computes `v_new = 0.99 * v_old + 0.01 * grad²` (fast decay)
- Host computes `v_hat = v_new / (1 - 0.999^t)` (slow correction)
- Mismatch causes:
  - Early training: bias_correction2 is too small → v_hat over-amplified
  - Late training: bias_correction2 is too large → v_hat under-amplified
- Effective learning rate is **inconsistent** across training timeline

**Fix Applied:**
```cuda
// AFTER (FIXED):
const float bias_correction1 = 1.0f - powf(ADAMW_BETA1, ...);  // Use constant
const float bias_correction2 = 1.0f - powf(ADAMW_BETA2, ...);  // Use constant
```

Also moved `ADAMW_BETA1`, `ADAMW_BETA2`, `ADAMW_EPSILON` outside anonymous namespace so `launchAdamWKernel` can access them.

**Status:** Fixed, plateau persists

---

### Issue #13: Token-Normalized Clipping Scales UP Instead of DOWN (CRITICAL - Dec 22)

**File:** `Phase2_TrainingLoop.cu` lines 1566-1577

**Bug Description:**
The gradient clipping logic computes `clip_coef` incorrectly, causing gradients to be scaled **UP** instead of **DOWN**:

```cpp
// CURRENT (BUG):
float effective_clip = clip_selection.effective_clip_norm * telemetry_decision.grad_scale_factor;
// effective_clip = per_token_limit * token_count = 1.0 * 3000 = 3000

if (result.normalized_grad_norm > effective_per_token_limit) {
    const float clip_coef = effective_clip / (result.grad_norm + 1e-8f);
    // clip_coef = 3000 / 4.5 = 667x  <-- SCALES UP!
    ctx.model->scaleGradients(clip_coef);
}
```

**Evidence from log:**
```
POST-GRADNORM preclip=4.5301 per_token=4.5301
PRE-OPTIMIZER batch=71 ... grad_norm=3120.0000  <-- Scaled UP to 3120!
```

The gradient norm was 4.5 but became 3120 after "clipping". This is a 693x increase!

**Impact:**
- Gradients are multiplied by ~500-700x every batch
- This causes massive overshooting, explaining the loss oscillation ("sine wave")
- Even though the optimizer later normalizes by variance, the initial gradient magnitudes cause instability
- This has been silently corrupting training since the TNC clipping was added

**Correct Logic:**
The intent of token-normalized clipping is: if per-token gradient exceeds threshold, scale DOWN to threshold.
```cpp
// FIXED:
if (result.normalized_grad_norm > effective_per_token_limit) {
    // Scale DOWN to per_token_limit, not up to effective_clip_norm
    const float clip_coef = effective_per_token_limit / (result.normalized_grad_norm + 1e-8f);
    ctx.model->scaleGradients(clip_coef);
    result.grad_norm *= clip_coef;  // Reflect actual post-clip norm
    result.normalized_grad_norm = effective_per_token_limit;
    result.gradient_clipped = true;
}
```

**Status:** Fixed, plateau persists

---

### Issue #14: TelemetryLattice Uses POST-CLIP Gradient Norm (CRITICAL - Dec 23)

**File:** `Phase2_TrainingLoop.cu` lines 1749-1751

**Bug Description:**
The TelemetryLattice receives the **post-clip** gradient norm (`result.grad_norm` after line 1581 multiplies by `clip_coef`), but the TelemetryControl evaluates **pre-clip** gradient norm. This creates a mismatch:

- TelemetryLattice baseline: ~1.0 (post-clip value, always clamped to `per_token_limit`)
- TelemetryControl input: ~3-12 (actual pre-clip gradient magnitude)
- Spike ratio: `3.0 / 1.0 = 3.0x` = mild, `12.0 / 1.0 = 12.0x` = **SEVERE**

**Evidence from log (training_17664701159314762.log):**
```
[TelemetryLattice] PRE-UPDATE batch=1 step=0 grad_norm=1.000000  <-- POST-CLIP (always 1.0)
[GradTrace] POST-GRADNORM preclip=8.4764 per_token=8.4764
[TelemetryControl] batch=21 Action=skip_step spike=severe(ratio=12.4293)  <-- Thinks it's 12x spike!
```

**Impact:**
- **62% of all batches were SKIPPED** due to false "severe spike" detection
- Training 17664701159314762.log: 379 skipped out of 610 batches
- The model literally wasn't learning because the optimizer never ran
- This is the ROOT CAUSE of the plateau after Issue #13 fix

**Root Cause:**
```cuda
// Line 1581 - BEFORE telemetry update
result.grad_norm *= clip_coef;  // Post-clip = 1.0

// Lines 1749-1751 - telemetry receives POST-CLIP value
float observations[5] = {
    result.loss,
    result.grad_norm,  // <-- Always ~1.0 (clamped)
    ...
};
```

The spike detection compares current (pre-clip) to baseline (post-clip=1.0), so normal gradients of 5-10 look like 5-10x spikes.

**Fix Applied:**
```cuda
float observations[5] = {
    result.loss,
    preclip_grad_norm,  // Use PRE-CLIP value for accurate baseline
    preclip_grad_norm,  // Stream 2 same
    ...
};
```

**Status:** FIXED (Dec 23) - Applied at Phase2_TrainingLoop.cu lines 1731-1759, using `preclip_grad_norm` for telemetry update instead of post-clip `result.grad_norm`

**Expected Outcome After Testing:**
- TelemetryLattice baseline now matches actual gradient magnitudes
- Spike detection ratios will be ~1.0-1.5x (normal variation) instead of 10-20x (false positives)
- Optimizer steps should run for 95%+ of batches instead of being skipped 62%
- Model should train normally without false plateau

---

### Issue #15: atomicMax Bug with Negative Floats (FIXED - Dec 23)

**Location:** `Flash_Attention_Kernal.cu` lines 221-232 (diagnostic kernel) and 345-351 (K-tensor stats)

**Bug Description:**
The diagnostic kernel used `atomicMax(reinterpret_cast<int*>(&float_var), __float_as_int(value))` to find max QK scores. This **fails catastrophically for negative floats** due to IEEE 754 vs two's complement mismatch:

```
-0.69f bits:   0xBF30A3D7 → int: -1087503913
-FLT_MAX bits: 0xFF7FFFFF → int: -8388609 (closer to zero, so "wins" atomicMax)
```

**Result:** `-FLT_MAX` incorrectly selected as maximum over `-0.69`, appearing in diagnostics as:
```
[AttnDiag] step=20 layer=4: qk=[-0.69, -FLT_MAX]  # Should be qk=[-0.69, -0.01]
```

**Evidence:**
- Pattern appeared at step 20, layer 4 when batch 21 had longest sequence (1905 tokens)
- `qk_min` was normal (-0.69) but `qk_max` was -FLT_MAX (broken)
- Root cause diagnosis: "qk_min normal but qk_max=-FLT_MAX → atomicMax bug (int compare on negative float)"

**Fix Applied:**
Added proper float atomics using CAS loops:
```cuda
__device__ __forceinline__ float atomicMaxFloat(float* address, float val) {
    int* address_as_int = reinterpret_cast<int*>(address);
    int old = *address_as_int, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_int, assumed,
            __float_as_int(fmaxf(val, __int_as_float(assumed))));
    } while (assumed != old);
    return __int_as_float(old);
}

__device__ __forceinline__ float atomicMinFloat(float* address, float val) {
    int* address_as_int = reinterpret_cast<int*>(address);
    int old = *address_as_int, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_int, assumed,
            __float_as_int(fminf(val, __int_as_float(assumed))));
    } while (assumed != old);
    return __int_as_float(old);
}

// Replace all atomicMax/Min calls for floats:
atomicMaxFloat(&qk_max, local_qk_max);
atomicMinFloat(&qk_min, local_qk_min);
```

**Test Result:** ✅ **BUG VERIFIED FIXED**
- Re-ran `analyze_attention_diagnostics.py --auto` on latest training data
- **Result:** "✅ No -FLT_MAX anomalies detected" across 1200 entries
- QK ranges now normal: 0.89-2.72 across all layers
- Entropy healthy: 79.38-81.26 (no collapse)
- Gradients flow properly: L5: 2.4e-06 → L0: 1.5e-05

**Status:** ❌ **PLATEAU PERSISTS AFTER FIX**
- This was a **diagnostic bug only** (didn't affect training forward/backward passes)
- Loss still plateaus at ~8.3-8.8 after initial drop from ~10.5
- Root cause of plateau is elsewhere

---

## Architecture Reference

```
vocab_size:     37,555
d_model:        768
num_layers:     12
num_heads:      12
num_kv_heads:   4 (GQA 3:1 ratio)
d_ff:           3,072
head_dim:       64
tie_embeddings: true
use_bias:       false
```

---

### Issue #7: Gradient Clipping Too Aggressive (FIXED - Dec 24)

**Log File:** Multiple runs through Dec 24

**Bug Description:**
The gradient clipping threshold `gradient_clip: 5.0` in `ai_config.json` was crushing natural gradient magnitudes. Raw gradient norms consistently ~5000-6000 but being clipped to 5.0.

**Evidence:**
- Pre-clip gradient norms: 5000-6000 (natural scale for 104M parameter model)
- Post-clip: 5.0 (1000x reduction!)
- This effectively reduced learning rate by 1000x

**Impact:**
- Model could not escape plateau because gradient magnitude was artificially crushed
- Effective LR: 0.0001 → ~0.0000001 due to clipping
- Not a learning rate problem, but a gradient magnitude problem

**Fix Applied:**
Changed `gradient_clip: 5.0` → `gradient_clip: 0.0` in `ai_config.json` to disable clipping entirely

**Status:** ✅ FIXED - Clipping disabled, plateau persists

---

### Issue #8: Content-Based Loss Weighting (NON-STANDARD - REMOVED Dec 24)

**Location:** `Phase2_TrainingLoop.cu` lines 574-604, 1050-1070

**Bug Description:**
Training loop had custom heuristic system to downweight "Boilerplate" and "MixedJunk" sequences:
- Decoded every batch's sequences via tokenizer to classify them
- Applied pattern matching for `"}} "`, `"{{ "`, `"<url>"`, etc.
- Downweighted "AtomHeavy" sequences (counterproductive to ScratchBlock design)
- Variable loss weights per sequence

**Why This is Wrong:**
- **Not standard practice** - GPT/LLaMA/Mistral weight all sequences equally
- **Performance overhead** - Unnecessary CPU decoding work every batch
- **Bandaid for bad data** - Data quality should be fixed in pipeline, not training loop
- **Training instability** - Variable weights cause gradient variance issues
- **Undermines ScratchBlock** - AtomTable specifically designed to handle structural tokens

**Fix Applied:**
Removed entire content-based weighting system:
- Deleted `SequenceClass` enum
- Deleted `classifySequence()` function (~75 lines)
- Deleted `computeContentLossWeights()` function (~30 lines)
- Deleted loss weight application code (~17 lines)

**Status:** ✅ REMOVED - Training now follows standard LLM practice, plateau persists

---

### Issue #9: Deprecated Tokenizer Backwards Compatibility (REMOVED Dec 24)

**Location:** `Layers/Tokenizer/` folder

**Bug Description:**
Entire `Layers/Tokenizer/` folder contained backwards compatibility wrappers:
- `GrimTokenizer.hpp` - 27-line alias to `Tokenizer::UniByte`
- `Tokenizer_GPU.cu/hpp` - Deprecated legacy BPE tokenizer
- Production code used `GRIM::GrimTokenizer` which just aliased to `Tokenizer::UniByte`

**Why This is Wrong:**
- Unnecessary indirection (alias to alias)
- Legacy code marked "DEPRECATED" still included in builds
- CMake referenced deleted files causing link errors

**Fix Applied:**
- Deleted entire `Layers/Tokenizer/` folder
- Updated all production code to use `GRIM::Tokenizer::UniByte` directly
- Updated CMakeLists.txt to remove Tokenizer_GPU.cu references
- Fixed legacy files (convert_vocab_to_binary.cpp, main_data_collection.cpp, causality_proof_tests.cu)

**Status:** ✅ REMOVED - Clean direct usage of UniByte tokenizer

---

## ✅ VERIFIED CORRECT (Not the Problem)

### 1. Parameter Counts
- **Embedding params:** 28,842,240 ✓ (37,555 × 768)
- **Encoder params:** 75,515,904 ✓ (6,292,992 per layer × 12)
- **LM head params:** 0 ✓ (tied to embeddings)
- **Total:** 104,408,320 ✓
- **No biases:** Confirmed intentional (architecture decision)

### 2. GQA Configuration
- Model initializes with correct GQA: `num_heads=12, num_kv_heads=4`
- W_qkv size: 983,040 elements (1280 × 768) ✓
- Checkpoint/model now use matching GQA dimensions

### 3. Loss Baseline
- Initial loss: 10.49 ≈ ln(37,555) = 10.53 ✓
- Random baseline matches expected value

### 4. Weight Updates
- Weights ARE changing (delta ≈ LR × sign(grad) for early AdamW batches)
- LM head weights update every batch
- Early AdamW behavior (unit-gradient normalization) is expected

### 5. Weight Tying Implementation
- `embedding_grads == lm_head_weight_grads` (same pointer) ✓
- LM head backward writes first (cuBLAS beta=0)
- Embedding backward accumulates (atomicAdd)
- Destructor correctly handles aliased pointers (no double-free)

### 6. Gradient Zeroing
- All gradient buffers zeroed before each backward pass via `zeroGradients()`
- Verified in `LanguageModel_Training.cu`

### 7. cuBLAS Beta Values
- `beta_accum = accumulate ? 1.0f : 0.0f` ✓
- Set correctly in `initBackwardContextRaw()`
- Currently `accumulate=false` which is correct for single-pass (no gradient accumulation)

### 8. Checkpoint Loading (Now Fixed)
- Phase1_Startup.cu now loads `num_kv_heads` from ai_config.json
- Serialization logging now shows actual checkpoint values

### 9. Embedding Layer (Comprehensive Test Suite - Jan 11, 2026)

**39/39 Tests Passed** - Complete diagnostic test suite verified:

| Category | Tests | Status |
|----------|-------|--------|
| Xavier Init | 2 | ✅ Pass |
| Lookup (forward) | 2 | ✅ Pass |
| Backward (gradients) | 2 | ✅ Pass |
| Layer Integration | 1 | ✅ Pass |
| Boundary/Memory | 2 | ✅ Pass |
| Integration (GRMT data) | 3 | ✅ Pass |
| Weight Tying | 1 | ✅ Pass |
| RMSNorm | 3 | ✅ Pass |
| Position Embeddings | 4 | ✅ Pass |
| Special Tokens | 1 | ✅ Pass |
| Gradient Tests | 10 | ✅ Pass |
| Edge Cases | 5 | ✅ Pass |
| Stability | 1 | ✅ Pass |
| Concurrent Streams | 1 | ✅ Pass |

**Key Verified Behaviors:**
- ✅ **Finite difference gradient check** - 100% pass, max relative error = 0.00
- ✅ **Gradient accumulation** - atomicAdd correctly accumulates across backward passes
- ✅ **Position embedding consistency** - Same position gets same embedding across all batches (Issue #19 regression check)
- ✅ **Weight tying gradients** - Flow correctly through RMSNorm to tied weights
- ✅ **Token ID validation** - Rule 20 Fail Loud correctly rejects invalid tokens
- ✅ **High contention atomicAdd** - Deterministic and accurate under 4096-way contention
- ✅ **Very long sequences** - 8192 tokens work at 5.3M tokens/sec, no NaN/Inf
- ✅ **Batch independence** - Batched results exactly match independent single-batch runs
- ✅ **Real GRMT data** - Forward pass works correctly with actual training sequences

**Conclusion:** Embedding layer is **NOT the cause** of training plateau. All forward/backward computations, gradient accumulation, weight tying, and position embeddings verified correct.

### 10. UnigramByte Tokenizer (Comprehensive Test Suite - Jan 11, 2026)

**67/67 Tests Passed** - Complete diagnostic test suite verified:

| Category | Tests | Status |
|----------|-------|--------|
| Byte Encode/Decode | 5 | ✅ Pass |
| Unigram LM | 5 | ✅ Pass |
| Aho-Corasick DFA | 5 | ✅ Pass |
| UniByte Orchestrator | 8 | ✅ Pass |
| AtomTable | 17 | ✅ Pass |
| Integration | 3 | ✅ Pass |
| Edge Cases | 5 | ✅ Pass |
| Unicode & Emoji | 3 | ✅ Pass |
| Multiple Structural | 4 | ✅ Pass |
| Path Detection | 2 | ✅ Pass |
| Numeric Edge Cases | 3 | ✅ Pass |
| Byte Fallback Control | 2 | ✅ Pass |
| Vocabulary Persistence | 3 | ✅ Pass |
| GPU Decode | 1 | ✅ Pass |

**Key Verified Behaviors:**
- ✅ **Byte fallback** - UTF-8 round-trip encoding/decoding works correctly (100% coverage)
- ✅ **Unigram Viterbi** - Optimal subword segmentation via dynamic programming
- ✅ **Aho-Corasick DFA** - O(n) multi-pattern matching for structural token detection
- ✅ **URL detection** - Case-insensitive detection of http://, https://, www., ftp://
- ✅ **Email detection** - Correct @ symbol and mailto: handling
- ✅ **Date/Time detection** - Various formats recognized and tokenized
- ✅ **Number detection** - Integers, floats, scientific notation, negative numbers
- ✅ **IP address vs decimal** - Correctly distinguishes 192.168.1.1 from 3.14159
- ✅ **Path detection** - Windows (C:\) and Unix (/) paths recognized
- ✅ **AtomTable registration** - All atom types (int, float, hex, binary, URL, email, date, time, IP, path, string, identifier)
- ✅ **AtomTable GPU upload** - Device memory correctly populated
- ✅ **Hash deduplication** - Duplicate atoms share same ID
- ✅ **Placeholder injection** - Atom placeholders [256-511] correctly injected
- ✅ **Vocabulary persistence** - Text and binary save/load work correctly
- ✅ **GPU decode** - Device-side token-to-text reconstruction

**Conclusion:** UnigramByte tokenizer is **NOT the cause** of training plateau. All tokenization, structural detection, atom handling, and GPU operations verified correct.

---

## 🔴 KNOWN ISSUES FOUND

### Issue #1: Gradient Doubling in Tied Embeddings (FIXED)
**File:** `BackwardPhase3_InputLayer.cu` line ~272  
**Bug:** When `tie_embeddings=true`, code did `launchResidualAdd(A, A, A)` which doubled gradients  
**Fix Applied:** Skip merge when `embedding_grads == lm_head_weight_grads` (aliased)  
**Status:** Fixed, needs rebuild and retest

### Issue #2: Weight Tying Gradient Imbalance (DISPROVEN)
**Hypothesis:** Weight tying causes LM head to receive **both** output gradients and input gradients (accumulated), making its effective gradient ~2x larger than it should be relative to encoder layers.

**Test Conducted:** Set `tie_embeddings: false` in ai_config.json, rebuilt model with separate embedding/LM head weights (95.5M total params vs 66.6M tied)

**Results:** ❌ **HYPOTHESIS DISPROVEN**
- Plateau **persists** with untied weights (loss 10.57 → 8.44 in 151 batches, then stalls)
- Gradient components all decrease **proportionally** (no imbalance):
  - Batch 1: `emb=0.0062 lm=1.6561 attn=0.3530 ffn=0.1428 rms=0.0037`
  - Batch 141: `emb=0.0056 lm=1.4730 attn=0.3216 ffn=0.1021 rms=0.0026`
  - All components drop ~10-30% together (no dominance)

**Conclusion:** Gradient imbalance from weight tying is **NOT the root cause**. The plateau is a deeper architectural or optimization issue unrelated to tied embeddings.

---

### Issue #3: Learning Rate Warmup + Optimizer State Corruption (DISPROVEN)
**Hypothesis:** Ultra-low warmup LR (2e-6 starting, 50 steps) creates unstable AdamW momentum/variance estimates during early training. Bias correction amplifies tiny m/v values, causing optimizer state to get trapped in local minimum.

**Evidence (Initial):**
- Training log `17664237716101789.log` showed `lr=0.00000200` at batch 1 (50x lower than base 1e-4)
- AdamW bias correction: `step=1 → correction1=0.1, correction2=0.001` (10-1000x amplification)
- Plateau occurred in batches 1-50 (during warmup phase)

**Test Conducted:** Removed warmup entirely (`warmup_steps: 0`), started at full LR (1e-4) from batch 1

**Results:** ❌ **HYPOTHESIS DISPROVEN**
- Training log `17664247970142263.log`: `lr=0.00010000` at batch 1 (full LR immediately)
- Plateau **still occurs** in same loss range (8.4-8.9) at batch ~170
- Gradient components still decay proportionally:
  - Batch 1: `emb_lm_tied=1.6561 attn=0.3530 ffn=0.1428 rms=0.0037`
  - Batch 171: `emb_lm_tied=1.1427 attn=0.2759 ffn=0.0840 rms=0.0021`
  - All components drop ~30% together (proportional)

**Additional Finding:** DynamicLR controller **decreased** LR during plateau (1e-4 → 8.3e-5), making problem worse! This suggests optimizer/controller doesn't recognize plateau as needing exploration.

**Conclusion:** Warmup schedule and early optimizer state corruption are **NOT the root cause**. Plateau occurs regardless of initial LR or warmup strategy. Problem is likely deeper: optimization dynamics or model architecture.

---

### Issue #4: Training Data Diversity (DISPROVEN)
**Hypothesis:** Training data lacks diversity, causing all batches to produce nearly identical gradient directions. Evidence from gradient curvature diagnostic showed 99.45% gradient alignment (cosine similarity).

**Test Conducted:** Created comprehensive data quality inspector (`inspect_training_data_quality.py`) to analyze GRMT training data with multiple diversity metrics:
- Token distribution analysis (Zipf's law, frequency concentration)
- Sequence-level diversity (duplicates, length distribution)
- N-gram repetition patterns (2/3/4-grams)
- Pairwise sequence similarity (Jaccard index)
- Structural pattern detection (common prefixes)

**Results:** ✅ **DATA QUALITY IS GOOD - HYPOTHESIS DISPROVEN**

**Training Data Statistics:**
- **Sequences:** 6,733 unique (0% duplicates)
- **Total tokens:** 8,594,154
- **Unique tokens:** 28,352 (75.5% of vocab coverage)
- **Sequence similarity:** Mean Jaccard = 0.10 (only 10% token overlap)
- **Prefix diversity:** 98-99% unique prefixes at all lengths (5/10/20 tokens)
- **High similarity pairs:** 0 out of 4,950 pairs (all <0.23)
- **N-gram repetition:** Normal for natural text (2-gram 76%, 3-gram 36%, 4-gram 17%)
- **Zipf exponent:** 0.726 (slightly low but within acceptable range)
- **No structural issues:** No HTML artifacts, boilerplate text, or pattern contamination

**Key Findings:**
1. Sequences are genuinely diverse (mean similarity 10%, all pairs <23%)
2. No duplicate sequences or common prefix clustering
3. Token distribution follows natural language patterns
4. Sequence length variance is healthy (mean=1276, std=483, range=6-2492)

**Conclusion:** Training data diversity is **NOT the root cause** of 99.45% gradient alignment. Possible sources considered:
1. **Model architecture:** Weight tying, GQA attention patterns, or residual connections
2. **Optimization dynamics:** AdamW momentum accumulation or update gating

**Recommended Next Investigation:** Batch composition strategy was tested in Issue #5 and is not the root cause.

---

### Issue #5: Batch Composition Strategy (DISPROVEN)
**Hypothesis:** `SIMILARITY_GROUPED` batching creates length-based clusters that collapse effective diversity from 6,733 sequences → ~20-30 length buckets, causing 99.45% gradient alignment.

**How SIMILARITY_GROUPED Works:**
```cpp
// From Batching_GPU.cu line 495-497
if (opts.strategy == PackingStrategy::SIMILARITY_GROUPED) {
    breaks_similarity = !areSimilarLengths(current.min_seq_len, seq_len, 0.30f);
    // Only allows sequences within 30% length variance in same batch
}
```

**Theory:**
1. **Length Clustering**: Sorts all sequences by length → groups into ~20-30 length buckets
2. **Batch Construction**: Consecutive batches from same bucket have identical lengths (e.g., all 1591-1591 tokens)
3. **Gradient Math**: Length-similar sequences → similar token distributions → identical gradient directions
4. **Plateau Timing**: First ~20-30 batches learn each cluster, remaining 2,500+ batches repeat same patterns

**Test Conducted:** Changed `batch_strategy` from `SIMILARITY_GROUPED` to `RANDOM` (GREEDY packing) in `ai_config.json`, rebuilt training executable, re-ran training.

**Results:** ❌ **HYPOTHESIS DISPROVEN**
- Training logs `17664301660264094.log` and `17664308988073139.log` both still show plateau
- Loss trajectory identical: 10.57 → ~8.5 in first 50 batches, then stalls
- Gradient component collapse pattern unchanged
- Batching strategy change had **no effect on plateau**

**Conclusion:** Batch composition strategy is **NOT the root cause**. The plateau is deeper than data ordering. Most likely causes remaining:
1. **Vanishing gradients through encoder layers** (attn/ffn/rms collapse while lm_head stays stable)
2. **Model architecture issue** (GQA, residual connections, RMSNorm)
3. **Optimizer dynamics** (AdamW momentum accumulation, update gating, weight decay)

---

### Issue #6: AdamW Step Counter + Bias Correction Division-by-Zero (FIXED but plateau persists)
**Bug #1:** Optimizer step counter incremented TWICE per batch
- `updateWeights()` incremented step at line 542 of `LanguageModel_Training.cu`
- Training loop incremented step again at line 1684 of `Phase2_TrainingLoop.cu`
- **Result:** step = 2×batch_number (e.g., step=237 for batch=119)
- **Impact:** AdamW bias correction used wrong timestep (converged 2x faster than intended)

**Bug #2:** Division by zero when step=0
- AdamW kernel computed `bias_correction1 = 1.0 - beta1^step`
- When step=0: `1.0 - 0.9^0 = 1.0 - 1.0 = 0.0` → divides by zero
- **Result:** All weights became NaN immediately after first optimizer update

**Fixes Applied:**
1. Removed duplicate `optimizer_state->step++` from `updateWeights()` (LanguageModel_Training.cu line 542)
2. Added `step+1` offset in AdamW kernel bias correction to avoid division by zero (AdamW_Kernal_GPU.cu line 76-77)

**Test Results:** ❌ **PLATEAU PERSISTS**
- Training log `17664319304198326.log`: Division by zero fixed (no NaN weights)
- Loss trajectory **UNCHANGED**: 10.57 → ~8.5 in first 50 batches, then stalls
- Plateau occurs at same batch range and loss magnitude as before

**Conclusion:** Step counter bugs were **NOT the root cause** of plateau. They were separate bugs that corrupted training in different ways (wrong bias correction timestep, then NaN explosion), but fixing them did NOT resolve the underlying plateau issue.

---

## � CONFIRMED: VANISHING GRADIENT BUG

### Full Epoch Evidence (12-layer model, 2555 batches)

**Training Log:** `training_17663877358178526.log`
- **Duration:** 6.5 hours (02:15 → 08:59)
- **Start Loss:** 10.49 (random baseline ✓)
- **End Loss:** 8.63 (avg 8.42)
- **Validation:** Loss 8.55, PPL 5158

### Gradient Component Collapse — DEFINITIVE PROOF

| Component | Batch 1 | Batch 51 | Batch 2551 | Collapse |
|-----------|---------|----------|------------|----------|
| **lm_head** | 3.17 | 7.01 | 3.05 | **4%** ✅ STABLE |
| **attn** | **3.94** | **0.84** | 1.38 | **65%** ⚠️ |
| **ffn** | **1.12** | **0.21** | 0.18 | **84%** ⚠️ |
| **rms** | 0.030 | 0.005 | 0.006 | **80%** ⚠️ |

**CRITICAL OBSERVATION:** 
- attn/ffn/rms gradients collapse by **65-84%** in first 50 batches
- LM head gradient spikes then stabilizes (healthy)
- Loss plateaus immediately after encoder gradient collapse
- 0.84^12 ≈ 0.11 (matches the ~10-20% residual we see!)

### Loss Trajectory — Plateau After Batch 50

| Phase | Batches | Loss | Improvement |
|-------|---------|------|-------------|
| **Rapid Learning** | 1-50 | 10.5 → 8.5 | **2.0** |
| **Plateau** | 50-2555 | 8.5 → 8.5 | **0.0** |

**The model learns in 50 batches, then STOPS for 2500+ batches.**

---
## ❌ DISPROVEN: Softmax Temperature Hypothesis

**Test:** Training run `17664369072452783` with:
- SOFTMAX_TEMPERATURE = 1.0 (changed from 4.0)
- ScratchBlock disabled
- 6-layer model

**Result:** Plateau persists!
- Loss: 10.57 → 8.09 (124 batches), then stuck ~8.5
- Gradient collapse: attn 0.35→0.95→0.31, ffn 0.14→0.31→0.08
- Same pattern as before - encoder gradients collapse while emb_lm stays stable

**Conclusion:** Temperature was not the root cause.

---

## ❌ DISPROVEN: Encoder Gradient Collapse (Root Cause)

**Update (Dec 28):** Single-batch plateau shows this is not sufficient to explain the plateau; treat as a symptom, not the cause.

### Observed Pattern (Historical)

**All configurations show same pattern:**

| Test Config | Loss Trajectory | attn Gradient |
|-------------|-----------------|---------------|
| temp=4.0, 12-layer, ScratchBlock ON | 10.6→8.1→stuck | peak→collapse |
| temp=1.0, 6-layer, ScratchBlock OFF | 10.6→8.1→stuck | peak→collapse |

The plateau appears **upstream of** softmax temperature, ScratchBlock, and layer count.

### Gradient Evolution (log `17664369072452783`)

```
Batch   attn    ffn     rms_gamma  emb_lm
1       0.35    0.14    0.0037     1.49
30      0.95    0.31    0.0096     2.10  (peak)
60      0.72    0.18    0.0039     1.75
90      0.44    0.10    0.0027     1.67
121     0.31    0.08    0.0021     1.82  (collapsed)
```

**Key insight:** LM head (emb_lm) stabilizes ~1.8, but encoder COLLAPSES to ~25% of peak.

---

## ✅ CHECKED HYPOTHESES (Not Root Cause)

### Hypothesis 1: Residual Connection Backward Bug (CHECKED - Found FFN bug)

Pre-norm transformers need TWO gradient paths:
1. **Direct residual:** `grad_residual = grad_output` (full magnitude)
2. **Through layer:** `grad_layer = backward_layer(grad_output)` (attenuated)

**Result:** Checked during Issue #10 investigation. Found FFN transpose bug (fixed). Residual paths look correct. Plateau persists.

### Hypothesis 2: Flash Attention GQA Scale (CHECKED - Correct)

GQA (3 Q heads per KV head) requires gradient normalization:
- Backward accumulates 3 Q-heads' gradients into each KV gradient
- `gqa_grad_scale = 1.0f / heads_per_kv_group` should normalize

**Result:** Code inspection shows `gqa_grad_scale` correctly applied to dV (line 857), dK (line 991, 1023). dQ correctly NOT scaled (Q heads are independent). Implementation matches theory.

### Hypothesis 3: RMSNorm Backward Scale (CHECKED - Correct)

RMSNorm backward: `dx = (dy - mean(x * dy) * x / rms²) * gamma / rms`

**Result:** Implementation in `RMSNorm_Kernel_GPU.cu` is mathematically correct. `inv_rms` clamped to 100.0f to prevent explosion.

### Hypothesis 4: AdamW Effective Update (BUG FOUND - Dec 22)

See **Issue #12** below.

---

## ✅ VERIFIED CORRECT: Encoder Backward Residual Merge Logic (Dec 23)

**File:** `BackwardPhase2_Encoder.cu` Steps 1-8

**Pre-Norm Transformer Forward (per layer):**
```
layer_input → RMSNorm1 → QKV → Attention → W_o → + layer_input → residual1
                                                       ↑
                                            (skip connection 1)
residual1 → RMSNorm2 → W1 → GELU → W2 → + residual1 → layer_output
                                             ↑
                                   (skip connection 2)
```

**Mathematical Backward Derivation:**
```
Forward equations:
  residual1 = attn_out + layer_input    (skip 1)
  layer_output = ffn_out + residual1    (skip 2)

Backward (chain rule):
  grad_residual1 = grad_layer_output + grad_through_ffn   (from skip 2)
  grad_layer_input = grad_residual1 + grad_through_attn   (from skip 1)
  
Expanding:
  grad_layer_input = (grad_layer_output + grad_through_ffn) + grad_through_attn
                   = grad_layer_output + grad_through_ffn + grad_through_attn
```

**Code Trace (BackwardPhase2_Encoder.cu):**

| Step | Operation | Result |
|------|-----------|--------|
| **1** | `grad_ffn_input = ctx.current_grad` (copy) | Save grad_layer_output for FFN path |
| **2** | Reconstruct `residual1 = attn_out + layer_input` | Forward reconstruction |
| **3** | FFN backward on `grad_ffn_input` | Computes grad through W2, GELU, W1 |
| **4** | RMSNorm2 backward on `grad_ffn_input` | Now grad w.r.t. residual1 via FFN |
| **5** | `grad_attn_input = grad_ffn_input + ctx.current_grad` | Merge FFN + skip = grad_residual1 ✓ |
| **5b** | `ctx.current_grad = grad_attn_input` | Update to grad_residual1 |
| **6** | Attention backward receives `grad_attn_input` | grad_attn_out = grad_residual1 (correct: d(residual1)/d(attn_out)=1) |
| **7** | RMSNorm1 backward outputs `grad_qkv_input` | Grad w.r.t. layer_input via attention |
| **8** | `ctx.current_grad = grad_qkv_input + ctx.current_grad` | = grad_through_attn + grad_residual1 ✓ |

**Final Result:**
```
ctx.current_grad = grad_through_attn + grad_residual1
                 = grad_through_attn + (grad_through_ffn + grad_layer_output)
                 = grad_through_attn + grad_through_ffn + grad_layer_output  ✓
```

**Conclusion:** The residual merge logic is **MATHEMATICALLY CORRECT**. The encoder gradient collapse is NOT caused by incorrect residual merging.

**Key Insight:** Passing `grad_residual1` to attention backward IS correct because:
- Forward: `residual1 = attn_out + layer_input`
- Therefore: `d(residual1)/d(attn_out) = 1`
- So: `grad_attn_out = grad_residual1 * 1 = grad_residual1`

---

## � NEW FINDINGS (Dec 23 Evening)

### Observation 1: Mysterious Loss=0.0000 Events

**Evidence from logs:**
```
[2025-12-22 14:26:54] Epoch 1/1
[2025-12-22 14:26:52] [GradTrace] POST-FORWARD loss=10.5721
[2025-12-22 14:26:52] [GradTrace] POST-BACKWARD batch=1 loss=10.5721
[2025-12-22 14:26:54] [GradTrace] POST-FORWARD loss=0.0000
[2025-12-22 14:26:54] [GradTrace] POST-BACKWARD batch=2 loss=0.0000
[2025-12-22 14:26:56] [GradTrace] POST-FORWARD loss=0.0000
```

**Count:** 20 occurrences of `loss=0.0000` across training logs

**Impact:**
- Loss drops from 10.5 to 0.0 immediately after first batch
- Subsequent batches stay at 0.0 for entire run
- This is **catastrophic** - indicates loss computation completely broken in some runs

**Potential Causes:**
1. NaN/Inf propagation caught and zeroed by loss clamping **(disproven)**
2. Empty batch (all padding tokens) → no valid predictions  
3. Bug in UnifiedLoss computation when all predictions are correct (unlikely)
4. Loss buffer not properly initialized/copied from GPU

**Status:** Needs urgent investigation - this could be the true plateau cause

---

### Observation 2: Repeated Training Restarts

**Evidence from logs:**
```
[2025-12-22 17:12:02] [ConfigAdjust] auto_stop_plateau_patience 12 -> 1
[2025-12-23 17:20:36] [ConfigAdjust] auto_stop_plateau_patience 12 -> 1
[2025-12-23 18:38:13] [ConfigAdjust] auto_stop_plateau_patience 12 -> 1
[2025-12-23 18:50:30] [ConfigAdjust] soft_restart_max_step_window 50 -> 1531
[2025-12-23 18:50:30] [ConfigAdjust] soft_restart_cooldown_steps 200 -> 1531
```

**Pattern:**
- Plateau detection triggers, reduces patience from 12 → 1
- Soft restart mechanism activates
- Training attempts to recover from checkpoint
- Cycle repeats

**Implications:**
- Training is **repeatedly detecting the plateau** and trying to recover
- Auto-stop mechanism working as designed (detects stuck training)
- Recovery mechanisms (soft restart) not fixing the underlying issue


---

### Observation 3: Current Configuration Analysis

**Focal Loss Settings (ai_config.json):**
```json
"focal": {
    "alpha": 1.0,
    "enabled": true,
    "gamma": 1.0
}
```

**Note:** Gamma=1.0 means focal loss collapses to standard cross-entropy.
- Focal loss formula: `L = α(1-p_t)^γ * CE`
- When γ=1.0: `(1-p_t)^1 = (1-p_t)` -> just scales CE by confidence
- **No hard example focusing** - defeats the purpose of focal loss

**Status (Dec 28):** Disproven as plateau root cause. Loss math confirms focal/label-smoothing multipliers are neutral.

---

### Observation 4: Batch Composition Strategy

**Current Setting:**
```json
"batch_strategy": "SIMILARITY_GROUPED"
```

**Potential Issue:**
- Grouping similar sequences may create **homogeneous batches**
- This could lead to gradient variance collapse (all samples similar → similar gradients)
- Model may overfit to batch patterns rather than learning general features
- High gradient alignment (99.45% from previous tests) could be caused by this

**Data Quality Verified (Dec 22):**
- 6,733 unique sequences
- Only 10% token overlap (Jaccard)
- 0% duplicates
- 98-99% prefix diversity

**Conclusion:** Data is diverse, but batch grouping may be hiding that diversity from the model

---

## 📊 TRAINING METRICS (Latest Run)

**Loss Trajectory:**
```
Batch 1:     loss=10.5721 (random baseline ✓)
Batch 100:   loss=8.6279  (dropped 1.9 in 100 batches)
Batch 200:   loss=8.5713  (dropped 0.06 in next 100 batches) ← PLATEAU
Batch 600+:  loss=8.3-8.9 (oscillating, no progress)
```

**Learning Rate Schedule:**
```
Step 100:  lr=0.00010000  (base LR)
Step 200:  lr=0.00007793  (decay started)
```

**Attention Diagnostics (analyze_attention_diagnostics.py):**
- ✅ No -FLT_MAX anomalies (Issue #15 fix verified)
- ✅ Entropy: 79.38-81.26 (healthy range)
- ✅ QK range: 0.89-2.72 (normal variation)
- ✅ Gradient flow: L5: 2.4e-06 → L0: 1.5e-05 (proper propagation)
- ⚠️  6 entropy collapse events (20%+ drop, transient)
- ✅ No gradient explosion/vanishing detected

---



## 📁 Key Files

| File | Purpose |
|------|---------|  
| `inspect_training_data_quality.py` | Data quality inspector (DISPROVEN data diversity hypothesis) |
| `BackwardPhase1_OutputLayer.cu` | LM head → encoder gradient |
| `BackwardPhase2_Encoder.cu` | Layer-by-layer encoder backward |
| `BackwardPhase3_InputLayer.cu` | Embedding backward (FIXED) |
| `BackwardContext.hpp` | Shared context for backward |
| `Phase2_TrainingLoop.cu` | Training orchestration |
| `GradNormGPU.cu` | Gradient component computation |

---

## 🧪 Test Commands

```powershell
# Rebuild training executable
cd D:\G.R.I.M\resources\models\GRIM-text\training\TrainingLoop
cmake --build build --config Release --target train_gpu

# Run training
cd D:\G.R.I.M\resources\models\GRIM-text\training
.\TrainingLoop\build\Release\train_gpu.exe

# Analyze latest log
python D:\G.R.I.M\verify_training_math.py
```
