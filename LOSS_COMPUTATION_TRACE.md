# GRIM-text Loss Computation Trace

**Generated:** December 22, 2025  
**Purpose:** Complete data flow trace for debugging training plateau at 8.0-8.7 loss range

---

## Executive Summary

Loss computation uses a **unified single-kernel architecture** that combines:
- Label-smoothed cross-entropy
- Focal weighting for hard examples
- Per-token sample weights

**Key Insight:** All loss terms computed in ONE kernel pass - no gradient overwrites, no sequential accumulation bugs.

---

## Pipeline Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                         TRAINING LOOP                                 │
│                    (Phase2_TrainingLoop.cu)                          │
└──────────────────────────┬───────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│  1. MODEL FORWARD PASS                                               │
│     → Embedding → Encoder → LM Head → cached_logits                 │
│     → Shape: [batch_size * seq_len, vocab_size]                     │
└──────────────────────────┬───────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│  2. COMPUTE LOSS (LanguageModel::computeLossBatch)                  │
│     File: training/TrainingOps.cu:39-171                            │
│                                                                      │
│     Inputs:                                                          │
│     • logits [D]: cached_logits from forward                        │
│     • targets [D]: ground truth token IDs                           │
│     • sequence_weights [D]: per-sequence content weighting          │
│     • grad_logits [D]: pre-allocated gradient buffer                │
│                                                                      │
│     Creates LossComputationInputs struct:                           │
│     • context.logits = training_state_.cached_logits                │
│     • context.targets = training_state_.cached_targets              │
│     • context.sequence_weights (optional, for content weighting)    │
│     • context.vocab_size, batch_size, seq_len                       │
│     • grad_logits = training_state_.grad_logits                     │
│                                                                      │
│     Calls: computeLossHost(loss_inputs, scratch)                    │
└──────────────────────────┬───────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│  3. HOST WRAPPER (ComputeLossHost_GPU.cu:137-242)                   │
│     File: Shared/Loss/ComputeLoss/ComputeLossHost_GPU.cu            │
│                                                                      │
│     Validation:                                                      │
│     • Non-null buffers (logits, targets, grad_logits)               │
│     • Positive dimensions (batch_size, seq_len, vocab_size)         │
│     • Token count ≤ max_tokens (default 8192)                       │
│                                                                      │
│     Scratch Buffer Management:                                       │
│     • Allocates loss_values [D]: per-token loss array               │
│     • Allocates loss_accumulator [D]: reduction scratch             │
│     • Reuses buffers across batches (Rule 22: centralized)          │
│                                                                      │
│     Diagnostic Logging (one-time):                                   │
│     • First 10 logits values                                         │
│     • First 5 target token IDs                                       │
│     • Logit AT TARGET position for token 0                          │
│     • Max logit in first 100 elements                                │
│                                                                      │
│     Calls: Loss::launchLossPipeline(ctx, cfg, buffers)              │
└──────────────────────────┬───────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│  4. LOSS ORCHESTRATOR (ComputeLoss_GPU.cu:40-185)                   │
│     File: Shared/Loss/ComputeLoss/ComputeLoss_GPU.cu                │
│                                                                      │
│     Configuration Mapping:                                           │
│     unified_cfg.focal_alpha = cfg.focal.alpha (default 1.0)         │
│     unified_cfg.focal_gamma = cfg.focal.gamma (default 2.0)         │
│     unified_cfg.smoothing_epsilon = cfg.label_smoothing.epsilon     │
│     unified_cfg.strict_mode = true (ALWAYS fail loud)               │
│                                                                      │
│     Static Context (Rule 22 compliance):                             │
│     • UnifiedLossContext persists across batches                    │
│     • Allocates telemetry buffers once, reuses forever              │
│     • No per-call allocations                                        │
│                                                                      │
│     Calls: loss_context.compute(unified_cfg, inputs, outputs)       │
│     Returns: Telemetry with loss stats, gradient norms, errors      │
└──────────────────────────┬───────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│  5. UNIFIED LOSS KERNEL (UnifiedLoss_GPU.cu:88-340)                 │
│     File: Shared/Loss/UnifiedLoss/UnifiedLoss_GPU.cu                │
│                                                                      │
│     ** SINGLE KERNEL - NO GRADIENT OVERWRITES **                    │
│                                                                      │
│     Per-Token Processing (parallel across tokens):                   │
│     ┌───────────────────────────────────────────────────────────┐  │
│     │ A. SOFTMAX (numerically stable)                           │  │
│     │    max_logit = max(logits[0..vocab_size-1])              │  │
│     │    sum_exp = Σ exp(logit[i] - max_logit)                 │  │
│     │    p_i = exp(logit[i] - max_logit) / sum_exp             │  │
│     │    p_t = softmax[target]                                  │  │
│     └───────────────────────────────────────────────────────────┘  │
│                                                                      │
│     ┌───────────────────────────────────────────────────────────┐  │
│     │ B. LABEL SMOOTHING TARGETS                                │  │
│     │    q_on = 1 - ε  (target class)                          │  │
│     │    q_off = ε/(V-1)  (off-target classes)                 │  │
│     │    ε = smoothing_epsilon (0.15 default)                   │  │
│     └───────────────────────────────────────────────────────────┘  │
│                                                                      │
│     ┌───────────────────────────────────────────────────────────┐  │
│     │ C. CROSS-ENTROPY (label-smoothed)                         │  │
│     │    log_p_t = log(p_t)                                     │  │
│     │    log_p_off = Σ_{i≠t} log(p_i)                          │  │
│     │    CE_smooth = -q_on * log_p_t - q_off * log_p_off       │  │
│     └───────────────────────────────────────────────────────────┘  │
│                                                                      │
│     ┌───────────────────────────────────────────────────────────┐  │
│     │ D. FOCAL WEIGHTING                                        │  │
│     │    focal_weight = (1 - p_t)^γ                            │  │
│     │    γ = focal_gamma (2.0 default)                          │  │
│     │    Emphasizes hard examples (low p_t)                     │  │
│     └───────────────────────────────────────────────────────────┘  │
│                                                                      │
│     ┌───────────────────────────────────────────────────────────┐  │
│     │ E. FINAL LOSS                                             │  │
│     │    L = α * (1-p_t)^γ * CE_smooth * sample_weight         │  │
│     │    α = focal_alpha (1.0 default)                          │  │
│     └───────────────────────────────────────────────────────────┘  │
│                                                                      │
│     ┌───────────────────────────────────────────────────────────┐  │
│     │ F. GRADIENT COMPUTATION (full derivation)                 │  │
│     │                                                            │  │
│     │    ∂L/∂z_i = α * (1-p_t)^(γ-1) * [                      │  │
│     │        (1-p_t) * (p_i - q_i) +          [CE term]        │  │
│     │        γ * p_t * CE_smooth * (p_i - δ)  [focal term]     │  │
│     │    ]                                                       │  │
│     │                                                            │  │
│     │    Where:                                                  │  │
│     │    • p_i = softmax probability for class i                │  │
│     │    • q_i = smoothed target (q_on or q_off)                │  │
│     │    • δ = Kronecker delta (1 if i==target, else 0)         │  │
│     │                                                            │  │
│     │    CRITICAL: Single gradient write, no accumulation       │  │
│     │              No overwrite bugs possible                    │  │
│     └───────────────────────────────────────────────────────────┘  │
│                                                                      │
│     ┌───────────────────────────────────────────────────────────┐  │
│     │ G. TELEMETRY UPDATE (atomic ops)                          │  │
│     │    atomicAdd(&telemetry->loss_sum, loss)                  │  │
│     │    atomicAdd(&telemetry->loss_sq_sum, loss²)              │  │
│     │    atomicMaxFloat(&telemetry->loss_max, loss)             │  │
│     │    atomicMinFloat(&telemetry->loss_min, loss)             │  │
│     │    atomicAdd(&telemetry->grad_norm_sum, ||∇||)            │  │
│     │    atomicMaxFloat(&telemetry->grad_norm_max, ||∇||)       │  │
│     │    atomicAdd(&telemetry->focal_weight_sum, focal_weight)  │  │
│     │    atomicAdd(&telemetry->valid_count, 1)                  │  │
│     │    if (p_t < 0.5): atomicAdd(&hard_example_count, 1)     │  │
│     └───────────────────────────────────────────────────────────┘  │
│                                                                      │
│     Special Cases:                                                   │
│     • Masked tokens (target < 0): loss=0, grad=0, skip telemetry   │
│     • NaN/Inf logits: loss=0, grad=0, increment error counters     │
│     • Invalid sum_exp: Early exit with error code                   │
│                                                                      │
│     Debug Capture (first valid token only):                          │
│     • target, max_logit, sum_exp, p_t                               │
│     • ce_smooth, focal_weight, sample_weight                        │
│     • focal_alpha, final loss                                       │
│     • Stored in telemetry->debug_* fields                           │
└──────────────────────────┬───────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│  6. REDUCTION KERNEL (UnifiedLoss_GPU.cu:342-370)                   │
│                                                                      │
│     Sums token_losses[0..total_tokens-1] → scratch buffer          │
│     Block-wise reduction with shared memory                         │
│     Final atomic add to global accumulator                          │
└──────────────────────────┬───────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│  7. RESULTS EXTRACTION                                               │
│                                                                      │
│     ComputeLoss_GPU.cu:                                             │
│     • Copies loss_sum from device to host                           │
│     • Calculates average_loss = total / valid_tokens               │
│     • Populates LossBreakdown struct:                               │
│       - cross_entropy = total loss (focal+smoothing integrated)     │
│       - label_smoothing = 0 (legacy field, integrated)              │
│       - focal = 0 (legacy field, integrated)                        │
│       - total = cross_entropy                                       │
│                                                                      │
│     Returns to TrainingOps.cu → Phase2_TrainingLoop.cu             │
└──────────────────────────┬───────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│  8. BACKWARD PASS                                                    │
│     File: Phase2_TrainingLoop.cu                                    │
│                                                                      │
│     Gradient Flow (CRITICAL):                                        │
│     1. grad_logits already populated by loss kernel                 │
│     2. Token normalization scaling:                                  │
│        base_scale = 1.0 / valid_tokens                              │
│        Applied during backward (line 1377-1383)                     │
│     3. Backprop: LM Head → Encoder → Embedding                      │
│        Each layer reads grad_logits, computes dW, propagates dX     │
│     4. Gradient accumulation in GradAccumulationController          │
│                                                                      │
│     Token Normalization Math:                                        │
│     • Loss kernel computes: dL_sum/dz (per-token, sum units)        │
│     • Backward scales by 1/T: dL_avg/dz (averaged units)            │
│     • This makes gradients batch-size invariant                     │
└──────────────────────────┬───────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│  9. GRADIENT PROCESSING PIPELINE                                     │
│     File: Phase2_TrainingLoop.cu:1400-1600                          │
│                                                                      │
│     A. Compute Gradient Norm:                                        │
│        result.grad_norm = computeGradNorm(should_sync)              │
│        • Syncs CPU-GPU every 10 batches (performance)               │
│        • Computes L2 norm across all parameter gradients            │
│        • Component breakdown: emb, lm, attn, ffn, rms, sb           │
│                                                                      │
│     B. Token-Normalized Gradient:                                    │
│        result.normalized_grad_norm =                                │
│            computeNormalizedGrad(grad_norm, clip_selection.stats)   │
│        • Scales by sqrt(ref_tokens/valid_tokens)                    │
│        • Makes norm batch-size invariant                            │
│                                                                      │
│     C. TelemetryControl Decision:                                    │
│        telemetry_decision = evaluateTelemetryControl(               │
│            TelemetryInputs{                                         │
│                global_step, current_loss, grad_norm,                │
│                normalized_grad_norm, valid_tokens, reference_tokens │
│            }                                                         │
│        )                                                             │
│        • Spike detection: normalized vs baseline gradient           │
│        • Decay detection: gradient health check                     │
│        • Outlier regime: loss volatility                            │
│        • Returns: grad_scale_factor (0.65 if decay, 1.0 normal)    │
│                                                                      │
│     D. Gradient Clipping:                                            │
│        effective_clip = base_clip * telemetry_decision.scale        │
│        ctx.model->clipGradients(effective_clip, valid_tokens)       │
│        • Base clip: 1.0 (from clip_selection)                       │
│        • Scaled by telemetry decision (BUG: applied here + optimizer)│
│                                                                      │
│     E. Optimizer Step:                                               │
│        scaleOptimizerMoments(telemetry_decision.scale)              │
│        ctx.optimizer.update(effective_lr)                           │
│        • BUG: grad_scale_factor applied AGAIN (double penalty)      │
│        • Effective penalty: 0.65² = 0.4225 (58% reduction!)         │
└───────────────────────────────────────────────────────────────────────┘
```

---

## Key Findings from Trace

### 1. **Loss Computation is CORRECT**
- Single unified kernel, no gradient overwrites
- Proper focal weighting for hard examples  
- Label smoothing applied correctly
- Gradients mathematically verified (see gradient derivation in kernel)

### 2. **Token Normalization is CORRECT**
- Loss kernel computes sum gradients: `dL_sum/dz`
- Backward scales by `1/valid_tokens`: `dL_avg/dz`
- Makes gradients batch-size invariant

### 3. **CONFIRMED BUGS**

#### Bug A: Double Gradient Scaling (Lines 1538-1539, 1603)
```cpp
// Line 1538-1539: FIRST application
effective_clip = clip_norm * telemetry_decision.grad_scale_factor;

// Line 1603: SECOND application (BUG!)
scaleOptimizerMoments(telemetry_decision.grad_scale_factor);
```
**Impact:** When decay detected (scale=0.65), effective penalty = 0.65² = 0.4225 (58% reduction, not 35%)

#### Bug B: Normalized vs Raw Comparison (FIXED)
```cpp
// OLD (TelemetryControl_GPU.cu:300-308):
const float baseline_grad = L2_grad_mu;  // RAW
const float current_grad = normalized_grad;  // NORMALIZED
spike_ratio = current_grad / baseline_grad;  // APPLES vs ORANGES

// FIXED:
const float normalized_baseline = d_computeNormalizedGrad(raw_baseline, ...);
spike_ratio = normalized_grad / normalized_baseline;  // APPLES vs APPLES
```
**Impact:** Batch size variance triggered false decay flags

### 4. **Telemetry Debug Fields**
First valid token's values captured in kernel:
- `debug_target`: Target token ID
- `debug_max_logit`: Max logit before softmax
- `debug_sum_exp`: Softmax denominator
- `debug_p_t`: Target probability
- `debug_ce_smooth`: Label-smoothed CE
- `debug_focal_weight`: (1-p_t)^γ
- `debug_sample_weight`: Content weight
- `debug_loss`: Final loss value

**Access:** `getLastTelemetry()` after `computeLossBatch()`

---

## Configuration Parameters

From `ai_config.json` → `training.config.loss`:

```json
{
  "focal_alpha": 1.0,
  "focal_gamma": 2.0,
  "smoothing_epsilon": 0.15,
  "strict_mode": true
}
```

**Effective Formula:**
```
L = 1.0 * (1 - p_t)^2.0 * [
    -(1-0.15) * log(p_t) - 0.15/(V-1) * Σ_{i≠t} log(p_i)
] * sample_weight
```

Where:
- `p_t` = softmax probability of target class
- `V` = vocab_size
- `sample_weight` = sequence content weighting (default 1.0)

---

## Gradient Formula (Full Derivation)

From kernel line 276-333:

```
∂L/∂z_i = α * (1-p_t)^(γ-1) * [
    (1-p_t) * (p_i - q_i) +           // CE term scaled by focal weight
    γ * p_t * CE_smooth * (p_i - δ)   // Focal derivative term
]
```

**Component Breakdown:**
1. **CE term:** `(1-p_t) * (p_i - q_i)`
   - `p_i` = softmax output for class i
   - `q_i` = smoothed target (0.85 for correct, 0.15/(V-1) for wrong)
   - Standard softmax CE gradient with label smoothing

2. **Focal term:** `γ * p_t * CE_smooth * (p_i - δ)`
   - Derivative of focal weight `(1-p_t)^γ` w.r.t. logit
   - Modulates gradient by how hard the example is

3. **Focal base:** `α * (1-p_t)^(γ-1)`
   - Common factor for both terms
   - When γ=2: `(1-p_t)^1 = (1-p_t)`

**Numerical Stability:**
- Softmax uses max-logit subtraction
- Log clamped to -100 (prevents -inf)
- p_t clamped to [ε, 1-ε] where ε=1e-7
- All NaN/Inf checked, fail-loud in strict mode

---

## Performance Characteristics

### GPU Kernel Stats (typical batch)
- **Block size:** 256 threads
- **Grid size:** `ceil(total_tokens / 256)` blocks
- **Shared memory:** 1 KB per block (reduction kernel)
- **Global memory access:**
  - Coalesced reads: logits, targets
  - Scattered writes: grad_logits (strided by vocab_size)
  - Atomic writes: telemetry accumulators

### CPU-GPU Synchronization Points
1. **Forward pass:** Async, no sync
2. **Loss computation:** Kernel async + stream sync (line 241)
3. **Gradient norm:** Conditional sync every 10 batches (line 1410)
4. **Optimizer step:** No sync (GPU-only)

**Rule 24 compliance:** Minimize CPU-GPU sync for performance

---

## Debug Checklist for Plateau Issue

### ✅ Validated (Not the Problem)
- [x] Loss formula mathematically correct
- [x] Gradient derivation verified
- [x] Token normalization proper
- [x] No gradient overwrites (single kernel)
- [x] Softmax numerically stable
- [x] Focal weighting works (hard_example_count tracked)

### ⚠️ Requires Fix (Root Causes)
- [ ] **Double gradient scaling** (Bug A) - clip AND optimizer
- [x] **Normalized vs raw comparison** (Bug B) - FIXED
- [ ] TelemetryControl false positives still possible (validate with new logs)

### 🔍 Next Steps
1. **Remove double-scaling:**
   - Option A: Remove scale from clip thresholds (line 1538-1539)
   - Option B: Remove scale from optimizer (line 1603)
   - Recommendation: Remove from optimizer (clip should be absolute)

2. **Validate decay detection fix:**
   - Run training with normalized comparison
   - Check for [DECAY] flags in logs
   - Should be rare after batch size normalization

3. **Monitor telemetry:**
   - Add debug logging for first token: `getLastTelemetry().debug_*`
   - Track p_t distribution (should increase as model learns)
   - Monitor hard_example_ratio (should decrease)

---

## Files Reference

### Core Loss Pipeline
- `training/TrainingOps.cu:39-171` - computeLossBatch entry point
- `Shared/Loss/ComputeLoss/ComputeLossHost_GPU.cu:137-242` - Host wrapper
- `Shared/Loss/ComputeLoss/ComputeLoss_GPU.cu:40-185` - Orchestrator
- `Shared/Loss/UnifiedLoss/UnifiedLoss_GPU.cu:88-340` - Kernel implementation

### Gradient Processing
- `training/Phases/Phase2_TrainingLoop.cu:1400-1600` - Gradient ops
- `Shared/Telemetry/TelemetryControl_GPU.cu:300-312` - Spike/decay detection (FIXED)
- `Shared/Telemetry/TelemetryControl_GPU.cu:334-342` - Decay check

### Configuration
- `ai_config.json` → `training.config.loss` - Loss hyperparameters
- `Shared/Loss/Loss.hpp` - LossConfig struct definitions

---

## Telemetry Fields

```cpp
struct UnifiedLossTelemetry {
    // Aggregates
    uint32_t valid_count;         // Non-masked tokens
    uint32_t masked_count;        // Masked/invalid tokens
    float loss_mean;              // Average loss
    float loss_variance;          // Loss variance
    float loss_min, loss_max;     // Loss range
    float grad_norm_mean;         // Average gradient L2 norm
    float grad_norm_max;          // Max gradient L2 norm
    float focal_weight_mean;      // Average focal weight
    uint32_t hard_example_count;  // Tokens with p_t < 0.5
    
    // Error tracking
    uint32_t nan_count;           // NaN in logits/loss
    uint32_t inf_count;           // Inf in logits/loss
    uint32_t exit_max_logit_nan;  // Early exit: NaN max_logit
    uint32_t exit_sum_exp_zero;   // Early exit: zero denominator
    uint32_t exit_loss_nan;       // Early exit: NaN loss
    uint32_t exit_success;        // Successful computations
    
    // Debug capture (first valid token)
    int32_t debug_target;
    float debug_max_logit;
    float debug_sum_exp;
    float debug_p_t;
    float debug_ce_smooth;
    float debug_focal_weight;
    float debug_sample_weight;
    float debug_focal_alpha;
    float debug_loss;
    
    int32_t error_code;  // 0 = OK, negative = error
};
```

---

## Conclusion

**Loss computation is NOT the problem.** The plateau is caused by **downstream gradient processing bugs:**

1. **Primary Issue:** Double gradient scaling (58% penalty instead of 35%)
2. **Secondary Issue:** False decay detection from mixed units (FIXED)

**Next Action:** Remove one application of `grad_scale_factor` (recommend removing from optimizer, keep clip absolute).
