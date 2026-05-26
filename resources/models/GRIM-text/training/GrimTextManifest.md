# GRIM-Text Training Stale Code Investigation Checklist

**Chronological execution order from Phase 1 Startup → Phase 2 Training Loop → Phase 3 Cleanup**

Use this checklist to systematically audit each file in the order it's used during training focusing on a systematic review the primary goal is finding either mismtaches: things that throw away progress from training or stale usages. when starting a new file for investigation you should ask yourself what in this file is redudent, unused, used only for backwards compatibiliity, or is not implemented properly do not start going through multiple files at a time we have done that time and time again so we are going to do this one right and through each one by one in order this is to not only help identify mode collapse/loss plateau but also to make up for maintenance debt top to bottom when finished with each file provide an indepth rundown of findings suspicouns and then wait for instruction before making changes.

---

## PHASE 1: STARTUP & INITIALIZATION

### 1.1 Orchestrator Entry Point

- [] **train_gpu.cu** (main entry) 
  - Calls Phase1 → Phase2 → Phase3 sequentially


---

### 1.1a Logging Infrastructure (grim_log_recorder library)

- [] **Shared/LogRecorder/LogRecorder.cu** 
  - Centralized logging system used throughout training
  - Records diagnostics, equation logging, and error messages
  - Part of grim_log_recorder static library (shared by all targets)


---

### 1.2 Configuration Loading & Validation

- [] **train_gpu.cu + Phase1_Startup.cu / Phase1_Startup.hpp** 
  - `train_gpu.cu` loads the validated training startup config root from canonical `ai_config.json`
  - `Phase1_Startup` receives that config handoff and performs startup validation/initialization against it
  - Path validation (model, data, checkpoint dirs)

---

### 1.3 Training Data Loading

- [] **Shared/DataLoader/DataLoader.cu + training_data_loader.hpp** 
  - PrepareTrainingDataFromCache: Reads merged_verified_cache.jsonl → tokenizes → writes single GRMT file
  - training_data_loader.hpp: GRMTDataLoader reads .grmt binary format for Phase1_Startup

---

### 1.4 Tokenizer Initialization (UnigramByte Library)

- [] **Shared/UnigramByte/Byte.cu** 
  - Byte fallback tokenizer (raw UTF-8 bytes 0x00-0xFF)
  - Provides 100% coverage for unknown characters/emojis

- [] **Shared/UnigramByte/Unigram.cu** 
  - Unigram Language Model tokenizer (statistical subword segmentation)
  - Viterbi decoding for optimal segmentation


- [] **Shared/UnigramByte/UniByte.cu** 
  - Combined Unigram + Byte fallback (GrimTokenizer alias)


- [] **Shared/UnigramByte/AtomTable.cu** 
  - Atom token management (numbers, URLs, emails, paths, dates, code)
  - Dedicated embeddings for structural elements

- [] **Shared/UnigramByte/AhoCorasick.cu** 
  - O(n) multi-pattern matching for structural token detection
  - 50-100x faster than std::regex for URL/email/number prefixes
  - Detects: http://, https://, www., ftp://, ws://, wss://, file://, @, 0x, 0b

---

### 1.5 Model Weight Initialization

- [] **Shared/TrainingState/TrainingTensors.cu**  & FIXED
  - Weight allocation and Xavier initialization


---

### 1.6 Embedding Initialization

- [] **Shared/TensorContract/TensorContract_GPU.cu**
- GPU autograd/logit system Tape Based

---

### 1.7 Stream & Resource Initialization

- [] **Shared/StreamController/StreamController_GPU.cu**
  - CUDA stream creation and synchronization
  - **Rule 22**: All streams created via TrainingState controller, never raw `cudaStream_t`

---

### 1.8 Training State GPU Setup

- [] **Shared/TrainingState/TrainingStateGPU.cu**  & FIXED (4 passes)
  - Allocates GPU buffers, cuBLAS handle, CUDA streams
  - Allocates: gradient buffers, optimizer state (m, v), intermediate tensors
  
---

- [ ] **Shared/GPUBuffer/GPUBuffer.cu**
  - Low-level GPU buffer memory management utilities
  - Wrapper for cudaMalloc/cudaFree with error checking
  - Pattern to check: Verify all allocations track ownership
  - Pattern to check: Verify no memory leaks (all allocations freed)
  - **Rule 22**: Should only be called from TrainingState, not directly by layers

---

### 1.9 Positional Bias Initialization

- [] **Shared/PBM/PositionalBiasMethod.cu**  & FIXED
  - Initializes ALiBi slopes and RoPE frequencies
  - **Issue #47**: ALiBi slopes scale relative to max_seq_len - VERIFIED ✅ (`m_max = target_bias / d_min`, geometric interpolation)
  - **Issue #78**: ALIBI_MAX_BIAS = 0.0f (capping DISABLED) - correct after Issue #84 root cause fix. Stale "NOT RECOMMENDED" comment in HyperParameters FIXED.
  - **FIXED**: Rule 20 — `launchRoPERotationGQA()` and `launchRoPERotationGQA_backward()` had silent `return` on null pointers/invalid params → changed to `throw std::runtime_error()`
  - **FIXED**: Rule 20 — Duplicate PBM init fallback in `InitTrainingState.cu::initTrainingState()` REMOVED. Now throws if PBM not pre-initialized by `initPBM()`.
  - Slopes are NEGATIVE (FlashAttention uses `+= slope * col_idx`) ✅
  - Non-GQA launcher properly DELETED (Rule 20) ✅
  - `ensurePBM()` float equality safe (config assignment, no arithmetic) ✅
  - No stale code found ✅

---

### 1.9a Tensor Conversion Utilities

- [] **Shared/TensorConversion/TensorConversion.cu**
  - Tensor format conversion utilities (BHSD↔BSM, BHSD↔BSHD bf16, QKV split/merge GQA)
  - **DELETED**: Dead float-only conversions (BHSD↔BHDS, BHSD↔BSHD float, BSHD→BHSD float) — only reachable via unused dispatcher
  - **DELETED**: Unused `convert()` dispatcher, `can_convert_inplace()`, `convert_inplace()`, `is_conversion_supported()` from TensorContract_GPU.cu
  - **DELETED**: `Layout::BHDS` enum value, `make_BHDS()` factory from TensorContract_GPU.hpp
  - Remaining functions all have direct production callers (bf16 for FlashAttention, BSM for head merge/split, QKV split/merge for GQA)
  - Rule 20 compliant: `split_qkv_gqa`/`merge_qkv_grads_gqa` throw on null pointers and invalid dims

---

### 1.10 Training State Initialization

- [] **InitTrainingState.cu**
  - Coordinates allocation of all GPU resources for training
  - **FIXED**: `num_kv_heads` sourced from compile-time `HyperParameters::DEFAULT_NUM_KV_HEADS` instead of `cfg.num_kv_heads` — latent desync bug from agent corruption. Now reads JSON config like every other callsite.
  - **DELETED**: 10 dead FlashAttention bf16 Tensor fields + 2 size_t fields from TrainingState (~56MB dead GPU). `FlashAttentionLayer::ensureScratch()` self-manages these buffers; autograd `ScaledDotProductAttentionGradFn` self-allocates backward buffers. Same dead code deleted from `InitInferenceState.cu`.
  - **DELETED**: Per-token text feature side-channel and its dimension constant; atom metadata now uses numeric values, atom mask, atom flags, atom strings, and slot maps only
  - Initialization order verified: StreamController → cuBLAS → PBM → TrainingTensors → activation caches → gradient buffers → loss scratch → ScratchBlock
  - Rule 20 compliant: throws on uninitialized cuBLAS, PBM, TrainingTensors
  - Rule 22 compliant: all streams via `stream_ctrl.getPrimaryStream()`

---

### 1.11 Startup Model GPU Assembly

- [] **Phases/Startup/Model/ModelGpuAssembly.cu** (`LanguageModel::initGPU`) —
  - Startup/Model ownership is explicit: this source assembles durable GPU model layers after CUDA, streams, cuBLAS, and PBM are initialized.
  - Creates GPU encoder layers, EmbeddingLayer, LMHeadLayer, optional ReasoningHead, ExecutionBlock, DecodeTimeSlotSelector, DecodeTimeNumPolicy, and MTP heads.
  - `cfg.num_kv_heads` remains sourced from runtime JSON via grouped hyperparameter views, not compile-time defaults.
  - Rule 20 compliance: throws on use_gpu=false, missing StreamController/cuBLAS/PBM, null tied embedding data, or not-ready layer weights.
  - Does not allocate TrainingState activation caches, optimizer state, parameter groups, checkpoints, forward/backward, or Phase2 training loop state.
  - **DELETED**: vague root-level `TrainingOps.cu`; active builds now compile the Startup/Model-owned source.

---

### 1.12 Common Language Model Utilities

- [] **Common/grim_language_model_gpu.cu**  (3 passes)
  - High-level language model orchestration on GPU
  - **FIXED**: Removed unused `grim_scale_buffer.hpp` include (no callers in this file)
  - **FIXED**: Deleted backwards-compat scaffolding (`has_v2_fields`) and stale tombstone comments
  - **FIXED**: `normalizeProbabilities()` now throws on invalid sum (no silent return)
  - **FIXED**: Removed debug spew in `sampleFromLogits()` and unreachable code after throws
  - **CLEANED**: `GRIM/grim_language_model_cuda.hpp` CPU fallback class blocks removed (EncoderLayer/GrimEncoder/LMHead/TextGenerator), dead accessors and members deleted
  - **REFACTORED**: `ALiBiPositionalBias` wrapper deleted; PBM access now goes through model-level `PBM::PBMStateOwner` and `LanguageModel::getPBMState()` / `getPBMSpec()`.
  - **FIXED (Vocab authority cleanup)**: Deleted LanguageModel vocab.bin size detection/override; model construction now uses caller-supplied vocab size (GRMT header for training, tokenizer token-space size for inference).
  - **FIXED (Pass 3)**: Constructor now validates `num_heads > 0` and `d_model % num_heads == 0` BEFORE computing `d_head` (prevents divide-by-zero/UB during positional init).
  - **DELETED (Payload Inference Cleanup)**: staged prompt APIs were removed; inference callers now build `BatchPayload` and enter through payload-only logits/generation methods.
  - **FIXED (Pass 3)**: Numeric prediction host copy in generation now checks `cudaMemcpyAsync` + stream sync result and throws on failure (previously ignored sync result).
  - **FIXED (Pass 3)**: Scratch toggles hardened — enabling ScratchBlock or scratch pool now throws if backing objects are uninitialized; only disable-without-init is treated as no-op.

- [] **GRIM/grim_language_model_cuda.hpp** (2 passes)
  - **DELETED**: `parameterGroupsStale()` declaration — unimplemented method, zero callers, no valid purpose (intended arch-hash check was never coded)
  - **DELETED**: `last_param_group_arch_hash_` member — companion to above, never read or written
  - **DELETED**: `applyActivationQuantization()` declaration — unimplemented method for unimplemented feature, zero callers. Activation quantization config loading stays (Phase1 infrastructure), but no QuantizationLayer is wired to any forward path
  - **DELETED**: `activation_quantizer_` member — `std::unique_ptr<Quantization::QuantizationLayer>` that was never assigned, always nullptr
  - **DELETED**: `#include "Quantization_GPU.hpp"` — no longer needed after activation_quantizer_ removal
  - **DELETED**: `LanguageModel::alibi_` member and `GrimEmbeddingStack::alibi_` wrapper ownership. `LanguageModel::initPBM()` initializes the model-level `PBM::PBMStateOwner` after `StreamController` exists; `PBMSpec` is only a non-owning attention view.
  - **DELETED**: `getAlibiPtr()` accessor — zero callers, returned the dead `alibi_` member above
  - **NOTE (not fixed)**: `HardcodedPattern` enum is duplicated in `LanguageModelConfig` and `HardcodedStates_GPU.hpp` with `static_cast` bridge in Phase1. Fragile but diagnostic-only — defer unification.

- [] **Common/ScaleBuffer.cu**  & HARDENED
  - Buffer scaling utilities (43 lines, kernel + 3 wrappers)
  - **FIXED**: `scaleDeviceBuffer(float*, ...)` — replaced `if (!data || count == 0) return;` silent early-return with throws (Rule 20). Identity scale check (`scale ≈ 1.0`) kept as optimization.
  - **FIXED**: `scaleDeviceBuffer(Tensor&, ...)` — replaced `if (!tensor.data || numel == 0) return;` with throws.
  - **FIXED**: `scaleGradBuffer(Tensor&, ...)` — replaced `if (!has_grad() || numel == 0) return;` with throws. Caller MUST ensure gradient is allocated before calling.
  - No hardcoded scale values found. All scale factors come from callers.
  - `AutogradTraining.cu` has its own inline `scaleGradBuffer(float*, ...)` with different signature (raw pointer) — not a conflict, different call sites.

---

### 1.13 Optimizer Initialization

- [] **Shared/Optimizers/AdamW/AdamW_Kernal_GPU.cu**  & HARDENED
  - **FIXED**: Manifest path stale in previous entry (`training/Shared/...`) — canonical path is `resources/models/GRIM-text/Shared/Optimizers/AdamW/AdamW_Kernal_GPU.cu`
  - AdamW update verified as decoupled weight decay: `param -= lr * (adam_update + wd * param)` (not L2-regularization in gradient moment path)
  - **FIXED**: Rule 20 input guards in `launchAdamWKernel()` — validates `learning_rate` finite and `>= 0`, `weight_decay` finite and `>= 0`, `step >= 0`, and non-null CUDA stream
  - **FIXED**: Bias-correction denominator validation added before inversion (prevents divide-by-zero/NaN propagation on invalid optimizer state)
  - **FIXED**: Added immediate CUDA kernel launch error check (`cudaGetLastError`) with group name in exception
  - Optimizer states (`m_states`, `v_states`) are allocated centrally in `OptimizerState::allocate()` and bound to parameter groups in `Startup/Model/ParameterGroupRegistration::buildParameterGroups(model)` ✅

---

### 1.14 Stability Overrides & Feature Flags

- [] **Shared/Stability/StabilityOverrides.cpp**  & DELETED
  - **DELETED**: Entire `Shared/Stability/` library was DEAD CODE — `loadStabilityOverrides()` had ZERO callers
  - Stability overrides were already loaded via `ai_config_paths.hpp` → `TrainingHyperparameters` → `Phase1_Startup.cu` → `StabilityOverrides` struct
  - Three competing data structures collapsed to one pipeline: JSON → `ai_config_paths` → `Phase1_Startup.hpp::StabilityOverrides`
  - **DELETED** `clip_abs` field: no absolute gradient clipping exists in training loop — dead config with no implementation
  - **RENAMED** `clip_norm` → `clip_per_token` for consistency with JSON key and actual semantics
  - **FIXED**: `clip_per_token` now WIRED into `hp.grad_clip_norm` override when `stability.enabled=true` (was loaded but silently discarded)
  - **FIXED**: `stability_override_clip_abs` + `stability_override_clip_norm` in `ai_config_paths.hpp` consolidated to single `stability_override_clip_per_token`
  - Removed from CMakeLists.txt, removed `clip_abs` from `ai_config.json`
  - **NOTE (not fixed)**: `SkipGradGuard = true` hardcoded at Phase2_TrainingLoop.cu:2281 permanently disables GradGuard spike detection — separate concern

---

### 1.15 GRIM-TS Layer Initialization

- [] **Layers/GRIMTS/GRIM-TS.cu**  & FIXED
  - GRIM-TS (Guess-Reward Integrated Memory - Training System): GPU-resident speculative decoding + RL cache
  - ACTIVELY USED by Phase2_TrainingLoop.cu (InitializeGuessCache, CacheGuessBatchGPU, ApplyRewardBatchGPU, GetCacheTelemetry, BeginMicroValidation, CompleteMicroValidation). TelemetryLattice depends on this.
  - **FIXED (Rule 20)**: Converted 11× `fprintf(stderr)+return false` → `throw std::runtime_error()` in InitializeGuessCache
  - **FIXED (Rule 20)**: Removed `return cudaErrorInvalidValue` from delegate registration (dead code)
  - **DELETED (Rule 26)**: Entire `GRIMTS::Delegates` namespace — 4 delegate types, 4 registration kernels, 4 clear kernels, 8 host-side registration functions (~200 lines). Zero callers ever registered callbacks. `Shared/Delegate/Delegate.hpp` also deleted (zero remaining includers).
  - **DELETED**: `#include <windows.h>` — zero Windows API calls in file
  - **DELETED (Rule 20)**: `NotifyMutation`/`NotifyEviction` stubs AND all 8 call sites — no-op stubs violate single-owner telemetry. Eviction telemetry `atomicAdd` inlined at eviction site.
  - **DELETED (Rule 26)**: `CacheMutationKind` and `EvictionReason` enums — zero remaining users after Notify* deletion.
  - **KEPT**: 12 global mutable variables in anonymous namespace — properly `{}` initialized, protected by mutexes, with clear lifecycle (init/shutdown). Appropriate pattern for GPU cache state.
  - **KEPT**: Logging namespace (actively used — RegisterLogCallback called from Phase2_TrainingLoop.cu GuessCacheScope)
  - File reduced from 1,995 → ~1,700 lines.

---

### 1.16 Equation-Based Diagnostic Logging

- [] **Shared/EquationLogging/EquationLogging.cu** 
  - Centralized equation-based diagnostic logging (Rules 20 & 21)
  - [*_EQUATION] format markers with mathematical formulas
  - Expected vs Actual value comparison with anomaly detection
  - Pattern to check: Verify all tensor operations have equation logging available
  - Pattern to check: Verify anomaly thresholds defined (score > 100, LSE > 50, gradient > 1e6)
  - Pattern to check: Verify includes tensor shapes for dimensional analysis

---

---

## PHASE 2: TRAINING LOOP (Forward Pass)

### 2.0 Inference State Initialization (if needed)

- [] **Layers/InitInferenceState/InitinferenceState.cu**  & FIXED
  - Initializes inference-specific GPU state (KV cache, etc.)
  - Used before validation batches or standalone inference
  - **FIXED**: 3x `std::cerr + return` → `throw std::runtime_error()` (Rule 20: StreamController, cuBLAS, GQA config)
  - **FIXED**: ScratchBlock init `catch` swallowed exception → re-throws with context (Rule 20: config says enabled, MUST succeed)
  - **FIXED**: Ghost memory summary counted DELETED buffers (layer_cache_bytes, embedding_cache_bytes) → now counts only actually-allocated activation buffers
  - **FIXED**: `cudaDeviceSynchronize` failure `std::cerr + return` → `throw std::runtime_error()` (Rule 20)
  - **DELETED**: Unused `cudaError_t err;` declaration (dead variable)
  - **DELETED**: `scratch_pool = nullptr` with misleading "first use" comment — ScratchBlock inference doesn't use pool
  - NOT dead code: Called by startup/model allocation and serialization paths that need inference runtime validation/allocation.

---

### 2.0a Hardcoded Debug States (testing/debugging only?)

- [ ] **Layers/HardcodedStates/HardcodedStates_GPU.cu**
  - Hardcoded test inputs or debug states for validation
  - Likely for debugging specific edge cases or regression tests
  - Pattern to check: Verify NOT used in production training
  - Pattern to check: Check if this is wrapped in `#ifdef DEBUG` or similar
  - **STALE CODE CHECK**: May be dead code - verify has any callers

---

### 2.1 Forward Pass Orchestrator

- [] **Forward_GPU.cu**  & FIXED (103→97 lines)
  - NOT a forward pass orchestrator — is `GPUGrimEncoder::Impl` layer container only
  - Creates `GPUEncoderLayer` instances from `EncoderLayerConstructionHP`, stores in `gpu_layers_` vector, exposes `getLayer()`; loop counts come from config-owned `num_layers`
  - Actual forward orchestration lives in `AutogradTraining.cu` (section 4.1)
  - **FIXED**: `FWD_ERROR + std::abort()` → `throw std::runtime_error()` (Rule 20), validation moved before config copy
  - **DELETED**: `FWD_ERROR` macro — only 2 usages, both replaced by the throw
  - No stale code, no dead functions; encoder public API is construction + layer accessors only ✅

- [x] **Inference_GPU.cu** DELETED
  - Phase2 inference owns generation and the read-only payload scorer `scoreInferencePrefillLogits(BatchPayload)` through shared forward.

---

### 2.1b Batch Composition

- [] **Shared/Batching/Batching_GPU.cu**
  - Dynamic batch composition: GREEDY, BEST_FIT_DECREASING, SIMILARITY_GROUPED packing
  - Token budget management, overflow handling, gradient accumulation grouping
  - Sequence lengths are supplied directly by Phase1 data loading; no separate catalog layer
  - BatchPayload.cu: Excellent Rule 20 compliance — thorough cross-check validation
---

### 2.1c Activation Quantization (Int8/FP16)

- [] **Layers/Quantization/Quantization_GPU.cu**  (code quality OK, fully unwired)
  - Activation quantization for inference speedup (Int8 symmetric, FP16)
  - 280 lines total (.cu + .hpp), 3 CUDA kernels (quantize, dequantize, fused quantize-dequantize)
  - **TODO**: Wire up when ready for inference optimization. Needs: construct `QuantizationLayer` in InitInferenceState, call at configured activation points, fix `stream` field to use centralized controller (Rule 22)
  - Implementation postponed. Current state is clean and self-contained, with no impact on existing training or inference code paths. No Rule 20 issues found in current implementation, but not yet wired to any forward paths to first focus on training stability and correctness.

---

### 2.2 Embedding + Position Encoding

- [] **AutogradTraining.cu + TensorContract_GPU.cu**

  **Forward Path Trace:**
  1. `token_ids = reinterpret_cast<int*>(ts->cached_token_ids_tensor.data)` Rule 20 null check
  2. `emb_weights = ts->tensors_->embedding_weights` Rule 20 null check + shape validation
  3. `embedding_scale = 1.0f` Issue #140: removed √d_model (correct for tied weights + ALiBi/RoPE)
  4. `emb_output = autograd::embedding(emb_weights, token_ids, total_tokens, stream, 1.0f)`
  5. Learned position embeddings: `use_learned_pos_emb = (positional_encoding == NONE)` → **FALSE** with ALIBI_ROPE → SKIPPED.
  6. Embedding dropout: training-only branch calls `autograd::dropout(emb, 0.15, step*2654435761+500, stream)`; inference skips dropout entirely ✅
  7. ScratchBlock forward operates in-place on `intermediates.embedding_tensor.data` ✅ Issue #90 fixed

  **Backward Path Trace (PCGrad for tied weights):**
  1. LM head matmul backward writes `grad_W` to `embedding_weights.grad` (shared buffer) ✅
  2. Backward chain reaches `EmbeddingGradFn::apply()` ✅
  3. PCGrad buffer available (`g_pcgrad_temp_buffer` snapshotted at construction time) ✅
  4. Step 1: Zero temp buffer ✅
  5. Step 2: `kernel_embedding_backward` writes to temp buffer (NOT shared grad buffer) ✅
  6. Step 3: `kernel_pcgrad_combine(weight_grad, pcgrad_buffer, vocab_size, d_model)` ✅
  7. Formula: `g_final = g_lm*(1 - proj_coef) + g_emb` where `proj_coef = g_lm·g_emb / ||g_lm||²` ✅
  8. `g_skip_embedding_backward_for_tied_weights = false` — correct (Issue #109/#110) ✅

  **Kernel Math Verification:**
  - `kernel_embedding_forward`: `output[i] = weight[token_id][i] * scale` (scale=1.0 for debugging)
  - `kernel_embedding_backward`: `atomicAdd(&weight_grad[token_id][i], grad[i] * scale)`
  - `generatePositionIds`: `position_ids[idx] = idx % seq_len`

  **Weight Initialization:**
  - `embedding_weights = Tensor::zeros({vocab_size, d_model})` then `xavier_uniform_()`
  - Position embedding weights: NOT allocated for ALIBI_ROPE (data=nullptr) ✅ Issue #96
  - Weight tying: `lm_head_weights = Tensor::from_ptr(embedding_weights.data, ...)` + `share_grad()`
  - PCGrad buffer: `Tensor::zeros({vocab_size, d_model})` in Phase1_Startup when `tie_embeddings=true`

  **Config Verified:**
  - `tie_embeddings: true` ✅
  - `positional_encoding: {use_learned:false, use_rope:true, use_alibi:true}` → ALIBI_ROPE ✅
  - `dropout_rate: 0.15` ✅
  - `embedding_scale = 1.0f` for debugging only not to be a workaround 

  **Manifest Entry Corrections (stale items from pre-audit):**
  - ~~"Token embedding lookup with √d_model scaling"~~ → REMOVED (Issue #140: scale=1.0f)
  - ~~"Issue #113: Sinusoidal position embeddings VERIFY present"~~ → DELETED from code (manifest line 141)
  - ~~"Sinusoidal MUST be applied with ALiBi/RoPE"~~ → NOT APPLIED (sinusoidal kernel deleted)
  - Shape verified: `[total_tokens, d_model]` = `[batch*seq_len, d_model]` ✅

  **⚠️ ARCHITECTURAL OBSERVATION (NOT necessarily a bug):**
  With sinusoidal deleted and learned position embeddings disabled for ALIBI_ROPE, identical tokens
  at different positions enter the encoder with IDENTICAL vectors. Position differentiation comes
  ONLY from ALiBi bias and RoPE rotation INSIDE attention. This is the LLaMA/Mistral/Gemma pattern
  and is architecturally valid. However, Issue #113 originally flagged this as causing `avg_cos=0.90`
  hidden state correlation and mode collapse. If mode collapse persists, this is the place to
  investigate — consider re-adding sinusoidal position embeddings (within autograd graph this time)
  or verifying that ALiBi/RoPE provide sufficient positional differentiation for the model size.

  **Diagnostic Code Assessment:**
  - Issue #93/#95 diagnostic blocks (cudaMemcpy stats, column variance) are INSIDE the
    `if (use_learned_pos_emb)` guard → NEVER EXECUTED with ALIBI_ROPE. Dead diagnostic code.
  - Issue #91 diagnostic (full embedding dump after ScratchBlock) is behind
    `ENABLE_EXPENSIVE_DIAGNOSTICS` compile-time guard. Not a performance concern.

  ---

  **🔴 BUG A: ScratchBlock backward NEVER CALLED (parameters frozen) — [FIXED Issue #141]**

  `AutogradTraining.cu` line 1507 guards ScratchBlock backward with:
  ```cpp
  if (ctx.scratch_block && ctx.scratch_block->isEnabled() &&
      intermediates.embedding_tensor.has_grad()) {
  ```
  But `intermediates.embedding_tensor` is a NON-LEAF tensor (dropout output). Its `grad_`
  member is NEVER allocated because `ensure_grad()` is never called on it. The autograd
  chain writes gradients to GradFn-internal buffers (`DropoutGradFn::input_grad`), NOT to
  `tensor.grad_data()`. Therefore `has_grad()` → **false** → ScratchBlock backward is SKIPPED.

  **Impact:** `atom_projection_` and `atom_type_embeddings_`
  receive ZERO gradients. ScratchBlock parameters are FROZEN for the entire training run.
  The model cannot learn numeric/structural reasoning.

  **Fix (implemented):** Option (b) chosen — added `grad_output_tap` mechanism to base `GradFn`
  struct. `DropoutGradFn::apply()` copies `grad_output` to tap buffer before applying dropout mask.
  ScratchBlock backward uses this buffer instead of the NULL `tensor.grad_data()`.
  - `TensorContract_GPU.hpp`: Added `grad_output_tap`/`grad_output_tap_count` to `GradFn`
  - `TensorContract_GPU.cu`: `DropoutGradFn::apply()` copies to tap before mask
  - `AutogradTraining.cu`: Removed `has_grad()` guard, uses tap buffer
  - `TrainingState_GPU.hpp`: Added `scratchblock_grad_tap` Tensor
  - `Phase1_Startup.cu`: Allocates tap buffer when `use_scratch_block=true`

  ---

  **🟡 BUG B: positional_encoding config HARDCODED — JSON ignored (Rule 20 violation) — [FIXED Issue #141]**

  `Phase1_Startup.cu` line 748:
  ```cpp
  model_config.positional_encoding = GRIM::HyperParameters::DEFAULT_POSITIONAL_ENCODING;
  ```
  This ALWAYS sets `ALIBI_ROPE` regardless of `ai_config.json`'s `positional_encoding` block:
  ```json
  "positional_encoding": { "use_learned": false, "use_rope": true, "use_alibi": true }
  ```
  The JSON fields `use_learned`, `use_rope`, `use_alibi` are **NEVER PARSED** into the model config.
  The `parsePositionalEncodingType()` function exists but is **NEVER CALLED**.

  **Impact:** Currently benign (hardcoded default matches JSON intent). But if the JSON config is
  changed (e.g., `"use_rope": false` to test pure ALiBi), the change is **silently ignored**. This
  violates Rule 20 (no silent fallbacks). The config file is decorative for this setting.

  **Fix (implemented):** Parsed `positional_encoding` from JSON in `Phase1_Startup.cu` `loadConfiguration()`.
  - `Phase1_Startup.cu`: Added JSON parsing block for `positional_encoding` object
  - `HyperParameters_GPU.hpp`: Added `positional_encoding` field to `ModelArchitecture` struct
  - `Phase1_Startup.cu`: `initializeModel()` uses `arch.positional_encoding` instead of hardcoded default

  ---

  **🟡 BUG C: PCGrad `assert` is only guard against division-by-zero NaN — [FIXED Issue #141]**

  `kernel_pcgrad_combine` line 1574:
  ```cuda
  assert(total_norm_sq > 1e-12f && "kernel_pcgrad_combine: g_lm has near-zero norm!");
  s_proj_coef = total_dot / total_norm_sq;
  ```
  In Release builds (`NDEBUG`), `assert()` compiles to nothing. If `total_norm_sq ≈ 0` for any
  vocab row (e.g., numerical underflow in LM head backward), `0/0 = NaN` propagates:
  ```
  lm_row[i] = 0.0f * (1.0f - NaN) + 0.0f = NaN
  ```
  Currently doesn't trigger because LM head backward (full matmul) produces dense gradients for all
  vocab rows. But this is a LATENT BUG — any future change that makes LM head gradients sparse
  (e.g., sampled softmax, top-k loss) would cause NaN corruption in the entire gradient buffer.

  **Fix (implemented):** Replaced `assert` with runtime guard in `TensorContract_GPU.cu`:
  ```cuda
  if (total_norm_sq < 1e-12f) {
      s_proj_coef = 0.0f;  // Zero g_lm → nothing to project onto
  } else {
      s_proj_coef = total_dot / total_norm_sq;
  }
  ```

  ---

  **🟡 BUG D: Dead diagnostic code in wrong code path — [FIXED Issue #141]**

  Issue #93/#95 diagnostic blocks (token embedding stats, position embedding stats, column variance
  analysis) at lines 345-510 are INSIDE the `if (use_learned_pos_emb)` guard. With ALIBI_ROPE,
  `use_learned_pos_emb = false`, so these ~170 lines of diagnostic code (3x cudaMemcpy, 3x O(n)
  loops) NEVER execute. The diagnostics were written to debug embedding issues but are completely
  unreachable in the current configuration.

  **Fix (implemented):** Deleted ~170 lines of dead diagnostic code from `AutogradTraining.cu`.
  Kept only: `cudaFreeAsync`, `autograd::add()`, and `AG_INFO` log.

  ---

  **🟡 BUG E: copilot-instructions.md stale/contradictory with actual code — [FIXED Issue #141]**

  Line 736 of `.github/copilot-instructions.md` says:
  > `addSinusoidalPositionEmbeddingsKernel` in `AutogradTraining.cu`, called UNCONDITIONALLY
  > after token embedding lookup when using ALiBi/RoPE

  This kernel was **DELETED**. The instructions claim it exists and runs unconditionally. This
  creates false expectations for any contributor (or AI agent) reading the instructions.

  **Fix (implemented):** Updated `.github/copilot-instructions.md` Issue #113 section to
  accurately state that sinusoidal position embeddings were DELETED and no position embeddings
  are added when using ALiBi/RoPE. Added comprehensive Issue #141 entry.

  ---

  **🟡 BUG F: ScratchBlock gradient tap stale when dropout path not active — [FIXED Issue #142]**

  The Issue #141 tap hook existed only in `DropoutGradFn::apply()`. With embedding dropout disabled
  (`dropout_rate=0`) or identity-dropout paths, `embedding_tensor.grad_fn` is `AddGradFn` or
  `EmbeddingGradFn`, so tap data could remain stale while ScratchBlock backward still ran.

  **Fix (implemented):**
  - `TensorContract_GPU.cu`: Added tap copy support to `AddGradFn::apply()` and `EmbeddingGradFn::apply()`
  - `TensorContract_GPU.cu`: Hardened tap copy in `DropoutGradFn::apply()` (size checks + cudaMemcpy error checks)
  - `TensorContract_GPU.hpp`: Added `grad_output_tap_written` flag in `GradFn`
  - `AutogradTraining.cu`: Arms tap with `grad_output_tap_written=false`, clears tap buffer before backward, and throws if tap was not written before ScratchBlock backward

  ---

  **🟡 BUG G: ScratchBlock tap allocation not fail-loud — [FIXED Issue #142]**

  `Phase1_Startup.cu` allocated `scratchblock_grad_tap` with unchecked `cudaMalloc` then logged success.

  **Fix (implemented):** Added explicit `cudaMalloc` error check with throw in `Phase1_Startup.cu`.

  ---

  **🟡 BUG H: Shared architecture loader ignored positional_encoding (server/runtime path) — [FIXED Issue #142]**

  `HyperParameters::loadModelArchitecture()` did not receive authored positional fields from the raw `training.config` PBM leaves.
  `grim_text_server.cpp` also did not propagate `arch.positional_encoding` into `LanguageModelConfig`.

  **Fix (implemented):**
  - `HyperParameters_GPU.hpp`: Consume the authored positional encoding mode after raw config parsing
  - `grim_text_server.cpp`: Propagate `num_kv_heads`, `tie_embeddings`, and `positional_encoding` from `ModelArchitecture`

  ---

  **🟡 BUG I: Shared architecture loader still ignored `num_kv_heads`/`tie_embeddings` — [FIXED Issue #142]**

  `HyperParameters::loadModelArchitecture()` parsed only a subset of architecture fields.
  `num_kv_heads` and `tie_embeddings` remained at defaults in shared runtime paths, even if JSON changed.

  **Fix (implemented):**
  - `HyperParameters_GPU.hpp`: Added parsing for `training.config.num_kv_heads` and `training.config.tie_embeddings`

  ---

  **🟡 BUG J: ScratchBlock enabled + missing tap buffer could silently skip backward — [FIXED Issue #142]**

  `executeAutogradBackward()` previously gated ScratchBlock backward with `&& ts->scratchblock_grad_tap.data`.
  If ScratchBlock was enabled but tap allocation failed or was skipped by an alternate init path, backward could be skipped silently.

  **Fix (implemented):**
  - `AutogradTraining.cu`: Added explicit throw when ScratchBlock is enabled but `scratchblock_grad_tap.data` is null

  ---

  **🟡 BUG K: `use_learned` positional mode parsed but position table never allocated — [FIXED Issue #143]**

  `Phase1_Startup.cu` maps the removed learned-position mode to
  `PositionalEncodingType::NONE`, and `AutogradTraining.cu` treats `NONE` as the
  additive learned-position path. But `TrainingTensors::initializeParams()` never
  allocated `position_embedding_weights` for `NONE`, so the forward gate:
  `if (use_learned_pos_emb && position_embedding_weights.data)` stayed false.
  Result: learned positional mode silently degraded to "no additive position embeddings".

  **Fix (implemented):**
  - `TrainingTensors.cu`: Allocate/init `position_embedding_weights` when `positional_encoding == NONE`
  - `Phase1_Startup.cu`: Parse positional encoding with fail-loud type validation

  ---

  **🔴 RE-AUDIT FINDING 1 (PERF CRITICAL): `g_debug_sync_after_every_kernel = true` — [FIXED]**

  `TensorContract_GPU.cu` line 175 had debug flag hardcoded to `true`, causing
  `cudaStreamSynchronize()` after EVERY kernel launch via `trackKernelLaunch()`.
  This serializes the entire GPU pipeline — ~10-50x throughput reduction.

  **Fix:** Changed to `false`. Comment updated to warn about performance impact.

  ---

  **🟡 RE-AUDIT FINDING 2: Stale Issue #92 AIAYN comments in embedding kernels — [FIXED]**

  4 locations in `TensorContract_GPU.cu` referenced deleted Issue #92 sqrt(d_model) scaling.
  Updated to reference Issue #140 (scale=1.0f, AIAYN removed).

  ---

  **🟡 RE-AUDIT FINDING 3 (RULE 20): `assert()` in embedding kernels compiled out in Release — [FIXED]**

  `kernel_embedding_forward` and `kernel_embedding_backward` used `assert()` for OOB
  token ID bounds checking. In Release builds (`NDEBUG`), `assert()` is a no-op —
  OOB access would silently corrupt GPU memory. Replaced with `if + printf + __trap()`
  which works in both Debug and Release builds. `__trap()` causes `cudaErrorLaunchFailure`
  which is caught by `trackKernelLaunch()` host-side error checking.

  ---

  **🟡 RE-AUDIT FINDING 4: Dead lambda in `autograd::add()` — [FIXED]**

  3-line unused lambda declared but never invoked (actual addition uses `TensorContract::add()`).
  Deleted.

  ---

  **🔴 RE-AUDIT FINDING 5 (CORRECTNESS): Inference autograd context omitted ScratchBlock side-channel pointers — [FIXED]**

  `initAutogradContext(..., batch_size, seq_len, ...)` did not populate
  `token_numeric_values` / `token_numeric_mask` from
  `TrainingState` caches. With ScratchBlock enabled this caused:
  `ScratchBlockLayer::forward requires token numeric side-channel` during sampling/inference.

  **Fix (implemented):**
  - `AutogradTraining.cu`: both `initAutogradContext` overloads now default
    ScratchBlock side-channel pointers from `TrainingState` cached buffers.

  ---

  **🟡 RE-AUDIT FINDING 6 (RULE 20): Learned positional mode could silently skip position embeddings in inference — [FIXED]**

  `InitinferenceState.cu` did not allocate `position_embedding_weights` for
  `positional_encoding == NONE` (learned/additive mode). Forward used:
  `if (use_learned_pos_emb && position_embedding_weights.data)` so a null table
  silently degraded to "no additive position embeddings".

  **Fix (implemented):**
  - `InitinferenceState.cu`: allocate `position_embedding_weights` when
    `cfg.positional_encoding == NONE`.
  - `AutogradTraining.cu`: fail loud (`throw`) when learned mode is selected
    but `position_embedding_weights.data` is null.

  ---

  **🟡 RE-AUDIT FINDING 7 (PERF CRITICAL): Seeded dropout forced host synchronization per call — [FIXED]**

  `autograd::dropout(x, p, seed, ...)` called `cudaStreamSynchronize(stream)`
  before `cudaFree(mask)`. This serialized execution for every dropout call
  (embedding + per-layer dropout), causing avoidable throughput collapse.

  **Fix (implemented):**
  - `TensorContract_GPU.cu`: replaced sync+free with stream-ordered
    `cudaFreeAsync(mask, stream)` and explicit error checking.

---

### 2.3 ScratchBlock Atom Substitution

- [] **Layers/ScratchBlock/ScratchBlockReasoning_GPU.cu**  & FIXED
  - In-place token substitution for structural atoms (numbers, URLs, emails, paths, dates)
  - **Issue #90**: Buffer desync after autograd::add() (Jan 2026) - FIXED
    - **Root Cause**: `autograd::add(emb, pos_emb)` created NEW buffer, but ScratchBlock operated on `cached_embeddings` (old buffer), Layer 0 received stale pre-ScratchBlock data
    - **Fix**: Added cudaMemcpyAsync after `ctx.scratch_block->forward()` to copy output back to `ctx.embedding_tensor.data`
  - **Issue #141 (BUG A)**: ScratchBlock backward NEVER called (Feb 2026) - FIXED
    - **Root Cause**: Backward guard checked `embedding_tensor.has_grad()` which was ALWAYS false (dropout output is non-leaf, gradients written to internal buffer, not tensor.grad_data())
    - **Symptom**: `atom_projection_` and `atom_type_embeddings_` received ZERO gradients for entire training
    - **Fix**: Added `grad_output_tap` field to GradFn base struct, DropoutGradFn copies `grad_output` to tap buffer before applying mask
    - **Implementation**: `scratchblock_grad_tap` buffer allocated in TrainingState (Phase1_Startup.cu), tap set before `loss_tensor.backward()`, ScratchBlock backward uses captured gradient
    - Files modified: TensorContract_GPU.hpp (GradFn tap), TensorContract_GPU.cu (DropoutGradFn tap copy), AutogradTraining.cu (tap setup + ScratchBlock backward), TrainingState_GPU.hpp (tap buffer), Phase1_Startup.cu (allocation)
  - Pattern verified: cudaMemcpyAsync after forward() copies output back to embedding_tensor.data
  - Pattern to verify: Atoms are packed into the vocab at the right length.

---

### 2.4 Shared/ScratchBlock (Pinned Memory Pool)

- [x] **Shared/ScratchBlock/ScratchBlockPool_GPU.{hpp,cu}** DELETED
  - The pool was not ScratchBlock reasoning state; it was a pinned CPU→GPU batch-upload staging workaround.
  - Its only live consumer was `uploadBatchToDevice()`, so the subsystem was removed instead of preserving a TrainingState catch-all field.
  - Batch upload now copies `BatchPayload` host vectors directly into TrainingState device cache tensors.
  - Deleted associated TrainingState members and LanguageModel scratch-pool toggles.

---

### 2.5 Encoder Layers (6-12x this sequence)

For each encoding layer (Layer 0 → Layer 11):

#### 2.5a Per-Token Rare Token Weighting

- [ ] **Shared/RareTokens/RareTokens_GPU.cu**
  - Weights batch composition by rare token frequency
  - No known issues
  - Pattern to check: Verify applied BEFORE forward pass

#### 2.5b Layer Normalization (Pre-Attention)

- [ ] **Layers/Encoding/Encoding_GPU.cu** (RMSNorm pre-attention)
  - RMSNorm forward: `y = (x - μ) / sqrt(σ² + ε) * γ + β`
  - **Issue #105**: Diagnostic formula for expected RMSNorm output - correctly fixed (NOT a bug)
  - Pattern to check: Verify RMSNorm computes `output_rms = input_rms * gamma_rms / sqrt(input_rms² + eps)`
  - Pattern to check: Verify gamma is learned parameter (not frozen)

#### 2.5c QKV Projection

- [x] **TensorContract/autograd QKV path**
  - QKV projection is `autograd::matmul(ln1_out, W_qkv, transpose_b=true)` so gradients stay on the tape.
  - `Layers/Attention/QKV_Projector.{hpp,cu}` is deleted; do not recreate it.
  - Pattern to check: `qkv_out -> autograd::split_and_reshape_qkv() -> Flash Attention`, with split/merge backed only by TensorConversion raw layout kernels.

#### 2.5d Positional Bias Application

- [ ] **Shared/PBM/PositionalBiasMethod.cu**
  - Adds ALiBi or RoPE bias to attention scores
  - **Issue #47**: Max sequence length scaling - VERIFY applied
  - **Issue #78**: Max bias clamping (-10.0) prevents underflow - VERIFY applied
  - Pattern to check: Look for `max(abs(slope)) <= max_bias / d_min` validation

#### 2.5e Flash Attention (Forward)

- [ ] **Layers/FlashAttention/Flash_Attention_Kernal.cu**
  - Flash Attention v2 forward pass
  - **Issue #72**: GQA dK/dV buffer allocation for num_heads - VERIFY (not num_kv_heads)
  - **Issue #78**: ALiBi bias integration - VERIFY
  - GQA support with heads_per_kv_group - VERIFY kernel logic
  - Pattern to check: Verify allocates dK/dV with size `[b * seq * num_heads * hd]`, not `[b * seq * num_kv_heads * hd]`
  - Pattern to check: Verify causal mask applied (training uses causal attention)

#### 2.5f Attention Output Projection

- [ ] **Layers/Encoding/Encoding_GPU.cu** (W_o projection + bias)
  - Projects attention output back to hidden size
  - Uses `autograd::broadcast_add()` for bias (Issue #97)
  - **Issue #97**: Bias addition MUST use autograd::broadcast_add() - VERIFY NOT using launchFFNBiasAdd()
  - Pattern to check: Search for `autograd::broadcast_add(attn_out, b_o)` - MUST be present
  - Pattern to check: Search for `launchFFNBiasAdd` - should ONLY appear in deprecated paths, NOT in production forward

#### 2.5g Residual + LayerScale

- [ ] **Layers/Encoding/Encoding_GPU.cu**
  - Adds `gamma_attn[d] * attention_output[t,d]` to input (per-channel LayerScale residual)
  - **Issue #129**: LayerScale init_value changed from 0.1 → 1.0 - VERIFY in config
  - Pattern to check: Verify `ai_config.json` has `layer_scale.init_value = 1.0`
  - Pattern to check: If init_value=0.1, gradients vanish (10x attenuation)

#### 2.5h Layer Normalization (Pre-FFN)

- [ ] **Layers/Encoding/Encoding_GPU.cu** (RMSNorm pre-FFN)
  - Same RMSNorm as 2.5b
  - Pattern to check: Verify output shape matches FFN input expectation

#### 2.5i Feed-Forward Network

- [ ] **Layers/FeedForward/Feed_Forward_GPU.cu**
  - FFN: `W2 * GELU(W1 * x + b1) + b2`
  - **Issue #97**: Bias additions MUST use autograd::broadcast_add() - VERIFY
  - **Issue #25**: FFN post-GELU cache MUST be written - VERIFY `args.cache_ffn_output` populated
  - Pattern to check: Search for `autograd::broadcast_add(ffn_hidden, b2)` - MUST be present
  - Pattern to check: Search for `cudaMemcpyAsync(...post_gelu, args.cache_ffn_output)` - MUST be present
  - Pattern to check: NO `launchFFNBiasAdd()` calls without autograd wrapper

#### 2.5j GELU Activation

- [ ] **Shared/Activations/GELU/GELU.cu**
  - GELU forward/backward
  - No known issues
  - Pattern to check: Verify uses accurate approximation (not simplified)

#### 2.5k Residual + LayerScale

- [ ] **Layers/Encoding/Encoding_GPU.cu**
  - Adds per-channel LayerScale vectors before residual adds:
    - `residual1[t,d] = input[t,d] + gamma_attn[d] * attn_output[t,d]`
    - `output[t,d] = residual1[t,d] + gamma_ffn[d] * ffn_output[t,d]`
  - Verify `gamma_attn`/`gamma_ffn` have shape `[1, d_model]`, not scalar `[1]`

---

### 2.6 LM Head (Language Model Output)

- [ ] **Layers/LMHead/lm_head_GPU.cu**
  - Projects hidden states to vocabulary logits
  - **Issue #125**: Hidden state centering was ROW-WISE (WRONG) - VERIFY FIXED in AutogradTraining
  - **Issue #132**: MUST apply center_columns() THEN center_rows() - VERIFY BOTH applied
  - **Issue #98**: With tied embeddings, logits MUST be scaled by √d_model - VERIFY autograd::scale() applied
  - Pattern to check: Verify uses autograd::center_columns() THEN autograd::center_rows()
  - Pattern to check: Verify when `tie_embeddings=true`, applies `autograd::scale(logits, sqrt(d_model))`
  - Pattern to check: Search for deprecated `launchCenterHiddenStates()` - if present, MUST use column centering

---

### 2.7 Numeric Head (Auxiliary Loss)

- [ ] **Layers/NumericHead/numeric_head_GPU.cu**
  - Auxiliary prediction head for numeric values (Huber loss)
  - No known issues in forward pass
  - Pattern to check: Verify forward output shape matches training batch

---

### 2.7a Teacher Logits (Knowledge Distillation)

- [ ] **Shared/TeacherLogits/TeacherLogits_GPU.cu**
  - Handles teacher model logits for knowledge distillation
  - Stores/loads pre-computed teacher predictions
  - Pattern to check: Verify teacher logits properly aligned with student batch
  - Pattern to check: Check if distillation loss is actually used (could be dead code)
  - Pattern to check: Verify temperature scaling applied if using distillation
  - **STALE CODE CHECK**: Verify knowledge distillation is enabled in config (or mark as unused)

---

### 2.8 Dropout (if enabled)

- [ ] **Shared/Dropout/Dropout_GPU.cu**
  - Dropout regularization
  - Pattern to check: Verify disabled during validation

---

---

## PHASE 2: TRAINING LOOP (Loss Computation)

### 3.1 Loss Computation Orchestrator

- [] **LanguageModel_Training.cu**  & FIXED
  - Backward orchestrator: delegates to `executeAutogradBackward()` in AutogradTraining.cu
  - **FIXED (Issue #138)**: Added CUDA event-based timing to decompose `computeGradNorm` into kernel time vs backward pipeline drain
  - `computeGradNorm` now logs `wall_time` (3-53ms) AND `gpu_kernel_time` (~1ms) separately — eliminates misleading timing variance
  - **Issue #57**: Numeric head backward VERIFIED called from `executeAutogradBackward()` when enabled ✅
  - Backward path verified: `model->backward(loss)` → `executeAutogradBackward(ctx)` → `loss_tensor.backward(nullptr)` + numeric head backward (if enabled) ✅
  - Pattern verified: Loss stored in `training_state.autograd_intermediates.loss_tensor`, gradients flow to all parameters ✅

---

### 3.2 Unified Loss (Text Cross-Entropy)

- [] **Shared/Loss/ComputeLoss/AutogradLoss.cu** 
  - **ONLY LOSS PATH** - Combines CE + focal + label smoothing + entropy regularization
  - `autograd::unified_loss()` returns scalar Tensor with NLLLossGradFn → LogSoftmaxGradFn chain ✅
  - Gradient formula: `∂L/∂logits = (softmax - one_hot) / N` (mean reduction) ✅
  - Called exclusively from `computeAutogradLoss()` in AutogradTraining.cu ✅
  - No stale paths found — unified_loss is the sole production loss computation
  - Formula: `L = α(1-p_t)^γ * CE_smooth + λ * H(p)`
  - **Rule 20**: `UnifiedLoss_GPU.cu` and `ComputeLoss_GPU.cu` DELETED - VERIFY NOT referenced
  - Pattern to check: Verify config controls focal (alpha, gamma), smoothing (epsilon), entropy (lambda)
  - Pattern to check: Search for `autograd::unified_loss()` - should be THE loss function
  - Pattern to check: Inspect ai_config.json for loss configuration:
    - `entropy_reg.lambda` should be 0 (was masking true CE in Issue #133)
    - `focal.enabled` should be false (no benefit at uniform)
    - `weight_decay` should be 0.1 (not 0.3, too aggressive)
    - verify NOT at extreme values like lambda=0.1

---

### 3.3 Loss Batch Computation

- [] **Shared/Loss/ComputeLoss/ComputeLossBatch.cu**  & DELETED — separate validation/eval autograd loop removed
  - **FULLY REFACTORED**: Inline loss code (~400 lines) DELETED, delegates to `computeAutogradLoss(autograd_ctx, payload)` ✅
  - File now contains ONLY: GPU copies → autograd context setup → forward pass → loss config build → `computeAutogradLoss()` call → return
  - **DELETED (362 lines total)**:
    - Dead `LossContext::TensorViews` construction (~30 lines)
    - Dead `Loss::LossConfig` construction (~10 lines) 
    - Distillation/preference copy code (~50 lines)
    - Static warning checks, dead variables
    - Inline `ag_loss_config` construction
    - Inline `autograd::unified_loss()` call + readback
    - Spike diagnostic block (~80 lines)
    - Inline `NumericLossInputs`/`launchNumericLoss` (~50 lines)
    - Inline learned weighting (~50 lines)
    - All STEP-A through STEP-J fprintf diagnostics
  - **DEAD INCLUDES REMOVED**: `AutogradLoss.hpp`, `LossContext.hpp`, `NumericLoss_GPU.hpp`, `TeacherLogits_GPU.hpp`
  - `FORWARD_DIAG` block kept (compile-time guarded, zero cost when disabled) ✅
  - Mean reduction verified: handled by `computeAutogradLoss()` using `payload.valid_tokens` ✅
  - Zero compile errors confirmed ✅

---

### 3.4 Numeric Loss (Auxiliary Head)

- [] **Shared/Loss/NumericLoss/NumericLoss_GPU.cu**  (integrated into computeAutogradLoss)
  - Huber loss for numeric predictions
  - **NOW CALLED FROM**: `computeAutogradLoss()` in AutogradTraining.cu ✅
  - **FIXED (Issue #137)**: `scaleNumericGradKernel` uses `1/valid_text_tokens` (same denominator as text CE mean reduction)
  - **FIXED (Issue #137)**: `log_var` gradients normalized by `1/(1 + loss²)` to bound their contribution
  - **FIXED (Issue #137)**: Weight+bias gradients post-scaled by `sqrt(N_atoms / valid_tokens)` to normalize dense accumulation variance
  - **Issue #74**: Mean reduction verified — `scaleNumericGradKernel` applies `1.0f / valid_text_tokens` ✅
  - **Issue #74b**: Race condition FIXED — `cudaStreamSynchronize()` added between `numericLossKernel` and `scaleNumericGradKernel` ✅

---

### 3.5 Old Loss Sub-Modules (DELETED — Rule 26)

- [] **ENTIRE OLD LOSS SYSTEM DELETED** ✅ Dead code — production uses `autograd::unified_loss()` exclusively
  - **Files DELETED:**
    - `Shared/Loss/CrossEntropy/CrossEntropy_GPU.cu` + `.hpp` (standalone CE kernel, zero callers)
    - `Shared/Loss/LabelSmoothing/LabelSmoothing_GPU.cu` + `.hpp` (zero callers)
    - `Shared/Loss/TKML/TKML_GPU.cu` + `.hpp` (distillation KL, zero callers)
    - `Shared/Loss/Preference/Preference_KL_GPU.cu` + `.hpp` (preference KL, zero callers)
    - `Shared/Loss/Divergence/Divergence_GPU.cu` + `.hpp` (token masking + guess feedback, zero callers)
    - Entire directories removed: `CrossEntropy/`, `LabelSmoothing/`, `TKML/`, `Preference/`, `Divergence/`
  - **Loss.hpp GUTTED:** All structs and function declarations deleted:
    - `Loss::Term` enum, `Loss::LossContext`, `Loss::LossBreakdown`, `Loss::DeviceBuffers`, `Loss::AuxiliaryBatchViews`
    - `Loss::LossConfig` (composite) and all sub-configs: `LabelSmoothingConfig`, `DistillationConfig`, `PreferenceKLConfig`, `FocalLossConfig`, `MaskConfig`, `GuessFeedbackConfig`, `LimitsConfig`, `EntropyRegConfig`
    - All function declarations: `computeLossTerms`, `applyLabelSmoothing`, `accumulateDistillationKL`, `accumulatePreferenceKL`, `applyFocalLossScaling`, `applyTokenMasking`, `blendGuessFeedback`, `computeInfoNCELoss`, `computeCosineSimilarityLoss`, `validate`
  - **LossContext.hpp DELETED:** final loss-options wrapper moved into `HyperParameters::LossConfigHP` in `HyperparameterGroupings.hpp`; no loss-owned/model-owned config wrapper remains.
  - **CMakeLists.txt UPDATED:** Removed `CrossEntropy_GPU.cu` from `grim_training_kernels` STATIC library
  - **Verification:** `Loss::LossContext` was NEVER constructed in any .cu file. All 5 sub-module functions only had declarations + implementations — zero callers.
  - **Production loss path**: `BatchPayload + LossConfigHP → AutogradContext → computeAutogradLoss() → unified_loss()` via AutogradLoss.cu
  - **Still alive:**
    - `HyperParameters::LossConfigHP` — config flow: authoritative `TrainingHyperparameters` → Phase2/startup local `lossConfigHP(...)` derivation → autogradTrainingStep → computeAutogradLoss/MTP → unified_loss
    - `autograd::unified_loss()` — the one true loss path (AutogradLoss.cu/hpp)
    - `NumericLoss/NumericLoss_GPU.cu/hpp` — auxiliary numeric regression head (called from AutogradTraining.cu)

---

---

## PHASE 2: TRAINING LOOP (Backward Pass & Gradient Flow)

### 4.1 Backward Orchestrator

- [] **AutogradTraining.cu**  & FULLY REFACTORED (2070 lines, 4 major functions)
  - **PRIMARY BACKWARD PATH**: `executeAutogradBackward(ctx)` — handles text loss + numeric loss + learned weighting + ScratchBlock backward
  - **PRIMARY LOSS PATH**: `computeAutogradLoss(ctx, loss_config, mtp_alpha_effective)` — text CE + auxiliary loss assembly, returns `LossResult`
  - **PHASE2 TRAINING ORCHESTRATOR**: `Phase2_TrainingLoop.cu::processBatch(...)` — upload + explicit `Forward::executeModelForward(...)` + loss + backward
  - **FIXED (Issue #140)**: Removed √d_model embedding scaling (scale=1.0f) — eliminates 27.7x gradient asymmetry for tied weights
  - **FIXED (Issue #141)**: ScratchBlock backward now uses gradient tap buffer — `atom_projection_` and `atom_type_embeddings_` are trained ✅
  - **FIXED (Issue #141)**: `positional_encoding` config parsed from JSON (was hardcoded to ALIBI_ROPE)
  - **FIXED (Issue #141)**: PCGrad NaN guard changed from `assert` to runtime check (prevents 0/0 in Release builds)
  - **FIXED (Issue #142)**: Gradient tap support added to `AddGradFn` and `EmbeddingGradFn` (not just DropoutGradFn)
  - **DELETED (~170 lines)**: Dead Issue #93/#95 diagnostic code inside `if (use_learned_pos_emb)` guard (never executed with ALIBI_ROPE)
  - Backward verified: `loss_tensor.backward(nullptr)` → propagates through full autograd graph → all parameters receive gradients ✅
  - Numeric head backward verified: `numeric_head_output.backward(&grad_numeric_tensor)` called when enabled ✅
  - ScratchBlock backward verified: uses `scratchblock_grad_tap` buffer (Issue #141 fix) ✅
  - PCGrad verified: `kernel_pcgrad_combine` orthogonalizes embedding gradient when `tie_embeddings=true` ✅
  - **Issue #125**: Column centering verified — `CenterColumnsGradFn` correctly reduces hidden state correlation ✅
  - **Issue #132**: Row centering verified — applied AFTER column centering to eliminate gradient sign flip ✅
  - **Issue #127**: `centered_encoder_output` member verified — stored in ForwardContext, prevents use-after-free ✅

---

### 4.2 GradFn Lifecycle & Buffer Ownership

- [] **Shared/TensorContract/TensorContract_GPU.cu** (All GradFn subclasses)  & FIXED
  - Core autograd system
  - **Issue #126**: RMSNormGradFn stale input_grad pointer (Feb 2026) - FIXED
    - **Root Cause**: `capture_inputs()` stored raw pointer to `x.grad_data()` borrowed from input tensor. When `x` destructed (temporary from `autograd::add()`), buffer freed, leaving dangling pointer
    - **Symptom**: CUDA error 1 (invalid argument) when accessing freed memory during backward
    - **Fix**: RMSNormGradFn now allocates OWN gradient buffer via `cudaMalloc()` instead of borrowing. Added `bool owns_input_grad` to track ownership, destructor frees when owned
  - **Issue #127**: CenterColumnsGradFn use-after-free of RMSNormGradFn (Feb 2026) - FIXED
    - **Root Cause**: Local `Tensor centered_encoder_output` went out of scope after forward(), deleting CenterColumnsGradFn, which OWNED the RMSNormGradFn (transferred via `capture_input()`). `lm_input_tensor.grad_fn` pointer became dangling
    - **Symptom**: Crash after `[MATMUL-BWD-TO-A] About to call a_grad_fn->apply()` with invalid vtable access
    - **Fix**: Added `centered_encoder_output` member to ForwardContext (persists until clearIntermediates). `lm_input_tensor.grad_fn` now points to `centered_encoder_output.grad_fn` instead of temp's grad_fn
  - **Issue #142**: Gradient tap hardening (Feb 2026) - IMPLEMENTED
    - **Root Cause**: ScratchBlock backward needs encoder input gradient, but DropoutGradFn wrote to internal buffer (non-leaf tensor pattern)
    - **Fix**: Added `float* grad_output_tap` field to base `GradFn` struct
    - **Implementation**:
      - DropoutGradFn::apply(): Copies `grad_output` to tap buffer BEFORE applying dropout mask (with size checks + error checks)
      - AddGradFn::apply(): Added tap copy support (accumulates to tap when set)
      - EmbeddingGradFn::apply(): Added tap copy support for backward transparency
    - Files modified: TensorContract_GPU.hpp (tap field), TensorContract_GPU.cu (tap copies in all 3 GradFn types)
  - **Issue #56**: Return statements in forward functions - VERIFIED all return output (FFN bug fixed in Issue #71)
  - **Critical checks verified**:
    - `RMSNormGradFn::capture_inputs()` - allocates grad buffer with `cudaMalloc()`, sets `owns_input_grad=true` ✅
    - `RMSNormGradFn::~RMSNormGradFn()` - frees own buffer only when `owns_input_grad=true` ✅
    - `CenterColumnsGradFn::~CenterColumnsGradFn()` - does NOT delete input's GradFn (ownership transferred to persistent tensor) ✅
    - SDPA forward/GradFn setup allocates GQA backward buffers for num_heads (not num_kv_heads) to prevent overflow (Issue #72); no duplicate `ScaledDotProductAttentionGradFn::save()` path ✅
  - Pattern verified: ALL GradFn subclasses set `data_cached = true` (Issue #48 fix) ✅
  - Pattern verified: `grad_output_tap` size checks before copy, cudaMemcpy error checked ✅

---

### 4.3 Attention Backward (Flash Attention)

- [ ] **Layers/FlashAttention/Flash_Attention_Kernal.cu**
  - Flash Attention v2 backward pass
  - **Issue #84**: Preprocessing kernel (flash_bwd_dot_do_o_kernel) MUST be called FIRST - VERIFY present
  - **Issue #72 + #85**: GQA dK/dV temporary buffers use `num_heads`, then reduce to `num_kv_heads` with a plain sum - VERIFY present
  - **Issue #87**: Removed dQ/dK normalization that was crushing gradients - VERIFY NOT present
  - Pattern to check: Search for `flash_bwd_dot_do_o_kernel<<<` to verify preprocessing kernel call
  - Pattern to check: Search for `kernel_reduce_gqa_grads_BSHD_bf16_to_BHSD_fp32` and verify no `gqa_grad_scale` multiplication
  - Pattern to check: Verify NO scaling of dQ/dK to dV magnitude (Issue #87 would re-appear here)

---

### 4.4 Attention Diagnostics (Optional)

- [ ] **Layers/FlashAttention/AttentionDiagnostics.cu**
  - Logs Q/K/V magnitudes and distributions
  - Pattern to check: Verify diagnostic thresholds match current model scale

---

### 4.5 Encoder Backward Pass

#### 4.5a RMSNorm Backward (Pre-Attention)

- [ ] **Layers/Encoding/Encoding_GPU.cu** (RMSNorm backward)  
  - Computes gradient w.r.t. input and parameters (gamma, beta)
  - **Issue #126**: GradFn owns its gradient buffer - VERIFY
  - Pattern to check: Verify backward computes both input and weight gradients

#### 4.5b Attention Backward

- [ ] Flash Attention backward (see 4.3)

#### 4.5c Bias (W_o) Backward

- [ ] **Shared/TensorContract/TensorContract_GPU.cu** (BiasAddGradFn)
  - Backward for `autograd::broadcast_add(attn_out, b_o)`
  - **Issue #97**: Backward pools gradients and accumulates - VERIFY working correctly
  - Pattern to check: Verify BiasAddGradFn sums over batch dimension correctly
- [ ] **BiasGradientKernel.cu**
  - Low-level bias gradient summation kernel
  - Called by BiasAddGradFn backward pass
  - Pattern to check: Verify uses atomicAdd for gradient accumulation
  - Pattern to check: Verify NO race conditions in accumulation
#### 4.5d LayerScale + Residual Backward

- [ ] **Layers/Encoding/Encoding_GPU.cu**
  - Residual addition backward: `grad_input[t,d] += gamma[d] * grad_residual[t,d]`
  - Gamma backward: `grad_gamma[d] += sum_t(grad_residual[t,d] * sublayer_output[t,d])`
  - Pattern to check: Verify gamma grad is summed locally; CE/root backward already applies mean scaling

#### 4.5e RMSNorm Backward (Pre-FFN)

- [ ] Same as 4.5a

#### 4.5f FFN Backward (W1, W2, Biases)

- [ ] **Layers/FeedForward/Feed_Forward_GPU.cu**
  - Backward through W1, GELU, W2
  - **Issue #97**: Bias backward accumulates properly - VERIFY
  - **Issue #25**: FFN cache used for backward - VERIFY was actually written in forward
  - Pattern to check: Verify gradient accumulates into both b1 and b2
  - Pattern to check: Trace: grad_output → grad_W2 (GEMM with cached_ffn_hidden) → grad_hidden → GELU backward → grad_W1

#### 4.5g GELU Backward

- [ ] **Shared/Activations/GELU/GELU.cu**
  - GELU backward via Jacobian
  - Pattern to check: Verify numerical stability

#### 4.5h Encoding Layer Stacking (Layer 11 → Layer 0)

- [ ] Repeat 4.5a-g for each encoder layer (reverse order in backward)

---

### 4.6 LM Head Backward

- [ ] **Layers/LMHead/lm_head_GPU.cu**
  - Backward through LM head weight matrix and centering layers
  - **Issue #98**: With tied embeddings, backward scales gradient by √d_model - VERIFY
  - Pattern to check: Verify backward of `autograd::scale()` applies scaling
  - Pattern to check: Verify backward of center_rows() and center_columns() work correctly

---

### 4.7 Numeric Head Backward

- [ ] **Shared/Loss/NumericLoss/NumericLoss_GPU.cu** + **Layers/NumericHead/numeric_head_GPU.cu**
  - Numeric loss backward
  - **Issue #57**: VERIFY numeric_head_output.backward() is ACTUALLY CALLED in LanguageModel_Training.cu
  - **Issue #74**: Gradients scaled by 1/count - VERIFY applied
  - Pattern to check: In LanguageModel_Training.cu, search for backward call on numeric head

---

### 4.8 Embedding Backward

- [ ] **Shared/TensorContract/TensorContract_GPU.cu** (autograd::embedding backward)
  - Backward for embedding lookup
  - **Issue #92**: Backward includes √d_model scaling - VERIFY present
  - **Issue #19**: PCGrad for tied embedding/LM head weight conflicts - VERIFY logic correct
  - Pattern to check: Verify backward does weighted scatter-add with scale
  - Pattern to check: If tie_embeddings=true, verify PCGrad applied: `g_emb = g_emb - proj_{g_lm}(g_emb)`

---

### 4.9 ScratchBlock Backward

- [ ] **Layers/ScratchBlock/ScratchBlockReasoning_GPU.cu**
  - Backward through atom token substitution
  - Pattern to check: Verify backward propagates gradients to original token embeddings

---

---

## PHASE 2: TRAINING LOOP (Optimization Step)

### 5.1 Gradient Accumulation & Clipping

- [] **Phase2_TrainingLoop.cu** (post-accumulation clipping section)  & REFACTORED
  - **Issue #139**: Per-component gradient clipping (Feb 2026) - emb_lm_tied was dominating 88-99.6% of total gradient norm, crushing encoder gradients to near-zero when joint clipping applied single coefficient to ALL text parameters
  - **OLD**: Two-group clipping (text=emb+attn+ffn+rms, num=numeric_head) - when emb=99% of text_norm, clip_coef≈0.2 scaled ALL text params equally
  - **NEW**: Three independent clips:
    1. `emb clip` - clips LM_HEAD (+ EMBEDDING if untied) independently
    2. `enc clip` - clips ATTENTION + FFN + RMSNORM + SCRATCHBLOCK independently  
    3. `num clip` - clips NUMERIC_HEAD independently
  - Each component gets full per-token budget (avoids cross-contamination)
  - **Effect**: Encoder gradients maintain natural magnitude instead of being crushed by embedding spikes
  - Added `[EMB_GRAD_EQUATION]` gradient spike diagnostic - decomposes LM head matmul gradients by: `grad_B[vocab,d] = Σ_t hidden[t,d] × grad_logits[t,vocab]`
    - Shows decomposition: contribution from target-token positions vs contribution from all other positions
    - Includes hidden state statistics (H_rms, H_mean, H_max), grad_logits statistics (min/max/mean), and expected vs actual gradient magnitude
    - Detects WEIGHT_PARADOX_SOURCE when total contribution has wrong sign relative to loss gradient
  - **Batch Strategy Cleanup**: Removed SIMILARITY_GROUPED/RANDOM/WEIGHTED_RANDOM options, forced GREEDY (highest-loss tokens first) for consistent training
  - Files modified: Phase2_TrainingLoop.cu (clipping section ~lines 1650-1720, diagnostic ~lines 1800-1950)

- [] **Shared/TNC/Token-normalized_clipping.cu** REMOVED
  - Batch token stats are now owned by `BatchPayload` and authored during Phase1 planned-batch construction
  - Phase2 reads `HyperParameters::gradientClippingHP()` before calling `GradClip::clipGradientNorms`
  - Removed the unused clip-selection wrapper and dead `computeBatchTokenStats` recomputation path

---

### 5.2 Gradient Computation & Statistics

- [ ] **Shared/Gradients/GradientCC_GPU.cu**
  - Core gradient computation kernels on GPU
  - Handles gradient accumulation and synchronization
  - Pattern to check: Verify gradient buffers are zeroed before accumulation
  - Pattern to check: Verify atomicAdd used for concurrent accumulation

- [ ] **Shared/Gradients/GradientCC_Host_GPU.cu**
  - CPU-GPU gradient bridge code
  - Host-side gradient coordination and transfer
  - Pattern to check: Verify proper CPU-GPU synchronization
  - Pattern to check: Verify no unnecessary CPU-GPU copies

- [ ] **Shared/GradNorm/GradNormGPU.cu**
  - Computes total gradient norm across all parameters
  - **Issue #24**: cudaStreamSynchronize() overhead (7+ seconds/batch) - VERIFY `sync_for_host_read=false` used
  - Pattern to check: Verify sync_for_host_read parameter passed correctly (false except every 10 iterations)

- [ ] **Shared/Gradients/GradStatsCollector.cu**
  - Collects per-layer, per-component gradient norms and cosine similarities
  - Pattern to check: Verify includes all components: text_ce, attention, ffn, embedding, lm_head, numeric

---

### 5.3 AdamW Optimizer Step

- [ ] **Shared/Optimizers/AdamW/AdamW_Kernal_GPU.cu**
  - Applies AdamW update: `param -= lr * m_hat / (sqrt(v_hat) + eps) + weight_decay_term`
  - Pattern to check: Verify decoupled weight decay formula
  - Pattern to check: Verify uses the scheduled learning rate provided by Phase2

---

### 5.4 Soft Restart Mechanism

- [ ] **Shared/SoftRestart/SoftRestart.cu**
  - Resets optimizer state on plateau detection
  - Pattern to check: Verify uses telemetry signal (not just loss stalling)

---

### 5.6 Telemetry System

- [ ] **Shared/Telemetry/TelemetryLattice_GPU.cu**
  - 8-level hierarchical anomaly detection
  - 10-dimensional per-metric vectors (mu, sigma, volatility, slope, persistence, etc.)
  - Pattern to check: Verify metrics feed properly from loss computation

- [ ] **Shared/Telemetry/TelemetryControl_GPU.cu**
  - Telemetry configuration and control logic
  - Manages telemetry enable/disable flags and thresholds
  - Pattern to check: Verify threshold values loaded from config, not hardcoded
  - Pattern to check: Verify telemetry can be disabled (performance overhead when enabled)
  - Pattern to check: Check for any always-on telemetry that should be conditional

---

### 5.7 Validation & Checkpointing

- [] **Phase2_TrainingLoop.cu** (validation section)  & FIXED
  - Periodic validation and checkpoint saving
  - **Issue #85**: Validation token budget exceeds training buffer size (Jan 2026) - FIXED
    - **Root Cause**: Hardcoded Phase2 token budget `8192` exceeded training allocation (batch_size × max_seq_len = 7168)
    - **Symptom**: STATUS_STACK_BUFFER_OVERRUN crash (exit -1073740791) after "Created N validation batches"
    - **Fix**: Changed to use Phase1-authored token budget instead of hardcoded constant
    - Added logging: `"[Val] Token budget: X (model limit: Y)"`
  - **Issue #115**: Diagnostic buffer mismatch (Feb 2026) - FIXED
    - **Root Cause**: Diagnostics read `cached_encoder_output` (pre-centering) instead of `centering_scratch_tensor` (post-centering)
    - **Fix**: Added `use_centering` parameter to diagnostic functions, conditionally select buffer based on `cfg.lm_head_center_hidden_states`
  - Pattern verified: Uses config limit, not hardcoded constants
  - Pattern verified: Validation respects same buffer size as training

- [ ] **Shared/TrainingState/TrainingStateGPU.cu**
  - Training state resource management (buffers, streams, cuBLAS handle)
  - Pattern to check: Verify centralized resource allocation (no distributed allocations)
  - Pattern to check: Verify no double-frees (especially aliased pointers like tied embedding)

---

---

## PHASE 3: CLEANUP & FINALIZATION

### 6.1 Final Checkpoint Save

- [ ] **Layers/Serialization/Serialization_GPU.cu**
  - Low-level FlatBuffers serialization kernels
  - Saves model weights to checkpoint format
  - **GQA compatibility**: Validates checkpoint num_kv_heads matches config - VERIFY validation present
  - Pattern to check: Verify saves all weights including adaptive components (LayerScale, RMSNorm gains, embedding scale)

- [ ] **Common/grim_model_serialization.cu**
  - High-level serialization orchestration
  - Coordinates checkpoint save/load operations
  - Pattern to check: Verify proper error handling for I/O failures
  - Pattern to check: Verify checkpoint versioning and compatibility checks

---

### 6.2 Training Summary

- [ ] **Phase3_Cleanup.cu**
  - Writes training summary FlatBuffer
  - Pattern to check: Verify includes final metrics and diagnostics

---

### 6.3 Resource Cleanup

- [ ] **TrainingState** resource cleanup
  - Frees GPU buffers, destroys streams, destroys cuBLAS handle
  - Pattern to check: Verify no double-frees (especially for aliased pointers like tied embedding)

---

---

## STALE CODE VERIFICATION

Use this section to track stale code patterns found during audit:

- [ ] Dead `Embedding_GPU.cu` - confirm NOT referenced
- [] Dead `CrossEntropy_GPU.cu` - ✅ DELETED (Rule 26, along with all old loss sub-modules)
- [ ] Deprecated `launchFFNBiasAdd()` - confirm NOT in production forward pass
- [ ] Deprecated `launchCenterHiddenStates()` - confirm NOT used (replaced by autograd)
- [ ] Old `UnifiedLoss_GPU.cu` - confirm NOT referenced
- [ ] Old `ComputeLoss_GPU.cu` - confirm NOT referenced
- [ ] Stale `EmbeddingLayer` class - confirm NOT instantiated
- [] Hardcoded Phase2 token-budget constant - ✅ DELETED; token budget is Phase1-authored capacity/config
- [ ] Raw `cudaStream_t` allocations - confirm NOT in layer modules
- [ ] Raw `cudaMalloc()` outside TrainingState - confirm NOT present in layers

---

## Quick Reference: Known Issues by File

| File | Issue # | Fix Required | Status |
|------|---------|--------------|--------|
| Phase1_Startup.cu | #106 | W_qkv scaling (1/√d_m) | Check |
| Phase1_Startup.cu | #107 | LCG PRNG splitmix64 | Check |
| Phase1_Startup.cu | #110 | PCGrad allocation | Check |
| TrainingTensors.cu | #106 | W_qkv init scaling | Check |
| TrainingTensors.cu | #107 | LCG PRNG quality | Check |
| TensorContract_GPU.cu | #126 | RMSNormGradFn ownership | Check |
| TensorContract_GPU.cu | #127 | CenterColumnsGradFn lifecycle | Check |
| TensorContract_GPU.cu | #92 | Embedding scale √d_m | Check |
| lm_head_GPU.cu | #125 | Column centering (WRONG) | Fix in AutogradTraining |
| lm_head_GPU.cu | #132 | Col + row centering | Check AutogradTraining |
| lm_head_GPU.cu | #98 | Embedding scale matching | Check |
| Feed_Forward_GPU.cu | #97 | autograd::broadcast_add | Check |
| Encoding_GPU.cu | #97 | autograd::broadcast_add | Check |
| Encoding_GPU.cu | #129 | LayerScale init_value=1.0 | Check ai_config.json |
| Flash_Attention_Kernal.cu | #84 | Preprocessing kernel | Check |
| Flash_Attention_Kernal.cu | #72+73 | GQA reduction scale | Check |
| Flash_Attention_Kernal.cu | #87 | NO dQ/dK crushing | Check |
| Flash_Attention_Kernal.cu | #78 | ALiBi max bias clamp | Check |
| AutogradTraining.cu | #125 | Column centering | Check |
| AutogradTraining.cu | #132 | Row centering | Check |
| AutogradTraining.cu | #127 | CenterColumnsGradFn | Check |
| LanguageModel_Training.cu | #57 | Numeric backward call | Check |
| AutogradLoss.cu | unified | ONLY loss system | Verify |
| NumericLoss_GPU.cu | #74 | Mean reduction scaling | Check |
| NumericLoss_GPU.cu | #74b | Kernel sync | Check |
| ScratchBlockReasoning_GPU.cu | #90 | Buffer copy after forward | Check |
| PositionalBiasMethod.cu | #47 | ALiBi slope scaling | Check |
| PositionalBiasMethod.cu | #70 | RoPE NTK scaling | Check |
| PositionalBiasMethod.cu | #78 | ALiBi max bias | Check |
| Phase2_TrainingLoop.cu | #85 | Validation token budget | Check |
| Phase2_TrainingLoop.cu | #115 | Diagnostic buffer select | Check |
| GradNormGPU.cu | #24 | Sync overhead | Check |

---

## Investigation Tips

1. **Use grep to search patterns**:
   ```powershell
   grep -r "pattern" resources/models/GRIM-text/training/ --include="*.cu" --include="*.hpp"
   ```

2. **Cross-reference issue numbers**: Open PLATEAU_BUG_INVESTIGATION.md and search issue number

3. **Trace data flow**: Follow tensor transformations using the file dependency graph (bottom of original manifest)

4. **Search for dead code patterns**:
   - Search for `Embedding_GPU` - should have ZERO matches
   - Search for `CrossEntropy_GPU` or `cross_entropy_loss()` - should have ZERO matches (files DELETED)
   - Search for `launchFFNBiasAdd` in forward paths - should have ZERO matches

5. **Verify fixes**:
   - For each issue listed, search for the fix keyword(s)
   - Example: Issue #92 → search for `embedding_scale` or `sqrt(d_model)`
   - If fix keyword not found, that's your problem!

---

**Last Updated**: February 7, 2026 (Loss System Dead Code Purge)  
**Status**: Loss computation sections (3.1-3.5), backward orchestrator (4.1-4.2), ScratchBlock (2.3), and Phase2 optimization (5.1, 5.7) fully documented with completed refactoring work  
**Recent Updates**:
- **Old Loss System DELETED (Rule 26)**: Loss.hpp gutted, LossContext::TensorViews deleted, 10 dead .cu/.hpp files removed, 5 directories purged, CrossEntropy_GPU.cu removed from CMakeLists. BatchPayload confirmed as single source of truth via AutogradContext.
- ComputeLossBatch.cu deleted; Phase2 no longer runs a second validation/eval autograd loop
- AutogradTraining.cu fully refactored (computeAutogradLoss, autogradTrainingStep, executeAutogradBackward)
- Issue #140 (embedding scale removed), #141 (ScratchBlock gradient tap + positional_encoding config + PCGrad NaN guard), #142 (gradient tap hardening)
- Issue #139 (per-component gradient clipping: emb/enc/num independent budgets)
- Issue #137 (numeric loss gradient normalization fixes)
- Issue #138 (computeGradNorm timing decomposition)
- Issues #85 (validation token budget), #115 (diagnostic buffer mismatch)

**Next Step**: Continue systematic audit through remaining forward pass (sections 2.x) and backward pass (sections 4.3+) components
