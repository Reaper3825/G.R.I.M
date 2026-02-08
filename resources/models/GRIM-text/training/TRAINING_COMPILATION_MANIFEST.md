# GRIM-Text Training Stale Code Investigation Checklist

**Chronological execution order from Phase 1 Startup → Phase 2 Training Loop → Phase 3 Cleanup**

Use this checklist to systematically audit each file in the order it's used during training focusing on a systematic review the primary goal is finding either mismtaches: things that throw away progress from training or stale usages. when starting a new file for investigation you should ask yourself what in this file is redudent, unused, used only for backwards compatibiliity, or is not implemented properly do not start going through multiple files at a time we have done that time and time again so we are going to do this one right and through each one by one in order this is to not only help identify mode collapse/loss plateau but also to make up for maintenance debt top to bottom when finished with each file provide an indepth rundown of findings suspicouns and then wait for instruction before making changes.

---

## PHASE 1: STARTUP & INITIALIZATION

### 1.1 Orchestrator Entry Point

- [x] **train_gpu.cu** (main entry) ✅ AUDITED
  - Calls Phase1 → Phase2 → Phase3 sequentially
  - Removed unused includes (`<iostream>`, `<cuda_runtime.h>`, `<exception>`), unused `EmitModuleWarning` import
  - Fixed `fprintf(stderr)` bypass → now uses `EmitModuleInfo` for consistent logging
  - Fixed stale `train_gpu_orchestrator.cu` reference in copilot-instructions.md
  - Pattern to check: `throw std::runtime_error()` on config load errors

---

### 1.1a Logging Infrastructure (grim_log_recorder library)

- [x] **Shared/LogRecorder/LogRecorder.cu** ✅ AUDITED
  - Centralized logging system used throughout training
  - Records diagnostics, equation logging, and error messages
  - Part of grim_log_recorder static library (shared by all targets)
  - **FIXED**: Removed hardcoded fallback path `kDefaultLogsPath` (Rule 20). InitLogRecorder now throws if rootPath is empty.
  - **FIXED**: Phase1_Startup.cu now passes `config.paths.log_dir` to InitLogRecorder().
  - **DELETED**: All dead device-side logging code (RecordLayerLog, RecordLayerLogSimple, DeviceBufferLogger, device kernels, DeviceLogBuffer/DeviceLogDelegate types, GetDeviceLogBuffer, InstallDefaultDeviceLogger, RegisterDeviceLogCallback, ClearDeviceLogCallbacks) — zero callers in any kernel.
  - **DELETED**: LogMirrorScope class — zero callers.
  - **DELETED**: ParseModuleOverrideSpec — zero external callers.
  - **DELETED**: RegisterDefaultLoggingProfiles (LogRecorder version) — never called; Phase1_Startup has its own copy that IS called.
  - **DELETED**: ParseModuleLogLevelString (LogRecorder version with fallback param) — Phase1 has own copy without fallback that IS called.
  - **DELETED**: GRIM::InitLogRecorder() no-arg wrapper — zero callers.
  - **KEPT**: FlushDeviceLogs/ResetDeviceLogs as no-ops (callers exist in Phase2/Phase3/Tests).
  - **KEPT**: RecordLayerLogHost (1 caller in Phase2_TrainingLoop.cu).
  - File reduced from 1015 → ~676 lines.

---

### 1.2 Configuration Loading & Validation

- [x] **Phase1_Startup.cu / Phase1_Startup.hpp** ✅ AUDITED
  - Configuration loading from ai_config.json
  - Path validation (model, data, checkpoint dirs)
  - **Issue #106**: Reseeding W_qkv with 1/sqrt(d_model) scaling - ✅ Applied (xavier_seed passed to initializeAutogradTensors)
  - **Issue #107**: Fixed LCG PRNG correlation with splitmix64 - ✅ Applied (Tensor::xavier_uniform_() with Philox PRNG)
  - **Issue #110**: PCGrad buffer allocation for tied embeddings - ✅ Applied (allocatePCGradBuffer + g_skip_embedding_backward=false)
  - **Rule 20**: max_seq_len defaults changed from 512 → 0 - ✅ Verified (defaults=0, throws if still 0)
  - **FIXED**: Deleted `DEBUG_BATCH_PREP_CORRUPTION` define + 7 dead `#if` blocks (resolved investigation, Rule 20)
  - **FIXED**: `catch(...)` in checkpoint scanning now catches `std::exception&` and logs filename + message
  - **FIXED**: CUDA RNG failure now throws instead of silently degrading (Rule 20: training with uncontrolled RNG = undefined)
  - **FIXED**: Stability override validation - throws if `batch_size <= 0` when stability enabled (Rule 20)
  - **FIXED**: `actual_batch_size` ternary simplified - `batch_size` already overridden by loadConfiguration, no redundant access
  - **NOTE**: `debug_gradient_attribution` block exists but hardcoded `false` (production disabled) - kept for Issue #60 debugging
  - **NOTE**: Double ai_config.json snapshot load (once for hyperparams, once for tokenizer_config) - minor, not a training bug
  - File reduced from ~1828 → ~1760 lines.

---

### 1.3 Training Data Loading

- [x] **Shared/DataLoader/DataLoader.cu + training_data_loader.hpp** ✅ AUDITED
  - PrepareTrainingDataFromCache: Reads merged_verified_cache.jsonl → tokenizes → writes single GRMT file
  - training_data_loader.hpp: GRMTDataLoader reads .grmt binary format for Phase1_Startup
  - **FIXED**: `catch(...)` now `catch(const std::exception&)` + counts/logs malformed JSONL lines
  - **FIXED**: Hardcoded `character_coverage = 0.9995f` → uses `GRIM::HyperParameters::TOKENIZER_CHARACTER_COVERAGE`
  - **FIXED**: Deleted dead `loadBinaryFormat()` from training_data_loader.hpp (~35 lines, never called, Rule 20)
  - **FIXED**: Deleted dead `stripBosEosMarkers()` — `stripHtmlTags(<[^>]+>)` already strips `<s>` and `</s>` 
  - **FIXED**: Removed wrong `targets[0] = -1` unconditional masking — BOS→first_token is valid training signal
  - **FIXED (SPAGHETTI)**: Removed 80/10/10 split + chunk_size splitting + validation_data.grmt/test_data.grmt output. DataLoader now writes ALL sequences to single GRMT. Phase1_Startup owns train/val splitting and sliding windows — clear ownership, no data waste.
  - **FIXED**: Unsupported vocab.bin version now throws (was printing and continuing silently, Rule 20)
  - HTML cleaning pipeline correct: static regex, entity decoding ✅
  - GRMT v6 format with byte_lengths, vocab validation, non-finite sanitization ✅
  - `totalVocabSize()` correctly used for header (includes byte+atom ranges) ✅

---

### 1.4 Tokenizer Initialization (UnigramByte Library)

- [x] **Shared/UnigramByte/Byte.cu** ✅ AUDITED
  - Byte fallback tokenizer (raw UTF-8 bytes 0x00-0xFF)
  - Provides 100% coverage for unknown characters/emojis
  - Pattern to check: Verify byte token IDs are [0-255]
  - Pattern to check: No allocations during tokenization (zero-copy std::string_view)

- [x] **Shared/UnigramByte/Unigram.cu** ✅ AUDITED
  - Unigram Language Model tokenizer (statistical subword segmentation)
  - Viterbi decoding for optimal segmentation
  - **CRITICAL**: Trie-based prefix matching for fast encoding
  - Pattern to check: Verify `buildTrie()` called in constructor (NOT lazy)
  - Pattern to check: Search for `addPiece()` with explicit token_id (Rule 20: no auto-ID fallback)
  - Pattern to check: Verify Viterbi kernel runs SEQUENTIALLY (O(n) dependency, NOT parallelized across positions)
  - **STALE CODE CHECK**: Auto-ID `addPiece()` overload confirmed DELETED (line 463 comment)

- [x] **Shared/UnigramByte/UniByte.cu** ✅ AUDITED
  - Combined Unigram + Byte fallback (GrimTokenizer alias)
  - Token layout: [0-255] = bytes, [256-511] = atoms, [512+] = unigram vocab
  - Pattern to check: Verify ATOM_TOKEN_BASE = 256 offset applied
  - Encoding: detectStructures() → segment → Unigram encode per segment → Byte fallback internal to Unigram

- [x] **Shared/UnigramByte/AtomTable.cu** ✅ AUDITED
  - Atom token management (numbers, URLs, emails, paths, dates, code)
  - Dedicated embeddings for structural elements
  - Pattern to check: Verify `entries_[]` array access subtracts ATOM_TOKEN_BASE
  - No `= {-1}` patterns found. No Rule 20 violations.

- [x] **Shared/UnigramByte/AhoCorasick.cu** ✅ AUDITED
  - O(n) multi-pattern matching for structural token detection
  - 50-100x faster than std::regex for URL/email/number prefixes
  - Detects: http://, https://, www., ftp://, ws://, wss://, file://, @, 0x, 0b
  - DFA built eagerly in DetectorState constructor with build() → BFS failure links ✅
  - Detection confirmed BEFORE Viterbi encoding (detectStructures → encodeInternal) ✅

---

### 1.5 Model Weight Initialization

- [x] **Shared/TrainingState/TrainingTensors.cu** ✅ AUDITED & FIXED
  - Weight allocation and Xavier initialization
  - **Issue #107**: Philox PRNG (cuRAND) now used — supersedes old LCG and Issue #106 scaling
  - **FIX**: Numeric head weights now Xavier-initialized (was all-zeros with dead TODO)
  - **FIX**: `zeroGrad()` now zeros LayerScale grads (was missing layer_scale1/2)
  - **FIX**: `position_embedding_weights.ensure_grad()` and `.zero_grad()` guarded with null check
  - **FIX**: Dead `launchXavierInit` extern "C" deleted from LanguageModel_Training.cu
  - Weight tying verified: `from_ptr()` + `share_grad()` + `owns_data=false` ✅
  - GQA dims correct: `total_qkv_dim = d_model + 2*kv_dim` ✅
  - FFN shapes correct per Issue #89: W1=[d_model, d_ff], W2=[d_ff, d_model] ✅

---

### 1.6 Embedding Initialization

- [ ] **Shared/TensorContract/TensorContract_GPU.cu** (autograd::embedding)
  - Production embedding forward/backward (replaces Embedding_GPU.cu)
  - **Issue #92**: Embedding scale √d_model applied - VERIFY in forward pass
  - **Issue #92**: Backward gradient scaling includes 1/sqrt(d_model) - VERIFY in backward
  - Pattern to check: Search for `embedding_scale = sqrt(d_model)` parameter
  - Pattern to check: Verify Constructor does NOT call buildTrie() lazily (must be in Phase1)
  - **STALE CODE CHECK**: Verify `Layers/Embedding/Embedding_GPU.cu` is NOT called (dead code per Rule 20)

---

### 1.7 Stream & Resource Initialization

- [ ] **Shared/StreamController/StreamController_GPU.cu**
  - CUDA stream creation and synchronization
  - **Rule 22**: All streams created via TrainingState controller, never raw `cudaStream_t`
  - Pattern to check: Search for any standalone `cudaStreamCreate()` calls (should be in TrainingState)
  - No known issues in this module itself

---

### 1.8 Training State GPU Setup

- [ ] **Shared/TrainingState/TrainingStateGPU.cu**
  - Allocates GPU buffers, cuBLAS handle, CUDA streams
  - Allocates: gradient buffers, optimizer state (m, v), intermediate tensors
  - **Rule 22**: All GPU resources managed centrally - VERIFY no stragglers in other modules
  - Pattern to check: Verify `allocatePCGradBuffer()` called when tie_embeddings=true
  - Pattern to check: Search for `cudaMalloc()` calls - should NOT be in layer/kernel modules

- [ ] **Shared/GPUBuffer/GPUBuffer.cu**
  - Low-level GPU buffer memory management utilities
  - Wrapper for cudaMalloc/cudaFree with error checking
  - Pattern to check: Verify all allocations track ownership
  - Pattern to check: Verify no memory leaks (all allocations freed)
  - **Rule 22**: Should only be called from TrainingState, not directly by layers

---

### 1.9 Positional Bias Initialization

- [ ] **Shared/PBM/PositionalBiasMethod.cu**
  - Initializes ALiBi slopes and RoPE frequencies
  - **Issue #47**: ALiBi slopes scale relative to max_seq_len - VERIFY implementation
  - **Issue #70**: RoPE NTK scaling for context > 2048 - VERIFY implemented
  - **Issue #78**: ALiBi max bias clamping (ALIBI_MAX_BIAS = -10.0) - VERIFY present
  - Pattern to check: Search for `m_max = target_bias / d_min` and slope interpolation
  - Pattern to check: Search for `ALIBI_MAX_BIAS = -10.0f` constant definition
  - Pattern to check: Verify slopes are NEGATIVE (library uses `+= slope * col_idx`)

---

### 1.9a Tensor Conversion Utilities

- [ ] **Shared/TensorConversion/TensorConversion.cu**
  - Tensor format conversion utilities (e.g., FP32 ↔ FP16, host ↔ device)
  - Used for moving tensors between CPU/GPU or precision conversions
  - Pattern to check: Verify proper synchronization after async copies
  - Pattern to check: Check for memory leaks in temporary conversion buffers
  - **STALE CODE CHECK**: Verify actually used in production, or mark as dead code

---

### 1.10 Training State Initialization

- [ ] **InitTrainingState.cu**
  - Detailed training state initialization
  - Coordinates allocation of all GPU resources
  - Pattern to check: Verify calls to TrainingState allocation methods
  - Pattern to check: Verify proper initialization order (streams → cuBLAS → PBM → tensors)

---

### 1.11 GPU Layer Initialization & Wiring

- [ ] **TrainingOps.cu** (initGPU method)
  - Wires GPU encoder layers to TrainingTensors memory
  - Creates EmbeddingRuntime pointing to TrainingTensors buffers
  - **Rule 20**: TrainingTensors is ONLY initialization path - NO legacy CPU paths
  - **Issue #91**: apply_rms_norm=false (fused embedding norm DISABLED)
  - **Issue #92**: embedding_scale = sqrt(d_model) - VERIFY present
  - Pattern to check: Search for `embedding_runtime->token_buffer = training_state_.tensors_->` - MUST point to TrainingTensors
  - Pattern to check: Verify `owns_token_buffer = false` (TrainingTensors owns memory)
  - Pattern to check: Verify NO CPU embedder fallback paths
  - Pattern to check: Search for `useExternalWeights()` call - wires encoder layers to TrainingTensors
  - **STALE CODE CHECK**: Verify NO allocations in EmbeddingRuntime (just pointers to TrainingTensors)

---

### 1.12 Common Language Model Utilities

- [ ] **Common/grim_language_model_gpu.cu**
  - High-level language model orchestration on GPU
  - Contains model-level methods and coordination logic
  - Pattern to check: Verify proper initialization sequence
  - Pattern to check: Verify error handling is fail-loud (Rule 20)

- [ ] **Common/ScaleBuffer.cu**
  - Buffer scaling utilities (e.g., for embedding/logit scaling)
  - Pattern to check: Verify scales are applied consistently
  - Pattern to check: Check for any hardcoded scale values

---

### 1.13 Optimizer Initialization

- [ ] **Shared/Optimizers/AdamW/AdamW_Kernal_GPU.cu**
  - Sets up AdamW optimizer state (m_states, v_states)
  - No known issues
  - Pattern to check: Verify uses decoupled weight decay, not L2 regularization

---

### 1.14 Stability Overrides & Feature Flags

- [ ] **Shared/Stability/StabilityOverrides.cpp**
  - Stability override flags and feature toggles
  - Runtime stability controls (e.g., disable specific optimizations if causing issues)
  - Pattern to check: Verify flags are loaded from config, not hardcoded
  - Pattern to check: Search for any hotfix flags that should be permanent
  - **STALE CODE CHECK**: Look for temporary stability flags that are no longer needed

---

### 1.15 GRIM-TS Layer Initialization

- [ ] **Layers/GRIMTS/GRIM-TS.cu**
  - GRIM-TS (Training Signal) specific layer implementation
  - Additional training-specific layers or signals
  - Pattern to check: Verify integration with main model forward/backward
  - Pattern to check: Check if this is actually used (might be dead code)

---

### 1.16 Equation-Based Diagnostic Logging

- [ ] **Shared/EquationLogging/EquationLogging.cu**
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

- [ ] **Layers/InitInferenceState/InitInferenceState.cu**
  - Initializes inference-specific GPU state (KV cache, etc.)
  - Used before validation batches or standalone inference
  - Pattern to check: Verify proper cleanup after inference
  - Pattern to check: Check if this duplicates training state init (potential for consolidation)
  - **STALE CODE CHECK**: Verify actually used (may be dead code if validation uses training state)

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

- [ ] **Forward_GPU.cu**
  - Coordinates forward pass through all layers (training mode)
  - Pattern to check: Verify layer call order: Embedding → ScratchBlock → EncodingLayers → LMHead → NumericHead

- [ ] **Inference_GPU.cu**
  - Inference-mode forward pass (no gradient computation)
  - Used during validation/evaluation
  - Pattern to check: Verify dropout DISABLED during inference
  - Pattern to check: Verify no gradient tracking overhead
  - **STALE CODE CHECK**: Verify this path is actually used during validation (or mark as dead code)

---

### 2.1b Batch Composition & Dynamic Sequences

- [ ] **Shared/Batching/Batching_GPU.cu**
  - Dynamic batch composition based on sequence length and token rarity
  - Packing strategies: SIMILARITY_GROUPED, BEST_FIT_DECREASING, RARITY_WEIGHTED_RANDOM
  - Token budget management and overflow handling
  - Pattern to check: Verify batch composition uses content weighting (rare token scoring)
  - Pattern to check: Check for hardcoded batch sizes (should use config)
  - Pattern to check: Verify packing efficiency calculation considers padding

- [ ] **Shared/DynaSeqs/DynaSeq_GPU.cu**
  - Dynamic sequence packing and unpacking
  - Handles variable-length sequences in batches
  - Pattern to check: Verify proper padding and masking for variable lengths

---

### 2.1c Activation Quantization (Int8/FP16)

- [ ] **Layers/Quantization/Quantization_GPU.cu**
  - Activation quantization for inference speedup (Int8 symmetric, FP16)
  - Configurable per-layer: embeddings, encoder outputs, QKV cache, logits
  - Pattern to check: Verify quantization config loaded from `ai_config.json` -> `training.config.activation_quantization`
  - Pattern to check: Verify symmetric quantization formula: `q = round(x / scale)` where `scale = max(abs(x)) / 127`
  - Pattern to check: Check if quantization is actually enabled in config (may be dead code if disabled)
  - **STALE CODE CHECK**: Verify quantization paths are used, or mark as experimental/disabled

---

### 2.2 Embedding + Position Encoding

- [ ] **Shared/TensorContract/TensorContract_GPU.cu** (autograd::embedding, sinusoidal)
  - Token embedding lookup with √d_model scaling
  - **Issue #113**: Sinusoidal position embeddings added (AIAYN style) - VERIFY present
  - **Issue #113**: Sinusoidal MUST be applied with ALiBi/RoPE - VERIFY unconditional
  - Pattern to check: Search for `PE(pos,2i) = sin(pos/10000^(2i/d_model))` implementation
  - Pattern to check: Verify sinusoidal applied even when learning=true for position encoding
  - Pattern to check: Verify shape is `[batch*seq_len, d_model]`

---

### 2.3 ScratchBlock Atom Substitution

- [ ] **Layers/ScratchBlock/ScratchBlock_GPU.cu**
  - In-place token substitution for structural atoms (numbers, URLs, emails, paths, dates)
  - **Issue #90**: Buffer desync after autograd::add() - VERIFY fixed
  - **Issue #90**: After forward(), output MUST be copied back to embedding_tensor.data
  - Pattern to check: Search for `cudaMemcpyAsync after ctx.scratch_block->forward()`
  - Pattern to check: Verify LayerBlock uses AtomTable with token ID offsets (ATOM_TOKEN_BASE = 256)

---

### 2.4 Shared/ScratchBlock (Backup Path)

- [ ] **Shared/ScratchBlock/ScratchBlock_GPU.cu**
  - Alternative ScratchBlock implementation (verify which one is used)
  - Pattern to check: Determine which version is active - should only have ONE ScratchBlock

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

- [ ] **Layers/Attention/QKV_Projector.cu**
  - Projects hidden states to Q, K, V
  - No known issues
  - Pattern to check: Verify output passed to Flash Attention

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
  - Adds `λ * attention_output` to input (LayerScale residual)
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
  - Adds `λ * ffn_output` to input (same LayerScale scalar)
  - Same as 2.5g - verify consistent λ value

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

- [ ] **LanguageModel_Training.cu**
  - Calls `autograd::unified_loss()` for text ce loss
  - **Issue #57**: MUST call `numeric_head_output.backward(&grad_numeric)` if numeric head enabled
  - **Issue #57**: CRITICAL - verify numeric_head backward IS called
  - Pattern to check: Search for `if (ctx.model->hasNumericHead()) { numeric_head_output.backward(...) }`
  - Pattern to check: Verify BOTH `loss_tensor.backward()` and potentially numeric backward called

---

### 3.2 Unified Loss (Text Cross-Entropy)

- [ ] **Shared/Loss/ComputeLoss/AutogradLoss.cu**
  - **ONLY LOSS PATH** - Combines CE + focal + label smoothing + entropy regularization
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

- [ ] **Shared/Loss/ComputeLoss/ComputeLossBatch.cu**
  - Applies unified_loss per batch
  - Pattern to check: Verify uses model config loss settings
  - Pattern to check: Check for mean reduction: `loss / num_valid_tokens`

---

### 3.4 Numeric Loss (Auxiliary Head)

- [ ] **Shared/Loss/NumericLoss/NumericLoss_GPU.cu**
  - Huber loss for numeric predictions
  - **Issue #74**: Missing 1/count mean reduction in gradients - VERIFY scaleNumericGradKernel applied
  - **Issue #74b**: Race condition sync between kernels - VERIFY `cudaStreamSynchronize()` present
  - Pattern to check: Search for `scaleNumericGradKernel` with `1.0f / count` scaling
  - Pattern to check: Search for sync between `numericLossKernel` and `scaleNumericGradKernel`

---

### 3.5 CrossEntropy Legacy Path

- [ ] **Shared/Loss/CrossEntropy/CrossEntropy_GPU.cu**
  - **STALE CODE** - Should NOT be called in production (Rule 20)
  - Verify it's NOT referenced anywhere
  - Pattern to check: Search codebase for `cross_entropy_loss()` calls - if ANY found, that's a bug

---

---

## PHASE 2: TRAINING LOOP (Backward Pass & Gradient Flow)

### 4.1 Backward Orchestrator

- [ ] **AutogradTraining.cu**
  - Main backward pass orchestration
  - Calls `.backward()` on loss tensor
  - **Issue #125**: Column centering applied to encoder output - VERIFY present
  - **Issue #132**: Row centering applied AFTER column centering - VERIFY BOTH applied
  - **Issue #127**: RMSNormGradFn use-after-free with centered tensors - VERIFY fixed
  - Pattern to check: Verify `ctx.encoder_output = autograd::center_rows(autograd::center_columns(...))`
  - Pattern to check: Search for `CenterColumnsGradFn` - should own its buffers and destroy correctly
  - Pattern to check: Verify `centered_encoder_output` is a member of ForwardContext (NOT local variable)

---

### 4.2 GradFn Lifecycle & Buffer Ownership

- [ ] **Shared/TensorContract/TensorContract_GPU.cu** (All GradFn subclasses)
  - Core autograd system
  - **Issue #126**: RMSNormGradFn buffer ownership - VERIFY owns gradient buffer, not borrowed
  - **Issue #127**: CenterColumnsGradFn ownership of input tensors - VERIFY proper lifecycle
  - **Issue #56**: Return statements in forward functions - VERIFY ALL return output, no missing returns
  - **Critical checks**:
    - `RMSNormGradFn::capture_inputs()` - allocates grad buffer with `cudaMalloc()`
    - `RMSNormGradFn::~RMSNormGradFn()` - frees own buffer only
    - `CenterColumnsGradFn::~CenterColumnsGradFn()` - does NOT delete input's GradFn
    - `ScaledDotProductAttentionGradFn::save()` - allocates GQA buffers correctly
  - Pattern to check: Search for `bool owns_X` flags tracking ownership
  - Pattern to check: Verify ALL GradFn subclasses set `data_cached = true` (Issue #48 stale caching fix)

---

### 4.3 Attention Backward (Flash Attention)

- [ ] **Layers/FlashAttention/Flash_Attention_Kernal.cu**
  - Flash Attention v2 backward pass
  - **Issue #84**: Preprocessing kernel (flash_bwd_dot_do_o_kernel) MUST be called FIRST - VERIFY present
  - **Issue #72 + #73**: GQA gradient reduction and scaling - VERIFY `gqa_grad_scale = 1.0 / heads_per_kv_group`
  - **Issue #87**: Removed dQ/dK normalization that was crushing gradients - VERIFY NOT present
  - Pattern to check: Search for `flash_bwd_dot_do_o_kernel<<<` to verify preprocessing kernel call
  - Pattern to check: Search for dK/dV reduction with `gqa_grad_scale` multiplication
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
  - Residual addition backward: `grad_input += λ * grad_residual`
  - Pattern to check: Verify scales gradient by layer scalar

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

- [ ] **Layers/ScratchBlock/ScratchBlock_GPU.cu**
  - Backward through atom token substitution
  - Pattern to check: Verify backward propagates gradients to original token embeddings

---

---

## PHASE 2: TRAINING LOOP (Optimization Step)

### 5.1 Gradient Accumulation & Clipping

- [ ] **Shared/TNC/Token-normalized_clipping.cu**
  - Token-normalized gradient clipping
  - Pattern to check: Verify divides by token normalization, not just global norm
  - Pattern to check: Verify adaptive clipping enabled

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
  - Pattern to check: Verify uses correct learning rate (dynamically adjusted if enabled)

---

### 5.4 Dynamic Learning Rate

- [ ] **Shared/Dynamic_LR/DynamicLR.cu**
  - Adjusts learning rate based on training signals
  - Pattern to check: Verify uses loss/gradient trends (not just epoch number)

---

### 5.5 Soft Restart Mechanism

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

- [ ] **Shared/TrainingState/TrainingStateGPU.cu** + Phase2_TrainingLoop.cu
  - Periodic validation and checkpoint saving
  - **Issue #85**: Validation token budget must respect training buffer size - VERIFY uses config limit
  - Pattern to check: Search for `kDefaultMaxTokensPerBatch` - should use `ctx.model->getConfig().max_tokens_per_batch` instead

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
- [ ] Dead `CrossEntropy_GPU.cu` - confirm NOT referenced  
- [ ] Deprecated `launchFFNBiasAdd()` - confirm NOT in production forward pass
- [ ] Deprecated `launchCenterHiddenStates()` - confirm NOT used (replaced by autograd)
- [ ] Old `UnifiedLoss_GPU.cu` - confirm NOT referenced
- [ ] Old `ComputeLoss_GPU.cu` - confirm NOT referenced
- [ ] Stale `EmbeddingLayer` class - confirm NOT instantiated
- [ ] Hardcoded `kDefaultMaxTokensPerBatch` - confirm uses config value
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
| ScratchBlock_GPU.cu | #90 | Buffer copy after forward | Check |
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
   - Search for `CrossEntropy_GPU` or `cross_entropy_loss()` - should have ZERO matches
   - Search for `launchFFNBiasAdd` in forward paths - should have ZERO matches

5. **Verify fixes**:
   - For each issue listed, search for the fix keyword(s)
   - Example: Issue #92 → search for `embedding_scale` or `sqrt(d_model)`
   - If fix keyword not found, that's your problem!

---

**Last Updated**: February 7, 2026  
**Status**: Checklist ready for systematic investigation  
**Next Step**: Work through checklist section by section, marking boxes as you go
