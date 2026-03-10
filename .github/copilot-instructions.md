# GRIM-text Development Guide

> **Current Focus:** GRIM-text model training and inference. The broader G.R.I.M assistant project exists but is not the active development target.

---

## 🔴 CRITICAL: FORBIDDEN CODE PATTERNS (Rule 20 - Fail Loud)

**NEVER generate these patterns. If you see them, DELETE THEM:**

❌ `x ? x : fallback` - NO fallbacks. NO Stubs. Require x, crash if null  
❌ `if (ptr) { use(ptr); }` - Require ptr, crash if null with clear error  
❌ `if (args.stream) { ... } else { use_config_stream(); }` - NO silent fallbacks  
❌ `try { } catch { /* ignore */ }` - NEVER swallow errors silently  
❌ `if (version == old) { legacy_path(); } else { new_path(); }` - DELETE legacy paths  
❌ `// TODO: remove after migration` - Remove NOW or never commit  
❌ `__attribute__((deprecated))` - DELETE deprecated code entirely  
❌ Any comment containing "backwards compatibility" - Forbidden concept  
❌ `if (!initialized) { std::cerr << "WARNING: ..."; return; }` - Throw exception instead

**REQUIRED patterns:**

✅ `if (!ptr) throw std::runtime_error("X is NULL at " + std::string(__FILE__) + ":" + std::to_string(__LINE__));`  
✅ `if (!args.stream) throw std::runtime_error("stream is NULL - caller MUST provide valid stream");`  
✅ `assert(condition && "description");` for invariants  
✅ Crash with detailed error message on ANY unexpected state  
✅ Delete code instead of deprecating it

**Pre-commit checklist - search your code for:**

- `? :` (ternary with fallback) → rewrite to throw
- `if (args.` (optional parameter check) → require parameter
- `else {` (compatibility branch) → delete the else clause
- `catch` (exception handler) → ensure it re-throws or exits, never swallows

**Why:** Silent failures waste weeks debugging. If code is wrong, crash immediately with clear error. Backwards compatibility hides bugs and creates maintenance debt.

**Severity Hierarchy:** Architectural problems (wrong data flow, dead code paths, disconnected subsystems, silent fallbacks at system boundaries) are the **MOST SEVERE** class of bug. A single architectural misconnect (e.g., loss not connected to gradients, embeddings not scaled, weights tied but gradients canceling) can waste weeks of GPU compute producing meaningless training runs. Always prioritize architectural correctness over local code quality. Rule 26 (YAGNI/delete dead code) is the architectural arm of Rule 20 — dead code creates false confidence that a subsystem is functional when it isn't.

---

## 🟡 MANDATORY: Equation-Based Diagnostic Logging (Rule 21)

**When adding diagnostic logging for AI/ML math operations, ALWAYS use the `[*_EQUATION]` format:**

This format is non-negotiable for training/inference debugging because **you cannot argue with hard mathematical facts**.

**Required Format Structure:**

```
[OPERATION_EQUATION] operation_name: mathematical_formula
  INPUT (description): shape=[dims] min=X max=Y rms=Z
  ACTUAL result: shape=[dims] min=X max=Y rms=Z
```

**Optional additions (use only when applicable):**
```
  PARAMETERS: param1=value1, param2=value2  // Only if operation has configurable params
  EXPECTED result = formula                  // Only if result is predictable from inputs
                  = substituted_values
                  = computed_value
  [ANOMALY] description                      // Only if something is wrong
```

**Statistics explanation:**
- `min/max` = minimum/maximum value in tensor
- `rms` = Root Mean Square = sqrt(mean(x²)) = typical size/scale of values in the tensor
- `shape` = tensor dimensions [batch, seq_len, features] etc.

**Examples of Good Logging Tags:**

- `[GRAD_A_EQUATION]` - Matrix A weight gradient computation
- `[ATTN_SCORE_EQUATION]` - Attention score computation (Q @ K^T / sqrt(d))
- `[LSE_EQUATION]` - Log-Sum-Exp computation
- `[RMSNORM_EQUATION]` - RMSNorm computation
- `[LOSS_EQUATION]` - Loss computation (cross-entropy, focal, etc.)

**Mandatory Elements:**

1. **Mathematical equation** - The EXACT formula being computed
2. **Input shapes** - Tensor dimensions for dimensional analysis
3. **Input statistics** - min, max, rms (and optionally mean, std)
4. **Expected value** - What the result SHOULD be (only when predictable)
5. **Actual value** - What was actually computed
6. **Anomaly detection** - Flag when actual >> expected or contains NaN/Inf

**When to Add Equation Logging:**

- ✅ ANY new forward/backward kernel implementation
- ✅ When debugging gradient explosion/vanishing
- ✅ When investigating loss anomalies
- ✅ When verifying weight initialization
- ✅ Any GEMM operation (grad_A = C^T @ B, grad_B = A^T @ C, etc.)

---

## 🔴 ACTIVE BUG INVESTIGATION

**READ FIRST:**

1. [docs/LOG_FILE_CONVENTION.md](../docs/LOG_FILE_CONVENTION.md) - **CRITICAL: Always verify which log file before making claims**
2. [docs/PLATEAU_BUG_INVESTIGATION.md](../docs/PLATEAU_BUG_INVESTIGATION.md) - Check "CURRENT RUN TRACKING" section first

Training plateau bug under investigation. The docs above track what has been verified correct, known issues found and fixed, diagnostic data, and next steps. Check here first to avoid re-investigating ruled-out causes.

---

## Project Overview

GRIM-text is a custom transformer model with:
- Flash Attention v2, cuBLAS, custom fused CUDA kernels
- **ScratchBlock Reasoning Layer** - Structured reasoning with atom detection (numbers, URLs, emails, paths, dates, code literals)
- **Unigram + Byte Fallback Tokenizer** - High-quality subword segmentation with 100% UTF-8 coverage
  - Token layout: [0-255] = bytes, [256-511] = atom placeholders, [512+] = unigram vocab
- **Grouped Query Attention (GQA)** - num_heads=12, num_kv_heads=4
- **Unified Loss System** - Focal loss + label smoothing + entropy reg in single autograd kernel
- **TelemetryLattice** - Hierarchical streaming statistics (8 levels, 5 metric streams)
- Config driven via `ai_config.json`
- Runs as HTTP server (`grim_text_server.exe`) on port 11435
- Training executable: `train_gpu.exe` (three-phase architecture)

**GRIM-text is a SEPARATE build from the main GRIM program.** GRIM-text MUST NOT include headers from `../../../../core/` or any G.R.I.M main-program libraries.

---

## Build & Training

**Build GRIM-text:**

```powershell
cd resources/models/GRIM-text/training/TrainingLoop
cmake --build build --config Release --target train_gpu
cmake --build build --config Release --target grim_text_server
```

**Run training:**

```powershell
cd resources/models/GRIM-text/training
.\TrainingLoop\build\Release\train_gpu.exe
```

**Tokenizer self-test (37 tests):**

```powershell
cd resources/models/GRIM-text/training/build
cmake --build . --config Release --target unigrambyte_self_test
.\Release\unigrambyte_self_test.exe
```

**CMake cache note:** When removing `.cu` files from CMakeLists.txt, clean the cache to remove stale device-link objects:
```powershell
Remove-Item -Recurse -Force build\CMakeFiles\grim_training_kernels.dir
# or: cmake --build build --config Release --clean-first
```

---

## Three-Phase Training Architecture

Entry point is `train_gpu.cu` → `executePhase1()` → `executePhase2()` → `executePhase3()`. Data flows via `TrainingContext` struct (no globals).

- **Phase 1: Startup** ([Phase1_Startup.cu](resources/models/GRIM-text/training/Phases/Phase1_Startup.cu)) - config, tokenizer, data loading, model init, optimizer setup
- **Phase 2: Training Loop** ([Phase2_TrainingLoop.cu](resources/models/GRIM-text/training/Phases/Phase2_TrainingLoop.cu)) - batching, forward/backward, gradient clipping, validation, checkpointing
- **Phase 3: Cleanup** ([Phase3_Cleanup.cu](resources/models/GRIM-text/training/Phases/Phase3_Cleanup.cu)) - final checkpoint, training summary, resource cleanup

If modifying training logic, edit the appropriate phase file, not `train_gpu.cu`.

---

## Architecture Details

### Unified Loss System

Use `autograd::unified_loss()` in `AutogradLoss.cu` — this is the ONLY loss path.
- Formula: `L = α(1-p_t)^γ * CE_smooth + λ * H(p)`
- `cross_entropy_loss()` is a wrapper that calls `unified_loss()` with plain CE config
- Config: `ai_config.json` → `training.config.loss`
- **DELETED**: `UnifiedLoss_GPU.cu`, `ComputeLoss_GPU.cu` — old path was disconnected from autograd gradients

### Centralized Controller Pattern — MANDATORY

All GPU resource management MUST go through `TrainingState`. VIOLATIONS ARE BUGS:
- **CUDA Streams**: `training_state.stream_ctrl.getPrimaryStream()` — NEVER create raw `cudaStream_t` locals
- **cuBLAS**: `training_state.cublas_handle` — NEVER create separate handles
- **Gradient Buffers**: via autograd system (`ctx.model->zeroGradients()`, `ctx.model->backward()`)
- **Optimizer States**: `training_state.optimizer_m_states/optimizer_v_states` — ParameterGroup holds pointers only
- Structs store pointers only. TrainingState owns allocations.

### Tokenization: Unigram + Byte Fallback

- **Files**: `resources/models/GRIM-text/Shared/UnigramByte/`
- **Key Implementation**: GrimTokenizer.hpp (alias to UniByte), AhoCorasick.cu (pattern matching), Unigram.cu (vocab + Viterbi)
- Trie must be built before encoding — constructor auto-builds with special tokens
- Aho-Corasick DFA built during DetectorState construction, not lazily

### Grouped Query Attention (GQA)

- Config: num_heads=12, num_kv_heads=4, heads_per_kv_group=3
- W_qkv shape: `[(num_heads + 2*num_kv_heads) * head_dim, d_model]` = `[1280, 768]`
- Backward kernel MUST apply `gqa_grad_scale = 1.0f / heads_per_kv_group` to dV/dK
- MHA and GQA checkpoints are incompatible — serialization validates and throws on mismatch
- Old GQA checkpoints (num_kv_heads=0) cannot load into current model; must retrain

### Tied Embedding / LM Head Weight Gradients

When `tie_embeddings=true`, LM head backward and embedding backward both write to the SAME gradient buffer via PyTorch-style direct accumulation (same approach as GPT-2/LLaMA):
- `g_final = g_lm + g_emb` (accumulated into shared buffer)
- LM head: dense GEMM (`grad_W = centered^T @ grad_logits`)
- Embedding: sparse scatter-add (`grad_W[tok] += grad_encoder[t]`)
- Weight tying aliasing: `embedding_grads` and `lm_head_weight_grads` are the SAME pointer. NEVER zero both separately, add both to param groups, or free both.

### Per-Component Gradient Clipping

Three independent clips (Issue #139):
1. **emb clip** — LM_HEAD (+ EMBEDDING if untied)
2. **enc clip** — ATTENTION + FFN + RMSNORM + SCRATCHBLOCK
3. **num clip** — NUMERIC_HEAD

NEVER clip components jointly when one dominates L2 norm — it crushes the smaller ones.

### ALiBi / RoPE Position Encoding

- Position info injected INSIDE attention, NOT in residual stream. No position embeddings added to token embeddings.
- **ALiBi**: Slopes capped via `ALIBI_MAX_BIAS = -10.0f` in `HyperParameters_GPU.hpp`. Ensures `exp(-10) ≈ 0.000045` (computable) not `exp(-256) ≈ 0` (underflow → gradient explosion).
- **FlashAttention expects NEGATIVE slopes** (library uses `+= slope * col_idx`)
- **Always match `max_seq_len` to actual context length** — mismatched slopes cause weak attention at distance
- **RoPE NTK scaling**: `effective_theta = theta * (max_seq_len / 2048)^(rotary_dim / (rotary_dim - 2))` when max_seq_len > 2048

---

## Known Issues / Pitfalls

### C++ / CUDA

- **C++ Vector Invalidation**: NEVER hold references (`auto& node`) across vector mutations (`emplace_back`, `push_back`) — reallocation invalidates all references.
- **C++ Array Initialization**: `int arr[256] = {-1}` only sets first element to -1, rest are 0. Use `std::fill()`.
- **Autograd Return Statements**: ALWAYS explicitly `return output;` from autograd forward functions. Missing return causes the grad_fn chain to be destroyed during forward pass → illegal memory access in backward.
- **Atomic Kernel Ordering**: When kernel B reads data written by kernel A via `atomicAdd`, you MUST `cudaStreamSynchronize` between them even on the same stream.
- **GPU Gradient Norm Sync**: `cudaStreamSynchronize` inside `computeGradNorm` drains the entire backward pipeline. Pass `sync_for_host_read=false` for 9 out of 10 batches; only sync when logging gradient components.

### Config / Initialization

- **Hardcoded Defaults Violate Rule 20**: Config struct fields for algorithmic parameters MUST default to `0`. Throw if still 0 after loading. Example: `int max_seq_len = 0;` not `= 512`.
- **Validation Token Budget**: Validation MUST use `ctx.model->getConfig().max_tokens_per_batch`, not hardcoded constants — buffer overflow crash otherwise.
- **`positional_encoding` config**: Parsed from JSON in `loadConfiguration()`. Do not hardcode `DEFAULT_POSITIONAL_ENCODING`.

### Training Data

- **HTML artifacts** corrupt tokenizer vocab — always strip tags before training. `DataLoader.cu` handles this automatically via `stripHtmlTags()` / `decodeHtmlEntities()` / `normalizeWhitespace()`.
- **AtomTable Token IDs** include `ATOM_TOKEN_BASE` (256) offset. When accessing `entries_[]` array: `uint32_t idx = id - ATOM_TOKEN_BASE`.
- **Sliding Window Overlap**: `overlap_len = raw_overlap - 1` (reduces by 1 when raw_overlap > 0) to avoid masking the same boundary target in two consecutive windows.

### Architecture

- **Embedding scale = 1.0**: Do NOT scale embeddings by `sqrt(d_model)`. GRIM-text uses ALiBi/RoPE inside attention — the AIAYN scaling has no purpose and creates a 27.7x gradient asymmetry with tied weights.
- **Sandwich Norm removed** (Issue #148): Architecture is standard pre-norm (`output = input + LayerScale(sublayer_output)`). Post-residual RMSNorm constrained hidden norms to a hypersphere → mode collapse.
- **LayerScale init_value = 1.0** in `ai_config.json`. Value 0.1 causes catastrophic gradient vanishing through encoder layers.
- **`per_token_grad_scale=true` is REQUIRED**: Gradient RMS ~1e-6 with ~3000 tokens is CORRECT. Disabling causes 3000x effective LR explosion.
- **ScratchBlock backward**: Uses `grad_output_tap` on `DropoutGradFn` — set tap before `loss_tensor.backward()`, then pass captured gradient to ScratchBlock backward. Do NOT check `has_grad()` on dropout outputs (always false for non-leaf tensors).
- **FFN Post-GELU Cache**: `EncodingLayer::forward()` MUST write post-GELU activations to `args.cache_ffn_output` via `cudaMemcpyAsync` after `ffn_->forward()`. Forgetting this fills the cache with garbage → corrupted W2 gradients.
- **ScratchBlock Buffer Desync**: After `autograd::add(emb, pos_emb)`, copy `ts->cached_embeddings` back to `ctx.embedding_tensor.data` after ScratchBlock forward. Layer 0 will receive stale pre-ScratchBlock data otherwise.
- **Encoder Bias Autograd**: Use `autograd::broadcast_add()` for all bias additions (b_qkv, b_o, b1, b2). Raw `launchFFNBiasAdd` bypasses autograd → zero bias gradients.
- **Encoder Activation Centering**: Center cached activations (`cached_ln1_output`, `cached_ffn_input`, etc.) BEFORE weight gradient GEMMs to eliminate systematic gradient bias from non-zero mean.
- **Hidden State Centering**: Apply BOTH column centering (`Σ_t h[t,d] = 0`) AND row centering (`Σ_d h[t,d] = 0`) before LM head. Column alone reduces cosine correlation; row eliminates gradient sign flips from non-zero row sums.

### FlashAttention

- **Missing Preprocessing Kernel** (Issue #84): `flash_bwd_dot_do_o_kernel` MUST be launched BEFORE `flash_bwd_dq_dk_dv_loop_kernel`. Without it, `dsoftmax_sum` buffer contains garbage for most m_blocks → dQ/dK explosion.
- **GQA Buffer Sizing** (Issue #72): Allocate `dk_bf16`/`dv_bf16` for `num_heads` (12), NOT `num_kv_heads` (4). The Dao-AILab backward kernel writes using query head index. Both allocation sites in `TensorContract_GPU.cu` MUST use `num_heads`.
- **GQA Reduction Scaling** (Issue #73): Apply `gqa_grad_scale = 1.0f / heads_per_kv_group` IN the reduction kernel. The external FlashAttention library does NOT apply this internally.

### Diagnostic Pitfalls

- **RMSNorm diagnostic formula**: `expected_output_rms = input_rms * gamma_rms / sqrt(input_rms² + eps)` — not just `gamma_rms`. With small Xavier-init embeddings (rms≈0.006), epsilon contributes ~20%.
- **Xavier Init LCG**: Uses splitmix64-style per-element seed + 16 iterations. Single-iteration LCG produces correlated outputs (avg|cos| ≈ 0.37 instead of expected 0.036).
- **Diagnostic buffer selection**: Read `centering_scratch_tensor` (post-centering), not `cached_encoder_output` (pre-centering), in diagnostic functions.
- **Wall-time vs GPU-time**: `cudaStreamSynchronize` timing includes draining prior pipeline work. Use CUDA events (`cudaEventRecord`/`cudaEventElapsedTime`) to isolate actual kernel time.
- **Mean reduction double-application**: Loss backward already scales by `1/N`. Do NOT apply additional `1/tokens` scaling in parameter gradient kernels (RMSNorm gamma, etc.).
- **LibTorch gradient comparisons**: Only valid when baseline uses IDENTICAL config (d_model, num_layers, num_heads, batch_tokens). Different configs produce inherently different gradient magnitudes.

### Deleted Code — Do Not Recreate

- `UnifiedLoss_GPU.cu`, `ComputeLoss_GPU.cu` — replaced by `AutogradLoss.cu`
- `Embedding_GPU.cu` kernels/launchers/EmbeddingLayer class — dead code, only `destroyEmbeddingRuntime()` remains
- `ScaleGradFn` / `autograd::scale()` — deleted (embedding scale removed)
- Value extraction head (`value_extraction_weight_`, `value_extraction_bias_`) — deleted (Issue #142). ScratchBlock is a reasoning layer, not scalar regression.
- `rms_post_attn_gamma_`, `rms_post_ffn_gamma_` — sandwich norm deleted (Issue #148)
- GPU delegate system (`Shared/Delegate/Delegate.hpp`) — zero registered callbacks, deleted per Rule 26

---

## Key Files

| File | Purpose |
|------|---------|
| `resources/models/GRIM-text/training/train_gpu.cu` | Training entry point (orchestrator) |
| `resources/models/GRIM-text/training/Phases/Phase1_Startup.cu` | Config, model init, data loading |
| `resources/models/GRIM-text/training/Phases/Phase2_TrainingLoop.cu` | Training loop, diagnostics |
| `resources/models/GRIM-text/training/Phases/Phase3_Cleanup.cu` | Checkpoint, cleanup |
| `resources/models/GRIM-text/Shared/TensorContract_GPU.cu` | Autograd system, all GradFn structs |
| `resources/models/GRIM-text/Shared/Loss/ComputeLoss/AutogradLoss.cu` | Unified loss |
| `resources/models/GRIM-text/Layers/Flash_Attention_Kernal.cu` | FlashAttention forward/backward |
| `resources/models/GRIM-text/Layers/Encoding_GPU.cu` | Transformer encoder layer |
| `resources/models/GRIM-text/Layers/Feed_Forward_GPU.cu` | FFN layer |
| `resources/models/GRIM-text/Training/AutogradTraining.cu` | Forward/backward orchestration |
| `resources/models/GRIM-text/Training/TrainingState_GPU.hpp` | GPU resource ownership |
| `resources/models/GRIM-text/Shared/UnigramByte/` | Tokenizer |
| `resources/models/GRIM-text/training/schemas/grim_transformer_model.fbs` | FlatBuffer schema |
| `ai_config.json` | All model/training configuration |
| `docs/PLATEAU_BUG_INVESTIGATION.md` | Active training investigation |
