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

### 🔴 Rule 20 — OWNERSHIP TAXONOMY (Boundary Discipline)

Every piece of state in the training loop MUST belong to **exactly one** of three categories. Mixing categories in a single struct, or letting one category leak across a boundary into another, is a Rule 20 violation of equal severity to silent fallbacks.

**Category 1 — Graph-owned (TRANSIENT)**
Lives only inside the autograd tape window (forward → loss assembly → backward).
Examples: `loss_tensor`, `logits_tensor`, `encoder_layer_outputs`, all `GradFn` saved tensors, dropout grad taps, aux-loss `autograd::add` results.
- ❌ MUST NOT be read by any code that runs after `Tensor::backward()` returns.
- ❌ MUST NOT be stored in a struct that also holds Category 2 or Category 3 state.
- ✅ MUST be cleared by a single RAII owner at the boundary (e.g. `AutogradStepScope`).
- ✅ If a "late consumer" needs information from a Category 1 object, snapshot a **reduced/scalar** form into a Category 2 telemetry struct **before** `clear()` runs. Never extend the lifetime of the tensor.

**Category 2 — Persistent training state (DURABLE)**
Lives across many steps; the contents are meaningful between steps.
Examples: `ParameterGroup.tensor` data, parameter `grad_` buffers (across the accumulation window), `optimizer_m_states` / `optimizer_v_states`, LR scheduler state, `optimizer_state.step`, `current_micro_step`, `global_step`, `best_val_loss`, telemetry lattice, EMA, checkpoint metadata, `BatchDiagnostics` snapshots, `BatchLogTape`.
- ✅ Owned by `TrainingState` or by long-lived loop structs (`TrainingContext.optimizer`, etc.).
- ❌ MUST NOT alias or wrap a Category 1 tensor.
- Parameter grads are Category 2 only because the accumulation window deliberately persists them; intermediate (activation) grads are Category 1.

**Category 3 — Workspace / cache (PERSISTENT BUFFER, STALE CONTENTS)**
A buffer that persists for performance reasons but whose contents are meaningless across the boundary.
Examples: `cached_token_ids_tensor`, `cached_targets_tensor`, `cached_token_numeric_values`, `cached_token_to_slot_map`, `d_read_gate_accum`, deferred-cleanup queue object, `cublas_handle`, `stream_ctrl`.
- ✅ Owned by `TrainingState` directly (NOT by `AutogradIntermediates`).
- ✅ Re-zeroed / re-filled at the start of the next pass; never read across the boundary as if the previous pass's contents were valid.
- ❌ A Category 3 buffer in a Category 1 struct is a violation, even if `clear()` "skips freeing it" — that is an admission the field is in the wrong struct.

**Boundary rules:**

1. **Single boundary owner.** `autograd_intermediates.clear()` MUST be called from exactly one site (an RAII scope around the autograd step). Multiple call sites = ownership smell.
2. **Single teardown owner.** `flushDeferredCleanup()` is owned by `Tensor::backward()`. Additional calls outside it are forbidden unless code between them enqueues new deferred work (it does not, today).
3. **Tape sealing is explicit.** Once `loss_tensor` is read out as a host scalar, no further `autograd::add` / tape mutation is permitted on it. Functions that finish loss assembly and functions that read the loss scalar must be distinct.
4. **No "exception field" in `clear()`.** If a struct's `clear()` method has to skip a field, that field belongs to a different category — move it.

**Pre-commit checklist for ownership:**

- New field added to `AutogradIntermediates`? → It MUST be Category 1. If it's a buffer that persists, move it to `TrainingState`.
- New `autograd_intermediates.clear()` call site? → Delete it; route through the RAII scope instead.
- New `flushDeferredCleanup()` call outside `Tensor::backward()`? → Delete it.
- New "let me peek at `intermediates.X` after backward" code? → Snapshot it into a Category 2 diagnostics struct **before** the boundary instead.

**Why:** Category mixing is how `logits_tensor` (Category 1, ~tokens × vocab floats of GPU memory) ended up surviving the autograd boundary just because a histogram wanted it, and how `d_read_gate_accum` (Category 3 workspace) ended up with a special-case exemption inside a Category 1 `clear()`. Both are silent contracts that the next refactor will break. Enforcing the taxonomy at the type / file-organization level removes the entire class of bug.

---

## 🟡 Equation-Based Diagnostic Logging (Rule 21)

When adding diagnostic logging for ML math, use the `[*_EQUATION]` format:

```
[OPERATION_EQUATION] name: formula
  INPUT (desc): shape=[...] min=X max=Y rms=Z
  ACTUAL result: shape=[...] min=X max=Y rms=Z
  [ANOMALY] description     // only if wrong
```

Required: equation, input shape + (min/max/rms), actual result. Optional: PARAMETERS, EXPECTED (when predictable), ANOMALY tag. Use for any new forward/backward kernel, GEMM, or when debugging grad explosion/vanishing/loss anomalies.

---

## Reference Docs

- [docs/LOG_FILE_CONVENTION.md](../docs/LOG_FILE_CONVENTION.md) — verify which log file before making claims
- [docs/PLATEAU_BUG_INVESTIGATION.md](../docs/PLATEAU_BUG_INVESTIGATION.md) — active training investigation notes

---

## Project Overview

GRIM-text is a custom transformer:
- Flash Attention v2, cuBLAS, custom fused CUDA kernels
- **GQA**: num_heads=12, num_kv_heads=4
- **Tokenizer**: Unigram + byte fallback. Token layout: [0-255]=bytes, [256-511]=atom placeholders, [512+]=unigram
- **ScratchBlock Reasoning Layer** with atom detection
- **Unified Loss**: focal + label smoothing + entropy reg, single autograd kernel
- **TelemetryLattice**: hierarchical streaming stats
- Config: `ai_config.json`. Server: `grim_text_server.exe` (port 11435). Trainer: `train_gpu.exe` (three-phase).

**GRIM-text is a SEPARATE build.** MUST NOT include headers from `../../../../core/` or any G.R.I.M main-program libraries.

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

### Unified Loss
Use `autograd::unified_loss()` in `AutogradLoss.cu` — the ONLY loss path. Formula: `L = α(1-p_t)^γ * CE_smooth + λ * H(p)`. `cross_entropy_loss()` is a wrapper. Config: `ai_config.json → training.config.loss`.

### Centralized Controller — MANDATORY
All GPU resources go through `TrainingState`. Structs hold pointers only.
- Streams: `training_state.stream_ctrl.getPrimaryStream()` — never raw `cudaStream_t`
- cuBLAS: `training_state.cublas_handle` — never separate handles
- Gradients: `ctx.model->zeroGradients()` / `backward()`
- Optimizer states: `training_state.optimizer_m_states/optimizer_v_states`

### GQA
- `W_qkv` shape: `[(num_heads + 2*num_kv_heads) * head_dim, d_model]` = `[1280, 768]`
- Backward MUST apply `gqa_grad_scale = 1.0f / heads_per_kv_group` to dV/dK
- MHA and GQA checkpoints are incompatible — serialization throws on mismatch

### Tied Embedding / LM Head Gradients
With `tie_embeddings=true`, LM head and embedding backward write to the SAME buffer (PyTorch-style direct accumulation). `embedding_grads` and `lm_head_weight_grads` ARE the same pointer — never zero, register, or free both.

### Per-Component Gradient Clipping
Three independent clips:
1. **emb** — LM_HEAD (+ EMBEDDING if untied)
2. **enc** — ATTENTION + FFN + RMSNORM + SCRATCHBLOCK
3. **num** — NUMERIC_HEAD

Never clip jointly when one component dominates the L2 norm.

### Position Encoding
Position injected INSIDE attention, never in the residual stream.
- **ALiBi**: slopes capped via `ALIBI_MAX_BIAS = -10.0f`. FlashAttention expects NEGATIVE slopes. Match `max_seq_len` to actual context length.
- **RoPE NTK**: `effective_theta = theta * (max_seq_len / 2048)^(rotary_dim / (rotary_dim - 2))` when `max_seq_len > 2048`.

---

## Known Footguns

### C++ / CUDA
- Never hold references across vector mutations (`emplace_back` invalidates).
- `int arr[256] = {-1}` only sets element 0. Use `std::fill()`.
- Always explicitly `return output;` from autograd forward functions — missing return destroys the grad_fn chain.
- When kernel B reads atomicAdd output of kernel A, `cudaStreamSynchronize` between them even on the same stream.
- `computeGradNorm` sync drains the backward pipeline. Pass `sync_for_host_read=false` except when logging.

### Config
- Algorithmic config fields default to `0` and throw if not loaded (Rule 20). No hardcoded defaults like `int max_seq_len = 512;`.
- Validation must use `ctx.model->getConfig().max_tokens_per_batch`, never a hardcoded constant.

### Training Data
- HTML must be stripped pre-tokenization. `DataLoader.cu` handles this.
- `AtomTable` IDs include `ATOM_TOKEN_BASE` (256) offset: `idx = id - ATOM_TOKEN_BASE`.
- Sliding window: `overlap_len = raw_overlap - 1` to avoid duplicate boundary targets.

### Architecture
- **Embedding scale = 1.0** — do NOT scale by `sqrt(d_model)`. ALiBi/RoPE go inside attention; AIAYN scaling creates a 27.7× gradient asymmetry with tied weights.
- **`per_token_grad_scale=true` is REQUIRED** — gradient RMS ~1e-6 with ~3000 tokens is correct. Disabling causes ~3000× LR explosion.
- **LayerScale init_value = 1.0**. Value 0.1 causes catastrophic gradient vanishing.
- **γ_final**: registered as `RMSNORM` with `wd_mult=1.0` AND `lr_mult=0.1`. Without both, γ_final inflates as a logit temperature → mode collapse. Empirically the inflation gradient still wins; set `lm_head_centering.freeze_final_rms_gamma=true` in `ai_config.json` to hold it at 1.0 (LM head W absorbs scale, GPT-2-style).
- **Standard pre-norm only** — sandwich norm was deleted (Issue #148).
- **FFN post-GELU cache**: `EncodingLayer::forward()` MUST `cudaMemcpyAsync` post-GELU activations to `args.cache_ffn_output`, else W2 gradients are corrupted.
- **ScratchBlock buffer desync**: copy `ts->cached_embeddings` back to `ctx.embedding_tensor.data` after ScratchBlock forward, else Layer 0 sees stale data.
- **Encoder bias**: use `autograd::broadcast_add()` for b_qkv, b_o, b1, b2. Raw `launchFFNBiasAdd` bypasses autograd → zero bias gradients.
- **Encoder activation centering**: center cached activations (`cached_ln1_output`, etc.) BEFORE weight gradient GEMMs.
- **Hidden state centering**: column-center h before LM head; row-center the LM head WEIGHT (not h) via `autograd::center_rows(weights_)` inside `LMHeadLayer::forward`. Watch telemetry stream 38 (`rho_raw_rms_spread`): healthy 1.0–1.5×, >2× warn, >4× anomaly.
- **ScratchBlock backward**: set `grad_output_tap` on `DropoutGradFn` before `loss_tensor.backward()`, then pass captured gradient to ScratchBlock backward. Do not check `has_grad()` on dropout outputs.

### FlashAttention
- **dot_do_o preprocessing kernel** MUST run before `dq_dk_dv_loop_kernel`, else `dsoftmax_sum` is garbage → dQ/dK explosion (Issue #84).
- **GQA backward dk/dv buffers**: allocate for `num_heads` (12), not `num_kv_heads` (4). Dao-AILab kernel writes by query head index (Issue #72).
- **GQA reduction**: apply `gqa_grad_scale = 1.0f / heads_per_kv_group` in the reduction kernel (Issue #73).

### Diagnostics
- RMSNorm expected output: `input_rms * gamma_rms / sqrt(input_rms² + eps)` — not just `gamma_rms`.
- Xavier init uses splitmix64 seed + 16 LCG iterations. A single iteration produces correlated outputs.
- Read `cached_encoder_output` (post-centering) for hidden-state diagnostics.
- For kernel timing use CUDA events, not `cudaStreamSynchronize` wall-time.
- Loss backward already applies `1/N`. Do NOT add another `1/tokens` scaling in parameter grad kernels.
- LibTorch gradient comparisons require IDENTICAL config (d_model, num_layers, num_heads, batch_tokens).

### Deleted — Do Not Recreate
- `UnifiedLoss_GPU.cu`, `ComputeLoss_GPU.cu` — replaced by `AutogradLoss.cu`
- `Embedding_GPU.cu` kernels / `EmbeddingLayer` class
- `ScaleGradFn` / `autograd::scale()`
- Value extraction head (`value_extraction_weight_/_bias_`) — Issue #142
- `rms_post_attn_gamma_`, `rms_post_ffn_gamma_` — Issue #148
- GPU delegate system (`Shared/Delegate/Delegate.hpp`)
- `centering_scratch_tensor` — single buffer is the source of truth

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
