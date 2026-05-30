# Training VRAM Breakdown

Facts from config, startup log, and allocation math. Source: `memoryusestartup.txt`, `TrainingState::allocateStepDeviceWorkspaces()`, and Phase1_Startup.

---

## Config (from startup log)

- **Cache allocation:** batch=8, seq_len=1024, tokens=8192 (log line: `Cache allocation: batch=8, seq_len=1024, tokens=8192`).
- **Model:** d_model=768, num_layers=12, num_heads=12, d_ff=3072, max_seq_len=1024.
- **Vocab:** 2512 (DataLoader: "Vocab size: 2512").
- **Params:** total_params=107,798,346 (getModelStats: embedding=1,929,216, encoder=103,937,402, lm_head=1,931,728).

---

## Startup allocation (after `Startup::initializeTrainingRuntime`)

All sizes float32, 4 bytes per element.

**Parameter-related (4 copies: weights, grads, Adam m, Adam v):**

- 107,798,346 × 4 × 4 = 1,724,773,536 bytes ≈ **1.61 GiB**.
- **Running tally: 1.61 GiB**

**Pre-allocated TrainingState upload workspaces (max_tokens=8192):**

Durable LM-head input/logits snapshots were deleted. Forward outputs now live in `AutogradIntermediates` only; TrainingState keeps reusable upload/staging buffers such as targets, token ids, numeric side-channels, atom side-channels, and sequence weights.

The remaining fixed upload buffers are tiny at this config (well under 1 MiB total).
- **Running tally: ~1.61 GiB**

**Startup total (params + upload workspaces):** ~1.61 GiB.

Historical note: TrainingState-owned `grad_encoder`, `grad_ffn_*`, `grad_attn_*`, `grad_q/k/v`, `grad_qkv_*`, and `grad_logits_tensor` buffers were deleted. TensorContract owns activation/intermediate gradient lifecycle through `Tensor.grad_` and GradFn scratch.

---

## First forward/backward (autograd intermediates)

Per-layer tensors owned directly by `ModelForwardOutputs` (one retained slot per layer across 12 layers): ln1_out, ln2_out, qkv_out, Q_bhsd, K_bhsd, V_bhsd, attn_out_bhsd, attn_out, proj_out, scaled_proj, residual1, ffn_out, scaled_ffn, output, ffn_gate_out, ffn_silu_out, ffn_linear1_out, ffn_swiglu_out. Shapes [8192, d_model] or [8192, d_ff] or [8192, kv_dim] with d_model=768, d_ff=3072, qkv fused 1280, kv_dim=256.

Approximate per-layer total: 704 MiB. Over 12 layers: 704 × 12 = 8,448 MiB ≈ **8.25 GiB**.

Flash Attention backward workspace (per layer, batch=8, seqlen=1024, n_heads=12, head_dim=64): dq_accum + dsoftmax_sum ≈ **0.67 GiB** (12 layers). ScaledDotProductAttentionGradFn also allocates **dq_bf16, dk_bf16, dv_bf16, dout_bf16** per layer (TensorContract_GPU.cu): 12 × (4 × 8×1024×12×64×2) ≈ **0.87 GiB**.

**First-batch additional (autograd + FA scratch + FA bf16 grads):** ~8.25 + 0.67 + 0.87 ≈ **9.79 GiB**.
- **Running tally: 1.61 + 9.79 = 11.40 GiB**

---

## Running tally (total accounted for)

| Item | GiB | Cumulative GiB |
|------|-----|----------------|
| Params (weights, grads, Adam m, Adam v) | 1.61 | 1.61 |
| Pre-allocated upload buffers | ~0.00 | 1.61 |
| First-batch autograd (12-layer intermediates) | 8.25 | 9.86 |
| First-batch Flash Attention scratch (dq_accum + dsoftmax_sum) | 0.67 | 10.53 |
| First-batch Flash Attention bf16 grads (dq/dk/dv/dout × 12 layers) | 0.87 | 11.40 |
| GRIM-TS (this run: disabled) | 0 | 11.31 |
| Telemetry (lattice 8×5 + control disabled) | ~0.000004 | 11.40 |

**Total accounted for: 11.40 GiB**

---

## Sum vs reported 40 GB

Device total: 40,441 MB (from log: "Memory: 40441 MB"). Our breakdown accounts for **11.40 GiB** of *tracked* allocations (params, upload workspaces, first-batch autograd + FA). So **40,441 MB − 11.40 GiB ≈ 28–29 GiB is unaccounted**: we do not attribute it to any buffer in this doc. Likely sources include CUDA allocator reserved/fragmentation, cuBLAS/cuDNN workspace, and other runtime pools. Until we measure or instrument those, the gap remains unexplained.

---

## Observed snapshots (from logs)

| When | Log line | Interpretation |
|------|----------|----------------|
| Pre-validation checkpoint | `GPU memory: 37197 MB free / 40441 MB total` | **37,197 MB = ~37.2 GiB used**; remaining capacity = 40,441 − 37,197 = **3,244 MB = ~3.2 GiB free**. Steady state between steps (autograd intermediates cleared). So at that moment ~37 GiB was in use; we only account for ~12 GiB of it. |
| Peak (first fwd/bwd) | — | We *account* for **~11.31 GiB**; actual device *used* at peak is not logged here. If at peak the driver reports e.g. 35–37 GiB used, we still only explain ~11.3 GiB; the rest is the same unaccounted ~28 GiB. |

---

## Logits (this config)

There is no durable TrainingState logits cache and no logits pointer on the forward result. Full-batch logits are graph-owned by `AutogradIntermediates::logits_tensor` during the active forward/loss/backward window, then released at `AutogradStepScope` teardown. Inference prefill copies the last-token logits directly from `AutogradIntermediates::logits_tensor` before clearing intermediates.

---

## GRIM-TS (Guess cache)

**This run:** Log line: `[GuessCache][INFO] Guess cache disabled (guess_aux.enabled=false)`. No GRIM-TS buffers are allocated; **0 bytes** GPU VRAM.

**When enabled** (guess_aux.enabled=true): `GuessCacheScope::OwnedBuffers::allocate` owns the cache buffers with capacity = `kDefaultGuessCacheCapacity` = 16,384 (GuessCacheTraining.hpp), diversity_bloom_bits = 65,536, pinned_buffer_size = 8,192. Sizes use `sizeof(GRIMTS::GuessRecord)` and `sizeof(GRIMTS::GuessMetadata)`:

| Buffer | Formula | Bytes |
|--------|--------|--------|
| records | capacity × 96 | 1,572,864 |
| keys | capacity × 8 | 131,072 |
| size | 1 × 4 | 4 |
| evict_cursor | 1 × 4 | 4 |
| diversity_bloom | ((65536+31)/32) × 4 | 8,192 |
| calibration_offset | 4 | 4 |
| single_meta_buffer | 32 | 32 |
| single_reward_buffer | 4 | 4 |

GPU total: 1,712,180 bytes ≈ **1.63 MiB**. Pinned buffers (pinned_meta 8192×32, pinned_rewards 8192×4) are host memory (cudaMallocHost), not VRAM.

So with GRIM-TS enabled at default capacity, GPU use is about **1.6 MiB**. It does not account for the ~29 GB gap.

---

## Telemetry (lattice + control)

**This run:** Log: `[Telemetry] Lattice initialized: 8 levels, 5 streams, GPU-resident (Pattern B)` and `Telemetry control: DISABLED (monitoring only)`. So the lattice is allocated; control may or may not allocate (initGPU when enabled).

**TelemetryLattice** (TelemetryLattice_GPU.cu): total_states = num_levels × num_streams = 8 × 5 = 40. Allocations:

| Buffer | Formula | Bytes |
|--------|--------|--------|
| levels_ | 40 × sizeof(LatticeLevelState) | 40 × 96 = 3,840 |
| observations_ | num_streams × 4 | 5 × 4 = 20 |
| scratch_vectors_ | num_streams × 10 × 4 | 5 × 40 = 200 |
| d_error_flag_ | sizeof(int) | 4 |

LatticeLevelState = TelemetryState (20 float + 2 uint32) + stride (uint32_t) + last_update (uint32_t); TelemetryState_GPU.hpp gives 96 bytes per level state. Lattice GPU total: **4,064 bytes** (~4 KiB).

**TelemetryControl** (TelemetryControl_GPU.cu), when initGPU() is called: d_config_ (sizeof(TelemetryControlConfig)), d_state_ 64, d_decision_ 48, d_input_ 32 (TelemetryControl_GPU.hpp static_asserts). Config is one struct of floats/ints/bools; total control is on the order of **~0.5 KiB** if allocated.

**Telemetry total (this run):** lattice 4,064 B; control 0 (disabled). **~4 KiB** GPU. Negligible for the running tally; cumulative stays 11.40 GiB.

---

## Where the numbers come from

- **HyperParameters_GPU.hpp + HyperparameterGroupings.hpp / trainingFixedShapeHP():** `HyperParameters_GPU.hpp` validates and computes the startup fixed-shape rectangle from root `StartupConfig` / `TrainingHyperparameters` facts, then `trainingFixedShapeHP()` slices the immutable view (`batch_size`, `max_seq_len`, `max_tokens_per_batch`). `finalizeLanguageModelConfig(..., ModelExecutionMode::TRAINING)` lives in `HyperParameters_GPU.hpp` and mirrors the validated root facts into `LanguageModelConfig` without accepting a grouping payload.
- **Startup/Model/ModelGpuAssembly.cu:** `Startup::initializeTrainingRuntime()` and `Startup::initializeInferenceRuntime()` slice `HyperParameters::TrainingStateWorkspaceHP` from finalized config and pass that grouping plus the primary stream to `TrainingState::allocateStepDeviceWorkspaces()`.
- **TrainingStateGPU.cu:** owns allocation of target/token staging tensors and `sequence_weights_tensor`; LM-head input/logits are not durable TrainingState allocations.
- **memoryusestartup.txt**: "Cache allocation: batch=8, seq_len=1024, tokens=8192"; "total_params=107798346"; older logs may mention `Allocated cached_logits`, but that durable snapshot allocation has been deleted.
- **GRIM-TS:** `GuessCacheScope::OwnedBuffers::allocate` in GuessCacheTraining.cu (capacity, `sizeof(GRIMTS::GuessRecord)`=96, `sizeof(GRIMTS::GuessMetadata)`=32); GuessCacheScope uses kDefaultGuessCacheCapacity=16384, diversity_bloom_bits=65536, pinned_buffer_size=8192. GRIM-TS.hpp static_assert(sizeof(GuessRecord)==96).
- **Telemetry:** TelemetryLattice_GPU.cu constructor: levels_ = num_levels × num_streams × sizeof(LatticeLevelState); observations_ = Tensor::zeros({num_streams}); scratch_vectors_ = Tensor::zeros({num_streams, 10}); d_error_flag_ = 4 bytes. TelemetryState_GPU.hpp LatticeLevelState = TelemetryState + stride + last_update (96 bytes). Log: "8 levels, 5 streams". TelemetryControl_GPU.cu initGPU: d_config_, d_state_ (64), d_decision_ (48), d_input_ (32).

---

## Cleanup (all allocations cleared properly)

| Allocation | Cleared where |
|------------|----------------|
| **Params, grads, optimizer m/v** | Tensor / vector members of TrainingState; freed in ~TrainingState when members destruct. |
| **Pre-allocated upload workspaces** (cached_targets, token caches, sequence_weights) | Tensor members; Tensor::~Tensor() → release() → cudaFree(data). |
| **class_weights_tensor** | Tensor member; `Tensor::~Tensor()` releases class-balanced weights. |
| **PBM (alibi_slopes, rope_inv_freq)** | `TrainingContext::pbm_owner` (`PBM::PBMStateOwner`) RAII releases PBM buffers and upload event; consumers borrow the same `PBMState` instead of a duplicate view struct. |
| **TeacherLogits / reference_logits** | `TeacherLogits::Buffer` RAII destructor releases device storage. |
| **Optimizer states** | `Training::OptimizerContext::optimizer_state.clear()` clears dedicated optimizer-state owner tensors. |
| **Guess cache (GRIM-TS)** | `GuessCacheScope::OwnedBuffers` RAII member; released when `ctx.guess_cache_scope.reset()` runs before model teardown. |
| **Debug grad buffers** | ~TrainingState: `freeDebugGradBuffers()` (assigns Tensor() to release). |
| **GradNorm scratch** | `std::unique_ptr<GradNormScratch>` member; `GradNormScratch::~GradNormScratch()` releases GPU/pinned buffers. |
| **Autograd intermediates** (`layer_outputs`, `embedding_tensor`, `encoder_layer_outputs`, `logits_tensor`, `loss_tensor`, etc.) | Cleared after each backward in training loop; at shutdown, **Phase3 releaseResources()** calls `ctx.model->getTrainingState().autograd_intermediates.clear()` before `ctx.model.reset()` so grad_fns and intermediates are released in a controlled order. When ~TrainingState runs, AutogradIntermediates member destructor also clears. |
| **Per-layer grad_fns** (ScaledDotProductAttentionGradFn dq_accum, dsoftmax_sum, dq/dk/dv/dout_bf16, saved_*) | When intermediates are cleared or Tensors destruct, grad_fn refcount drops; ~ScaledDotProductAttentionGradFn calls release_saved() which cudaFrees all 11 buffers. |
| **Telemetry (lattice, control)** | Owned by TrainingContext (ctx.telemetry); when ctx is destroyed, unique_ptrs destruct; TelemetryLattice_GPU and TelemetryControl_GPU destructors cudaFree their buffers. |
| **KV/decode BF16/FP32 buffers** | `DeviceAllocation` RAII members/vectors release typed CUDA buffers. |
| **StreamController / cublas_handle** | streams owned by `StreamController`; cuBLAS owned by `CublasHandleOwner` RAII member. |

Shutdown order: Phase3 `releaseResources()` → clear autograd_intermediates → ctx.model.reset() → ~LanguageModel → ~TrainingContext members in reverse declaration order, including `ctx.gpu_model` (layer topology teardown) and `ctx.pbm_owner` (PBM release), then the destroyed model's `TrainingState` teardown (TeacherLogits RAII, DeviceAllocation KV/decode buffers, GradNorm unique_ptr, cuBLAS owner, Tensor members). Telemetry is released when TrainingContext is destroyed after Phase3.

---

## Full audit of GPU allocations (training path)

Every GPU allocation site that can run during training, with source and size formula. Excludes inference-only (InitInferenceState) and tests.

| Component | Buffer / allocation | Source | Formula (bytes) |
|-----------|---------------------|--------|------------------|
| **InitTrainingState** | Params (weights) | Phase1 / model load | `total_params × 4` |
| | Grads | InitTrainingState / optimizer | `total_params × 4` |
| | Adam m, Adam v | TrainingStateGPU.cu | `total_params × 4` each |
| | cached_targets_tensor | TrainingStateGPU.cu | `max_logit_tokens × 1 × 4` |
| | cached_token_ids_tensor | TrainingStateGPU.cu | `1 × max_tokens × 4` (int32) |
| | cached_token_numeric_values | TrainingStateGPU.cu | `1 × max_tokens × 4` |
| | cached_token_atom_mask | TrainingStateGPU.cu | `1 × max_tokens × 1` (uint8) |
| | cached_token_atom_flags | TrainingStateGPU.cu | `1 × max_tokens × 4` (uint32) |
| | sequence_weights_tensor | TrainingStateGPU.cu | `max_batch_size × 1 × 4` |
| **Phase1_Startup** | class_weights_tensor | Phase1_Startup.cu (class_balanced) | `vocab_size × 4` |
| **TrainingStateGPU** | Guess cache (records, keys, bloom, etc.) | TrainingStateGPU.cu | When enabled: ~1.63 MiB (see GRIM-TS) |
| | Debug grad norm buffers | TrainingStateGPU.cu | When debug: small per-param |
| **ScaledDotProductAttentionGradFn** (per layer, at first backward) | dq_accum | TensorContract_GPU.cu | `round(seq,128)×batch×n_heads×round(head_dim,32|64)×4` |
| | dsoftmax_sum | TensorContract_GPU.cu | `round(seq,128)×batch×n_heads×4` |
| | dq_bf16, dout_bf16 | TensorContract_GPU.cu | `batch×seq×num_heads×head_dim×2` each |
| | dk_bf16, dv_bf16 | TensorContract_GPU.cu | `batch×seq×num_heads×head_dim×2` each (sized for num_heads) |
| **ModelForwardOutputs per-layer retained tensors** (per layer × 12) | ln1_out, ln2_out, qkv_out, Q_bhsd, K_bhsd, V_bhsd, attn_out_bhsd, attn_out, proj_out, scaled_proj, residual1, ffn_out, scaled_ffn, output, FFN intermediate tensors | ModelForwardOutputs.hpp / Encoding | `[tokens, d_model]` or `[tokens, kv_dim]` or `[tokens, d_ff]`; see Geometry map |
| | qkv_out | Encoding | `[tokens, d_model + 2×kv_dim]×4` |
| | Q_bhsd, K_bhsd, V_bhsd, attn_out_bhsd | Encoding | `[batch, heads, seq, head_dim]×4` |
| | ffn_gate_out, ffn_silu_out, ffn_linear1_out, ffn_swiglu_out | Encoding | `[tokens, d_ff]×4` |
| **GradNorm** | d_partial_sums | GradNormGPU.cu | `max_groups × 4` (GPU); h_partial_sums / h_metrics are host |
| **TeacherLogits / ReferenceLogits** | Buffer | TeacherLogits_GPU.hpp ensureCapacity | When used: `tokens × vocab_size × 4` per buffer |
| **Autograd (ephemeral)** | LayerScaleGradFn input_grad / input_data | GradFns/LayerScaleGradFn.cu | Non-leaf only: `element_count × 4`; gamma vectors are `[1, d_model]` (freed after backward) |
| **FlashAttentionLayer** | ensureScratch (fwd bf16, softmax_lse) | FlashAttention layer | Fwd-only scratch; backward uses GradFn buffers above |
| **Telemetry** | Lattice / control | TelemetryLattice_GPU.cu, TelemetryControl_GPU.cu | ~4 KiB + control if enabled |
| **ScratchBlock** | Backward temps (e.g. atom caches) | ScratchBlockReasoning_GPU.cu | Per-call temporary; not pre-allocated pool |

Config used for numeric examples: `max_tokens = 8192`, `max_logit_tokens = 8192`, `batch = 8`, `seq_len = 1024`, `d_model = 768`, `d_ff = 3072`, `num_heads = 12`, `head_dim = 64`, `vocab_size = 2512`, `num_layers = 12`.

---

## Geometry → memory map

Symbols: **T** = max_tokens (batch × seq_len), **T_logit** = max_logit_tokens, **B** = batch, **S** = seq_len, **D** = d_model, **F** = d_ff, **H** = num_heads, **H_kv** = num_kv_heads, **d** = head_dim, **V** = vocab_size, **L** = num_layers. All element sizes in bytes: float = 4, bfloat16 = 2, uint16 = 2, uint8 = 1, int32 = 4.

| Buffer | Dimensions (shape) | Bytes formula | Example (this config) |
|--------|--------------------|---------------|------------------------|
| Params (weights) | — | `total_params × 4` | 431,193,384 → 1.61 GiB / 4 copies |
| cached_targets_tensor | [T_logit, 1] | `T_logit × 4` | 32.8 KiB |
| cached_token_ids_tensor | [1, T] | `T × 4` | 32.8 KiB |
| cached_token_numeric_values | [1, T] | `T × 4` | 32.8 KiB |
| cached_token_atom_mask | [1, T] | `T × 1` | 8 KiB |
| cached_token_atom_flags | [1, T] | `T × 4` | 32.8 KiB |
| sequence_weights_tensor | [B, 1] | `B × 4` | 32 B |
| class_weights_tensor | [V] | `V × 4` | 2512×4 = 10 KiB |
| **Per-layer tensors on ModelForwardOutputs** | | | |
| ln1_out, ln2_out | [T, D] | `T × D × 4` | 25.2 MiB each |
| qkv_out | [T, D + 2×H_kv×d] | `T × (D + 2×H_kv×d) × 4` | 8192×1280×4 = 41.9 MiB |
| proj_out, scaled_proj, residual1, ffn_out, scaled_ffn, output | [T, D] | `T × D × 4` | 25.2 MiB each |
| Q_bhsd, attn_out_bhsd | [B, H, S, d] | `B × H × S × d × 4` | 8×12×1024×64×4 = 25.2 MiB each |
| K_bhsd, V_bhsd | [B, H_kv, S, d] | `B × H_kv × S × d × 4` | 8×4×1024×64×4 = 8.4 MiB each |
| attn_out | [T, D] | `T × D × 4` | 25.2 MiB |
| ffn_gate_out, ffn_silu_out, ffn_linear1_out, ffn_swiglu_out | [T, F] | `T × F × 4` | 100.7 MiB each |
| **Per-layer FA backward (ScaledDotProductAttentionGradFn)** | | | |
| dq_accum | [B, round(S,128), H, round(d,32|64)] fp32 | `round(S,128)×B×H×round(d,32|64)×4` | 1024×8×12×64×4 = 25.2 MiB |
| dsoftmax_sum | [B, H, round(S,128)] fp32 | `B × H × round(S,128) × 4` | 8×12×1024×4 = 393 KiB |
| dq_bf16, dout_bf16 | [B, S, H, d] | `B × S × H × d × 2` | 8×1024×12×64×2 = 12.6 MiB each |
| dk_bf16, dv_bf16 | [B, S, H, d] | `B × S × H × d × 2` | 12.6 MiB each |
| **GradNorm** | d_partial_sums | [max_groups] | `max_groups × 4` | small |
| **Teacher/Reference logits** | Buffer | [tokens, V] | `tokens × V × 4` (when used) | 82.4 MiB per buffer |

Example totals for this config: startup (params + pre-allocated + small caches) ≈ 1.61 GiB; first forward adds 12 × retained per-layer `ModelForwardOutputs` slots ≈ 8.25 GiB; first backward adds 12 × (dq_accum + dsoftmax_sum + 4×bf16) ≈ 0.67 + 0.87 ≈ 1.54 GiB FA backward. Total accounted: 1.61 + 9.79 = **11.40 GiB**. The ~29 GB gap vs 40 GB device total remains unaccounted (fragmentation, cuBLAS/cuDNN workspace, or other allocators).
