# Training Memory Investigation (A100 40GB OOM)

**Config:** batch_size=10, max_seq_len=1024, d_model=1024, num_layers=12, num_heads=16, num_kv_heads=4, d_ff=1024, vocab~10k

## Preallocated TrainingState Step Workspaces

Allocation owner: `TrainingState::allocateStepDeviceWorkspaces()` in `Shared/TrainingState/TrainingStateGPU.cu`. `InitTrainingState.cu` and `InitInferenceState.cu` pass the authored `LanguageModelConfig` plus the primary stream; no cache-specific grouped HP object is constructed.

All sizes use `max_cached_batch * max_cached_seq_len` = 10 * 1024 = **10,240 tokens**.

| Buffer | Size Formula | Est. MB |
|--------|--------------|---------|
| cached_encoder_output | 10240 x 1024 x 4 | 40 |
| cached_logits | 10240 x vocab x 4 | **390** (vocab=10k) |

Historical note: the old TrainingState-owned `grad_encoder`, `grad_ffn_*`, `grad_q/k/v`, `grad_qkv_*`, `grad_logits_tensor`, and centering scratch buffers were deleted. TensorContract owns activation/intermediate gradient lifecycle through `Tensor.grad_` and GradFn scratch.

**Logits dominate** -- if `actual_vocab_size` > 10k, this scales linearly. At vocab=50k: ~2 GB for cached logits alone.

---

## Flash Attention (Per Layer, Per Forward)

Created in `scaled_dot_product_attention` when `requires_grad=true`. **12 layers** x one set of buffers per forward.

**Forward (transferred to grad_fn):** q_bf16, k_bf16, v_bf16, out_bf16, softmax_lse
**Backward workspace:** dq_accum, dsoftmax_sum, dq_bf16, dk_bf16, dv_bf16, dout_bf16

For batch=10, seq=1024, heads=16, head_dim=64:
- q_elems = 10 x 1024 x 16 x 64 = 10.5M -> 21 MB bf16
- dk/dv sized for **num_heads** (Issue #72): 21 MB x 2 = 42 MB
- Per layer: ~130 MB. **x 12 layers = 1.5 GB** (released after backward via release_saved).

---

## Hardcoded / Config-Independent Allocations

### 1. Unigram Tokenizer Runtime Workspace
```cpp
// HyperParameters_GPU.hpp
constexpr size_t UNIGRAM_MAX_SEQUENCE_LENGTH = DEFAULT_MAX_SEQ_LEN * 4;  // default floor, currently 1024*4 = 4096
```
- `UNIGRAM_MAX_SEQUENCE_LENGTH` is only the default-capacity floor for generic/server tokenizer runtime init.
- Corpus training/GRMT generation dynamically finalizes tokenizer runtime state with `longest_normalized_e_step_segment` via `UnigramTrainingRuntimeReport`; long concept rows must not be capped by this static floor.
- `d_viterbi_scores`, `d_viterbi_prev`, `d_viterbi_tokens`: `workspace_max_length + 1` elements each.
- **Not a major factor**, but tokenizer workspace capacity is now workload-sized for corpus encoding instead of being silently tied to model `max_seq_len`.

### 2. Guess Cache (when enabled)
- Capacity: `kDefaultGuessCacheCapacity` = 16384, or config `guess_cache.initial_capacity`
- ai_config has `guess_cache.enabled: false` -- so likely not allocated.

### 3. ScratchBlock
- `scratch_block_max_atoms`: 8192 in config
- Atom embedding gradient scratch: 8192 x 96 x 4 = 3 MB

---

## Potential Culprits

### 1. **actual_vocab_size > 10,000**
`cached_logits` and `grad_logits` are `[max_tokens, vocab_size]`.
If vocab from tokenizer/GRMT is 20k-50k, this becomes 800 MB-2 GB for those two buffers.

**Check:** Log `ctx.config.actual_vocab_size` at startup.

### 2. **d_ff mismatch**
Config shows `d_ff: 1024` (same as d_model). Typical is `4 x d_model`.
If some path uses `DEFAULT_D_FF = 3072` or `4*1024`, per-layer ForwardIntermediates and TensorContract GradFn scratch grow.
Verify `arch.d_ff` matches intent.

### 3. **cuBLAS / library workspace**
cuBLAS and Flash Attention use internal workspace. On A100 this can be hundreds of MB.
`cublasSetMathMode(..., CUBLAS_TF32_TENSOR_OP_MATH)` enables Tensor Cores and can increase workspace.

### 4. **Gradient accumulation and graph retention**
With `gradient_accumulation_steps=6`, we do 6 full forward+backward passes before optimizer step.
If intermediates or grad_fns are retained across these passes (e.g. in a shared structure), memory can pile up.
Current flow appears to be one batch forward -> one batch backward -> release -> next batch.
Worth confirming no extra retention.

### 5. **Fragmentation**
Long training runs can fragment GPU memory. Allocating many similar-sized buffers and freeing them in a different order can leave unusable gaps.

---

## Issue #85: Flash Attention Backward Bugs (Critical)

Traced the entire FA2 backward path and compared against upstream Dao FA2. Found:

### BUG 1: dK/dV stride mismatch with GQA (memory corruption)

`init_bwd_params_contiguous` set `dk_row_stride = n_kv_heads * head_dim` (256) but Issue #72
allocated dk/dv buffers for `num_heads` (16), not `num_kv_heads` (4). The kernel writes dK/dV at
`bidh * dk_head_stride` where bidh ranges 0-15. With row stride 256, Head 4 Row 0 aliases
Head 0 Row 1 -- concurrent thread blocks race-condition causing silently corrupted dK/dV gradients.

**Fix**: `dk_row_stride = n_heads * head_dim`.

### BUG 2: GQA reduction applied wrong scaling

Reduction kernel divided by `sqrt(heads_per_kv_group)`, halving dK/dV gradients.
Correct operation (chain rule, upstream) is plain sum. K/V trained at half learning rate.

**Fix**: Remove the `1/sqrt` scaling.

### Confirmed Working

- Preprocessing kernel (Issue #84) IS running, correctly computes dsoftmax_sum
- dQ accumulation across n_blocks is correct (load-add-store through dq_accum)
- Non-parallel kernel is valid (perf, not correctness issue)
- scale_softmax_rp_dropout initialized correctly

---

## Recommended Next Steps

1. **Add memory logging** at init and before/after first batch to see where usage jumps.
2. **Verify `actual_vocab_size`** -- if much larger than 10k, reduce batch size or max_seq_len.
3. **Confirm `d_ff`** -- ensure it matches intended architecture (e.g. 1024 vs 4096).
4. **Try reducing batch_size to 4-6** as a test -- if OOM disappears, it's batch-dependent allocation.
5. **Check for double init** -- ensure `InitTrainingState` and `InitInferenceState` are not both run for training.
