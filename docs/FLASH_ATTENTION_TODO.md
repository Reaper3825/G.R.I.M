# FlashAttention v2 Integration TODO

## Remaining Work
- [ ] Remove or exclude legacy FlashAttention implementation files from build (`Flash_Attention_Kernal_old.*`), once new paths are live.

## Completed
- [x] Replaced `Flash_Attention_Kernal.hpp` with FA v2 C API declarations.
- [x] Added `softmax_lse` helper for size/binding of dense FP32 LSE buffers.
- [x] Added per-layer `cached_softmax_lse` to `TrainingState` and allocated it in init (training + inference).
- [x] Added per-step FA workspace buffers (`fa_dq_accum`, `fa_dsoftmax_sum`) to `TrainingState` and allocated them in init.
- [x] Replaced `flashAttentionForward` calls with `flash_attn_fwd_ex` and passed per-layer `softmax_lse` (Encoding forward path).
- [x] Replaced `flashAttentionBackward` calls with `flash_attn_bwd_ex` and passed `softmax_lse`, `fa_dq_accum`, and `fa_dsoftmax_sum` (BackwardPhase2).
- [x] Wired `cached_softmax_lse[layer]` through `GPUGrimEncoder::forwardGPU`, `ForwardWithCache_GPU.cu`, and `ComputeLossBatch.cu`.
- [x] Removed `getFlashAttentionWorkspaceSize` usage from `Encoding_GPU.cu` (no FA workspace in encoder workspace now).
- [x] Selected BF16 path and implemented Q/K/V/Out conversion helpers; `is_bf16` matches the buffers.
- [x] Ensured FA input caches are post-RoPE BHSD and preserved for backward.
- [x] Updated tests/diagnostics to the FA v2 API.
