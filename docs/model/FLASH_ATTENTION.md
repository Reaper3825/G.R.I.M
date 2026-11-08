# FlashAttention v2: Current Integration and Verification

FlashAttention v2 is the compiled attention implementation. This is no longer
a pending migration from the handwritten attention/backward phases.

## Implemented flow

- [TrainingLoop/CMakeLists.txt](../../resources/models/GRIM-text/training/TrainingLoop/CMakeLists.txt)
  builds `grim_flash_attention` from the GRIM wrapper and diagnostic units,
  using the pinned external FlashAttention source and its required headers.
- [Flash_Attention_Kernal.cu](../../resources/models/GRIM-text/Layers/FlashAttention/Flash_Attention_Kernal.cu)
  adapts the pinned upstream API; it is not the historical handwritten kernel
  described by the archived entropy investigation.
- [EncoderSelfAttention_GPU.cu](../../resources/models/GRIM-text/Layers/FlashAttention/EncoderSelfAttention_GPU.cu)
  assembles attention projections and invokes the shared TensorContract path.
- `autograd::scaled_dot_product_attention()` in
  [AutogradAttention.cu](../../resources/models/GRIM-text/Shared/TensorContract/AutogradAttention.cu)
  converts FP32 BHSD tensors to BF16 BSHD, calls `flash_attn_fwd_ex()`,
  and converts the output back.

## Forward/backward ownership

The SDPA operation allocates Q/K/V/output conversion buffers and FP32 LSE
`[batch, query_heads, sequence]`. For a differentiable call it transfers those
buffers to `ScaledDotProductAttentionGradFn`, together with backward workspace.
Non-differentiable calls release the operation-local buffers without attaching
a backward node. These are not per-layer persistent TrainingState LSE caches.

The GradFn calls `flash_attn_bwd_ex()` with the saved scale, causal policy,
ALiBi slopes, dropout probability, and dropout seed. It clears backward output
scratch before use and preserves the forward/backward padding-bound contract.
GQA scratch and its reduction must follow the actual query/KV head geometry;
do not revive the old manual encoder-backward path.

The optional attention-off-by-one epilogue changes output and LSE before
the GradFn captures them. Non-causal padded attention requires uploaded
sequence lengths. These are explicit current policies, not alternate legacy
attention implementations.

## Verification still requiring execution

After a relevant kernel or pinned-API change, explicitly requested GPU checks
should exercise forward/backward shape, causal/padding policy, GQA, dropout-mask
reproduction, and finite gradients. The build defines a `flash_attention_test`
target, but target existence is not a passing test result.

No trainer or attention test executable was built or run for this documentation
update. Historical entropy notes are in
[legacy/FLASH_ATTENTION_DIAGNOSTIC_ENTROPY_BUG.md](../legacy/FLASH_ATTENTION_DIAGNOSTIC_ENTROPY_BUG.md).
