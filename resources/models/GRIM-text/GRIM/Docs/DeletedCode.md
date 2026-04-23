# Deleted Code — Do Not Recreate

| Removed | Reason |
|---------|--------|
| `UnifiedLoss_GPU.cu`, `ComputeLoss_GPU.cu` | Replaced by `AutogradLoss.cu` — old path was disconnected from autograd gradients |
| `Embedding_GPU.cu` kernels / `EmbeddingLayer` class | Dead; only `destroyEmbeddingRuntime()` remains |
| `ScaleGradFn` / `autograd::scale()` | Embedding scale removed (see [LMHead.md](LMHead.md)) |
| Value extraction head (`value_extraction_weight_/_bias_`) | Issue #142 — ScratchBlock is a reasoning layer, not scalar regression |
| `rms_post_attn_gamma_`, `rms_post_ffn_gamma_` | Sandwich norm deleted (Issue #148) |
| GPU delegate system (`Shared/Delegate/Delegate.hpp`) | Zero registered callbacks — Rule 26 |
| `centering_scratch_tensor` | Single buffer (`cached_encoder_output`) is the source of truth |
