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
| `TrainingState::cached_batch_size`, `cached_seq_len`, `cached_valid_tokens` | Phase1 `BatchPayload` owns per-step geometry; TrainingState must not preserve current-batch shadow metadata |
| `TrainingState::max_cached_batch`, `max_cached_seq_len`, `max_cached_tokens`, `max_logit_tokens` | `RunCapacity` / `LanguageModelConfig` own authored capacity; Tensor shapes own allocated capacity |
| `TrainingState::kv_cache_len`, `kv_cache_capacity`, `kv_cache_k`, `kv_cache_v`, `kv_cache_softmax_lse`, `decode_q_bf16`, `decode_attn_out_bf16`, `decode_attn_out_fp32` | Moved to `GenerationState`; autoregressive inference state is not training state |
| `TrainingState::decode_kv_bf16` | Dead decode scratch; K/V are converted directly into per-layer KV cache buffers |
| `TrainingState::d_loss_scratch`, `d_loss_sum_scratch` | Dead preallocated loss scratch; active `AutogradLoss.cu` allocates per-call `per_token_loss` / `d_loss_sum` and frees them after loss assembly |
| `TrainingState::grad_logits_tensor` | Dead logits-gradient mirror; LM-head logits are non-leaf, so `LogSoftmaxGradFn` allocates its own input-gradient workspace and passes that view to the upstream logits `GradFn` |
| `Shared/TensorContract/LOGIT_INTEGRATION_GUIDE.md`, `INTEGRATION_SUMMARY.md`, `logit_integration_example.cpp` | Obsolete migration artifacts for a nonexistent `autograd::cross_entropy()` API and deleted TrainingState logits-gradient buffers |
| `Shared/ScratchBlock/ScratchBlockPool_GPU.{hpp,cu}`, `TrainingState::scratch_pool`, `LanguageModel::configureScratchPool()`, `LanguageModel::isScratchPoolInitialized()` | Dead pinned-staging subsystem used only by `uploadBatchToDevice()`; batch upload now copies `BatchPayload` host vectors directly into TrainingState device cache tensors. ScratchBlock reasoning owns its own layer buffers. |
| `TrainingState::num_heads`, `TrainingState::num_kv_heads` | Dead GQA architecture shadows; `LanguageModelConfig` / `ModelArchitecture` own authored dimensions, `TensorContract::GQADims` is the call payload, and `GenerationState::KVCacheShape` owns allocated generation-cache geometry. |
| `TrainingState::MTPDiagnostics`, `TrainingState::mtp_diagnostics` | Moved to `Shared/MTP/MTPDiagnostics.hpp` and routed through `Autograd::LossResult` / `Training::BatchResult`; per-step MTP telemetry is not TrainingState-owned GPU resource state. |
