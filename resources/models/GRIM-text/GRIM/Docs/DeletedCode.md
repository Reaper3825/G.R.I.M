# Deleted Code — Do Not Recreate

| Removed | Reason |
|---------|--------|
| `UnifiedLoss_GPU.cu`, `ComputeLoss_GPU.cu` | Replaced by `AutogradLoss.cu` — old path was disconnected from autograd gradients |
| `Embedding_GPU.cu` kernels / `EmbeddingLayer` class | Dead; only `destroyEmbeddingRuntime()` remains |
| `ScaleGradFn` / `autograd::scale()` | Embedding scale removed (see [LMHead.md](LMHead.md)) |
| Value extraction head (`value_extraction_weight_/_bias_`) | Issue #142 — ScratchBlock is a reasoning layer, not scalar regression |
| `rms_post_attn_gamma_`, `rms_post_ffn_gamma_` | Sandwich norm deleted (Issue #148) |
| GPU delegate system (`Shared/Delegate/Delegate.hpp`) | Zero registered callbacks — Rule 26 |
| `centering_scratch_tensor` | Live LM-head input tensor is the source of truth inside the forward/autograd boundary |
| `TrainingState::cached_logits_tensor` | Live logits are observed through `AutogradIntermediates::logits_tensor` inside the active boundary; no durable logits mailbox or result pointer |
| `TrainingState::cached_encoder_output` | Live LM-head input is observed through `lm_head_input_tensor` or `encoder_output_tensor`; no durable hidden-state mailbox |
| `TrainingState::cached_batch_size`, `cached_seq_len`, `cached_valid_tokens` | Phase1 `BatchPayload` owns per-step geometry; TrainingState must not preserve current-batch shadow metadata |
| `TrainingState::max_cached_batch`, `max_cached_seq_len`, `max_cached_tokens`, `max_logit_tokens` | Deleted shadow capacity state. `HyperparameterGroupings.hpp::trainingFixedShapeHP()` / `LanguageModelConfig` own authored `batch_size`, `max_cached_seq_len`, and token budget; Tensor shapes own allocated capacity |
| `TrainingState::cached_num_layers`, `ModelAllocationState::model_max_cached_seq_len` | Dead shadow mirrors; layer count belongs to `LanguageModelConfig` / model topology, and sequence capacity belongs to `HyperparameterGroupings.hpp::trainingFixedShapeHP()` / payload or generation paths |
| `TrainingState::kv_cache_len`, `kv_cache_capacity`, `kv_cache_k`, `kv_cache_v`, `kv_cache_softmax_lse`, `decode_q_bf16`, `decode_attn_out_bf16`, `decode_attn_out_fp32` | Deleted with the model-owned KV decode path; Phase2 inference owns generation chronology and uses shared full-context scoring |
| `TrainingState::inference_exec_memory`, `has_inference_exec_memory`, `decode_selector_*`, `single_token_*` | Persistent execution memory and selector result moved to `GenerationState`; single-token scratch deleted with the KV decode path |
| `TrainingState::decode_kv_bf16` | Dead decode scratch; K/V are converted directly into per-layer KV cache buffers |
| `TrainingState::d_loss_scratch`, `d_loss_sum_scratch`, `cross_entropy_*_tensor`, `TrainingContext::ce_workspace_owner`, `CrossEntropyForwardWorkspaceOwner` | Deleted legacy/failed kitchen-drawer and Phase1 loss scratch ownership. Active CE scalar reduction scratch is owned by `NLLLossGradFn::capture_inputs()` and released by `NLLLossGradFn::release_saved()`; full-vocab/per-token loss scratch must not be resurrected as durable runtime allocation. |
| `TrainingState::grad_logits_tensor` | Dead logits-gradient mirror; LM-head logits are non-leaf, so `LogSoftmaxGradFn` allocates its own input-gradient workspace and passes that view to the upstream logits `GradFn` |
| `Shared/TensorContract/LOGIT_INTEGRATION_GUIDE.md`, `INTEGRATION_SUMMARY.md`, `logit_integration_example.cpp` | Obsolete migration artifacts for a nonexistent `autograd::cross_entropy()` API and deleted TrainingState logits-gradient buffers |
| `Layers/Attention/QKV_Projector.{hpp,cu}` | Stale wrapper name; QKV projection kernels were already deleted and the only remaining BHSD→BSM reshape now lives directly in `autograd::reshape_bhsd_to_flat()` via `TensorConversion::convert_BHSD_to_BSM()` |
| `Shared/ScratchBlock/ScratchBlockPool_GPU.{hpp,cu}`, `TrainingState::scratch_pool`, `LanguageModel::configureScratchPool()`, `LanguageModel::isScratchPoolInitialized()` | Dead pinned-staging subsystem used only by `uploadBatchToDevice()`; batch upload now copies `BatchPayload` host vectors directly into TrainingState device cache tensors. ScratchBlock reasoning owns its own layer buffers. |
| `Layers/GRIMTS/GRIM-TS.{hpp,cu}`, `Layers/GRIMTS/GuessCacheTraining.{hpp,cu}`, `training/Phases/Startup/GuessCache/GuessCacheInit.{hpp,cu}`, `TrainingContext::guess_cache_*` | Guess cache / GRIM-TS training integration removed. Startup, Phase2, telemetry, and cleanup no longer route logits or state through a dead speculative-cache subsystem. |
| `TrainingState::num_heads`, `TrainingState::num_kv_heads` | Dead GQA architecture shadows; `LanguageModelConfig` owns authored/root dimensions, `HyperparameterGroupings.hpp` owns grouped encoder/startup views, and `GenerationState::KVCacheShape` owns allocated generation-cache geometry. |
| `TrainingState::MTPDiagnostics`, `TrainingState::mtp_diagnostics` | Moved to `Shared/MTP/MTPDiagnostics.hpp` and routed through `Autograd::LossResult` / `Training::BatchResult`; per-step MTP telemetry is not TrainingState-owned GPU resource state. |
| `DataLoadInputs` | Deleted local Phase1 data-load mirror; `LoadTrainingData()` consumes `TokenizerHP` / `DataLoadingHP` and reads `StartupConfig.max_seq_len` directly, while `DataInfo` remains only the durable GRMT/data summary fact stored on `TrainingContext`. |
| `Layers/ReasoningHead/reasoning_head_GPU.{hpp,cu}`, `Forward::ReasoningHeadOutput`, `ParamGroupType::REASONING_HEAD` | Dead architecture. ExecutionBlock owns live structured reasoning, ScratchBlock owns atom detection/injection, and the old ReasoningHead had no active training loss or runtime path when execution was enabled. Do not recreate atom-op logits as a parallel subsystem. |
