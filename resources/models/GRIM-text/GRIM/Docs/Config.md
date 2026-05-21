# Configuration (`ai_config.json`)

All model and training configuration lives in `ai_config.json`.

Typed config ownership is split into exactly two code locations:

- `Shared/HyperParameters/HyperParameters_GPU.hpp` owns authored hyperparameter/config structs, constants, JSON loading, derivation, and validation.
- `Shared/HyperParameters/HyperparameterGroupings.hpp` owns grouped read views used by phase code. It must not include or call `ai_config_paths.hpp` directly, and it must not own authored defaults/constants; it may only slice/derive from `HyperParameters_GPU.hpp` structs and constants. Phase 1 authors exactly one static model config set (`TrainingContext.model_config`) via `startupLanguageModelConfig()`; model construction receives it once, `LanguageModel` stores its own copy, and Phase 2 consumes `ctx.model_config` directly. Do not rebuild, forward-declare, or route alternate static model config wrappers through registration/training paths. Server/runtime inference uses `inferenceLanguageModelConfig()`.

`HyperparameterGroupings.hpp` must never be wrapped, mirrored, or forward-declared from consumer files. If a subsystem needs a grouping type, include `HyperparameterGroupings.hpp` and accept the explicit grouping payload. If a subsystem already receives the actual config (`LanguageModelConfig`, `TrainingHyperparameters`, or `StartupConfig`), do not pass a second sidecar policy/config beside it. Put authored fields on the actual config owner, slice them through an existing grouping only when needed, and have the consumer read one authoritative source. Config access helpers/facades must live only in `HyperParameters_GPU.hpp` or `HyperparameterGroupings.hpp`; every other file consumes typed config/grouping payloads passed to it instead of reaching through another module. This is especially important for `ParameterGroupRegistration`: precision/type metadata is stamped on registered `ParameterGroup` entries from the model's actual `LanguageModelConfig`, not from a parallel policy object.

`startupLanguageModelConfig()` is also the single policy boundary that stamps training startup models with `ModelExecutionMode::TRAINING`. `inferenceLanguageModelConfig()` is the matching boundary for server/runtime inference and stamps `ModelExecutionMode::INFERENCE` plus the single-row prompt/cache capacity. Do not rely on the `LanguageModelConfig` default (`INFERENCE`) in training code; training `BatchPayload` upload is forbidden for inference-mode models.

Startup TrainingState cache capacity is authored on `LanguageModelConfig` as the values the allocator actually consumes: `max_cached_batch` and `max_tokens_per_batch`. `validateLanguageModelCacheCapacity()` validates those consumed capacities when startup/inference configs are produced, and `TrainingState::allocateStepDeviceWorkspaces()` validates the same authored config before allocating reusable tensors. `max_cached_seq_len` belongs to prompt/KV/payload capacity paths, not the TrainingState token-cache validation path. Per-batch/prompt geometry belongs to `BatchPayload`, not cache-capacity groupings.

`BatchPayload` owns per-batch/per-prompt data and geometry: token IDs, targets, sequence lengths, numeric values, atom metadata, slot maps, and current valid-token counts. Config/grouping code owns static and computed startup values only; it must not absorb prompt contents or current batch facts.

Mixed precision is configured once at startup via `training.config.precision.parameter_groups`. These authored fields live on `LanguageModelConfig` as per-parameter-group precision values. `ParameterGroupRegistration` is the durable handoff: it validates the authored values, stamps each registered `ParameterGroup`, and stamps the owning `Tensor`'s TensorContract precision metadata. `LanguageModelConfig` does not own a model-wide precision mode, and there is no sidecar precision policy. Runtime TensorContract operators consume tensor metadata; raw FP32↔BF16 conversion kernels belong only in `Shared/TensorConversion`. A `bf16_compute` request must fail during registration until the matching TensorContract compute consumer is wired, never silently run as FP32.

Phase/startup code may consume grouped views for subsystem construction, loss, tokenizer, optimizer, and payload-specific contracts, but static model configuration itself is not wrapped: no `ModelArchitectureHP`, no `ParameterRegistrationHP`, and no alternate architecture mirror. If Phase2/diagnostics need static model fields, read `TrainingContext.model_config` directly. If a new startup subsystem needs a repeated non-static slice of config, add that view to `HyperparameterGroupings.hpp`; if it needs a new authored field/default/constant, add that to `HyperParameters_GPU.hpp` first. Do not create a third config owner. Loss runtime config follows this rule through `LossConfigHP` / `lossConfigHP()`: Phase 1 builds one durable `TrainingContext.loss_config` grouping immediately after hyperparameter validation, and Phase 2 passes that grouping explicitly to autograd/loss calls. `LossConfigHP` has no runtime `initialized` sentinel; validity belongs to the Phase1 `StartupConfig`/hyperparameter validation and grouping construction boundary. `TrainingLoopState`, `AutogradContext`, and `LanguageModel` must not store, rebuild, revalidate, or mutate loss config.

Tokenizer startup/build paths consume `TokenizerHP` / `tokenizerHP()` directly. `TokenizerHP` carries the resolved tokenizer artifact paths (`data_path`, `vocab_path`), rebuild policy (`force_rebuild_vocab`), and tokenizer settings; do not pass `StartupConfig.paths`, `PathConfig`, rebuild booleans, or a tokenizer-artifact path wrapper as a second tokenizer payload. `UniByte` stores that grouping as its tokenizer HP snapshot; `train_tokenizer`, `DataLoader`, `LoadTrainingData`, `tokenizer_runner`, and `tokenizer_self_test` must not hand-build a separate tokenizer config mirror from scattered `TokenizerConfig`, `TrainingHyperparameters`, and `PathConfig` fields. `LoadTrainingData()` consumes `DataLoadingHP` for data-loading policy and reads the configured `StartupConfig.max_seq_len` directly for sliding windows; do not reintroduce a local data-load input wrapper or a capacity grouping that just mirrors config fields. Isolated tokenizer tests construct explicit `TokenizerHP` fixtures instead of a wrapper struct.

`tokenizer.min_cleaned_text_length` owns the minimum rendered/cleaned text byte length required before `DataLoader.cu` encodes a concept/plaintext row into GRMT. Keep this threshold in `ai_config.json` and pass it through `TokenizerConfig` → `TokenizerHP`; do not reintroduce a file-local DataLoader constant.

`LanguageModel` owns only `LanguageModelConfig` plus runtime/model state. It must not retain `TrainingHyperparameters` pointers or expose pass-through accessors for Phase 1 config. After `TrainingContext.model_config` is authored, Phase2 and diagnostics must consume that handoff directly instead of reading `ctx.config.hyperparameters.architecture` or rebuilding static model groupings.

Grouped construction views also own repeated derived dimensions for their consumers. For encoder construction, `EncoderLayerConstructionHP` carries validated GQA/QKV dimensions (`head_dim`, `heads_per_kv_group`, `kv_dim`, `qkv_dim`, `is_gqa`) so layer code does not recompute config geometry locally.

PBM positional encoding construction uses `PBMConstructionHP` from `HyperparameterGroupings.hpp`. `Shared/PBM/PositionalBiasMethod.*` consumes that grouped view plus explicit runtime options only; it must not define its own authored defaults or hand-copy `LanguageModelConfig` fields.

Encoder-facing autograd helpers consume grouped attention snapshots sliced from `EncoderLayerConstructionHP` (`EncoderSelfAttentionHP`). Encoder files must not build local `TensorContract::GQADims` wrappers or store encoder-owned GQA snapshots; the grouped HP snapshot is the source for Q/K/V geometry.

Layer classes may store a durable copy of their grouped construction view when forward/runtime methods need the values after startup-local grouping objects are destroyed. Name those members as HP/grouping snapshots (for example `hp_`), not as authored config owners, and keep borrowed tensor ownership outside the grouping. Forward-time runtime handles (`cudaStream_t`, `cublasHandle_t`) belong to per-call payload/request structs, not layer configs or late mutator methods; startup constructors may take an init stream only for allocation.

ScratchBlock startup construction uses `ScratchBlockConstructionHP` / `scratchBlockConstructionHP()`. That grouping owns static values such as `d_model`, `scratch_block_max_atoms`, `scratch_block_atom_embedding_dim`, atom token range, and `scratch_block_atom_scale`. It must not own the CUDA stream; `ModelGpuAssembly.cu` supplies the startup init stream when constructing `ScratchBlockLayer`.

RMSNorm shape/epsilon travels through the same grouped construction HP as the owning layer (`EncoderLayerConstructionHP::rms_epsilon`, `LMHeadLayerConstructionHP::rms_epsilon`). Runtime CUDA streams and gamma tensor ownership stay outside the grouping.

Encoder dropout is owned by `LanguageModelConfig` in `HyperParameters_GPU.hpp`, sliced through `EncoderLayerConstructionHP`, and passed to FFN construction through `FeedForwardLayerConstructionHP`. Do not add layer-local dropout defaults or copy `dropout_rate` through ad-hoc config structs.

Config value logging is owned by `training/Phases/ConfigDump.{hpp,cu}`. Phase/startup modules may log lifecycle/status messages, but they MUST NOT print per-field config values (for example architecture dimensions, logging/tape knobs, telemetry lattice dimensions, or feature gate values) outside `ConfigDump`. If a value is useful in startup logs, add it to `ConfigDump` instead of emitting it at the consumer site.

## Fail-loud defaults (Rule 20)
Algorithmic config fields MUST default to `0` and throw if not loaded. Hardcoded defaults like `int max_seq_len = 512;` are forbidden — they hide silent fallbacks.

```cpp
int max_seq_len = 0;  // ✅
// ...load from JSON...
if (max_seq_len == 0)
    throw std::runtime_error("max_seq_len missing from ai_config.json");
```

## Validation token budget
Validation MUST use Phase1-authored config facts (`ctx.config.max_seq_len`, `ctx.config.hyperparameters.batch_size`, `ctx.model_allocation.model_max_tokens_per_batch`, or a non-duplicate explicit `HyperparameterGroupings.hpp` view), never the `LanguageModel` config accessor and never a hardcoded constant — buffer overflow crash otherwise. Batch scheduling consumes configured `max_seq_len` and `batch_size` directly; it must not derive sequence length from `max_tokens_per_batch` or a capacity wrapper.

## `per_token_grad_scale=true`
Required. See [Encoder.md](Encoder.md).

## Loss config
`training.config.loss` — see [Loss.md](Loss.md).

## LM head centering
`training.config.lm_head_centering.freeze_final_rms_gamma=true` — see [LMHead.md](LMHead.md).
