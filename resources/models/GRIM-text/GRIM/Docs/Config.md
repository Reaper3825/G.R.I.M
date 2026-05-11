# Configuration (`ai_config.json`)

All model and training configuration lives in `ai_config.json`.

Typed config ownership is split into exactly two code locations:

- `Shared/HyperParameters/HyperParameters_GPU.hpp` owns authored hyperparameter/config structs, constants, JSON loading, derivation, and validation.
- `Shared/HyperParameters/HyperparameterGroupings.hpp` owns grouped read views used by phase code. It must not include or call `ai_config_paths.hpp` directly, and it must not own authored defaults/constants; it may only slice/derive from `HyperParameters_GPU.hpp` structs and constants. Startup model construction uses `startupLanguageModelConfig()`, and startup tensor registration uses `parameterRegistrationHP()`.

`startupLanguageModelConfig()` is also the single policy boundary that stamps training startup models with `ModelExecutionMode::TRAINING`. Do not rely on the `LanguageModelConfig` default (`INFERENCE`) in training code; training `BatchPayload` upload is forbidden for inference-mode models.

Phase/startup code should consume these grouped views instead of hand-copying scattered config fields. If a new startup subsystem needs a repeated slice of config, add that view to `HyperparameterGroupings.hpp`; if it needs a new authored field/default/constant, add that to `HyperParameters_GPU.hpp` first. Do not create a third config owner.

Grouped construction views also own repeated derived dimensions for their consumers. For encoder construction, `EncoderLayerConstructionHP` carries validated GQA/QKV dimensions (`head_dim`, `heads_per_kv_group`, `kv_dim`, `qkv_dim`, `is_gqa`) so layer code does not recompute config geometry locally.

PBM positional encoding construction uses `PBMConstructionHP` from `HyperparameterGroupings.hpp`. `Shared/PBM/PositionalBiasMethod.*` consumes that grouped view plus explicit runtime options only; it must not define its own authored defaults or hand-copy `LanguageModelConfig` fields.

Encoder-facing autograd helpers consume grouped attention snapshots sliced from `EncoderLayerConstructionHP` (`EncoderSelfAttentionHP`). Encoder files must not build local `TensorContract::GQADims` wrappers or store encoder-owned GQA snapshots; the grouped HP snapshot is the source for Q/K/V geometry.

Layer classes may store a durable copy of their grouped construction view when forward/runtime methods need the values after startup-local grouping objects are destroyed. Name those members as HP/grouping snapshots (for example `hp_`), not as authored config owners, and keep borrowed tensor ownership outside the grouping. Forward-time runtime handles (`cudaStream_t`, `cublasHandle_t`) belong to per-call payload/request structs, not layer configs or late mutator methods; startup constructors may take an init stream only for allocation.

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
Validation MUST use `ctx.model->getConfig().max_tokens_per_batch`, never a hardcoded constant — buffer overflow crash otherwise.

## `per_token_grad_scale=true`
Required. See [Encoder.md](Encoder.md).

## Loss config
`training.config.loss` — see [Loss.md](Loss.md).

## LM head centering
`training.config.lm_head_centering.freeze_final_rms_gamma=true` — see [LMHead.md](LMHead.md).
