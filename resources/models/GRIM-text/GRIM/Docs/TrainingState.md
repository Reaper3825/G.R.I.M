# TrainingState — Training GPU Resource Controller

Training-owned GPU resources MUST go through `TrainingState`. Generation-owned GPU resources live in `GenerationState` (`Shared/InferenceState/GenerationState_GPU.hpp`) and are owned directly by `LanguageModel`, not smuggled through TrainingState.

`TrainingState` owns runtime allocations only. Cache capacity is authored on `LanguageModelConfig` by Phase1 startup config construction, but the runtime owner must not accept raw config. `LanguageModel::initTrainingState()` and `LanguageModel::initInferenceState()` verify resource startup order, slice `HyperParameters::TrainingStateWorkspaceHP` from the finalized root config, and pass only that explicit workspace grouping plus the primary stream to `TrainingState::allocateStepDeviceWorkspaces()`. `max_cached_seq_len` belongs to prompt/KV/payload capacity paths, not the TrainingState token-cache boundary, and root config validation stays on the caller-owned config side rather than inside the TrainingState allocator. Inference startup is not a manual CUDA/model/bootstrap path and is not owned by `grim_text_server`; `train_gpu --inference` calls `executePhase1(...INFERENCE)`, owns the returned `TrainingContext`, and exposes Phase2 inference through the internal worker endpoint.

Trainable parameter verification is not a `TrainingState` responsibility. `Startup/Model/ParameterGroupRegistration.{hpp,cu}` runs after `LanguageModel::initGPU(weight_init_seed)` in the main startup path and is the single registration-time gate for layer-owned parameter tensors: non-null data, non-null grad buffer, non-empty shape, config-disabled tensor absence, and duplicate data-alias detection. Do not add individual embedding/LM-head/encoder weight checks to `initTrainingState()`.

`TrainingState` owns two different resource classes:

- **Tensor members** (`Tensor cached_*`, optimizer states, `class_weights_tensor`, `read_gate_accum_tensor`) release themselves through `Tensor::~Tensor()`. Never call `cudaFree` on `Tensor::data` from the destructor.
- **RAII non-Tensor members** release themselves through their own destructors: `TeacherLogits::Buffer`, `std::unique_ptr<GradNormScratch>`, and `CublasHandleOwner`.

`TrainingState::~TrainingState()` is defaulted. Do not add a central destructor cleanup list; resource ownership must live on the field type itself.

RAII helper modules:
- `Shared/TrainingState/CublasHandleOwner_GPU.{hpp,cu}` owns `cublasDestroy` for `TrainingState::cublas_handle`.
- `CublasHandleOwner` keeps the raw handle private. Use `outParam()` only at `cublasCreate` sites and `.get()` at every raw `cublasHandle_t` borrow site; do not pass the owner object itself across runtime payload boundaries.
- `Shared/TrainingState/DeviceAllocation_GPU.{hpp,cu}` owns raw CUDA device allocations used by typed runtime owners, including `GenerationState` KV/decode buffers.

`ScratchBlockPool` was deleted. Batch upload copies `BatchPayload` host vectors directly into the TrainingState device cache tensors, and ScratchBlock reasoning owns its own layer buffers.

`ScratchBlockLayer` itself is **not** a TrainingState allocation. It is durable model topology owned by `LanguageModel` and assembled in `LanguageModel::initGPU(weight_init_seed)` from `HyperParameters::scratchBlockConstructionHP()` plus the explicit startup init stream. `initTrainingState()` must not call `std::make_unique<ScratchBlockLayer>()`, hand-copy `ScratchBlockConfig`, or reset `scratch_block_layer_`.

`read_gate_accum_tensor` is TrainingState-owned Category 3 workspace and must come online during `initTrainingState()` / `initInferenceState()` through `TrainingState::allocateStepDeviceWorkspaces()`, not lazily inside any autograd helper and not anywhere in parameter registration. It is a fixed `[2]` device buffer for execution-block read telemetry: `[sum_of_gate_values, token_count]`.

The `cached_targets_tensor`, `cached_mtp_shifted_targets_tensor`, `cached_token_*`, `cached_seq_lengths_tensor`, and `sequence_weights_tensor` allocations live in `Shared/TrainingState/TrainingStateGPU.cu` under `TrainingState::allocateStepDeviceWorkspaces()`. They are TrainingState-owned Category 3 runtime substrate: per-step device staging with stale contents across boundaries. `initTrainingState()` and `initInferenceState()` must not hand-allocate those fields directly; they should only pass `HyperParameters::TrainingStateWorkspaceHP` and the stream to the TrainingState owner. CE scalar reduction scratch is graph-owned by `NLLLossGradFn::capture_inputs()` and released by `NLLLossGradFn::release_saved()`; it must not be added to `TrainingState` or Phase1 startup state. LM-head input and logits are not TrainingState caches; diagnostics/inference must consume explicit live forward outputs before `AutogradIntermediates::clear()`.

`TrainingState` does not own GQA architecture dimensions. Use `LanguageModelConfig` for authored/root model dimensions, `HyperparameterGroupings.hpp` grouped views for encoder/startup consumers, and `GenerationState::KVCacheShape` for allocated generation-cache geometry. Encoder files must pass the grouped HP snapshot directly instead of constructing local GQA dimension shadows.

`TrainingState` does not own MTP diagnostics. `Shared/MTP/MTPDiagnostics.hpp` defines the host-side payload, `Autograd::LossResult` carries the producer snapshot, and `Training::BatchResult` carries the log-interval snapshot consumed by `Diagnostics::runMtpDiagnostic()`.

| Resource | Access |
|----------|--------|
| CUDA streams | `training_state.stream_ctrl.getPrimaryStream()` |
| cuBLAS handle | `training_state.cublas_handle` (`CublasHandleOwner`; create with `outParam()`, borrow as raw `cublasHandle_t` through `.get()`) |
| Parameter gradients | `Tensor.grad_` via registered `ParameterGroup` tensors; zero with `zeroParameterGradients(model.parameterGroups(), stream)` at the accumulation-window boundary |
| Optimizer states | Not TrainingState-owned; `Training::OptimizerContext::optimizer_state` owns Adam/RAdam moment tensors and `Startup/Model/ParameterGroupRegistration` binds them to `LanguageModel` parameter groups. |

Phase2 inference session state is separate from `TrainingState`:

| Resource | Access |
|----------|--------|
| Persistent inference execution memory | `generation_state_.exec_memory` / `has_exec_memory` |
| Decode-time selector result | `generation_state_.decode_selector` |
| Decode execution trace state | `generation_state_.execution_trace_by_row` / `trace_state_by_row` |

Phase2 owns token-by-token generation chronology. `LanguageModel` does not own a generation session, KV cursor, KV cache, or single-token decode scratch; `training/Phases/Phase2_InferenceLoop.cu` owns the read-only scorer (`scoreInferencePrefillLogits(...)`) that evaluates the caller-authored `BatchPayload` objects Phase2 passes through shared forward.

`TrainingState` does **not** own activation/intermediate gradient lifecycle. The tape-based autograd system owns those through `Tensor.grad_` and `GradFn` scratch buffers inside the autograd boundary. LM-head logits are non-leaf tensors; `LogSoftmaxGradFn` allocates its non-leaf input-gradient workspace and passes that view directly to the upstream logits `GradFn`. Do not add TrainingState mirrors for logits gradients.

Phase1/Phase1Startup hands off durable state before the graph is created: authored capacity/config, startup layer allocations, registered parameter groups and persistent parameter grad buffers, `TrainingState` cache tensors, class-weight storage ownership, stream/cuBLAS owners, and the grouped `LossConfigHP`. Per-step autograd graph tensors and GradFn saved buffers are created later by Phase2 from the prepared payload and must be released at the tape boundary. Full-vocab GradFn saved buffers and scalar CE reduction scratch are Category 1; `NLLLossGradFn` owns the CE scratch for the active graph window so it must not be hidden in `TrainingState`, `TrainingContext`, `AutogradIntermediates`, or local loss guards.

The `cached_token_*` / `cached_targets_tensor` / `cached_mtp_shifted_targets_tensor` / `cached_seq_lengths_tensor` fields are reusable device storage only. `BatchPayload` owns host-side batch semantics and geometry; `LanguageModel::uploadBatchToDevice()` is implemented in `Shared/Batching/BatchDeviceUpload.cu` and copies that payload into TrainingState-owned device buffers, then returns `BatchDeviceBindings`, which is the canonical per-step device view consumed by forward/loss. This upload code is runtime batching glue, not `Startup/Model` work: startup assembles durable layers, while upload runs once per train/eval/inference payload. `uploadBatchToDevice()` is not a lazy bootstrap path; callers must arrive with runtime already initialized. `BatchDeviceBindings.d_seq_lengths` is required for padding-aware residual and LM-head hidden centering; `BatchDeviceBindings.d_target_ids` and `d_mtp_shifted_targets` are the only loss-facing target mirrors. Target masking in loss is not an activation mask. Training/eval code must not infer the current batch by directly reading these cache fields. Inference prompt ingestion now enters through `Batching::buildInferenceBatchPayload()` (`BatchPayloadMode::InferencePrefill`) using tokenizer-authored token IDs, numeric values, atom masks/flags, atom entry IDs, and slot maps; Phase2 inference owns the generation loop and repeatedly calls `scoreInferencePrefillLogits(const BatchPayload&)`, which uploads that payload through the same explicit sync boundary before calling `Shared/Forward/ModelForward_GPU.hpp` with a read-only `ModelForwardGraphPolicy`. There is no model-owned `primeGenerationSession()` / `continueGenerationSession()` path and no separate single-token decode graph. Inference must not enter `AutogradContext` or any training-only forward adapter.

Streams and cuBLAS are borrowed into per-call runtime payloads (`AutogradContext`, `Forward::ModelForwardRequest`, `Forward::ModelForwardRuntimePayload`, decode policy calls). Layers must not keep forward-time copies of those handles in config/member state; startup construction may use a clearly named init stream only for allocation/initialization.

`Shared/Forward/ModelForwardRuntimePayload.hpp` owns the mutable runtime sink boundary for shared forward. `Forward::ModelForwardRequest` carries the immutable model/input side, while `executeModelForward(...)` receives `ModelForwardRuntimePayload` directly as a sibling payload. Training callers fill that runtime payload from `TrainingState`; Phase2's `scoreInferencePrefillLogits(...)` fills it from `GenerationState` plus `AutogradIntermediates`. Shared forward must write only those explicit payload sinks.

Inference session state is **not** `TrainingState`: persistent execution memory, decode trace vectors, and decode-time selector result are `GenerationState` fields reset by Phase2 inference. Training's `execution_trace_by_row` / `trace_state_by_row` are training/eval forward diagnostics only.

MTP shifted targets are TrainingState-owned Category 3 upload workspace, not AutogradIntermediates. `cached_mtp_shifted_targets_tensor` is head-major (`[mtp_k, max_tokens]`) so every MTP head receives a stable distinct slice through backward. Never reuse a single per-head scratch target buffer inside `computeAutogradMtpAuxiliaryLosses()`; the upload boundary must provide all head slices through `BatchDeviceBindings.d_mtp_shifted_targets` before loss assembly begins, while `mtp_k` remains payload/model-config semantics. The shared `MTP_GPU.cu` file owns only the accuracy kernel and must not reach into TrainingState, AutogradContext, or LM-head representation policy.

Weight initialization seeds are **not** TrainingState-owned. Phase 1 owns the RNG hierarchy and passes `ctx.rng.init_seed` directly into `LanguageModel::initGPU(weight_init_seed)`, where Pattern B layers self-allocate weights with deterministic offsets.

**Violations are bugs:**
- ❌ Raw `cudaStream_t` locals
- ❌ Separate cuBLAS handles
- ❌ Training allocations owned by anything other than `TrainingState`
- ❌ Generation allocations owned by anything other than `GenerationState`

See [Autograd.md](Autograd.md) for the ownership taxonomy that governs which buffers belong here vs. inside the autograd tape.

## Key file
`resources/models/GRIM-text/Shared/TrainingState/TrainingState_GPU.hpp`
