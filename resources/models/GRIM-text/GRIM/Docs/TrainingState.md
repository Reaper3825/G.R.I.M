# TrainingState — Training GPU Resource Controller

Training-owned GPU resources MUST go through `TrainingContext::training_state` / `TrainingContext::requireTrainingState(...)`, except the explicit batch-upload boundary now owned by `BatchPayload`/`BatchDeviceStorage`. `TrainingState` is no longer owned by `LanguageModel` and there is no `LanguageModel::getTrainingState()` tunnel; startup, Phase2, diagnostics, checkpointing, telemetry, and optimizer helpers must consume the direct `TrainingContext` owner or an explicit `TrainingState&` parameter. Generation-owned GPU resources live in `TrainingContext::generation_state` / `TrainingContext::requireGenerationState(...)`, not behind `LanguageModel`, and are not smuggled through TrainingState.

`TrainingState` owns only training runtime state that is truly training-owned: stream/cuBLAS wrappers, execution-runtime diagnostics, class weights, and the read-gate telemetry workspace. The shared-forward sink is no longer a `TrainingState` or `GenerationState` field; each shared-forward call returns its own explicit `ModelForwardOutputs` owner, and Phase2 clears that Category 1 object at the forward/loss/backward or forward/sample boundary. The autograd loss root is also no longer TrainingState-owned; each training batch creates its own explicit `AutogradLossState` sibling, and Phase2 `processBatch()` clears that sibling through its local step-state guard. It no longer owns per-batch token/target upload buffers. `Startup/Model/ModelGpuAssembly.cu` now brings up only the TrainingState-owned runtime pieces (`read_gate_accum_tensor`, stream/cuBLAS readiness, PBM readiness). Fixed-shape batch upload capacity is still authored on config, but the explicit reusable device owner for that boundary is `Shared/Batching/BatchDeviceStorage`, attached to `BatchPayload`, not `TrainingState`. Inference startup is not a manual CUDA/model/bootstrap path and is not owned by `grim_text_server`; `train_gpu --inference` calls `executePhase1(...INFERENCE)`, owns the returned `TrainingContext`, and exposes Phase2 inference through the internal worker endpoint.

Trainable parameter verification is not a `TrainingState` responsibility. `Startup/Model/ParameterGroupRegistration.{hpp,cu}` runs after `GRIMText::Training::Startup::assembleGpuModel(config, training_state, gpu_model_state, parameter_registry, weight_init_seed)` in the main startup path and is the single registration-time gate for layer-owned parameter tensors: non-null data, non-null grad buffer, non-empty shape, config-disabled tensor absence, and duplicate data-alias detection. The adjacent `Startup/Model/ParameterRegistry.hpp` declares the durable writable-parameter owners (embedding, LM head, encoder, FFN, execution block, selector, and per-layer LoRA pairs) plus their header-only parameter-group inventory slices; that registry is tensor-only. In LoRA mode, registration requires every base tensor to be frozen with no gradient allocation and builds an adapter-only inventory with class-specific FP32 precision, zero weight decay, and one `A`/`B` pair per enabled class per layer. The `.cu` remains the only place that performs registration-time validation and writes the durable `ParameterGroup` inventory. Encoder topology stays on `GpuModelState`, while parameter bundles live on `TrainingContext::parameter_registry`; do not add individual embedding/LM-head/encoder weight checks to the startup runtime allocator.

`TrainingState` owns two different resource classes:

- **Tensor members** (`class_weights_tensor` and `read_gate_accum_tensor`) release themselves through `Tensor::~Tensor()`. Per-call shared-forward tensors live on the returned `ModelForwardOutputs` owner, not on `TrainingState`. Never call `cudaFree` on `Tensor::data` from the destructor.
- **RAII non-Tensor members** release themselves through their own destructors: `TeacherLogits::Buffer`, `std::unique_ptr<GradClip::ClipScratch>`, and `CublasHandleOwner`.

`TrainingState::~TrainingState()` is defaulted. Do not add a central destructor cleanup list; resource ownership must live on the field type itself.

RAII helper modules:
- `Shared/TrainingState/CublasHandleOwner_GPU.{hpp,cu}` owns `cublasDestroy` for `TrainingState::cublas_handle`.
- `CublasHandleOwner` keeps the raw handle private. Use `outParam()` only at `cublasCreate` sites and `.get()` at every raw `cublasHandle_t` borrow site; do not pass the owner object itself across runtime payload boundaries.
- `Shared/TrainingState/DeviceAllocation_GPU.{hpp,cu}` owns raw CUDA device allocations used by typed runtime owners, including `GenerationState` KV/decode buffers.

`ScratchBlockPool` was deleted. Batch upload copies `BatchPayload` host vectors directly into the payload-attached `BatchDeviceStorage` owner. ScratchBlock's remaining layer-local buffers are transient implementation staging only; they are not semantic data owners.

`ScratchBlockLayer` itself is **not** a TrainingState allocation. It is durable startup-owned model topology on `TrainingContext::gpu_model` (`Startup::GpuModelState`) and is assembled in `GRIMText::Training::Startup::assembleGpuModel(config, training_state, gpu_model_state, parameter_registry, weight_init_seed)` from `HyperParameters::scratchBlockConstructionHP()` plus the explicit startup init stream. The startup runtime allocator must not call `std::make_unique<ScratchBlockLayer>()`, hand-copy `ScratchBlockConfig`, or reset the startup-owned layer pointer.

Current teardown note: the startup-owned layer classes / tensor-owner bundles still marked for removal are (`GPUGrimEncoder`, `EmbeddingLayer`, `LMHeadLayer`, `ScratchBlockLayer`, `MtpHeadParameterTensors`). `ExecutionBlockLayer` is already deleted; execution-block state now flows through registry-owned tensors plus `ModelForwardExecutionRuntime::execution_diag`. `TrainingState` must not absorb the remaining classes as a rescue refactor, and new runtime payloads must not grow fresh back-pointers to keep those classes alive. The decode-time slot selector subsystem is deleted entirely (see Docs/DeletedCode.md). If a path still needs data from one of these classes during the deletion window, pass the exact tensors/runtime facts explicitly and keep shrinking the compatibility surface.

`read_gate_accum_tensor` is TrainingState-owned Category 3 workspace and must come online during startup runtime allocation through `TrainingState::allocateReadGateWorkspace()`, not lazily inside any autograd helper and not anywhere in parameter registration. It is a fixed `[2]` device buffer for execution-block read telemetry: `[sum_of_gate_values, token_count]`.

Per-batch token/target/numeric/atom/slot-map upload buffers are no longer TrainingState fields. They live on `Shared/Batching/BatchDeviceStorage`, and `Phase1` attaches one shared owner across planned train/val payloads while ad hoc inference payloads can attach their own owner locally. `Batching::uploadBatchToDevice()` copies host payload arrays into that explicit owner and returns `BatchDeviceBindings`, the only reader-facing device view for the step. CE scalar reduction scratch is graph-owned by `NLLLossGradFn::capture_inputs()` and released by `NLLLossGradFn::release_saved()`; it must not be added to `TrainingState` or Phase1 startup state. LM-head input and logits are not TrainingState caches; diagnostics/inference must consume the explicit live `ModelForwardOutputs` returned by shared forward before that per-call owner is cleared.

`TrainingState` does not own GQA architecture dimensions. Use `LanguageModelConfig` for authored/root model dimensions, `HyperparameterGroupings.hpp` grouped views for encoder/startup consumers, and `GenerationState::KVCacheShape` for allocated generation-cache geometry. Encoder files must pass the grouped HP snapshot directly instead of constructing local GQA dimension shadows.


| Resource | Access |
|----------|--------|
| CUDA streams | `ctx.requireTrainingState("caller").stream_ctrl.getPrimaryStream()` or an explicit `training_state.stream_ctrl.getPrimaryStream()` parameter borrow |
| cuBLAS handle | `training_state.cublas_handle` (`CublasHandleOwner`; create with `outParam()`, borrow as raw `cublasHandle_t` through `.get()`) |
| Parameter gradients | `Tensor.grad_` via registered `ParameterGroup` tensors; zero with `zeroParameterGradients(parameter_registry.requireParameterGroups(...), stream)` at the accumulation-window boundary |
| Optimizer states | Not TrainingState-owned; `Training::OptimizerContext::optimizer_state` owns Adam/RAdam moment tensors and `Startup/Model/ParameterGroupRegistration` binds them to `StartupParameterRegistry` parameter groups. |

Phase2 inference session state is separate from `TrainingState`:

| Resource | Access |
|----------|--------|
| Session KV cache | `ctx.requireGenerationState("caller").kv_cache` |

Phase2 owns token-by-token generation chronology. `LanguageModel` does not own a generation session, KV cursor, KV cache, or single-token decode scratch; `training/Phases/Phase2_InferenceLoop.cu` explicitly builds its read-only `ModelForwardRequest` / `ModelForwardRuntimePayload`, calls `Forward::executeModelForward(...)`, and samples from the caller-authored `BatchPayload` objects it rebuilds each step.

`TrainingState` does **not** own activation/intermediate gradient lifecycle. The tape-based autograd system owns those through `Tensor.grad_` and `GradFn` scratch buffers inside the autograd boundary. LM-head logits are non-leaf tensors; `LogSoftmaxGradFn` allocates its non-leaf input-gradient workspace and passes that view directly to the upstream logits `GradFn`. Do not add TrainingState mirrors for logits gradients.

Phase1/Phase1Startup hands off durable state before the graph is created: authored capacity/config, startup layer allocations, registered parameter groups and persistent parameter grad buffers, payload-attached/shared `BatchDeviceStorage` owners, class-weight storage ownership, stream/cuBLAS owners, and the grouped `LossConfigHP`. Per-step autograd graph tensors and GradFn saved buffers are created later by Phase2 from the prepared payload and must be released at the tape boundary. Full-vocab GradFn saved buffers and scalar CE reduction scratch are Category 1; `NLLLossGradFn` owns the CE scratch for the active graph window so it must not be hidden in `TrainingState`, `TrainingContext`, `forward_outputs`, `AutogradLossState`, or local loss guards.

`BatchPayload` owns host-side batch semantics and carries the explicit `BatchDeviceStorage` owner for upload backing. `Batching::uploadBatchToDevice()` is implemented in `Shared/Batching/BatchDeviceUpload.cu` and copies that payload into its attached storage, then returns `BatchDeviceBindings`, which is the canonical per-step device view consumed by forward/loss. This upload code is runtime batching glue, not `Startup/Model` work: startup assembles durable layers, while upload runs once per train/eval/inference payload. `Batching::uploadBatchToDevice()` is not a lazy bootstrap path for training; Phase1 attaches the shared storage owner up front for planned train/val payloads. `BatchDeviceBindings.d_target_ids` is the loss-facing target mirror. Target masking in loss is not an activation mask. Training/eval code must not infer the current batch by directly reading any global TrainingState cache because that path no longer exists. Inference prompt ingestion now enters through `Batching::buildInferenceBatchPayload()` (`BatchPayloadMode::InferencePrefill`) using tokenizer-authored token IDs, numeric values, atom masks/flags, and atom entry IDs; inference payloads materialize an all-`-1` slot map and never accept execution bindings. Phase2 inference owns the generation loop and repeatedly uploads that payload, authors a read-only `ModelForwardGraphPolicy`, calls `Shared/Forward/ModelForward_GPU.hpp`, and consumes the returned `ModelForwardOutputs` only inside that call window. There is no model-owned `primeGenerationSession()` / `continueGenerationSession()` path and no separate single-token decode graph. Inference must not enter `AutogradContext` or any training-only forward adapter.

Streams and cuBLAS are borrowed into per-call runtime payloads (`AutogradContext`, `Forward::ModelForwardRequest`, `Forward::ModelForwardRuntimePayload`, decode policy calls). Layers must not keep forward-time copies of those handles in config/member state; startup construction may use a clearly named init stream only for allocation/initialization. Do not respond to the layer-class removal work by moving these borrowed runtime handles onto replacement singleton objects; the direct-handoff destination is explicit per-call payloads, not a new god-object.

`Shared/Forward/ModelForwardRuntimePayload.hpp` reserves the mutable runtime boundary for shared forward. `Forward::ModelForwardRequest` carries the immutable model/input side, and `executeModelForward(...)` returns the per-call `ModelForwardOutputs` owner directly.

Inference session state is **not** `TrainingState`: `GenerationState` owns only the KV cache reset by Phase2 inference. The live shared-forward sink is the returned `ModelForwardOutputs` object for the current call window, not durable session state.


Weight initialization seeds are **not** TrainingState-owned. Phase 1 owns the RNG hierarchy and passes `ctx.rng.init_seed` directly into `GRIMText::Training::Startup::assembleGpuModel(ctx.config, ctx.requireTrainingState("ModelAllocated"), ctx.gpu_model, ctx.parameter_registry, weight_init_seed)`, where Pattern B layers self-allocate weights with deterministic offsets.

**Violations are bugs:**
- ❌ Raw `cudaStream_t` locals
- ❌ Separate cuBLAS handles
- ❌ Batch upload buffers owned by `TrainingState`
- ❌ Generation allocations owned by anything other than `GenerationState`

See [Autograd.md](Autograd.md) for the ownership taxonomy that governs which buffers belong here vs. inside the autograd tape.

## Key file
`resources/models/GRIM-text/Shared/TrainingState/TrainingState_GPU.hpp`
