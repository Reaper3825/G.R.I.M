# TrainingState — Training GPU Resource Controller

Training-owned GPU resources MUST go through `TrainingState`. Generation-owned GPU resources live in `GenerationState` (`Shared/InferenceState/GenerationState_GPU.hpp`) and are owned directly by `LanguageModel`, not smuggled through TrainingState. GRIM-TS guess-cache buffers are owned by `GRIMTS::Training::GuessCacheScope`, not `TrainingState`; that scope only borrows `TrainingState.stream_ctrl` for the primary stream.

`TrainingState` owns two different resource classes:

- **Tensor members** (`Tensor cached_*`, optimizer states, `class_weights_tensor`, `read_gate_accum_tensor`) release themselves through `Tensor::~Tensor()`. Never call `cudaFree` on `Tensor::data` from the destructor.
- **RAII non-Tensor members** release themselves through their own destructors: `TeacherLogits::Buffer`, `std::unique_ptr<GradNormScratch>`, and `CublasHandleOwner`.

`TrainingState::~TrainingState()` is defaulted. Do not add a central destructor cleanup list; resource ownership must live on the field type itself.

RAII helper modules:
- `Shared/TrainingState/CublasHandleOwner_GPU.{hpp,cu}` owns `cublasDestroy` for `TrainingState::cublas_handle`.
- `Shared/TrainingState/DeviceAllocation_GPU.{hpp,cu}` owns raw CUDA device allocations used by typed runtime owners, including `GenerationState` KV/decode buffers.
- `Layers/GRIMTS/GuessCacheTraining.{hpp,cu}` owns GRIM-TS guess-cache records, keys, bloom, calibration, and pinned async-transfer buffers through `GuessCacheScope::OwnedBuffers`.

`ScratchBlockPool` was deleted. Batch upload copies `BatchPayload` host vectors directly into the TrainingState device cache tensors, and ScratchBlock reasoning owns its own layer buffers.

`TrainingState` does not own GQA architecture dimensions. Use `LanguageModelConfig` / `ModelArchitecture` for authored `num_heads`, `num_kv_heads`, and `head_dim`; pass `TensorContract::GQADims` at kernel/layer boundaries; use `GenerationState::KVCacheShape` for allocated generation-cache geometry.

`TrainingState` does not own MTP diagnostics. `Shared/MTP/MTPDiagnostics.hpp` defines the host-side payload, `Autograd::LossResult` carries the producer snapshot, and `Training::BatchResult` carries the log-interval snapshot consumed by `Diagnostics::runMtpDiagnostic()`.

| Resource | Access |
|----------|--------|
| CUDA streams | `training_state.stream_ctrl.getPrimaryStream()` |
| cuBLAS handle | `training_state.cublas_handle` (`CublasHandleOwner`, borrowed as raw `cublasHandle_t`) |
| Parameter gradients | `Tensor.grad_` via `ctx.model->zeroGradients()` / `ctx.model->backward()` |
| Optimizer states | Not TrainingState-owned; `Training::OptimizerContext::optimizer_state` owns Adam/RAdam moment tensors and `LanguageModel::bindOptimizerState()` binds them to parameter groups. |

Generation state is a separate owner:

| Resource | Access |
|----------|--------|
| Autoregressive KV cursor | `generation_state_.kv_cache_len` |
| BF16 per-layer K/V cache | `generation_state_.kv_cache.k` / `.v` |
| KV capacity / geometry | `generation_state_.kv_cache.shape` |
| FlashAttention decode LSE scratch | `generation_state_.kv_cache.softmax_lse` |
| Single-token decode scratch | `generation_state_.decode_scratch` |

Training-time sampling must call `LanguageModel::ensureKVCacheAllocated()` explicitly. `initTrainingState()` does not seed generation capacity or allocate decode buffers.

`TrainingState` does **not** own activation/intermediate gradient lifecycle. TensorContract owns those through `Tensor.grad_` and `GradFn` scratch buffers inside the autograd boundary. LM-head logits are non-leaf tensors; `LogSoftmaxGradFn` allocates its non-leaf input-gradient workspace and passes that view directly to the upstream logits `GradFn`. Do not add TrainingState mirrors for logits gradients.

The `cached_token_*` / `cached_targets_tensor` fields are reusable device storage only. `BatchPayload` owns host-side batch semantics and geometry; `LanguageModel::uploadBatchToDevice()` copies that payload into TrainingState-owned device buffers and returns `BatchDeviceBindings`, which is the canonical per-step device view consumed by forward/loss. Training/eval code must not infer the current batch by directly reading these cache fields. Phase-2 inference still writes its single-sequence data into the same reusable storage as temporary backing, but `LanguageModel::executeInferenceForward_()` now builds an explicit per-call `BatchDeviceBindings` view plus a geometry-only `BatchPayload`, then calls `Shared/Forward/ModelForward_GPU.hpp` with `ModelForwardMode::InferencePrefill`; inference must not enter `AutogradContext` or `executeAutogradForward()`.

Do not add shared MTP shifted-target buffers to `TrainingState`. Each MTP head owns a distinct shifted-target tensor in `AutogradIntermediates` because `NLLLossGradFn` stores raw target pointers through backward; a shared reusable buffer causes all heads to read the last uploaded horizon.

Weight initialization seeds are **not** TrainingState-owned. Phase 1 owns the RNG hierarchy and passes `ctx.rng.init_seed` directly into `LanguageModel::initGPU(weight_init_seed)`, where Pattern B layers self-allocate weights with deterministic offsets.

**Violations are bugs:**
- ❌ Raw `cudaStream_t` locals
- ❌ Separate cuBLAS handles
- ❌ Training allocations owned by anything other than `TrainingState`
- ❌ Generation allocations owned by anything other than `GenerationState`

See [Autograd.md](Autograd.md) for the ownership taxonomy that governs which buffers belong here vs. inside the autograd tape.

## Key file
`resources/models/GRIM-text/Shared/TrainingState/TrainingState_GPU.hpp`
