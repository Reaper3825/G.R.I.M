# Autograd / TensorContract

Implementation: `resources/models/GRIM-text/Shared/TensorContract_GPU.cu` (all `GradFn` structs), `resources/models/GRIM-text/training/Autograd/AutogradTraining.cu` (forward/backward orchestration), and small loss primitives under `resources/models/GRIM-text/training/Autograd/`.

## Prepared payload boundary
Phase1/Phase1Startup owns semantic batch construction. By the time Phase2 calls autograd, `BatchPayload` is already complete: token IDs, targets, masks, execution teacher steps, selector targets, numeric values, and slot maps are Phase1-authored data. Autograd code must consume this prepared payload and matching `BatchDeviceBindings`; it must not rebuild, infer, repair, or silently synthesize missing supervision.

Embedding lookup follows the same prepared-payload rule as CE/NLL loss: `autograd::embedding()` consumes `BatchPayload` plus `BatchDeviceBindings` and reads `bindings.d_input_ids` at the forward/backward launch boundary. `EmbeddingGradFn` must not copy or own token IDs, and it must not cache `d_model` as per-op state; backward derives token geometry from `BatchPayload` and the captured embedding weight shape.

`AutogradContext` and `Forward::ModelForwardRequest` must not mirror `batch_size`, `seq_len`/`max_seq_len`, or accumulation-window normalization policy. Batch geometry is read directly from the caller-owned `BatchPayload` (validated against `BatchDeviceBindings`). `executeAutogradBackward()` seeds backward with the default root gradient, while Phase2 owns any accumulation normalization later at the optimizer-window boundary.

Forward runtime handles are sibling payload data, not layer state: `AutogradContext` carries the `cudaStream_t` and `cublasHandle_t` borrowed from `TrainingState`, and `Forward::ModelForwardRequest` passes them through to encoder, FFN, LM head, execution-block, and selector forwards. Do not patch those handles into layer configs or mutate layers with late setter calls.

Dropout is a training-only TensorContract primitive. `autograd::dropout()` must not carry an identity/non-training branch, and inference/prefill code must not call it as a no-op wrapper. The training/inference boundary owns that policy: `TrainingGraph` may apply embedding/layer dropout, while `InferencePrefill` skips dropout calls entirely.

Static loss configuration is an explicit boundary payload, not Autograd runtime state. Phase2 derives `LossConfigHP` from authoritative `TrainingContext.config.hyperparameters` only at the explicit training loss boundary after `processBatch()` runs `Forward::executeModelForward(...)`, and passes both the required `const BatchPayload&` and `LossConfigHP&` into `computeAutogradLoss()` and the nested loss-assembly helpers. `AutogradContext` must not be the primary reader for batch semantics inside loss assembly, and it must not store, default-construct, assign, or revalidate loss config after that boundary. Runtime loss inputs such as class-balanced device weights remain separate `TrainingState` buffers.

The explicit Phase2 training path must not pass a sidecar `LanguageModelConfig` beside `LanguageModel& model`. `LanguageModel` already owns the authoritative startup `LanguageModelConfig` and autograd/forward read it through `model.getConfig()`. The only extra config payload allowed at the loss boundary is the explicit loss grouping that the model does not own.

Phase1/Phase1Startup also completes the durable handoff long before the autograd graph exists: model config/capacity, RNG/init seed, layer construction, registered parameter groups and their persistent grad buffers, payload-attached `BatchDeviceStorage` owners for the upload boundary, and stream/cuBLAS owners are already established before Phase2 builds any per-batch graph. Phase2 then derives the explicit loss grouping from `ctx.config.hyperparameters` before entering autograd. The graph-time objects created later (`TrainingState::forward_outputs.logits_tensor`, `LogSoftmaxGradFn` saved log-probs, `NLLLossGradFn`, CE scalar reduction scratch, per-op saved caches, and non-leaf gradient scratch) are Category 1 tape state; they must not recreate or reinterpret startup-authored semantics.



## Primitive extraction rule
Do not add new catch-all executors around `AutogradTraining.cu`. Extract narrow primitives first:
- A primitive has one reason to exist and one mutation target.
- A loss primitive may mutate `AutogradLossState::loss_tensor` plus Category 1 retained tensors only, then return host telemetry scalars.
- It must fail loud when Phase1-prepared data cannot be represented by the current model state.

The execution-entangled decode-time slot selector (`DecodeTimeSlotSelector` tensors, `DecodeTimeNumPolicy` candidate ops, `AutogradSelectorSupervisionLoss` CE) was deleted; selection is being rebuilt as an execution-independent numeric-meaning selector per `docs/ATOM_SELECTOR_IMPLEMENTATION_PLAN.md`. Do not recreate selector logic that derives its candidate set from `ExecutionMemory`.

The NumberEncoder numeric-meaning input path is `autograd::number_encode()` in `Shared/TensorContract/GradFns/NumberEncoderGradFn.{hpp,cu}`. Forward consumes the eight registry-owned `NumberEncoderParameterTensors` (passed as a non-owning `NumberEncoderForwardParams` view from `executeModelForward`) plus the payload digit-place channels (`BatchPayload::atom_digit_*` / `atom_global_features` uploaded as `BatchDeviceBindings::d_atom_digit_*`), and emits a dense `[total_tokens, d_model]` tensor that is zero on every non-atom row; shared forward fuses it via `autograd::residual_add` into the embedding before dropout. Backward is batch-aware (re-reads indices/features/masks from backward bindings, like `EmbeddingGradFn`), writes only atomicAdd accumulations into the eight registered leaf grad buffers, and propagates nothing upstream — AtomTable metadata is never trainable. Padding digit slots have mask 0 and contribute exactly zero in both directions. The channels carry CURRENT-token arg_number metadata only; next-token atom metadata is supervision and must never enter this input path.

The Rule 20 ownership taxonomy in `.github/copilot-instructions.md` is the authoritative contract for tape-bound vs. persistent state. This doc covers the implementation-level traps.

`Shared/Forward/ModelForwardOutputs.hpp` is the canonical home for the top-level shared-forward sink, the per-layer retained tensor vectors it owns directly, and the retained shared-forward payload types (`Forward::ExecutionBlockOutput`, `Forward::ExecutionBlockStepOutput`, `Forward::ExecutionRecord`, `Forward::ExecStepMetrics`). Do not recreate a separate ownership/include layer or a second helper-type wrapper for forward-result storage.

## Mandatory `return` in autograd forwards
Always explicitly `return output;` from any autograd forward function. A missing return destroys the `grad_fn` chain during forward → illegal memory access in backward.

## Mandatory GradFn operation names
Every concrete `GradFn` constructor must set `op_name` to a non-empty static string before the node is attached to a tensor. TensorContract grad-flow diagnostics fail loud on a missing name so debug logs can attribute backward anomalies to the exact operation instead of emitting anonymous tape nodes.

## Boundary call sites (single-owner rule)
- `forward_outputs.clear()` / `loss_state.clear()` — exactly **one** owner (the Phase2 `processBatch()` local step-state clear guard for each active training batch; shutdown mirrors only the retained shared-forward sink because `AutogradLossState` is batch-local explicit state).
- `flushDeferredCleanup()` — owned by `Tensor::backward()`. No external calls.
- Tape sealing: once `AutogradLossState::loss_tensor` is read as a host scalar, no further `autograd::add` / tape mutation. Loss-assembly and loss-readout functions must be distinct.

## `forward_outputs` / `AutogradLossState` checklist
- New field? → MUST be Category 1 (graph-owned, transient). If it persists, move to `TrainingState`.
- A `clear()` method that has to skip a field is admitting the field is in the wrong struct.
- Need a value after `backward()`? Snapshot a **scalar/reduced** form into a Category 2 telemetry struct **before** `clear()` runs.

## Atomic kernel ordering
When kernel B reads data written by kernel A via `atomicAdd`, you MUST `cudaStreamSynchronize` between them — even on the same stream.

## Fail-hard CenterColumns kernels
`center_columns*` autograd wrappers launch kernels that may deliberately `trap` on invalid sequence lengths. Every forward/backward CenterColumns launch must immediately check `cudaGetLastError()` and then `cudaStreamSynchronize(stream)` so launch failures and device-side traps surface at the centering call site, not several kernels later.

## Gradient norm sync
`cudaStreamSynchronize` inside `computeGradNorm` drains the entire backward pipeline. Pass `sync_for_host_read=false` for non-logging steps; only sync when logging gradient components.

## Registered parameter gradient lifecycle
Parameter gradient zeroing is owned by the registry-owned TensorContract `ParameterGroup` inventory. `executeAutogradBackward()` calls `zeroParameterGradients(parameter_registry.requireParameterGroups(...), stream)` only when `accumulate=false`; it must not enumerate embedding, LM-head, encoder, execution-block, or selector tensors directly. Adding a trainable tensor means placing its inventory entry on the startup parameter-registry boundary (`Startup/Model/ParameterRegistry.hpp` and `ParameterGroupRegistration`) so optimizer stepping, clipping, diagnostics, and zeroing all see the same source of truth.

## Tensor precision metadata
Tensor precision is TensorContract metadata, just like shape/layout. Startup `ParameterGroupRegistration` maps the authored `LanguageModelConfig::parameter_precision_*` value onto both the registered `ParameterGroup` and the owning `Tensor::compute_precision`. TensorContract operators are responsible for reading that metadata at operation boundaries and invoking conversion when needed; layers, optimizer code, and diagnostics must not carry a second precision policy object.

`Shared/TensorConversion` owns raw precision conversion operators (`convert_fp32_to_bf16`, `convert_bf16_to_fp32`) and layout conversion kernels. TensorContract may call TensorConversion, but conversion kernels must not be copied into GradFns or layer files. Persistent parameter data, gradient buffers, and optimizer moments remain FP32 unless an explicit TensorContract operator changes that contract. During gradient accumulation, leaf gradients are still Category 2 state and must be accumulated additively.

## Gradient connectivity verification
`verifyGradientsAreConnected()` scans each checked gradient tensor in full when computing finite/nonzero/RMS diagnostics. Do not reintroduce prefix sampling caps: a zero prefix (for example the first rows of `attnWqkv.grad`) is not evidence that the full parameter tensor missed gradient signal.

During gradient accumulation, existing parameter grad buffers already contain prior microbatch contributions. A post-backward nonzero/RMS check proves only that the cumulative buffer is nonzero, not that the current backward pass delivered signal. Accumulation-slot verification snapshots expected signal tensors before `AutogradLossState::loss_tensor.backward()` and measures the finite pre/post delta after backward.

Parameter connectivity verification has two independent channels. TensorContract leaf accumulation increments a host-side delivery counter on the parameter's shared gradient tensor; an active parameter with no new delivery is a structural failure even when an old accumulated gradient is nonzero. Full-tensor RMS/delta probing remains the numerical check. A parameter that has never established a nonzero numerical signal fails on its second consecutive zero-signal active check. Once it has established nonzero history, later structurally delivered zero-signal checks are retained as rate-limited saturation/dead-zone warnings rather than being mislabeled as graph disconnection.

If gradient connectivity verification fails after `AutogradLossState::loss_tensor.backward()` returns, `executeAutogradBackward()` must throw immediately. It must not return a recoverable-looking `success=false` result because Category 2 parameter gradient buffers have already been mutated by backward; continuing or retrying inside the same training process would risk carrying contaminated gradients.

ExecutionBlock gradient verification must be driven by `AutogradLossState` loss-added flags (`exec_op_ce_added`, `exec_arg_ce_added`, `exec_write_ce_added`) set by `computeAutogradLoss()`. Do not infer active execution supervision by scanning `exec_outputs_per_row`: that vector contains diagnostics for inactive rows and masked/padded teacher steps that may never be added to `loss_tensor`.

Execution loss assembly indexes execution diagnostics by payload row. When execution outputs exist, `ModelForwardOutputs::exec_outputs_per_row.size()` must exactly match `BatchPayload::batch_size`, and a non-empty `BatchPayload::execution_active` mask must have the same length. Mismatches must throw before any row-indexed access; never rely on `!exec_outputs_per_row.empty()` as a geometry guard.

## QKV attention boundary
Autograd attention owns the QKV tape boundary. `autograd::split_and_reshape_qkv()` creates the `SplitAndReshapeQKVGradFn` and delegates only raw layout movement to `TensorConversion::split_qkv_gqa()` / `merge_qkv_grads_gqa()`. Encoder code must not call TensorConversion QKV split/merge directly.

QKV projection is `autograd::matmul(ln1_out, W_qkv, transpose_b=true)` and must remain a normal TensorContract tape operation. Do not recreate `Layers/Attention/QKV_Projector.{hpp,cu}`; that stale wrapper no longer projected QKV and its remaining BHSD→BSM step was folded into the autograd wrapper `autograd::reshape_bhsd_to_flat()`, which calls `TensorConversion::convert_BHSD_to_BSM()` for the raw geometry and owns only the backward/tape bridge.

QKV-specific diagnostics live in `Shared/TensorContract/AutogradQKVDiagnostics.hpp/.cu`, next to `AutogradAttention.cu`. Keep `[QKV_EQUATION]`, `QKV_PROJECTION_EQUATION`, and `GRIM_DEBUG_QKV` NaN/Inf scans there so diagnostics observe the autograd path instead of creating an encoder-local parallel path.

`ScaledDotProductAttentionGradFn` owns the `Tensor::grad_` shared-pointer owners for captured Q/K/V gradient buffers, not only raw `q_grad` / `k_grad` / `v_grad` pointers. This is required so attention-internal Q/K/V tensors can be scoped locally without leaving SDPA backward with dangling pointers into destroyed non-leaf gradient tensors. Do not revert SDPA capture to raw grad pointers only.

`ScaledDotProductAttentionGradFn` must not grow a separate `save()` helper. SDPA forward owns one coherent setup path: run FlashAttention forward, keep the exact BF16 Q/K/V/O buffers and matching `softmax_lse`, stamp the resolved softmax scale/dropout metadata on the GradFn, and allocate backward workspaces there. A second save path risks uninitialized LSE or mismatched scale/dropout state.

## GradFn accumulation contract
GradFns must never overwrite a persistent leaf gradient buffer during backward. If a backward kernel writes directly into `tensor.grad_data()`, it must use additive writes (`+=` or `atomicAdd`) because `ensure_grad()` only allocates/zeroes the buffer once; step/microbatch zeroing owns the accumulation window.

For non-leaf inputs, a GradFn may write its local Jacobian result into an owned temporary buffer, but that buffer must be zeroed before use and additive writes are still preferred so the same kernel is safe for both owned and leaf buffers. If a GradFn must use a shared forward kernel that assigns into its output (for example centering kernels), keep the owned temporary buffer and explicitly accumulate that temporary into the leaf grad buffer before continuing the chain.

Use `Shared/TensorContract/GradientAccumulation.hpp` for generic `dst += src * scale` pass-through or scratch-to-leaf accumulation. Do not add per-translation-unit `kernel_accumulate_grad` copies in GradFns; operator-specific derivatives may still use their own additive kernels when they compute a real local Jacobian (for example GELU, RMSNorm, embedding scatter-add, or broadcast reductions).

## GradFn saved-buffer lifecycle
GradFns own Category 1 saved forward data and non-leaf gradient scratch only for the active tape window. Leaf tensors reuse their persistent `Tensor.grad_` buffers; non-leaf tensors allocate owned scratch, zero it before use, and release it from `release_saved()` / RAII deleters after `Tensor::backward()` has synchronized. Large loss buffers follow this same immediate GradFn lifecycle: `LogSoftmaxGradFn` owns the saved log-probability buffer, `NLLLossGradFn` borrows that saved buffer through the upstream GradFn, owns CE forward scalar scratch from `capture_inputs()`, owns its `grad_log_probs` backward scratch, and releases all of that tape-local state from `release_saved()`. Promoting full-vocab loss workspaces to durable preallocation would require a dedicated loss-workspace owner, not `TrainingState`.

CE scalar reduction scratch is a GradFn-owned forward scratch buffer. It is allocated in `NLLLossGradFn::capture_inputs()`, used immediately by `computeCrossEntropyForwardFromLogProbs()` to produce the host `mean_loss`, and released by `NLLLossGradFn::release_saved()`. It must not be added to `TrainingState`, `TrainingContext`, or `AutogradContext`.
