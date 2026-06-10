# Unified Loss

Public autograd entry point: `resources/models/GRIM-text/Shared/Loss/ComputeLoss/AutogradLoss.cu`.

Cross-entropy / NLL implementation: `resources/models/GRIM-text/Shared/Loss/ComputeLoss/CrossEntropyNLL.cu` via `computeCrossEntropyForwardFromLogProbs()` and `computeCrossEntropyBackwardToLogProbs()`. These helpers take `BatchPayload`, `BatchDeviceBindings`, a `CrossEntropyTargetSelection` source descriptor, and a caller-owned `CrossEntropyForwardWorkspace` view. In the production autograd path, that view is created only inside `NLLLossGradFn::capture_inputs()` from GradFn-owned scalar scratch; public callers must not pass CE workspace state. Callers must not manually pass `num_tokens`/`vocab_size` scalar geometry or pre-resolved raw target pointers. CUDA kernels receive `payload.total_tokens` / `payload.vocab_size` and the resolved target device pointer only at the device ABI boundary because `BatchPayload` is host-only, and they receive the durable `HyperParameters::LossConfigHP` grouping by value rather than exploded focal/smoothing/entropy scalar argument lists.

Primary text CE enters as `autograd::unified_loss(logits, BatchPayload, BatchDeviceBindings, LossConfigHP, d_class_weights, stream)`. Auxiliary shifted-target heads (MTP) enter through `autograd::unified_loss_for_mtp_head(logits, BatchPayload, BatchDeviceBindings, head_idx, LossConfigHP, d_class_weights, stream)`, but the corresponding `logits` tensors must already have been emitted by `Forward::executeModelForward(...)` into the caller-owned `ModelForwardOutputs` sink when the request’s `ModelForwardGraphPolicy.emit_mtp_logits` flag is set. MTP targets are Phase1-authored on `BatchPayload`, uploaded by `Batching::uploadBatchToDevice()`, and read from `BatchDeviceBindings.d_mtp_shifted_targets`; MTP count/validity comes from `BatchPayload` and config, not bindings. Loss assembly must not allocate or upload target buffers, and it must not create MTP logits.

`autograd::unified_loss()` is the **primary text loss path**. `cross_entropy_loss()` and the model-owned loss-options side channel are deleted; callers pass the explicit `HyperParameters::LossConfigHP` grouping derived from authoritative `TrainingContext.config.hyperparameters`. Phase 2 must not store or thread a second loss-config owner through `TrainingContext` or `TrainingLoopState`; it derives the grouping only at the immediate autograd call boundary, which passes both the explicit `const BatchPayload&` and `LossConfigHP&` into `computeAutogradLoss()` and MTP loss assembly. `AutogradContext` must not store that grouping or become the primary source for batch semantics inside loss assembly. Class-balanced device weights are runtime `TrainingState` buffers and are passed separately.

Phase1 handoff boundary: model vocab/capacity, parameter registration, class-weight storage ownership, and batch semantic construction are all decided before any loss graph node exists. Runtime loss assembly consumes the Phase1-authored `BatchPayload`, uploaded `BatchDeviceBindings`, the caller-derived `LossConfigHP`, current graph logits, and optional class-weight device buffer. `LogSoftmaxGradFn` owns the saved log-probability buffer needed by log-softmax backward. `NLLLossGradFn::capture_inputs()` owns the CE forward scalar reduction scratch (`loss_sum`, `valid_count`, `weight_sum`), saves the host `mean_loss`/`valid_count`/`weight_sum`, and `NLLLossGradFn::release_saved()` releases that scratch together with its `grad_log_probs` backward buffer at the tape boundary. Loss GradFns must not allocate or synthesize targets, masks, MTP geometry, or a second loss-config owner after the handoff.

There is no separate validation/eval loss pass. Phase2 derives epoch metrics from the training batches already executed through the explicit shared-forward + autograd loss/backward path in `Phase2_TrainingLoop.cu`. Runtime payload upload lives in `Shared/Batching/BatchDeviceUpload.cu`; loss math lives in the autograd loss primitives only.

Issue-investigation diagnostics such as the old finite-difference gradient verifier and Token 277 collapse probe are deleted from the production loss path. Do not reintroduce one-off diagnostic kernels here; add focused tests or guarded tooling outside the unified loss primitive instead.

## Formula
$$L = \alpha (1 - p_t)^\gamma \cdot \mathrm{CE}_{\text{smooth}} + \lambda \cdot H(p)$$

(focal + label smoothing + entropy regularization in a single autograd kernel)

## Config
`ai_config.json → training.config.loss_*` direct leaves. Do not recreate a nested `training.config.loss` sidecar object; `ai_config_paths.hpp` requires and parses the flat `loss_label_smoothing_*`, `loss_focal_*`, `loss_entropy_reg_*`, `loss_class_balanced_*`, `loss_preference_*`, `loss_distillation_*`, and `loss_masking_*` leaves.

## Gradient clipping

The active training loop uses `GRIM::GradClip::clipGradientNorms()` with one **registered global RMS** over every clipping-owned `ParameterGroup` and one global scale coefficient when the configured limit is exceeded. `clipGradientNorms()` owns the `TrainingState.grad_norm_scratch` allocation/validation contract and returns a `ClipResult` containing the measured global/component metrics plus `measured_group_count` for validating the clipping topology actually used.

Gradient diagnostics must report `preclip_grad_rms` from that same `ClipResult`. Component-level RMS values are telemetry/diagnostic signals only; they are not clipping buckets unless `clipGradientNorms()` is changed to clip by those buckets. Diagnostics must not perform a second grad-norm measurement or add an extra stream sync.

## Footgun: double mean reduction
Loss backward already applies `1/N`. Do **not** add another `1/tokens` scaling in parameter gradient kernels (RMSNorm γ, etc.).

## Execution auxiliary normalization

ExecutionBlock auxiliary losses assembled in `AutogradTraining.cu` must be added to `loss_tensor` as one normalized aggregate, not as raw per-step sums. The local execution objective is accumulated over active scalar loss terms — transition loss, division penalties, arg REINFORCE loss, and each structured selection CE decision — then divided by that active term count before a single `autograd::add()` into the main loss.

Loss decomposition at the training boundary is explicit: `LossResult` / `BatchResult` carry `text_loss`, `mtp_loss`, and `execution_loss`. Do not recreate a blended `aux_loss` bucket; it hides the real source of non-text loss and corrupts telemetry semantics. (The old `selector_loss` channel was deleted with the execution-entangled decode-time selector; new numeric supervision heads will add their own explicit channels.)

Why: text CE is already averaged over valid tokens. Raw execution sums make rows with more teacher steps exert more loss pressure and make batch composition change the effective execution-loss weight.

Execution entropy is monitoring-only, not added to `loss_tensor`. Its row loop must mirror execution supervision masking: skip inactive rows, skip rows with no unmasked real steps, fail loud if `computeEntropyLoss(...)` returns null data, and average by the number of monitored rows rather than `payload.batch_size`.
