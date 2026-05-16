# Unified Loss

Public autograd entry point: `resources/models/GRIM-text/Shared/Loss/ComputeLoss/AutogradLoss.cu`.

Cross-entropy / NLL implementation: `resources/models/GRIM-text/Shared/Loss/ComputeLoss/CrossEntropyNLL.cu` via `computeCrossEntropyForwardFromLogProbs()` and `computeCrossEntropyBackwardToLogProbs()`.

Primary text CE now enters as `autograd::unified_loss(logits, BatchPayload, BatchDeviceBindings, LossConfigHP, d_class_weights, stream)`. Auxiliary shifted-target heads (MTP) enter through `autograd::unified_loss_for_mtp_head(logits, BatchPayload, BatchDeviceBindings, head_idx, LossConfigHP, d_class_weights, stream)`. MTP targets are Phase1-authored on `BatchPayload`, uploaded by `LanguageModel::uploadBatchToDevice()`, and read from `BatchDeviceBindings.d_mtp_shifted_targets`; loss assembly must not allocate or upload target buffers.

`autograd::unified_loss()` is the **primary text loss path**. `cross_entropy_loss()` and the model-owned loss-options side channel are deleted; callers pass the durable `HyperParameters::LossConfigHP` grouping authored by Phase 1 as `TrainingContext.loss_config`. Phase 2 must not rebuild or wrap loss hyperparameters inside `TrainingLoopState`; it passes the Phase1 grouping directly into `autogradTrainingStep()`. `AutogradContext` borrows that grouping by required reference and does not run a second `initialized`/config-validity check. Class-balanced device weights are runtime `TrainingState` buffers and are passed separately.

There is no separate validation/eval loss pass. Phase2 derives epoch metrics from the training batches already executed through `autogradTrainingStep()`. Runtime payload upload lives in `Shared/Batching/BatchDeviceUpload.cu`; loss math lives in the autograd loss primitives only.

## Formula
$$L = \alpha (1 - p_t)^\gamma \cdot \mathrm{CE}_{\text{smooth}} + \lambda \cdot H(p)$$

(focal + label smoothing + entropy regularization in a single autograd kernel)

## Config
`ai_config.json → training.config.loss`

## Gradient clipping

The active training loop uses `GRIM::GradClip::clipGradientNorms()` with one **registered global RMS** over every clipping-owned `ParameterGroup` and one global scale coefficient when the configured limit is exceeded. `clipGradientNorms()` owns the `TrainingState.grad_norm_scratch` allocation/validation contract and returns a `ClipResult` containing the measured global/component metrics plus `measured_group_count` for validating the clipping topology actually used.

Gradient diagnostics must report `preclip_grad_rms` from that same `ClipResult`. Component-level RMS values are telemetry/diagnostic signals only; they are not clipping buckets unless `clipGradientNorms()` is changed to clip by those buckets. Diagnostics must not perform a second grad-norm measurement or add an extra stream sync.

## Footgun: double mean reduction
Loss backward already applies `1/N`. Do **not** add another `1/tokens` scaling in parameter gradient kernels (RMSNorm γ, etc.).

## Execution auxiliary normalization

ExecutionBlock auxiliary losses assembled in `AutogradTraining.cu` must be added to `loss_tensor` as one normalized aggregate, not as raw per-step sums. The local execution objective is accumulated over active scalar loss terms — transition loss, division penalties, arg REINFORCE loss, and each structured selection CE decision — then divided by that active term count before a single `autograd::add()` into the main loss.

Why: text CE is already averaged over valid tokens. Raw execution sums make rows with more teacher steps exert more loss pressure and make batch composition change the effective execution-loss weight.

Execution entropy is monitoring-only, not added to `loss_tensor`. Its row loop must mirror execution supervision masking: skip inactive rows, skip rows with no unmasked real steps, fail loud if `computeEntropyLoss(...)` returns null data, and average by the number of monitored rows rather than `payload.batch_size`.
