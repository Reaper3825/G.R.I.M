# Unified Loss

Implementation: `resources/models/GRIM-text/Shared/Loss/ComputeLoss/AutogradLoss.cu`.

`autograd::unified_loss()` is the **only** loss path. `cross_entropy_loss()` is a wrapper that calls it with plain-CE config.

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
