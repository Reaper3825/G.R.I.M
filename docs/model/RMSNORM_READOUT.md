# RMSNorm and Readout

Current RMSNorm math and last-row readout live together here. The historical
training-log analysis is kept separately from the implementation contract.

## Row-local normalization

[RMSNormGradFn.cu](../../resources/models/GRIM-text/Shared/TensorContract/GradFns/RMSNormGradFn.cu)
implements, for each row of width `D`:

```text
s = sqrt(sum_i x_i*x_i / D + epsilon)
y_i = x_i * gamma_i / s
```

Gamma is learned feature-wise, shared across rows, and not a row-scalar.
There is no normalization beta/mean-subtraction term. Nonuniform gamma may
change directions/angles; output RMS is not automatically one after gamma
learns or when epsilon is significant.

For upstream derivative `dy`, the input and gamma contributions are:

```text
a = sum_j dy_j * gamma_j * x_j / (D * s*s)
dx_i += (dy_i * gamma_i - x_i * a) / s
dgamma_i += sum_rows dy_i * x_i / s
```

The input kernel uses `+=`; gamma contributions use `atomicAdd`.
These do not overwrite persistent leaf gradients or apply another loss mean.
Fresh-window zeroing and accumulation belong to [autograd](AUTOGRAD.md) and
[training math](TRAINING_MATH.md), not an RMSNorm-local reset.

## Parameter and graph ownership

[ParameterRegistry.hpp](../../resources/models/GRIM-text/training/Phases/Startup/Model/ParameterRegistry.hpp)
owns per-layer `rms1_gamma`/`rms2_gamma` and LM-head `final_rms_gamma`.
[ParameterGroupRegistration.cu](../../resources/models/GRIM-text/training/Phases/Startup/Model/ParameterGroupRegistration.cu)
applies the configured freezing/registration policy. Final gamma has no
current hardcoded `lr_mult=0.1` registration.

Saved inputs, operation outputs, and non-leaf gradients belong to the graph.
Durable parameter gradients remain on the registry inventory. Frozen gamma
does not imply that no input-gradient computation is needed.
See [parameters](PARAMETERS.md) for shapes and exact configured gates.

## LM-head and inference readout

[lm_head_GPU.cu](../../resources/models/GRIM-text/Layers/LMHead/lm_head_GPU.cu)
validates the rows being read out, applies final RMSNorm to those actual rows,
then projects them (with the configured optional head-side SwiGLU adapter).
The math is row-local; it does not borrow the first token's normalized state
for another token.

[CachedInferenceForward.cu](../../resources/models/GRIM-text/Shared/Forward/CachedInferenceForward.cu)
copies the last logits row from the cached forward.
[Phase2_InferenceLoop.cu](../../resources/models/GRIM-text/training/Phases/Phase2_InferenceLoop.cu)
samples that row at prefill/continuation. Removed centering/PC1 interventions
are not part of this current readout.

## Historical Issue #34

The January 13, 2026 report described a manual inference path that normalized
the first encoder row while projecting the last, with space-only continuation
as the reported symptom. Its old output-forward file is not the current
implementation surface.

The current invariant is that the actual selected projected row receives its
configured normalization. That source wiring does not prove that every new
repetition symptom has the same cause or that a deployed binary was rebuilt.

[RMSNorm training analysis](investigations/RMSNORM_TRAINING_ANALYSIS.md) retains
the old run-specific dimensions, measurements, and health assessments.
Those values are not a new model-health certificate. No GPU or training
regression run was executed for this documentation update.
