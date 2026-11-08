# Training Math: Current Source Contracts

This page describes current PT/SFT training math and the source boundaries to
verify. It is not a passing finite-difference test, current collapse diagnosis,
or a recipe to restore removed encoder/readout centering or PC1 projection.

## 1. Prepared payload, forward, and loss

[Phase2_TrainingLoop.cu](../../resources/models/GRIM-text/training/Phases/Phase2_TrainingLoop.cu)
uploads the prepared `BatchPayload`, calls shared forward, and supplies its
outputs plus matching device bindings to
[AutogradTraining.cu](../../resources/models/GRIM-text/training/Autograd/AutogradTraining.cu).
Geometry and valid target counts come from the payload, not a parallel loss
workspace/config owner.

The primary objective is atom-insertion BCE when `EnableAtomIdentification`
is set; otherwise it is text `autograd::unified_loss()`. Text batches with
configured retrieval and queries also contribute weighted local-atom retrieval
loss. Neural register-machine selection/transition and conventional MTP losses
are not current objectives.

### Text CE

[AutogradLoss.cu](../../resources/models/GRIM-text/Shared/Loss/ComputeLoss/AutogradLoss.cu)
owns saved log probabilities in `TextLossGradFn`.
[CrossEntropyNLL.cu](../../resources/models/GRIM-text/Shared/Loss/ComputeLoss/CrossEntropyNLL.cu)
uses target `-1` as zero supervision, rejects invalid target ranges, and verifies
the measured forward count against `payload.lm_valid_tokens`.

For plain CE over `N` supervised positions:

```text
L = -sum_t log(p[t,target[t]]) / N
dL/dz[t,v] = (p[t,v] - one_hot(target[t])[v]) / N
```

Backward fuses the NLL-family derivative and log-softmax Jacobian directly
into logits gradients from the saved probabilities. Smoothing, focal,
negative-entropy regularization, and class balancing use their configured
derivatives; they do not share the plain-CE shortcut. Class-balanced reduction
uses the sum of target-class weights. See
[LOSS_ARCHITECTURE_REFACTOR_FEB2026.md](LOSS.md).

## 2. Microbatch accumulation

Let configured accumulation count be `M`; a particular window may contain
fewer microbatches at the epoch end. Let primary supervised counts be `n_i`,
with `N_window = sum_i n_i`.

[AccumulationWeights.hpp](../../resources/models/GRIM-text/Shared/Batching/AccumulationWeights.hpp)
implements the per-objective backward-root scale:

```text
s_i = M * n_i / N_window
```

`runEpoch()` computes separate counts/scales for primary supervision and
retrieval queries. Host telemetry reports original unscaled microbatch means;
`computeAutogradLoss()` builds the scaled connected roots, and
`executeAutogradBackward()` seeds backward with `1.0`.

[GradientCC_GPU.cu](../../resources/models/GRIM-text/Shared/Gradients/GradientCC_GPU.cu)
then applies configured `1/M` normalization once, before RMS measurement:

```text
g_window = (1/M) * sum_i s_i * gradient(L_i)
         = sum_i (n_i/N_window) * gradient(L_i)
```

This includes a short final window without reducing its contribution by its
missing slots. Each objective is weighted by its own supervision count.
For class-balanced microbatch losses this is a supervised-row-weighted mean of
those microbatch objectives, not necessarily the same as one corpus-wide
class-weight-sum reduction.

Do not add another `1/N` in parameter-gradient kernels or another root `1/M`.

## 3. Persistent gradients and tying

`executeAutogradBackward()` zeroes the registry inventory only at a fresh
window (`accumulate=false`). `ensure_grad()` does not clear each contribution.

For `Y = X @ W.T`, MatMulGradFn in
[AutogradAttention.cu](../../resources/models/GRIM-text/Shared/TensorContract/AutogradAttention.cu)
implements `dX = G @ W`, `dW = G.T @ X`, using accumulation beta `1.0`.
[EmbeddingGradFn.cu](../../resources/models/GRIM-text/Shared/TensorContract/GradFns/EmbeddingGradFn.cu)
scatter-adds the matching token/type-gated embedding contributions.
Both paths add to the tied buffer rather than relying on overwrite ordering.

[ParameterGroupRegistration.cu](../../resources/models/GRIM-text/training/Phases/Startup/Model/ParameterGroupRegistration.cu)
registers tied embedding/LM-head weights once and rejects duplicate data-pointer
registration. Optimizer/clipping/zeroing all consume that same inventory.

## 4. GQA and readout

The GQA reduction in `AutogradAttention.cu` sums grouped query-head dK/dV
contributions into KV-head gradients, using `+=`. It does **not** multiply by
the former `1/sqrt(heads_per_kv_group)` balancing factor.

[lm_head_GPU.cu](../../resources/models/GRIM-text/Layers/LMHead/lm_head_GPU.cu)
applies row-local final RMSNorm and optional head-side SwiGLU residual adapter
before projection. Configured token-type gating, positional behavior, and
normalization remain distinct from removed centering/PC1 interventions.

## 5. Clipping and optimizer

`clipGradientNorms()` first normalizes accumulated gradients, then measures
registered global RMS:

```text
RMS = sqrt(sum_all_registered_elements g*g / registered_element_count)
```

When enabled and above the configured limit, one global coefficient scales
the gradients. Component metrics are diagnostics, not independent clipping
budgets. `runEpoch()` owns optimizer timing and
[OptimizerUpdate_GPU.cu](../../resources/models/GRIM-text/Shared/Optimizers/OptimizerUpdate_GPU.cu)
dispatches the configured AdamW or RAdamW update.

Both optimizer step implementations apply group LR/decay multipliers.
[AdamW_Kernal_GPU.cu](../../resources/models/GRIM-text/Shared/Optimizers/AdamW/AdamW_Kernal_GPU.cu)
uses `m_hat * rsqrt(v_hat + ADAMW_EPSILON)`, with epsilon inside the square root.
[RAdamW_Kernal_GPU.cu](../../resources/models/GRIM-text/Shared/Optimizers/RAdamW/RAdamW_Kernal_GPU.cu)
uses its rectification policy and `sqrt(v_hat) + epsilon` in the rectified
denominator. Do not collapse these distinct formulas into one alleged kernel.
Bias correction uses the optimizer step's actual chronology.

Final RMS gamma has no current hardcoded slow-LR `0.1` registration.
LoRA mode optimizes only registered A/B pairs while base tensors are frozen.
See [LEARNED_PARAMETERS.md](PARAMETERS.md).

## 6. Verification boundaries

Validate derivatives against the exact configured objective, additive
branched/tied gradients, padding/type gates, partial windows, and optimizer
resume state using focused tests before making runtime correctness claims.
Historical fixed token IDs, gradient magnitudes, and weight-decay ratios do
not describe every current model. Diagnose a present run with
[LOG_FILE_CONVENTION.md](../LOG_FILE_CONVENTION.md).

No trainer/model/GPU target was built or run for this documentation maintenance.
