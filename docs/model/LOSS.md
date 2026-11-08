# Loss Architecture Refactor: Current TextLoss Path

The February 2026 work established saved log probabilities as the loss
forward/backward source. The current implementation retains that principle,
but no longer uses the historical separate NLLLossGradFn -> LogSoftmaxGradFn
backward chain. This page describes the implementation now, not that migration
checklist or a production-readiness certificate.

## Public boundary and objective assembly

`autograd::unified_loss(logits, payload, bindings, loss_config, class_weights, stream)`
is declared/implemented in
[AutogradLoss.hpp](../../resources/models/GRIM-text/Shared/Loss/ComputeLoss/AutogradLoss.hpp)
and [AutogradLoss.cu](../../resources/models/GRIM-text/Shared/Loss/ComputeLoss/AutogradLoss.cu).
It requires the prepared host `BatchPayload`, matching uploaded
`BatchDeviceBindings`, and explicit `LossConfigHP`. Callers do not supply a
second geometry/target/workspace owner. The old `cross_entropy_loss()` public
entrypoint and manual ComputeLossBatch orchestration are not current APIs.

[AutogradTraining.cu](../../resources/models/GRIM-text/training/Autograd/AutogradTraining.cu)
selects atom-insertion BCE or text CE as the primary objective. For text batches
with configured retrieval and queries, it adds weighted sequence-local retrieval
loss through autograd. There is no neural register-machine execution auxiliary
loss or conventional MTP objective in this assembly.

## Forward

`TextLossGradFn::capture_inputs()`:

1. Captures the leaf-gradient destination or upstream logits producer.
2. Creates a non-owning, non-differentiable logits view and calls
   `autograd::log_softmax()` once. It owns the resulting saved log probabilities;
   it does not attach a LogSoftmaxGradFn for the production text loss.
3. Creates local scalar reduction scratch and calls
   `computeCrossEntropyForwardFromLogProbs()`.
4. Retains the host mean and class-weight sum. Valid target count remains
   `BatchPayload::lm_valid_tokens`; the forward helper checks its measured count
   against that authored count.

The kernel in
[CrossEntropyNLL.cu](../../resources/models/GRIM-text/Shared/Loss/ComputeLoss/CrossEntropyNLL.cu)
skips `target == -1` and rejects out-of-range targets. For `V > 1`, smoothing
assigns `q_target = 1-epsilon` and `q_other = epsilon/(V-1)`.

```text
CE = -sum_v q_v * log(p_v)
focal CE = alpha * (1-p_target)^gamma * CE   only when focal is enabled
entropy term = lambda * sum_v p_v * log(p_v)
row loss = class_weight[target] * (configured CE + entropy term)
```

The entropy term uses negative entropy, not `+lambda * H(p)`.
The reduction divides by valid token count, or by the sum of target-class
weights when class-balanced weighting is enabled. Focal parameters do not
scale ordinary CE when focal is disabled.

## Backward

`TextLossGradFn::apply()` receives the original payload and bindings, validates
saved logits geometry, and reads the upstream scalar derivative on the
provided stream. `computeCrossEntropyBackwardToLogits()` fuses the NLL-family
derivative and log-softmax Jacobian in registers:

```text
dL/dz_v = dL/dlog(p_v) - p_v * sum_j dL/dlog(p_j)
```

For unweighted plain CE this reduces to `(p_v - q_v)/N`, multiplied by the
upstream scalar derivative. Focal, entropy, and class weighting require their
full configured derivative, not the plain-CE shortcut.

The fused kernel adds directly to the upstream logits-gradient destination.
The active tape is notified of that contribution; passing the destination
back through `apply()` as a second contribution would double-count it.
Saved probabilities, not a new softmax from mutated logits, determine backward.

## Accumulation and lifetime

The host diagnostic means are unscaled. Phase 2 supplies separate
supervised-row accumulation scales for primary and retrieval objectives;
the connected roots carry those scales and backward is seeded with `1.0`.
The optimizer-window clipping boundary applies the existing configured
accumulation normalization once. See [TRAINING_MATH_AUDIT.md](TRAINING_MATH.md).

Saved log probabilities and non-leaf gradient destinations are tape-owned.
Forward reduction scratch is local to capture, not persistent TrainingState
loss workspace. The batch clear scope releases retained forward/loss roots;
TensorContract owns backward teardown.

The CUDA reduction uses atomic additions; source-level chain-rule consistency
does not establish bitwise deterministic reductions, measured performance,
or a passing GPU gradient test. Such claims require explicitly executed tests.
