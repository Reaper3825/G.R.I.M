# Autograd Migration: Current Ownership and Gradient Flow

The January 2026 migration moved parameter gradients into TensorContract.
The current implementation is registry-owned, tape-driven autograd, not the
former manual output/encoder/input backward phases or encoder gradient getters.

## Current flow

1. Phase 1 allocates parameter tensor owners in
   [ParameterRegistry.hpp](../../resources/models/GRIM-text/training/Phases/Startup/Model/ParameterRegistry.hpp).
2. [ParameterGroupRegistration.cu](../../resources/models/GRIM-text/training/Phases/Startup/Model/ParameterGroupRegistration.cu)
   validates and publishes the complete inventory. `ParameterGroup` entries
   borrow the registry's tensors; they are not second parameter owners.
3. [Phase2_TrainingLoop.cu](../../resources/models/GRIM-text/training/Phases/Phase2_TrainingLoop.cu)
   uploads a prepared `BatchPayload`, calls shared forward, and supplies the
   returned graph outputs and matching device bindings to loss/backward.
4. [AutogradTraining.cu](../../resources/models/GRIM-text/training/Autograd/AutogradTraining.cu)
   assembles the connected objective and calls
   `loss_tensor.backward(nullptr, 1.0f, ctx.payload, ctx.device_bindings)`.
5. TensorContract GradFns contribute to the tape's gradient destinations.
   The epoch loop owns accumulation completion, normalization/clipping, and
   configured optimizer dispatch.

## Persistent gradients

`executeAutogradBackward()` zeroes the registered parameter gradients only when
`accumulate == false`. Subsequent microbatches add to those same buffers.
`ensure_grad()` is not a per-backward reset.

For example, `MatMulGradFn` in
[AutogradAttention.cu](../../resources/models/GRIM-text/Shared/TensorContract/AutogradAttention.cu)
uses accumulation beta `1.0f`; embedding scatter contributions accumulate through
[EmbeddingGradFn.cu](../../resources/models/GRIM-text/Shared/TensorContract/GradFns/EmbeddingGradFn.cu).
Tied embedding/LM-head storage is registered once, so both graph branches
contribute to one gradient buffer and receive one optimizer update.

Raw pointers still exist at CUDA/cuBLAS launch boundaries. The invariant is
correct ownership and accumulation, not the absence of every `float*`.

## Ownership and teardown

- Registry tensors and parameter gradients persist across the accumulation
  window; optimizer moments belong to persistent optimizer state.
- The current batch's shared-forward result and `AutogradLossState` are transient.
  `processBatch()` uses `ProcessBatchStepStateClearScope` to clear them.
- Saved tensors and non-leaf gradients belong to the tape, not the registry.
- Late diagnostics consume reduced snapshots, not live graph tensors.

In LoRA mode the optimizer inventory contains only trainable A/B pairs; the
separate base checkpoint inventory remains read-only. See
[LEARNED_PARAMETERS.md](PARAMETERS.md).

## Source verification

The relevant implementation boundaries are `buildParameterGroups()`,
`processBatch()`, and `executeAutogradBackward()` in the sources above.
This document describes source wiring; it does not assert that a training run
or a GPU gradient test was executed during documentation maintenance.
