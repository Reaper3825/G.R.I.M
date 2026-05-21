# TrainingState Cache Lifecycle Investigation

## Finding

`TrainingState` cache tensors have two different meanings in the current system:

- reusable device storage owned by `TrainingState`, addressed through explicit per-call views such as `BatchDeviceBindings`;
- implicit "last forward/current batch" mailboxes read directly from `TrainingState` by diagnostics, inference, and some production helper logic.

The second meaning is the lifecycle leak. It lets code outside the active forward/loss boundary rediscover runtime values by reading durable `TrainingState` fields. Those fields have capacity and allocation ownership, but they do not carry a payload id, step id, valid row count, mode, stream completion token, or freshness invariant.

The clearest instance is `cached_logits_tensor`. It began as a diagnostic snapshot, but it now acts as a graph-external logits side channel.

## Expected Boundary

The intended runtime boundary should be:

```text
BatchPayload + BatchDeviceBindings
  -> executeModelForward()
  -> active logits/hidden views
  -> loss and diagnostics observe explicit current-step views
  -> reduced telemetry snapshots are kept if needed after clear
  -> AutogradIntermediates clear
```

The current problematic path is:

```text
executeModelForward()
  -> copies logits/hidden into TrainingState durable cache tensors
  -> loss/backward run
  -> external code reads TrainingState cache tensors as "the current step"
```

This works only by convention. The cache tensors are durable buffers with stale contents across boundaries, so any direct value reader has to trust ordering and caller discipline that is not represented in the type system.

## Confirmed Leak: `cached_logits_tensor`

Owner and writer:

- `Shared/TrainingState/TrainingStateGPU.cu`: `TrainingState::allocateStepDeviceWorkspaces()` allocates the full `[max_tokens_per_batch, vocab_size]` logits slab.
- `Shared/Forward/ModelForward_GPU.cu`: `Forward::executeModelForward()` computes the LM-head `logits_tensor`, copies it into `TrainingState::cached_logits_tensor`, then stores the live autograd tensor in `AutogradIntermediates::logits_tensor`.

Correct active consumer:

- `training/Autograd/AutogradTraining.cu`: `computeAutogradLoss()` consumes `AutogradIntermediates::logits_tensor`, not the cache.

Direct cache readers:

- `training/Diagnostics/PredictionDistributionDiagnostic.cu`: copies logits from `cached_logits_tensor` for prediction distribution and logit trace.
- `training/Diagnostics/LogitScaleDiagnostic.cu`: copies logits from `cached_logits_tensor` for logit-scale statistics.
- `Layers/GRIMTS/GuessCacheTraining.cu`: copies prediction logits from `cached_logits_tensor` for guess-cache update logic.
- `training/Inference_GPU.cu`: inference prefill returns last-token logits by copying from `cached_logits_tensor`. Single-token decode does not use the slab; it copies directly from the live LM-head result.

Why this is a boundary leak:

- The forward result does not expose logits, so callers that need logits after forward are forced to know about the side effect.
- The cache can outlive the forward graph and contains stale data outside the active row range.
- The readers do not receive a value that proves "these logits belong to this payload and this forward."
- Diagnostics are conceptually outside gradient math, but that does not require them to be outside the active forward observation boundary.

## Related Leak: `cached_encoder_output`

`cached_encoder_output` has the same diagnostic side-channel shape as logits.

Owner and writer:

- `Shared/TrainingState/TrainingStateGPU.cu`: allocated as `[max_tokens_per_batch, d_model]`.
- `Shared/Forward/ModelForward_GPU.cu`: written from the actual LM-head input after optional centering.

Direct readers:

- `training/Diagnostics/LogitScaleDiagnostic.cu`: reads the buffer to compute hidden-state norms for the logit-scale equation.
- `training/Diagnostics/RhoDiagnostic.cu`: reads the buffer as the LM-head input layer in rho buildup.
- `GRIM/Docs/Diagnostics.md`: explicitly documents `cached_encoder_output` as the diagnostic source of truth.

This is less production-critical than logits, but it is the same architectural pattern. A hidden-state diagnostic should receive an explicit current-forward hidden view or a reduced snapshot, not discover it through durable global state.

## Suspect Area: `cached_token_ids_tensor` And Token Side Channels

The token staging tensors are not all bugs. They are valid as reusable backing storage when accessed through `BatchDeviceBindings`.

Valid pattern:

- `Shared/Batching/BatchDeviceUpload.cu` copies `BatchPayload` host arrays into `TrainingState` device buffers.
- It returns `BatchDeviceBindings`, a non-owning view valid only until the next upload.
- Forward/loss code consumes `BatchPayload` plus `BatchDeviceBindings`, not `TrainingState` directly.

This is the right shape because the active batch identity is carried by the call frame.

Direct readers that should be audited:

- `training/Diagnostics/GradientNormDiagnostic.cu`: reads `ts.cached_token_ids_tensor.data` directly for embedding-gradient equation diagnostics.
- `training/Inference_GPU.cu`: uses `cached_token_ids_tensor`, `cached_token_numeric_values`, `cached_token_atom_mask`, `cached_token_atom_flags`, and `cached_token_to_slot_map` as inference session staging and decode-time state.
- `training/Inference_GPU.cu`: builds inference prefill bindings directly from `TrainingState` buffers and updates decode positions in place.

Why this is suspicious:

- `cached_token_ids_tensor` is allocated as step workspace, but inference decode treats it as session state across forward steps.
- The token side-channel fields can be valid upload backing for a single call and invalid as hidden generation state.
- `GRIM/Docs/InferenceBoundary.md` already lists moving token id, numeric, atom, slot-map, and inference snapshot state out of `TrainingState` and into `GenerationState` as unfinished work.

The token buffers are therefore a mixed case:

- training/eval upload through `BatchDeviceBindings`: acceptable reusable backing storage;
- diagnostics reading token ids directly from `TrainingState`: diagnostic escape hatch;
- inference decode storing session tokens in `TrainingState`: likely ownership leak, because generation state should own generation session data.

## Lower-Risk Capacity Checks

Some references to cache tensors are not value leaks:

- `Startup/Model/ModelAllocationState.cu`: checks allocated tensor shapes against `HyperparameterGroupings.hpp::trainingFixedShapeHP()`.
- `Shared/Batching/BatchDeviceUpload.cu`: checks logits capacity before upload.
- `training/Diagnostics/BoundaryDiagnostic.cu`: reports allocation capacity.
- `training/Autograd/AutogradTraining.cu`: validates that the logits buffer can hold the current payload.

These sites are using cache tensors as allocation/capacity facts. They should still avoid implying that the cache contains current semantic values.

## Rule For Future Audits

A `TrainingState.cached_*` direct read is safe only if it reads allocation metadata or constructs an explicit per-call view immediately after the write boundary.

It is suspicious if the code:

- reads `.data` from `TrainingState.cached_*` after forward/loss/backward instead of receiving an explicit view;
- uses a cache field to infer "current batch", "current logits", "current token", or "current sequence";
- preserves values across `AutogradIntermediates::clear()`;
- uses a training cache as generation session state;
- copies from a cache without checking the active row count from `BatchPayload` or a returned result object.

## Remediation Direction

Recommended order:

1. Add explicit forward result views for logits and LM-head input.
   - `Forward::ModelForwardResult` should expose the live logits view and row/column shape.
   - Training should pass that view to diagnostics and guess-cache while still inside the active step scope.
   - Inference prefill should copy return logits from the live forward result before clearing intermediates.

2. Convert diagnostics to accept explicit observation inputs.
   - Diagnostics may stay outside gradient math.
   - They should not recover observed tensors through `TrainingState`.
   - Diagnostics must run inside the lifecycle graph of the information they need.
   - If a value is needed after the graph boundary, it should be moved to run within the graph boundary lifetime.

3. Split generation session buffers from training step workspace.
   - Move inference token ids, numeric values, atom masks/flags, and slot maps into `GenerationState` or a typed inference staging owner.
   - Keep `TrainingState` upload buffers as training/eval/inference-prefill backing only when represented by `BatchDeviceBindings`.

4. Delete or narrow global snapshots.
   - Once logits and hidden diagnostics use explicit views, remove `cached_logits_tensor` and `cached_encoder_output` from production contracts.
   - If full-tensor snapshots remain necessary, make them diagnostic-owned and attach freshness metadata: step, mode, rows, cols, and source stream.

## Open Questions

- `GuessCacheTraining` appears to be monitoring-only now. It was originally intended as something different, but current usage looks like an attempt to keep a dead feature alive. Treat it as a candidate for removal or reduction to explicit telemetry before designing a new logits handoff for it.
- Inference prefill needs further investigation. The unresolved design question is whether it should share the same full-forward result type as training or copy into an inference-specific logits output buffer owned by `GenerationState`.
- Diagnostics should not run outside the lifecycle graph of the information they need. They can run outside gradient math, but not outside the lifetime/freshness boundary for the tensors they observe.
- Additional non-`TrainingState` caches need further investigation, especially layer-local diagnostic caches or GradFn-adjacent caches that survive beyond the graph boundary.

## Current Conclusion

`cached_logits_tensor` is not a memory leak. It is a semantic lifecycle leak: a durable cache field is acting as an untracked current-forward output. `cached_encoder_output` follows the same diagnostic pattern, and `cached_token_ids_tensor` plus side-channel caches show a broader ownership split where valid upload backing storage has also become an escape hatch for diagnostics and inference session state.
