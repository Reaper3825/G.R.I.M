# Optimizer Window

The **Optimizer Window** is the single architectural boundary where accumulated gradients become durable optimizer state and updated parameter weights.

Location: `training/Phases/Phase2_TrainingLoop.cu`, inside the `should_step` branch after `advanceAccumulationOrThrow()` reports a full accumulation window.

## Boundary sequence

The window is intentionally narrow and ordered:

1. Accumulation gate confirms the full microbatch window is complete.
2. Registered gradient clipping consumes the accumulated parameter grads.
3. Pre-step diagnostics that need gradient/tying state run from Phase 2.
4. Phase 2 calls `launchOptimizerUpdate(...)` exactly once.
5. Post-step finite/weight diagnostics run after the optimizer stream is synchronized.
6. `completeOptimizerStepAfterFullAccumulationWindow(...)` owns optimizer-step bookkeeping and gradient clearing.

## Ownership policy

- `Phase2_TrainingLoop.cu` owns **when** the Optimizer Window opens/closes, because it owns accumulation, clipping, diagnostics, checkpoint cadence, and loop bookkeeping.
- `Shared/Optimizers/OptimizerUpdate_GPU.{hpp,cu}` owns configured optimizer dispatch. It is the only training orchestration path that may branch on `OptimizerKind`.
- `OptimizerUpdate_GPU.hpp` must include `Shared/HyperParameters/HyperparameterGroupings.hpp` directly. Do not forward declare `OptimizerUpdateHP`; this boundary consumes the grouping contract by value/reference, not an opaque private type.
- `Shared/Optimizers/AdamW/AdamW_Kernal_GPU.{hpp,cu}` owns only AdamW kernels and AdamW all-group stepping.
- `Shared/Optimizers/RAdamW/RAdamW_Kernal_GPU.{hpp,cu}` owns only RAdamW kernels and RAdamW all-group stepping.
- AdamW and RAdamW files must not include or dispatch to each other. Cross-optimizer selection belongs only in `OptimizerUpdate_GPU.cu`.

## Forbidden patterns

- Do not branch on `optimizer_kind` or `OptimizerKind` in Phase 2.
- Do not call `launchAdamWStep(...)` or `launchRAdamWStep(...)` from Phase 2.
- Do not read `LanguageModel::getConfig()` to configure optimizer updates.
- Do not add embedding-freeze skip logic to Phase 2; the configured optimizer boundary owns it.
- Do not move gradient clearing into `launchOptimizerUpdate(...)`; accumulation-window bookkeeping owns clearing after diagnostics/checkpoint decisions.

## Required call shape

Phase 2 passes only the completed-window facts:

- registered `ParameterGroup` vector
- grouped `OptimizerUpdateHP`
- current learning rate
- current optimizer step
- primary CUDA stream

The dispatcher validates these inputs fail-loud, selects the concrete optimizer, and invokes the concrete optimizer implementation. The concrete optimizer files validate their own per-group state before launching kernels.

## Non-ownership

The Optimizer Window does not own backward seed scaling. The `1 / accum_steps` scaling is applied before/during backward accumulation so the window receives already-normalized accumulated parameter gradients.
