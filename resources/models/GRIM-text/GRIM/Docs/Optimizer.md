# Optimizer Window

The **Optimizer Window** is the single architectural boundary where accumulated gradients become durable optimizer state and updated parameter weights.

Location: `training/Phases/Phase2_TrainingLoop.cu`, owned by `runEpoch()` after `processBatch()` returns and `advanceAccumulationOrThrow()` reports a full accumulation window.

`OptimizerContext` in `training/Phases/Phase1_Startup.hpp` is a durable state carrier only (`optimizer_step`, optimizer state tensors, soft-restart controller, resumable accumulation-slot cursor). It must not own accumulation-window progression logic. `processBatch()` is the microbatch/autograd boundary only: it uploads one `BatchPayload`, runs forward/loss/backward, emits post-backward diagnostics, and returns. It must not read or mutate `OptimizerContext`, advance accumulation slots, complete optimizer steps, or call `launchOptimizerUpdate()`.

## Boundary sequence

The window is intentionally narrow and ordered:

1. `runEpoch()` authors a `BatchAutogradPlan` from the current accumulation slot and active `batch_idx`.
2. `processBatch()` consumes that immutable plan and produces/accumulates gradients for one microbatch.
3. `runEpoch()` advances the accumulation gate and confirms the full microbatch window is complete.
4. Phase2 scales the registered parameter gradients once by `1 / accum_steps` for the completed window.
5. Registered gradient clipping consumes the normalized parameter grads.
6. Pre-step diagnostics that need gradient/tying state run from the epoch-owned optimizer window.
7. The optimizer window calls `launchOptimizerUpdate(...)` exactly once.
8. Post-step finite/weight diagnostics run after the optimizer stream is synchronized.
9. `runEpoch()` completes optimizer-step bookkeeping and gradient clearing at the end of the finished accumulation window.

## Ownership policy

- `runEpoch()` owns **when** the Optimizer Window opens/closes, because it owns chronological batch iteration, accumulation-slot advancement, clipping timing, diagnostics timing, checkpoint cadence, and loop bookkeeping.
- `processBatch()` owns only the per-microbatch autograd boundary. It receives `BatchAutogradPlan` by required reference and must not reach into `ctx.optimizer`.
- `Shared/Optimizers/OptimizerUpdate_GPU.{hpp,cu}` owns configured optimizer dispatch. It is the only training orchestration path that may branch on `OptimizerKind`.
- `OptimizerUpdate_GPU.hpp` must include `Shared/HyperParameters/HyperparameterGroupings.hpp` directly. Do not forward declare `OptimizerUpdateHP`; this boundary consumes the grouping contract by value/reference, not an opaque private type.
- `Shared/Optimizers/AdamW/AdamW_Kernal_GPU.{hpp,cu}` owns only AdamW kernels and AdamW all-group stepping.
- `Shared/Optimizers/RAdamW/RAdamW_Kernal_GPU.{hpp,cu}` owns only RAdamW kernels and RAdamW all-group stepping.
- AdamW and RAdamW files must not include or dispatch to each other. Cross-optimizer selection belongs only in `OptimizerUpdate_GPU.cu`.

## Forbidden patterns

- Do not branch on `optimizer_kind` or `OptimizerKind` in Phase 2.
- Do not read or mutate `ctx.optimizer` inside `processBatch()`.
- Do not call `advanceAccumulationOrThrow()`, `completeOptimizerWindowBookkeepingOrThrow()`, or `launchOptimizerUpdate()` from `processBatch()`.
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

Autograd does not own accumulation-window normalization. `executeAutogradBackward()` runs with the default root seed, while the Optimizer Window owns the single post-accumulation `1 / accum_steps` scaling pass over the registered parameter gradients before clipping and optimizer update.
