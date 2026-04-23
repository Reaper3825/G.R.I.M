# Autograd / TensorContract

Implementation: `resources/models/GRIM-text/Shared/TensorContract_GPU.cu` (all `GradFn` structs), `resources/models/GRIM-text/Training/AutogradTraining.cu` (forward/backward orchestration).

The Rule 20 ownership taxonomy in `.github/copilot-instructions.md` is the authoritative contract for tape-bound vs. persistent state. This doc covers the implementation-level traps.

## Mandatory `return` in autograd forwards
Always explicitly `return output;` from any autograd forward function. A missing return destroys the `grad_fn` chain during forward → illegal memory access in backward.

## Boundary call sites (single-owner rule)
- `autograd_intermediates.clear()` — exactly **one** call site (the RAII `AutogradStepScope`).
- `flushDeferredCleanup()` — owned by `Tensor::backward()`. No external calls.
- Tape sealing: once `loss_tensor` is read as a host scalar, no further `autograd::add` / tape mutation. Loss-assembly and loss-readout functions must be distinct.

## `AutogradIntermediates` checklist
- New field? → MUST be Category 1 (graph-owned, transient). If it persists, move to `TrainingState`.
- A `clear()` method that has to skip a field is admitting the field is in the wrong struct.
- Need a value after `backward()`? Snapshot a **scalar/reduced** form into a Category 2 telemetry struct **before** `clear()` runs.

## Atomic kernel ordering
When kernel B reads data written by kernel A via `atomicAdd`, you MUST `cudaStreamSynchronize` between them — even on the same stream.

## Gradient norm sync
`cudaStreamSynchronize` inside `computeGradNorm` drains the entire backward pipeline. Pass `sync_for_host_read=false` for non-logging steps; only sync when logging gradient components.
