# Autograd / TensorContract

Implementation: `resources/models/GRIM-text/Shared/TensorContract_GPU.cu` (all `GradFn` structs), `resources/models/GRIM-text/training/Autograd/AutogradTraining.cu` (forward/backward orchestration), and small loss primitives under `resources/models/GRIM-text/training/Autograd/`.

## Prepared payload boundary
Phase1/Phase1Startup owns semantic batch construction. By the time Phase2 calls autograd, `BatchPayload` is already complete: token IDs, targets, masks, MTP shifted targets, execution teacher steps, selector targets, numeric values, and slot maps are Phase1-authored data. Autograd code must consume this prepared payload and matching `BatchDeviceBindings`; it must not rebuild, infer, repair, or silently synthesize missing supervision.

Forward runtime handles are sibling payload data, not layer state: `AutogradContext` carries the `cudaStream_t` and `cublasHandle_t` borrowed from `TrainingState`, and `Forward::ModelForwardRequest` passes them through to encoder, FFN, LM head, reasoning head, and selector forwards. Do not patch those handles into layer configs or mutate layers with late setter calls.

## Primitive extraction rule
Do not add new catch-all executors around `AutogradTraining.cu`. Extract narrow primitives first:
- A primitive has one reason to exist and one mutation target.
- A loss primitive may mutate `AutogradIntermediates::loss_tensor` and Category 1 tensors only, then return host telemetry scalars.
- It must fail loud when Phase1-prepared data cannot be represented by the current model state.

Current extracted primitives:
- `AutogradSelectorSupervisionLoss.cu` — final-state decode-time selector CE. It consumes Phase1-authored selector targets, owns detached selector inputs in `AutogradIntermediates`, adds CE into `loss_tensor`, and returns the weighted selector scalar.

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

## Fail-hard CenterColumns kernels
`center_columns*` autograd wrappers launch kernels that may deliberately `trap` on invalid sequence lengths. Every forward/backward CenterColumns launch must immediately check `cudaGetLastError()` and then `cudaStreamSynchronize(stream)` so launch failures and device-side traps surface at the centering call site, not several kernels later.

## Gradient norm sync
`cudaStreamSynchronize` inside `computeGradNorm` drains the entire backward pipeline. Pass `sync_for_host_read=false` for non-logging steps; only sync when logging gradient components.

## GradFn accumulation contract
GradFns must never overwrite a persistent leaf gradient buffer during backward. If a backward kernel writes directly into `tensor.grad_data()`, it must use additive writes (`+=` or `atomicAdd`) because `ensure_grad()` only allocates/zeroes the buffer once; step/microbatch zeroing owns the accumulation window.

For non-leaf inputs, a GradFn may write its local Jacobian result into an owned temporary buffer, but that buffer must be zeroed before use and additive writes are still preferred so the same kernel is safe for both owned and leaf buffers. If a GradFn must use a shared forward kernel that assigns into its output (for example centering kernels), keep the owned temporary buffer and explicitly accumulate that temporary into the leaf grad buffer before continuing the chain.
