# GPU memory accounting

`MemoryMeasurer` retains the existing `[PEAK_MEMORY]` fields and sampling points.
`GRIM::MemoryAccounting::Enabled` in
`Shared/Diagnostics/MemoryAllocationTracker.hpp` is the single compile-time gate
for the allocation hooks and checkpoint logs. Accounting starts with allocations,
not with Phase 2, so live startup tensors are included in the persistent baseline.

Each checkpoint also emits `[GRAPH_MEMORY]` with the same `sample`, `owner`,
`batch`, and `accumulation_slot` context and byte/MiB fields:

| Category | Contents |
| --- | --- |
| `storage` | Tensor data and uncategorized tracked scratch allocations |
| `graph_gradient` | Per-consumer gradients, including common capture, direct GradFn buffers, QKV split gradients and non-leaf Tensor gradients |
| `engine_gradient` | GradFn pending-gradient accumulators allocated when backward contributions arrive |
| `leaf_gradient` | Tensor-owned leaf gradients, normally persistent parameter gradients |
| `saved_for_backward` | Owned saved inputs, weight copies, probabilities and dropout masks |
| `attention_buffer` | FlashAttention BF16 saved tensors, BF16 backward buffers and workspaces |

Categories are explicitly assigned at allocation/capture sites; they are not
guessed from allocation-label text. Storage is counted once by allocation pointer.
Non-owning Tensor views do not register allocations. Shared owners do not increment
allocation counts. The hooks do not retain tensors, reset the ledger at graph
boundaries, or change the original CUDA free operation.

`[GraphAllocationSizes]` prints a deterministic per-category, per-allocation-label
inventory at `forward.outputs_live`, `backward.complete_graph_live` and
`microbatch.graph_cleared`. Its indented entries follow `[ForwardOutputSizes]`:
`label category=... allocations=... bytes=... MiB=... pending_free_bytes=...`.
The allocation labels identify operations, not individual layer instances.

## Reconciliation

The current microbatch's `microbatch.persistent_floor` establishes both a device
usage baseline and a tracked-allocation baseline. Each subsequent checkpoint logs:

```
device_delta_bytes  = current device usage - baseline device usage
tracked_delta_bytes = current tracked bytes - baseline tracked bytes
residual_bytes      = device_delta_bytes - tracked_delta_bytes
```

The signed residual is intentionally not clamped. `coverage=tensor_contract`
means the ledger covers TensorContract allocations and its Tensor factories,
including callers elsewhere that use those factories. It does not cover every
raw allocation in other subsystems, CUDA library allocations, other processes,
allocator overhead or changes between the two sequential observations. Those
remain in the residual. `inventory_complete=true` means no bookkeeping failures
were observed, **not** complete coverage of device memory.

The forward report preserves its original `total_bytes` and `total_MiB` fields
for compatibility and adds `unique_data_bytes`, `unique_data_MiB`, `aliased_bytes`
and a per-entry `unique_data` flag. This deduplicates identical data pointers used
by retained full-tensor views; it is not a general overlapping-slice analyzer.
**Do not add forward inventory bytes to tracked bytes**: tracked storage already
includes those allocations.

## Release boundaries and failures

Synchronous allocations retire only when their CUDA free succeeds. Async frees
retain their bytes under `pending_free_bytes` until a nonblocking event query
confirms completion. These pending bytes are a subset of tracked bytes, not an
additional category. No stream/device synchronization is introduced. Generation
IDs prevent delayed release bookkeeping from deleting a newer allocation that
reused the same address. CUDA allocation/free return semantics are preserved.

If async completion cannot be observed, bytes remain conservatively counted and
`observation_errors` increases. Diagnostic exceptions do not terminate training.
A failed inventory query invalidates the reconciliation baseline. Check the
error fields before treating any residual as exact. Checkpoints are sampled
boundaries, not a measurement of every transient CUDA peak.

## Validation

`Tests/MemoryAccounting/memory_accounting_test.cpp` is a host-only test against a
minimal fake CUDA backend. Compile it with C++17 and add
`Tests/MemoryAccounting` to its include path. Do not add this fake include path
to training or runtime targets. It covers shared owners, async completion, failed
frees, failed allocations, observation failures, device isolation, category
changes and pointer reuse. It does not validate CUDA execution on a real device.
