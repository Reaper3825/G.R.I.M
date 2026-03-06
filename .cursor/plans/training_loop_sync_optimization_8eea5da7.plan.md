---
name: Training Loop Sync Optimization
overview: Reduce GPU-to-CPU synchronization points in the Phase2 training loop without sacrificing accuracy. The current hot path has 5-9 syncs per batch, many of which are diagnostic-only or redundant, and several massive blocking cudaMemcpy calls that drain the GPU pipeline.
todos:
  - id: a1-remove-redundant-sync
    content: Remove redundant cudaStreamSynchronize at Phase2:3784 (measureGradientNorms already syncs internally)
    status: completed
  - id: a2-fix-device-sync
    content: Replace cudaDeviceSynchronize with cudaStreamSynchronize in sampleWeightStats (TrainingDiagnostics.cu:58)
    status: completed
  - id: b1-gate-collapse-diag
    content: Gate collapse diagnostic block (128 blocking copies) behind shouldSyncDiagnostics or replace with GPU argmax kernel
    status: completed
  - id: b2-remove-token277
    content: Remove Token277 pre/post weight snapshots entirely (model stabilized, no longer in collapse)
    status: pending
  - id: b3-gate-emb-grad
    content: Gate embedding gradient equation diagnostic behind shouldSyncDiagnostics
    status: completed
  - id: c1-batch-loss-sync
    content: Batch text loss and numeric loss cudaMemcpyAsync into a single sync point
    status: completed
  - id: c2-split-gradnorm
    content: Split measureGradientNorms into launch/sync phases to overlap GPU compute with CPU work
    status: completed
  - id: d1-batch-diag-copies
    content: Replace individual blocking cudaMemcpy in diagnostic blocks with batched async + single sync
    status: completed
  - id: d2-gpu-argmax-kernel
    content: Replace 128-iteration host argmax detection loop with single GPU histogram kernel
    status: completed
isProject: false
---

# Training Loop Sync Point Optimization

## Current Sync Inventory

I audited every `cudaDeviceSynchronize`, `cudaStreamSynchronize`, and blocking `cudaMemcpy` in the training hot path. Below is a per-batch classification.

### Tier 1: EVERY-BATCH Syncs (Hot Path)

These fire on **every single batch** regardless of diagnostic flags:


| #   | Location | Sync Type | Purpose |
| --- | -------- | --------- | ------- |


1. **Loss retrieval** ([AutogradTraining.cu:1044-1045](resources/models/GRIM-text/training/Autograd/AutogradTraining.cu)) -- `cudaStreamSynchronize` after `cudaMemcpyAsync` of scalar loss. **Unavoidable** (need loss for NaN check and logging), but could be deferred/overlapped.
2. **Numeric loss retrieval** ([AutogradTraining.cu:960-962](resources/models/GRIM-text/training/Autograd/AutogradTraining.cu)) -- `cudaStreamSynchronize` for numeric head loss scalar. Only active when numeric head is enabled. **Could be batched with #1.**
3. **Guess cache logit D2H** ([Phase2:1976-1985](resources/models/GRIM-text/training/Phases/Phase2_TrainingLoop.cu)) -- `cudaStreamSynchronize` for pred_logits copy. Only when `guess_aux_enabled && guess_cache_ready`. **Could use async pipeline.**
4. **measureGradientNorms** ([GradNormGPU.cu:211](resources/models/GRIM-text/Shared/GradNorm/GradNormGPU.cu)) -- `cudaStreamSynchronize` inside the function to read D2H per-group sums.
5. **REDUNDANT Phase2 stream sync** ([Phase2:3784](resources/models/GRIM-text/training/Phases/Phase2_TrainingLoop.cu)) -- Another `cudaStreamSynchronize(stream)` immediately after `measureGradientNorms()` returns, which already synced internally. **Eliminate this -- pure waste.**
6. **Collapse diagnostic: targets D2H** ([Phase2:3327-3330](resources/models/GRIM-text/training/Phases/Phase2_TrainingLoop.cu)) -- Blocking `cudaMemcpy` for flat_targets. Runs every `diag_interval=10` batches when equation logging is on.
7. **Collapse diagnostic: up to 128 logit row copies** ([Phase2:3352-3355](resources/models/GRIM-text/training/Phases/Phase2_TrainingLoop.cu)) -- Individual blocking `cudaMemcpy` per sampled position inside the argmax detection loop. **This is catastrophic -- 128 serial blocking copies.**
8. ~~**Token277 pre-optimizer snapshot**~~ ([Phase2:4338-4341](resources/models/GRIM-text/training/Phases/Phase2_TrainingLoop.cu)) -- **REMOVE ENTIRELY** (model stabilized, no longer in collapse).
9. ~~**Token277 post-optimizer delta**~~ ([Phase2:4481-4484](resources/models/GRIM-text/training/Phases/Phase2_TrainingLoop.cu)) -- **REMOVE ENTIRELY**.
10. **Post-accumulation gradient norm** ([Phase2:4224-4230](resources/models/GRIM-text/training/Phases/Phase2_TrainingLoop.cu)) -- `cudaStreamSynchronize` for gradient clipping norm. Only runs on accumulation-complete batches (`should_step`). **Necessary for clipping correctness.**

### Tier 2: PERIODIC Diagnostic Syncs (Gated by `shouldSyncDiagnostics`)

These only fire when equation logging is enabled AND `(batch_idx + 1) % GRIM_SYNC_INTERVAL == 0`:

1. **Logit prediction distribution** ([Phase2:2128](resources/models/GRIM-text/training/Phases/Phase2_TrainingLoop.cu)) -- Blocking `cudaMemcpy` of `100 * vocab_size * 4` bytes (~1.6 MB for vocab=4096).
2. **PtPvDump row fetches** ([Phase2:2335](resources/models/GRIM-text/training/Phases/Phase2_TrainingLoop.cu)) -- Per-position blocking `cudaMemcpy` for positions beyond the initial sample (up to 20 extra copies).
3. **Logit statistics copy** ([Phase2:2470](resources/models/GRIM-text/training/Phases/Phase2_TrainingLoop.cu)) -- Another blocking `cudaMemcpy` of logit sample (50 positions).
4. **Hidden state + weight row copies** ([Phase2:2635, 2685, 2807, 2890, 3079](resources/models/GRIM-text/training/Phases/Phase2_TrainingLoop.cu)) -- Multiple blocking `cudaMemcpy` calls for various diagnostics.
5. **sampleWeightStats PRE/POST optimizer** ([TrainingDiagnostics.cu:58](resources/models/GRIM-text/training/Diagnostics/TrainingDiagnostics.cu)) -- Uses `cudaDeviceSynchronize()` (global device sync!) followed by `cudaStreamSynchronize`. **Two syncs where one stream sync would suffice.**
6. **GradStats flushAndLog** ([Phase2:4111-4133](resources/models/GRIM-text/training/Phases/Phase2_TrainingLoop.cu)) -- `cudaStreamSynchronize` for gradient stats D2H.
7. **sampleOptimizerMomentStats** ([Phase2:4442-4454](resources/models/GRIM-text/training/Phases/Phase2_TrainingLoop.cu)) -- `cudaMemcpyAsync` + `cudaStreamSynchronize` for moment buffer sampling.
8. **computePerComponentUpdateTrace** ([TrainingDiagnostics.cu:2031](resources/models/GRIM-text/training/Diagnostics/TrainingDiagnostics.cu)) -- Internal `cudaStreamSynchronize`.

### Tier 3: Rare/One-Shot Syncs

1. **Embedding gradient equation** ([TrainingDiagnostics.cu:129-157](resources/models/GRIM-text/training/Diagnostics/TrainingDiagnostics.cu)) -- Full vocab gradient D2H (~16 MB) + token IDs D2H. Fires every `emb_grad_diag_interval=10` batches when equation logging on.
2. **Inference sample** ([Phase2:268, 387](resources/models/GRIM-text/training/Phases/Phase2_TrainingLoop.cu)) -- Two `cudaDeviceSynchronize()` calls bracketing `model->generate()`. Only fires periodically at log_interval.
3. **Validation** ([Phase2:1333](resources/models/GRIM-text/training/Phases/Phase2_TrainingLoop.cu)) -- `cudaDeviceSynchronize()` safety drain. Once per epoch.
4. **cudaMemGetInfo** ([Phase2:4949](resources/models/GRIM-text/training/Phases/Phase2_TrainingLoop.cu)) -- Implicitly syncing on some drivers. Once per epoch.

---

## Optimization Plan

### Phase A: Eliminate Redundant Syncs (Zero Risk)

**A1.** Remove the redundant `cudaStreamSynchronize(stream)` at [Phase2:3784](resources/models/GRIM-text/training/Phases/Phase2_TrainingLoop.cu) -- `measureGradientNorms()` already syncs internally at [GradNormGPU.cu:211](resources/models/GRIM-text/Shared/GradNorm/GradNormGPU.cu).

**A2.** In `sampleWeightStats` ([TrainingDiagnostics.cu:58](resources/models/GRIM-text/training/Diagnostics/TrainingDiagnostics.cu)), replace `cudaDeviceSynchronize()` with `cudaStreamSynchronize(stream)` -- there is no reason to drain ALL device streams when only the primary stream's data is needed.

### Phase B: Gate Diagnostics That Currently Run Every Batch (Low Risk)

**B1.** Gate the collapse diagnostic block ([Phase2:3310-3392](resources/models/GRIM-text/training/Phases/Phase2_TrainingLoop.cu)) behind `shouldSyncDiagnostics()` instead of the current `kCollapseDiagEnabled` (which fires every 10 batches). The 128 individual blocking `cudaMemcpy` calls in the argmax detection loop are the single worst offender. Alternatively, replace the 128 serial blocking copies with a single GPU argmax reduction kernel.

**B2.** **Remove all Token277 diagnostics entirely** (model stabilized, no longer in collapse):

- Pre-optimizer weight snapshot ([Phase2:4328-4351](resources/models/GRIM-text/training/Phases/Phase2_TrainingLoop.cu))
- Post-optimizer weight delta ([Phase2:4468-4515](resources/models/GRIM-text/training/Phases/Phase2_TrainingLoop.cu))
- `computeToken277Diagnostic` call in collapse block ([Phase2:3376-3381](resources/models/GRIM-text/training/Phases/Phase2_TrainingLoop.cu))

Eliminates 2+ syncs per optimizer-step batch and D2H copies in the collapse diagnostic path.

**B3.** Gate the embedding gradient equation diagnostic ([Phase2:3851-3887](resources/models/GRIM-text/training/Phases/Phase2_TrainingLoop.cu)) behind `shouldSyncDiagnostics()` instead of its own `emb_grad_diag_interval`.

### Phase C: Batch and Overlap D2H Transfers (Medium Effort)

**C1.** Batch the text loss and numeric loss retrieval into a single `cudaStreamSynchronize` -- issue both `cudaMemcpyAsync` calls before syncing once.

**C2.** For `measureGradientNorms`, split it into launch + sync phases: launch the kernel and issue the async D2H copy, then do CPU work (logging, stats), then sync only when the result is actually needed. This overlaps GPU compute with CPU logging.

### Phase D: Replace Blocking Copies with Async Patterns (Higher Effort)

**D1.** Replace all blocking `cudaMemcpy(..., cudaMemcpyDeviceToHost)` in the `shouldSyncDiagnostics` block with `cudaMemcpyAsync` + a single `cudaStreamSynchronize` at the end of the diagnostic section. Currently there are ~10 individual blocking copies that each drain the pipeline independently.

**D2.** Replace the 128-iteration argmax detection loop ([Phase2:3349-3368](resources/models/GRIM-text/training/Phases/Phase2_TrainingLoop.cu)) with a single GPU kernel that computes per-position argmax and writes a histogram of dominant tokens. This eliminates 128 blocking copies entirely.

---

## Expected Impact

**Before optimization (typical batch, equation logging on):**

- Hot path: 5-7 syncs (loss, numeric loss, guess cache, gradnorm x2, token277 x2 when collapse token was tracked)
- Every 10 batches: +3-5 syncs (collapse diag, emb grad)
- On diagnostic interval: +8-12 syncs (logit copies, weight samples, moment samples)

**After optimization:**

- Hot path: 2 syncs (loss retrieval, gradnorm -- both unavoidable)
- Every N batches (diagnostic interval only): +consolidated syncs (batched D2H)
- Optimizer-step batches: +1 sync (post-accum norm -- unavoidable for clipping)

**Estimated sync reduction: ~60-75% fewer pipeline stalls on typical batches.**

---

## Files to Modify

- [Phase2_TrainingLoop.cu](resources/models/GRIM-text/training/Phases/Phase2_TrainingLoop.cu) -- Main training loop (most changes)
- [TrainingDiagnostics.cu](resources/models/GRIM-text/training/Diagnostics/TrainingDiagnostics.cu) -- `sampleWeightStats` device sync fix
- [AutogradTraining.cu](resources/models/GRIM-text/training/Autograd/AutogradTraining.cu) -- Batch loss + numeric loss sync
- [GradNormGPU.cu](resources/models/GRIM-text/Shared/GradNorm/GradNormGPU.cu) -- Split launch/sync API (Phase C2)

