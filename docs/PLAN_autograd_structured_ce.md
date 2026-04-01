# Plan: Autograd-Connected Structured CE for Execution Block

## Review corrections folded into this revision

This revision explicitly incorporates the feedback from plan review:

- The new CE path must be gated on **teacher availability**, not on `diag_out`.
- The preferred implementation is **logits-space CE**, not probability-space NLL.
- If a probability-space fallback is used, **forward and backward must both clamp**.
- Validation must prove **gradient deltas, ablation behavior, and faster selection learning**, not merely “non-zero grads”.
- Teacher-slot indexing is treated as an **unproven contract that must be verified and enforced**, not as a settled assumption.
- Host-side CE deletion happens **only after** scalar logging is cleanly re-threaded, to avoid losing comparability or double-counting.

---

## Problem

The execution block already computes a teacher-supervised structured CE term (`exec_structured_ce`), but it does so on the **host**:

- copy `p_op`, `p_arg1`, `p_arg2`, `p_write` to CPU
- compute `-log(p[target])` on CPU
- add the result to `result.loss_value`

That means the structured CE term currently contributes **zero gradients**. The actual autograd-connected execution signal is still dominated by `transition_loss`, which supervises the soft value outcome rather than directly supervising the discrete selection decisions.

This is the core learning gap: the model is told whether the **result** was wrong, but not directly told which **selection decision** was wrong.

---

## Current code facts this plan is anchored to

1. `computeAutogradLoss()` currently computes `exec_structured_ce` by copying execution probabilities to host and evaluating CE there. It contributes to `result.loss_value`, not to `intermediates.loss_tensor`.
2. `executeStep()` currently computes `transition_loss` only inside the existing `if (diag_out)` block. Training currently passes a per-step `ExecutionBlockStepOutput`, but the new CE path must **not** repeat this silent dependency pattern.
3. `ConceptExecutionSequenceBuilder` emits teacher slots as `base_slot + local_idx`.
4. `DataLoader.cu` defaults `concept_exec_base_slot` to `0`, but it is configurable via `GRIM_CONCEPT_EXEC_BASE_SLOT`.
5. `ExecutionBlockConfig` supports `num_scratch_slots > 0`, while current host-side CE code in `AutogradTraining.cu` hardcodes `S = 0` with a `TODO`.
6. Existing autograd cache ownership already has two explicit patterns:
   - **owned saved forward copy** (`SoftmaxGradFn::save()`, many execution-block GradFns)
   - **documented borrowed cache** with an ownership flag / contract (`LogSoftmaxGradFn::save(copy=false)`, tape-based cache contracts such as `MatMulGradFn`)

Conclusion: the current indexing logic works under today’s defaults, but it is **not yet a proven invariant** once non-zero scratch slots or non-zero base slots are allowed.

---

## Preferred formulation: logits-space selection CE

### Why change the formulation

The original draft used probability-space NLL:

$$
L = -\log p_t
$$

with backward:

$$
\frac{\partial L}{\partial p_t} = -\frac{1}{p_t}
$$

That is mathematically valid, but it is not the cleanest or most stable formulation. The better formulation here is **logits → stable CE**.

### Preferred forward loss

For a tiny logits vector $z \in \mathbb{R}^N$ and target index $t$:

$$
L(z, t) = \log\left(\sum_j e^{z_j - z_{\max}}\right) + z_{\max} - z_t
$$

where:

$$
z_{\max} = \max_j z_j
$$

### Preferred backward

$$
\frac{\partial L}{\partial z_j} = \operatorname{softmax}(z)_j - \mathbf{1}[j = t]
$$

Benefits:

- no `-1 / p` singularity
- no forward `-log(0)` / underflow issue
- cleaner optimization behavior than probability-space NLL
- easy to implement for tiny distributions (`N \in [4, 8]`)

### Fallback rule if probability-space NLL is used anyway

If reuse pressure forces a probability-space NLL path, this plan requires:

- forward: `-log(max(p_t, eps))`
- backward: `-1 / max(p_t, eps)`

with the **same epsilon in both places**. Backward-only clamping is explicitly forbidden.

---

## Implementation plan

### Step 0 — Prove and enforce the teacher-slot indexing contract

Before adding any new CE term, add a small validation / normalization helper at the training boundary.

This is mandatory because current code mixes:

- builder output: `base_slot + local_idx`
- runtime memory indices: `[0, V)`
- value-slot distributions: `[S, V)` mapped to `[0, V-S)`

The plan must not assume that:

- `arg1_slot - S` is always correct
- `arg2_slot - S` is always correct
- `write_slot` should always be used unshifted

Those formulas are only valid if the teacher slots are proven to already be in the exact memory index space the runtime expects.

#### Required action

Add a helper with fail-loud validation, conceptually:

- `normalizeTeacherSelectionTargets(ts_k, slot_base, S, V)`

That helper must:

1. define the authoritative slot contract
2. normalize teacher targets into:
   - `arg1_target_idx` in `[0, V-S)`
   - `arg2_target_idx` in `[0, V-S)`
   - `write_target_idx` in `[0, V)`
3. throw if the contract is violated

Two acceptable contract choices:

1. **Memory-slot contract**
   - teacher slots are already runtime memory indices in `[0, V)`
   - args must lie in `[S, V)`
   - normalization is `arg_idx = slot - S`, `write_idx = slot`

2. **Base-slot contract**
   - teacher slots are `base_slot + local_idx`
   - normalize via `slot - base_slot`, then map to the runtime index space

This plan does **not** assume which one is correct until code-level confirmation is added.

#### Fail-loud rule

If teacher supervision is present and slot normalization cannot be proven, training must **throw**, not silently “best effort” the targets.

---

### Step 1 — Make the CE path depend on teacher availability, not diagnostics presence

The structured CE path must live **outside** the optional diagnostics block.

#### What is wrong with the old draft

The original draft proposed:

```cpp
if (diag_out && expected_target) {
    // compute new CE
}
```

That is a bad training gate. If `diag_out` is ever null in a real training path, the model silently loses the new gradients.

#### Required behavior

The CE path must be gated only on:

- teacher row exists
- teacher step exists
- step mask says the step is real
- normalized teacher indices passed validation

It must **not** be gated on:

- `diag_out`
- debug mode
- whether host-side record snapshots are requested

#### Minimum-scope implementation choice

Use the existing per-step output object as the training carrier, but harden the contract:

- in training, if teacher supervision exists for `(b, k)` and the per-step output carrier is missing, **throw**
- compute `selection_ce_*` regardless of whether host diagnostics are enabled
- keep record / metrics / state snapshot logic under separate diagnostics-only guards

#### Cleaner alternative if the current API gets messy

Split the concepts explicitly:

- `ExecutionBlockStepTrainingOutput` — always present in training, carries `transition_loss` and new `selection_ce_*`
- `ExecutionBlockStepDiagnostics` — optional, host-facing snapshots / metrics only

This is not required for the first implementation, but the plan allows it if the current `diag_out` contract becomes too confusing.

---

### Step 2 — Add a lightweight logits-space CE primitive

Implement a small GradFn next to the existing execution-block GradFns, conceptually:

- `SelectionCrossEntropyGradFn`

Saved state:

- logits grad_fn
- logits shape
- saved softmax probabilities for the tiny distribution
- target index

#### Saved-buffer lifetime rule (must match existing cache logic)

This GradFn must not save a raw pointer with an implied lifetime.

Use one of the repo’s **existing** cache ownership contracts:

1. **Default path for the first implementation: owned saved copy**
   - mirror `SoftmaxGradFn::save()`
   - allocate a tiny device buffer for the saved probabilities during forward
   - copy the distribution into that buffer on the same stream
   - free it in `release_saved()` and in the destructor as the fallback safety net

2. **Optional future optimization: explicit borrowed cache**
   - only if profiling shows the tiny copy matters
   - mirror `LogSoftmaxGradFn::save(copy=false)` / the tape-based cache contract style
   - add an `owns_saved_probs` flag
   - document that the caller owns the buffer lifetime through `apply()` / `release_saved()`
   - borrowed mode is invalid unless the source buffer lifetime is proven

For this feature, the plan chooses **owned copy by default**:

- the distribution is tiny (`N \in [4, 8]` or at worst `V` for write selection)
- the project explicitly favors learning correctness over memory savings here
- this avoids creating a hidden dependency on `diag_out`, temporary tensors, or host snapshots

Fail-loud rule:

- if `apply()` runs and `saved_probs == nullptr`, throw
- if a borrowed-cache mode is ever added, the ownership mode must be explicit in code and comments
- `release_saved()` must clear the pointer in both modes and free only when ownership is true

Forward output:

- scalar tensor `[1,1]`
- stable logits-space CE

Backward behavior:

$$
\nabla_z L = p - y
$$

where $p$ is the saved softmax distribution and $y$ is the one-hot target.

This should attach directly to the logits tensors that already exist in `executeStep()`:

- `arg1_logits`
- `arg2_logits`
- `op_logits`
- `write_logits`

Important: compute the CE **while those logits still exist and still carry the full grad_fn chain**. Do not wait until only detached diagnostic copies remain.

---

### Step 3 — Compute per-selection CE immediately after logits creation

Wire the CE terms at the exact points where the logits are created:

- after `arg1_logits` → `selection_ce_arg1`
- after `arg2_logits` → `selection_ce_arg2`
- after `op_logits`   → `selection_ce_op`
- after `write_logits` → `selection_ce_write`

Each CE tensor should be a scalar `[1,1]` tensor stored in the step output used by training.

Only compute these terms when teacher supervision is available for that `(batch, step)` pair and target normalization succeeded.

Do **not** compute them from `p_*` diagnostic copies.

---

### Step 4 — Accumulate the new CE terms into autograd and into a clean scalar logging path

Host-side CE deletion is a dependency chain, not an afterthought.

#### Required order of operations

1. Add the autograd-connected per-step CE tensors
2. Accumulate them into:
   - `intermediates.loss_tensor` for backward
   - a single scalar execution-CE accumulator for reporting
3. Copy that single scalar to host for:
   - `result.loss_value`
   - finite checks
   - log output
4. Only then delete the legacy host-side CE loop

#### Recommendation: transitional parity phase

During a short debug / validation window:

- compute both the legacy host CE and the new device CE
- compare them within tolerance
- include **only one** of them in `result.loss_value`

This keeps comparability while avoiding silent drift or double-counting.

---

### Step 5 — Remove the legacy host CE path and clean up the dead causal scalar path

After parity is established and scalar reporting is in place:

- delete the `cudaMemcpy`-driven structured CE loop
- delete the host `safe_nll()` monitoring path or demote it to a temporary parity-only debug path
- remove the always-zero `exec_causal_loss` accumulator from the reported total, unless a real scalar monitoring path is wired for it

The current `exec_causal_loss_sum` / `exec_causal_count` path is dead reporting code and should not remain as fake signal.

---

## Config surface

Keep an explicit enable switch plus a dedicated execution CE weight.

Recommended initial config:

```json
"structured_ce_enabled": true,
"structured_ce_weight": 0.1
```

Rule 20 behavior:

- if `structured_ce_enabled == true` and teacher supervision is present, `structured_ce_weight` must be `> 0`, or training must throw
- do not silently configure “CE enabled” with zero effect
- if an ablation run wants transition-only behavior, it must disable CE explicitly via `structured_ce_enabled = false` or an equivalent explicit experiment switch
- ablation must not rely on `structured_ce_weight = 0` as an implicit off switch

---

## Files expected to change

| File | Purpose |
|---|---|
| `resources/models/GRIM-text/Layers/ExecutionBlock/execution_block_data_stream_GPU.cu` | Add the lightweight logits-space CE GradFn and wire per-step CE creation next to the logits tensors. |
| `resources/models/GRIM-text/Layers/ExecutionBlock/execution_block_GPU.hpp` | Add `selection_ce_*` tensors to the step output used by training. |
| `resources/models/GRIM-text/training/Autograd/AutogradTraining.cu` | Normalize teacher targets, accumulate CE tensors into `loss_tensor`, add clean scalar logging, then delete legacy host CE. |
| `resources/models/GRIM-text/training/Phases/Phase1_Startup.cu` | Parse and validate `structured_ce_enabled` and `structured_ce_weight`. |
| `resources/models/GRIM-text/GRIM/grim_language_model_cuda.hpp` (or equivalent config struct) | Add the runtime config fields for `structured_ce_enabled` and `structured_ce_weight` if they do not already exist. |
| `ai_config.json` | Add the explicit enable knob plus the weight once the code path exists. |

---

## Risks and mitigations

### 1. Index-space bug on teacher targets

**Risk:** CE is wired to the wrong slots because teacher slot ids are assumed rather than normalized.

**Mitigation:** make slot normalization a prerequisite step and throw on any mismatch.

### 2. Loss double-counting during cutover

**Risk:** both the legacy host CE and the new device CE are added to `result.loss_value`.

**Mitigation:** parity mode may compute both, but the reported total may include only one.

### 3. Gradient magnitude shift

**Risk:** four CE terms per step overpower the text loss or the existing transition loss.

**Mitigation:** start with `structured_ce_weight = 0.1`, then compare parameter-group gradient norms on identical batches.

### 4. Numerical instability if a probability-space fallback is chosen

**Risk:** forward `-log(p_t)` produces `Inf` / `NaN` when `p_t` underflows.

**Mitigation:** the preferred path is logits-space CE; if a fallback path is used, clamp in forward and backward with the same epsilon.

### 5. Saved-buffer lifetime bug in the new CE GradFn

**Risk:** `SelectionCrossEntropyGradFn` saves probabilities via a dangling pointer or via an undocumented borrowed buffer, causing use-after-free in backward.

**Mitigation:** use the same explicit ownership contract as the existing autograd code:

- default to an owned device copy
- only allow borrowed-cache mode with an explicit ownership flag and caller-lifetime contract
- throw if backward runs without a valid saved buffer

---

## Validation criteria

The previous “non-zero grad” validation is insufficient, because these parameters already receive some indirect gradient from `transition_loss`.

The revised validation bar is:

### 1. Gradient norm delta when CE is enabled

On the same teacher-forced batch, compare CE-off vs CE-on for parameters such as:

- `W_op_select`
- `w_arg1_select`
- `w_arg2_select`
- write-head selection parameters

Expected result: enabling CE produces a measurable gradient norm increase on the selection-related parameters.

### 2. Ablation proves the CE branch is the cause

Run an ablation where:

- `structured_ce_enabled = false`

or the CE accumulation branch is explicitly disabled in the experiment harness.

Expected result: the added gradient delta disappears.

### 3. Selection accuracy improves faster than transition-only baseline

Track teacher-supervised execution selection quality:

- op top-1 accuracy
- arg1 top-1 accuracy
- arg2 top-1 accuracy
- write top-1 accuracy
- optional exact-step match accuracy

Expected result: CE-enabled training improves these metrics faster than transition-only training.

### 4. Scalar parity during cutover

While both implementations coexist temporarily:

- device CE scalar should match legacy host CE within tolerance

### 5. Stability

Expected result:

- no NaN / Inf in the CE scalar
- no NaN / Inf in the affected gradients
- no regression in total loss finite checks

---

## Bottom line

The new structured CE path should be implemented as a **logits-space, autograd-connected loss on the execution decisions themselves**, with:

- explicit teacher-target normalization
- fail-loud handling of any missing training output carrier
- scalar logging in place before host CE removal
- validation based on gradient deltas, ablation, and faster learning of op / arg / write decisions

That gives the execution block the direct supervision it is currently missing, without smuggling new loss logic behind a diagnostics gate or relying on fragile index assumptions.
