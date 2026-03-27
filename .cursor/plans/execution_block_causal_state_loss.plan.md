---
name: ExecutionBlock causal state loss
overview: Same as prior, plus optional expected_target for teacher-aligned transition_loss (vs internal expected), read_consistency_loss for slot-value truth, hard-gate curriculum (warmup / threshold schedule), and per-slot scale-aware epsilon for Δstate and changed_slots counts. L_exec merged into loss_tensor.
todos:
  - id: extend-step-output
    content: Step output + optional expected_target scalar/buffer; read_consistency_loss; v_soft fields; document [1,1] scalars
    status: pending
  - id: ground-truth-and-transition
    content: expected_internal = op(v1,v2) per kernelFourOps; hard gate uses |v_out - expected_internal|; soft transition_loss = |v_soft - expected_target| if teacher/target present else |v_soft - expected_internal|; pass target from teacher_steps into executeStep or fused args
    status: pending
  - id: fix-hard-transition-gate
    content: kStageTransitionInvalid when hard error > threshold ONLY after warmup (training_step >= exec_gate_warmup_steps) OR use scheduled threshold; document interaction with curriculum
    status: pending
  - id: state-delta-losses
    content: Per-slot epsilon_i = max(1e-6, 0.01 * |state_before[i]|) for hinge + changed_slots count; write_consistency before mutation unchanged
    status: pending
  - id: read-consistency-loss
    content: read_consistency_loss — |v1_actual - v1_expected_from_teacher| + arg2 when targets exist; else temporal/self-consistency stub or skip (document as next bottleneck)
    status: pending
  - id: valid-mask-penalty
    content: Penalize M.values[i] where valid_mask[i]==0 (reuse snapshots or M after step per spec); accumulate into state_integrity_loss or dedicated term per aggregate formula
    status: pending
  - id: write-mismatch
    content: write_mismatch_loss = max(p_write) * transition_error (use soft transition scalar for gradients); add write_entropy_penalty = f(entropy(p_write)) to discourage collapse (reuse/align with existing entropy aux patterns)
    status: pending
  - id: trace-consistency
    content: trace_consistency_loss — reuse transition_error / same scalar if redundant; else assert single source of truth in diagnostics
    status: pending
  - id: l-exec-aggregate
    content: L_exec includes w6*read_consistency_loss when defined; weights + transition_hard_threshold + gate_warmup_steps + optional threshold schedule
    status: pending
  - id: fail-hard-validation
    content: Same throws as before; gates disabled or threshold inflated during warmup; NaN/write_slot/multi-slot unchanged
    status: pending
  - id: training-loop-autograd
    content: Wire L_exec into intermediates.loss_tensor via autograd::add (same pattern as MTP aux in AutogradTraining.cu) so backward sees penalties; extend result.loss_value / logging; avoid CPU-only CE-style path for these terms
    status: pending
  - id: tests-docs
    content: Extend ExecutionBlockTest.cu / smoke for non-finite throw and loss tensor connectivity; short DOCUMENTATION.md note on new losses
    status: pending
isProject: false
---

# ExecutionBlock: causal deterministic state transitions (plan)

## Objective

Enforce **causal, deterministic state evolution** in `[ExecutionBlockLayer::executeStep](resources/models/GRIM-text/Layers/ExecutionBlock/execution_block_GPU.cu)` by adding **validation + differentiable penalty tensors**, without redesigning the block (keep softmax + STE, memory layout, and kernel structure).

---

## Critical fixes (addendum — non-negotiable semantics)

### Fix 1 — HARD correctness gate (not just loss)

Inside `executeStep`, **after** computing `transition_error` (hard path: `|v_out - expected|`):

```cpp
if (transition_error > config_.transition_hard_threshold) {
    atomicMax(d_numeric_error_flag_, kStageTransitionInvalid);
}
```

**Why:** Distinguish **invalid state / broken machine semantics** (“this must not happen”) from **suboptimal but valid** softmax mass. Loss alone cannot replace that distinction.

**Plan items:** Add `transition_hard_threshold` to `ExecutionBlockConfig`, new stage `kStageTransitionInvalid`, existing post-step sync + `EXEC_CHECK` so training throws like other stages — subject to **Fix 8 (curriculum)** so early training does not abort every step.

---

### Fix 2 — Enforce single-slot mutation (strict)

**Soft hinge** on non-write slots is not sufficient.

**Hard gate** after computing per-slot deltas; use **per-slot `epsilon_i`** from **Fix 9** (not a global constant):

```cpp
int changed_slots = count(|state_after[i] - state_before[i]| > epsilon_i);
if (changed_slots > 1) {
    atomicMax(d_numeric_error_flag_, kStageMultiSlotMutation);
}
```

**Why:** Without this, memory **smears** across slots; it is not a register machine.

**Implementation note:** Implement `count` on-device (reduction kernel or atomic add to shared counter then compare). `changed_slots > 1` is intentional: exactly one slot may change (the write slot); **zero** changes may also be invalid depending on your write kernel — confirm against actual write path (if write always mutates one slot, `!= 1` may be stricter; the spec as given is `**> 1`**). Apply **Fix 8** if this gate false-positives during warmup.

---

### Fix 3 — Write consistency BEFORE mutation

**Wrong:** apply physical write, then compare `state_after` to `v_out` as “write consistency” (that only checks kernel plumbing, not semantic overwrite).

**Right:** compute **before** the memory write:

- `expected_slot_value = v_out` (the value about to be written)
- `actual_slot_before = state_before[write_slot]` (from snapshot already taken at step start when `diag_out != nullptr`)

```text
write_consistency_loss = |actual_slot_before - expected_slot_value|
```

**Why:** Penalize **writing garbage into a slot that already had meaning** (large overwrite surprise), not post-write identity.

**Ordering in `executeStep`:** snapshot `state_before` → compute `v_out` and `write_slot` → compute `write_consistency_loss` from `state_before[write_slot]` and `v_out` → then perform the write → snapshot `state_after` for integrity / `changed_slots`.

---

### Fix 4 — Gradient signal to selection (STE-friendly soft transition)

**Problem:** `transition_loss` on **hard** `v_out` weakly reaches `p_op`, `p_arg1`, `p_arg2`.

**Minimal fix:** keep **forward = hard** (current execution path unchanged for the actual write and `v_out` used for the record).

**Also compute** a **soft** scalar:

```text
v_soft = Σ_k p_op[k] * result_k
```

where each `result_k` is `op_k(v1, v2)` with the **same** `(v1, v2)` and safe-div rules as hard ops (use **STE’d or soft-read** `v1,v2` consistently with existing `p_arg1`/`p_arg2` path so gradients flow to slot selection).

**Then** (for **loss / backward**) — align with **Fix 6**:

```text
if expected_target provided:
    transition_loss = |v_soft - expected_target|
else:
    transition_loss = |v_soft - expected_internal|
```

where `expected_internal = op(v1, v2)` with safe-div rules. **Hard gate (Fix 1)** still uses `transition_error_hard = |v_out - expected_internal|` (machine correctness). Teacher alignment is primarily the **soft** path via `expected_target`.

**Why:** Provides `**∂L/∂p_op`** (and slot heads if `v1,v2` are soft) without removing STE forward behavior.

---

### Fix 5 — Strengthen write mismatch (entropy anti-collapse)

Keep:

```text
write_mismatch_loss = write_confidence * transition_error
```

(use the **soft** `transition_error` / magnitude tied to Fix 4 for gradient into `p_write` where intended.)

**Add:**

```text
write_entropy_penalty = low_entropy(p_write)
```

i.e. penalize **collapsed** `p_write` (always the same slot). Reuse or align with existing **entropy regularization** machinery (`entropy_weight`, `computeEntropyLoss`, `ExecStepMetrics`) so you do not duplicate conflicting definitions.

**Why:** Mismatch alone does not stop **always writing to the same register**.

---

### Fix 6 — `expected_target` (optional): teach **correct answers**, not only valid ALU behavior

**Problem:** If `transition_loss` only compares `v_soft` to **expected_internal** = `op(v1, v2)` from current register contents, the model can learn **internally consistent garbage** (wrong operands, wrong intermediates) and still get low transition loss.

**Add** an optional per-step **teacher / spec scalar** `expected_target` (e.g. from `teacher_steps[k].expected_value` or equivalent — align with existing batch payload).

**Soft transition loss (training):**

```text
if target present for this step:
    transition_loss = |v_soft - expected_target|
else:
    transition_loss = |v_soft - expected_internal|
```

**Hard gate / diagnostics:** Keep **transition_error_hard** = `|v_out - expected_internal|` (machine semantics: “did the hard path apply the op to the read values correctly?”). Optionally add a **separate** diagnostic or loss term for `|v_out - expected_target|` when target exists (do not conflate unless product-wise you want one gate).

**Plumbing:** Extend `executeStep(...)` with optional `const float* expected_target` (scalar per step) or pass via a small device buffer set by `AutogradTraining` from `payload->teacher_steps`.

**Why this is huge:** Without `expected_target`, training optimizes **valid computation**; with it, it can optimize **correct answers** relative to supervision.

---

### Fix 7 — READ validity: `read_consistency_loss` (beyond `kStageSlotUninit`)

**Gap:** Existing checks ensure a slot is **initialized** (`kStageSlotUninit`, etc.) but **not** that the slot holds the **semantically correct value for this step**. The model can write garbage early, **reuse it consistently**, and pass slot-init / transition / write-local checks while remaining **globally wrong**.

**Add `read_consistency_loss`:**

- When **external truth** exists (teacher specifies operand values or slot contents for this step): compare **actual read** vs **expected from spec**, e.g. `|v1_actual - v1_expected_from_state|` and similarly for `v2` if available.
- **Without** external truth: options — (a) **temporal consistency** (e.g. regularize reads against bootstrap literals or prior-step teacher); (b) **skip** the term but document this as the **next bottleneck** after this patch.

**Weights:** Add `w6 * read_consistency_loss` to `L_exec` when the term is active.

**Non-goal for v1:** Do not rip out `kStageSlotUninit`; **add** this loss as orthogonal pressure.

---

### Fix 8 — Hard gate curriculum (avoid killing training early)

**Risk:** Fix 1 (`transition_error > threshold` → throw) during **early training** when outputs are noisy → **constant crashes**.

**Mitigations (pick one or combine):**

1. **Warmup:** if `training_step < exec_gate_warmup_steps`, **do not** set `kStageTransitionInvalid` (or skip `EXEC_CHECK` for that stage only). Loss terms still apply.
2. **Scheduled threshold:** start with **large** `transition_hard_threshold` and **anneal** down to the target value over N steps.

Pass `training_step` (or threshold override) from `AutogradTraining` / context into `executeStep` or set on `ExecutionBlockLayer` at forward start.

**Multi-slot gate (Fix 2):** Consider the same warmup for `kStageMultiSlotMutation` if false positives appear before the write path stabilizes.

---

### Fix 9 — Scale-aware `epsilon` for Δstate (hinge + `changed_slots`)

**Problem:** A **constant** `epsilon` for `|Δstate[i]| > epsilon` is brittle: too small → false positives (float noise); too large → **silent corruption**.

**Define per slot `i` (use magnitude of pre-state or max of before/after):**

```text
epsilon_i = max(1e-6, 0.01 * |state_before[i]|)
```

Use `epsilon_i` for:

- Hinge penalty on non-write slots (compare `|Δ_i|` to `epsilon_i`).
- Counting **changed_slots** for Fix 2 (slot `i` “changed” if `|state_after[i] - state_before[i]| > epsilon_i`).

Implement on GPU (per-element); no single global constant for value comparisons.

---

## Ground truth in code today


| Concept                                                  | Location                                                                                                                                                                                                                                                     |
| -------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Per-step forward, `v1` / `v2`, op logits, `v_out`, write | `[execution_block_GPU.cu](resources/models/GRIM-text/Layers/ExecutionBlock/execution_block_GPU.cu)` (`executeStep`, ~1909+)                                                                                                                                  |
| Step diagnostics struct                                  | `[ExecutionBlockStepOutput](resources/models/GRIM-text/Layers/ExecutionBlock/execution_block_GPU.hpp)` — already has `p_`*, `v_out`, `state_before_`*, `state_after_*`, `record`                                                                             |
| Training always passes `diag_out`                        | `[AutogradTraining.cu](resources/models/GRIM-text/training/Autograd/AutogradTraining.cu)` (~700–714): `ExecutionBlockStepOutput step_diag` + `&step_diag`                                                                                                    |
| Autograd-connected total loss                            | `[intermediates.loss_tensor](resources/models/GRIM-text/training/Autograd/AutogradTraining.cu)` — text CE + MTP via `autograd::add`                                                                                                                          |
| Host-only exec CE today                                  | `computeAutogradLoss` adds `exec_structured_ce` to `**result.loss_value` only** (~1284–1384); **not** merged into `loss_tensor` in current code — new `**L_exec` must use `autograd::add(loss_tensor, …)`** if gradients through ExecutionBlock are required |
| Teacher per-step targets                                 | `payload->teacher_steps[b][k]` already carries `expected_value`, slots, `op_id` — use to populate **expected_target** and **read_consistency** when present (`[AutogradTraining.cu](resources/models/GRIM-text/training/Autograd/AutogradTraining.cu)`)      |


---

## Hard requirements → implementation mapping

### 1. Transition correctness (critical)

After `v1`, `v2`, selected `op` (0..3), and `v_out`:

- Compute **expected_internal** on-device with the **same safe divide** semantics as existing op kernels (align with `kernelFourOps` / division clamp — do not invent a different `safe(v2)`).
- **Hard (gate + diagnostics):** `transition_error_hard = abs(v_out - expected_internal)` — drives **Fix 1** (with **Fix 8** curriculum) and internal trace consistency.
- **Soft (training loss):** `v_soft = Σ_k p_op[k] * result_k(v1,v2)` then **Fix 6:** `transition_loss = abs(v_soft - expected_target)` if teacher target present, else `abs(v_soft - expected_internal)`.
- Persist per-step tensors on `ExecutionBlockStepOutput`.

**Constraint:** Soft path aligns to **teacher outcome** when available; hard path enforces **register-machine semantics**.

### 2. State delta enforcement

Using **state_before_values** / **state_after_values** `[V,1]` (already copied when `diag_out != nullptr`):

- **Write consistency (Fix 3):** `write_consistency_loss = abs(state_before[write_slot] - v_out)` computed **before** the physical write.
- **Non-write slots:** hinge on `abs(state_after[i] - state_before[i])` vs **epsilon_i** (**Fix 9**), not a global constant.
- **Fix 2 (hard):** `changed_slots = count_i(|Δ_i| > epsilon_i)`; if `changed_slots > 1` → `kStageMultiSlotMutation` (optional **Fix 8** warmup).

Accumulate non-write violations into **state_integrity_loss**.

### 2b. Read consistency (**Fix 7**)

- **read_consistency_loss:** when teacher (or spec) gives expected operand values / slot contents, penalize `|v1_actual - v1_expected|` (and `v2`). Otherwise temporal consistency or **skip** with documented follow-up.

### 3. Slot validity

Penalty when `**valid_mask[i] == 0`** but `**values[i] != 0`** (use **before or after** snapshot per single clear rule — recommend **after step** on `M` or `state_after` + `state_after_valid` for consistency with mutation).

Fold into `**state_integrity_loss`** unless you need a separate diagnostic tensor.

### 4. Write legitimacy

- **write_confidence** = `max(p_write)` (GPU reduction on `[1,V]`).
- **write_mismatch_loss** = `write_confidence * transition_error` — prefer the **soft** transition magnitude from Fix 4 so gradients reach `p_write` where possible.
- **Fix 5 — write_entropy_penalty:** penalize **low** entropy(`p_write`) (collapse = always writing the same slot). Reuse or align with existing `entropy_weight` / `computeEntropyLoss` / `ExecStepMetrics` so definitions stay consistent.

### 5. Trace consistency

**trace_consistency_loss:** tie to **hard** `|v_out - expected_internal|` (same as **transition_error_hard** / gate signal). Do **not** double-count the **soft** `transition_loss` in `L_exec` unless you intentionally split “trace log” vs “train” semantics in the struct.

### 6. Aggregate `L_exec` (inside ExecutionBlock)

```
L_exec = w1 * transition_loss          // Fixes 4+6: |v_soft - expected_target| or |v_soft - expected_internal|
       + w2 * state_integrity_loss
       + w3 * write_consistency_loss   // Fix 3
       + w4 * write_mismatch_loss
       + w5 * write_entropy_penalty    // Fix 5
       + w6 * read_consistency_loss    // Fix 7 (omit or zero if no read targets)
```

Defaults: `w1=1.0`, `w2=0.5`, `w3=0.5`, `w4=0.25`, `w5` TBD, `w6` TBD when teacher read targets exist.

Expose weights via `ExecutionBlockConfig` optional floats so Phase1/TrainingOps can tune without recompile.

### 7. Fail-hard validation (existing style)

On violation:

- Set **stage** on `d_numeric_error_flag_` — **kStageTransitionInvalid** (Fix 1), **kStageMultiSlotMutation** (Fix 2), plus existing `kStage*` values.
- **EXEC_CHECK** / post-kernel sync pattern **identical** to current `executeStep` validation (see `kernelCheckFinite`, slot validation).
- **Fix 8 (curriculum):** while `training_step < exec_gate_warmup_steps`, **suppress** transition and/or multi-slot hard failures **or** compare against a **loosened** `effective_transition_threshold`; after warmup, enforce full gates. Optionally **anneal** threshold from high → target.

**Throw** on (only when the corresponding gate is **active**):

- NaN in hard transition error / required penalty tensors.
- **write_slot** outside `[S, V)` (value-slot range).
- **transition_error_hard > effective_transition_threshold** (Fix 1 + schedule).
- **changed_slots > 1** (Fix 2, with **epsilon_i** from Fix 9).
- Hinge loss on non-write slots still handles small numerical slack **without** throwing.

**No** downgrade to warnings.

---

## Implementation rules (recap)

- **GPU-only** math for these losses; no new CPU-side forward for penalties.
- **Do not** remove softmax/STE, change tensor layouts, or move selection off the current path.
- **Prefer** existing `**autograd::`** ops (`add`, `scale_scalar`, reductions) for scalar losses so `**L_exec` chains into `executeStep`’s tensors**.

---

## Output shape: `ExecutionBlockStepOutput` extensions

Add fields (names per spec):

- `Tensor transition_error` or `transition_error_hard` — `[1,1]` hard `|v_out - expected_internal|` (Fix 1 gate).
- `Tensor transition_loss` — soft `|v_soft - expected_target|` or `|v_soft - expected_internal|` (Fixes 4 + 6).
- Optional scalar/bool or sentinel for **whether `expected_target` was used** (debug).
- `Tensor state_integrity_loss`, `write_consistency_loss`, `write_mismatch_loss`, `write_entropy_penalty`, `read_consistency_loss` — scalars.
- Optional: alias **trace_consistency_loss** to hard transition error to avoid duplicate storage.

Return or store `**L_exec`** scalar tensor:

- Either as **additional field** on `ExecutionBlockStepOutput` (`exec_aggregate_loss`) **or** accumulated only in the training loop from step outputs — prefer **per-step `L_exec` on `step_diag`** plus **loop `autograd::add`** for clarity and debugging.

---

## Training loop compatibility

**A. Encoder forward** (where `executeStep` is invoked — same file, execution-block loop ~700+):

1. Pass **global `training_step`** (or effective threshold / gate-enable flags) into `executeStep` for **Fix 8**.
2. When `payload->teacher_steps` has data for `(b, k)`, pass **expected_target** and any **read expectations** for **Fixes 6–7** (device scalar or small buffer uploaded per step).

**B. `computeAutogradLoss`** (`[AutogradTraining.cu](resources/models/GRIM-text/training/Autograd/AutogradTraining.cu)`):

1. After building `intermediates.loss_tensor` (and MTP adds), iterate `exec_outputs_per_row[b].steps`, sum **per-step `L_exec`** with `autograd::add` (and optional global weight `execution_block_causal_loss_weight` in config if you want a master switch).
2. Include the same quantity in **result.loss_value** and **finite checks**.
3. Log components similarly to `exec_structured_ce` / `exec_entropy`.

This matches the **MTP** pattern (~1205–1207): differentiable auxiliary merged into `**loss_tensor`** so `**loss_tensor.backward`** sees ExecutionBlock penalties.

---

## Inference / `diag_out == nullptr`

Current repo calls `**executeStep`** from training with `**diag_out` set**. If inference later calls with `nullptr`:

- Either **always** allocate minimal device temporaries for validation inside `executeStep` (no host logic), **or** document that causal loss + hard checks run only when `diag_out != nullptr`.

The plan default: **training path is authoritative**; inference parity can be a follow-up.

---

## Files to touch (expected)


| File                                                                                                                                                                             | Change                                                                                                                                                                                                                                        |
| -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `[execution_block_GPU.hpp](resources/models/GRIM-text/Layers/ExecutionBlock/execution_block_GPU.hpp)`                                                                            | `ExecutionBlockStepOutput` + `ExecutionBlockConfig`: `transition_hard_threshold`, `exec_gate_warmup_steps`, optional threshold schedule, loss weights `w1..w6` (no global constant epsilon — Fix 9 is per-slot formula)                       |
| `[execution_block_GPU.cu](resources/models/GRIM-text/Layers/ExecutionBlock/execution_block_GPU.cu)`                                                                              | Order: `state_before` snapshot → `v_out`/selection → **write_consistency from `state_before[ws]`** → **soft `v_soft`/loss** (Fix 6 branch) → **hard gates** (Fix 8) → physical write → `state_after` → **epsilon_i** deltas / `changed_slots` |
| `[AutogradTraining.cu](resources/models/GRIM-text/training/Autograd/AutogradTraining.cu)`                                                                                        | Forward: pass `training_step`, `expected_target`, read targets into `executeStep`; loss: merge Σ `L_exec` into `intermediates.loss_tensor`                                                                                                    |
| `[Phase1_Startup.cu](resources/models/GRIM-text/training/Phases/Phase1_Startup.cu)` / `[TrainingOps.cu](resources/models/GRIM-text/training/TrainingOps.cu)` (if config plumbed) | New optional weights                                                                                                                                                                                                                          |
| `[ExecutionBlockTest.cu](resources/models/GRIM-text/Tests/ExecutionBlockTest.cu)`                                                                                                | Smoke + non-finite / flag behavior                                                                                                                                                                                                            |


**No** serialization change unless new **trainable** parameters are added (this plan adds **loss terms**, not weights — config-only floats may still go through existing config paths without checkpoint tensor changes).

---

## Acceptance assertion

After implementation, **without** paying `L_exec` (weights → 0), behavior matches baseline; with weights **on**, the model receives **gradient pressure** so that:

- **transition_loss** chases **teacher `expected_target`** when present (Fix 6), else **expected_internal** (Fix 4); wrong **hard** math can trip **Fix 1** when gates are active post-warmup (Fix 8).
- **changed_slots > 1** fails hard (Fix 2) with **scale-aware epsilon_i** (Fix 9); hinge + mask penalties cover smaller drift.
- **read_consistency_loss** (Fix 7) pressures correct **operand values**, not only initialized slots.
- Confident writes paired with wrong math are penalized via **write_mismatch_loss**; collapsed **p_write** via **write_entropy_penalty** (Fix 5).

If the graph **does not** connect these scalars to **intermediates.loss_tensor**, the requirement “required training signal” is **not** met.

---

## Relation to other plans

This plan is **orthogonal** to [persistent execution trace / Pattern B](persistent_execution_trace_31b8f378.plan.md): trace feedback encodes **history**; this work enforces **local step correctness** of the register machine. Integrate in `executeStep` with ordering **Fix 3** (write consistency before write; gates after hard `transition_error`; soft loss alongside without changing hard forward semantics).