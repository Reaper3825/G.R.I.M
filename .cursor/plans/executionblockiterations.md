You are refactoring GRIM’s numeric reasoning system to an execution-first architecture.

This is a hard architectural change. Do not preserve legacy behavior. Do not add compatibility layers.

Implementation checklist and file-level notes: [.cursor/plans/execution-first_numeric_refactor_6c99d23f.plan.md](execution-first_numeric_refactor_6c99d23f.plan.md).

---

# CORE INVARIANTS (FAIL IF VIOLATED)

## 1. No value prediction exists anywhere in the system.

* Remove NumericHead completely.
* Remove all regression losses on numeric values.
* No `hidden_state → scalar` mapping for answers.

If any path produces a numeric **result** without going through ExecutionBlock → **FAIL**.

## 2. ExecutionBlock is the only source of numeric truth.

All numeric updates must follow:

`(state₀, op, args) → ExecutionBlock → state₁`

**Correct value with incorrect op or args is still WRONG.**

## 3. Structured prediction only.

Per step the model outputs logits (then argmax or sampled indices):

| Head | Shape | Meaning |
|------|--------|---------|
| `op_logits` | `[num_ops]` | Operation |
| `arg1_logits` | `[num_slots]` | First operand slot |
| `arg2_logits` | `[num_slots]` | Second operand slot |
| `write_slot_logits` | `[num_slots]` | Result destination slot |

**Loss (base, per step):**

`L = CE(op) + CE(arg1) + CE(arg2) + CE(write_slot)`

**No numeric loss term. Ever.**

## 4. Step-wise supervision is mandatory.

The dataset must provide **ordered** steps:

`step_k: (op, arg1, arg2, write_slot)` with teacher targets for each head.

Do **not** collapse multi-step problems into single-step supervision.

---

# INPUT AND SLOTS

* Literals enter **only** at parse time: bind each `<NUM>` to a slot and fill `numeric_values` / `token_to_slot_map`.
* During **generation**, slots come **only** from **input** or from **execution write-back**. **No dynamic slot creation** (no new slot indices invented by the sampler).
* `<NUM>` **must** always map to a bound slot; if not → **throw**.

---

# EXECUTION-GROUNDED LEARNING

After each supervised step:

`v_out = ExecutionBlock(op, arg1, arg2)` (using current slot state and `write_slot` as implemented in ExecutionBlock).

Compare `v_out` to `expected_value` for that step (from the teacher trace). This comparison **gates** routing losses below—it is **not** a regression target for a value head.

---

## Step X — Execution-consistency loss (REQUIRED)

If `v_out != expected_value`:

* `CE(arg1) *= α`
* `CE(arg2) *= α`

**Constraint:** `α > 1`, configurable.

**Purpose:** tie numeric correctness to **operand slot** selection.

Still **cross-entropy on logits**, not MSE on `v_out`.

---

## Step Y — Joint structured loss (REQUIRED — choose one and document)

**Option A — joint head**

* `L_joint = CE(joint_logits, target_joint)` where `joint_logits` is a single discrete distribution over a feasible encoding of `(op, arg1, arg2)` and `target_joint` is the teacher class index.
* Weight `λ_joint` in total loss if used.

**Option B — preferred (no extra head)**

If execution is incorrect (`v_out != expected_value`):

* `CE(op)`, `CE(arg1)`, `CE(arg2)`, `CE(write_slot)` each `*= β` for that step (or a documented per-head schedule).

**Constraints:**

* `β ≥ α` (full-head bump must be at least as strong as Step X’s operand bump, unless you explicitly replace Step X).
* **Do not** silently stack Step X and Step Y-B; document **stack vs replace** in config.

**Purpose:** penalize factorized errors (e.g. right op, wrong slots) that look good under independent CE.

---

## Step Z — Generation execution loop (MUST MATCH TRAINING)

At **every** decode step:

1. Run the **structured head** (same heads as training).
2. **Validate** `(op, arg1, arg2)` [and `write_slot` per your mask rules].

**If valid:**

* Execute **immediately** (ExecutionBlock).
* Write result to the chosen slot; update numeric state.
* **Bind** the next emitted `<NUM>` to that slot (`token_to_slot_map`, `numeric_values`).

**Then** continue LM token generation (`forwardStep` / sampling).

**If invalid:**

* **Fail hard** (no silent skip, no corrective execute).

Execution is **interleaved** with token generation—not batched only at end of sequence.

---

# SCRATCHBLOCK CONSTRAINT

For **arithmetic-tagged** batches:

* Disable **all** numeric-derived features (magnitude, sign, value-shaped embeddings, etc.).
* Only allow **type** embedding: “this token is a number.”

If magnitude information leaks through hidden state → **FAIL**.

---

# EXECUTION DEPENDENCY

If arithmetic is required for the batch:

* ExecutionBlock **must** run.
* No fallback, no approximation path that skips execution.

If execution is skipped or disabled → **training step fails**.

---

# INVALID PREDICTIONS

If:

* invalid `op`, or
* invalid slot index (out of range / masked slot),

Then:

* **Do not** execute.
* **Do not** correct silently.
* Apply **full penalty** on the structured heads (training); at inference **fail hard**.

No silent recovery.

---

# NUMERIC TOKEN POLICY

* Mask all **literal** numeric vocabulary IDs; only `<NUM>` is allowed for numeric surface form.
* Do **not** supervise numeric **answers** via the LM head.
* `<NUM>` must map to a slot (see INPUT AND SLOTS).

---

# NUMERIC STATE MODEL

Maintain explicit state each step, e.g.:

`state_k = { slots: [v₀, v₁, …], transitions: [(op, args → result_slot, v_out), …] }`

The model learns **transitions** (op + slot choices), not free-form number prediction.

**Slot identity:** two slots are distinct even if they hold the same value; do not merge slots by value.

---

# OPTIONAL — SEQUENCE CONSISTENCY

After the **full** sequence (multi-step example), optionally compare **final** slot value to ground truth; if mismatch, apply an **additional** penalty across the step losses to reinforce long chains. This does **not** reintroduce value regression on hidden states—it is an extra signal on top of structured CE + Steps X/Y.

---

# NO BACKWARDS COMPATIBILITY

* No dual code paths (legacy numeric head vs new stack). Old behavior is **removed**, not feature-flagged beside the new path.
* Old checkpoints either **refuse to load** with a clear error or require an explicit **non-default** migration tool—not silent weight drops.
* No `predictNumericValue`, no `kernelNumericLoss`, no generation fill-in for `<NUM>` from hidden state.
* DataLoader and batch code: **throw** on violation; do not `cerr` and continue.

Full rationale and file-level ownership: [execution-first_numeric_refactor_6c99d23f.plan.md](execution-first_numeric_refactor_6c99d23f.plan.md).

---

# FILE INTENT (SEPARATION)

Single **primary owner** per concern (cross-check at boundaries is OK; duplicated business logic is not):

| Concern | Owns it |
|--------|---------|
| Batch validity, side-channel lengths, slot map consistency | `BatchPayload` (+ builder) |
| Literal → `<NUM>` + slot assignment in corpus | `DataLoader`, `UniByte`, `AtomTable` |
| Config surface | `grim_language_model_cuda.hpp`, `ai_config.json` |
| Device caches | `TrainingState_GPU`, init training/inference state |
| Forward order, autograd, structured + LM outputs | `AutogradTraining.cu` |
| Training batch entry | `ComputeLossBatch.cu` |
| Deterministic math on slots | `execution_block_GPU.*` |
| “Type only” vs value injection | `ScratchBlockReasoning_GPU.*` |
| Prefill / step forward | `Inference_GPU.cu` |
| Step Z orchestration | `grim_language_model_gpu.cu` |
| Literal masking at sample time | `Sampling.cu` |
| Checkpoint format | `grim_model_serialization.cu` |

Structured head implementation: **new module or explicit extension**—do not hide structured logits inside ExecutionBlock as the only API if training loss lives elsewhere without a single contract.

---

# HARD-FAIL ERROR LOGIC

**Throw** (e.g. `std::runtime_error`) on contract breaks. **Do not** clamp, substitute, or skip.

| Situation | Action |
|-----------|--------|
| Side-channel length mismatch vs `token_ids` | Throw |
| `<NUM>` / atom without valid slot in map | Throw |
| Arithmetic batch but ExecutionBlock off or null | Throw before step |
| Missing or malformed per-step teacher sequence | Throw |
| Step Z: invalid structured prediction at decode | Throw (no execute) |
| `<NUM>` emitted without bound slot | Throw |
| Slot id outside `[0, num_slots)` or “new” slot at decode | Throw |
| ScratchBlock value features on arithmetic batch | Throw (misconfig) |
| ExecutionBlock internal numeric / slot error flag after step | Throw |

**Not a throw:** `v_out != expected` in training → apply Step X / Y loss scaling (learning signal).

---

# FORBIDDEN

Do **not**:

* Reintroduce NumericHead or any scalar value head for answers.
* Add numeric regression anywhere on predicted vs target **values**.
* Let the LM emit raw numeric strings (mask literals).
* Add execution fallback or “fix up” invalid ops/slots by executing a default.
* Collapse multi-step teacher traces into one step.

---

# REQUIRED IMPLEMENTATION

1. Remove all numeric **value** supervision (`kernelNumericLoss`, etc.).
2. Implement structured head: `op_logits`, `arg1_logits`, `arg2_logits`, `write_slot_logits` with shapes above.
3. Implement **Step X** (amplify `CE(arg1)`, `CE(arg2)` when `v_out != expected`).
4. Implement **Step Y** (document Option A vs B; resolve stacking with Step X).
5. Enforce ScratchBlock gating for arithmetic batches.
6. Enforce execution dependency (fail if execution off).
7. Implement step-wise dataset / batch format.
8. Refactor **Step Z** generation: interleave structured head → execute → bind → LM step.
9. Enforce numeric token masking and `<NUM>`–slot binding.
10. Fail-hard validation everywhere (batch build, forward, generate).

---

# SUCCESS CONDITION

The model should only be **correct** when:

* correct **operations**,
* correct **operand and write slots**,
* correct **execution sequence** (Step Z in sync with training).

If it can guess the **answer** without the **correct** execution path → implementation is incorrect.

If training destabilizes → **fix supervision and wiring**, not these constraints.
