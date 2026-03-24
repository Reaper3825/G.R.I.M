You are modifying GRIM’s training system to enforce **execution-first numeric reasoning**.

The goal is to ensure the model CANNOT solve arithmetic via hidden state approximation and MUST rely on ExecutionBlock.

This is not optional. This is a hard constraint.

---

## GOAL

Force the model to learn:

(state, op, args) → execution → state'

NOT:

hidden_state → predicted number

---

## CORE RULE

Numeric correctness MUST come ONLY from ExecutionBlock.

The model must NOT be able to compute numeric results internally.

---

## STEP 1 — REMOVE ALL VALUE-BASED SUPERVISION

Delete all training paths that supervise numeric values.

Specifically:

* Remove NumericHead loss (kernelNumericLoss)
* Remove any regression on numeric_values
* Remove any objective that compares predicted value to ground truth

There must be ZERO gradient path encouraging:
→ “predict the number directly”

---

## STEP 2 — STRUCTURED SUPERVISION ONLY

Training targets must be:

For each step:

* op_target
* arg1_target
* arg2_target
* write_slot_target

Loss:

L_total =
CE(op_logits, op_target)

* CE(arg1_logits, arg1_target)
* CE(arg2_logits, arg2_target)
* CE(write_slot_logits, write_slot_target)

No numeric loss.

---

## STEP 3 — EXECUTION-GROUNDED CORRECTNESS

After each predicted step:

1. Execute:
   v_out = ExecutionBlock(op, args)

2. Compare against ground truth:

If v_out != expected_value:
→ apply penalty to (op, arg1, arg2)

Do NOT train value prediction.

All numeric correctness is enforced through:
→ correctness of op + slot selection

---

## STEP 4 — HARD DISABLE VALUE LEAKAGE

For arithmetic-tagged batches:

* Disable all value-based ScratchBlock injection:

  * remove log-magnitude features
  * remove sign features
  * remove numeric-derived embeddings

Only allow:

* type embedding (this is a number)

Hidden state must NOT contain enough information to reconstruct numeric values.

---

## STEP 5 — EXECUTION DEPENDENCY ENFORCEMENT

During training:

If correct answer requires execution:

* model MUST go through ExecutionBlock
* no shortcut paths allowed

If model produces correct value WITHOUT correct op:
→ treat as WRONG

Correctness is defined as:

* correct op
* correct arguments
* correct transition

NOT correct numeric output alone

---

## STEP 6 — STEP-WISE SUPERVISION (CRITICAL)

Each training example must include ordered steps:

Example:
"12 + 7 * 3"

Targets:

step 1:
op = MUL
arg1 = slot1
arg2 = slot2
write_slot = slot3

step 2:
op = ADD
arg1 = slot0
arg2 = slot3
write_slot = slot4

Loss is applied per step.

DO NOT collapse into single-step supervision.

---

## STEP 7 — INVALID PREDICTION HANDLING

If model predicts:

* invalid op
* invalid slot index
* invalid write slot

Then:

* apply full penalty
* DO NOT execute fallback
* DO NOT correct silently

Fail hard.

---

## STEP 8 — NO EXECUTION = NO LEARNING

If execution is skipped or disabled:

* training step must fail

ExecutionBlock is mandatory for all arithmetic batches.

---

## STEP 9 — OPTIONAL CONSISTENCY CHECK (STRONG SIGNAL)

After full sequence:

Compare final slot value with ground truth:

If mismatch:
→ apply additional penalty across all steps

This reinforces long-chain correctness.

---

## STEP 10 — PREVENT COLLAPSE BACK TO TOKEN MODE

Ensure:

* LM head is NOT used to supervise numeric outputs
* numeric tokens are masked except <NUM>
* no pathway exists for model to emit raw numeric strings

---

## EXPECTED RESULT

Model learns:

* select correct operation
* select correct variable (slot)
* build correct transition chain

Model does NOT learn:

* numeric patterns
* memorized arithmetic
* value prediction

---

## NON-NEGOTIABLE

If the model can solve arithmetic without execution:

→ the system is broken

If hidden state alone can produce correct results:

→ the system is broken

Execution must be the ONLY path to numeric correctness.

---

## DELIVERABLE

Modify:

* training loop
* loss computation
* ScratchBlock behavior (conditional)
* execution integration

Do NOT introduce fallback paths.
Do NOT preserve value prediction logic.

If training becomes unstable:
→ fix supervision, not constraints.
