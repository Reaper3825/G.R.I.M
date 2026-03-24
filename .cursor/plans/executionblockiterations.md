# FINAL EXECUTION SYSTEM IMPLEMENTATION — STRICT ENFORCEMENT PROMPT

Reference plan: 

This is NOT a design task.
This is a **surgical implementation task**.

You are enforcing an already-defined system.

---

## PRIMARY GOAL

Make the codebase match EXACTLY this invariant:

```text
tokens → token_to_slot_map → ExecutionMemory.values
→ ExecutionBlock (op_table) → updated values
→ StateEncoder(values) → state
```

No deviations. No partial compliance.

---

## GLOBAL ENFORCEMENT RULE

If any code path allows:

* numeric values from hidden state
* numeric values from tokenizer after bootstrap
* numeric values outside ExecutionMemory.values

→ DELETE IT

Do NOT adapt. Do NOT fallback.

---

## STEP 1 — HARD DETECTION PASS (DO THIS FIRST)

Search entire codebase and identify ALL instances of:

* token_numeric_values used outside bootstrap
* numeric MLP decode paths
* hidden → float conversion for execution
* scratch slot numeric usage
* multiple write paths to “state”

For each occurrence:

```text
IF used for execution → DELETE
IF ambiguous → ERROR and stop
```

Do not proceed until all violations are removed or accounted for.

---

## STEP 2 — TOKEN → SLOT SYSTEM (MANDATORY)

Implement:

```text
int32_t token_to_slot_map[total_tokens]
```

Rules:

* `<NUM>` MUST map to valid slot
* non-state tokens → -1
* mapping MUST persist across forward + autoregressive steps

Thread through:

* BatchPayload
* TrainingState
* InferenceState
* CUDA kernels

If any execution path lacks slot mapping:

→ THROW (fail hard)

---

## STEP 3 — BOOTSTRAP (ONLY LITERAL ENTRY POINT)

After ExecutionMemory init:

```text
if slot_id >= 0:
    M.values[slot_id] = literal
```

Constraints:

* training: detached
* inference: prompt only

After this:

```text
LITERALS MUST NEVER BE USED AGAIN FOR EXECUTION
```

Any later usage → FAIL

---

## STEP 4 — EXECUTION BLOCK (CORE REWRITE)

Inside `executeStep`:

### REPLACE ALL VALUE SOURCES WITH:

```text
slot_i = token_to_slot_map[arg1]
slot_j = token_to_slot_map[arg2]

v_i = M.values[slot_i]
v_j = M.values[slot_j]
```

If:

* slot invalid
* slot == -1
* slot out of range

→ THROW

---

### EXECUTION (NON-NEGOTIABLE)

```text
v_out = op_table[op_id](v_i, v_j)
```

Constraints:

* NO gradients
* NO hidden state involvement
* NO approximation

---

### WRITE (ONLY PATH)

```text
target_slot ∈ [S .. V-1]

M.values[target_slot] = v_out
```

If:

* write touches scratch
* write happens elsewhere
* blending affects scratch

→ FAIL

---

## STEP 5 — STATE SYSTEM (STRICT)

### DELETE:

* state write heads
* state logits
* scratch-based state storage

---

### IMPLEMENT:

```text
state0 = StateEncoder(M.values, mask)
state1 = StateEncoder(M.values_updated)
```

State must NEVER be:

* stored in memory
* written to scratch
* predicted directly

---

## STEP 6 — SCRATCHBLOCK CORRECTION

In:

`kernelLookupAtomEmbeddingsWithValue`

If:

```text
slot_id >= 0
```

Then:

```text
has_value = false
```

REMOVE:

* numeric literal injection into embedding bands

ADD:

```text
embedding += slot_embedding[slot_id]
```

ScratchBlock must NOT contain numeric truth

---

## STEP 7 — GATHER + ARG DOMAIN

Candidates:

```text
atoms + value_slots_only
```

EXCLUDE:

* scratch slots
* state embeddings

Arg heads must operate ONLY on this set

---

## STEP 8 — NUMERIC HEAD (CONTAINMENT)

NumericHead is:

* decode only
* optional supervision

It MUST NOT:

* influence execution
* provide intermediate values
* act as truth source

Inference must read:

```text
M.values[result_slot]
```

---

## STEP 9 — AUTOGRAD BOUNDARY

Enforce:

```text
NO gradient through:
    op_table
    M.values writes
```

If gradients are detected crossing this boundary:

→ FAIL

---

## STEP 10 — FAIL-HARD VALIDATION

Add explicit runtime checks:

* `<NUM>` without slot → ERROR
* slot out of bounds → ERROR
* execution using hidden decode → ERROR
* write to scratch → ERROR
* dual write paths → ERROR

No warnings. No silent fixes.

---

## FINAL VERIFICATION CHECKLIST

System is correct ONLY IF:

* arithmetic is exact (bit-consistent)
* same inputs → identical outputs
* removing NumericHead does NOT break reasoning
* removing ScratchBlock numeric bands does NOT break execution
* multi-step reasoning uses ONLY slot updates

---

## IMPLEMENTATION RULE

If something conflicts with this system:

```text
DELETE IT
```

Do NOT:

* preserve compatibility
* keep legacy paths
* introduce fallbacks

---

## END STATE

You are not building a “model that predicts numbers.”

You are building:

```text
A deterministic execution engine controlled by a transformer
```

If the transformer disappears, execution must still be correct.

If execution disappears, the system must fail.

---

Execute exactly as specified.
