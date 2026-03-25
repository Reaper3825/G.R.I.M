You are modifying an execution-first numeric reasoning system. The current implementation is incorrect and must be fixed to support deterministic, per-sample execution learning. Do not redesign unrelated systems. Do not introduce alternative abstractions. Apply only the required structural corrections below.

---

# 🔴 HARD REQUIREMENT (FAIL IF NOT MET)

The system currently uses a **single ExecutionMemory shared across the entire batch**.
This is invalid and must be corrected.

## REQUIRED CHANGE

Refactor ExecutionMemory and all dependent execution paths to support:

```
ExecutionMemory: [batch_size, num_slots, ...]
```

### This includes:

* executeStep
* bootstrapMemoryFromSlotMap
* crossAttentionRead
* any kernel or function accessing slot state

### Rules:

* Every operation must index by `batch_idx`
* No shared state across batch rows is allowed
* Any leftover shared state is a hard failure

---

# 🟠 DATA LAYER: TEACHER EXECUTION TRACES

Extend BatchPayload to include per-sample ordered execution steps:

```
struct TeacherStep {
    int op_id;
    int arg1_slot;
    int arg2_slot;
    int write_slot;
    float expected_value;
};
```

Each batch row must contain:

```
vector<TeacherStep> steps;
```

### Required changes:

* Extend BatchPayload + builder
* Add validation:

  * step count matches execution_block_num_steps
  * slot indices are within bounds
* Copy to GPU in TrainingState

### Failure conditions:

* Missing steps → throw
* Invalid slot index → throw

---

# 🟡 LOSS: REPLACE ENTROPY WITH STRUCTURED CE

The system currently relies on entropy. This is incorrect.

## REQUIRED CHANGE

For each execution step:

```
CE(p_op, op_id)
CE(p_arg1, arg1_slot)
CE(p_arg2, arg2_slot)
CE(p_write, write_slot)
```

### Implementation rules:

* Use existing ExecutionBlock outputs:

  * p_op
  * p_arg1
  * p_arg2
  * p_write
* Do NOT create a new head
* Map slot indices correctly into logits

### Entropy:

* Remove as primary loss
* Keep only as optional small auxiliary (low weight)

---

# 🔴 REMOVE CONFLICTING SYSTEM

ReasoningHead operates on atoms, not slots. This conflicts with execution-first design.

## REQUIRED CHANGE

EITHER:

* Remove ReasoningHead from arithmetic path

OR:

* Gate it behind config so it never runs during execution training

### Rule:

There must be only ONE source of execution decisions.

---

# 🟢 EXECUTION CONSISTENCY (STEP X / Y)

After each step:

```
if (v_out != expected_value)
```

Apply penalties:

## Step X:

* Multiply CE(arg1) and CE(arg2)

## Step Y:

* Multiply ALL CE terms (op, arg1, arg2, write)

### REQUIRED:

* Implement BOTH
* Add config:

  * step_x_multiplier
  * step_y_multiplier
  * step_y_overrides_x (bool)

### Behavior:

* If override = true → only Step Y applies
* Else → multipliers stack

---

# 🔵 GENERATION (STEP Z)

Fix generation so execution actually runs during decoding.

## REQUIRED LOOP PER TOKEN:

1. Run forward → get hidden state
2. Decode:

   ```
   (op, arg1, arg2, write) = argmax from execution logits
   ```
3. Execute deterministically → update ExecutionMemory
4. When `<NUM>` is generated:

   * Resolve value from bound slot
   * Populate numeric output
   * DO NOT throw

---

## REQUIRED CHANGES

* Persist ExecutionMemory across generation steps
* Reset only when sequence resets
* Remove `<NUM>` throw path
* Enforce:

  * invalid op/arg → hard failure

---

# 🔴 STRICT FAILURE POLICY

The system must fail hard in all of the following:

* Shared execution state across batch
* Missing teacher steps
* Invalid slot indices
* Invalid execution operations during training or inference
* `<NUM>` generated without slot binding

No warnings. No fallbacks. No silent correction.

---

# 🚫 DO NOT

* Do not reintroduce NumericHead
* Do not create duplicate reasoning systems
* Do not bypass execution with regression
* Do not allow soft execution (must be deterministic)

---

# ✅ SUCCESS CONDITION

The system is correct when:

* Each batch sample executes independently
* ExecutionBlock decisions are directly supervised via CE
* Execution state evolves deterministically per step
* Generation produces `<NUM>` via slot binding, not prediction
* No shared or ambiguous execution state exists

---

Implement exactly this. No simplifications. No substitutions.
