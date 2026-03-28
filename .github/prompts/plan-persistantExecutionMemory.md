You are modifying the existing ExecutionBlock implementation.
Do NOT redesign the system. Do NOT change structure.
Apply ONLY the following targeted corrections.

---

## OBJECTIVE

Fix the mismatch between:

* soft training path
* hard execution path

Ensure:

1. Causal consistency between training and execution
2. Write head produces deterministic behavior
3. Trace/state becomes REQUIRED for decision making

---

## FIX 1 — DUAL TRANSITION SUPERVISION (CRITICAL)

### PROBLEM

Current:
transition_loss = |v_soft - target|

But execution uses v_out (hard), not v_soft.

### CHANGE

Replace transition loss with:

L_transition =
λ_soft * |v_soft - target|

* λ_hard * |v_out - target|

### IMPLEMENTATION

Inside executeStep:

* Keep existing v_soft computation
* Add:

Tensor hard_transition_loss = Tensor::zeros({1,1}, stream);

kernelAbsDiff<<<1,1,0,stream>>>(
hard_transition_loss.data,
v_out.data,
target_ptr,
nullptr, 0, 0.0f
);

* Modify aggregation:

transition_loss =
lambda_soft * soft_transition_loss

* lambda_hard * hard_transition_loss

Use config:
config_.lambda_soft_transition
config_.lambda_hard_transition

DEFAULT:
lambda_soft = 1.0
lambda_hard = 1.0

---

## FIX 2 — WRITE HEAD STRAIGHT-THROUGH ESTIMATOR (CRITICAL)

### PROBLEM

Write selection is:

* forward: argmax
* backward: softmax

This creates non-deterministic learning.

### CHANGE

Introduce STE for write distribution.

### IMPLEMENTATION

After computing p_write:

1. Compute one-hot:

int write_idx = argmax(p_write)

Tensor p_write_hard = zeros_like(p_write)
p_write_hard[write_idx] = 1

2. Replace gradient path:

p_write = stop_grad(p_write_hard - p_write) + p_write

3. Use:

* p_write_hard for forward execution
* p_write (STE version) for gradients

---

## FIX 3 — MAKE TRACE/STATE MANDATORY INPUT (CRITICAL)

### PROBLEM

Current:
context' = context + trace_vec + trace_state

This allows model to ignore trace.

### CHANGE

Replace ADD with CONCAT.

### IMPLEMENTATION

Replace:

context_prime = add(add(context, trace_vec), trace_state)

WITH:

Tensor decision_input = concat(context, trace_vec, trace_state)

Then:

* Replace ALL uses of context/context_prime in:

  * arg selection
  * op selection
  * write head

With decision_input

### REQUIRED DIMENSION UPDATE

Update:
W_op_select_
w_arg1_select_
w_arg2_select_

Input dim becomes:
3 * d_model

DO NOT change output shapes.

---

## FIX 4 — WRITE OVERWRITE PENALTY

### PROBLEM

Model can overwrite existing valid slots without penalty.

### CHANGE

Add penalty when writing to already valid slot.

### IMPLEMENTATION

Before write:

Tensor overwrite_penalty = zeros({1,1})

if (M.valid_mask[write_slot] == 1):
overwrite_penalty = 1.0

Add to loss:

L_exec += config_.overwrite_penalty_weight * overwrite_penalty

---

## FIX 5 — ARGUMENT DUPLICATION PENALTY

### PROBLEM

Model can select same slot twice (v1 == v2)

### IMPLEMENTATION

After arg selection:

if (arg1_slot == arg2_slot):
duplicate_penalty = 1.0
else:
duplicate_penalty = 0.0

Add to loss:

L_exec += config_.arg_duplicate_penalty_weight * duplicate_penalty

---

## FIX 6 — TRACE STATE NORMALIZATION

### PROBLEM

trace_state grows unbounded.

### IMPLEMENTATION

After:
trace_state = trace_state + encoded_step

Add:

kernelL2Normalize<<<1,1,0,stream>>>(
trace_state.data,
d_model
)

---

## FIX 7 — INJECTION GATE INITIALIZATION

### PROBLEM

sigmoid(0) = 0.5 → too strong early injection

### CHANGE

Initialize gate bias negative.

### IMPLEMENTATION

In constructor:

Initialize w_inject_gate_ with small negative bias:

for each j:
w_inject_gate_[j] = -2.0f

This gives:
sigmoid ≈ 0.12 initial gate

---

## FIX 8 — LOSS NORMALIZATION (STABILITY)

### PROBLEM

Loss terms can dominate unevenly.

### IMPLEMENTATION

For each loss component L_i:

L_i = L_i / (mean(L_i over batch) + 1e-6)

Apply BEFORE aggregation.

---

## CONSTRAINTS

* Do NOT change kernel structure
* Do NOT remove any existing validation
* Do NOT alter ExecutionMemory layout
* Do NOT introduce new global systems
* All changes must remain GPU-compatible
* Preserve fail-hard behavior

---

## SUCCESS CONDITION

After patch:

1. v_soft ≈ v_out during training
2. Write head produces stable deterministic slot selection
3. Model behavior depends on trace_state (cannot ignore it)
4. No multi-slot mutation violations
5. Loss remains stable across steps

---

Apply ONLY these changes. No extra improvements.
