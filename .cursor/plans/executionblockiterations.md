Good questions. Implement exactly as follows:

---

## Fix 1 — Differentiable context

Use the matmul approach.

* Build a constant `[1, total_tokens]` vector with values `1 / total_tokens`
* Compute:

  * `context = matmul(avg_vec, H)` → shape `[1, d_model]`
* Do NOT add a new `autograd::mean` op
* Ensure this stays inside the autograd graph

---

## Fix 2 — Remove detach in arg selection

Do NOT use `.detach()` anywhere in arg selection.

* Apply masking directly to logits:

  * `logits += (1 - mask) * (-1e9)`
* Then pass logits into `autograd::softmax`

Do NOT create a custom autograd mask op.

---

## Fix 3 — Split decode path

Change decode to:

1. Run MLP ONLY on atom rows `[0 → num_atoms)`
2. Memory rows `[num_atoms → C)` use `M.values` directly
3. Rebuild `[C,1]` via concatenation:

   * `decoded_values = concat(atom_decoded, mem_values)`

Keep ordering: atoms first, memory second.

---

## Fix 4 — Stabilize injection gate

* Add a **config hyperparameter**: `inject_gate_temp`
* Forward:

  * `gate = sigmoid((H · w_gate) * inject_gate_temp)`
* Update backward kernel to include `inject_gate_temp` in gradient:

  * multiply derivative by `inject_gate_temp`

Do NOT make this parameter learnable.
Do NOT move this to autograd.

---

## Fix 5 — Memory slot growth

Use Option A.

* Remove:

  * `M.num_filled = min(V, M.num_filled + 1)`
* Cross-attention must operate over ALL `V` slots
* Slot visibility is controlled ONLY by `valid_mask`
* Do NOT introduce host-side counting or thresholds

---

## Fix 6 — Op extensibility

* Change validation:

  * from `num_ops == 4`
  * to `num_ops > 0`

* Replace hardcoded `4` with `config_.num_ops` where applicable

* Keep current 4 ops implementation intact

* Do NOT introduce dispatch tables or function pointers yet

---

## Constraints

* No CPU sync
* No structural rewrites
* No moving logic out of GPU
* Keep tensor shapes consistent
* Preserve current execution flow

---

Implement only these changes. Do not refactor unrelated code.
