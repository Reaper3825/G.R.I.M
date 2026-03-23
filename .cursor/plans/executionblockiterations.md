You are fixing a CUDA-based differentiable execution layer (`ExecutionBlockLayer`). The system is already functional, but it has critical architectural and gradient flow issues that must be corrected without changing overall structure.

Apply the following fixes precisely.

---

## 1. Make context fully differentiable

Current implementation:

* `kernelComputeContext` computes mean of H outside autograd.

Problem:

* This breaks gradient flow from op selection back into H.

Fix:

* Remove `kernelComputeContext`.
* Replace with autograd-based mean reduction over H:

  * context = mean(H, dim=0) → shape [1, d_model]
* Ensure this participates in autograd graph.

---

## 2. Remove detach in argument selection

Current implementation:

* arg logits are detached before masking:

  * `arg1_detached = arg1_logits.detach()`

Problem:

* This kills gradient flow into `w_arg*_select`.

Fix:

* Do NOT detach logits.
* Apply mask directly to logits:

  * masked_logits = logits + (1 - mask) * (-1e9)
* Then apply softmax to masked_logits.

---

## 3. Fix value decoding path (avoid wasted compute + gradient noise)

Current implementation:

* MLP decode runs on ALL candidates (atoms + memory)
* Then masked:

  * `atom_decoded = mlp_out * atom_mask`

Problem:

* Memory slots are unnecessarily passed through MLP
* Creates useless gradients and wasted compute

Fix:

* Split decode path:

  * For atom candidates → run MLP decode
  * For memory slots → directly use stored values
* Do NOT compute MLP on memory slots at all

---

## 4. Stabilize injection gate

Current implementation:

* sigmoid(H[slot] · w_gate)

Problem:

* Can saturate early → vanishing gradients
* Backward path is manually constructed → fragile

Fix:

* Add stabilization:

  * Option A: scale logits (e.g. * 0.5)
  * Option B: clamp pre-sigmoid values
  * Option C (preferred): use temperature-controlled sigmoid

Do NOT remove gate, only stabilize it.

---

## 5. Fix memory slot growth (enable real competition)

Current implementation:

* `M.num_filled = min(V, M.num_filled + 1)`

Problem:

* Forces sequential slot usage
* Prevents learned slot selection

Fix:

* Remove forced increment
* Let `p_write` determine slot usage
* `valid_mask` should control which slots are active

Optional:

* Set `num_filled = max(num_filled, index_of_max(p_write))`

---

## 6. Prepare op system for extensibility (do NOT fully implement yet)

Current implementation:

* Hardcoded 4 ops (+, -, *, /)

Problem:

* Not scalable
* Cannot support tool system or generalized reasoning

Fix:

* Abstract op selection:

  * Replace assumption `num_ops == 4`
  * Keep current ops but structure code so ops are index-based and extensible
* Do NOT implement new ops yet, only refactor structure

---

## Constraints

* Do NOT change overall execution flow
* Do NOT move logic out of GPU
* Do NOT introduce CPU syncs
* Maintain full differentiability
* Preserve existing tensor shapes unless explicitly required

---

## Goal

After fixes:

* All decision paths must be differentiable
* No unnecessary gradient noise
* Memory must be learnable, not sequential
* Execution block must behave as a true state-mutating layer inside the forward pass

---
