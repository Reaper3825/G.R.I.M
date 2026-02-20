# Training Math Correctness Audit

**Purpose:** Audit the training pipeline for mathematical correctness. Mode collapse / plateau should be traced to root causes in the math, not masked by heuristic patches.

**Last updated:** Feb 19, 2026

---

## 1. Loss and Gradient Chain (Forward → Backward)

### 1.1 Loss forward
- **Location:** `AutogradLoss.cu` — `kernelNLLLossForward`, `unified_loss()`
- **Formula:** `L = (1/N) * Σ_t -log(p_target[t])` for valid tokens (target ≠ -1, mask ≠ 0)
- **Valid count:** Computed in-kernel via `atomicAdd(valid_count, 1)` — single source of truth
- **Output:** `mean_loss = loss_sum / h_valid_count` scalar
- **Status:** ✓ Matches PyTorch `F.cross_entropy(..., reduction='mean')`

### 1.2 NLL backward (grad w.r.t. log_probs)
- **Location:** `AutogradLoss.cu` — `kernelNLLLossBackward`
- **Formula:** `grad_log_probs[t,v] = grad_v * inv_valid_count`
  - At target: `grad_v = -q_on` (plain CE: -1), so `grad_log_probs[t, target] = -1/N`
  - At non-target: `grad_v = 0` (or smoothing/focal terms)
- **inv_valid_count:** `1.0f / valid_count` — same N as forward
- **Status:** ✓ Correct mean reduction

### 1.3 LogSoftmax backward (grad w.r.t. logits)
- **Location:** `TensorContract_GPU.cu` — `kernel_log_softmax_backward`
- **Formula:** `grad_logits[i] = grad_log_p[i] - exp(log_p[i]) * Σ_j grad_log_p[j]`
- **Chain:** For CE, `Σ grad_log_p = -1/N` → `grad_logits[v] = (p_v - 1_{v=target}) / N`
- **Status:** ✓ Standard softmax Jacobian, matches PyTorch

### 1.4 Matmul backward (LM head: logits = h @ W^T)
- **Location:** `TensorContract_GPU.cu` — `MatMulGradFn::apply()`
- **Forward:** C = A @ B^T with A = h [T, D], B = W [V, D], C = logits [T, V]
- **grad_A (grad_input → encoder):** `grad_A = grad_C @ B` → [T, V] @ [V, D] = [T, D] ✓
- **grad_B (grad_W):** `grad_B = grad_C^T @ A` → [V, T] @ [T, D] = [V, D] ✓
- **Status:** ✓ Verified (transpose_b branch lines 4941–5083)

---

## 2. Gradient Scaling and Accumulation

### 2.1 Root gradient scale (Issue #149 — CRITICAL)
- **Location:** `Phase2_TrainingLoop.cu` line ~1603, `AutogradTraining.cu` — `loss.backward(nullptr, ctx.grad_scale)`
- **Formula:** `grad_scale = 1.0f / accum_steps`
- **Purpose:** With M micro-batches, each backward produces mean gradient. Summing them without scale gives `M × mean_grad`. Scaling root by 1/M yields correct accumulated mean.
- **Status:** ✓ Fixed (Feb 18). Verify `grad_scale` is passed through to `backward()`.

### 2.2 per_token_grad_scale (copilot #44)
- **Config:** `ai_config.json` → `per_token_grad_scale: true`
- **Note:** Doc says gradient RMS ~1e-6 is correct when valid_tokens ~3000. If disabled, effective LR becomes ~3000× larger. Ensure this is not a second scaling layer that could double-scale or conflict.

### 2.3 GQA gradient scaling (Q/K/V balance)
- **Location:** `TensorContract_GPU.cu` — `kernel_reduce_gqa_grads_BSHD_bf16_to_BHSD_fp32`
- **Setup:** FlashAttention backward outputs dK, dV per *query* head (12). For GQA (4 KV heads), we sum 3 Q-head gradients per KV head.
- **Issue:** Using `gqa_grad_scale = 1/heads_per_kv_group` (1/3) made ||dK||, ||dV|| = (1/√3)||dQ|| → **Q received ~1.7× larger gradient magnitude than K and V** for three roles that should be comparable.
- **Fix:** Use `gqa_grad_scale = 1/sqrt(heads_per_kv_group)` so that per-head gradient magnitudes for Q, K, V are comparable (no structural 1.7× imbalance). True backprop gradient for the shared KV parameter is the sum (no scale); we trade that for balanced updates.
- **Status:** ✓ Applied. Verify kernel uses `sqrtf(heads_per_kv_group)` in scale.

---

## 3. Weight Tying (Embedding + LM Head)

### 3.1 Memory aliasing
- **Location:** `lm_head_GPU.cu`, `buildParameterGroups` in `LanguageModel_Training.cu`
- **Setup:** When `tie_embeddings=true`, `lm_head.weights()` and `embedding.tokenWeights()` share data and grad via `share_grad()`.
- **Parameter groups:** Embedding is NOT registered; only `embedding_lm_head_tied` is registered. One optimizer update on shared buffer.
- **Status:** ✓ No double update (avoids double Adam step)

### 3.2 Gradient accumulation order

- **Doc (EMBEDDING_ARCHITECTURE.md):** LM head backward overwrites grad (beta=0); embedding backward accumulates (atomicAdd).
- **Order:** LM head backward runs as part of autograd chain; embedding backward runs from encoder input grad. Both write to same `grad` buffer. Must verify order and that combined grad = grad_LM + grad_embedding (no overwrite of one by the other).

### 3.3 Embedding scale (Issue #140)

- **Fix:** `embedding_scale = 1.0f` (was sqrt(d_model) = 27.7)
- **Reason:** sqrt(d_model) on embedding forward created 27.7× gradient asymmetry between embedding path and LM head path. Structural tokens received amplified embedding grads.
- **Status:** ✓ Removed. Verify `embedding_scale` is 1.0 in `AutogradTraining.cu` embedding forward.

---

## 4. AdamW Optimizer

### 4.1 Update formula

- **Location:** `AdamW_Kernel_GPU.cu`
- **Formula:** `params[i] = param - lr * (m_hat / sqrt(v_hat + ε) + weight_decay * param)`
- **Moments:** `m_new = β1*m + (1-β1)*g`, `v_new = β2*v + (1-β2)*g²`
- **Bias correction:** `m_hat = m_new / (1 - β1^t)`, `v_hat = v_new / (1 - β2^t)` (Issue #12: use ADAMW_BETA1, ADAMW_BETA2, not hardcoded 0.9/0.999)
- **Status:** ✓ Standard AdamW. Verify HyperParameters match kernel betas.

### 4.2 Weight decay effectiveness (PLATEAU_BUG doc)

- **Observation:** With small init (||W|| ~ 0.00023), `weight_decay_term = lr × wd × W ≈ 7e-10` vs `adam_update ~ 3e-4` → weight decay ~400,000× weaker.
- **Implication:** If math is correct, weight decay may be too weak to prevent norm growth. That is a hyperparameter choice, not a formula bug.

---

## 5. Centering / Pre-LM-Head Operations

### 5.1 center_columns, center_rows

- **Location:** `TensorContract_GPU.cu`, `lm_head_GPU.cu`
- **column centering:** `y[t,d] = x[t,d] - mean_t(x[:,d])` — removes shared direction across positions
- **row centering:** `y[t,d] = x[t,d] - mean_d(x[t,:])` — per-token mean subtraction
- **Backward:** Linear ops → backward is same centering on grad.
- **Status:** ✓ Issue #125/#132. Ensure backward uses correct centering (column vs row) for each op.

### 5.2 project_out_pc1

- **Location:** `TensorContract_GPU.cu` — `autograd::project_out_pc1`
- **Forward:** `h̃[t] = h[t] - (h[t]·g)*g` with g = PC1 via power iteration
- **Backward:** `grad_h = (I - gg^T) * grad_h̃` (g stop-gradient)
- **Status:** Autograd node exists. Not used when `center_hidden_states=true` (else-if in lm_head). Verify both can run when config has both (pipeline change from previous edit).

---

## 6. Areas to Audit for Potential Errors

| Area | Risk | Action |
|------|------|--------|
| **valid_count consistency** | Forward uses kernel atomicAdd; backward uses same value. | Confirm NLLLossGradFn receives `h_valid_count` from same forward run. |
| **Gradient sign** | Wrong sign → ascent instead of descent. | Finite-diff check in `kernelFiniteDiffGradVerify` — verify enabled and passing. |
| **Tied grad accumulation** | LM head overwrites, embedding adds. | Trace backward order; ensure combined grad = grad_LM + grad_emb. |
| **Bias correction (Adam)** | β2 mismatch (Issue #12) caused inconsistent LR. | Confirm `launchAdamWKernel` uses ADAMW_BETA1/BETA2 in `powf()`. |
| **Grad accumulation scale** | Bug #149: missing 1/accum_steps. | Confirm `grad_scale` flows from Phase2 → backward(root=grad_scale). |
| **Padding / valid_mask** | valid_mask=nullptr → mask=1.0 in kernel. | Padding must use target=-1. Ensure dataloader sets target=-1 for pad. |

---

## 7. Known Fixed Issues (Verify Still Applied)

- **Issue #149:** Gradient accumulation — `grad_scale = 1/accum_steps` at loss.backward()
- **Issue #140:** embedding_scale = 1.0 (no sqrt(d_model))
- **Issue #125:** center_columns (not center_rows) for reducing cos(h_i, h_j)
- **Issue #12:** Adam bias correction uses ADAMW_BETA1/ADAMW_BETA2
- **Issue #60:** Weight tying — do not add embedding and lm_head to optimizer separately
- **GQA Q/K/V balance:** `gqa_grad_scale = 1/sqrt(heads_per_kv_group)` so Q, K, V get comparable gradient magnitude (was 1/heads_per_kv_group → Q 1.7× larger)

---

## 8. Suggested Verification Steps

1. **Gradient check:** Run `kernelFiniteDiffGradVerify` on a few (token, vocab) pairs; ensure sign(analytical) == sign(FD).
2. **Loss scaling:** For one batch, compute `loss * valid_tokens` and compare to `sum(per_token_loss)`; should match.
3. **Grad magnitude:** After backward, sample `grad_logits` at target positions; expect `|grad| ≈ (1-p_t)/N` for plain CE.
4. **Tied buffer:** When tied, confirm `embedding.grad_data() == lm_head.grad_data()` and that only one parameter group references it.
