# Fact-Check: "Rank Collapse of the Weight Gradient" Statement

**Source:** User-provided statement (algebra of encoder weight gradient → rank-1 → trap).  
**Checked against:** Codebase (encoder backward, RMSNorm, matmul grad), `equation_trace.txt`, `BACKWARD_PASS_ARCHITECTURE.md`, `PLATEAU_BUG_INVESTIGATION.md`.

---

## 1. Weight gradient formula

**Claim:** ∇W^l = (∂L/∂output)^T × RMSNorm(h^l), where RMSNorm(h^l) ∈ R^{N×D} is the layer input (activation matrix).

**Verdict: ✓ Correct.**

- Encoder is **pre-norm**: first sublayer does `ln1_out = RMSNorm(input)`, then `qkv_out = ln1_out @ W_qkv^T` (Encoding_GPU.cu:522–549).
- Matmul backward: `grad_W = grad_output^T @ input` (TensorContract_GPU.cu, BACKWARD_PASS_ARCHITECTURE.md). So for the QKV projection, **grad_W_qkv = grad_qkv^T @ ln1_out** = (upstream grad)^T @ RMSNorm(h). Same idea for other encoder linears (FFN, etc.): weight gradient uses the **cached layer input**, which is RMSNorm(residual) at each sublayer.

---

## 2. When ρ ≈ 0.99, activation matrix is rank-1

**Claim:** When ρ ≈ 0.99, every row of the activation matrix points in approximately the same direction â; RMSNorm(h^l)[t,:] ≈ c_t · â; so the activation matrix is rank-1.

**Verdict: ✓ Correct.**

- RMSNorm scales each row by a scalar (1/rms), so it does **not** change row directions. If hidden states are nearly aligned (high pairwise cosine ρ), then h[t,:] ≈ α_t · g for some g. After RMSNorm, rows remain proportional to g (up to per-row scale), so all rows ≈ c_t · â for a single direction â. Hence the matrix is rank-1.

---

## 3. Weight gradient collapses to rank-1

**Claim:** ∇W^l = (∑_t g[t]) × â^T (rank-1), regardless of diversity of g[t] = ∂L/∂output[t].

**Verdict: ✓ Correct in spirit; one small precision.**

- With activation rows A[t,:] = c_t · â, we have (∇W)[i,j] = Σ_t g[t,i] A[t,j] = â[j] · Σ_t c_t g[t,i]. So ∇W = **u** ⊗ â^T with **u**[i] = Σ_t c_t g[t,i]. So the left factor is **(Σ_t c_t · g[t])**, not (Σ_t g[t]). If c_t ≈ const (e.g. ≈ 1 after norm), then it’s ≈ (Σ_t g[t]) × â^T. So the rank-1 conclusion is correct; the exact formula uses c_t-weighted sum.

---

## 4. Rank-1 update cannot create diversity

**Claim:** Δoutput[t] = ΔW × input[t] = u × (â·input[t]); with input[t] ≈ c_t·â, this is ≈ c_t · constant_vector, so all positions move in the same direction.

**Verdict: ✓ Correct.**

- ΔW = u ⊗ â^T ⇒ Δoutput[t] = ΔW @ input[t] = u (â·input[t]). If input[t] ∝ â, then â·input[t] is a scalar, so Δoutput[t] ∝ u for all t. So the update is one global direction; it cannot by itself create new directions.

---

## 5. Self-consistency trap

**Claim:** Encoder needs diverse activations for high-rank weight gradient and high-rank gradient for diverse activations; once ρ is ~0.8–0.9, rank-1 approximation holds and neither can bootstrap the other.

**Verdict: ✓ Consistent with code and trace.**

- equation_trace.txt shows final avg_cos = 0.7973 (batch 1), and trajectory ρ → 0.97+, PC1 variance 33% → 95%+; encoder grad share drops (e.g. 53% → 2%). So high ρ and rank-dominated dynamics are observed; the “trap” description is consistent.

---

## 6. AdamW does not fix rank-1 gradient

**Claim:** For ∇W[i,j] ≈ u[i]a[j], AdamW gives update[i,j] ≈ sign(u[i])·sign(a[j]), still rank-1 (outer product of sign vectors).

**Verdict: ✓ Reasonable.**

- AdamW normalizes per element; if gradient is u⊗a, then (in the limit of that structure) m and v inherit that structure, so the update remains an outer product of two vectors, hence rank-1. Magnitude is normalized, rank is not.

---

## 7. Position-specific gradient gets summed away

**Claim:** LM head backward is position-specific (different W[target[t]] per t), but ∇W^l = Σ_t g[t] × input[t]^T; when input[t] ≈ â for all t, this becomes (Σ_t g[t]) ⊗ â^T, so position-specific information is lost in the sum.

**Verdict: ✓ Correct.**

- equation_trace.txt (lines 361–366) derives the same: when all h[t] ≈ same direction, g_tracked and g_other both ∝ h̄, so they cancel and ||∇W[36]|| shrinks (1.60 → 0.20). So the “diverse signal destroyed by summation over rank-1 activation” is correct.

---

## 8. Production LLMs (D=4096, LayerNorm, batch, vocab)

**Claim:** Production transformers avoid this due to D=4096+ (ρ baseline 1/D ≈ 0.016), LayerNorm (GPT-2), larger batch, larger vocab.

**Verdict: ✓ Mostly correct; one typo.**

- **Typo:** Baseline correlation for random unit vectors in D dimensions is **1/√D**, not 1/D. So for D=4096, baseline ≈ 1/√4096 = 1/64 ≈ **0.016**. The number 0.016 is correct; the formula should read **ρ = 1/√D ≈ 0.016**, not 1/D.
- **LayerNorm:** GPT-2 uses LayerNorm (mean subtraction), which can reduce shared component; GRIM uses RMSNorm (no mean subtraction), so the statement is correct.
- **Larger D, batch, vocab:** All are plausible reasons production models have lower ρ and less rank collapse.

---

## 9. Your model: ρ(0)=0.21, 12 layers → 0.80

**Claim:** Model at D=768 has ρ(0)=0.21 (6× above isotropic baseline 0.036), and 12 layers of causal attention push it to ~0.80 before training.

**Verdict: ✓ Supported by equation_trace.txt.**

- **ρ(0) = 0.210:** equation_trace.txt line 573: “MEASURED: ρ(0) = 0.210 (forward-only with random weights)”. Line 597: isotropic baseline 1/√768 = 0.036; line 598: “MEASURED: avg|cos| = 0.210”. So 0.21 ≈ 6× 0.036. ✓
- **Final ρ:** Trace reports “Layer 0 output: avg_cos ≈ 0.257” and “Final output: avg_cos = 0.7973” (lines 65–66). So “push to 0.80” is correct. (Different runs show different trajectories; e.g. line 576: ρ 0.210 → … → 0.971.)

---

## 10. Fix must be inside the encoder

**Claim:** The fix has to prevent ρ from rising (not just project out after the fact).

**Verdict: ✓ Consistent with docs.**

- equation_trace.txt (e.g. line 756) and Issue #149 indicate that projecting PC1 out *after* the encoder crushes ||h⊥|| and doesn’t fix the encoder dynamics; fixes need to operate inside the encoder (e.g. per-layer decorrelation, spectral norm, etc.).

---

## Summary

| Claim | Verdict |
|-------|--------|
| ∇W^l = (∂L/∂output)^T × RMSNorm(h^l) | ✓ Correct |
| ρ ≈ 0.99 ⇒ activation matrix rank-1 | ✓ Correct |
| ∇W^l rank-1: (∑_t g[t])×â^T | ✓ Correct (exact: (∑_t c_t·g[t])×â^T) |
| Rank-1 update ⇒ same direction for all t | ✓ Correct |
| Self-consistency trap | ✓ Consistent with trace |
| AdamW preserves rank-1 structure | ✓ Reasonable |
| Position-specific grad summed away | ✓ Correct |
| Production: 1/D ≈ 0.016 | ⚠ Typo: should be **1/√D** ≈ 0.016 |
| ρ(0)=0.21, 12 layers → 0.80 | ✓ In trace |
| Fix inside encoder | ✓ Consistent with docs |

**Bottom line:** The statement is **mathematically and empirically sound** modulo (1) the small precision on the left factor of ∇W^l (c_t-weighted sum), and (2) the baseline correlation formula (1/√D, not 1/D). The rank-collapse story and the need for encoder-internal interventions are well supported by your code and equation trace.
