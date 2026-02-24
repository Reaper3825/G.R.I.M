# Rank-1 effect on the training loop (math only)

Notation and recurrence only. No causal or semantic claims.

---

## 1. Notation

- Layer index: `l` in {0, ..., L-1}
- Batch: `N` tokens, feature dimension `D`
- Pre-norm linear (e.g. QKV): layer input `X^l` in R^(N x D), weights `W^l`, output `Y^l = X^l (W^l)^T`
- `X^l` = RMSNorm(h^l) row-wise; `h^l` = residual input to layer l
- Loss L; upstream gradient at this layer output: `G^l` in R^(N x d), `G^l[t,i] = dL/dY^l[t,i]`

---

## 2. Backward: weight gradient

Matmul backward (output = input @ W^T, gradient w.r.t. W):

```
  dW^l = (G^l)^T @ X^l    in R^(d x D)
```

Entry-wise:

```
  (dW^l)[i,j] = sum_t G^l[t,i] * X^l[t,j]
```

---

## 3. Rank-1 assumption on X^l

Suppose every row of `X^l` is proportional to the same vector `a` in R^D:

```
  row_t(X^l) = c_t * a^T    for all t,  with  c_t in R,  a in R^D
```

So `X^l` has rank at most 1 (all rows parallel to `a`).

Then:

```
  (dW^l)[i,j] = sum_t G^l[t,i] * c_t * a[j] = a[j] * sum_t c_t * G^l[t,i]
```

Define `u^l` in R^d by:

```
  u^l[i] = sum_t c_t * G^l[t,i]
```

Then:

```
  dW^l = u^l @ a^T    (rank-1: outer product)
```

---

## 4. Optimizer step

Update:

```
  W^l := W^l - eta * step^l
```

where `step^l` = Optimizer(dW^l) (e.g. AdamW).

**Rank-1 preservation condition:**

If `step[i,j] = f(dW[i,j])` for some function `f`, then `step` is rank-1 only if `f(u[i] * a[j]) = p(u[i]) * q(a[j])` for some functions `p, q` (i.e., `f` is separable in its argument's factors).

**AdamW separability:**

AdamW: `step[i,j] = eta * m[i,j] / (sqrt(v[i,j]) + eps)`

- If `eps = 0` and momentum states `m, v` remain separable across updates, then `step` is separable.
- If `eps ≠ 0`, the `+ eps` term breaks exact separability (though effect may be small if `sqrt(v) >> eps`).

---

## 5. Effect of rank-1 step on layer output

Let `step^l = u_tilde @ a_tilde^T`. The change in this layer’s output for the same input `X^l` is:

```
  dY^l = X^l @ (step^l)^T = X^l @ a_tilde @ u_tilde^T
```

Row t:

```
  dY^l[t,:] = (X^l[t,:] . a_tilde) * u_tilde^T
            = (c_t * a^T . a_tilde) * u_tilde^T
            = lambda_t * u_tilde^T
```

with `lambda_t = c_t * (a . a_tilde)` scalar.

So every row of `dY^l` is proportional to `u_tilde`. The update adds a rank-1 matrix to the layer output; all positions get a multiple of the same vector `u_tilde`.

---

## 6. One training iteration (algorithm)

**Given:** `h^l`, `W^l`, `G^l`, optimizer state

1. **Normalize**
   - `X^l = RMSNorm(h^l)`

2. **Weight gradient**
   - `dW^l = (G^l)^T @ X^l`

3. **If** `row_t(X^l) = c_t * a^T` for all t **then**
   - `dW^l = u^l @ a^T`  where  `u^l[i] = sum_t c_t * G^l[t,i]`

4. **Optimizer step**
   - `step^l = Optimizer(dW^l)`   (rank(step^l) <= 1 under (3) and per-element optimizer)

5. **Update weights**
   - `W^l := W^l - eta * step^l`

6. **Change in layer output** (for same `X^l`)
   - `dY^l = -eta * X^l @ (step^l)^T`
   - If `X^l` is rank-1 as above: `dY^l[t,:] = lambda_t * u_tilde^T` for some scalar `lambda_t`

**Output:** Updated `W^l`; when `X^l` is rank-1, the change in this layer’s output `dY^l` has every row proportional to the same vector.

---

## 7. Summary identities

| Quantity | Formula |
|----------|---------|
| Weight gradient | `dW^l = (G^l)^T @ X^l` |
| If X^l rank-1 (row_t = c_t * a^T) | `dW^l = u^l @ a^T`,  `u^l[i] = sum_t c_t * G^l[t,i]` |
| Rank-1 step | `step^l = u_tilde @ a_tilde^T` |
| Change in output (rank-1 X^l) | `dY^l[t,:] = lambda_t * u_tilde^T`,  `lambda_t = c_t * (a . a_tilde)` |

All “then” conclusions hold **if** `X^l` has the rank-1 form above.
---

## 8. Numerical instantiation (Session 17714708285407192, Batch 1)

**Model configuration:**
```
D = 768                  (d_model)
V = 10277                (vocab_size)
L = 12                   (num_layers)
H = 12                   (num_heads)
d_h = D/H = 64           (head_dim)
N = 6780                 (valid tokens in this batch)
sqrt(D) = 27.71
```

**Weight initialization:**
```
W_emb in R^(10277 x 768):   W_ij ~ Uniform[-a, +a],  a = 0.02332
                             Expected ||W[row]|| = a * sqrt(D/3) = 0.02332 * sqrt(256)
                                                 = 0.02332 * 16 = 0.373
                             (NOT sqrt(D/3) = 15.97 — that's missing the 'a' factor)
                             
W_qkv in R^(1280 x 768):     a = sqrt(6/(1280+768)) = 0.0541
                             W_ij ~ Uniform[-0.0541, +0.0541]
                             
W_o in R^(768 x 768):        a = 0.0643, scaled by 1/sqrt(24) for residual
                             effective: [-0.0131, +0.0131]
```

**Measured values (Batch 1):**
```
avg_cos(h_i, h_j) = ρ = 0.7973        (hidden state alignment)
logit_max = 1.6740                     (max logit value)
loss = 9.2669                          (cross-entropy, ln(10277) = 9.238)
total_grad_norm = 2.1785
  ├─ lm_head: 0.9470
  ├─ attention: 1.3467
  ├─ ffn: 1.4262
  └─ rms_norm: 0.0197
```

### 8.1 RMSNorm output constraint

After final RMSNorm:
```
||h[t]|| = sqrt(D) = 27.71    for all t
```

With avg_cos = 0.7973, the hidden state matrix H has:
```
cos(h[i], h[j]) ≈ 0.80    for most pairs (i,j)

PC1 (first principal component) explains ~54% of variance
If g = PC1 direction (unit vector), then:
  h[t] = alpha_t * g + h_perp[t]
  where alpha_t = h[t] · g ≈ 0.8 * ||h[t]|| ≈ 0.8 * 27.71 ≈ 22.2
```

### 8.2 Weight gradient with rank-1 hidden states

**LM head backward** (tied with embedding, V=10277, D=768):

```
dW = (G_logits)^T @ H        in R^(10277 x 768)

Entry [v,d]:
  (dW)[v,d] = sum_{t=1}^{6780} G[t,v] * h[t,d]
```

**If rank-1:** `h[t,:] = c_t * a^T` where `a` in R^768, then:
```
(dW)[v,d] = a[d] * sum_{t=1}^{6780} c_t * G[t,v]
          = a[d] * u[v]
          
where u[v] = sum over 6780 tokens of (c_t * G[t,v])

dW = u @ a^T    (rank-1 matrix: 10277 x 768)
```

**Measured gradient norm (Batch 1):**
```
||dW_emb||_F = 0.9470
```

With rank-1 form:
```
||dW||_F = ||u|| * ||a||
0.9470 = ||u|| * ||a||

If ||a|| ≈ 27.71 (normalized hidden norm), then:
||u|| ≈ 0.9470 / 27.71 ≈ 0.0342
```

### 8.3 Optimizer step (AdamW)

**Learning rate:** eta ≈ 3e-4 (typical)

**AdamW update:**
```
m[v,d] := beta1 * m[v,d] + (1-beta1) * dW[v,d]
v[v,d] := beta2 * v[v,d] + (1-beta2) * dW[v,d]^2
step[v,d] = eta * m[v,d] / (sqrt(v[v,d]) + eps)
```

**With rank-1 gradient** `dW = u @ a^T`:
```
After many steps, momentum m also becomes rank-1:
m ≈ m_u @ a^T    (aligned with gradient direction)

step = u_tilde @ a_tilde^T    (rank-1 update)
```

### 8.4 Logit computation

**Current forward pass:**

Define `γ_t,v = cos(h[t], W[v])` (cosine between hidden state and weight row).

```
logit[t,v] = h[t] · W[v]
           = ||h[t]|| * ||W[v]|| * γ_t,v
           = 27.71 * ||W[v]|| * γ_t,v
```

**At initialization:**
```
Expected ||W[v]|| = 0.373  (from Xavier with a=0.02332)
Random γ_t,v ≈ 0  (expected for random unit vectors in R^768)

Expected logit ≈ 27.71 * 0.373 * 0 ≈ 0
Measured logit_max = 1.6740

This suggests either:
  (a) ||W[v]|| grew slightly, or
  (b) γ_t,v = cos(h[t], W[v]) is non-zero (some alignment)

If γ_t,v ≈ 0.16 (typical for batch):  
  ||W[v_max]|| ≈ 1.674 / (27.71 * 0.16) ≈ 0.377  (near init)

Note: avg_cos(h_i, h_j) = 0.7973 measures HIDDEN STATE alignment,
      γ_t,v measures HIDDEN-to-WEIGHT alignment (different quantity).
```

### 8.5 Change in layer output after rank-1 update

**Layer l update:**
```
W^l := W^l - eta * step^l
where step^l = u_tilde @ a_tilde^T    (rank-1)
```

**Next forward (same input X^l):**
```
dY^l = -eta * X^l @ (step^l)^T
     = -eta * X^l @ a_tilde @ u_tilde^T
```

**For token t with** `X^l[t,:] = c_t * a^T`:
```
dY^l[t,:] = -eta * c_t * (a · a_tilde) * u_tilde^T
          = lambda_t * u_tilde^T

where lambda_t = -eta * c_t * (a · a_tilde)
```

**Numerical example (Layer 0 → Layer 1):**

**Consistent interpretation (unit vectors):**

Decompose `X^l[t,:] = c_t * a^T` where `a` is a unit direction vector.

From RMSNorm: `||X^l[t,:]|| = sqrt(D) = 27.71`, so `|c_t| = 27.71` (since `||a|| = 1`).

```
let a, a_tilde be unit vectors (common directions)
let cos(a, a_tilde) ≈ 0.8  (directional alignment)

Then:
  a · a_tilde = cos(a, a_tilde) = 0.8  (dot product of unit vectors IS cosine)
  
  lambda_t = -eta * c_t * (a · a_tilde)
           = -3e-4 * 27.71 * 0.8
           ≈ -0.00663

All 6780 tokens receive a change proportional to same u_tilde:
  dY[1,:] ≈ -0.0066 * u_tilde^T
  dY[2,:] ≈ -0.0066 * u_tilde^T    (slightly different lambda, same direction)
  ...all parallel to u_tilde^T
```

**Key insight:**

ALL tokens get updates in the SAME DIRECTION `u_tilde^T`. Only the scalar magnitude `lambda_t = -eta * c_t * cos(a, a_tilde)` varies per position.

### 8.6 Gradient magnitude analysis

**Total parameter count:**
```
Embedding (tied): V * D = 10277 * 768 = 7,892,736
Encoder (12 layers):
  ├─ W_qkv: 12 * 1280 * 768 = 11,796,480
  ├─ W_o: 12 * 768 * 768 = 7,077,888
  ├─ W1: 12 * 3072 * 768 = 28,311,552
  └─ W2: 12 * 768 * 3072 = 28,311,552
Total: ~83M parameters
```

**Gradient distribution (Batch 1):**
```
Component       Norm    % of total   Params
─────────────────────────────────────────────
lm_head         0.9470  43.5%        7.9M
attention       1.3467  61.8%        18.9M
ffn             1.4262  65.5%        56.6M
rms_norm        0.0197   0.9%        18K
─────────────────────────────────────────────
Total (L2)      2.1785  100%         83M

Frobenius norm: ||G|| = sqrt(sum of squared component norms)
                      = sqrt(0.947^2 + 1.347^2 + 1.426^2 + 0.020^2)
                      = sqrt(0.897 + 1.814 + 2.033 + 0.0004)
                      = sqrt(4.744) = 2.178 ✓
```

**Per-parameter gradient magnitude:**
```
Average gradient magnitude = ||G|| / sqrt(num_params)
                          = 2.1785 / sqrt(83M)
                          = 2.1785 / 9110
                          ≈ 2.39e-4

With learning rate 3e-4:
Average weight change = 3e-4 * 2.39e-4 = 7.17e-8 per param
```

### 8.7 Loss landscape at initialization

**Random prediction:**
```
Expected loss = ln(V) = ln(10277) = 9.238
Measured loss = 9.2669
Delta = 0.029    (slightly worse than random!)
```

**With avg_cos = 0.7973:**
```
Hidden states are 80% aligned → logits are correlated
Softmax applies to correlated scores → NOT uniform distribution
Entropy < ln(V) → loss can exceed random baseline
```

**Required logit spread for learning:**
```
For loss to decrease:
  max(logit[t,:]) - logit[t, target[t]] must grow

With rank-1 updates:
  logit[t,v] = lambda_t * (u_tilde · W[v])
  
Only W[v] that aligns with u_tilde gets boosted
→ Winner-take-all dynamics
→ Mode collapse to single token
```

### 8.8 Algebraic recurrence (rank-1 approximation)

**State at batch k:**

Define a unit direction `g(k)` in R^D (e.g., first principal component of hidden states).

For each token t, define projection coefficient:
```
α_t(k) = h_t(k) · g(k)    in R
```

For each vocab row v, define projection coefficient:
```
β_v(k) = W_v(k) · g(k)    in R
```

Decompose hidden states and weights:
```
h_t(k) = α_t(k) * g(k) + h_perp,t(k)    where h_perp,t(k) · g(k) = 0
W_v(k) = β_v(k) * g(k) + W_perp,v(k)    where W_perp,v(k) · g(k) = 0
```

**Logit decomposition:**
```
logit[t,v](k) = h_t(k) · W_v(k)
              = [α_t(k) * g(k) + h_perp,t(k)] · [β_v(k) * g(k) + W_perp,v(k)]
              = α_t(k) * β_v(k) + h_perp,t(k) · W_perp,v(k)
```

**Rank-1 approximation:**

If `h_perp,t(k)` and `W_perp,v(k)` are negligible (rank-1 dominance):
```
logit[t,v](k) ≈ α_t(k) * β_v(k)
```

**After RMSNorm constraint:**
```
||h_t(k)|| = sqrt(D) = 27.71    for all t

If rank-1: h_t(k) ≈ α_t(k) * g(k), then:
  |α_t(k)| ≈ 27.71    (g(k) is unit vector)
```

**Weight gradient (from Section 3):**
```
dW = u(k) ⊗ a(k)    where u_v(k) = sum_t c_t(k) * G[t,v](k)

If a(k) = g(k) (hidden direction), then gradient is rank-1 aligned with g.
```

**Update step:**
```
W_v(k+1) = W_v(k) - eta * step_v(k)

Projection onto g:
  β_v(k+1) = β_v(k) - eta * [step_v(k) · g(k)]
```

**Recurrence for logits along g:**
```
logit[t,v](k) ≈ α_t(k) * β_v(k)

After update:
  logit[t,v](k+1) ≈ α_t(k+1) * β_v(k+1)
```

**Numerical example (Batch 1 → Batch 2):**
```
Batch 1:
  Define g(1) = PC1 of H (explains 54% variance)
  α_t(1) ≈ 22.2  (projection of ||h|| = 27.71 at cos ≈ 0.80)
  β_max(1) = max_v |β_v(1)| ≈ measured from logit_max / α_typical
           ≈ 1.674 / 22.2 ≈ 0.0754

  If v_max is dominant token (tok3237):
    logit[t, v_max](1) ≈ α_t(1) * β_max(1) ≈ 22.2 * 0.075 ≈ 1.67 ✓

Gradient concentrates u(1) in direction of v_max error:
  u_v_max(1) is large (negative, wants to increase W[v_max])
  
Update increases β_v_max:
  β_v_max(2) ≈ β_v_max(1) + eta * |u_v_max(1)|
```

**Measured trajectory (qualitative):**
```
Batch   Observed                     Rank-1 interpretation
  1     logit_max = 1.67             α ≈ 22.2, β_max ≈ 0.075
  2     logit_max = 2.98             β_max increased to ≈ 0.134
  3     logit_max = 6.92             β_max increased to ≈ 0.312
  10    logit_max = 7.06             β_max ≈ 0.318 (saturated)
```

**Key observation:**

The recurrence is driven by the rank-1 gradient structure:
```
dW = u ⊗ g    implies    β_v(k+1) - β_v(k) ∝ u_v(k)

If u concentrates mass on v_max, then β_v_max grows exponentially while others shrink.
This is pure winner-take-all from the rank-1 gradient structure.
```