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

If the optimizer applies a per-element map (e.g. scaled sign or normalized gradient), then `dW^l = u^l @ a^T` implies `step^l` is also an outer product of two vectors (e.g. `step^l = u_tilde @ a_tilde^T`). So rank(step^l) <= 1.

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
