# Rank Geometry and Gradient Interpretation

Exact fixed-input algebra and its qualifications live together here, not in
competing collapse proofs/fact-check summaries.

## Fixed-input algebra

For `X[N,D]`, `W[d,D]`, `Y = X @ W.T`, and `G = dL/dY[N,d]`:

```text
dW = G.T @ X
dX = G @ W
```

The transposed MatMulGradFn branch in
[AutogradAttention.cu](../../resources/models/GRIM-text/Shared/TensorContract/AutogradAttention.cu)
implements these contributions with accumulation beta `1.0`.

If `X = c @ a.T` exactly, with `c[N,1]`, `a[D,1]`:

```text
dW = (G.T @ c) @ a.T = u @ a.T
u_i = sum_t G[t,i] * c_t
rank(dW) <= 1
```

Retain the `c_t` weighting. This is one contribution; different branch/window
contributions sum and can have higher rank.

For an arbitrary actual optimizer update `U`, holding `X` fixed:

```text
W_new = W - eta * U
deltaY = -eta * X @ U.T = c @ (-eta * U @ a).T
rank(deltaY) <= 1
```

This output-change bound does not require a rank-one optimizer step.

## Interpretation limits

| Claim | Qualification |
|---|---|
| High average cosine proves exact rank one | No; it neither establishes exact proportionality nor bounds every singular value |
| RMSNorm preserves all row directions | Nonuniform feature-wise gamma can change directions/angles |
| Exactly proportional rows stay proportional under shared RMSNorm gamma | Yes; row-scalar normalization plus a shared diagonal transform cannot increase rank |
| Rank-one raw gradients force rank-one AdamW/RAdamW steps | Not generally; nonlinear scaling, epsilon, moment history, and decay matter |
| Fixed-input linear rank bound proves an unrecoverable transformer trap | No; changing inputs, residuals, nonlinearities, other layers, and later updates are outside this identity |
| Random signed cosine baseline is `1/D` | Its mean is zero and standard deviation is `1/sqrt(D)`; average absolute cosine is different |

[RMSNormGradFn.cu](../../resources/models/GRIM-text/Shared/TensorContract/GradFns/RMSNormGradFn.cu)
uses `y_i = x_i * gamma_i / sqrt(mean(x*x) + epsilon)`.
Gamma is feature-wise, not a row-scalar. Compare actual tensors rather than
assuming pre/post-normalization cosine remains unchanged.

## Optimizer counterexample

Take rank-one `G = [[1,2],[2,4]]` and apply `f(g)=g/(abs(g)+0.1)` elementwise.
Its determinant is approximately `-0.0201115184`, so the result has rank two.
This refutes arbitrary elementwise rank preservation; it is not a reproduced
GRIM optimizer step.

[AdamW_Kernal_GPU.cu](../../resources/models/GRIM-text/Shared/Optimizers/AdamW/AdamW_Kernal_GPU.cu)
uses moment history and `rsqrt(v_hat+epsilon)`;
[RAdamW_Kernal_GPU.cu](../../resources/models/GRIM-text/Shared/Optimizers/RAdamW/RAdamW_Kernal_GPU.cu)
has rectified/unadapted phases, epsilon, and decay. Neither establishes an
unconditional separable map from the current gradient.

## Historical measurements

[Equation trace](investigations/equation_trace.txt) and the
[plateau investigation](../PLATEAU_BUG_INVESTIGATION.md) retain their original
run scope. They do not diagnose a current model or prove a particular
intervention is necessary. Removed centering/PC1 paths are not requirements
to reinstate. Use [training math](TRAINING_MATH.md) and named current logs
for an actual regression investigation.
