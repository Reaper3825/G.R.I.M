# Weight Initialization

## Xavier init via splitmix64
Per-element seed mixed with splitmix64, then 16 LCG iterations. A **single** LCG iteration produces correlated outputs (`avg|cos| ≈ 0.37` instead of expected `0.036`).

## Embedding scale = 1.0
Do **not** scale embeddings by `sqrt(d_model)`. See [LMHead.md](LMHead.md) for the full rationale.
