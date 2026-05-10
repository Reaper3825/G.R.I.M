# Weight Initialization

## Xavier init via splitmix64
Per-element seed mixed with splitmix64, then 16 LCG iterations. A **single** LCG iteration produces correlated outputs (`avg|cos| ≈ 0.37` instead of expected `0.036`).

## Residual projection scaling
Residual projection weights (`W_o` and FFN `W2`) are Xavier-initialized, then scaled by `gpuModelInitializationHP().residual_scale = 1 / sqrt(2 * num_layers)` during startup construction. This scale is required construction plumbing, not a runtime branch and not a constructor fallback: `EncoderConstructionBindings`, `EncodingLayer`, and `FeedForwardLayer` must fail loud if it is missing or non-positive.

## Embedding scale = 1.0
Do **not** scale embeddings by `sqrt(d_model)`. See [LMHead.md](LMHead.md) for the full rationale.
