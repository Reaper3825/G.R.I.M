# Weight Initialization

## Xavier init via splitmix64
Per-element seed mixed with splitmix64, then 16 LCG iterations. A **single** LCG iteration produces correlated outputs (`avg|cos| ≈ 0.37` instead of expected `0.036`).

## Residual projection scaling
Residual projection weights (`W_o` and FFN `W2`) are Xavier-initialized, then scaled by `encoderLayerConstructionHP().residual_scale = 1 / sqrt(2 * num_layers)` during startup construction. This derived value belongs to grouped construction HP, not `EncoderConstructionBindings`: bindings carry borrowed startup resources only. `GPUGrimEncoder`, `EncodingLayer`, and `FeedForwardLayer` must fail loud if the grouped value is missing or non-positive.

## Embedding scale = 1.0
Do **not** scale embeddings by `sqrt(d_model)`. See [LMHead.md](LMHead.md) for the full rationale.
