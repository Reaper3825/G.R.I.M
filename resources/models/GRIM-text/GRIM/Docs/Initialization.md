# Weight Initialization

## Xavier init via Philox
`Tensor::xavier_uniform_()` fills 2D weight tensors with Xavier uniform values using cuRAND Philox. The initializer computes:

`scale = sqrt(6 / (fan_in + fan_out))`

and samples `U(-scale, +scale)` on the provided CUDA stream. After sampling,
the initializer computes the realized tensor mean and subtracts it so every
initialized weight matrix with more than one element has exact sample mean 0.
This mean-centering pass does not renormalize variance afterward; it only
removes small residual sample bias from the realized Xavier draw.

## Residual projection init gain
Residual projection weights (`W_o` and FFN `W2`) use `Tensor::xavier_uniform_with_gain_()` during startup construction:

`scale = sqrt(6 / (fan_in + fan_out)) * residual_projection_init_gain`

`encoderLayerConstructionHP().residual_projection_init_gain = 1 / sqrt(2 * num_layers)`

This is parameter initialization only. Runtime residual connections stay inside the tape through `autograd::add`, with optional `autograd::layer_scale`; no raw residual-path scaling is performed during forward or backward.

The derived gain belongs to grouped construction HP, not startup bindings. Bindings carry borrowed startup resources only. `EncodingLayer` and `FeedForwardLayer` fail loud if the grouped gain is missing or non-positive.

## Embedding scale = 1.0
Do **not** scale embeddings by `sqrt(d_model)`. See [LMHead.md](LMHead.md) for the full rationale.
