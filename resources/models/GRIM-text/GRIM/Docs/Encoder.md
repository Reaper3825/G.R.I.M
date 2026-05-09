# Encoder Layer (Attention + FFN)

Implementation: `resources/models/GRIM-text/Layers/Encoding_GPU.cu`, `resources/models/GRIM-text/Layers/Feed_Forward_GPU.cu`.

## Pre-norm only
Standard pre-norm: `output = input + LayerScale(sublayer_output)`. Sandwich norm was deleted (Issue #148) — post-residual RMSNorm constrained hidden norms to a hypersphere → mode collapse.

## LayerScale
`init_value = 1.0` in `ai_config.json`. `0.1` causes catastrophic gradient vanishing through encoder layers.

## Bias additions through autograd
Use `autograd::broadcast_add()` for **all** biases (`b_qkv`, `b_o`, `b1`, `b2`). Raw `launchFFNBiasAdd` bypasses autograd → zero bias gradients.

## FFN post-GELU cache
`EncodingLayer::forward()` MUST `cudaMemcpyAsync` post-GELU activations into `args.cache_ffn_output` after `ffn_->forward()`. Forgetting this leaves the cache as garbage → corrupted W2 gradients.

## Activation centering before weight grads
Center cached activations (`cached_ln1_output`, `cached_ffn_input`, …) **before** the weight-gradient GEMMs to eliminate systematic bias from non-zero mean.

## Residual centering is per-sequence and padding-aware
`center_encoder_residuals` MUST use `autograd::center_columns_by_sequence_lengths(x, bindings.d_seq_lengths, payload.batch_size, payload.max_seq_len, stream)`, not global `center_columns(x)` or fixed-row `center_columns_by_sequence(x, payload.max_seq_len, stream)`, on flattened `[batch_size * seq_len, d_model]` tensors.

Global column-centering over the full flat matrix makes sample A depend on sample B via the batch-wide mean. Fixed-row per-sequence centering still leaks PAD activations into real-token means because `BatchPayload` pads input IDs to `Tokenizer::PAD_TOKEN_ID` and those rows produce real embeddings. The safe equation is:

`h[b,t,d] = h[b,t,d] - mean_{u < seq_lengths[b]}(h[b,u,d])` for valid rows, and padded rows are zeroed.

This removes the shared within-sequence direction without crossing batch-row ownership boundaries or letting PAD rows steer the mean. If any centered row has `seq_lengths[b] <= 1`, fail loud; centering a single valid row would erase the residual stream.

## `per_token_grad_scale=true` is REQUIRED
Gradient RMS ~1e-6 with ~3000 tokens is **correct**. Disabling causes ~3000× effective LR explosion.
