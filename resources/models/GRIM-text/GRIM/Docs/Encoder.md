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

## `per_token_grad_scale=true` is REQUIRED
Gradient RMS ~1e-6 with ~3000 tokens is **correct**. Disabling causes ~3000× effective LR explosion.
