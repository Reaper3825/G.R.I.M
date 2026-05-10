# Encoder Layer (Attention + FFN)

Implementation: `resources/models/GRIM-text/Layers/Encoding/Encoding_GPU.cu`, `resources/models/GRIM-text/Layers/FeedForward/Feed_Forward_GPU.cu`.

## Pre-norm only
Standard pre-norm: `output = input + LayerScale(sublayer_output)`. Sandwich norm was deleted (Issue #148) — post-residual RMSNorm constrained hidden norms to a hypersphere → mode collapse.

## LayerScale
LayerScale uses one learnable gamma vector per residual branch, not one scalar:

`residual1[t,d] = input[t,d] + gamma_attn[d] * attn_out[t,d]`

`output[t,d] = residual1[t,d] + gamma_ffn[d] * ffn_out[t,d]`

Each gamma tensor has shape `[1, d_model]` and is initialized from `training.config.layer_scale.init_value`. Use `init_value = 1.0` in `ai_config.json`; `0.1` causes catastrophic gradient vanishing through encoder layers.

Backward keeps TensorContract's parameter convention: `grad_gamma[d] = sum_t(grad_out[t,d] * sublayer_out[t,d])`. Do **not** divide by token count inside LayerScale; cross-entropy/root backward already mean-scales `grad_out`.

## Bias additions through autograd
Use `autograd::broadcast_add()` for **all** biases (`b_qkv`, `b_o`, `b1`, `b2`). Raw `launchFFNBiasAdd` bypasses autograd → zero bias gradients.

## Tensor shape ownership
Layer code must not crack open `Tensor::shape` for matmul/activation compatibility checks. If an object is already a `Tensor`, shape validation and layout conversion belong inside TensorContract/autograd operations such as `autograd::matmul()`, `autograd::elementwise_mul()`, and `autograd::broadcast_add()`.

## Forward runtime handle ownership
Encoder/FFN/LM-head/reasoning/selector layers must not store forward-time `cudaStream_t` or `cublasHandle_t`, and must not expose late `setStream()` / `setCublasHandle()` mutators. Startup may pass an init stream for self-allocation only. Actual forward execution handles come from the caller's payload/request (`AutogradContext` → `Forward::ModelForwardRequest`, or the inference/decode equivalent) and are passed into each forward call.

## Dropout HP ownership
Encoder and FFN dropout rates must come from `HyperParameters_GPU.hpp` → `EncoderLayerConstructionHP` → `FeedForwardLayerConstructionHP`. `EncodingConfig` stores the grouped encoder HP snapshot plus PBM pointer; `FeedForwardLayer` stores its grouped FFN HP snapshot directly as `hp_`. Do not reintroduce layer-local dropout defaults, thin FFN config wrappers, or forward-runtime handle fields.

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
