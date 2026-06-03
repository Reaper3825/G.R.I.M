# Encoder Layer (Attention + FFN)

Implementation: `resources/models/GRIM-text/Layers/Encoding/Encoding_GPU.cu`, `resources/models/GRIM-text/Layers/FeedForward/Feed_Forward_GPU.cu`. QKV autograd operations and QKV diagnostics live under `resources/models/GRIM-text/Shared/TensorContract/` beside `AutogradAttention.cu`.

## Pre-norm only
Standard pre-norm: `output = input + LayerScale(sublayer_output)`. Sandwich norm was deleted (Issue #148) — post-residual RMSNorm constrained hidden norms to a hypersphere → mode collapse.

## LayerScale
LayerScale uses one learnable gamma vector per residual branch, not one scalar:

`residual1[t,d] = input[t,d] + gamma_attn[d] * attn_out[t,d]`

`output[t,d] = residual1[t,d] + gamma_ffn[d] * ffn_out[t,d]`

Each gamma tensor has shape `[1, d_model]` and is initialized from `training.config.layer_scale_init`. Use `layer_scale_init = 1.0` in `ai_config.json`; `0.1` causes catastrophic gradient vanishing through encoder layers.

Backward keeps TensorContract's parameter convention: `grad_gamma[d] = sum_t(grad_out[t,d] * sublayer_out[t,d])`. Do **not** divide by token count inside LayerScale; cross-entropy/root backward already mean-scales `grad_out`.

## Bias additions through autograd
Use `autograd::broadcast_add()` for **all** biases (`b_qkv`, `b_o`, `b1`, `b2`). Raw `launchFFNBiasAdd` bypasses autograd → zero bias gradients.

## Tensor shape ownership
Layer code must not crack open `Tensor::shape` for matmul/activation compatibility checks. If an object is already a `Tensor`, shape validation and layout conversion belong inside TensorContract/autograd operations such as `autograd::matmul()`, `autograd::elementwise_mul()`, and `autograd::broadcast_add()`.

## Forward runtime handle ownership
Encoder/FFN/LM-head/reasoning/selector layers must not store forward-time `cudaStream_t` or `cublasHandle_t`, and must not expose late `setStream()` / `setCublasHandle()` mutators. Startup may pass an init stream for self-allocation only. Actual forward execution handles come from the caller's payload/request (`AutogradContext` → `Forward::ModelForwardRequest`, or the inference/decode equivalent) and are passed into each forward call.

PBM follows the same rule for forward-time ownership: Phase1 owns PBM initialization/readiness, but encoder layers do **not** cache a borrowed `PBMState*` as durable layer state. The shared forward request carries the borrowed Phase1-owned PBM state, `forwardEncodingLayer(...)` takes that borrowed PBM explicitly, and `encoderSelfAttentionForward(...)` takes PBM as an explicit signature input for the RoPE/ALiBi math call.

## Encoder type ownership
`GPUGrimEncoder` is only a container of `EncodingLayer` instances. `grim_language_model_cuda.hpp` may forward-declare `EncodingLayer` for pointer access, but it must not mint a second public alias such as `GPUEncoderLayer`. The concrete layer type is owned by `Layers/Encoding/Encoding_GPU.hpp`; callers that need layer methods should include that header and work with `EncodingLayer*` directly.

## Dropout HP ownership
Encoder and FFN dropout rates must come from `HyperParameters_GPU.hpp` → `EncoderLayerConstructionHP` → `FeedForwardLayerConstructionHP`. `EncodingLayer` stores the grouped encoder HP snapshot directly as `hp_`; `FeedForwardLayer` stores its grouped FFN HP snapshot directly as `hp_`. Do not reintroduce layer-local dropout defaults, thin FFN config wrappers, hidden PBM pointer state, or forward-runtime handle fields.

The attention runtime dropout mode bit rides on `HyperparameterGroupings.hpp::EncoderSelfAttentionHP::dropout_enabled`, sliced explicitly by the caller as `encoderSelfAttentionHP(hp, dropout_enabled)`. Do not keep a second `dropout_enabled` sidecar on `EncoderSelfAttentionForwardRequest`, and do not infer attention dropout mode from training-vs-inference identity inside the attention facade.

## FFN parameter ownership
FFN trainable tensors (`W_gate`, `W1`, `W2`, optional `b2`) are durable registry state owned by `ParameterRegistry::StartupParameterRegistry::feed_forward_parameter_tensors`. `ParameterGroupRegistration::initializeFeedForwardParameterTensors()` allocates and Xavier-initializes one bundle per encoder layer, and shared forward passes either the direct registry-owned `FeedForwardParameterTensors` bundle or a detached copy of that same registry owner type when parameter-graph connection is disabled. `GPUGrimEncoder`, `EncodingLayer`, and `FeedForwardLayer` do not borrow or cache FFN parameter views during construction. `FeedForwardLayer` stores HP only and `forward()` consumes registry-owned parameter tensor bundles, not layer-local view structs. Do not add `Tensor` members, borrowed-view fields, or weight accessors back to `Feed_Forward_GPU.hpp`.

## Encoder dimension HP ownership
Encoder GQA/QKV derived dimensions (`head_dim`, `heads_per_kv_group`, `kv_dim`, `qkv_dim`, `is_gqa`) are computed on the HyperParameters-owned typed config surface before grouping assignment. `HyperparameterGroupings.hpp` only slices those finalized values into `EncoderLayerConstructionHP`; `Encoding_GPU.cu` must consume those fields directly from `hp_` and must not recompute config geometry in layer methods.

Encoder-facing autograd calls (`split_and_reshape_qkv`, `rope_rotation`, `reshape_bhsd_to_flat`) take an `EncoderSelfAttentionHP` snapshot sliced from the owning `EncoderLayerConstructionHP`. Do not construct a local `TensorContract::GQADims` in encoder files; TensorContract may keep GQA payload structs only for lower-level tensor/view APIs.

`Encoding_GPU.cu` must call `autograd::split_and_reshape_qkv()` only. That wrapper owns the tape node and delegates the raw layout split/merge to `TensorConversion::split_qkv_gqa()` / `merge_qkv_grads_gqa()` internally. Calling `TensorConversion` directly from encoder forward bypasses `SplitAndReshapeQKVGradFn` and disconnects `Q/K/V` from the `qkv_out -> W_qkv / b_qkv` gradient path.

There is no live `QKV_Projector` module. QKV projection is the `autograd::matmul(ln1_out, W_qkv, transpose_b=true)` call because it must be a tape node. The deleted `Layers/Attention/QKV_Projector.{hpp,cu}` wrapper had stopped projecting QKV and only forwarded BHSD→BSM reshape to TensorConversion, so it was removed to avoid a false second ownership path.

## QKV diagnostics ownership
QKV-specific diagnostic code belongs next to the autograd attention implementation, not inside `Encoding_GPU.cu`. Use `Shared/TensorContract/AutogradQKVDiagnostics.hpp/.cu` for:
- `GRIM_DEBUG_QKV` NaN/Inf tensor scans around QKV/SDPA tensors. These scans are silent on finite tensors and fail loud with `[QKV_NONFINITE] FATAL ...` plus a thrown exception on any non-finite value.
- `[QKV_EQUATION]` and `QKV_PROJECTION_EQUATION` Rule 21 logging.
- QKV projection shape/bias validation used by those diagnostics.

`Encoding_GPU.cu` may call the diagnostic API at the attention boundary, but it must not own QKV sampling buffers, QKV non-finite scan kernels, or duplicated QKV equation logic. This keeps the visible path as: encoder orchestration → TensorContract/autograd attention API → raw TensorConversion layout kernels.

## Encoder residual diagnostics ownership
Per-layer residual/output Rule 21 logging lives in `Layers/Encoding/EncoderDiagnostics.hpp/.cu`, not inline in `Encoding_GPU.cu`. `forwardEncodingLayer(...)` passes the full live stack (`input`, raw/actual attention branch, `residual1`, raw/actual FFN branch, `output`, LayerScale gamma pointers, payload geometry, stream, layer index) to the helper after `output` is materialized.

The helper emits one summary equation entry plus one ordered per-row stack entry for each flat `[batch_size * max_seq_len]` row. Per-row logs must be emitted under a single phase key and rely on the tape's stable phase sort so equal phase/layer entries retain flat-row insertion order; do not split a single row stack across multiple phases or the flushed log will appear out of order.

## Encoder residual diagnostics ownership
`Layers/Encoding/EncoderDiagnostics.{hpp,cu}` owns the Rule 21 residual-stack diagnostic emitted after `forwardEncodingLayer()` computes `output = residual1 + ffn_branch`. `Encoding_GPU.cu` must pass the full live stack (`input`, raw attention output, actual attention branch used in the residual add, `residual1`, raw FFN output, actual FFN branch used in the residual add, `output`, optional LayerScale tensors, payload geometry, stream, and layer index) and must not rebuild host-side diagnostic sampling logic inline.

The diagnostic emits one compact summary equation plus `LAYER_RESIDUAL1_ROW_EQUATION` and `LAYER_RESIDUAL2_ROW_EQUATION` entries for every flattened payload row. Row entries report `(row, batch_row, seq_pos, valid)` and row-level min/max/mean/RMS for the tensors that participate in each residual equation. This is intentionally Debug-gated and skipped on non-initial accumulation slots because it performs blocking D2H copies of the active encoder stack.

## FFN post-GELU cache
`forwardEncodingLayer()` MUST `cudaMemcpyAsync` post-GELU activations into `args.cache_ffn_output` after the explicit FFN compute call. Forgetting this leaves the cache as garbage → corrupted W2 gradients.

## Activation centering before weight grads
Center cached activations (`cached_ln1_output`, `cached_ffn_input`, …) **before** the weight-gradient GEMMs to eliminate systematic bias from non-zero mean.

## Residual centering is per-sequence, padding-aware, and causal-prefix only
`center_encoder_residuals` MUST use `autograd::center_columns_by_causal_prefix_lengths(x, payload.seq_lengths, payload.batch_size, payload.max_seq_len, stream)`, not global `center_columns(x)`, fixed-row `center_columns_by_sequence(x, payload.max_seq_len, stream)`, or full-sequence `center_columns_by_sequence_lengths(...)`, on flattened `[batch_size * seq_len, d_model]` tensors.

Global column-centering over the full flat matrix makes sample A depend on sample B via the batch-wide mean. Full-sequence per-row centering avoids cross-batch leakage but still leaks future positions `t+1...N` into token `t`, which is forbidden for an autoregressive model. Fixed-row per-sequence centering also leaks PAD activations into real-token means because `BatchPayload` pads input IDs to `Tokenizer::PAD_TOKEN_ID` and those rows produce real embeddings. The safe equation is:

`h[b,0,d] = h[b,0,d]`

`h[b,t,d] = h[b,t,d] - mean_{u < t}(h[b,u,d])` for valid `t > 0`, and padded rows are zeroed.

This removes the running within-sequence shared direction without crossing batch-row ownership boundaries, letting PAD rows steer the mean, erasing the first token, or leaking future tokens into the current position. If any centered row has `seq_lengths[b] <= 1`, fail loud; there is no meaningful strict-past context to center against.

`center_encoder_residuals` applies to the live intra-layer residual stream inside `forwardEncodingLayer()` at `residual1 = center_columns_by_causal_prefix_lengths(input + attn_branch, ...)`. Do **not** apply a second causal-prefix centering pass to the committed layer output in `ModelForward_GPU.cu` before handing it to the next layer. Re-applying the same non-orthogonal prefix-centering operator at the layer boundary amplifies early-token rows (`seq_pos` 1-3 showed runaway row-RMS growth while branch RMS stayed small) and makes `input_{l+1}` larger than the raw `output_l` it was derived from.

## `per_token_grad_scale=true` is REQUIRED
Gradient RMS ~1e-6 with ~3000 tokens is **correct**. Disabling causes ~3000× effective LR explosion.
