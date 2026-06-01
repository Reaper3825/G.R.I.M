# Grouped Query Attention

Config: `num_heads=12`, `num_kv_heads=4`, `heads_per_kv_group=3`.

## Shapes
`W_qkv`: `[(num_heads + 2*num_kv_heads) * head_dim, d_model]` = `[1280, 768]`

## FlashAttention support
FlashAttention v2 supports GQA/MQA when `num_heads % num_kv_heads == 0`. GRIM passes `n_heads`, `n_kv_heads`, and `h_h_k_ratio = n_heads / n_kv_heads` into both contiguous forward/backward and KV-cache forward wrappers. Forward reads K/V through `query_head / h_h_k_ratio`.

At the TensorContract boundary, the finalized grouped ratio also rides on `HyperparameterGroupings.hpp::FlashAttentionRuntimeHP::heads_per_kv_group` so SDPA/backward reduction consumes the HyperParameters-owned value instead of recomputing `num_heads / num_kv_heads` locally.

## Terminology note
In GRIM docs, `MHA` means the equal-head special case `num_kv_heads == num_heads`. It is not a separate FlashAttention kernel family. Wrapper helper names should use `flash_attn`, not `mha`, because the same dispatch covers MHA, MQA, and GQA through `h_h_k_ratio`.

## Backward reduction
Backward MUST reduce per-query-head dK/dV contributions into each grouped KV head with a plain sum. Do **not** scale by `1 / heads_per_kv_group` or `1 / sqrt(heads_per_kv_group)`; upstream FlashAttention expands dK/dV to query-head slots and then uses `sum_out` across the grouped-head axis. GRIM mirrors this by allocating temporary dK/dV buffers for `num_heads`, then reducing to `num_kv_heads` in `ScaledDotProductAttentionGradFn`.

## Checkpoint compatibility
MHA and GQA checkpoints are **incompatible** — serialization throws on mismatch. Old GQA checkpoints with `num_kv_heads=0` cannot load into the current model; retrain.

See [FlashAttention.md](FlashAttention.md) for the matching backward buffer sizing rules.
