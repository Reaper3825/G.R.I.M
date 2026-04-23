# Grouped Query Attention

Config: `num_heads=12`, `num_kv_heads=4`, `heads_per_kv_group=3`.

## Shapes
`W_qkv`: `[(num_heads + 2*num_kv_heads) * head_dim, d_model]` = `[1280, 768]`

## Backward scaling
Backward MUST apply `gqa_grad_scale = 1.0f / heads_per_kv_group` to dV/dK.

## Checkpoint compatibility
MHA and GQA checkpoints are **incompatible** — serialization throws on mismatch. Old GQA checkpoints with `num_kv_heads=0` cannot load into the current model; retrain.

See [FlashAttention.md](FlashAttention.md) for the matching backward buffer sizing rules.
