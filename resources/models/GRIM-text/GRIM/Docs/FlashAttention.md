# FlashAttention (v2)

Implementation: `resources/models/GRIM-text/Layers/FlashAttention/Flash_Attention_Kernal.cu`. External library is Dao-AILab FlashAttention v2.

## Pinned Bridges-2 namespace/API contract
`scripts/run_train_on_bridges2.sh --sync-fas` uses `scripts/bridges2_ensure_flash_attention.sh` to pin `external/flash-attention` to the superproject gitlink and nested Cutlass to `bbe579a9e3beb6ea6626d9227ec32d0dae119a49`.

At that FlashAttention revision, headers respect `FLASH_NAMESPACE`; GRIM defines it as `grim_flash` before including upstream headers. Therefore GRIM's wrapper must dispatch `compute_attn`, `compute_dq_dk_dv_seqk_parallel`, `convert_dQ`, and `compute_dot_do_o` through `grim_flash`, not `flash`, on Linux and Windows.

Forward `compute_attn` also requires the explicit softcap template parameter:
`compute_attn<..., Is_even_K, Is_softcap, Return_softmax>(params)`. GRIM disables softcap with `FLASHATTENTION_DISABLE_SOFTCAP`, so wrapper calls must pass `/*Is_softcap=*/false`.

If Bridges-2 reports `namespace "flash" has no member "compute_attn"`, do not edit vendored FlashAttention source. Verify the tracked wrapper namespace/signature contract first, then rerun FAS sync if the remote dependency tree is dirty.

## Backward launch contract — seqK-parallel only
Use the pinned Dao backward launch shape:

1. `flash_bwd_dot_do_o_kernel<Clear_dQaccum=true>` over `(num_m_block, batch, heads)`.
2. `compute_dq_dk_dv_seqk_parallel` over `(num_n_block, batch, heads)`.
3. `convert_dQ` over `(num_m_block, batch, heads)` with `nsplits=1` while `deterministic=false`.

Do **not** route GRIM training through the direct `compute_dq_dk_dv` launch shape `(batch, heads, 1)`. In the May 2026 Bridges-2 gradient explosion investigation, that direct path produced corrupt K/V gradients under GQA+dropout while dQ looked small. The first full-buffer explosion was:

`SDPA.apply dv_bf16_post_bwd rms≈5.9e9..2.1e10, max_abs≈2.06e12`

and it propagated downstream into `grad_v_fp32`, `SplitQKV.merge grad_V_bhsd`, then `BiasAdd.apply grad_bias_accum` / qkv bias groups.

## Issue #84 — `dot_do_o` preprocessing
`flash_bwd_dot_do_o_kernel` MUST run **before** the seqK-parallel main backward kernel. Without it, `dsoftmax_sum` is garbage for most m_blocks → dQ/dK explosion.

## Issue #72 — GQA backward dk/dv buffers
Allocate `dk_bf16` / `dv_bf16` for `num_heads` (12), **not** `num_kv_heads` (4). The Dao-AILab backward kernel writes using the query head index. Allocation sites in `Shared/TensorContract/AutogradAttention.cu` must use `num_heads`.

`dk_row_stride` and `dv_row_stride` must also be `num_heads * head_dim`. Using `num_kv_heads * head_dim` aliases query-head writes across sequence rows and corrupts K/V gradients.

## Issue #73 — GQA reduction scaling
Use a plain sum in the reduction kernel. The grouped KV gradient is the chain-rule sum of per-query-head contributions; upstream FlashAttention reduces with sum semantics.

## ALiBi slopes
FlashAttention expects **negative** slopes (library uses `+= slope * col_idx`). See [PositionEncoding.md](PositionEncoding.md).
