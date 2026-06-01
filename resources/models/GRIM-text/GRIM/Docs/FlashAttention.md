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

Validation signal after switching GRIM to the seqK-parallel contract: Bridges-2 session `1779067735859356347` (`w008.ib.bridges2.psc.edu`, `2026-05-17 21:28 EDT`) no longer showed the explosion. In `train_latest.err`, representative registered gradient checks stayed around `preclip_registered_global≈4.9e-5..7.0e-5`, `enc_rms_pre≈4.9e-5..7.0e-5`, `clipped=NO`, with top groups at ordinary `~1e-4..6e-4` RMS instead of qkv bias at `5.0966443327488e13` RMS. In the matching `latest_run.log`, 312 FlashAttention backward samples showed `dK`/`dV` at ordinary `~1e-7` RMS and the old `SDPA.apply dv_bf16_post_bwd≈1e10` / qkv-bias `5.0966443327488e13` signatures were absent.

## Diagnostics ownership
Optional FlashAttention equation tracing and D2H sampling live in `resources/models/GRIM-text/Layers/FlashAttention/AttentionDiagnostics.hpp/.cu`.

`Flash_Attention_Kernal.cu` may call narrow pre/post forward/backward hooks from that helper, but it must not inline host-side attention-score reconstruction, per-head LSE scans, or raw tensor sample conversion logic. Keep the kernel wrapper focused on validation, parameter stamping, kernel launch, and fail-loud CUDA error handling.

## Softmax scale plumbing
`autograd::scaled_dot_product_attention(..., scale, ...)` resolves `scale == 0.0f` to the canonical `1 / sqrt(head_dim)` default. Any non-zero scale must be finite and positive, and it must be passed unchanged into both `flash_attn_fwd_ex` and the matching `flash_attn_bwd_ex` call. The wrapper then stamps `params.scale_softmax`, `params.scale_softmax_log2`, and `params.scale_softmax_rp_dropout` from that same resolved scale.

Do not reintroduce a local `1 / sqrt(head_dim)` inside the contiguous forward/backward param initialization path; that silently changes the caller's attention equation and desynchronizes backward from the saved forward LSE.

## KV-cache forward dispatch
`flash_attn_fwd_kvcache()` is inference-only, but its kernel dispatch must still mirror `flash_attn_fwd_ex`: respect the caller's `head_dim`, `causal`, and `is_bf16` flags, then fail loud when compile-time gates such as `GRIM_FLASHATTN_HDIM64_ONLY`, `GRIM_FLASHATTN_CAUSAL_ONLY`, or `GRIM_FLASHATTN_BF16_ONLY` make a requested mode unavailable. Do not hardcode `run_flash_attn_fwd_hdim64<cutlass::bfloat16_t, true>` in this path.

## Issue #84 — `dot_do_o` preprocessing
`flash_bwd_dot_do_o_kernel` MUST run **before** the seqK-parallel main backward kernel. Without it, `dsoftmax_sum` is garbage for most m_blocks → dQ/dK explosion.

## Issue #72 — GQA backward dk/dv buffers
Allocate `dk_bf16` / `dv_bf16` for `num_heads` (12), **not** `num_kv_heads` (4). The Dao-AILab backward kernel writes using the query head index. Allocation sites in `Shared/TensorContract/AutogradAttention.cu` must use `num_heads`.

`dk_row_stride` and `dv_row_stride` must also be `num_heads * head_dim`. Using `num_kv_heads * head_dim` aliases query-head writes across sequence rows and corrupts K/V gradients.

## Issue #73 — GQA reduction scaling
Use a plain sum in the reduction kernel. The grouped KV gradient is the chain-rule sum of per-query-head contributions; upstream FlashAttention reduces with sum semantics.

## ALiBi slopes
FlashAttention expects **negative** slopes (library uses `+= slope * col_idx`). See [PositionEncoding.md](PositionEncoding.md).
