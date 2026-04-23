# FlashAttention (v2)

Implementation: `resources/models/GRIM-text/Layers/Flash_Attention_Kernal.cu`. External library is Dao-AILab FlashAttention v2.

## Issue #84 — `dot_do_o` preprocessing
`flash_bwd_dot_do_o_kernel` MUST run **before** `flash_bwd_dq_dk_dv_loop_kernel`. Without it, `dsoftmax_sum` is garbage for most m_blocks → dQ/dK explosion.

## Issue #72 — GQA backward dk/dv buffers
Allocate `dk_bf16` / `dv_bf16` for `num_heads` (12), **not** `num_kv_heads` (4). The Dao-AILab backward kernel writes using the query head index. Both allocation sites in `TensorContract_GPU.cu` must use `num_heads`.

## Issue #73 — GQA reduction scaling
Apply `gqa_grad_scale = 1.0f / heads_per_kv_group` **in** the reduction kernel. The external library does not apply this internally.

## ALiBi slopes
FlashAttention expects **negative** slopes (library uses `+= slope * col_idx`). See [PositionEncoding.md](PositionEncoding.md).
