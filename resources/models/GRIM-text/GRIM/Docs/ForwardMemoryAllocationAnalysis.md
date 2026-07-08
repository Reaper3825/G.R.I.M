# Forward Memory Allocation Analysis

## Source

Analyzed logs:

- `resources/models/GRIM-text/training/logs/training_1783465557385087069.log`
- `resources/models/GRIM-text/training/logs/training_1783467717196987649.log`

Run date in logs:

- `2026-07-07`

Shared model/runtime context emitted by the runs:

- GPU: NVIDIA H100 80GB HBM3
- `d_model=768`
- `d_ff=3072`
- `num_layers=12`
- `num_heads=12`
- `num_kv_heads=4`
- `max_seq_len=1024`
- `vocab_size=10262`
- FlashAttention enabled
- GQA enabled
- MTP enabled

Batch geometry differs by run:

| Log | `batch_size` | `max_tokens_per_batch` |
|---|---:|---:|
| `training_1783465557385087069.log` | 12 | 12288 |
| `training_1783467717196987649.log` | 1 | 1024 |

## Report Scope

This report uses only measured values emitted by the log. It does not infer per-label liveness at `post_forward`, because `[ForwardAllocationSizes]` records cumulative successful allocations during `executeModelForward()`, while `[GPU_MEM] phase=post_forward` records device memory live at the sample point.

The report distinguishes these three measurements:

- `[GPU_MEM] pre_forward/post_forward`: live device memory at sampled boundaries.
- `[ForwardAllocationSizes]`: cumulative successful `cudaMallocOrThrow` allocation volume inside `executeModelForward()`, grouped by allocation label.
- `[ForwardOutputSizes]`: live tensors retained by `Forward::ModelForwardOutputs` after forward returns.

## Run-Wide Summary

Parsed complete blocks:

| Measurement | Count | Min MiB | Max MiB | Avg MiB |
|---|---:|---:|---:|---:|
| `[ForwardAllocationSizes].total_MiB` | 900 | 46138.95 | 46212.37 | 46165.79 |
| `[ForwardOutputSizes].total_MiB` | 900 | 15052.36 | 15064.08 | 15056.64 |
| Allocation total minus retained output total | 900 | 31086.59 | 31148.29 | 31109.15 |

Forward allocation structure was stable across the run:

| Measurement | Count | Min | Max | Avg |
|---|---:|---:|---:|---:|
| Allocations per forward | 900 | 944 | 944 | 944 |
| Allocation labels per forward | 900 | 65 | 65 | 65 |

Sampled live forward GPU delta:

| Measurement | Count | Min MiB | Max MiB | Avg MiB |
|---|---:|---:|---:|---:|
| `post_forward - pre_forward` | 19 | 40794.00 | 40874.00 | 40809.16 |

Derived from the run-wide averages:

| Relationship | Value MiB |
|---|---:|
| Avg live forward delta not retained in `ModelForwardOutputs` | 25752.52 |
| Avg cumulative forward allocation above sampled live forward delta | 5356.63 |

These two derived rows mean:

- About `15056.64 MiB` of the sampled live forward growth is directly retained by `ModelForwardOutputs`.
- About `25752.52 MiB` of the sampled live forward growth is live outside `ModelForwardOutputs` at `post_forward`.
- About `5356.63 MiB` of cumulative forward allocation volume does not appear in the sampled live forward delta. The current log does not identify which allocation labels account for that difference.

## Concrete Batch Samples

### Batch 1

Measured log entries:

| Field | Value |
|---|---:|
| `pre_forward device_used` | 2717.10 MiB |
| `[ForwardAllocationSizes].total_MiB` | 46169.50 MiB |
| `post_forward device_used` | 43591.10 MiB |
| `[ForwardOutputSizes].total_MiB` | 15057.23 MiB |
| `post_backward device_used` | 45597.10 MiB |

Derived values:

| Relationship | Value MiB |
|---|---:|
| `post_forward - pre_forward` | 40874.00 |
| Live forward delta not retained in `ModelForwardOutputs` | 25816.77 |
| Cumulative allocation above live forward delta | 5295.50 |

### Batch 50

Measured log entries:

| Field | Value |
|---|---:|
| `pre_forward device_used` | 28083.10 MiB |
| `[ForwardAllocationSizes].total_MiB` | 46178.89 MiB |
| `post_forward device_used` | 68897.10 MiB |
| `[ForwardOutputSizes].total_MiB` | 15058.73 MiB |
| `post_backward device_used` | 45607.10 MiB |

Derived values:

| Relationship | Value MiB |
|---|---:|
| `post_forward - pre_forward` | 40814.00 |
| Live forward delta not retained in `ModelForwardOutputs` | 25755.27 |
| Cumulative allocation above live forward delta | 5364.89 |

## Largest Forward Allocation Labels

Batch 1 contained `944` forward allocations across `65` labels, totaling `46169.50 MiB`.

Top allocation labels from batch 1:

| Label | Count | MiB | Max Request MiB |
|---|---:|---:|---:|
| `matmul_result` | 75 | 7821.19 | 481.03 |
| `MatMulGradFn_cache_a` | 75 | 4113.00 | 144.00 |
| `MatMulGradFn_grad_a` | 75 | 4113.00 | 144.00 |
| `BiasAddGradFn_grad_input` | 10 | 2630.20 | 481.03 |
| `broadcast_add_result` | 10 | 2630.20 | 481.03 |
| `ElementwiseMulGradFn_grad_a` | 13 | 1764.00 | 144.00 |
| `ElementwiseMulGradFn_grad_b` | 13 | 1764.00 | 144.00 |
| `SiluGradFn_input_grad` | 13 | 1764.00 | 144.00 |
| `emul_result` | 13 | 1764.00 | 144.00 |
| `silu_result` | 13 | 1764.00 | 144.00 |
| `RMSNormGradFn_cache` | 31 | 1089.00 | 36.00 |
| `RMSNormGradFn_input_grad` | 31 | 1089.00 | 36.00 |
| `rms_norm_result` | 31 | 1089.00 | 36.00 |
| `AddGradFn_grad_a` | 27 | 942.98 | 36.00 |
| `MulScalarGradFn_input_grad` | 27 | 942.98 | 36.00 |
| `add_result` | 27 | 942.98 | 36.00 |
| `mul_scalar_result` | 27 | 942.98 | 36.00 |
| `AddGradFn_grad_b` | 26 | 936.00 | 36.00 |
| `sdpa_result` | 24 | 864.00 | 36.00 |
| `SplitQKV_merged_qkv_grad` | 12 | 720.00 | 60.00 |
| `SliceColumnsGradFn_grad_x` | 4 | 576.00 | 144.00 |
| `MatMulGradFn_cache_b` | 75 | 574.89 | 30.06 |
| `DropoutGradFn_input_grad` | 13 | 468.00 | 36.00 |
| `dropout_seeded_result` | 13 | 468.00 | 36.00 |
| `qkv_split_Q` | 12 | 432.00 | 36.00 |

Measured grouping by label family:

| Family | Included labels | Batch 1 MiB |
|---|---|---:|
| Matmul result/cache/grad-a/cache-b | `matmul_result`, `MatMulGradFn_cache_a`, `MatMulGradFn_grad_a`, `MatMulGradFn_cache_b` | 16622.08 |
| Bias/broadcast add | `BiasAddGradFn_grad_input`, `broadcast_add_result` | 5260.40 |
| Silu/SwiGLU elementwise | `ElementwiseMulGradFn_grad_a`, `ElementwiseMulGradFn_grad_b`, `SiluGradFn_input_grad`, `emul_result`, `silu_result` | 8820.00 |
| RMSNorm | `RMSNormGradFn_cache`, `RMSNormGradFn_input_grad`, `rms_norm_result` | 3267.00 |
| Add/mul-scalar | `AddGradFn_grad_a`, `MulScalarGradFn_input_grad`, `add_result`, `mul_scalar_result`, `AddGradFn_grad_b` | 4707.92 |

The largest measured cumulative allocation labels are TensorContract/autograd labels and elementwise/FFN labels. This is a statement about emitted allocation labels only; it does not prove each label's bytes are still live at `post_forward`.

## Retained `ModelForwardOutputs` Breakdown

Batch 1 retained `224` tensors in `ModelForwardOutputs`, totaling `15057.23 MiB`.

Largest retained groups from batch 1:

| Retained group | Calculation from logged tensors | MiB |
|---|---|---:|
| FFN retained intermediates | `ffn_gate_out_per_layer`, `ffn_silu_out_per_layer`, `ffn_linear1_out_per_layer`, `ffn_swiglu_out_per_layer`: `4 * 12 * 144.00` | 6912.00 |
| Main logits plus MTP logits | `logits_tensor` + `mtp_logits_tensors[0..3]`: `5 * 481.03` | 2405.15 |
| QKV projection outputs | `qkv_out_per_layer`: `12 * 60.00` | 720.00 |
| Per-layer Q tensors | `Q_bhsd_per_layer`: `12 * 36.00` | 432.00 |
| Per-layer K tensors | `K_bhsd_per_layer`: `12 * 12.00` | 144.00 |
| Per-layer V tensors | `V_bhsd_per_layer`: `12 * 12.00` | 144.00 |
| Encoder layer outputs | `encoder_layer_outputs[0..11]`: `12 * 36.00` | 432.00 |
| LM-head MLP 36 MiB tensors | gate, silu, up, swiglu, residual: `5 * 36.00` | 180.00 |

The retained output report accounts for a stable `~15056.64 MiB` of forward-owned tensors. The retained FFN intermediate group is the largest measured retained subset inside `ModelForwardOutputs`.

## Loss And Failure Markers

Loss summary from `900` `[LossStats]` entries:

| Measurement | Value |
|---|---:|
| Min `loss_mean` | 5.9945 |
| Max `loss_mean` | 8.3207 |
| Avg `loss_mean` | 7.1537 |
| Batch 1 `loss_mean` | 8.1788 |
| Batch 900 `loss_mean` | 6.1153 |

Failure marker scan:

| Pattern group | Count |
|---|---:|
| `CUDA error`, `FAILED`, `out of memory`, `OOM`, `runtime_error`, `Exception`, `Non-finite` | 0 |

## Conclusions From Measured Data

1. The original forward memory rise is real in the log samples. The sampled live forward delta averages `40809.16 MiB` across the parsed `[GPU_MEM]` samples.

2. `ModelForwardOutputs` does not account for the full live forward delta. It retains an average of `15056.64 MiB`, leaving an average of `25752.52 MiB` live outside `ModelForwardOutputs` at the `post_forward` sample.

3. The scoped allocation ledger captured substantially more forward allocation volume than the sampled live delta. The average cumulative forward allocation total is `46165.79 MiB`, which is `5356.63 MiB` above the average sampled live forward delta. This is expected for a cumulative allocation ledger and does not mean `46165.79 MiB` is all live simultaneously.

4. The largest emitted allocation labels are `matmul_result`, `MatMulGradFn_cache_a`, `MatMulGradFn_grad_a`, `BiasAddGradFn_grad_input`, `broadcast_add_result`, and Silu/elementwise labels. These labels provide the strongest measured attribution for the forward allocation volume.

5. The largest retained `ModelForwardOutputs` subset is FFN retained intermediates at `6912.00 MiB`, followed by main plus MTP logits at `2405.15 MiB`.

## Batch-Size-1 Run: `training_1783467717196987649.log`

This run used the same model shape as the batch-size-12 run, but with `batch_size=1` and `max_tokens_per_batch=1024`.

Run configuration emitted by the log:

| Field | Value |
|---|---:|
| `batch_size` | 1 |
| `max_tokens_per_batch` | 1024 |
| `max_seq_len` | 1024 |
| `gradient_accumulation_steps` | 2 |
| `vocab_size` | 10262 |
| `d_model` | 768 |
| `d_ff` | 3072 |
| `num_layers` | 12 |
| `num_heads` | 12 |
| `num_kv_heads` | 4 |

### Parsed Coverage

The log contains `987` `[ForwardAllocationSizes]` blocks and `987` `[ForwardOutputSizes]` blocks. It contains `986` `[LossStats]` rows. The tail of the file stops inside the final `[ForwardOutputSizes] batch=987` block, so batch `987` has forward allocation/output data but no matching loss row in this file.

| Last observed section | Batch |
|---|---:|
| Last `[ForwardAllocationSizes]` | 987 |
| Last `[ForwardOutputSizes]` | 987 |
| Last `[LossStats]` | 986 |

### Run-Wide Memory Summary

Parsed complete allocation/output blocks:

| Measurement | Count | Min MiB | Max MiB | Avg MiB |
|---|---:|---:|---:|---:|
| `[ForwardAllocationSizes].total_MiB` | 987 | 4341.14 | 4373.12 | 4368.51 |
| `[ForwardOutputSizes].total_MiB` | 987 | 1254.19 | 1254.48 | 1254.23 |
| Allocation total minus retained output total | 987 | 3086.95 | 3118.64 | 3114.28 |

Sampled live GPU memory deltas from `[GPU_MEM]` rows:

| Measurement | Count | Min MiB | Max MiB | Avg MiB |
|---|---:|---:|---:|---:|
| `post_forward - pre_forward` | 20 | 4472.00 | 4542.00 | 4480.10 |
| `post_backward - pre_forward` | 20 | 1654.00 | 4750.00 | 1813.70 |

Derived from run-wide averages:

| Relationship | Value MiB |
|---|---:|
| Avg live forward delta not retained in `ModelForwardOutputs` | 3225.87 |
| Avg cumulative forward allocation minus sampled live forward delta | -111.59 |

The negative allocation-minus-live value means the sampled live forward growth is larger than the cumulative allocations observed inside the scoped forward ledger. The current log therefore does not support treating `[ForwardAllocationSizes]` as a complete explanation for every byte of sampled live forward growth in this batch-size-1 run. It does establish that `ModelForwardOutputs` accounts for about `1254.23 MiB` of the average `4480.10 MiB` sampled live forward delta.

### Concrete Batch Samples

#### Batch 1

Measured log entries:

| Field | Value |
|---|---:|
| `pre_forward device_used` | 2703.10 MiB |
| `[ForwardAllocationSizes].total_MiB` | 4358.67 MiB |
| `post_forward device_used` | 7245.10 MiB |
| `[ForwardOutputSizes].total_MiB` | 1254.22 MiB |
| `post_backward device_used` | 7453.10 MiB |

Derived values:

| Relationship | Value MiB |
|---|---:|
| `post_forward - pre_forward` | 4542.00 |
| Live forward delta not retained in `ModelForwardOutputs` | 3287.78 |
| Cumulative allocation minus live forward delta | -183.33 |

#### Batch 50

Measured log entries:

| Field | Value |
|---|---:|
| `pre_forward device_used` | 5813.10 MiB |
| `[ForwardAllocationSizes].total_MiB` | 4371.35 MiB |
| `post_forward device_used` | 10285.10 MiB |
| `[ForwardOutputSizes].total_MiB` | 1254.29 MiB |
| `post_backward device_used` | 7467.10 MiB |

Derived values:

| Relationship | Value MiB |
|---|---:|
| `post_forward - pre_forward` | 4472.00 |
| Live forward delta not retained in `ModelForwardOutputs` | 3217.71 |
| Cumulative allocation minus live forward delta | -100.65 |

#### Batch 950

Measured log entries:

| Field | Value |
|---|---:|
| `pre_forward device_used` | 5809.10 MiB |
| `[ForwardAllocationSizes].total_MiB` | 4370.43 MiB |
| `post_forward device_used` | 10285.10 MiB |
| `[ForwardOutputSizes].total_MiB` | 1254.19 MiB |
| `post_backward device_used` | 7467.10 MiB |

Derived values:

| Relationship | Value MiB |
|---|---:|
| `post_forward - pre_forward` | 4476.00 |
| Live forward delta not retained in `ModelForwardOutputs` | 3221.81 |
| Cumulative allocation minus live forward delta | -105.57 |

### Largest Forward Allocation Labels

Batch 2 is used as the representative full-label sample because it has the steady `944` allocations and `65` labels. Batch 1 had `938` allocations and `59` labels, with the same dominant large labels but fewer tiny optional labels.

Batch 2 contained `944` forward allocations across `65` labels, totaling `4371.42 MiB`.

Top allocation labels from batch 2:

| Label | Count | MiB | Max Request MiB |
|---|---:|---:|---:|
| `matmul_result` | 75 | 651.30 | 40.09 |
| `MatMulGradFn_cache_b` | 75 | 574.54 | 30.06 |
| `MatMulGradFn_cache_a` | 75 | 342.75 | 12.00 |
| `MatMulGradFn_grad_a` | 75 | 342.75 | 12.00 |
| `BiasAddGradFn_grad_input` | 10 | 219.18 | 40.09 |
| `broadcast_add_result` | 10 | 219.18 | 40.09 |
| `ElementwiseMulGradFn_grad_a` | 13 | 147.00 | 12.00 |
| `ElementwiseMulGradFn_grad_b` | 13 | 147.00 | 12.00 |
| `SiluGradFn_input_grad` | 13 | 147.00 | 12.00 |
| `emul_result` | 13 | 147.00 | 12.00 |
| `silu_result` | 13 | 147.00 | 12.00 |
| `RMSNormGradFn_cache` | 31 | 90.75 | 3.00 |
| `RMSNormGradFn_input_grad` | 31 | 90.75 | 3.00 |
| `rms_norm_result` | 31 | 90.75 | 3.00 |
| `AddGradFn_grad_a` | 27 | 78.11 | 3.00 |
| `MulScalarGradFn_input_grad` | 27 | 78.11 | 3.00 |
| `add_result` | 27 | 78.11 | 3.00 |
| `mul_scalar_result` | 27 | 78.11 | 3.00 |
| `AddGradFn_grad_b` | 26 | 78.00 | 3.00 |
| `sdpa_result` | 24 | 72.00 | 3.00 |
| `SplitQKV_merged_qkv_grad` | 12 | 60.00 | 5.00 |
| `SliceColumnsGradFn_grad_x` | 4 | 48.00 | 12.00 |
| `DropoutGradFn_input_grad` | 13 | 39.00 | 3.00 |
| `dropout_seeded_result` | 13 | 39.00 | 3.00 |
| `qkv_split_Q` | 12 | 36.00 | 3.00 |

Measured grouping by label family for batch 2:

| Family | Included labels | Batch 2 MiB |
|---|---|---:|
| Matmul result/cache/grad-a/cache-b | `matmul_result`, `MatMulGradFn_cache_a`, `MatMulGradFn_grad_a`, `MatMulGradFn_cache_b` | 1911.34 |
| Bias/broadcast add | `BiasAddGradFn_grad_input`, `broadcast_add_result` | 438.36 |
| Silu/SwiGLU elementwise | `ElementwiseMulGradFn_grad_a`, `ElementwiseMulGradFn_grad_b`, `SiluGradFn_input_grad`, `emul_result`, `silu_result` | 735.00 |
| RMSNorm | `RMSNormGradFn_cache`, `RMSNormGradFn_input_grad`, `rms_norm_result` | 272.25 |
| Add/mul-scalar | `AddGradFn_grad_a`, `MulScalarGradFn_input_grad`, `add_result`, `mul_scalar_result`, `AddGradFn_grad_b` | 390.44 |

These listed families account for `3747.39 MiB` of the batch 2 cumulative allocation total, or about `85.73%` of `4371.42 MiB`.

### Retained `ModelForwardOutputs` Breakdown

Batch 1 retained `224` tensors in `ModelForwardOutputs`, totaling `1254.22 MiB`.

Largest retained groups from batch 1:

| Retained group | Calculation from logged tensors | MiB |
|---|---|---:|
| FFN retained intermediates | `ffn_gate_out_per_layer`, `ffn_silu_out_per_layer`, `ffn_linear1_out_per_layer`, `ffn_swiglu_out_per_layer`: `4 * 12 * 12.00` | 576.00 |
| Main logits plus MTP logits | `logits_tensor` + `mtp_logits_tensors[0..3]`: `5 * 40.09` | 200.45 |
| QKV projection outputs | `qkv_out_per_layer`: `12 * 5.00` | 60.00 |
| Per-layer Q tensors | `Q_bhsd_per_layer`: `12 * 3.00` | 36.00 |
| Per-layer K tensors | `K_bhsd_per_layer`: `12 * 1.00` | 12.00 |
| Per-layer V tensors | `V_bhsd_per_layer`: `12 * 1.00` | 12.00 |
| Encoder layer outputs | `encoder_layer_outputs[0..11]`: `12 * 3.00` | 36.00 |
| LM-head MLP 3 MiB tensors | gate, silu, up, swiglu, residual: `5 * 3.00` | 15.00 |

The retained output report accounts for a stable `~1254.23 MiB` of forward-owned tensors. The retained FFN intermediate group is again the largest measured retained subset inside `ModelForwardOutputs`.

### Loss And Failure Markers

Loss summary from `986` `[LossStats]` entries:

| Measurement | Value |
|---|---:|
| Min `loss_mean` | 6.95 |
| Max `loss_mean` | 9.21 |
| Avg `loss_mean` | 7.98 |
| Avg `valid_tokens` | 798.91 |
| Avg `masked_tokens` | 225.09 |

Failure marker scan:

| Pattern group | Count |
|---|---:|
| `CUDA error`, `FAILED`, `out of memory`, `OOM`, `runtime_error`, `Exception`, `Non-finite` | 0 |

### Comparison To Batch-Size-12 Run

Measured averages:

| Measurement | Batch size 12 | Batch size 1 | Ratio |
|---|---:|---:|---:|
| `[ForwardAllocationSizes].total_MiB` avg | 46165.79 | 4368.51 | 10.57x |
| `[ForwardOutputSizes].total_MiB` avg | 15056.64 | 1254.23 | 12.00x |
| Sampled `post_forward - pre_forward` avg | 40809.16 | 4480.10 | 9.11x |
| Retained FFN intermediates | 6912.00 | 576.00 | 12.00x |
| Main logits plus MTP logits | 2405.15 | 200.45 | 12.00x |

The retained `ModelForwardOutputs` total scales exactly with token capacity in these two runs: `15056.64 MiB / 1254.23 MiB = 12.00x`, matching `12288 / 1024 = 12.00x`. The cumulative forward allocation ledger scales less than 12x (`10.57x`) because it includes fixed or weakly token-scaled allocations, and because the relative contribution of `MatMulGradFn_cache_b` is larger in the batch-size-1 label table.

### Conclusions From Measured Data

1. The single-batch run reduces retained `ModelForwardOutputs` from `15056.64 MiB` to `1254.23 MiB`, a measured `12.00x` reduction that matches the token-capacity reduction from `12288` to `1024`.

2. The sampled live forward delta drops from `40809.16 MiB` to `4480.10 MiB`, a measured `9.11x` reduction.

3. The cumulative forward allocation ledger drops from `46165.79 MiB` to `4368.51 MiB`, a measured `10.57x` reduction.

4. `ModelForwardOutputs` accounts for `1254.23 MiB` of the average `4480.10 MiB` sampled live forward delta, leaving `3225.87 MiB` of sampled live growth outside `ModelForwardOutputs`.

5. The largest batch-size-1 allocation families remain matmul/cache/grad, Silu/SwiGLU elementwise, bias/broadcast add, add/mul-scalar, and RMSNorm. Their absolute MiB totals are smaller than batch size 12, but the dominant label families are unchanged.

6. The log has no failure, OOM, CUDA error, exception, runtime error, or non-finite markers in the scanned pattern group.

## Evidence Boundaries

This report does not claim:

- which individual allocation labels are still live at `post_forward`;
- which buffers are safe to recompute, defer, fuse, or delete;
- whether any allocation is unnecessary.

Those require a live-allocation ledger or owner/lifetime instrumentation that records allocation and free events by pointer/label across the forward boundary. The current report establishes allocation volume, retained output volume, sampled live forward growth, and the measured gap between those views.