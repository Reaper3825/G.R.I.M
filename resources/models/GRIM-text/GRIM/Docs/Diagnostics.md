# Diagnostics & Telemetry

## RMSNorm expected output
$$\text{expected\_output\_rms} = \frac{\text{input\_rms} \cdot \gamma_{\text{rms}}}{\sqrt{\text{input\_rms}^2 + \epsilon}}$$

Not just `gamma_rms`. With small Xavier-init embeddings (rms ≈ 0.006), epsilon contributes ~20%.

## Hidden-state buffer
Read `cached_encoder_output` (post-centering, overwritten after LM head forward) for hidden-state diagnostics. `centering_scratch_tensor` was deleted — single buffer is the source of truth.

## Kernel timing
Use CUDA events (`cudaEventRecord` / `cudaEventElapsedTime`) — not `cudaStreamSynchronize` wall-time. Sync timing includes draining prior pipeline work.

## LibTorch baselines
Gradient comparisons are valid **only** when the baseline uses an IDENTICAL config (`d_model`, `num_layers`, `num_heads`, `batch_tokens`). Different configs produce inherently different gradient magnitudes.

## Loss / gradient scaling double-application
Loss backward already applies `1/N`. Do **not** add another `1/tokens` scaling in parameter grad kernels.

## TelemetryLattice
Hierarchical streaming statistics: 8 levels, 5 metric streams. Stream 38 (`rho_raw_rms_spread`) is the canonical hidden-state health signal — see [LMHead.md](LMHead.md).

## External references
- [docs/LOG_FILE_CONVENTION.md](../../../../../docs/LOG_FILE_CONVENTION.md) — verify which log file before making claims.
- [docs/PLATEAU_BUG_INVESTIGATION.md](../../../../../docs/PLATEAU_BUG_INVESTIGATION.md) — active training investigation notes.
