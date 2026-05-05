# Diagnostics & Telemetry

## RMSNorm expected output
$$\text{expected\_output\_rms} = \frac{\text{input\_rms} \cdot \gamma_{\text{rms}}}{\sqrt{\text{input\_rms}^2 + \epsilon}}$$

Not just `gamma_rms`. With small Xavier-init embeddings (rms ≈ 0.006), epsilon contributes ~20%.

## Hidden-state buffer
Read `cached_encoder_output` (post-centering, overwritten after LM head forward) for hidden-state diagnostics. `centering_scratch_tensor` was deleted — single buffer is the source of truth.

## Per-step batch geometry
Diagnostics must use the Phase1-authored `BatchPayload` for `batch_size`, `max_seq_len`, `total_tokens`, sequence lengths, and LM valid-token counts. Authored capacity lives in `LanguageModelConfig` / `RunCapacity`; actual allocation capacity lives in Tensor shapes. `TrainingState` must not mirror current-batch geometry (`cached_batch_size` / `cached_seq_len` / `cached_valid_tokens`) or authored capacity (`max_cached_batch` / `max_cached_seq_len` / `max_cached_tokens` / `max_logit_tokens`) as shadow state.

## Kernel timing
Use CUDA events (`cudaEventRecord` / `cudaEventElapsedTime`) — not `cudaStreamSynchronize` wall-time. Sync timing includes draining prior pipeline work.

## Gradient norm diagnostics
`runGradientNormClipDiagnostic()` consumes the `ClipResult` produced by `GRIM::GradClip::clipGradientNorms()`. It does **not** launch grad-norm kernels, allocate scratch, or synchronize the stream. The only grad-norm measurement in the hot loop is the clipping-owned measurement on the optimizer-step boundary; diagnostics and gradient-dependent telemetry may read that measured result but must not create a second measurement path.

Encoder RMS in the clip result is optional telemetry. If the summed encoder-like group count is zero (disabled modules, early init, sparse routing), emit `NaN` for `encoder_rms_pre`; only non-finite sums with a positive count are fatal.

Clip-result diagnostics validate the `ClipResult` contract directly: pre/post global RMS values must be finite and non-negative, post RMS must not exceed pre RMS, clipped results must actually reduce RMS, and unclipped results must preserve RMS within tolerance. GradNorm group-count checks compare `GradMetrics.groups_processed` against `ClipResult.measured_group_count`, not against `model->parameterGroups().size()`, so future filtered/fused clipping topologies remain valid.

`emb_rms_pre` is logged in the main `POST-CLIP-MEASURE` line because embedding/LM-head rows are often the highest-variance component. The previous embedding RMS used for spike deltas lives in `TrainingLoopState.diagnostics.prev_emb_rms`; logging sinks only emit text and must not own evolving numeric diagnostic state.

`computeEmbGradEquation()` is **not async-safe**: it performs blocking D2H copies for token IDs and embedding gradients so CPU code can compute row/frequency statistics. It is permitted only inside the `shouldSyncDiagnostics()` path after the clipping-owned measurement has completed; do not call it from ordinary per-batch async paths.

## LibTorch baselines
Gradient comparisons are valid **only** when the baseline uses an IDENTICAL config (`d_model`, `num_layers`, `num_heads`, `batch_tokens`). Different configs produce inherently different gradient magnitudes.

## Loss / gradient scaling double-application
Loss backward already applies `1/N`. Do **not** add another `1/tokens` scaling in parameter grad kernels.

## TelemetryLattice
Hierarchical streaming statistics: 8 levels, 5 metric streams. Stream 38 (`rho_raw_rms_spread`) is the canonical hidden-state health signal — see [LMHead.md](LMHead.md).

## Accumulation-window logging
`ctx.optimizer.optimizer_step.step` is the zero-based optimizer step index supplied to AdamW/RAdamW and LR scheduling. `ctx.optimizer.accumulation_position` is the private in-progress index inside the accumulation window, not a second optimizer step. Diagnostics that fire at the optimizer boundary must consume only the single configured accumulation window size and format it as `accum_window=N`; do not pass separate completed/required log values or derive a parallel step path. Telemetry must receive the same zero-based `optimizer_step` used by the optimizer kernel, even though `completeOptimizerStep()` increments the stored step before telemetry is emitted.

## External references
- [docs/LOG_FILE_CONVENTION.md](../../../../../docs/LOG_FILE_CONVENTION.md) — verify which log file before making claims.
- [docs/PLATEAU_BUG_INVESTIGATION.md](../../../../../docs/PLATEAU_BUG_INVESTIGATION.md) — active training investigation notes.
