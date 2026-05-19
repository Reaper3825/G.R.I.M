# Diagnostics & Telemetry

## RMSNorm expected output
$$\text{expected\_output\_rms} = \frac{\text{input\_rms} \cdot \gamma_{\text{rms}}}{\sqrt{\text{input\_rms}^2 + \epsilon}}$$

Not just `gamma_rms`. With small Xavier-init embeddings (rms ≈ 0.006), epsilon contributes ~20%.

## Hidden-state buffer
Read `cached_encoder_output` (the actual LM-head matmul input, written after `LMHeadLayer::forward`) for hidden-state diagnostics. `centering_scratch_tensor` was deleted — single diagnostic snapshot is the source of truth. Runtime inference consumers must not read this global snapshot for decode decisions; the decode-time selector consumes the explicit live prefill hidden pointer before `AutogradIntermediates::clear()`.

## Logit-scale population
`LOGIT_SCALE_EQUATION` computes both actual `logit_std` and expected `logit_std` over the same LM-valid target rows: positions where `BatchPayload.target_ids[pos] >= 0`. The required count is `BatchPayload.lm_valid_tokens`; diagnostics must derive concrete row indices from `target_ids` and fail if the counted positions disagree with `lm_valid_tokens`. Do not use a contiguous prefix of the rectangular `[batch_size * max_seq_len]` buffer, and do not include padding/final-position/execution-slot-masked rows. The expected formula uses `h_rms_rms = sqrt(mean_t(rms(h_t)^2))`, not arithmetic `h_rms_mean`, because it is a variance estimate.

## LM-head GEMM equation tracing
`ENABLE_GEMM_EQUATION_LOGS` in `Shared/VerboseLogging.hpp` enables focused LM-head GEMM tracing when the BatchLogTape accepts `Debug` and `skipThisPass()` is false. The diagnostic implementation lives in `Shared/TensorContract/LMHeadGemmDiagnostics.{hpp,cu}`; production LM-head and matmul code should only call its narrow hook functions. Forward emits `[LM_HEAD_GEMM_EQUATION]` around `logits = lm_input @ W_eff^T`, including the executed centering order, actual GEMM geometry, sampled `lm_input` / `W_eff` / logits min-max-RMS, `W_eff` row-mean residual, and expected-vs-actual logit RMS. The `sample_shape` fields are bounded diagnostic prefixes (`lm_input` rows capped at 128, `W_eff` rows capped at 512, logits prefix capped at 64x256); use `lm_input_actual_shape`, `logits_actual_shape`, and clamp booleans in the `GEOMETRY` line when determining whether a pass is a full training batch or an autoregressive generation prefill. Backward emits `[LM_HEAD_GEMM_BACKWARD_EQUATION]` plus forced grad-flow stats for `grad_logits`, `grad_lm_input_pre_centering`, and `grad_W_eff_pre_center_rows` before `CenterRowsGradFn` / `CenterColumnsGradFn` propagate the gradient. Use this path for Issue #132 / early hidden-state alignment collapse investigations.

## Per-step batch geometry
Diagnostics must use the Phase1-authored `BatchPayload` for `batch_size`, `max_seq_len`, `total_tokens`, sequence lengths, and LM valid-token counts. Authored capacity lives in `LanguageModelConfig` / `RunCapacity`; actual allocation capacity lives in Tensor shapes. `TrainingState` must not mirror current-batch geometry (`cached_batch_size` / `cached_seq_len` / `cached_valid_tokens`) or authored capacity (`max_cached_batch` / `max_cached_seq_len` / `max_cached_tokens` / `max_logit_tokens`) as shadow state.

## Kernel timing
Use CUDA events (`cudaEventRecord` / `cudaEventElapsedTime`) — not `cudaStreamSynchronize` wall-time. Sync timing includes draining prior pipeline work.

## QKV finite scans
`GRIM_DEBUG_QKV` enables full-tensor finite checks around encoder QKV / SDPA boundaries. Clean tensors are silent; any NaN/Inf emits a `[QKV_NONFINITE] FATAL ...` module error with counts and first offending index/value, then throws immediately. Do not log `nan=0 inf=0` summaries — they hide the real anomaly signal.

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
Hierarchical streaming statistics: 8 levels. Stream 38 (`rho_raw_rms_spread`) is the canonical hidden-state health signal — see [LMHead.md](LMHead.md). Streams 39–44 track h↔W LM-head alignment, streams 45–46 track unigram-frequency-direction alignment, stream 47 tracks the raw `lm_head_w_rms_rms` term used by the logit-scale equation, streams 48–54 publish the compact init-time tied-embedding / optimizer-group invariant mirror directly into `telemetry_<session>.csv`, and streams 55–57 track final-layer rho signed/centered/mean-vector diagnostics (`rho_raw_avg_signed_dot`, `rho_centered_avg_abs_dot`, `rho_mean_vector_rms`). The full init-facts key/value dump is emitted through LogRecorder into `training_<session>_tape.log`; there is no separate init-facts CSV. `TelemetryCsvLogger` resolves stream names through the `MetricStream` single source of truth. `view_telemetry.py` repairs legacy CSV rows named `unknown` by using `stream_idx` before graphing.

Train-loss spike / EWMA detection is owned by TelemetryLattice. Do not add separate per-batch scalar spike detectors or diagnostic subscribers; they drift from telemetry-owned stream statistics. `Shared/Loss/LossSignals` owns only validation high-loss patience (`validation_high`) for auto-stop; it must not compute spike or delta signals.

## Atom-token display
Diagnostics that summarize arbitrary token IDs without per-token side channels must not call `UniByte::decode({tid})` on atom IDs. Atom tokens are type placeholders (`<INT>`, `<FLOAT>`) and require `atom_entry_ids` plus an `AtomTable` to reconstruct original text. Aggregate displays such as Rho top-token logs should render atom IDs by type; full sample reconstruction must use the atom-aware side-channel decode path.

## Accumulation-window logging
`ctx.optimizer.optimizer_step.step` is the zero-based optimizer step index supplied to AdamW/RAdamW and LR scheduling. `OptimizerContext` owns the private in-progress accumulation-slot cursor exposed by `accumulationSlot()`; one slot is exactly one `BatchPayload` upload → forward → loss → backward pass. There is no separate secondary counter/lifecycle, but there are two chronological Phase2 boundaries: `processBatch()` owns the microbatch/autograd boundary, while `runEpoch()` owns accumulation-slot advancement and the optimizer-window boundary after `processBatch()` returns. Diagnostics that fire at the optimizer boundary must consume only the single configured accumulation window size and format it as `accum_window=N`; do not pass separate completed/required log values or derive a parallel step path. Telemetry must receive the same zero-based `optimizer_step` used by the optimizer kernel, even though `completeOptimizerStepAfterFullAccumulationWindow()` increments the stored step before telemetry is emitted.

## External references
- [docs/LOG_FILE_CONVENTION.md](../../../../../docs/LOG_FILE_CONVENTION.md) — verify which log file before making claims.
- [docs/PLATEAU_BUG_INVESTIGATION.md](../../../../../docs/PLATEAU_BUG_INVESTIGATION.md) — active training investigation notes.
