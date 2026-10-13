# Gradient and residual-gate MetricStreams

`training/Diagnostics/GradientTelemetryCapture.cu` owns capture and registration.
All observations flow through `MetricStream`, `getMetricStreamName()`, the normal
telemetry lattice update, and `telemetry_<session>.csv`. The existing 29-column
CSV schema stays intact: select a series by `stream_name`, use `raw_observation`
for the measured value, and use the standard lattice statistics for trends.
There is no raw-row exporter or additional CUDA execution stream.

## Stream registration

Existing IDs and retired holes remain unchanged. IDs 94–101 identify global
capture measurements, 102–121 contain five RMS-ranked groups, and 122–141 contain
five energy-ranked groups. Each ranking has four fields per rank: group index,
gradient RMS, energy percentage, and parameter count. The `group_index` matches
the existing TOP-GRAD-GROUPS/TOP-ENERGY-GROUPS log indices; rank identities can
change from step to step.

Starting at ID 142, startup registers 34 named streams per encoder layer when
residual gating is enabled, then five streams per registered parameter group.
Names include the actual layer or parameter name. These model-specific IDs are
stable for the same architecture/registration order; use names to compare runs
with different models. `getMetricStreamName()` resolves dynamic names from the
lattice's startup registration table. Storage grows to fit the registered model,
with the authored `telemetry_lattice_num_streams` as a capacity floor. The maximum
storage capacity is 4096 streams; exceeding it fails at startup.

## Completed optimizer windows

Every registered group with a gradient publishes `<parameter>_grad_rms_pre`,
`_grad_sum_sq_pre`, `_grad_energy_pct`, `_grad_rms_rank`, and `_grad_energy_rank`.
These reuse the accumulation-normalized, pre-clip sums from clipping. Energy
percentage uses all registered gradients as the denominator; zero total energy
produces zero shares. Ranks are one-based, with ties in registration order.

Global streams include total pre-clip squared gradient energy, post-clip RMS,
clipping status/scale, combined residual-gate energy percentage, and the
one-based optimizer iteration. The existing `grad_norm_mean` still contains
pre-clip RMS. Per-layer gate weights, biases, and post-clip gradients publish
mean, population standard deviation, RMS, min, and max. They are sampled before
the optimizer mutates parameters. `layerN_attention_residual_gate_bias_grad_postclip_mean`
preserves the signed bias gradient. Parameter statistics are not update deltas.

## Gate behavior

At the configured `log_interval`, forward capture reduces multipliers and logits
for each layer, excluding padding through the existing device sequence-length
bindings. It includes all real tokens, including tokens whose direct objective
is masked. Both tensors publish mean, population standard deviation, RMS, min,
and max. Multipliers also publish the fraction below identity (1), fractions at
or below 0.1 / at or above 1.9, and the mean derivative of `2 * sigmoid(logit)`.
The saturation thresholds are diagnostic bins, not training controls.

If multiple microbatches are sampled in an accumulation window, the last sample
is published at that window's telemetry update. `gate_forward_sample_global_step`
and `gate_forward_sample_token_count` identify the source sample. If none was
sampled, forward metrics are absent for that update. Unmeasured capture slots
use a missing-sample sentinel internally and are skipped by the lattice and CSV;
they are not zeros or repeated stale measurements. NaN/Inf in a measured summary
still fails validation.

## Synchronization and plotting

Reductions run on the existing primary stream while their inputs are live.
Only small summary buffers are copied into pinned host storage, before the
optimizer boundary's existing synchronization. After that boundary, capture
publishes host observations for the normal lattice update. There is no separate
capture synchronization. CSV export retains its existing host readback boundary,
but now snapshots all lattice states in one transfer instead of one transfer
per stream and level.

Plot parameter energy percentages and RMS against `global_step` alongside loss
and learning rate. Plot gate signed bias gradients, multiplier mean/std, and
saturation fractions by layer. The normal `optimizer_iteration` stream aligns
window measurements; the gate sample-step stream identifies sampled microbatch
measurements. Missing samples should remain gaps in plots.
