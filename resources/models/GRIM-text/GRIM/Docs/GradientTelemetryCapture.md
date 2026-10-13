# Gradient and residual-gate MetricStreams

`training/Diagnostics/GradientTelemetryCapture.cu` owns capture and registration.
All observations flow through `MetricStream`, `getMetricStreamName()`, the normal
telemetry lattice update, and `telemetry_<session>.csv`. The existing 29-column
CSV schema stays intact: select a series by `stream_name`, use `raw_observation`
for the measured value, and use the standard lattice statistics for trends.
There is no raw-row exporter or additional CUDA execution stream.

## Stream registration

Existing IDs and retired holes remain unchanged. IDs 94–101 identify global
capture measurements. IDs 102–141 formerly held top-five rank slots and are now
retired: training no longer selects, sorts, logs, or publishes ranked groups.
Their historical names remain available for older CSVs; these slots are not reused.

Starting at ID 142, startup registers 34 named streams per encoder layer when
residual gating is enabled, then four streams per registered parameter group.
Names include the actual layer or parameter name. These model-specific IDs are
stable for the same architecture/registration order; use names to compare runs
with different models. `getMetricStreamName()` resolves dynamic names from the
lattice's startup registration table. Storage grows to fit the registered model,
with the authored `telemetry_lattice_num_streams` as a capacity floor. The maximum
storage capacity is 4096 streams; exceeding it fails at startup.

## Completed optimizer windows

Every registered group with a gradient publishes `<parameter>_grad_rms_pre`,
`_grad_sum_sq_pre`, `_grad_energy_pct`, and `_grad_parameter_count`.
These reuse the accumulation-normalized, pre-clip sums from clipping. Energy
percentage uses all registered gradients as the denominator; zero total energy
produces zero shares. Measurements are published in registration order, with no
training-time ranking. Missing gradients remain missing observations. Parameter
count allows offline size comparisons; energy equals count times squared RMS.

## Offline gradient intensity and tied ratio

The raw inputs for size-normalized comparisons are already captured for every
gradient-bearing group: squared gradient sum and parameter count, plus the
global squared gradient sum. `embedding_lm_head_tied` represents the shared
embedding/LM-head tensor once; its two uses do not double its parameter count.
The denominator population is the registered gradient-bearing tensors measured
by clipping, including zero-gradient tensors, rather than all model storage.

`view_telemetry.py` computes, for each group and optimizer window:

```
parameter_count_pct = 100 * group_count / sum(all_group_counts)
gradient_energy_pct = 100 * group_sum_sq / global_sum_sq
R = gradient_energy_pct / parameter_count_pct
```

For the tied group this is the requested tied gradient energy / tied parameter
count share. R = 1 means model-average squared gradient per parameter; R = 3
means three times that average. R measures gradient intensity, not optimizer
update energy. No R calculation or new ratio stream runs during training.

The viewer exports all groups to `group_gradient_energy.csv` and the tied group
to `tied_gradient_ratio.csv`, with raw inputs, shares, and the offline ratio.
It plots parameter shares and R for every group, and the tied ratio with an
R = 1 reference. Counts are never inferred from older CSVs or filled across
missing windows. Missing counts, incomplete energy totals, and zero global
energy leave R undefined (CSV gaps), rather than reporting a misleading zero.

For actual optimizer movement, the next useful raw captures are full-group
`sum(delta_weight^2)` and pre-update `sum(weight^2)`. Together with group counts,
these support update-energy share, update RMS, and update RMS / weight RMS
offline. Adaptive and decoupled weight-decay contributions should be recorded
separately if their attribution is needed; total squared energy includes their
cross term. Existing sampled optimizer-type traces are not full named-group
update-energy measurements.

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

Run `python view_telemetry.py <telemetry.csv>` from the repository root. The viewer
creates `<telemetry>_gradient_gate/` beside the CSV with energy/RMS comparisons,
energy concentration curves, global clipping diagnostics, paginated heatmaps for
all four recorded metrics plus two offline ranks of every parameter group, and
all gate statistics by layer. Gradient plots use publication steps; forward gate heatmaps
label measured columns with the source sample step and preserve unsampled columns
in gray. Heatmap columns have equal observation width, so their step labels, rather
than column spacing, indicate elapsed training steps. Log heatmaps show zero values
in gray as well. Gate gradient and parameter means/minima/maxima use a diverging
color scale to preserve sign.

The viewer computes RMS and energy ranks from all named group observations after
training, with one-based ranks and ties in registration order. It also computes
top-five concentration offline; incomplete group measurements remain gaps.
Named parameter curves retain stable identities, and the scatter comparison uses
energy and RMS from the same latest energy observation. Historical rank-slot
plots remain available for older CSVs. New CSVs do not require rank-slot streams
or precomputed ranks. Dynamic group IDs now follow the four-metric registration
layout; compare runs by stream name rather than numeric dynamic IDs.
Capture streams use these compact comparison pages by default; add
`--capture-details` after the CSV path to
also generate the full raw/lattice-state atlas for each capture stream. Older CSVs
without capture streams retain their existing plots.
