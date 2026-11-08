# Telemetry

Lattice measurement, sigma/raw-mirror math, and the retained observer API are
one documentation domain. Measurements must not be mistaken for a connected
adaptive training controller.

## Lattice statistics

[TelemetryLattice_GPU.cu](../../resources/models/GRIM-text/Shared/Telemetry/TelemetryLattice_GPU.cu)
implements the update; [TelemetryState_GPU.hpp](../../resources/models/GRIM-text/Shared/Telemetry/TelemetryState_GPU.hpp)
defines its persistent state. For one stream/level observation `x`:

```text
mu_raw <- beta_mu * mu_raw + (1-beta_mu) * x
m2_raw <- beta_mu * m2_raw + (1-beta_mu) * x*x
beta_power <- beta_power * beta_mu
mu = mu_raw / (1-beta_power)
m2 = m2_raw / (1-beta_power)
sigma = sqrt(max(m2-mu*mu, epsilon*mu*mu, 0))
```

The kernel records `sigma_jump = sigma - old_sigma_prev`, updates `v_sigma`,
then assigns `sigma_prev = sigma`. Slope computation happens **after** this
assignment: the field named `sigma_prev` contains current sigma at that point.
Do not document its slope denominator as the prior update's sigma.

Raw and normalized first-sample slopes are zero. The scale-aware floor can
produce nonzero first-sample sigma for nonzero observations; a zero observation
can still have zero sigma. The guard, not an assumed positive initial variance,
prevents the first-sample slope spike.

| Metric | Dependency |
|---|---|
| `sigma_tilde` | `sigma/(abs(mu)+epsilon)` |
| `sigma_jump` | Current minus stored prior sigma, without sigma division |
| `v_sigma` | EMA of squared sigma jump |
| `delta_raw` | Guarded `x-old_mu_prev`, without sigma |
| `delta_bar_raw` | EMA of raw delta, without sigma normalization |
| `delta_bar` | EMA of guarded `(x-old_mu_prev)/(sigma_prev+epsilon)` after assignment |
| `p` | EMA of normalized slope's sign |
| `k_out` | `k_out0 * (1 + alpha_v * safeSqrt(v_sigma,epsilon))` |
| `c_out` | `mu + k_out * sigma` |
| `outlier_raw` | `x-c_out`: no sigma division, but cutoff still depends on sigma |
| `r_out`, `ell_out` | EMAs of the sigmoid outlier gate |
| `mu_ex` | EMA of `max(0,x-c_out)` |
| `delta_mu` | Fast mean minus slow anchor mean, not sigma-normalized |
| `delta_sigma` | Sigma minus slow sigma anchor |

Higher levels update on their stride using the preceding level's mean and the
same bias correction/floor/slope guards. Raw mirrors remove particular
normalization dependencies, not every dependency of an adaptive cutoff.

## Active measurement and logging

[TelemetryUpdate.cu](../../resources/models/GRIM-text/Shared/Telemetry/TelemetryUpdate.cu),
the lattice, and [TelemetryCsvLogger.cu](../../resources/models/GRIM-text/Shared/Telemetry/TelemetryCsvLogger.cu)
remain measurement/logging paths. These are distinct from the retained decision
observer. Identify the log/CSV, stream/level, sample count, and active config
when interpreting spikes. Old CV/autocorrelation and multi-thousand-step
transient numbers are historical observations, not guarantees of these equations.
See [log attribution](../LOG_FILE_CONVENTION.md).

## Monitoring boundary

The April 2026 cleanup removed skip/quarantine/rescale/noise/cooldown/restart
actions from the observer.
[TelemetryControl_GPU.hpp](../../resources/models/GRIM-text/Shared/Telemetry/TelemetryControl_GPU.hpp)
now exposes Continue/FatalError only. `ControlDecision` is 48 bytes with action,
spike severity, accumulation-bug flags, ratio, normalized gradient, error code,
and padding. Persistent observer state is 64 bytes.

[TelemetryControl_GPU.cu](../../resources/models/GRIM-text/Shared/Telemetry/TelemetryControl_GPU.cu)
can classify spike severity and repeated low-gradient/nonzero-loss conditions.
Its diagnostic normalized gradient is not a command to modify parameter grads.

[TelemetryInitInputs.cu](../../resources/models/GRIM-text/training/Phases/Startup/Telemetry/TelemetryInitInputs.cu)
authors observer config, and [Phase1_Startup.hpp](../../resources/models/GRIM-text/training/Phases/Phase1_Startup.hpp)
retains config/controller fields. Current phase call sites do not invoke
`TelemetryControl::evaluate()` as a per-step training controller.
Class/config/kernel existence alone is not evidence of live decisions.

Actual [Phase2 checks](../../resources/models/GRIM-text/training/Phases/Phase2_TrainingLoop.cu)
and [autograd checks](../../resources/models/GRIM-text/training/Autograd/AutogradTraining.cu)
reject invalid/empty payloads, verify gradient delivery, check finite gradients
at the clipping boundary, and check updated weights. Intended accumulation
weighting, global clipping, LR scheduling, and configured optimizer updates
remain active math, not the removed telemetry interventions.

No training, GPU telemetry, or benchmark was run for this consolidation.
