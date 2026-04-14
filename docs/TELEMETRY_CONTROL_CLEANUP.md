# TelemetryControl Cleanup — Breadcrumb Doc

**Date:** April 14, 2026  
**Scope:** Removal of all error-masking interventions from the TelemetryControl system.  
**Rationale:** The entire system was built around silent error handling (skip step, quarantine data, scale gradients, inject noise). These patterns cover up real bugs and poison training. `telemetry_control_active = false` was already hardcoded in Phase2, so ALL intervention logic was dead code. This cleanup formalizes that by deleting it.

**Rule applied:** Rule 20 (Fail Loud) — crash with clear error on unexpected state, never silently skip.

---

## Files Modified

### 1. `resources/models/GRIM-text/Shared/Telemetry/TelemetryControl_GPU.hpp`
*(Previously modified session — completed before this session's breadcrumb)*

**What was removed:**
- `TelemetryControlConfig` fields: `moderate_grad_scale`, `moderate_cooldown_extension`, `seq_len_regime_change_threshold`, `regime_change_suppression_steps`, `volatility_damping_threshold`, `max_volatility_damping`, `gradient_decay_threshold`, `max_decay_boost`, `progress_boost_threshold`, `max_progress_boost`, `outlier_frequency_trigger`, `outlier_persistence_trigger`, `anchor_drift_sigma_multiplier`, `soft_restart_cooldown_steps`, all `plateau_noise_*` fields.
- `ControlAction` enum values: `SkipStep`, `ExtendCooldown`, `TriggerSoftRestart`, `InjectPlateauNoise`, `ScaleGradients`. **Only `Continue=0` and `FatalError=1` remain.**
- `ControlDecision` fields: `grad_scale_factor`, `cooldown_extension`, `volatility_damping`, `decay_boost`, `progress_boost`, `outlier_detected`, `anchor_drift_detected`, `plateau_detected`. **Remaining fields:** `action`, `spike_severity`, `flags` (only `FLAG_ACCUMULATION_BUG`), `spike_ratio`, `normalized_grad`, `error_code`, `_reserved[5]` (48 bytes total, static_assert preserved).
- `TelemetryControlState_GPU` fields: all per-step tracking except `consecutive_zero_grad_steps`, `step_count`, `_reserved[14]` (64 bytes total, static_assert preserved).
- `launchPlateauNoiseInjection` declaration.

**What was kept:**
- Reference values (`reference_tokens`, `reference_seq_len`)
- Spike thresholds (diagnostic only: `spike_mild_threshold`, `spike_moderate_threshold`, `spike_severe_threshold`)
- Accumulation bug config (`min_grad_for_nonzero_loss`, `loss_threshold_for_grad_check`, `max_consecutive_zero_grad_steps`, `fail_loud_on_accumulation_bug`)
- Monitoring config (`warmup_steps`, `baseline_stabilization_steps`, `verbose_logging`)

**If something breaks:** Any code that accesses old `ControlDecision` fields (e.g., `decision.grad_scale_factor`, `decision.cooldown_extension`) will fail to compile. Search for `telemetry_decision.` or `decision\.` in .cu files.

---

### 2. `resources/models/GRIM-text/Shared/Telemetry/TelemetryControl_GPU.cu`
*(Previously modified session — completed before this session's breadcrumb)*

**What was removed:**
- Device functions: `d_computeVolatilityDamping`, `d_computeDecayBoost`, `d_computeProgressBoost`, `d_detectOutlierRegime`, `d_detectAnchorDrift`
- Entire `plateauNoiseKernel` `__global__` kernel and its `launchPlateauNoiseInjection` host wrapper
- All intervention logic from `controlDecisionKernel`: regime change tracking, outlier/drift detection, action priority logic, scaling factor computation, plateau detection

**What was kept:**
- `d_computeNormalizedGrad` — computes normalized grad for monitoring
- `d_computeSpikeSeverity` — spike severity 0-3 for diagnostic logging
- `d_checkAccumulationBug` — the only legitimate FatalError trigger (zero grads + non-zero loss = disconnected autograd)
- `controlDecisionKernel` rewritten to: accumulation bug check only → FatalError, everything else → Continue
- `describeDecision` simplified to log action + spike info + accumulation bug flag

**If something breaks:** The kernel still runs but only produces `Continue` or `FatalError`. If `launchPlateauNoiseInjection` is called anywhere (shouldn't be), it will fail to link. Search `grep -r "launchPlateauNoiseInjection"`.

---

### 3. `resources/models/GRIM-text/training/Phases/Phase2_TrainingLoop.cu`
**Primary file of this session's edits.**

#### 3a. Data corruption quarantine → crash
**Location:** `processBatch()`, inside the DataGuard token validity scan, after the two nested scan loops.

**Before:** ~80 lines — when `has_invalid_tokens == true`:
- Logged a `[DataGuard] SKIPPING` message
- Iterated `payload.seq_ids` and incremented `state.spike_counts[sid]` + inserted into `state.quarantined_seqs`
- Wrote a detailed report to `<checkpoint_dir>/bad_sequences.log` (decoded text preview, per-sequence corruption details)
- Reset `ctx.optimizer.current_micro_step = 0` and cleared `autograd_intermediates`
- Set `result.skipped = true`, `result.skip_reason = "data_corruption"`, incremented `ctx.global_step`, early returned

**After:** 6 lines — throws `std::runtime_error` with message:
```
DATA CORRUPTION: batch <N> contains token IDs outside vocab range [0, <vocab_size>) — fix data pipeline at <FILE>:<LINE>
```

**IMPORTANT:** The token scan code above this point is UNCHANGED — it still correctly detects `input_ids` outside `[0, actual_vocab_size)` and `target_ids` outside `[-1, actual_vocab_size)`. The only change is what happens when corruption is detected.

**If something breaks:** Training will terminate immediately if any token ID is out of range. The error message includes batch number and vocab size. Root cause: the data pipeline produced an invalid token ID. Check `DataLoader.cu`'s `buildBatchPayload` or tokenizer output.

#### 3b. `handleGradientSpike` call site removed
**Location:** `processBatch()`, after the `// === TIMING GUARD ===` block, before the telemetry section.

**Before:**
```cpp
// Handle gradient spikes
auto spike_start = std::chrono::steady_clock::now();
if (Internal::handleGradientSpike(ctx, state, payload, preclip_grad_rms, preclip_grad_rms,
                                  result.loss, clip_selection, batch_idx)) {
    ctx.optimizer.current_micro_step = 0;
    ctx.model->getTrainingState().autograd_intermediates.clear();
    result.skipped = true;
    result.skip_reason = "gradient_spike";
    ctx.global_step++;
    return result;
}
auto spike_elapsed_ms = std::chrono::duration<float, std::milli>(...).count();
```

**After:**
```cpp
// Gradient spike handling removed (Rule 20: spikes indicate real bugs, not something to silently skip)
```

**Note:** `spike_elapsed_ms` variable was also removed here. The PERF log that referenced it was updated (see §3d).

**If something breaks:** This shouldn't break anything — the `handleGradientSpike` function body was removed in a prior session. If somehow the old function body still exists somewhere, search `grep -n "handleGradientSpike" Phase2_TrainingLoop.cu` — should return 0 results.

#### 3c. Entire telemetry control block removed
**Location:** `processBatch()`, between the clipping setup and the LR computation.

**Before (~60 lines):**
```cpp
// === TELEMETRY CONTROL (GPU-NATIVE) ===
auto telemetry_start = std::chrono::steady_clock::now();
static GRIM::Telemetry::ControlDecision cached_telemetry_decision = []() { ... }();
static int telemetry_counter = 0;
const bool telemetry_control_enabled = ctx.config.hyperparameters.telemetry_control_enabled;
const bool telemetry_control_active = false;  // was already hardcoded false
const bool should_eval_telemetry = telemetry_control_active && (telemetry_counter == 0);
telemetry_counter = (telemetry_counter + 1) % 10;
GRIM::Telemetry::ControlDecision telemetry_decision = cached_telemetry_decision;
if (!telemetry_control_active) {
    telemetry_decision = GRIM::Telemetry::ControlDecision{};
    telemetry_decision.grad_scale_factor = 1.0f;   // field no longer exists in struct
    telemetry_decision.action = GRIM::Telemetry::ControlAction::Continue;
    cached_telemetry_decision = telemetry_decision;
}
auto telemetry_elapsed_ms = ...;
const bool allow_telemetry_actions = ...;
GRIM::Telemetry::ControlAction telemetry_action = telemetry_decision.action;
if (!allow_telemetry_actions && ...) {
    telemetry_action = GRIM::Telemetry::ControlAction::Continue;
    telemetry_decision.cooldown_extension = 0;   // field no longer exists in struct
}
```

**After:**
```cpp
// Telemetry control interventions removed (Rule 20: monitoring-only, crash on real bugs)
```

**Variables that no longer exist** (cascading effects already fixed):
- `cached_telemetry_decision`
- `telemetry_counter`
- `telemetry_control_enabled`
- `telemetry_control_active`
- `should_eval_telemetry`
- `telemetry_decision`
- `telemetry_elapsed_ms`
- `allow_telemetry_actions`
- `telemetry_action`
- `telemetry_scale`
- `safe_scale_factor`

#### 3d. Telemetry switch block removed + clipping simplified
**Location:** `processBatch()`, between LR computation and optimizer step setup.

**Before (spike cooldown + ~90-line switch):**
```cpp
result.learning_rate = scheduled_lr;

// Spike cooldown handling
if (state.grad_spike_cooldown > 0 && !ctx.config.stability.enabled) {
    state.grad_spike_cooldown--;
    const float floor_lr = std::max(1e-8f, ctx.config.stability.lr_min);
    const float spike_cap_lr = std::max(floor_lr, scheduled_lr * kGradSpikeLrFraction);
    result.learning_rate = std::min(result.learning_rate, spike_cap_lr);
}

// === TELEMETRY CONTROL ACTIONS ===
switch (telemetry_action) {
    case ControlAction::SkipStep: ...return result;
    case ControlAction::ExtendCooldown: state.grad_spike_cooldown += ...; break;
    case ControlAction::TriggerSoftRestart: GRIM::SoftRestart::scaleOptimizerMoments(...); break;
    case ControlAction::InjectPlateauNoise: ...noise injection... break;
    default: break;
}
```

**After:**
```cpp
result.learning_rate = scheduled_lr;
```

**Clipping also simplified.** Before:
```cpp
const float telemetry_scale = telemetry_control_active ? telemetry_decision.grad_scale_factor : 1.0f;
const float safe_scale_factor = std::max(telemetry_scale, 0.01f);
const float effective_per_token_limit = clip_selection.per_token_limit * safe_scale_factor;
```
After:
```cpp
const float effective_per_token_limit = clip_selection.per_token_limit;
```
Clipping limit is now always exactly `clip_selection.per_token_limit` — no scaling applied.

**If something breaks:** If per-token gradient clipping limits are suddenly hitting earlier/more aggressively than expected, this is because `safe_scale_factor` was previously always 1.0 anyway (telemetry was inactive), so behavior should be identical. If `GRIM::SoftRestart` header include causes linker error, it may be unused now — check if it's included in Phase2_TrainingLoop.cu header section.

#### 3e. PERF log updated
**Location:** `processBatch()`, the `[PERF] Pre-optimizer setup took` log line.

**Before:**
```cpp
"ms (spike=" + formatScalar(spike_elapsed_ms, 2) + 
"ms, telemetry=" + formatScalar(telemetry_elapsed_ms, 2) + 
"ms, clipping=" + ...
```
**After:**
```cpp
"ms (clipping=" + formatScalar(clipping_elapsed_ms, 2) + 
"ms, lr=" + formatScalar(lr_elapsed_ms, 2) + 
"ms, sample=" + formatScalar(sample_elapsed_ms, 2) + "ms)"
```

#### 3f. `isBatchQuarantined` function body deleted
**Location:** `Internal` namespace, approximately line 665 in original file, just before `buildEpochBatches`.

**Before:**
```cpp
bool isBatchQuarantined(
    const std::vector<uint32_t>& seq_ids,
    const std::unordered_set<uint32_t>& quarantined_seqs) {
    for (uint32_t sid : seq_ids) {
        if (quarantined_seqs.count(sid)) return true;
    }
    return false;
}
```
**After:** Entirely deleted. `buildEpochBatches` now directly follows the previous function.

---

### 4. `resources/models/GRIM-text/training/Phases/Phase2_TrainingLoop.hpp`

#### 4a. Constants removed
**Before:**
```cpp
constexpr int kGradSpikeCooldownSteps = 2;
constexpr float kGradSpikeLrFraction = 0.10f;
```
**After:** Both lines deleted. Nothing in the file or .cu references them anymore.

#### 4b. `TrainingLoopState` fields removed
**Before (first 3 fields of struct):**
```cpp
struct TrainingLoopState {
    // Spike tracking
    std::unordered_map<uint32_t, int> spike_counts;
    std::unordered_set<uint32_t> quarantined_seqs;
    int grad_spike_cooldown = 0;
    
    // Loss baseline tracking (for adaptive spike detection)
    float initial_loss = 0.0f;
    ...
```
**After:**
```cpp
struct TrainingLoopState {
    // Loss baseline tracking
    float initial_loss = 0.0f;
    ...
```

**IMPORTANT:** If any existing checkpoint serialization or memory layout depends on `TrainingLoopState`'s field order, it shouldn't matter — `TrainingLoopState` is in-memory only (not serialized to disk). But if something like `memcpy`-based state transfer exists, field order changed.

#### 4c. `handleGradientSpike` declaration removed
**Before:**
```cpp
/**
 * @brief Handle gradient spike detection and quarantine
 */
bool handleGradientSpike(
    TrainingContext& ctx,
    TrainingLoopState& state,
    const GRIM::Batching::BatchPayload& payload,
    float preclip_grad_rms,
    float preclip_norm_grad,
    float batch_loss,
    const GRIM::TNC::ClipSelection& clip_selection,
    int batch_idx);
```
**After:** Entire block deleted.

**Note:** `isBatchQuarantined` declaration was already gone (removed in a prior edit of this session before the breadcrumb confirms it).

#### 4d. Includes removed
**Before:**
```cpp
#include <functional>
#include <unordered_map>
#include <unordered_set>
```
**After:**
```cpp
#include <functional>
```
`<unordered_map>` and `<unordered_set>` removed. If any other code in Phase2_TrainingLoop.cu uses `std::unordered_map` or `std::unordered_set` that doesn't bring the header in another way, there will be a compile error. Quick check: `grep -n "unordered_" Phase2_TrainingLoop.cu`.

---

### 5. `resources/models/GRIM-text/training/Phases/Phase1_Startup.cu`
**Location:** Section "11b. Initialize telemetry control", inside `loadConfiguration()` or `executePhase1()`.

**Before (~50 lines):**
- Populated `plateau_noise_*` fields (7 fields)
- Populated intervention fields: `moderate_grad_scale`, `moderate_cooldown_extension`, `seq_len_regime_change_threshold`, `regime_change_suppression_steps`, `volatility_damping_threshold`, `max_volatility_damping`, `gradient_decay_threshold`, `max_decay_boost`, `progress_boost_threshold`, `max_progress_boost`, `outlier_frequency_trigger`, `outlier_persistence_trigger`, `anchor_drift_sigma_multiplier`, `soft_restart_cooldown_steps` (14 intervention fields)
- Had an if/else branch: `if (telemetry_control_enabled)` → log ENABLED + plateau noise params / `else` → override thresholds to 1000.0f to prevent any interventions

**After (~15 lines):**
- Declares `const auto& hp = ctx->config.hyperparameters;`
- Populates only 3 diagnostic spike thresholds: `spike_mild_threshold`, `spike_moderate_threshold`, `spike_severe_threshold`
- Populates 3 accumulation bug detection fields: `min_grad_for_nonzero_loss`, `loss_threshold_for_grad_check`, `max_consecutive_zero_grad_steps`
- Populates 4 monitoring config fields: `warmup_steps`, `baseline_stabilization_steps`, `verbose_logging`, `fail_loud_on_accumulation_bug`
- Logs `"Telemetry control: MONITORING ONLY (all interventions removed)"`
- Removed the `telemetry_control_enabled` local variable entirely

**If something breaks:** If the code previously relied on the `else` branch to override spike thresholds to 1000.0f when disabled, that safeguard is now gone. But since the kernel only ever produces `Continue` (interventions removed from kernel too), the thresholds are purely diagnostic—they don't trigger anything.

---

## Variables / Symbols Confirmed Deleted

| Symbol | File | Notes |
|--------|------|-------|
| `spike_counts` | Phase2_TrainingLoop.hpp + .cu | `std::unordered_map<uint32_t, int>` in `TrainingLoopState` |
| `quarantined_seqs` | Phase2_TrainingLoop.hpp + .cu | `std::unordered_set<uint32_t>` in `TrainingLoopState` |
| `grad_spike_cooldown` | Phase2_TrainingLoop.hpp + .cu | `int` in `TrainingLoopState` |
| `kGradSpikeCooldownSteps` | Phase2_TrainingLoop.hpp | `constexpr int = 2` |
| `kGradSpikeLrFraction` | Phase2_TrainingLoop.hpp | `constexpr float = 0.10f` |
| `handleGradientSpike` | Phase2_TrainingLoop.hpp + .cu | function decl + body both gone |
| `isBatchQuarantined` | Phase2_TrainingLoop.hpp + .cu | function decl + body both gone |
| `cached_telemetry_decision` | Phase2_TrainingLoop.cu | static local in `processBatch()` |
| `telemetry_counter` | Phase2_TrainingLoop.cu | static local in `processBatch()` |
| `telemetry_control_active` | Phase2_TrainingLoop.cu | was hardcoded `false` |
| `telemetry_control_enabled` | Phase2_TrainingLoop.cu + Phase1_Startup.cu | local var reading `hp.telemetry_control_enabled` |
| `should_eval_telemetry` | Phase2_TrainingLoop.cu | was always `false` |
| `telemetry_decision` | Phase2_TrainingLoop.cu | `ControlDecision` local |
| `telemetry_elapsed_ms` | Phase2_TrainingLoop.cu | timing var, removed from PERF log |
| `allow_telemetry_actions` | Phase2_TrainingLoop.cu | was always `false` |
| `telemetry_action` | Phase2_TrainingLoop.cu | was always `Continue` |
| `telemetry_scale` | Phase2_TrainingLoop.cu | was always `1.0f` |
| `safe_scale_factor` | Phase2_TrainingLoop.cu | was always `1.0f` |
| `spike_elapsed_ms` | Phase2_TrainingLoop.cu | timing var from deleted spike block |
| `ControlAction::SkipStep` | TelemetryControl_GPU.hpp | enum value removed |
| `ControlAction::ExtendCooldown` | TelemetryControl_GPU.hpp | enum value removed |
| `ControlAction::TriggerSoftRestart` | TelemetryControl_GPU.hpp | enum value removed |
| `ControlAction::InjectPlateauNoise` | TelemetryControl_GPU.hpp | enum value removed |
| `ControlDecision::grad_scale_factor` | TelemetryControl_GPU.hpp | struct field removed |
| `ControlDecision::cooldown_extension` | TelemetryControl_GPU.hpp | struct field removed |
| `launchPlateauNoiseInjection` | TelemetryControl_GPU.hpp + .cu | decl + body both gone |

---

## What Was NOT Changed

- `ai_config_paths.hpp` — The `TrainingParams` struct still has all `telemetry_*` fields including `telemetry_plateau_noise_*`, `telemetry_moderate_grad_scale`, etc. These fields are still parsed from JSON. They're just never consumed now. No build error, just dead config fields.
- `ai_config.json` — Unchanged. JSON telemetry config keys still present.
- `TelemetryLattice` — Unchanged. Still streams 8-level hierarchical stats.
- `TelemetryCsvLogger` — Unchanged.
- `GRIM::SoftRestart::scaleOptimizerMoments` — Still exists, now has zero callers in Phase2. May produce a linker warning if the include is still present.
- The token validity SCAN logic in `processBatch()` — still fully intact. Only the response to a positive scan result changed (crash instead of skip).

---

## Quick Diagnostic Commands (if build breaks)

```bash
# Check for any remaining references to deleted fields
grep -n "grad_scale_factor\|cooldown_extension\|telemetry_action\|telemetry_decision\|cached_telemetry" \
  resources/models/GRIM-text/training/Phases/Phase2_TrainingLoop.cu

# Check for stale references to removed ControlAction values
grep -rn "SkipStep\|ExtendCooldown\|TriggerSoftRestart\|InjectPlateauNoise" \
  resources/models/GRIM-text --include="*.cu" --include="*.hpp"

# Check TrainingLoopState usages still compile
grep -n "state\.spike_counts\|state\.quarantined_seqs\|state\.grad_spike_cooldown" \
  resources/models/GRIM-text/training/Phases/Phase2_TrainingLoop.cu

# Check unordered includes are still pulled in if needed
grep -n "unordered_" resources/models/GRIM-text/training/Phases/Phase2_TrainingLoop.cu
```
