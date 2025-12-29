# Telemetry-Driven Adaptive Control Integration

**Status:** Sensors working, actuators missing  
**Goal:** Wire TelemetryLattice signals into training control loops

---

## ✅ Phase 0: Infrastructure (COMPLETE)
- [x] TelemetryLattice correctly computes hierarchical statistics
- [x] Fix per-call cudaMalloc/cudaFree bug (removed blocking allocations)
- [x] Pre-allocate d_error_flag and scratch buffers once at init
- [x] Verify multi-scale cascade math (EMA propagation confirmed correct)

---

## 🔧 Phase 1: Dynamic Learning Rate Controller

**File:** `resources/models/GRIM-text/Shared/Dynamic_LR/DynamicLR.hpp`  
**Location:** `DynamicLRController::update()` method

### Tasks:
- [ ] Add `TelemetryVector` parameter to `update()` signature
  - Currently: `float update(float grad_norm, float loss)`
  - New: `float update(float grad_norm, float loss, const TelemetryVector* telemetry = nullptr)`

- [ ] **Adaptive Smoothing** using gradient volatility
  - Read `vec0_grad.v_sigma` (volatility-of-volatility)
  - Formula: `smoothing = smoothing_min + (smoothing_max - smoothing_min) * clamp(v_sigma / variance_reference, 0, 1)`
  - High volatility → more smoothing (stabilize LR)
  
- [ ] **Trend-Aware Adjustment** using medium-term baseline
  - Read `vec2_loss.delta_bar` (normalized slope over 4-step window)
  - If `delta_bar < -0.3` (strong downward trend), allow more aggressive LR increases
  - If `delta_bar > 0.2` (upward drift), apply conservative LR policy

- [ ] **Anomaly Detection** using outlier frequency
  - Read `vec0_grad.r_out` (gradient outlier gate)
  - If `r_out > 0.6`, extend cooldown period (anomalous gradients detected)
  - Log: `[DynamicLR] Anomaly detected: r_out={r_out}, cooldown extended`

### Integration Point:
`resources/models/GRIM-text/training/Phases/Phase2_TrainingLoop.cu` ~line 1480-1500

---

## 🔄 Phase 2: Soft Restart Controller

**File:** `resources/models/GRIM-text/training/Phases/Phase2_TrainingLoop.cu`  
**Location:** ~line 1770 (after validation)

### Tasks:
- [ ] **Regime Detection** using outlier persistence
  - Read `vec0_loss.r_out` (outlier frequency)
  - Read `vec0_loss.ell_out` (outlier persistence)
  - Trigger condition: `r_out > 0.7 && ell_out > 0.6` (persistent anomaly regime)
  - Current: only checks `val_loss > threshold`

- [ ] **Anchor Drift Detection** using medium-term baseline
  - Read `vec2_loss.delta_mu` (mean drift from slow anchor)
  - Trigger if `|delta_mu| > 2.0 * vec2_loss.sigma_tilde` (baseline has shifted)
  - Log: `[SoftRestart] Anchor drift detected: δμ={delta_mu}`

- [ ] **Combined Logic**
  ```cpp
  const bool loss_anomaly = (vec0_loss.r_out > 0.7f && vec0_loss.ell_out > 0.6f);
  const bool anchor_drift = (std::abs(vec2_loss.delta_mu) > 2.0f * vec2_loss.sigma_tilde);
  const bool should_restart = loss_anomaly || anchor_drift || (val_loss > threshold);
  ```

### Integration Point:
Replace existing `shouldTrigger()` call with telemetry-aware logic

---

## ✂️ Phase 3: Gradient Clipping (Adaptive Bounds)

**File:** `resources/models/GRIM-text/training/Phases/Phase2_TrainingLoop.cu`  
**Location:** ~line 1400-1450 (after backward pass, before optimizer step)

### Tasks:
- [ ] **Volatility-Scaled Clip Bound**
  - Read `vec0_grad.v_sigma` (gradient volatility)
  - Current: fixed `clip_value = base_clip`
  - New: `clip_value = base_clip * (1.0f + alpha * sqrt(v_sigma))` where `alpha = 0.5`
  - Rationale: Expand headroom during volatile periods to avoid over-clipping

- [ ] **Excess Magnitude Trimming**
  - Read `vec0_grad.mu_ex` (excess severity beyond adaptive threshold)
  - If `mu_ex > base_clip * 0.5`, apply secondary soft clip:
    ```cpp
    const float excess_scale = 1.0f / (1.0f + mu_ex / base_clip);
    adjusted_clip = clip_value * excess_scale;
    ```

- [ ] **Logging**
  - Log when clip bound adjusts: `[GradClip] base={base_clip} v_σ={v_sigma} adjusted={adjusted_clip}`

### Integration Point:
Before `ctx.optimizer.grad_controller->clipGradients()` call

---

## 📈 Phase 4: Gradient Scaling (NEW)

**File:** `resources/models/GRIM-text/training/Phases/Phase2_TrainingLoop.cu`  
**Location:** ~line 1340 (before backward pass, where `grad_scale` is computed)

### Tasks:
- [ ] **Baseline-Aware Scaling**
  - Read `vec2_grad.mu` (medium-term gradient baseline over 4 steps)
  - Read `vec0_grad.mu` (current gradient magnitude)
  - Compute ratio: `gradient_health = vec0_grad.mu / (vec2_grad.mu + epsilon)`
  - If `gradient_health < 0.5` (gradients decaying), apply boost:
    ```cpp
    const float boost = std::min(1.5f, 1.0f / gradient_health);
    grad_scale *= boost;
    ```

- [ ] **Volatility Damping**
  - Read `vec0_grad.v_sigma`
  - If `v_sigma > threshold` (e.g., 0.3), apply damping:
    ```cpp
    const float damp = 1.0f / (1.0f + v_sigma);
    grad_scale *= damp;
    ```
  - Rationale: Reduce effective learning when gradients are noisy

- [ ] **Combined Formula**
  ```cpp
  // Base: loss averaging scale
  float grad_scale = 1.0f / valid_tokens;
  
  // Telemetry adjustments
  if (ctx.telemetry.enabled && ctx.telemetry.lattice) {
      TelemetryVector vec0_grad, vec2_grad;
      readTelemetryVector(ctx.telemetry.lattice, 0, (int)MetricStream::GRAD_NORM_MEAN, &vec0_grad);
      readTelemetryVector(ctx.telemetry.lattice, 2, (int)MetricStream::GRAD_NORM_MEAN, &vec2_grad);
      
      const float gradient_health = vec0_grad.mu / (vec2_grad.mu + 1e-6f);
      const float boost = (gradient_health < 0.5f) ? std::min(1.5f, 1.0f / gradient_health) : 1.0f;
      const float damp = (vec0_grad.v_sigma > 0.3f) ? (1.0f / (1.0f + vec0_grad.v_sigma)) : 1.0f;
      
      grad_scale = grad_scale * boost * damp;
      
      ctx.logging.logger->log("[GradScale] health=" + formatScalar(gradient_health) + 
                              " boost=" + formatScalar(boost) + 
                              " damp=" + formatScalar(damp) + 
                              " final=" + formatScalar(grad_scale, 8));
  }
  ```

### Integration Point:
Replace/augment existing `grad_scale` computation before `ctx.model->backward()` call

---

## 🎯 Phase 5: Batch Construction (Curriculum Strength)

**File:** `resources/models/GRIM-text/training/Phases/Phase2_TrainingLoop.cu`  
**Location:** Batch selection logic (~line 800-1000)

### Tasks:
- [ ] **Drift-Aware Curriculum**
  - Read `vec2_loss.delta_mu` (baseline drift)
  - If `|delta_mu| > threshold`, increase rare token boost strength
  - Formula: `curriculum_strength = base_strength * (1.0f + 0.5f * |delta_mu|)`
  - Rationale: Drifting baseline needs harder examples to stabilize

- [ ] **Diversity Modulation**
  - Read `vec0_loss.sigma_tilde` (normalized volatility)
  - If `sigma_tilde < 0.1` (low diversity), increase batch variety
  - Adjust `content_weight_beta` based on volatility

### Integration Point:
Where `RareTokens::scoreSequences()` and content weighting is applied

---

## 🧪 Phase 6: Testing & Validation

- [ ] Create test case: Train with telemetry-driven control vs baseline
- [ ] Log all telemetry-triggered decisions to separate file
- [ ] Verify no NaN/Inf introduced by telemetry scaling factors
- [ ] Check memory usage stable (no leaks from telemetry reads)
- [ ] Benchmark: Ensure telemetry overhead < 5% of step time

---

## ⚡ Phase 6.5: Gradient Spike Detection & Fair Comparison

**File:** `resources/models/GRIM-text/training/Phases/Phase2_TrainingLoop.cu`  
**Location:** After backward pass, before telemetry update

### Gradient Spike Detection

- [ ] **Multi-timescale spike detection**
  - Compare `vec0_grad.mu` against `vec2_grad.mu` (baseline over 4 steps)
  - Spike condition: `vec0_grad.mu > spike_threshold * vec2_grad.mu`
  - Default `spike_threshold = 5.0` (tunable)
  - Log: `[GradSpike] DETECTED: current={mu0} baseline={mu2} ratio={ratio}`

- [ ] **Spike severity classification**
  ```cpp
  enum class SpikeLevel { None, Mild, Moderate, Severe };
  const float ratio = vec0_grad.mu / (vec2_grad.mu + 1e-6f);
  SpikeLevel level = (ratio < 3.0f) ? SpikeLevel::None :
                     (ratio < 5.0f) ? SpikeLevel::Mild :
                     (ratio < 10.0f) ? SpikeLevel::Moderate : SpikeLevel::Severe;
  ```

- [ ] **Response actions by severity**
  - `Mild`: Log only, no action
  - `Moderate`: Reduce grad_scale by 50%, extend cooldown
  - `Severe`: Skip optimizer step entirely, log FATAL-level warning

### Fair Comparison Across Batches (Normalization)

- [ ] **Token-normalized gradient metric**
  - Current problem: Longer sequences have larger accumulated gradients
  - Solution: Track `grad_norm / sqrt(valid_tokens)` as the canonical metric
  - Feed normalized metric to telemetry instead of raw grad_norm:
    ```cpp
    const float normalized_grad = grad_norm_sum / sqrtf(static_cast<float>(valid_tokens));
    observations[1] = normalized_grad;  // Stream 1: GRAD_NORM_MEAN
    ```

- [ ] **Batch-size invariant loss**
  - Verify loss is already averaged by `valid_tokens` (not summed)
  - Check: `loss_mean = loss_sum / valid_tokens` (confirmed in UnifiedLoss)

- [ ] **Sequence-length adjusted thresholds**
  - Store `avg_seq_len` in telemetry context
  - Adjust spike thresholds: `effective_threshold = base_threshold * sqrt(avg_seq_len / reference_seq_len)`
  - Reference: `reference_seq_len = 512` (typical training length)

### Accumulation Bug Detection

- [ ] **Gradient accumulation sanity check**
  - After `grad_controller->endBackward()`, verify gradient buffer non-zero
  - Check: `grad_l2_norm > epsilon` (if zero, accumulation failed)
  - Detect: If `grad_norm == 0` but `loss > 0`, flag accumulation bug:
    ```cpp
    if (result.grad_norm < 1e-10f && result.loss > 0.01f) {
        ctx.logging.logger->log("[FATAL] Accumulation bug: zero gradients with non-zero loss!");
        throw std::runtime_error("Gradient accumulation failure detected");
    }
    ```

- [ ] **Multi-step accumulation tracking**
  - If `gradient_accumulation_steps > 1`, track per-microbatch grad norms
  - Detect: Gradients not accumulating (first microbatch == final norm)
  - Detect: Gradients exploding during accumulation (final >> expected sum)

- [ ] **Weight-gradient correlation check**
  - Compute `dot(weights, gradients) / (||weights|| * ||gradients||)`
  - If consistently near -1 or +1, indicates degenerate optimization
  - Log warning: `[AccumCheck] weight-grad correlation={corr} (expected ~0)`

### Preventing False Alarms on Long Sequences

- [ ] **Length-aware telemetry normalization**
  - Problem: 2048-token sequence has 4x gradient magnitude of 512-token
  - Solution: Normalize by `sqrt(seq_len)` before feeding to telemetry
  - Already tracked: `ctx.telemetry.observations[4] = tokens_per_batch`

- [ ] **Dynamic threshold scaling**
  - Read `vec0_tokens.mu` (EMA of batch token count)
  - Scale all thresholds by `sqrt(vec0_tokens.mu / reference_tokens)`
  - Prevents: False spikes when transitioning from short to long sequences

- [ ] **Sequence length change detection**
  - Track `seq_len_change = abs(current_avg_len - vec2_tokens.mu)`
  - If `seq_len_change > 0.3 * vec2_tokens.mu`, suppress spike detection for 1 step
  - Log: `[SeqLen] Regime change detected: {old_len} -> {new_len}, suppressing alarms`

- [ ] **Warmup period for new sequence lengths**
  - After dataset epoch boundary, reset telemetry baselines
  - Or: Use `vec4_*` (16-step baseline) for spike detection during transitions

## 📊 Phase 7: Monitoring & Logging

- [ ] Add `[Telemetry]` log section showing active control decisions
- [ ] WebSocket broadcast of telemetry vectors for real-time dashboard
- [ ] Export telemetry time-series to training_status.fb for post-analysis
- [ ] Create Python script to visualize telemetry influence on LR/clipping

---

## 🎓 Phase 8: Documentation

- [ ] Update `copilot-instructions.md` with telemetry integration patterns
- [ ] Add docstring examples showing how to query telemetry in training loop
- [ ] Document tunable thresholds (v_sigma sensitivity, r_out trigger points)
- [ ] Write "Telemetry-Driven Training" section in README.md

---

## Priority Order:
1. **Gradient Scaling** (Phase 4) - Highest impact on training stability
2. **Dynamic LR** (Phase 1) - Second highest, affects convergence speed
3. **Gradient Clipping** (Phase 3) - Safety mechanism, pairs with scaling
4. **Soft Restart** (Phase 2) - Recovery mechanism for stuck states
5. **Batch Construction** (Phase 5) - Curriculum refinement
6. Testing & Docs (Phases 6-8)

---

## Notes:
- All telemetry reads should check `ctx.telemetry.enabled` before accessing lattice
- Use `readTelemetryVector()` not `readTelemetryState()` (avoid exposing internal state)
- Telemetry thresholds should be config-driven (add to `ai_config.json` → `training.telemetry_control`)
- **FAIL LOUD:** If telemetry read returns error, log and fall back to non-telemetry behavior (don't crash)
