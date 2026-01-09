# GRIM-text Telemetry & TelemetryControl Architecture

**Document Version:** 1.0  
**Last Updated:** January 2026  
**Author:** GRIM Development Team  
**Audience:** Technical reviewers, professor evaluation

---

## Executive Summary

The GRIM-text Telemetry system provides **GPU-native hierarchical training monitoring** with two tightly integrated components:

1. **TelemetryLattice** - Multi-scale streaming statistics engine that tracks training metrics across 5 temporal resolutions
2. **TelemetryControl** - Adaptive training controller that uses lattice telemetry to make runtime decisions (scale gradients, skip steps, trigger soft restarts, inject plateau noise)

### Key Architectural Decisions

1. **Pure GPU Execution**: All telemetry computation happens on GPU—no CPU round-trips during training
2. **Single D2H Transfer**: TelemetryControl produces one 48-byte `ControlDecision` struct per batch
3. **Multi-Scale Aggregation**: 5 temporal levels with exponential strides (1, 2, 4, 8, 16 steps)
4. **5 Metric Streams**: Loss, gradient norm (mean), gradient norm (max), learning rate, tokens per batch
5. **10-Element Telemetry Vector**: Rich statistical representation per (level, stream) combination
6. **Fail-Loud Policy**: NaN/Inf detection with immediate termination in strict mode
7. **No Per-Call Allocations**: Rule 22 compliant—all buffers pre-allocated at initialization

---

## System Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        TELEMETRY SYSTEM ARCHITECTURE                        │
│                                                                             │
│  TRAINING LOOP                 TELEMETRYLATTICE              TELEMETRYCONTROL
│  ─────────────                 ────────────────              ────────────────
│                                                                             │
│  ┌──────────┐   observations   ┌─────────────────┐   lattice   ┌──────────┐│
│  │ Forward  │──────────────────▶│  Level 0 (1×)   │────────────▶│  Spike   ││
│  │ Backward │   [5 floats]     │  Level 1 (2×)   │             │ Detect   ││
│  │   Loss   │                  │  Level 2 (4×)   │             │ Regime   ││
│  │ GradNorm │                  │  Level 3 (8×)   │             │ Change   ││
│  └──────────┘                  │  Level 4 (16×)  │             │ Drift    ││
│       │                        └────────┬────────┘             └────┬─────┘│
│       │                                 │                           │      │
│       │                        ┌────────▼────────┐             ┌────▼─────┐│
│       │                        │ 5 Metric Streams│             │ Control  ││
│       │                        │ × 5 Levels      │             │ Decision ││
│       │                        │ = 25 States     │             │ (48 B)   ││
│       │                        └─────────────────┘             └────┬─────┘│
│       │                                                             │      │
│       └─────────────────────────────────────────────────────────────┘      │
│                                   grad_scale_factor                        │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 1. TelemetryLattice: Hierarchical Streaming Statistics

### 1.1 Core Concept

The TelemetryLattice maintains **multi-scale exponential moving averages** across 5 temporal levels. Each level tracks training dynamics at a different time horizon:

| Level | Stride | Update Frequency | Time Horizon |
|-------|--------|------------------|--------------|
| **0** | 1 | Every step | Immediate (1 batch) |
| **1** | 2 | Every 2 steps | Short-term (2 batches) |
| **2** | 4 | Every 4 steps | Medium-term (4 batches) |
| **3** | 8 | Every 8 steps | Long-term (8 batches) |
| **4** | 16 | Every 16 steps | Trend (16 batches) |

**Why Multi-Scale?** Single-scale statistics are either too noisy (level 0) or too slow to react (level 4). By tracking multiple scales simultaneously, TelemetryControl can:
- Detect sudden spikes (level 0 vs level 2)
- Identify persistent drift (level 2 vs level 4)
- Distinguish transient noise from regime changes

### 1.2 Metric Streams

Each level tracks 5 parallel metric streams:

```cpp
enum class MetricStream : int {
    LOSS = 0,              // Training loss
    GRAD_NORM_MEAN = 1,    // Mean gradient L2 norm
    GRAD_NORM_MAX = 2,     // Max gradient magnitude
    LEARNING_RATE = 3,     // Current LR
    TOKENS_PER_BATCH = 4   // Valid tokens (for normalization)
};
```

**Total States:** 5 levels × 5 streams = **25 TelemetryState structs** on GPU

### 1.3 TelemetryState Structure

Each (level, stream) pair maintains a 120-byte state struct:

```cpp
struct TelemetryState {
    // === Fast Magnitude (updated every stride) ===
    float mu;            // EMA mean: μ_t = β_μ·μ_{t-1} + (1-β_μ)·x_t
    float m2;            // EMA second moment: m2_t = β_μ·m2_{t-1} + (1-β_μ)·x_t²
    float sigma;         // Standard deviation: σ = √(m2 - μ²)
    float sigma_tilde;   // Coefficient of variation: σ̃ = σ / (|μ| + ε)
    
    // === Slow Anchor (for drift detection) ===
    float mu_a;          // Anchor mean (slow EMA: β_a = 0.995)
    float sigma_a;       // Anchor std
    float delta_mu;      // Drift: μ - μ_a
    float delta_sigma;   // Volatility drift: σ - σ_a
    
    // === Meta-Volatility ===
    float v_sigma;       // EMA of (σ - σ_prev)² - "volatility of volatility"
    float sigma_prev;    // Previous σ for derivative
    
    // === Trend Detection ===
    float delta_bar;     // EMA of normalized slope: Δ̄ = β_δ·Δ̄ + (1-β_δ)·Δ̂
    float p;             // Directional bias ∈ [-1, +1]
    float mu_prev;       // Previous μ for slope
    
    // === Outlier Statistics ===
    float r_out;         // Outlier frequency (soft sigmoid gating)
    float ell_out;       // Outlier persistence ("run length")
    float mu_ex;         // Excess magnitude above threshold
    float k_out;         // Adaptive threshold: k = k₀·(1 + α_v·√v_σ)
    float c_out;         // Cutoff value: μ + k_out·σ
    
    // === Metadata ===
    uint32_t step_count; // Total updates to this state
    uint32_t initialized; // 1 after first update
};
```

### 1.4 10-Element Telemetry Vector

TelemetryControl reads a compressed 40-byte vector from each state:

```cpp
struct TelemetryVector {
    float mu;           // [0] Current magnitude baseline
    float sigma_tilde;  // [1] Scale-normalized volatility
    float v_sigma;      // [2] Meta-volatility
    float delta_bar;    // [3] Directional trend strength
    float p;            // [4] Directional bias (up/down tendency)
    float r_out;        // [5] Outlier frequency
    float ell_out;      // [6] Outlier persistence
    float mu_ex;        // [7] Excess severity
    float delta_mu;     // [8] Mean drift vs anchor
    float delta_sigma;  // [9] Volatility drift vs anchor
};
```

### 1.5 Update Algorithm (Per-Step)

The `updateTelemetryStateKernel` executes 9 mathematical steps per update:

```
┌─────────────────────────────────────────────────────────────────┐
│                 TELEMETRY UPDATE ALGORITHM                      │
│                                                                 │
│  INPUT: x_t (single scalar observation)                         │
│                                                                 │
│  STEP 1: Fast Magnitude                                         │
│    μ ← β_μ·μ + (1-β_μ)·x_t                                     │
│    m2 ← β_μ·m2 + (1-β_μ)·x_t²                                  │
│    σ ← √max(m2 - μ², ε)                                        │
│    σ̃ ← σ / (|μ| + ε)                                           │
│                                                                 │
│  STEP 2: Volatility-of-Volatility                               │
│    v_σ ← β_v·v_σ + (1-β_v)·(σ - σ_prev)²                       │
│    σ_prev ← σ                                                   │
│                                                                 │
│  STEP 3: Adaptive Outlier Threshold                             │
│    k_out ← k₀·(1 + α_v·√v_σ)                                   │
│    c_out ← μ + k_out·σ                                         │
│                                                                 │
│  STEP 4: Normalized Slope + Direction                           │
│    Δ̂ ← (x_t - μ_prev) / (σ_prev + ε)                           │
│    Δ̄ ← β_δ·Δ̄ + (1-β_δ)·Δ̂                                     │
│    p ← β_δ·p + (1-β_δ)·sign(Δ̂)                                │
│    μ_prev ← μ                                                   │
│                                                                 │
│  STEP 5: Soft Outlier Gating (sigmoid)                          │
│    w_out ← σ((x_t - c_out) / (σ + ε))                          │
│    r_out ← β_r·r_out + (1-β_r)·w_out                           │
│                                                                 │
│  STEP 6: Excess Magnitude                                       │
│    η ← max(0, x_t - c_out)                                     │
│    μ_ex ← β_μ·μ_ex + (1-β_μ)·η                                 │
│                                                                 │
│  STEP 7: Persistence (Run Length)                               │
│    ℓ_out ← β_run·ℓ_out + (1-β_run)·w_out                       │
│                                                                 │
│  STEP 8: Slow Anchor Drift                                      │
│    μ_a ← β_a·μ_a + (1-β_a)·μ                                   │
│    σ_a ← β_a·σ_a + (1-β_a)·σ                                   │
│    δ_μ ← μ - μ_a                                               │
│    δ_σ ← σ - σ_a                                               │
│                                                                 │
│  STEP 9: Update Metadata                                        │
│    step_count++                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 1.6 Hyperparameters

```cpp
struct TelemetryHyperParams {
    // EMA decay rates (higher = slower adaptation)
    float beta_mu = 0.95f;      // Fast mean/variance
    float beta_a = 0.995f;      // Slow anchor (20× slower)
    float beta_delta = 0.90f;   // Slope EMA
    float beta_r = 0.85f;       // Outlier frequency
    float beta_run = 0.80f;     // Persistence EMA
    float beta_v = 0.90f;       // Volatility-of-volatility
    
    // Outlier detection
    float k_out0 = 2.5f;        // Base threshold (sigmas)
    float alpha_v = 1.5f;       // Volatility inflation factor
    
    // Numerical stability
    float epsilon = 1e-7f;      // Floor for divisions
    
    // Fail-loud mode
    bool strict_mode = true;    // Crash on NaN/Inf
};
```

### 1.7 Higher-Level Updates

Higher levels (k > 0) update less frequently and aggregate from level k-1:

```cpp
for (int level = 1; level < num_levels; ++level) {
    const uint32_t stride = 1u << level;  // 2^k
    
    if ((global_step % stride) != 0) {
        continue;  // Not time to update this level
    }
    
    // Use level (k-1) μ as input for level k
    const float x_k = levels[level - 1].state.mu;
    
    // Apply same 9-step update algorithm
    updateState(&levels[level].state, x_k, hp);
}
```

---

## 2. TelemetryControl: Adaptive Training Controller

### 2.1 Core Concept

TelemetryControl reads multi-scale telemetry from the lattice and produces control decisions that adaptively modify training behavior. It runs a **single-thread GPU kernel** that directly accesses GPU-resident telemetry.

### 2.2 Control Actions

```cpp
enum class ControlAction : int {
    Continue = 0,           // Normal operation
    ScaleGradients = 1,     // Apply grad_scale_factor
    ExtendCooldown = 2,     // Extend LR cooldown
    SkipStep = 3,           // Skip optimizer step entirely
    TriggerSoftRestart = 4, // Reset optimizer momentum
    InjectPlateauNoise = 5, // Add noise to escape local minimum
    FatalError = 6          // Unrecoverable state (accumulation bug)
};
```

### 2.3 ControlDecision Structure (48 bytes)

```cpp
struct alignas(16) ControlDecision {
    // Primary action
    ControlAction action = ControlAction::Continue;
    
    // Gradient scaling (always applied, default 1.0)
    float grad_scale_factor = 1.0f;
    
    // LR cooldown extension (0 = no extension)
    int cooldown_extension = 0;
    
    // Detected spike severity
    SpikeSeverity spike_severity = SpikeSeverity::None;
    
    // Bit flags for detected conditions
    uint32_t flags = 0;
    static constexpr uint32_t FLAG_ACCUMULATION_BUG   = 1 << 0;
    static constexpr uint32_t FLAG_REGIME_CHANGE      = 1 << 1;
    static constexpr uint32_t FLAG_OUTLIER_REGIME     = 1 << 2;
    static constexpr uint32_t FLAG_ANCHOR_DRIFT       = 1 << 3;
    static constexpr uint32_t FLAG_GRADIENT_DECAY     = 1 << 4;
    static constexpr uint32_t FLAG_RAPID_PROGRESS     = 1 << 5;
    static constexpr uint32_t FLAG_PLATEAU_NOISE      = 1 << 6;
    
    // Telemetry values for logging
    float spike_ratio;         // current / baseline
    float volatility_damping;  // Applied damping
    float decay_boost;         // Applied boost
    float progress_boost;      // Reward rapid improvement
    float normalized_grad;     // Token-normalized gradient
    
    // Error code (0 = OK)
    int error_code;
};
```

### 2.4 Detection Algorithms

#### Check 1: Accumulation Bug (FATAL)

Detects when gradients are zero but loss is non-zero—indicates broken backward pass:

```cpp
bool accumulationBug = (loss > 0.01f && grad_norm < 1e-10f);

if (accumulationBug) {
    consecutive_zero_grad_steps++;
    if (consecutive_zero_grad_steps >= 3) {
        return FatalError;  // CRASH - training is broken
    }
}
```

#### Check 2: Regime Change (Suppression)

Detects batch size/sequence length changes that cause temporary gradient magnitude shifts:

```cpp
float change = abs(current_tokens - baseline_tokens);
float threshold = 0.3f * baseline_tokens;  // 30% change

if (change > threshold) {
    flags |= FLAG_REGIME_CHANGE;
    suppression_steps = 2;  // Skip spike detection for 2 steps
}
```

**Why?** When batch composition changes (e.g., 93 tokens → 3114 tokens), gradient norms scale by √(tokens). Without suppression, this looks like a severe spike.

#### Check 3: Gradient Spike Detection

```cpp
SpikeSeverity computeSpikeSeverity(float current, float baseline) {
    float ratio = current / baseline;
    
    if (ratio >= 10.0f) return Severe;    // Skip optimizer step
    if (ratio >= 5.0f)  return Moderate;  // Scale gradients 50%
    if (ratio >= 3.0f)  return Mild;      // Log warning
    return None;
}
```

**Token Normalization:** Gradients are normalized before comparison:
```cpp
// Make grad_norm independent of batch size
normalized_grad = raw_grad_norm * sqrt(reference_tokens / valid_tokens);
```

#### Check 4: Outlier Regime Detection

Uses Level 0 loss stream's r_out and ℓ_out:

```cpp
bool outlierRegime = (r_out > 0.95f && ell_out > 0.90f);
// Triggers soft restart if persistent outliers
```

#### Check 5: Anchor Drift Detection

Uses Level 2 loss stream's delta_mu and sigma_tilde:

```cpp
bool anchorDrift = (delta_mu > 5.0f * sigma_tilde && delta_mu > 0);
// Only upward drift (loss increasing) is pathological
// Negative drift = loss decreasing = good!
```

#### Check 6: Gradient Decay Detection (DISABLED)

Originally detected vanishing gradients, but normal gradient decay during convergence triggered false positives. Now disabled:

```cpp
// DISABLED: gradient_decay_threshold = 0.0f
// Normal convergence shows gradient decay - not pathological
```

#### Check 7: Plateau Detection

Uses rolling loss variance from TelemetryLattice:

```cpp
float loss_variance = sigma_tilde * mu * sigma_tilde * mu;

if (loss_variance < 0.001f && mu > 0.1f) {
    consecutive_low_variance_batches++;
    
    if (consecutive_low_variance_batches >= 50 &&
        batches_since_noise >= 500 &&
        noise_injections_this_epoch < 3) {
        return InjectPlateauNoise;
    }
}
```

### 2.5 Scaling Factor Computation (MOSTLY DISABLED)

The original design computed adaptive scaling factors, but empirical testing showed they caused training instability:

```cpp
// DISABLED: These features prevented convergence
// - Volatility damping: threshold = 100.0f (unreachable)
// - Decay boost: threshold = 0.0f (disabled)
// - Progress boost: threshold = 100.0f (unreachable)

// Active scaling only:
grad_scale_factor = 1.0f;  // Neutral by default

// Moderate spike: scale by 0.5
if (spike_severity == Moderate) {
    grad_scale_factor *= 0.5f;
}

// Floor at 0.01 to prevent gradient death
grad_scale_factor = max(grad_scale_factor, 0.01f);
```

### 2.6 Action Priority

```cpp
// Priority: Severe > Moderate > PlateauNoise > Outlier/Drift > Scale > Continue

if (spike_severity == Severe) {
    return SkipStep;
}
else if (spike_severity == Moderate) {
    return ScaleGradients;
}
else if (plateau_noise_triggered && !cooldown_active) {
    return InjectPlateauNoise;
}
else if ((outlier_regime || anchor_drift) && !cooldown_active) {
    return TriggerSoftRestart;
}
else if (grad_scale_factor != 1.0f) {
    return ScaleGradients;
}
else {
    return Continue;
}
```

---

## 3. Configuration Reference

### 3.1 TelemetryControlConfig

```cpp
struct TelemetryControlConfig {
    // Reference values for normalization
    float reference_seq_len = 512.0f;
    float reference_tokens = 720.0f;
    
    // Spike detection thresholds
    float spike_mild_threshold = 3.0f;      // > 3× = mild
    float spike_moderate_threshold = 5.0f;  // > 5× = moderate
    float spike_severe_threshold = 10.0f;   // > 10× = severe
    
    // Spike response
    float moderate_grad_scale = 0.5f;
    int moderate_cooldown_extension = 3;
    
    // Accumulation bug detection
    float min_grad_for_nonzero_loss = 1e-10f;
    float loss_threshold_for_grad_check = 0.01f;
    int max_consecutive_zero_grad_steps = 3;  // FATAL after this
    
    // Regime change suppression
    float seq_len_regime_change_threshold = 0.3f;  // 30% change
    int regime_change_suppression_steps = 2;
    
    // Outlier/drift triggers (RAISED to reduce false positives)
    float outlier_frequency_trigger = 0.95f;
    float outlier_persistence_trigger = 0.90f;
    float anchor_drift_sigma_multiplier = 5.0f;  // 5-sigma events
    
    // Soft restart cooldown
    int soft_restart_cooldown_steps = 10;
    
    // Warmup (skip detection during unstable early training)
    int warmup_steps = 100;
    int baseline_stabilization_steps = 50;
    
    // Plateau noise injection
    bool plateau_noise_enabled = true;
    int plateau_noise_patience = 50;
    float plateau_noise_variance_threshold = 0.001f;
    float plateau_noise_std = 0.001f;
    bool plateau_noise_proportional = true;
    int plateau_noise_cooldown = 500;
    int plateau_noise_max_per_epoch = 3;
};
```

### 3.2 LatticeConfig

```cpp
struct LatticeConfig {
    int num_levels = 5;           // k ∈ [0, 4]
    int num_streams = 5;          // LOSS, GRAD_MEAN, GRAD_MAX, LR, TOKENS
    TelemetryHyperParams hyperparams;
    cudaStream_t stream = nullptr;  // MUST be non-default (Rule 22)
};
```

---

## 4. GPU Memory Layout

### 4.1 TelemetryLattice Memory

| Buffer | Size | Purpose |
|--------|------|---------|
| `levels` | 25 × 128B = 3.2KB | LatticeLevelState array [5 levels × 5 streams] |
| `d_observations` | 5 × 4B = 20B | Input observations per update |
| `d_scratch_vectors` | 5 × 40B = 200B | Temp space for vector extraction |
| `d_error_flag` | 4B | Atomic error flag |
| **Total** | ~3.5KB | GPU-resident |

### 4.2 TelemetryControl Memory

| Buffer | Size | Purpose |
|--------|------|---------|
| `d_config_` | ~200B | Config mirror on GPU |
| `d_state_` | 64B | Persistent control state |
| `d_decision_` | 48B | Output decision buffer |
| `d_input_` | 32B | Input parameters |
| **Total** | ~350B | GPU-resident |

### 4.3 Transfer Sizes Per Update

| Direction | Size | Operation |
|-----------|------|-----------|
| H→D | 20B | 5 observations to lattice |
| D→H | 0B | Lattice update (pure GPU) |
| H→D | 32B | Control input |
| D→H | 48B | Control decision |
| **Total** | 100B/step | Minimal CPU-GPU transfer |

---

## 5. API Reference

### 5.1 TelemetryLattice API

```cpp
// Initialize lattice (allocates GPU memory)
TelemetryLattice* initTelemetryLattice(const LatticeConfig& config);

// Update with new observations (GPU kernel)
TelemetryError updateTelemetryLattice(
    TelemetryLattice* lattice,
    const float* observations,   // [5] Host array
    uint32_t global_step
);

// Read telemetry vector (GPU→CPU copy)
TelemetryError readTelemetryVector(
    const TelemetryLattice* lattice,
    int level,
    int stream_idx,
    TelemetryVector* out_vector
);

// Batched read (66× faster - single sync instead of 3)
TelemetryError readTelemetryBatched(
    const TelemetryLattice* lattice,
    int level0, int stream_idx0, TelemetryVector* out0,
    int level1, int stream_idx1, TelemetryVector* out1,
    int level2, int stream_idx2, TelemetryVector* out2
);

// Reset anchors after soft restart
TelemetryError resetTelemetryAnchors(TelemetryLattice* lattice, cudaStream_t stream);

// Cleanup
void freeTelemetryLattice(TelemetryLattice** lattice);
```

### 5.2 TelemetryControl API

```cpp
class TelemetryControl {
public:
    explicit TelemetryControl(const TelemetryControlConfig& config = {});
    ~TelemetryControl();
    
    // Initialize GPU resources (call once)
    void initGPU();
    
    // Evaluate and produce control decision
    ControlDecision evaluate(
        const TelemetryLattice* lattice,
        float raw_grad_norm,
        float loss,
        int valid_tokens,
        float avg_seq_len,
        uint32_t global_step,
        cudaStream_t stream  // NOT stored (Rule 22)
    );
    
    // Reset state at epoch boundary
    void reset();
    
    // Describe decision for logging
    std::string describeDecision(const ControlDecision& d) const;
};
```

---

## 6. Integration Example

```cpp
// === INITIALIZATION (Phase 1 Startup) ===

LatticeConfig lattice_cfg;
lattice_cfg.num_levels = 5;
lattice_cfg.num_streams = 5;
lattice_cfg.stream = primary_stream;
TelemetryLattice* lattice = initTelemetryLattice(lattice_cfg);

TelemetryControlConfig ctrl_cfg;
ctrl_cfg.reference_tokens = 720.0f;
ctrl_cfg.spike_severe_threshold = 10.0f;
TelemetryControl controller(ctrl_cfg);
controller.initGPU();

// === TRAINING LOOP (Phase 2) ===

for (uint32_t step = 0; step < max_steps; ++step) {
    // Forward + backward pass
    float loss = computeLoss(...);
    computeBackward(...);
    float raw_grad_norm = computeGradNorm(...);
    
    // Update telemetry
    float observations[5] = {
        loss,                    // LOSS
        raw_grad_norm,           // GRAD_NORM_MEAN
        max_grad_component,      // GRAD_NORM_MAX
        current_lr,              // LEARNING_RATE
        (float)valid_tokens      // TOKENS_PER_BATCH
    };
    updateTelemetryLattice(lattice, observations, step);
    
    // Get control decision
    ControlDecision decision = controller.evaluate(
        lattice, raw_grad_norm, loss, valid_tokens,
        avg_seq_len, step, primary_stream
    );
    
    // Apply decision
    if (decision.action == ControlAction::SkipStep) {
        LOG_WARNING("Skipping step " << step << " due to severe spike");
        continue;  // Skip optimizer
    }
    
    if (decision.action == ControlAction::TriggerSoftRestart) {
        resetOptimizerMomentum(...);
        resetTelemetryAnchors(lattice, primary_stream);
    }
    
    // Scale gradients
    scaleGradients(decision.grad_scale_factor);
    
    // Optimizer step
    adamWUpdate(...);
}

// === CLEANUP (Phase 3) ===

freeTelemetryLattice(&lattice);
```

---

## 7. Plateau Noise Injection

### 7.1 Purpose

When training loss variance approaches zero for many consecutive batches, the model may be stuck in a local minimum. Plateau noise injects small Gaussian perturbations to escape.

### 7.2 Kernel Implementation

```cpp
__global__ void plateauNoiseKernel(
    float* weights,
    size_t num_elements,
    float noise_std,
    bool proportional,
    unsigned long long seed
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_elements) return;
    
    // High-quality RNG (Philox)
    curandStatePhilox4_32_10_t rng;
    curand_init(seed, idx, 0, &rng);
    
    float noise = curand_normal(&rng) * noise_std;
    
    if (proportional) {
        noise *= max(abs(weights[idx]), 1e-6f);
    }
    
    weights[idx] += noise;
}
```

### 7.3 Safety Limits

- **Patience:** 50 consecutive low-variance batches before injection
- **Cooldown:** 500 batches between injections
- **Per-Epoch Limit:** Maximum 3 injections per epoch
- **Variance Threshold:** Only inject if loss variance < 0.001

---

## 8. File Reference

| File | Lines | Purpose |
|------|-------|---------|
| `TelemetryState_GPU.hpp` | 120 | State structs, hyperparams, error codes |
| `TelemetryLattice_GPU.hpp` | 180 | Lattice API declarations |
| `TelemetryLattice_Internal.hpp` | 45 | Internal struct for kernel access |
| `TelemetryLattice_GPU.cu` | 660 | Lattice kernels and host API |
| `TelemetryControl_GPU.hpp` | 346 | Control config, decision struct, class |
| `TelemetryControl_GPU.cu` | 725 | Control kernel and class implementation |

---

## Appendix A: Mathematical Notation

| Symbol | Meaning |
|--------|---------|
| μ | Exponential moving average of input |
| σ | Standard deviation (√(m2 - μ²)) |
| σ̃ | Coefficient of variation (σ/μ) |
| v_σ | Volatility-of-volatility (EMA of (Δσ)²) |
| Δ̄ | Normalized slope EMA |
| p | Directional bias ∈ [-1, +1] |
| r_out | Soft outlier frequency |
| ℓ_out | Outlier persistence (run length) |
| μ_ex | Excess magnitude above threshold |
| δ_μ | Drift from anchor mean |
| β_* | EMA decay parameters |
| k_out | Adaptive outlier threshold |

---

## Appendix B: Disabled Features and Rationale

Several features were disabled after empirical testing showed they harmed training:

| Feature | Original Purpose | Why Disabled |
|---------|------------------|--------------|
| Volatility Damping | Reduce gradients during high variance | Damped 35% of gradients, prevented learning |
| Decay Boost | Amplify weak gradients | False positives during normal convergence |
| Progress Boost | Accelerate during rapid improvement | Caused feedback loop: dip→boost→overshoot |
| Gradient Decay Detection | Detect vanishing gradients | Normal convergence looks like decay |

**Lesson:** Conservative defaults are safer. It's better to miss some pathologies than to falsely intervene and corrupt training.

---

*Document generated for GRIM-text v2.0 telemetry system.*
