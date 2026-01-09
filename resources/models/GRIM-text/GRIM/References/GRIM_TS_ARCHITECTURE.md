# GRIM-TS (Guess-Reward Integrated Memory - Training System) Architecture

## 1. Overview

**GRIM-TS** is a production-ready GPU-resident cache system for **speculative decoding** with **reinforcement learning integration**. It tracks generated guesses (hypotheses), their reward signals, and learns to predict which guess patterns are valuable during training.

### Design Philosophy

| Principle | Implementation |
|-----------|----------------|
| **GPU-Resident** | All cache data lives on GPU; no per-operation D2H transfers |
| **Rule 22 Compliant** | `TrainingState` owns all GPU memory; GRIM-TS uses pointers only |
| **Fail-Loud** | All null pointer / invalid state conditions trigger immediate `fprintf(stderr)` + return |
| **Warp-Cooperative** | Hash table operations use warp-level primitives (`__ballot_sync`, `__shfl_sync`) |
| **Zero CPU Allocation** | No `cudaMalloc` inside GRIM-TS; receives pre-allocated buffers |

### Key Capabilities

- **Adaptive Capacity**: Automatic grow/shrink recommendations based on fill ratio
- **Multi-Factor Eviction**: Reward variance, update frequency, diversity, recency scoring
- **Diversity Tracking**: Bloom filter penalizes repetitive guesses
- **Confidence Calibration**: Dynamic offset adjusts model confidence based on accuracy
- **Cache Warming**: High-quality sequence injection for bootstrap
- **Rich Telemetry**: Histograms, trend analysis, health indicators
- **Async Transfers**: Pinned memory pools for overlap with compute
- **GPU Delegate System**: Device-side callbacks for cache events

---

## 2. File Organization

| File | Lines | Purpose |
|------|-------|---------|
| [GRIM-TS.hpp](../../Layers/GRIMTS/GRIM-TS.hpp) | 463 | Public API, data structures, delegates, logging |
| [GRIM-TS.cu](../../Layers/GRIMTS/GRIM-TS.cu) | 1859 | GPU kernels, hash table operations, state management |

**Integration Points:**
- `TrainingState_GPU.hpp` → `GuessCacheBuffers` struct for centralized memory ownership
- `Phase2_TrainingLoop.cu/hpp` → `GuessCacheScope`, `GuessCacheBatchBuffers`, `MicroValidationScope`
- `ai_config.json` → `training.config.guess_cache` configuration section

---

## 3. Core Data Structures

### 3.1 GuessMetadata (24 bytes)

```cpp
struct GuessMetadata {
    uint64_t prompt_hash;      // Hash of prompt/context signature
    uint64_t guess_hash;       // Hash of generated guess text
    float    confidence;       // Model's confidence score [0.0, 1.0]
    uint16_t sequence_length;  // Token count of guess
    uint16_t prompt_length;    // Token count of prompt context
    uint32_t epoch;            // Training epoch when cached
};
```

**Key Composition**: `key = prompt_hash ⊕ (guess_hash × kMixPrime)` creates unique slot keys.

### 3.2 GuessRewardStats (76 bytes)

```cpp
struct GuessRewardStats {
    uint32_t total_attempts;     // How many times this guess was evaluated
    uint32_t positives;          // Positive reward count
    uint32_t negatives;          // Negative reward count
    float    cumulative_reward;  // Sum of all rewards
    float    ema_reward;         // Exponential moving average reward
    float    last_reward;        // Most recent reward
    float    best_reward;        // Highest reward ever seen
    float    worst_reward;       // Lowest reward ever seen
    float    last_updated_ts;    // GPU clock timestamp of last update
    float    created_ts;         // When entry was first created
    float    reward_mean;        // Running mean (Welford's algorithm)
    float    reward_m2;          // Running variance numerator (Welford's)
    float    normalized_last;    // Z-score of last reward
    uint32_t update_streak;      // Consecutive updates (frequency tracking)
    float    diversity_score;    // 1.0 = unique, lower = repetitive
    uint8_t  eviction_priority;  // 0 = protected, 255 = evict first
};
```

### 3.3 GuessRecord (100 bytes)

```cpp
struct GuessRecord {
    GuessMetadata   metadata;   // 24 bytes
    GuessRewardStats stats;     // 76 bytes
};
```

**Total Cache Memory**: `capacity × 100 bytes + capacity × 8 bytes (keys) + bloom filter + overhead`

### 3.4 GuessCacheDeviceState (Device-Side Access)

```cpp
struct GuessCacheDeviceState {
    GuessRecord* records;           // Main record array [capacity]
    unsigned int* size;             // Current entry count
    uint64_t* keys;                 // Hash keys for slot lookup
    unsigned int* evict_cursor;     // Round-robin eviction position
    uint32_t* diversity_bloom;      // Bloom filter for duplicate detection
    float* calibration_offset;      // Dynamic confidence calibration
    size_t capacity;                // Maximum entries
    size_t bloom_size;              // Bloom filter word count
};
```

Stored in device constant memory (`__device__ GuessCacheDeviceState d_cache_state`) for fast kernel access.

---

## 4. Hash Table Implementation

### 4.1 Key Space

| Value | Meaning |
|-------|---------|
| `0xFFFFFFFFFFFFFFFF` | Empty slot (`kEmptyKey`) |
| Any other value | Valid entry |

**Key Mixing**: Uses Fibonacci hashing prime `0x9E3779B185EBCA87` for good distribution.

### 4.2 Warp-Cooperative Slot Finding

```
┌─────────────────────────────────────────────────────────┐
│ Thread 0 checks slot (hash + 0) % capacity              │
│ Thread 1 checks slot (hash + 1) % capacity              │
│ ...                                                     │
│ Thread 31 checks slot (hash + 31) % capacity            │
├─────────────────────────────────────────────────────────┤
│ __ballot_sync() → collect match results                 │
│ __shfl_sync() → broadcast winning slot to all threads   │
└─────────────────────────────────────────────────────────┘
```

**Lookup Steps:**
1. **Match Check**: `__ballot_sync(mask, stored == key)` finds existing entries
2. **Empty Check**: `__ballot_sync(mask, stored == kEmptyKey)` finds free slots
3. **CAS Insert**: Winning thread atomically claims slot
4. **Repeat**: Scan next 32 slots if no match/empty found

### 4.3 Atomic Operations

```cuda
__device__ uint64_t AtomicCASKey(uint64_t* ptr, uint64_t expected, uint64_t desired) {
    return atomicCAS(reinterpret_cast<unsigned long long*>(ptr),
                     static_cast<unsigned long long>(expected),
                     static_cast<unsigned long long>(desired));
}
```

---

## 5. Multi-Factor Eviction System

When cache is full, the eviction algorithm scores candidates to find the **least valuable** entry:

### 5.1 Eviction Score Components

| Factor | Weight (default) | Description |
|--------|------------------|-------------|
| **Recency** | 0.25 | `exp(-age × staleness_decay_rate)` - recent = valuable |
| **Frequency** | 0.20 | `min(update_streak / 10, 1.0)` - frequent = valuable |
| **Variance** | 0.30 | `min(stddev / 2, 1.0)` - high variance = still learning |
| **Diversity** | 0.25 | Bloom filter score - unique = valuable |

### 5.2 Eviction Algorithm

```
┌─────────────────────────────────────────────────────────┐
│  1. Atomically increment evict_cursor                   │
│  2. Scan evict_window slots (default: 32)              │
│  3. For each slot:                                      │
│     - If empty and CAS succeeds → return slot          │
│     - If priority == 0 → skip (protected)              │
│     - Compute eviction score                           │
│  4. Select slot with LOWEST score                       │
│  5. Notify eviction delegate                           │
│  6. Overwrite slot with new key                        │
└─────────────────────────────────────────────────────────┘
```

### 5.3 Eviction Priority

- `0` = **Protected** (never evicted, set via cache warming)
- `128` = **Default** (normal eviction eligibility)
- `255` = **Evict First** (low-value entries)

---

## 6. Diversity Tracking (Bloom Filter)

### 6.1 Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `diversity_bloom_bits` | 65536 | Bloom filter size in bits |
| `diversity_hash_functions` | 4 | Number of hash probes |
| `diversity_penalty` | 0.15 | Penalty per duplicate detection |

### 6.2 Operations

**Add to Filter:**
```cuda
void BloomAdd(uint32_t* bloom, size_t bloom_size, uint64_t hash, int num_hashes) {
    for (int i = 0; i < num_hashes; ++i) {
        size_t bit_idx = (hash + i * kBloomHashPrime) % (bloom_size * 32);
        atomicOr(&bloom[bit_idx / 32], 1u << (bit_idx % 32));
    }
}
```

**Query Filter:**
- If **any** bit is 0 → definitely NOT in set (unique guess)
- If **all** bits are 1 → probably in set → apply diversity penalty

### 6.3 Impact on Eviction

New guess with low diversity score (similar to existing) gets `diversity_score = max(0.1, 1.0 - penalty)`.
During eviction, low diversity entries are **more likely** to be evicted.

---

## 7. Reward Processing

### 7.1 EMA Update with Adaptive Momentum

```cuda
float AdaptiveMomentum(float base_momentum, float confidence,
                       float momentum_floor, float momentum_ceiling) {
    // Blend base momentum with confidence-adaptive momentum
    float adaptive = momentum_floor + (momentum_ceiling - momentum_floor) * confidence;
    return clamp(0.5 * (base_momentum + adaptive), momentum_floor, momentum_ceiling);
}
```

**Higher confidence** → **higher momentum** → **smoother EMA** (trusted guesses change slowly)

### 7.2 Reward Normalization (Welford's Algorithm)

```cuda
float NormalizeReward(GuessRecord& record, float reward) {
    // Running mean/variance via Welford
    delta = reward - mean;
    mean += delta / attempts;
    delta2 = reward - mean;
    m2 += delta * delta2;
    
    variance = m2 / (attempts - 1);
    z_score = (reward - mean) / sqrt(variance);
    return clamp(z_score, -3.0, 3.0);
}
```

**Why Z-score?** Normalizes rewards across entries with different scales, enabling fair comparison.

### 7.3 Staleness Decay

```cuda
bool ApplyStalenessDecay(GuessRecord& record, float now, float grace_period, float decay_rate) {
    float elapsed = now - last_updated_ts;
    if (elapsed <= grace_period) return false;  // Within grace period
    
    float decay = exp(-elapsed * decay_rate);
    ema_reward *= decay;
    cumulative_reward *= decay;
    update_streak = 0;  // Reset frequency tracking
    return true;
}
```

**Grace period** (default 0.25s): No decay immediately after update
**Decay rate** (default 0.01/s): Exponential forgetting of stale entries

---

## 8. Confidence Calibration

### 8.1 Problem

Model confidence may be **miscalibrated** - predicting 0.9 confidence but only 0.7 accuracy.

### 8.2 Calibration Update

```cpp
void UpdateConfidenceCalibration(float observed_accuracy, float target_accuracy) {
    float error = target_accuracy - observed_accuracy;
    float new_offset = current_offset + adaptation_rate * error;
    calibration_offset = clamp(new_offset, -0.5, 0.5);
}
```

**Usage**: Add `calibration_offset` to model confidence before using for decisions.

---

## 9. GPU Kernels

### 9.1 CacheGuessKernel

```cuda
__global__ void CacheGuessKernel(const GuessMetadata* metadata_batch, size_t count)
```

**Per-thread Flow:**
1. Compose key from metadata
2. Warp-cooperative slot finding (with insert)
3. If slot found:
   - Update diversity bloom filter
   - Initialize record (if inserted) or update metadata (if existing)
   - Notify mutation delegate
4. If no slot: Attempt eviction

### 9.2 ApplyRewardKernel

```cuda
__global__ void ApplyRewardKernel(const GuessMetadata* metadata_batch,
                                  const float* reward_batch,
                                  float momentum, size_t count,
                                  GuessRewardStats* out_stats)
```

**Per-thread Flow:**
1. Compose key, find/create slot
2. Apply staleness decay
3. Normalize reward (Welford)
4. Update EMA with adaptive momentum
5. Update best/worst bounds
6. Accumulate telemetry
7. Notify reward delegate

### 9.3 WarmCacheKernel

```cuda
__global__ void WarmCacheKernel(const WarmingEntry* entries, size_t count,
                                bool replace_existing, float min_priority_protection)
```

**Purpose**: Inject high-quality pre-computed entries (e.g., from previous training run)

**Features**:
- Optional replacement of existing entries
- Priority-based eviction protection (high-priority entries are immune)
- Sets initial EMA reward from provided value

---

## 10. Memory Ownership (Rule 22)

### 10.1 Allocation in TrainingState

```cpp
// In TrainingState_GPU.hpp
bool allocateGuessCacheBuffers(size_t capacity, bool enable_diversity, 
                               size_t diversity_bloom_bits, size_t pinned_buffer_size);
```

**Allocated Buffers:**
| Buffer | Size | Purpose |
|--------|------|---------|
| `records` | `capacity × sizeof(GuessRecord)` | Main cache data |
| `keys` | `capacity × sizeof(uint64_t)` | Hash table keys |
| `size` | `sizeof(unsigned int)` | Entry count |
| `evict_cursor` | `sizeof(unsigned int)` | Eviction position |
| `diversity_bloom` | `bloom_bits / 8` | Bloom filter |
| `calibration_offset` | `sizeof(float)` | Confidence calibration |
| `single_meta_buffer` | `sizeof(GuessMetadata)` | Single-item transfer |
| `single_reward_buffer` | `sizeof(float)` | Single-item transfer |
| `pinned_meta` | `pinned_size × sizeof(GuessMetadata)` | Async transfer staging |
| `pinned_rewards` | `pinned_size × sizeof(float)` | Async transfer staging |

### 10.2 GRIM-TS Initialization

```cpp
bool InitializeGuessCache(const CacheConfig& config, 
                          const GuessCacheBuffers& buffers,  // Pointers only, no ownership
                          cudaStream_t primary_stream);
```

**Fail-Loud Checks:**
- Stream must not be nullptr or default stream
- `buffers.allocated` must be true
- All buffer pointers must be non-null
- Capacity must be > 0

### 10.3 No Dynamic Resize

```cpp
bool ResizeCache(size_t new_capacity, cudaStream_t stream) {
    fprintf(stderr, "[FATAL] GRIMTS::ResizeCache: Cannot resize dynamically!\n"
            "Rule 22: GPU allocations must come from TrainingState.\n");
    return false;
}
```

To resize: Shutdown → Free TrainingState buffers → Reallocate → Reinitialize

---

## 11. Training Loop Integration

### 11.1 GuessCacheScope (RAII)

```cpp
class GuessCacheScope {
public:
    explicit GuessCacheScope(TrainingState& training_state, 
                             size_t capacity, 
                             bool enable_async = true);
    ~GuessCacheScope();  // Calls ShutdownGuessCache()
    
    bool active() const;
};
```

**Usage in Phase2_TrainingLoop.cu:**
```cpp
GuessCacheScope cache_scope(ctx.training_state, config.guess_cache.initial_capacity);
if (!cache_scope.active()) {
    LOG_ERROR("GuessCache initialization failed");
}
```

### 11.2 MicroValidationScope

```cpp
class MicroValidationScope {
public:
    explicit MicroValidationScope(int step);  // Calls BeginMicroValidation
    ~MicroValidationScope();                  // Auto-complete if not called
    
    void complete(const MicroValidationPulse& pulse);
};
```

**MicroValidationPulse:**
```cpp
struct MicroValidationPulse {
    int   global_step;
    int   epoch;
    float train_loss;
    float learning_rate;
    float val_loss;
    float val_perplexity;
    float duration_ms;
    int   batches;
    int   sequences;
    float cache_fill_ratio;
    float cache_hit_rate;
    float avg_reward;
};
```

---

## 12. Telemetry System

### 12.1 Device-Side Accumulator

```cuda
struct CacheStatsMoment {
    float ema_sum;
    float stale_events;
    float confidence_sum;
    float diversity_sum;
    float reward_sq_sum;
    unsigned int updates;
    unsigned int hits;
    unsigned int misses;
    unsigned int evictions;
};

__device__ CacheStatsMoment d_cache_moment;  // Global accumulator
```

### 12.2 Host-Side Telemetry

```cpp
struct GuessCacheTelemetry {
    // Basic stats
    float fill_ratio;
    float average_ema;
    float stale_fraction;
    unsigned int total_records;
    
    // Extended stats
    size_t current_capacity;
    float average_confidence;
    float reward_variance;
    float diversity_score;
    float calibration_offset;
    
    // Histograms
    RewardHistogram ema_histogram;
    RewardHistogram reward_histogram;
    
    // Trends (EMAs)
    CacheTrendMetrics trends;
    
    // Health
    bool is_healthy;
    float health_score;  // 0.0 = bad, 1.0 = good
    const char* health_message;
};
```

### 12.3 Health Assessment Rules

| Condition | Health Score | Message |
|-----------|--------------|---------|
| `fill_ratio > 0.95` | 0.5 | "Cache nearly full" |
| `stale_fraction > 0.5` | 0.6 | "High staleness ratio" |
| `diversity_score < 0.3` | 0.7 | "Low diversity" |
| Otherwise | 1.0 | "OK" |

---

## 13. Delegate System (GPU-Side Callbacks)

### 13.1 Available Delegates

| Delegate | Arguments | Use Case |
|----------|-----------|----------|
| `CacheRewardDelegate` | `(GuessRecord*, float raw, float normalized)` | Track reward distribution |
| `CacheMutationDelegate` | `(GuessRecord*, int slot, MutationKind)` | Log cache changes |
| `CacheEvictionDelegate` | `(GuessRecord*, int slot, EvictionReason)` | Monitor eviction patterns |
| `CacheResizeDelegate` | `(size_t old, size_t new, int direction)` | Track capacity changes |

### 13.2 Registration

```cpp
cudaError_t RegisterCacheRewardCallback(CacheRewardDelegate::Callback callback, cudaStream_t stream);
```

**CRITICAL**: Stream must be non-null (Rule 22 enforcement)

### 13.3 Mutation/Eviction Kinds

```cpp
enum class CacheMutationKind {
    kInserted = 0,    // New entry
    kUpdated = 1,     // Metadata update
    kReplaced = 2,    // Eviction + insert
    kWarmed = 3,      // Cache warming
    kResized = 4      // Migration during resize
};

enum class EvictionReason {
    kCapacityLimit = 0,
    kStale = 1,
    kLowReward = 2,
    kLowDiversity = 3,
    kManualPurge = 4,
    kResize = 5
};
```

---

## 14. Configuration (ai_config.json)

```json
"guess_cache": {
    "enabled": true,
    "initial_capacity": 16384,
    "min_capacity": 4096,
    "max_capacity": 262144,
    "grow_threshold": 0.85,
    "shrink_threshold": 0.25,
    "grow_factor": 2.0,
    "shrink_factor": 0.1,
    "evict_window": 32,
    "staleness_decay_rate": 0.01,
    "variance_weight": 0.3,
    "frequency_weight": 0.2,
    "diversity_weight": 0.25,
    "recency_weight": 0.25,
    "enable_diversity_tracking": true,
    "diversity_bloom_bits": 65536,
    "diversity_hash_functions": 4,
    "diversity_penalty": 0.15,
    "confidence_floor": 0.01,
    "confidence_ceiling": 0.999,
    "confidence_adaptation_rate": 0.01,
    "momentum_floor": 0.55,
    "momentum_ceiling": 0.995,
    "staleness_grace_period": 0.25,
    "enable_async_transfers": true,
    "pinned_buffer_size": 8192,
    "num_streams": 2,
    "enable_histograms": true,
    "histogram_bins": 32,
    "telemetry_sample_interval": 1
}
```

---

## 15. Memory Budget Calculator

### 15.1 Per-Entry Cost

| Component | Size |
|-----------|------|
| `GuessRecord` | 100 bytes |
| Key (`uint64_t`) | 8 bytes |
| **Total per entry** | **108 bytes** |

### 15.2 Fixed Overhead

| Component | Size (default config) |
|-----------|----------------------|
| Bloom filter | 65536 / 8 = 8 KB |
| Pinned meta | 8192 × 24 = 192 KB |
| Pinned rewards | 8192 × 4 = 32 KB |
| Calibration | 4 bytes |
| Size/cursor | 8 bytes |
| **Total fixed** | **~232 KB** |

### 15.3 Example Configurations

| Capacity | Entry Memory | Fixed | Total |
|----------|--------------|-------|-------|
| 16,384 | 1.7 MB | 232 KB | **~2 MB** |
| 65,536 | 6.8 MB | 232 KB | **~7 MB** |
| 262,144 | 27.2 MB | 232 KB | **~27 MB** |

---

## 16. API Reference

### 16.1 Lifecycle

| Function | Description |
|----------|-------------|
| `InitializeGuessCache(config, buffers, stream)` | Initialize with pre-allocated buffers |
| `ShutdownGuessCache()` | Release references (not memory) |
| `ResetGuessCache(stream)` | Clear all entries |

### 16.2 Configuration

| Function | Description |
|----------|-------------|
| `GetCurrentConfig()` | Read current configuration |
| `UpdateConfig(config)` | Modify runtime configuration |

### 16.3 Cache Operations

| Function | Description |
|----------|-------------|
| `CacheGuessGPU(metadata, stream)` | Insert single entry |
| `CacheGuessBatchGPU(device_metadata, count, stream)` | Insert batch |
| `ApplyRewardGPU(metadata, reward, momentum, out, stream)` | Apply single reward |
| `ApplyRewardBatchGPU(device_metadata, device_rewards, count, momentum, out, stream)` | Apply batch rewards |

### 16.4 Async Operations

| Function | Description |
|----------|-------------|
| `CacheGuessBatchAsync(host_metadata, count)` | Non-blocking batch insert |
| `ApplyRewardBatchAsync(host_metadata, host_rewards, count, momentum)` | Non-blocking batch reward |
| `WaitForOperation(handle, timeout_ms)` | Block until complete |
| `IsOperationComplete(handle)` | Poll completion |

### 16.5 Cache Warming

| Function | Description |
|----------|-------------|
| `WarmCache(entries, count, config, stream)` | Inject pre-computed entries |
| `WarmCacheFromFile(filepath, config, stream)` | Load warming entries from file |

### 16.6 Telemetry

| Function | Description |
|----------|-------------|
| `GetCacheTelemetry(include_histograms)` | Full telemetry snapshot |
| `GetCurrentFillRatio()` | Quick fill ratio check |
| `GetCurrentSize()` | Entry count |
| `ComputeDiversityScore()` | Overall diversity metric |

### 16.7 Debug

| Function | Description |
|----------|-------------|
| `DumpCacheStats(filepath)` | Write stats to file or stdout |
| `ValidateCacheIntegrity()` | Verify size matches non-empty keys |
| `PrintCacheSummary()` | Quick console summary |

---

## 17. Logging System

### 17.1 Log Levels

```cpp
enum class LogLevel { Debug, Info, Warning, Error };
```

### 17.2 Usage

```cpp
// Register custom logger
GRIMTS::Logging::RegisterLogCallback([](LogLevel level, std::string_view message) {
    fprintf(stdout, "[GRIMTS:%d] %.*s\n", (int)level, (int)message.size(), message.data());
});

// Set minimum level
GRIMTS::Logging::SetMinLogLevel(LogLevel::Info);
```

### 17.3 Structured Log Events

| Function | When Called |
|----------|-------------|
| `LogCacheInitialized(config)` | After successful init |
| `LogCacheShutdown()` | After shutdown |
| `LogCacheReset()` | After reset |
| `LogCacheResize(old, new, success)` | After resize attempt |
| `LogCacheFault(reason)` | On error condition |
| `LogTelemetrySummary(telemetry)` | Periodic status |

---

## 18. Performance Characteristics

### 18.1 Operation Latency

| Operation | Typical Latency |
|-----------|-----------------|
| Single insert | ~5 μs |
| Single reward | ~5 μs |
| Batch insert (1K) | ~50 μs |
| Batch reward (1K) | ~50 μs |
| Telemetry read | ~100 μs (D2H sync) |

### 18.2 Scaling

- **Warp utilization**: 32 threads cooperate per lookup → 32 slots checked per iteration
- **Eviction window**: 32 slots scanned → O(1) eviction decision
- **Bloom filter**: O(k) where k = hash functions (default 4)

### 18.3 Memory Bandwidth

- **Insert**: ~108 bytes write per entry
- **Reward**: ~76 bytes read + write (stats only)
- **Eviction scan**: ~100 bytes read per candidate

---

## 19. Design Patterns

### 19.1 Warp-Cooperative Hashing

```
Benefits:
- 32× fewer iterations than single-thread scan
- __ballot_sync avoids branch divergence
- __shfl_sync broadcasts winner efficiently

Trade-offs:
- Requires active mask management
- Load imbalance if early exit
```

### 19.2 Device Constant Memory for State

```cuda
__device__ GuessCacheDeviceState d_cache_state;
__device__ KernelConfig d_kernel_config;
```

**Why**: Constant memory is cached and broadcast to all threads efficiently.

### 19.3 Atomic Accumulation for Telemetry

```cuda
atomicAdd(&d_cache_moment.ema_sum, record.stats.ema_reward);
```

**Why**: Avoids reduction kernel; telemetry is approximate anyway.

---

## 20. Future Enhancements (Not Implemented)

1. **Dynamic Resize**: Currently requires manual shutdown/reallocate cycle
2. **Multi-GPU Sharding**: Cache partitioned across GPUs
3. **Persistent Storage**: Checkpoint cache to disk
4. **Reward Prediction**: Use cache history to predict future rewards
5. **Automatic Warming**: Transfer high-value entries between training runs

---

## Appendix A: Hash Function Constants

| Constant | Value | Purpose |
|----------|-------|---------|
| `kEmptyKey` | `0xFFFFFFFFFFFFFFFF` | Sentinel for empty slot |
| `kMixPrime` | `0x9E3779B185EBCA87` | Fibonacci hashing (golden ratio) |
| `kBloomHashPrime1` | `0xC96C5795D7870F42` | Bloom filter double hashing |

---

## Appendix B: FNV-1a Hash Implementation

```cpp
uint64_t HashSignature(std::string_view text) {
    constexpr uint64_t kPrime = 1099511628211ull;
    uint64_t hash = 1469598103934665603ull;  // FNV offset basis
    for (char c : text) {
        hash ^= static_cast<uint64_t>(static_cast<unsigned char>(c));
        hash *= kPrime;
    }
    return hash;
}
```

**Properties**: Fast, good avalanche, no collisions for typical prompt/guess strings.
