#include <cfloat>
#include <cuda_runtime.h>
#include <atomic>
#include <mutex>
#include <limits>
#include <cmath>
#include <sstream>
#include <vector>
#include <algorithm>
#include <fstream>
#include <chrono>
#include <thread>
#include <cstring>
#include <stdexcept>
#include "grim-ts.hpp"
#include "../../Shared/HyperParameters/HyperParameters_GPU.hpp"
#include "../../Shared/StreamController/StreamController_GPU.hpp"

namespace {

using namespace GRIMTS;
namespace CacheEvents = GRIMTS::Delegates;
namespace CacheLog = GRIMTS::Logging;

//==============================================================================
// Global State
//==============================================================================

// Configuration (runtime modifiable)
CacheConfig g_config{};
std::mutex g_config_mutex;

// Logging infrastructure
std::mutex g_log_mutex;
std::vector<CacheLog::LogCallback> g_log_callbacks;
CacheLog::LogLevel g_min_log_level = CacheLog::LogLevel::Info;

// Micro-validation state
std::atomic<bool> g_micro_validation_active{false};
MicroValidationPulse g_last_micro_validation{};
std::mutex g_micro_validation_mutex;

// Trend tracking
CacheTrendMetrics g_trends{};
std::mutex g_trends_mutex;

// Device state
__device__ GuessCacheDeviceState d_cache_state;
GuessCacheDeviceState h_cache_state{};

// Host-side buffers for single-item operations
GuessMetadata* g_single_meta_buffer = nullptr;
float* g_single_reward_buffer = nullptr;

// Pinned buffer state (stream provided externally)
struct PinnedBuffers {
    GuessMetadata* pinned_meta = nullptr;
    float* pinned_rewards = nullptr;
    std::size_t capacity = 0;
    bool initialized = false;
};
PinnedBuffers g_pinned{};

// Primary stream reference (set during initialization)
cudaStream_t g_primary_stream = nullptr;

bool g_initialized = false;

//==============================================================================
// Constants - using centralized HyperParameters (Rule 20)
//==============================================================================

constexpr std::uint64_t kEmptyKey = 0xFFFFFFFFFFFFFFFFull;
constexpr std::uint64_t kMixPrime = 0x9E3779B185EBCA87ull;
constexpr std::uint64_t kBloomHashPrime1 = 0xC96C5795D7870F42ull;
constexpr unsigned int kWarpSizeConst = GRIM::HyperParameters::CUDA_WARP_SIZE;
constexpr float kVarianceEpsilon = GRIM::HyperParameters::EPSILON_VARIANCE;
constexpr float kNormalizedClamp = GRIM::HyperParameters::NORMALIZED_CLAMP;

//==============================================================================
// Device-side Telemetry Accumulators
//==============================================================================

struct CacheStatsMoment {
    float ema_sum = 0.0f;
    float stale_events = 0.0f;
    float confidence_sum = 0.0f;
    float diversity_sum = 0.0f;
    float reward_sq_sum = 0.0f;  // For variance calculation
    unsigned int updates = 0;
    unsigned int hits = 0;
    unsigned int misses = 0;
    unsigned int evictions = 0;
};

__device__ CacheStatsMoment d_cache_moment;
CacheStatsMoment h_cache_moment{};

//==============================================================================
// Delegate Storage (Device-Side)
//==============================================================================

__device__ CacheEvents::CacheRewardDelegate d_cache_reward_delegate;
__device__ CacheEvents::CacheMutationDelegate d_cache_mutation_delegate;
__device__ CacheEvents::CacheEvictionDelegate d_cache_eviction_delegate;
__device__ CacheEvents::CacheResizeDelegate d_cache_resize_delegate;

//==============================================================================
// Delegate Registration Kernels
//==============================================================================

__global__ void RegisterRewardCallbackKernel(CacheEvents::CacheRewardDelegate::Callback cb) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        d_cache_reward_delegate.Add(cb);
    }
}

__global__ void RegisterMutationCallbackKernel(CacheEvents::CacheMutationDelegate::Callback cb) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        d_cache_mutation_delegate.Add(cb);
    }
}

__global__ void RegisterEvictionCallbackKernel(CacheEvents::CacheEvictionDelegate::Callback cb) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        d_cache_eviction_delegate.Add(cb);
    }
}

__global__ void RegisterResizeCallbackKernel(CacheEvents::CacheResizeDelegate::Callback cb) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        d_cache_resize_delegate.Add(cb);
    }
}

__global__ void ClearRewardCallbacksKernel() {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        d_cache_reward_delegate.Clear();
    }
}

__global__ void ClearMutationCallbacksKernel() {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        d_cache_mutation_delegate.Clear();
    }
}

__global__ void ClearEvictionCallbacksKernel() {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        d_cache_eviction_delegate.Clear();
    }
}

__global__ void ClearResizeCallbacksKernel() {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        d_cache_resize_delegate.Clear();
    }
}

//==============================================================================
// Telemetry Reset
//==============================================================================

void ResetCacheTelemetry() {
    CacheStatsMoment zero{};
    cudaMemcpyToSymbol(d_cache_moment, &zero, sizeof(CacheStatsMoment));
    h_cache_moment = {};
}

//==============================================================================
// Utility Functions
//==============================================================================

template <typename T>
void SafeCudaFree(T*& ptr) {
    if (ptr) {
        cudaFree(ptr);
        ptr = nullptr;
    }
}

template <typename T>
void SafeCudaFreeHost(T*& ptr) {
    if (ptr) {
        cudaFreeHost(ptr);
        ptr = nullptr;
    }
}

__device__ __forceinline__ std::uint64_t AtomicCASKey(std::uint64_t* ptr,
                                                      std::uint64_t expected,
                                                      std::uint64_t desired) {
    return atomicCAS(reinterpret_cast<unsigned long long*>(ptr),
                     static_cast<unsigned long long>(expected),
                     static_cast<unsigned long long>(desired));
}

__device__ __forceinline__ std::uint64_t AtomicExchKey(std::uint64_t* ptr,
                                                       std::uint64_t value) {
    return atomicExch(reinterpret_cast<unsigned long long*>(ptr),
                      static_cast<unsigned long long>(value));
}

// Device-side timestamp baseline (set from host at init)
__device__ float d_timestamp_baseline = 0.0f;
__device__ float d_clock_scale = 1e-9f;  // Adjusted per-GPU at init

__device__ __forceinline__ float CurrentTimestampDevice() {
    // Returns seconds since cache initialization
    // clock64() is cycles, d_clock_scale converts to seconds based on actual GPU clock
    const double ticks = static_cast<double>(clock64());
    return static_cast<float>(ticks * d_clock_scale) - d_timestamp_baseline;
}

__device__ __forceinline__ float Clamp01(float value) {
    return fminf(fmaxf(value, 0.0f), 1.0f);
}

__device__ __forceinline__ float ClampRange(float value, float lo, float hi) {
    return fminf(fmaxf(value, lo), hi);
}

//==============================================================================
// Bloom Filter Operations (for diversity tracking)
//==============================================================================

__device__ __forceinline__ void BloomAdd(std::uint32_t* bloom, std::size_t bloom_size,
                                          std::uint64_t hash, int num_hashes) {
    if (!bloom || bloom_size == 0) return;
    
    std::uint64_t h1 = hash;
    std::uint64_t h2 = hash * kBloomHashPrime1;
    
    for (int i = 0; i < num_hashes; ++i) {
        std::uint64_t combined = h1 + i * h2;
        std::size_t bit_idx = combined % (bloom_size * 32);
        std::size_t word_idx = bit_idx / 32;
        unsigned int bit_mask = 1u << (bit_idx % 32);
        atomicOr(&bloom[word_idx], bit_mask);
    }
}

__device__ __forceinline__ bool BloomQuery(const std::uint32_t* bloom, std::size_t bloom_size,
                                            std::uint64_t hash, int num_hashes) {
    if (!bloom || bloom_size == 0) return false;
    
    std::uint64_t h1 = hash;
    std::uint64_t h2 = hash * kBloomHashPrime1;
    
    for (int i = 0; i < num_hashes; ++i) {
        std::uint64_t combined = h1 + i * h2;
        std::size_t bit_idx = combined % (bloom_size * 32);
        std::size_t word_idx = bit_idx / 32;
        unsigned int bit_mask = 1u << (bit_idx % 32);
        if ((bloom[word_idx] & bit_mask) == 0) {
            return false;  // Definitely not in set
        }
    }
    return true;  // Probably in set
}

//==============================================================================
// EMA and Reward Processing
//==============================================================================

__device__ float AdaptiveMomentum(float base_momentum, float confidence,
                                   float momentum_floor, float momentum_ceiling) {
    const float base_clamped = ClampRange(base_momentum, momentum_floor, momentum_ceiling);
    const float conf = Clamp01(confidence);
    const float adaptive = momentum_floor + (momentum_ceiling - momentum_floor) * conf;
    const float blended = 0.5f * (base_clamped + adaptive);
    return ClampRange(blended, momentum_floor, momentum_ceiling);
}

__device__ bool ApplyStalenessDecay(GuessRecord& record, float now,
                                     float grace_period, float decay_rate) {
    auto& stats = record.stats;
    const float elapsed = now - stats.last_updated_ts;
    if (elapsed <= grace_period) {
        return false;
    }
    const float decay = expf(-elapsed * decay_rate);
    stats.ema_reward *= decay;
    stats.cumulative_reward *= decay;
    stats.normalized_last *= decay;
    stats.update_streak = 0;  // Reset streak on staleness
    return true;
}

__device__ float NormalizeReward(GuessRecord& record, float reward) {
    auto& stats = record.stats;
    const float attempts = fmaxf(static_cast<float>(stats.total_attempts), 1.0f);
    const float delta = reward - stats.reward_mean;
    stats.reward_mean += delta / attempts;
    const float delta2 = reward - stats.reward_mean;
    stats.reward_m2 += delta * delta2;
    const float variance = (attempts > 1.0f)
                               ? fmaxf(stats.reward_m2 / (attempts - 1.0f), kVarianceEpsilon)
                               : kVarianceEpsilon;
    const float stddev = sqrtf(variance);
    float normalized = (reward - stats.reward_mean) / stddev;
    normalized = ClampRange(normalized, -kNormalizedClamp, kNormalizedClamp);
    stats.normalized_last = normalized;
    return normalized;
}

__device__ void AccumulateCacheTelemetry(const GuessRecord& record, 
                                          float reward, bool stale_event, bool is_hit) {
    atomicAdd(&d_cache_moment.ema_sum, record.stats.ema_reward);
    atomicAdd(&d_cache_moment.confidence_sum, record.metadata.confidence);
    atomicAdd(&d_cache_moment.diversity_sum, record.stats.diversity_score);
    atomicAdd(&d_cache_moment.reward_sq_sum, reward * reward);
    atomicAdd(&d_cache_moment.updates, 1u);
    if (stale_event) {
        atomicAdd(&d_cache_moment.stale_events, 1.0f);
    }
    if (is_hit) {
        atomicAdd(&d_cache_moment.hits, 1u);
    } else {
        atomicAdd(&d_cache_moment.misses, 1u);
    }
}

//==============================================================================
// Record Initialization
//==============================================================================

__device__ void InitializeRecord(GuessRecord& record, const GuessMetadata& metadata) {
    record.metadata = metadata;
    record.stats = {};
    record.stats.best_reward = -FLT_MAX;
    record.stats.worst_reward = FLT_MAX;
    const float now = CurrentTimestampDevice();
    record.stats.last_updated_ts = now;
    record.stats.created_ts = now;
    record.stats.diversity_score = 1.0f;
    record.stats.eviction_priority = 128;
}

__device__ inline void NotifyMutation(GuessRecord& record,
                                      int slot_index,
                                      CacheEvents::CacheMutationKind kind) {
    if (d_cache_mutation_delegate.Count() > 0) {
        d_cache_mutation_delegate.Broadcast(&record,
                                            slot_index,
                                            static_cast<int>(kind));
    }
}

__device__ inline void NotifyEviction(GuessRecord& record,
                                       int slot_index,
                                       CacheEvents::EvictionReason reason) {
    if (d_cache_eviction_delegate.Count() > 0) {
        d_cache_eviction_delegate.Broadcast(&record, slot_index, static_cast<int>(reason));
    }
    atomicAdd(&d_cache_moment.evictions, 1u);
}

//==============================================================================
// Reward Application (In-Place)
//==============================================================================

__device__ void ApplyRewardInPlace(GuessRecord& record, float reward, float momentum,
                                    float staleness_grace, float staleness_decay,
                                    float momentum_floor, float momentum_ceiling) {
    auto& stats = record.stats;
    stats.total_attempts++;
    if (reward >= 0.0f) {
        stats.positives++;
    } else {
        stats.negatives++;
    }

    stats.last_reward = reward;
    stats.cumulative_reward += reward;
    stats.update_streak++;
    
    const float now = CurrentTimestampDevice();
    const bool stale_event = ApplyStalenessDecay(record, now, staleness_grace, staleness_decay);

    const float normalized = NormalizeReward(record, reward);
    const float beta = AdaptiveMomentum(momentum, record.metadata.confidence,
                                         momentum_floor, momentum_ceiling);
    stats.ema_reward = beta * stats.ema_reward + (1.0f - beta) * normalized;
    
    if (reward > stats.best_reward) {
        stats.best_reward = reward;
    }
    if (reward < stats.worst_reward) {
        stats.worst_reward = reward;
    }
    stats.last_updated_ts = now;

    AccumulateCacheTelemetry(record, reward, stale_event, true);

    if (d_cache_reward_delegate.Count() > 0) {
        d_cache_reward_delegate.Broadcast(&record, reward, stats.normalized_last);
    }
}

//==============================================================================
// Key Composition
//==============================================================================

__device__ __forceinline__ std::uint64_t ComposeKey(const GuessMetadata& metadata) {
    std::uint64_t key = metadata.prompt_hash;
    key ^= metadata.guess_hash + kMixPrime + (key << 6) + (key >> 2);
    if (key == kEmptyKey) {
        key -= 1ull;
    }
    return key;
}

//==============================================================================
// Slot Finding (Warp-Cooperative)
//==============================================================================

struct SlotResult {
    int index;
    bool inserted;
    bool existing;
};

__device__ SlotResult WarpFindSlot(GuessCacheDeviceState state,
                                   std::uint64_t key,
                                   bool allow_insert,
                                   unsigned int lane,
                                   unsigned int mask) {
    SlotResult result{-1, false, false};
    const std::size_t capacity = state.capacity;
    if (capacity == 0) {
        return result;
    }

    for (std::size_t base = 0; base < capacity; base += kWarpSizeConst) {
        const std::size_t idx = (key + base + lane) % capacity;
        auto* key_ptr = &state.keys[idx];
        // Issue 2 FIX: Use atomic load to prevent torn reads during concurrent writes
        const std::uint64_t stored = atomicAdd(reinterpret_cast<unsigned long long*>(key_ptr), 0ull);

        unsigned int match_mask = __ballot_sync(mask, stored == key);
        if (match_mask) {
            int owner = __ffs(match_mask) - 1;
            result.index = __shfl_sync(mask, static_cast<int>(idx), owner);
            result.existing = true;
            return result;
        }

        if (!allow_insert) {
            continue;
        }

        unsigned int empty_mask = __ballot_sync(mask, stored == kEmptyKey);
        while (empty_mask) {
            int candidate_lane = __ffs(empty_mask) - 1;
            bool success = false;
            int success_idx = -1;
            int inserted_flag = 0;
            int existing_flag = 0;
            if (static_cast<int>(lane) == candidate_lane) {
                const std::uint64_t prev = AtomicCASKey(key_ptr, kEmptyKey, key);
                if (prev == kEmptyKey || prev == key) {
                    success = true;
                    success_idx = static_cast<int>(idx);
                    inserted_flag = (prev == kEmptyKey) ? 1 : 0;
                    existing_flag = (prev == key) ? 1 : 0;
                }
            }

            unsigned int success_mask = __ballot_sync(mask, success);
            if (success_mask) {
                int owner = __ffs(success_mask) - 1;
                result.index = __shfl_sync(mask, success_idx, owner);
                result.inserted = __shfl_sync(mask, inserted_flag, owner);
                result.existing = __shfl_sync(mask, existing_flag, owner);
                return result;
            }
            empty_mask &= ~(1u << candidate_lane);
        }
    }

    return result;
}

//==============================================================================
// Multi-Factor Eviction Scoring
//==============================================================================

__device__ float ComputeEvictionScore(const GuessRecord& record, float now,
                                       float variance_weight, float frequency_weight,
                                       float diversity_weight, float recency_weight,
                                       float staleness_decay_rate) {
    const auto& stats = record.stats;
    
    // Recency: How recently was this entry updated?
    const float age = now - stats.last_updated_ts;
    const float recency_score = expf(-age * staleness_decay_rate);
    
    // Frequency: How often is this entry updated?
    const float freq_score = fminf(static_cast<float>(stats.update_streak) / 10.0f, 1.0f);
    
    // Reward variance: High variance = entry is still being refined = more valuable
    const float attempts = fmaxf(static_cast<float>(stats.total_attempts), 1.0f);
    const float variance = (attempts > 1.0f) 
        ? fmaxf(stats.reward_m2 / (attempts - 1.0f), kVarianceEpsilon)
        : kVarianceEpsilon;
    const float stddev = sqrtf(variance);
    const float variance_score = fminf(stddev / 2.0f, 1.0f);  // Cap at 1
    
    // Diversity: Unique entries are more valuable
    const float diversity_score = stats.diversity_score;
    
    // Combine scores - higher = MORE valuable, should NOT be evicted
    float total = recency_weight * recency_score +
                  frequency_weight * freq_score +
                  variance_weight * variance_score +
                  diversity_weight * diversity_score;
    
    // Apply eviction priority modifier (0 = protected, 255 = evict first)
    const float priority_mod = 1.0f - (static_cast<float>(stats.eviction_priority) / 255.0f);
    total *= (0.5f + 0.5f * priority_mod);
    
    return total;
}

//==============================================================================
// Eviction with Multi-Factor Scoring
//==============================================================================

__device__ int EvictSlot(GuessCacheDeviceState state, std::uint64_t key,
                          int evict_window, float variance_weight, float frequency_weight,
                          float diversity_weight, float recency_weight, float staleness_decay) {
    if (state.capacity == 0 || state.evict_cursor == nullptr) {
        return -1;
    }

    const float now = CurrentTimestampDevice();
    unsigned int cursor = atomicAdd(state.evict_cursor, 1u);
    const int window = state.capacity < static_cast<std::size_t>(evict_window)
                           ? static_cast<int>(state.capacity)
                           : evict_window;
    
    float worst_score = FLT_MAX;
    int worst_idx = -1;

    for (int i = 0; i < window; ++i) {
        const std::size_t idx = (cursor + i) % state.capacity;
        auto* key_ptr = &state.keys[idx];
        const std::uint64_t stored = *key_ptr;
        
        // Try to grab an empty slot first
        if (stored == kEmptyKey) {
            if (AtomicCASKey(key_ptr, kEmptyKey, key) == kEmptyKey) {
                return static_cast<int>(idx);
            }
            continue;
        }

        // Score this slot for potential eviction
        const GuessRecord& record = state.records[idx];
        
        // Skip protected entries
        if (record.stats.eviction_priority == 0) {
            continue;
        }
        
        const float score = ComputeEvictionScore(record, now,
                                                  variance_weight, frequency_weight,
                                                  diversity_weight, recency_weight,
                                                  staleness_decay);
        
        if (score < worst_score) {
            worst_score = score;
            worst_idx = static_cast<int>(idx);
        }
    }

    if (worst_idx >= 0) {
        // Issue 3 FIX: Verify key hasn't changed since we scored it
        // Use CAS to atomically claim the slot - if key changed, another thread got there first
        auto* key_ptr = &state.keys[worst_idx];
        const std::uint64_t expected_key = atomicAdd(reinterpret_cast<unsigned long long*>(key_ptr), 0ull);
        const std::uint64_t prev = AtomicCASKey(key_ptr, expected_key, key);
        if (prev == expected_key) {
            // Successfully claimed slot
            NotifyEviction(state.records[worst_idx], worst_idx,
                           CacheEvents::EvictionReason::kCapacityLimit);
            return worst_idx;
        }
        // Another thread evicted this slot - retry would require loop redesign
        // For now, return failure (caller can retry entire operation)
        return -1;
    }
    return -1;
}

//==============================================================================
// Kernel Configuration (passed from host)
//==============================================================================

struct KernelConfig {
    int evict_window;
    float staleness_decay_rate;
    float staleness_grace_period;
    float variance_weight;
    float frequency_weight;
    float diversity_weight;
    float recency_weight;
    float momentum_floor;
    float momentum_ceiling;
    float diversity_penalty;
    int diversity_hash_functions;
    bool enable_diversity_tracking;
};

__device__ KernelConfig d_kernel_config;

//==============================================================================
// Cache Kernels
//==============================================================================

__global__ void CacheGuessKernel(const GuessMetadata* metadata_batch,
                                 std::size_t count) {
    const unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= count) {
        return;
    }
    
    const GuessMetadata metadata = metadata_batch[tid];
    GuessCacheDeviceState state = d_cache_state;
    KernelConfig cfg = d_kernel_config;
    
    const std::uint64_t key = ComposeKey(metadata);
    const unsigned int mask = __activemask();
    const unsigned int lane = threadIdx.x & (kWarpSizeConst - 1);
    
    SlotResult slot = WarpFindSlot(state, key, true, lane, mask);
    
    if (slot.index < 0) {
        slot.index = EvictSlot(state, key, cfg.evict_window,
                               cfg.variance_weight, cfg.frequency_weight,
                               cfg.diversity_weight, cfg.recency_weight,
                               cfg.staleness_decay_rate);
        slot.inserted = false;
        slot.existing = false;
        if (slot.index < 0) {
            return;  // Cache full and couldn't evict
        }
    }

    auto& record = state.records[slot.index];
    
    // Update diversity tracking via bloom filter
    float diversity = 1.0f;
    if (cfg.enable_diversity_tracking && state.diversity_bloom && state.bloom_size > 0) {
        if (BloomQuery(state.diversity_bloom, state.bloom_size,
                       metadata.guess_hash, cfg.diversity_hash_functions)) {
            // Similar guess exists
            diversity = fmaxf(0.1f, 1.0f - cfg.diversity_penalty);
        }
        BloomAdd(state.diversity_bloom, state.bloom_size,
                 metadata.guess_hash, cfg.diversity_hash_functions);
    }
    
    if (slot.inserted) {
        InitializeRecord(record, metadata);
        record.stats.diversity_score = diversity;
        atomicAdd(state.size, 1u);
        NotifyMutation(record, slot.index, CacheEvents::CacheMutationKind::kInserted);
    } else if (slot.existing) {
        record.metadata.confidence = metadata.confidence;
        record.stats.diversity_score = fminf(record.stats.diversity_score, diversity);
        NotifyMutation(record, slot.index, CacheEvents::CacheMutationKind::kUpdated);
    } else {
        // Eviction replacement
        InitializeRecord(record, metadata);
        record.stats.diversity_score = diversity;
        NotifyMutation(record, slot.index, CacheEvents::CacheMutationKind::kReplaced);
    }
}

__global__ void ApplyRewardKernel(const GuessMetadata* metadata_batch,
                                  const float* reward_batch,
                                  float momentum,
                                  std::size_t count,
                                  GuessRewardStats* out_stats) {
    const unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= count) {
        return;
    }

    GuessCacheDeviceState state = d_cache_state;
    KernelConfig cfg = d_kernel_config;
    
    const GuessMetadata metadata = metadata_batch[tid];
    const std::uint64_t key = ComposeKey(metadata);
    const unsigned int mask = __activemask();
    const unsigned int lane = threadIdx.x & (kWarpSizeConst - 1);
    
    SlotResult slot = WarpFindSlot(state, key, true, lane, mask);
    
    if (slot.index < 0) {
        slot.index = EvictSlot(state, key, cfg.evict_window,
                               cfg.variance_weight, cfg.frequency_weight,
                               cfg.diversity_weight, cfg.recency_weight,
                               cfg.staleness_decay_rate);
        slot.inserted = false;
        slot.existing = false;
        if (slot.index < 0) {
            if (out_stats) {
                out_stats[tid] = {};
            }
            return;
        }
    }

    auto& record = state.records[slot.index];
    
    // Update diversity tracking
    // Issue 6 FIX: Only add to bloom if NOT existing (avoid double-add from CacheGuess + ApplyReward)
    float diversity = 1.0f;
    if (cfg.enable_diversity_tracking && state.diversity_bloom && state.bloom_size > 0) {
        if (BloomQuery(state.diversity_bloom, state.bloom_size,
                       metadata.guess_hash, cfg.diversity_hash_functions)) {
            diversity = fmaxf(0.1f, 1.0f - cfg.diversity_penalty);
        }
        if (!slot.existing) {
            // Only add to bloom for new entries - existing entries already added
            BloomAdd(state.diversity_bloom, state.bloom_size,
                     metadata.guess_hash, cfg.diversity_hash_functions);
        }
    }
    
    if (slot.inserted) {
        InitializeRecord(record, metadata);
        record.stats.diversity_score = diversity;
        atomicAdd(state.size, 1u);
        NotifyMutation(record, slot.index, CacheEvents::CacheMutationKind::kInserted);
    } else if (slot.existing) {
        record.metadata.confidence = metadata.confidence;
        record.stats.diversity_score = fminf(record.stats.diversity_score, diversity);
        NotifyMutation(record, slot.index, CacheEvents::CacheMutationKind::kUpdated);
    } else {
        InitializeRecord(record, metadata);
        record.stats.diversity_score = diversity;
        NotifyMutation(record, slot.index, CacheEvents::CacheMutationKind::kReplaced);
    }

    ApplyRewardInPlace(record, reward_batch[tid], momentum,
                       cfg.staleness_grace_period, cfg.staleness_decay_rate,
                       cfg.momentum_floor, cfg.momentum_ceiling);
    
    if (out_stats) {
        out_stats[tid] = record.stats;
    }
}

//==============================================================================
// Warming Kernel
//==============================================================================

__global__ void WarmCacheKernel(const WarmingEntry* entries, std::size_t count,
                                 bool replace_existing, float min_priority_protection) {
    const unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= count) {
        return;
    }
    
    GuessCacheDeviceState state = d_cache_state;
    KernelConfig cfg = d_kernel_config;
    
    const WarmingEntry& entry = entries[tid];
    const std::uint64_t key = ComposeKey(entry.metadata);
    const unsigned int mask = __activemask();
    const unsigned int lane = threadIdx.x & (kWarpSizeConst - 1);
    
    SlotResult slot = WarpFindSlot(state, key, true, lane, mask);
    
    if (slot.existing && !replace_existing) {
        return;  // Don't replace existing
    }
    
    if (slot.index < 0) {
        slot.index = EvictSlot(state, key, cfg.evict_window,
                               cfg.variance_weight, cfg.frequency_weight,
                               cfg.diversity_weight, cfg.recency_weight,
                               cfg.staleness_decay_rate);
        if (slot.index < 0) {
            return;
        }
    }
    
    auto& record = state.records[slot.index];
    
    if (slot.inserted || !slot.existing) {
        InitializeRecord(record, entry.metadata);
        record.stats.ema_reward = entry.initial_reward;
        record.stats.cumulative_reward = entry.initial_reward;
        
        if (entry.priority >= min_priority_protection) {
            record.stats.eviction_priority = 0;  // Protected
        } else {
            record.stats.eviction_priority = static_cast<std::uint8_t>(
                255.0f * (1.0f - entry.priority));
        }
        
        if (slot.inserted) {
            atomicAdd(state.size, 1u);
        }
        NotifyMutation(record, slot.index, CacheEvents::CacheMutationKind::kWarmed);
    }
}

//==============================================================================
// Diversity Reset Kernel
//==============================================================================

__global__ void ResetDiversityBloomKernel(std::uint32_t* bloom, std::size_t word_count) {
    const unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < word_count) {
        bloom[tid] = 0;
    }
}

} // namespace

namespace GRIMTS {

//==============================================================================
// Helper: Update Device-Side Kernel Config
//==============================================================================

static void SyncKernelConfig() {
    std::lock_guard<std::mutex> lock(g_config_mutex);
    KernelConfig kcfg;
    kcfg.evict_window = g_config.evict_window;
    kcfg.staleness_decay_rate = g_config.staleness_decay_rate;
    kcfg.staleness_grace_period = g_config.staleness_grace_period;
    kcfg.variance_weight = g_config.variance_weight;
    kcfg.frequency_weight = g_config.frequency_weight;
    kcfg.diversity_weight = g_config.diversity_weight;
    kcfg.recency_weight = g_config.recency_weight;
    kcfg.momentum_floor = g_config.momentum_floor;
    kcfg.momentum_ceiling = g_config.momentum_ceiling;
    kcfg.diversity_penalty = g_config.diversity_penalty;
    kcfg.diversity_hash_functions = g_config.diversity_hash_functions;
    kcfg.enable_diversity_tracking = g_config.enable_diversity_tracking;
    cudaMemcpyToSymbol(d_kernel_config, &kcfg, sizeof(KernelConfig));
}

//==============================================================================
// Pinned Buffer Management (Uses TrainingState-owned pinned memory)
//==============================================================================

static void SetupPinnedBuffers(const GuessCacheBuffers& buffers) {
    if (g_pinned.initialized) return;
    
    // RULE 22: Pinned memory is allocated by TrainingState, we just use pointers
    if (buffers.pinned_meta && buffers.pinned_rewards && buffers.pinned_capacity > 0) {
        g_pinned.pinned_meta = static_cast<GuessMetadata*>(buffers.pinned_meta);
        g_pinned.pinned_rewards = buffers.pinned_rewards;
        g_pinned.capacity = buffers.pinned_capacity;
        g_pinned.initialized = true;
    }
}

static void ClearPinnedBufferRefs() {
    // RULE 22: We don't own these, just clear our pointers
    // TrainingState handles actual deallocation
    g_pinned.pinned_meta = nullptr;
    g_pinned.pinned_rewards = nullptr;
    g_pinned.capacity = 0;
    g_pinned.initialized = false;
}

//==============================================================================
// Primary Initialization (Rule 22 Compliant - No Allocations)
//==============================================================================

bool InitializeGuessCache(const CacheConfig& config, 
                          const GuessCacheBuffers& buffers,
                          cudaStream_t primary_stream) {
    // FAIL LOUD: Already initialized
    if (g_initialized) {
        fprintf(stderr, "[FATAL] GRIMTS::InitializeGuessCache: Already initialized! "
                "Call ShutdownGuessCache() first.\n");
        return false;
    }
    
    // FAIL LOUD: Null stream (Rule 22 - must come from StreamController)
    if (!primary_stream) {
        fprintf(stderr, "[FATAL] GRIMTS::InitializeGuessCache: primary_stream is nullptr! "
                "Stream must be obtained from TrainingState.stream_ctrl.getPrimaryStream().\n");
        return false;
    }
    GRIM::StreamController::fatalIfDefaultStream(primary_stream, "GRIMTS::InitializeGuessCache");
    
    // FAIL LOUD: Buffers not allocated
    if (!buffers.allocated) {
        fprintf(stderr, "[FATAL] GRIMTS::InitializeGuessCache: buffers.allocated is false! "
                "Call TrainingState::allocateGuessCacheBuffers() first.\n");
        return false;
    }
    
    // FAIL LOUD: Null required buffer pointers
    if (!buffers.records) {
        fprintf(stderr, "[FATAL] GRIMTS::InitializeGuessCache: buffers.records is nullptr!\n");
        return false;
    }
    if (!buffers.keys) {
        fprintf(stderr, "[FATAL] GRIMTS::InitializeGuessCache: buffers.keys is nullptr!\n");
        return false;
    }
    if (!buffers.size) {
        fprintf(stderr, "[FATAL] GRIMTS::InitializeGuessCache: buffers.size is nullptr!\n");
        return false;
    }
    if (!buffers.evict_cursor) {
        fprintf(stderr, "[FATAL] GRIMTS::InitializeGuessCache: buffers.evict_cursor is nullptr!\n");
        return false;
    }
    if (!buffers.calibration_offset) {
        fprintf(stderr, "[FATAL] GRIMTS::InitializeGuessCache: buffers.calibration_offset is nullptr!\n");
        return false;
    }
    if (!buffers.single_meta_buffer) {
        fprintf(stderr, "[FATAL] GRIMTS::InitializeGuessCache: buffers.single_meta_buffer is nullptr!\n");
        return false;
    }
    if (!buffers.single_reward_buffer) {
        fprintf(stderr, "[FATAL] GRIMTS::InitializeGuessCache: buffers.single_reward_buffer is nullptr!\n");
        return false;
    }
    
    // FAIL LOUD: Zero capacity
    if (buffers.capacity == 0) {
        fprintf(stderr, "[FATAL] GRIMTS::InitializeGuessCache: buffers.capacity is 0!\n");
        return false;
    }
    
    g_primary_stream = primary_stream;
    
    {
        std::lock_guard<std::mutex> lock(g_config_mutex);
        g_config = config;
    }
    
    // Store single-item buffer pointers (owned by TrainingState)
    g_single_meta_buffer = static_cast<GuessMetadata*>(buffers.single_meta_buffer);
    g_single_reward_buffer = buffers.single_reward_buffer;
    
    // Set up device state from pre-allocated buffers
    h_cache_state.records = static_cast<GuessRecord*>(buffers.records);
    h_cache_state.size = buffers.size;
    h_cache_state.keys = buffers.keys;
    h_cache_state.evict_cursor = buffers.evict_cursor;
    h_cache_state.diversity_bloom = buffers.diversity_bloom;  // Can be null if diversity disabled
    h_cache_state.calibration_offset = buffers.calibration_offset;
    h_cache_state.capacity = buffers.capacity;
    h_cache_state.bloom_size = buffers.bloom_words;
    
    // Copy to device constant memory
    cudaError_t err = cudaMemcpyToSymbol(d_cache_state, &h_cache_state, sizeof(h_cache_state));
    if (err != cudaSuccess) {
        fprintf(stderr, "[FATAL] GRIMTS::InitializeGuessCache: cudaMemcpyToSymbol failed! error=%s\n",
                cudaGetErrorString(err));
        return false;
    }
    
    // Issue 1 FIX: Calibrate device timestamp
    // Get GPU clock rate and set scale factor for proper time measurement
    int device = 0;
    cudaGetDevice(&device);
    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, device);
    // props.clockRate is in kHz, convert to scale: 1 / (clockRate * 1000) = seconds per tick
    float clock_scale = 1.0f / (static_cast<float>(props.clockRate) * 1000.0f);
    err = cudaMemcpyToSymbol(d_clock_scale, &clock_scale, sizeof(float));
    if (err != cudaSuccess) {
        fprintf(stderr, "[WARN] GRIMTS: Failed to set clock scale, using default\n");
    }
    // Set baseline to current clock64() value so timestamps start at ~0
    // We need a kernel to read clock64() and set baseline
    // For simplicity, set baseline to 0 - first timestamp will establish relative time
    float baseline = 0.0f;
    cudaMemcpyToSymbol(d_timestamp_baseline, &baseline, sizeof(float));
    
    SyncKernelConfig();
    ResetCacheTelemetry();
    
    // Set up pinned buffer references (if available)
    if (config.enable_async_transfers) {
        SetupPinnedBuffers(buffers);
    }
    
    // Reset trends
    {
        std::lock_guard<std::mutex> lock(g_trends_mutex);
        g_trends = {};
    }
    
    g_initialized = true;
    CacheLog::LogCacheInitialized(g_config);
    
    fprintf(stdout, "[INFO] GRIMTS: Cache initialized with capacity=%zu, diversity=%s\n",
            buffers.capacity, buffers.diversity_bloom ? "ON" : "OFF");
    
    return true;
}

//==============================================================================
// Shutdown (Rule 22 Compliant - No Deallocations)
//==============================================================================

void ShutdownGuessCache() {
    if (!g_initialized) {
        return;
    }
    
    // Sync to ensure all operations complete
    if (g_primary_stream) {
        cudaStreamSynchronize(g_primary_stream);
    } else {
        cudaDeviceSynchronize();
    }
    
    // RULE 22: Clear our references to TrainingState-owned pinned memory
    ClearPinnedBufferRefs();
    
    // RULE 22: We do NOT free GPU memory - TrainingState owns it
    // Just clear our pointers
    g_single_meta_buffer = nullptr;
    g_single_reward_buffer = nullptr;
    
    h_cache_state = {};
    cudaMemcpyToSymbol(d_cache_state, &h_cache_state, sizeof(h_cache_state));
    ResetCacheTelemetry();
    
    g_primary_stream = nullptr;
    g_initialized = false;
    CacheLog::LogCacheShutdown();
}

//==============================================================================
// Reset
//==============================================================================

void ResetGuessCache(cudaStream_t stream) {
    if (!g_initialized) {
        return;
    }
    
    cudaMemsetAsync(h_cache_state.records, 0, 
                    h_cache_state.capacity * sizeof(GuessRecord), stream);
    cudaMemsetAsync(h_cache_state.size, 0, sizeof(unsigned int), stream);
    cudaMemsetAsync(h_cache_state.keys, 0xFF, 
                    h_cache_state.capacity * sizeof(std::uint64_t), stream);
    cudaMemsetAsync(h_cache_state.evict_cursor, 0, sizeof(unsigned int), stream);
    
    if (h_cache_state.diversity_bloom && h_cache_state.bloom_size > 0) {
        cudaMemsetAsync(h_cache_state.diversity_bloom, 0,
                        h_cache_state.bloom_size * sizeof(std::uint32_t), stream);
    }
    
    float zero_cal = 0.0f;
    cudaMemcpyAsync(h_cache_state.calibration_offset, &zero_cal, sizeof(float),
                    cudaMemcpyHostToDevice, stream);
    
    ResetCacheTelemetry();
    
    {
        std::lock_guard<std::mutex> lock(g_trends_mutex);
        g_trends = {};
    }
    
    CacheLog::LogCacheReset();
}

//==============================================================================
// Configuration Access
//==============================================================================

CacheConfig GetCurrentConfig() {
    std::lock_guard<std::mutex> lock(g_config_mutex);
    return g_config;
}

void UpdateConfig(const CacheConfig& config) {
    {
        std::lock_guard<std::mutex> lock(g_config_mutex);
        g_config = config;
    }
    SyncKernelConfig();
}

//==============================================================================
// State Access
//==============================================================================

GuessCacheDeviceState GetDeviceState() {
    return h_cache_state;
}

std::size_t GetCurrentCapacity() {
    return h_cache_state.capacity;
}

std::size_t GetCurrentSize() {
    if (!g_initialized) return 0;
    unsigned int entries = 0;
    cudaMemcpy(&entries, h_cache_state.size, sizeof(unsigned int), cudaMemcpyDeviceToHost);
    return static_cast<std::size_t>(entries);
}

float GetCurrentFillRatio() {
    if (!g_initialized || h_cache_state.capacity == 0) return 0.0f;
    return static_cast<float>(GetCurrentSize()) / static_cast<float>(h_cache_state.capacity);
}

//==============================================================================
// Capacity Management (Rule 22 Compliant)
//==============================================================================

bool ResizeCache(std::size_t new_capacity, cudaStream_t stream) {
    // RULE 22: Dynamic resize requires TrainingState to provide new buffers
    // This function cannot allocate GPU memory directly
    // Issue 8 FIX: Throw instead of return false per Rule 20 (fail loud)
    throw std::runtime_error(
        "[FATAL] GRIMTS::ResizeCache: Cannot resize dynamically! "
        "Rule 22: GPU allocations must come from TrainingState. "
        "To resize: 1) ShutdownGuessCache(), 2) TrainingState::freeGuessCacheBuffers(), "
        "3) TrainingState::allocateGuessCacheBuffers(new_capacity, ...), "
        "4) InitializeGuessCache(config, buffers, stream)");
}

bool TryAutoResize(cudaStream_t stream) {
    // RULE 22: Auto-resize disabled - requires TrainingState buffer reallocation
    // Just return false, no resize possible without external reallocation
    return false;
}

std::size_t GetRecommendedCapacity() {
    if (!g_initialized) return 0;
    
    float fill = GetCurrentFillRatio();
    CacheConfig cfg = GetCurrentConfig();
    
    if (fill > cfg.grow_threshold) {
        return static_cast<std::size_t>(h_cache_state.capacity * cfg.grow_factor);
    }
    if (fill < cfg.shrink_threshold) {
        return static_cast<std::size_t>(h_cache_state.capacity * cfg.shrink_factor);
    }
    return h_cache_state.capacity;
}

//==============================================================================
// Telemetry (Enhanced)
//==============================================================================

GuessCacheTelemetry GetCacheTelemetry(bool include_histograms) {
    GuessCacheTelemetry telemetry{};
    if (!g_initialized || h_cache_state.capacity == 0) {
        return telemetry;
    }

    // Get entry count
    unsigned int entries = 0;
    if (cudaMemcpy(&entries, h_cache_state.size, sizeof(unsigned int), 
                   cudaMemcpyDeviceToHost) != cudaSuccess) {
        return telemetry;
    }
    
    telemetry.total_records = entries;
    telemetry.current_capacity = h_cache_state.capacity;
    telemetry.fill_ratio = h_cache_state.capacity > 0
        ? static_cast<float>(entries) / static_cast<float>(h_cache_state.capacity)
        : 0.0f;

    // Get accumulated stats
    cudaMemcpyFromSymbol(&h_cache_moment, d_cache_moment, sizeof(CacheStatsMoment));
    
    if (h_cache_moment.updates > 0) {
        const float denom = static_cast<float>(h_cache_moment.updates);
        telemetry.average_ema = h_cache_moment.ema_sum / denom;
        telemetry.stale_fraction = h_cache_moment.stale_events / denom;
        telemetry.average_confidence = h_cache_moment.confidence_sum / denom;
        telemetry.diversity_score = h_cache_moment.diversity_sum / denom;
        
        // Compute reward variance
        float mean_sq = h_cache_moment.reward_sq_sum / denom;
        float sq_mean = telemetry.average_ema * telemetry.average_ema;
        telemetry.reward_variance = mean_sq - sq_mean;
    }
    
    // Get calibration offset
    if (h_cache_state.calibration_offset) {
        cudaMemcpy(&telemetry.calibration_offset, h_cache_state.calibration_offset,
                   sizeof(float), cudaMemcpyDeviceToHost);
    }
    
    // Copy trend metrics
    {
        std::lock_guard<std::mutex> lock(g_trends_mutex);
        telemetry.trends = g_trends;
        
        // Update trend EMAs
        const float alpha = 0.1f;
        g_trends.fill_ratio_ema = alpha * telemetry.fill_ratio + 
                                  (1.0f - alpha) * g_trends.fill_ratio_ema;
        
        if (h_cache_moment.hits + h_cache_moment.misses > 0) {
            float hit_rate = static_cast<float>(h_cache_moment.hits) /
                            static_cast<float>(h_cache_moment.hits + h_cache_moment.misses);
            g_trends.hit_rate_ema = alpha * hit_rate + (1.0f - alpha) * g_trends.hit_rate_ema;
        }
        
        g_trends.total_inserts += h_cache_moment.updates;
        g_trends.total_evictions += h_cache_moment.evictions;
        g_trends.total_hits += h_cache_moment.hits;
        g_trends.total_misses += h_cache_moment.misses;
    }
    
    // Health assessment
    telemetry.is_healthy = true;
    telemetry.health_score = 1.0f;
    telemetry.health_message = "OK";
    
    if (telemetry.fill_ratio > 0.95f) {
        telemetry.health_score = 0.5f;
        telemetry.health_message = "Cache nearly full";
    }
    if (telemetry.stale_fraction > 0.5f) {
        telemetry.health_score = std::min(telemetry.health_score, 0.6f);
        telemetry.health_message = "High staleness ratio";
    }
    if (telemetry.diversity_score < 0.3f) {
        telemetry.health_score = std::min(telemetry.health_score, 0.7f);
        telemetry.health_message = "Low diversity";
    }
    
    if (telemetry.health_score < 0.5f) {
        telemetry.is_healthy = false;
    }

    return telemetry;
}

//==============================================================================
// Synchronous Cache Operations
//==============================================================================

cudaError_t CacheGuessGPU(const GuessMetadata& metadata, cudaStream_t stream) {
    if (!g_initialized) {
        return cudaErrorInitializationError;
    }
    cudaError_t err = cudaMemcpyAsync(g_single_meta_buffer, &metadata, 
                                       sizeof(GuessMetadata), cudaMemcpyHostToDevice, stream);
    if (err != cudaSuccess) {
        return err;
    }
    return CacheGuessBatchGPU(g_single_meta_buffer, 1, stream);
}

cudaError_t CacheGuessBatchGPU(const GuessMetadata* device_metadata,
                               std::size_t count,
                               cudaStream_t stream) {
    if (!g_initialized) {
        return cudaErrorInitializationError;
    }
    if (count == 0 || device_metadata == nullptr) {
        return cudaSuccess;
    }
    
    constexpr unsigned int kBlockSize = 128;
    const unsigned int grid = static_cast<unsigned int>((count + kBlockSize - 1) / kBlockSize);
    CacheGuessKernel<<<grid, kBlockSize, 0, stream>>>(device_metadata, count);
    return cudaGetLastError();
}

cudaError_t ApplyRewardGPU(const GuessMetadata& metadata,
                           float reward,
                           float momentum,
                           GuessRewardStats* device_out,
                           cudaStream_t stream) {
    if (!g_initialized) {
        return cudaErrorInitializationError;
    }
    cudaError_t err = cudaMemcpyAsync(g_single_meta_buffer, &metadata, 
                                       sizeof(GuessMetadata), cudaMemcpyHostToDevice, stream);
    if (err != cudaSuccess) {
        return err;
    }
    err = cudaMemcpyAsync(g_single_reward_buffer, &reward, sizeof(float), 
                          cudaMemcpyHostToDevice, stream);
    if (err != cudaSuccess) {
        return err;
    }
    return ApplyRewardBatchGPU(g_single_meta_buffer, g_single_reward_buffer,
                               1, momentum, device_out, stream);
}

cudaError_t ApplyRewardBatchGPU(const GuessMetadata* device_metadata,
                                const float* device_rewards,
                                std::size_t count,
                                float momentum,
                                GuessRewardStats* device_out,
                                cudaStream_t stream) {
    if (!g_initialized) {
        return cudaErrorInitializationError;
    }
    if (count == 0 || device_metadata == nullptr || device_rewards == nullptr) {
        return cudaSuccess;
    }
    
    constexpr unsigned int kBlockSize = 128;
    const unsigned int grid = static_cast<unsigned int>((count + kBlockSize - 1) / kBlockSize);
    ApplyRewardKernel<<<grid, kBlockSize, 0, stream>>>(device_metadata, device_rewards,
                                                       momentum, count, device_out);
    return cudaGetLastError();
}

//==============================================================================
// Async Operations
//==============================================================================

AsyncOperationHandle CacheGuessBatchAsync(const GuessMetadata* host_metadata,
                                           std::size_t count) {
    AsyncOperationHandle handle{};
    if (!g_initialized || !g_pinned.initialized || count == 0 || !g_primary_stream) {
        return handle;
    }
    
    if (count > g_pinned.capacity) {
        return handle;
    }
    
    // Use primary stream
    handle.stream = g_primary_stream;
    handle.is_valid = false;  // No completion event tracking
    
    // Copy to pinned memory
    std::memcpy(g_pinned.pinned_meta, host_metadata, count * sizeof(GuessMetadata));
    
    // Allocate device buffer and copy
    GuessMetadata* device_meta = nullptr;
    if (cudaMalloc(&device_meta, count * sizeof(GuessMetadata)) != cudaSuccess) {
        return handle;
    }
    
    cudaMemcpyAsync(device_meta, g_pinned.pinned_meta, count * sizeof(GuessMetadata),
                    cudaMemcpyHostToDevice, g_primary_stream);
    
    constexpr unsigned int kBlockSize = 128;
    const unsigned int grid = static_cast<unsigned int>((count + kBlockSize - 1) / kBlockSize);
    CacheGuessKernel<<<grid, kBlockSize, 0, g_primary_stream>>>(device_meta, count);
    
    // Schedule cleanup
    cudaFreeAsync(device_meta, g_primary_stream);
    
    handle.is_valid = true;
    return handle;
}

AsyncOperationHandle ApplyRewardBatchAsync(const GuessMetadata* host_metadata,
                                            const float* host_rewards,
                                            std::size_t count,
                                            float momentum) {
    AsyncOperationHandle handle{};
    if (!g_initialized || !g_pinned.initialized || count == 0 || !g_primary_stream) {
        return handle;
    }
    
    if (count > g_pinned.capacity) {
        return handle;
    }
    
    handle.stream = g_primary_stream;
    handle.is_valid = false;  // No completion event tracking
    
    // Copy to pinned memory
    std::memcpy(g_pinned.pinned_meta, host_metadata, count * sizeof(GuessMetadata));
    std::memcpy(g_pinned.pinned_rewards, host_rewards, count * sizeof(float));
    
    // Allocate device buffers
    GuessMetadata* device_meta = nullptr;
    float* device_rewards = nullptr;
    if (cudaMalloc(&device_meta, count * sizeof(GuessMetadata)) != cudaSuccess) {
        return handle;
    }
    if (cudaMalloc(&device_rewards, count * sizeof(float)) != cudaSuccess) {
        // Issue 4 FIX: Use async free on the stream to avoid race with pending ops
        cudaFreeAsync(device_meta, g_primary_stream);
        return handle;
    }
    
    cudaMemcpyAsync(device_meta, g_pinned.pinned_meta, count * sizeof(GuessMetadata),
                    cudaMemcpyHostToDevice, g_primary_stream);
    cudaMemcpyAsync(device_rewards, g_pinned.pinned_rewards, count * sizeof(float),
                    cudaMemcpyHostToDevice, g_primary_stream);
    
    constexpr unsigned int kBlockSize = 128;
    const unsigned int grid = static_cast<unsigned int>((count + kBlockSize - 1) / kBlockSize);
    ApplyRewardKernel<<<grid, kBlockSize, 0, g_primary_stream>>>(device_meta, device_rewards,
                                                               momentum, count, nullptr);
    
    cudaFreeAsync(device_meta, g_primary_stream);
    cudaFreeAsync(device_rewards, g_primary_stream);
    
    handle.is_valid = true;
    return handle;
}

bool WaitForOperation(const AsyncOperationHandle& handle, int timeout_ms) {
    if (!handle.is_valid || !g_primary_stream) return false;
    return cudaStreamSynchronize(g_primary_stream) == cudaSuccess;
}

bool IsOperationComplete(const AsyncOperationHandle& handle) {
    if (!handle.is_valid || !g_primary_stream) return true;
    return cudaStreamQuery(g_primary_stream) == cudaSuccess;
}

//==============================================================================
// Cache Warming
//==============================================================================

cudaError_t WarmCache(const WarmingEntry* entries,
                      std::size_t count,
                      const WarmingConfig& config,
                      cudaStream_t stream) {
    if (!g_initialized) {
        return cudaErrorInitializationError;
    }
    if (count == 0 || entries == nullptr) {
        return cudaSuccess;
    }
    
    // Allocate device buffer for entries
    WarmingEntry* device_entries = nullptr;
    cudaError_t err = cudaMalloc(&device_entries, count * sizeof(WarmingEntry));
    if (err != cudaSuccess) {
        return err;
    }
    
    err = cudaMemcpyAsync(device_entries, entries, count * sizeof(WarmingEntry),
                          cudaMemcpyHostToDevice, stream);
    if (err != cudaSuccess) {
        cudaFree(device_entries);
        return err;
    }
    
    constexpr unsigned int kBlockSize = 128;
    const unsigned int grid = static_cast<unsigned int>((count + kBlockSize - 1) / kBlockSize);
    WarmCacheKernel<<<grid, kBlockSize, 0, stream>>>(device_entries, count,
                                                      config.replace_existing,
                                                      config.min_priority_for_protection);
    
    err = cudaGetLastError();
    cudaFreeAsync(device_entries, stream);
    return err;
}

// Issue 9: Magic number and version for cache warming file format
static constexpr std::uint32_t kWarmingFileMagic = 0x47524D57;  // "GRMW"
static constexpr std::uint32_t kWarmingFileVersion = 1;
static constexpr std::size_t kMaxWarmingEntries = 10'000'000;  // 10M max to prevent OOM

cudaError_t WarmCacheFromFile(const char* filepath,
                              const WarmingConfig& config,
                              cudaStream_t stream) {
    if (!filepath) return cudaErrorInvalidValue;
    
    std::ifstream file(filepath, std::ios::binary);
    if (!file) {
        CacheLog::LogError("WarmCacheFromFile: cannot open file");
        return cudaErrorFileNotFound;
    }
    
    // Issue 9 FIX: Validate magic number
    std::uint32_t magic = 0;
    file.read(reinterpret_cast<char*>(&magic), sizeof(magic));
    if (magic != kWarmingFileMagic) {
        CacheLog::LogError("WarmCacheFromFile: invalid magic number (wrong file format)");
        return cudaErrorInvalidValue;
    }
    
    // Validate version
    std::uint32_t version = 0;
    file.read(reinterpret_cast<char*>(&version), sizeof(version));
    if (version != kWarmingFileVersion) {
        CacheLog::LogError("WarmCacheFromFile: unsupported file version");
        return cudaErrorInvalidValue;
    }
    
    // Read count with bounds check
    std::size_t count = 0;
    file.read(reinterpret_cast<char*>(&count), sizeof(count));
    if (count == 0) return cudaSuccess;
    if (count > kMaxWarmingEntries) {
        CacheLog::LogError("WarmCacheFromFile: count exceeds maximum (possible corruption)");
        return cudaErrorInvalidValue;
    }
    
    // Read entries
    std::vector<WarmingEntry> entries(count);
    file.read(reinterpret_cast<char*>(entries.data()), count * sizeof(WarmingEntry));
    
    if (!file) {
        CacheLog::LogError("WarmCacheFromFile: failed to read all entries");
        return cudaErrorInvalidValue;
    }
    
    return WarmCache(entries.data(), count, config, stream);
}

//==============================================================================
// Diversity Management
//==============================================================================

void ResetDiversityTracking(cudaStream_t stream) {
    if (!g_initialized || !h_cache_state.diversity_bloom || h_cache_state.bloom_size == 0) {
        return;
    }
    
    constexpr unsigned int kBlockSize = GRIM::HyperParameters::CUDA_BLOCK_SIZE_STANDARD;
    const unsigned int grid = static_cast<unsigned int>(
        (h_cache_state.bloom_size + kBlockSize - 1) / kBlockSize);
    ResetDiversityBloomKernel<<<grid, kBlockSize, 0, stream>>>(
        h_cache_state.diversity_bloom, h_cache_state.bloom_size);
}

float ComputeDiversityScore() {
    if (!g_initialized) return 0.0f;
    
    cudaMemcpyFromSymbol(&h_cache_moment, d_cache_moment, sizeof(CacheStatsMoment));
    if (h_cache_moment.updates == 0) return 1.0f;
    
    return h_cache_moment.diversity_sum / static_cast<float>(h_cache_moment.updates);
}

//==============================================================================
// Confidence Calibration
//==============================================================================

void UpdateConfidenceCalibration(float observed_accuracy, float target_accuracy) {
    if (!g_initialized || !h_cache_state.calibration_offset) return;
    
    CacheConfig cfg = GetCurrentConfig();
    float current_offset = 0.0f;
    cudaMemcpy(&current_offset, h_cache_state.calibration_offset, sizeof(float),
               cudaMemcpyDeviceToHost);
    
    float error = target_accuracy - observed_accuracy;
    float new_offset = current_offset + cfg.confidence_adaptation_rate * error;
    new_offset = std::max(-0.5f, std::min(0.5f, new_offset));  // Clamp
    
    cudaMemcpy(h_cache_state.calibration_offset, &new_offset, sizeof(float),
               cudaMemcpyHostToDevice);
}

float GetCalibrationOffset() {
    if (!g_initialized || !h_cache_state.calibration_offset) return 0.0f;
    
    float offset = 0.0f;
    cudaMemcpy(&offset, h_cache_state.calibration_offset, sizeof(float),
               cudaMemcpyDeviceToHost);
    return offset;
}

void ResetCalibration(cudaStream_t stream) {
    if (!g_initialized || !h_cache_state.calibration_offset) return;
    
    float zero = 0.0f;
    cudaMemcpyAsync(h_cache_state.calibration_offset, &zero, sizeof(float),
                    cudaMemcpyHostToDevice, stream);
}

//==============================================================================
// Export/Import
//==============================================================================

cudaError_t ExportCacheToHost(GuessRecord* host_buffer,
                              std::size_t buffer_capacity,
                              std::size_t* out_count) {
    if (!g_initialized || !host_buffer || !out_count) {
        return cudaErrorInvalidValue;
    }
    
    unsigned int entries = 0;
    cudaMemcpy(&entries, h_cache_state.size, sizeof(unsigned int), cudaMemcpyDeviceToHost);
    
    std::size_t copy_count = std::min(static_cast<std::size_t>(entries), 
                                       std::min(buffer_capacity, h_cache_state.capacity));
    
    cudaError_t err = cudaMemcpy(host_buffer, h_cache_state.records,
                                  copy_count * sizeof(GuessRecord), cudaMemcpyDeviceToHost);
    *out_count = copy_count;
    return err;
}

cudaError_t ImportCacheFromHost(const GuessRecord* host_buffer,
                                std::size_t count,
                                cudaStream_t stream) {
    if (!g_initialized || !host_buffer) {
        return cudaErrorInvalidValue;
    }
    
    std::size_t copy_count = std::min(count, h_cache_state.capacity);
    
    cudaError_t err = cudaMemcpyAsync(h_cache_state.records, host_buffer,
                                       copy_count * sizeof(GuessRecord),
                                       cudaMemcpyHostToDevice, stream);
    if (err != cudaSuccess) return err;
    
    // Issue 10 FIX: Rebuild keys array from imported records
    // First, reset all keys to empty
    err = cudaMemsetAsync(h_cache_state.keys, 0xFF,
                          h_cache_state.capacity * sizeof(std::uint64_t), stream);
    if (err != cudaSuccess) return err;
    
    // Build keys on host and upload (records contain metadata with hashes)
    std::vector<std::uint64_t> keys(h_cache_state.capacity, kEmptyKey);
    for (std::size_t i = 0; i < copy_count; ++i) {
        // Compose key from metadata in record
        std::uint64_t key = host_buffer[i].metadata.prompt_hash;
        key ^= host_buffer[i].metadata.guess_hash + kMixPrime + (key << 6) + (key >> 2);
        if (key == kEmptyKey) key -= 1ull;
        keys[i] = key;
    }
    
    err = cudaMemcpyAsync(h_cache_state.keys, keys.data(),
                          h_cache_state.capacity * sizeof(std::uint64_t),
                          cudaMemcpyHostToDevice, stream);
    if (err != cudaSuccess) return err;
    
    // Update size
    unsigned int new_size = static_cast<unsigned int>(copy_count);
    return cudaMemcpyAsync(h_cache_state.size, &new_size, sizeof(unsigned int),
                           cudaMemcpyHostToDevice, stream);
}

//==============================================================================
// Hashing
//==============================================================================

std::uint64_t HashSignature(std::string_view text) {
    constexpr std::uint64_t kPrime = 1099511628211ull;
    std::uint64_t hash = 1469598103934665603ull;
    for (char c : text) {
        hash ^= static_cast<std::uint64_t>(static_cast<unsigned char>(c));
        hash *= kPrime;
    }
    return hash;
}

std::uint64_t HashSignature(const void* data, std::size_t size) {
    if (!data || size == 0) return 0;
    constexpr std::uint64_t kPrime = 1099511628211ull;
    std::uint64_t hash = 1469598103934665603ull;
    const auto* bytes = static_cast<const unsigned char*>(data);
    for (std::size_t i = 0; i < size; ++i) {
        hash ^= static_cast<std::uint64_t>(bytes[i]);
        hash *= kPrime;
    }
    return hash;
}

//==============================================================================
// Micro-Validation
//==============================================================================

void BeginMicroValidation(int global_step, int epoch) {
    g_micro_validation_active.store(true, std::memory_order_release);
    std::lock_guard<std::mutex> lock(g_micro_validation_mutex);
    g_last_micro_validation = {};
    g_last_micro_validation.global_step = global_step;
    g_last_micro_validation.epoch = epoch;
}

void CompleteMicroValidation(const MicroValidationPulse& pulse) {
    {
        std::lock_guard<std::mutex> lock(g_micro_validation_mutex);
        g_last_micro_validation = pulse;
    }
    g_micro_validation_active.store(false, std::memory_order_release);
}

bool MicroValidationActive() {
    return g_micro_validation_active.load(std::memory_order_acquire);
}

MicroValidationPulse GetLastMicroValidationPulse() {
    std::lock_guard<std::mutex> lock(g_micro_validation_mutex);
    return g_last_micro_validation;
}

//==============================================================================
// Debug/Diagnostics
//==============================================================================

void DumpCacheStats(const char* filepath) {
    GuessCacheTelemetry tel = GetCacheTelemetry(true);
    
    std::ostringstream ss;
    ss << "=== GRIM-TS Cache Statistics ===\n";
    ss << "Capacity: " << tel.current_capacity << "\n";
    ss << "Records: " << tel.total_records << "\n";
    ss << "Fill Ratio: " << (tel.fill_ratio * 100.0f) << "%\n";
    ss << "Average EMA: " << tel.average_ema << "\n";
    ss << "Stale Fraction: " << (tel.stale_fraction * 100.0f) << "%\n";
    ss << "Average Confidence: " << tel.average_confidence << "\n";
    ss << "Diversity Score: " << tel.diversity_score << "\n";
    ss << "Calibration Offset: " << tel.calibration_offset << "\n";
    ss << "Health: " << (tel.is_healthy ? "OK" : "DEGRADED") 
       << " (score=" << tel.health_score << ")\n";
    ss << "Health Message: " << tel.health_message << "\n";
    ss << "\n--- Trends ---\n";
    ss << "Total Inserts: " << tel.trends.total_inserts << "\n";
    ss << "Total Updates: " << tel.trends.total_updates << "\n";
    ss << "Total Evictions: " << tel.trends.total_evictions << "\n";
    ss << "Total Hits: " << tel.trends.total_hits << "\n";
    ss << "Total Misses: " << tel.trends.total_misses << "\n";
    ss << "Resize Events: " << tel.trends.resize_events << "\n";
    ss << "Hit Rate EMA: " << (tel.trends.hit_rate_ema * 100.0f) << "%\n";
    
    if (filepath) {
        std::ofstream file(filepath);
        if (file) {
            file << ss.str();
        }
    } else {
        CacheLog::LogInfo(ss.str());
    }
}

bool ValidateCacheIntegrity() {
    if (!g_initialized) return false;
    
    // Issue 5 FIX: Synchronize stream first to get consistent snapshot
    if (g_primary_stream) {
        cudaStreamSynchronize(g_primary_stream);
    } else {
        cudaDeviceSynchronize();
    }
    
    // Read size FIRST (smaller, faster)
    unsigned int reported_size = 0;
    cudaMemcpy(&reported_size, h_cache_state.size, sizeof(unsigned int), cudaMemcpyDeviceToHost);
    
    // Then read keys
    std::vector<std::uint64_t> keys(h_cache_state.capacity);
    cudaMemcpy(keys.data(), h_cache_state.keys, 
               h_cache_state.capacity * sizeof(std::uint64_t), cudaMemcpyDeviceToHost);
    
    std::size_t non_empty = 0;
    for (std::size_t i = 0; i < h_cache_state.capacity; ++i) {
        if (keys[i] != kEmptyKey) {
            ++non_empty;
        }
    }
    
    if (non_empty != reported_size) {
        std::ostringstream msg;
        msg << "Integrity check failed: reported=" << reported_size 
            << " actual=" << non_empty;
        CacheLog::LogError(msg.str());
        return false;
    }
    
    return true;
}

void PrintCacheSummary() {
    DumpCacheStats(nullptr);
}

} // namespace GRIMTS

//==============================================================================
// Delegate Registration (GRIMTS::Delegates)
//==============================================================================

namespace GRIMTS::Delegates {

cudaError_t RegisterCacheRewardCallback(CacheRewardDelegate::Callback callback, cudaStream_t stream) {
    if (callback == nullptr) {
        return cudaErrorInvalidValue;
    }
    if (!stream) {
        fprintf(stderr, "[FATAL] RegisterCacheRewardCallback: stream is nullptr!\n");
        return cudaErrorInvalidValue;
    }
    RegisterRewardCallbackKernel<<<1, 32, 0, stream>>>(callback);
    return cudaStreamSynchronize(stream);
}

cudaError_t RegisterCacheMutationCallback(CacheMutationDelegate::Callback callback, cudaStream_t stream) {
    if (callback == nullptr) {
        return cudaErrorInvalidValue;
    }
    if (!stream) {
        fprintf(stderr, "[FATAL] RegisterCacheMutationCallback: stream is nullptr!\n");
        return cudaErrorInvalidValue;
    }
    RegisterMutationCallbackKernel<<<1, 32, 0, stream>>>(callback);
    return cudaStreamSynchronize(stream);
}

cudaError_t RegisterCacheEvictionCallback(CacheEvictionDelegate::Callback callback, cudaStream_t stream) {
    if (callback == nullptr) {
        return cudaErrorInvalidValue;
    }
    if (!stream) {
        fprintf(stderr, "[FATAL] RegisterCacheEvictionCallback: stream is nullptr!\n");
        return cudaErrorInvalidValue;
    }
    RegisterEvictionCallbackKernel<<<1, 32, 0, stream>>>(callback);
    return cudaStreamSynchronize(stream);
}

cudaError_t RegisterCacheResizeCallback(CacheResizeDelegate::Callback callback, cudaStream_t stream) {
    if (callback == nullptr) {
        return cudaErrorInvalidValue;
    }
    if (!stream) {
        fprintf(stderr, "[FATAL] RegisterCacheResizeCallback: stream is nullptr!\n");
        return cudaErrorInvalidValue;
    }
    RegisterResizeCallbackKernel<<<1, 32, 0, stream>>>(callback);
    return cudaStreamSynchronize(stream);
}

cudaError_t ClearCacheRewardCallbacks(cudaStream_t stream) {
    if (!stream) {
        fprintf(stderr, "[FATAL] ClearCacheRewardCallbacks: stream is nullptr!\n");
        return cudaErrorInvalidValue;
    }
    ClearRewardCallbacksKernel<<<1, 32, 0, stream>>>();
    return cudaStreamSynchronize(stream);
}

cudaError_t ClearCacheMutationCallbacks(cudaStream_t stream) {
    if (!stream) {
        fprintf(stderr, "[FATAL] ClearCacheMutationCallbacks: stream is nullptr!\n");
        return cudaErrorInvalidValue;
    }
    ClearMutationCallbacksKernel<<<1, 32, 0, stream>>>();
    return cudaStreamSynchronize(stream);
}

cudaError_t ClearCacheEvictionCallbacks(cudaStream_t stream) {
    if (!stream) {
        fprintf(stderr, "[FATAL] ClearCacheEvictionCallbacks: stream is nullptr!\n");
        return cudaErrorInvalidValue;
    }
    ClearEvictionCallbacksKernel<<<1, 32, 0, stream>>>();
    return cudaStreamSynchronize(stream);
}

cudaError_t ClearCacheResizeCallbacks(cudaStream_t stream) {
    if (!stream) {
        fprintf(stderr, "[FATAL] ClearCacheResizeCallbacks: stream is nullptr!\n");
        return cudaErrorInvalidValue;
    }
    ClearResizeCallbacksKernel<<<1, 32, 0, stream>>>();
    return cudaStreamSynchronize(stream);
}

cudaError_t ClearAllCallbacks(cudaStream_t stream) {
    ClearCacheRewardCallbacks(stream);
    ClearCacheMutationCallbacks(stream);
    ClearCacheEvictionCallbacks(stream);
    ClearCacheResizeCallbacks(stream);
    return cudaSuccess;
}

} // namespace GRIMTS::Delegates

//======================================================//
//  GRIM-TS Logging Implementation
//======================================================//

namespace GRIMTS::Logging {

void RegisterLogCallback(LogCallback callback) {
    if (!callback) return;
    std::lock_guard<std::mutex> lock(g_log_mutex);
    g_log_callbacks.push_back(std::move(callback));
    fprintf(stderr, "[DEBUG] GRIMTS::Logging::RegisterLogCallback - now have %zu callbacks\n", g_log_callbacks.size());
}

void ClearLogCallbacks() {
    std::lock_guard<std::mutex> lock(g_log_mutex);
    g_log_callbacks.clear();
}

void SetMinLogLevel(LogLevel level) {
    g_min_log_level = level;
}

void EmitLog(LogLevel level, std::string_view message) {
    fprintf(stderr, "[DEBUG] GRIMTS::EmitLog called with level=%d, callbacks=%zu, msg=%.*s\n", 
            static_cast<int>(level), g_log_callbacks.size(), (int)message.size(), message.data());
    if (static_cast<int>(level) < static_cast<int>(g_min_log_level)) {
        fprintf(stderr, "[DEBUG] GRIMTS::EmitLog - filtered by level (min=%d)\n", static_cast<int>(g_min_log_level));
        return;
    }
    std::lock_guard<std::mutex> lock(g_log_mutex);
    for (const auto& cb : g_log_callbacks) {
        if (cb) {
            cb(level, message);
        }
    }
}

void ReportCudaError(std::string_view action, cudaError_t err) {
    if (err == cudaSuccess) return;
    std::ostringstream msg;
    msg << "[GuessCache] " << action << " failed (code=" << static_cast<int>(err) << ")";
    msg << " " << cudaGetErrorString(err);
    EmitLog(LogLevel::Error, msg.str());
}

void LogCacheInitialized(const GRIMTS::CacheConfig& config) {
    std::ostringstream msg;
    msg << "[GuessCache] GPU cache ready (capacity=" << config.initial_capacity 
        << ", diversity=" << (config.enable_diversity_tracking ? "ON" : "OFF")
        << ", async=" << (config.enable_async_transfers ? "ON" : "OFF") << ")";
    EmitLog(LogLevel::Info, msg.str());
}

void LogCacheShutdown() {
    EmitLog(LogLevel::Info, "[GuessCache] Shutdown complete");
}

void LogCacheReset() {
    EmitLog(LogLevel::Info, "[GuessCache] Cache reset");
}

void LogCacheResize(std::size_t old_capacity, std::size_t new_capacity, bool success) {
    std::ostringstream msg;
    msg << "[GuessCache] Resize " << old_capacity << " -> " << new_capacity
        << (success ? " OK" : " FAILED");
    EmitLog(success ? LogLevel::Info : LogLevel::Error, msg.str());
}

void LogCacheFault(std::string_view reason) {
    std::ostringstream msg;
    msg << "[GuessCache] Cache faulted: " << reason;
    EmitLog(LogLevel::Error, msg.str());
}

void LogTelemetrySummary(const GRIMTS::GuessCacheTelemetry& telemetry) {
    std::ostringstream msg;
    msg << "[GuessCache] Telemetry: fill=" << (telemetry.fill_ratio * 100.0f) << "%"
        << ", records=" << telemetry.total_records
        << ", avg_ema=" << telemetry.average_ema
        << ", health=" << (telemetry.is_healthy ? "OK" : "DEGRADED");
    EmitLog(LogLevel::Info, msg.str());
}

} // namespace GRIMTS::Logging
