#pragma once
#include <string_view>
#include <cstdint>
#include <cstddef>
#include <limits>
#include <functional>
#include <array>
#include <cuda_runtime_api.h>

//==============================================================================
// GRIM-TS (Guess-Reward Integrated Memory - Training System)
// Production-Ready GPU-Resident Cache for Speculative Decoding + RL Integration
//
// Features:
//   - Adaptive capacity with automatic resize on high fill ratio
//   - Multi-factor eviction (reward variance, update frequency, diversity)
//   - Diversity tracking to penalize repetitive guesses
//   - Cache warming for high-quality sequence injection
//   - Confidence calibration based on training progress
//   - Rich telemetry with histograms and trend analysis
//   - Async GPU transfers via pinned memory pools
//   - Thread-safe host-side operations
//==============================================================================

namespace GRIMTS {

//------------------------------------------------------------------------------
// Configuration Constants (tunable via CacheConfig)
//------------------------------------------------------------------------------

struct CacheConfig {
    // Capacity management
    std::size_t initial_capacity = 16384;       // Starting cache size
    std::size_t min_capacity = 4096;            // Minimum after shrink
    std::size_t max_capacity = 262144;          // Maximum allowed (256K entries)
    float grow_threshold = 0.85f;               // Resize up when fill > this
    float shrink_threshold = 0.25f;             // Resize down when fill < this
    float grow_factor = 2.0f;                   // Multiply capacity by this
    float shrink_factor = 0.5f;                 // Shrink capacity by this
    
    // Eviction parameters
    int evict_window = 32;                      // Slots to scan for eviction
    float staleness_decay_rate = 0.01f;         // Per-second penalty for old entries
    float variance_weight = 0.3f;               // Eviction: high variance = more valuable
    float frequency_weight = 0.2f;              // Eviction: frequent updates = more valuable
    float diversity_weight = 0.25f;             // Eviction: unique guesses = more valuable
    float recency_weight = 0.25f;               // Eviction: recent = more valuable
    
    // Diversity tracking
    bool enable_diversity_tracking = true;
    int diversity_bloom_bits = 65536;           // Bloom filter size for hash diversity
    int diversity_hash_functions = 4;           // Number of hash functions
    float diversity_penalty = 0.15f;            // Penalty per duplicate detection
    
    // Confidence calibration
    float confidence_floor = 0.01f;             // Minimum confidence
    float confidence_ceiling = 0.999f;          // Maximum confidence
    float confidence_adaptation_rate = 0.01f;   // How fast calibration adjusts
    
    // EMA parameters
    float momentum_floor = 0.01f;               // Minimum EMA momentum
    float momentum_ceiling = 0.995f;            // Maximum EMA momentum
    float staleness_grace_period = 0.25f;       // Seconds before staleness decay
    
    // Async operation
    bool enable_async_transfers = true;
    std::size_t pinned_buffer_size = 8192;      // Pinned memory pool entries
    int num_streams = 2;                        // Concurrent CUDA streams
    
    // Telemetry
    bool enable_histograms = true;
    int histogram_bins = 32;                    // Bins for EMA/reward histograms
    int telemetry_sample_interval = 100;        // Sample every N operations
};

//------------------------------------------------------------------------------
// Core Data Structures
//------------------------------------------------------------------------------

struct GuessMetadata {
    std::uint64_t prompt_hash = 0;        // Hash of prompt/context signature
    std::uint64_t guess_hash = 0;         // Hash of the generated guess text
    float confidence = 0.0f;              // Model's confidence score
    std::uint16_t sequence_length = 0;    // Token count of guess
    std::uint16_t prompt_length = 0;      // Token count of prompt context
    std::uint32_t epoch = 0;              // Training epoch when cached
};

struct GuessRewardStats {
    std::uint32_t total_attempts = 0;
    std::uint32_t positives = 0;
    std::uint32_t negatives = 0;
    float cumulative_reward = 0.0f;
    float ema_reward = 0.0f;
    float last_reward = 0.0f;
    float best_reward = -std::numeric_limits<float>::infinity();
    float worst_reward = std::numeric_limits<float>::infinity();
    float last_updated_ts = 0.0f;
    float created_ts = 0.0f;              // When entry was first created
    float reward_mean = 0.0f;
    float reward_m2 = 0.0f;               // For running variance (Welford)
    float normalized_last = 0.0f;
    std::uint32_t update_streak = 0;      // Consecutive updates (for frequency)
    float diversity_score = 1.0f;         // 1.0 = unique, lower = repetitive
    std::uint8_t eviction_priority = 128; // 0 = protected, 255 = evict first
};

struct GuessRecord {
    GuessMetadata metadata;
    GuessRewardStats stats;
};

// Enforce size assumptions used by GuessCacheBuffers_GPU.cu::allocateGuessCacheBuffers()
static_assert(sizeof(GuessMetadata) == 32, "GuessMetadata size changed — update GUESS_METADATA_SIZE in GuessCacheBuffers_GPU.cu");
static_assert(sizeof(GuessRecord) == 96, "GuessRecord size changed — update GUESS_RECORD_SIZE in GuessCacheBuffers_GPU.cu");

//------------------------------------------------------------------------------
// Device-Side State (accessed by CUDA kernels)
//------------------------------------------------------------------------------

struct GuessCacheDeviceState {
    GuessRecord* records = nullptr;
    unsigned int* size = nullptr;
    std::uint64_t* keys = nullptr;
    unsigned int* evict_cursor = nullptr;
    std::uint32_t* diversity_bloom = nullptr;  // Bloom filter for diversity
    float* calibration_offset = nullptr;       // Dynamic confidence calibration
    int* slot_locks = nullptr;                 // Per-slot spin locks for Welford consistency
    std::size_t capacity = 0;
    std::size_t bloom_size = 0;
};

//------------------------------------------------------------------------------
// Rich Telemetry Structures
//------------------------------------------------------------------------------

struct HistogramBin {
    float lower_bound = 0.0f;
    float upper_bound = 0.0f;
    std::uint32_t count = 0;
};

struct RewardHistogram {
    static constexpr int kMaxBins = 64;
    std::array<HistogramBin, kMaxBins> bins{};
    int num_bins = 32;
    float total_samples = 0;
    float min_value = std::numeric_limits<float>::infinity();
    float max_value = -std::numeric_limits<float>::infinity();
};

struct CacheTrendMetrics {
    float fill_ratio_ema = 0.0f;           // Smoothed fill ratio
    float hit_rate_ema = 0.0f;             // Cache hit rate
    float eviction_rate_ema = 0.0f;        // Evictions per operation
    float avg_reward_trend = 0.0f;         // Is average reward improving?
    float diversity_trend = 0.0f;          // Is cache becoming more diverse?
    std::uint64_t total_inserts = 0;
    std::uint64_t total_updates = 0;
    std::uint64_t total_evictions = 0;
    std::uint64_t total_hits = 0;
    std::uint64_t total_misses = 0;
    int resize_events = 0;                 // How many times cache resized
    float last_resize_ts = 0.0f;
};

struct GuessCacheTelemetry {
    // Basic stats
    float fill_ratio = 0.0f;
    float average_ema = 0.0f;
    float stale_fraction = 0.0f;
    unsigned int total_records = 0;
    
    // Extended stats
    std::size_t current_capacity = 0;
    float average_confidence = 0.0f;
    float reward_variance = 0.0f;
    float diversity_score = 0.0f;          // Overall cache diversity
    float calibration_offset = 0.0f;       // Current confidence calibration
    
    // Histograms (optional)
    RewardHistogram ema_histogram{};
    RewardHistogram reward_histogram{};
    
    // Trends
    CacheTrendMetrics trends{};
    
    // Health indicators
    bool is_healthy = true;
    float health_score = 1.0f;             // 0.0 = bad, 1.0 = good
    const char* health_message = "OK";
};

//------------------------------------------------------------------------------
// Cache Warming / Injection Structures
//------------------------------------------------------------------------------

struct WarmingEntry {
    GuessMetadata metadata;
    float initial_reward = 1.0f;           // Starting EMA reward
    float priority = 1.0f;                 // Higher = more important to keep
};

struct WarmingConfig {
    bool replace_existing = true;         // Overwrite existing entries?
    bool respect_diversity = true;         // Skip if too similar to existing
    float min_priority_for_protection = 0.9f;  // Entries above this get eviction protection
};

//------------------------------------------------------------------------------
// Async Operation Handles
//------------------------------------------------------------------------------

struct AsyncOperationHandle {
    cudaStream_t stream = nullptr;
    cudaEvent_t completion_event = nullptr;
    bool is_valid = false;
    std::size_t operation_id = 0;
};

//------------------------------------------------------------------------------
// Pre-Allocated Buffer Injection (Rule 22 Compliant)
// TrainingState owns all GPU memory, GRIM-TS uses pointers
//------------------------------------------------------------------------------

struct GuessCacheBuffers {
    // Main cache structures (all allocated by TrainingState)
    void* records = nullptr;               // GuessRecord array [capacity]
    uint64_t* keys = nullptr;              // Hash keys [capacity]
    unsigned int* size = nullptr;          // Entry count (single value)
    unsigned int* evict_cursor = nullptr;  // Eviction position (single value)
    
    // Optional diversity bloom filter
    uint32_t* diversity_bloom = nullptr;
    size_t bloom_words = 0;
    
    // Calibration
    float* calibration_offset = nullptr;
    
    // Per-slot spin locks
    int* slot_locks = nullptr;
    
    // Single-item transfer buffers
    void* single_meta_buffer = nullptr;    // Single GuessMetadata
    float* single_reward_buffer = nullptr; // Single float reward
    
    // Pinned host memory for async transfers
    void* pinned_meta = nullptr;
    float* pinned_rewards = nullptr;
    size_t pinned_capacity = 0;
    
    // Capacity tracking
    size_t capacity = 0;
    bool allocated = false;
};

//------------------------------------------------------------------------------
// Core API
//------------------------------------------------------------------------------

// Lifecycle - RULE 22: Buffers must be pre-allocated by TrainingState
// FAILS LOUD if buffers are nullptr or capacity is 0
bool InitializeGuessCache(const CacheConfig& config, 
                          const GuessCacheBuffers& buffers,
                          cudaStream_t primary_stream);
void ShutdownGuessCache();
void ResetGuessCache(cudaStream_t stream = nullptr);

// Configuration (can be modified at runtime)
CacheConfig GetCurrentConfig();
void UpdateConfig(const CacheConfig& config);

// Step counter — call once per training step to advance the deterministic clock
void AdvanceStep(float step);

// Capacity management
bool ResizeCache(std::size_t new_capacity, cudaStream_t stream = nullptr);
bool TryAutoResize(cudaStream_t stream = nullptr);  // Based on fill ratio
std::size_t GetRecommendedCapacity();

// State access
GuessCacheDeviceState GetDeviceState();
GuessCacheTelemetry GetCacheTelemetry(bool include_histograms = false);
float GetCurrentFillRatio();
std::size_t GetCurrentSize();
std::size_t GetCurrentCapacity();

// Synchronous operations (block until complete)
cudaError_t CacheGuessGPU(const GuessMetadata& metadata,
                          cudaStream_t stream = nullptr);

cudaError_t CacheGuessBatchGPU(const GuessMetadata* device_metadata,
                               std::size_t count,
                               cudaStream_t stream = nullptr);

cudaError_t ApplyRewardGPU(const GuessMetadata& metadata,
                           float reward,
                           float momentum,
                           GuessRewardStats* device_out = nullptr,
                           cudaStream_t stream = nullptr);

cudaError_t ApplyRewardBatchGPU(const GuessMetadata* device_metadata,
                                const float* device_rewards,
                                std::size_t count,
                                float momentum,
                                GuessRewardStats* device_out = nullptr,
                                cudaStream_t stream = nullptr);

// Async operations (non-blocking with completion tracking)
AsyncOperationHandle CacheGuessBatchAsync(const GuessMetadata* host_metadata,
                                          std::size_t count);
AsyncOperationHandle ApplyRewardBatchAsync(const GuessMetadata* host_metadata,
                                           const float* host_rewards,
                                           std::size_t count,
                                           float momentum);
bool WaitForOperation(const AsyncOperationHandle& handle, int timeout_ms = -1);
bool IsOperationComplete(const AsyncOperationHandle& handle);

// Cache warming
cudaError_t WarmCache(const WarmingEntry* entries,
                      std::size_t count,
                      const WarmingConfig& config = {},
                      cudaStream_t stream = nullptr);

cudaError_t WarmCacheFromFile(const char* filepath,
                              const WarmingConfig& config = {},
                              cudaStream_t stream = nullptr);

// Diversity management
void ResetDiversityTracking(cudaStream_t stream = nullptr);
float ComputeDiversityScore();  // Overall cache diversity 0.0-1.0

// Confidence calibration
void UpdateConfidenceCalibration(float observed_accuracy, float target_accuracy);
float GetCalibrationOffset();
void ResetCalibration(cudaStream_t stream = nullptr);

// Export/Import (for checkpointing)
cudaError_t ExportCacheToHost(GuessRecord* host_buffer,
                              std::size_t buffer_capacity,
                              std::size_t* out_count);
cudaError_t ImportCacheFromHost(const GuessRecord* host_buffer,
                                std::size_t count,
                                cudaStream_t stream = nullptr);

// Utility
std::uint64_t HashSignature(std::string_view text);
std::uint64_t HashSignature(const void* data, std::size_t size);

// Debug/Diagnostics
void DumpCacheStats(const char* filepath = nullptr);  // nullptr = stdout
bool ValidateCacheIntegrity();
void PrintCacheSummary();

} // namespace GRIMTS

//------------------------------------------------------------------------------
// Logging Integration
//------------------------------------------------------------------------------

namespace GRIMTS::Logging {

enum class LogLevel { Debug, Info, Warning, Error };

using LogCallback = std::function<void(LogLevel level, std::string_view message)>;

void RegisterLogCallback(LogCallback callback);
void ClearLogCallbacks();
void SetMinLogLevel(LogLevel level);

// Internal logging functions
void EmitLog(LogLevel level, std::string_view message);
inline void LogDebug(std::string_view message) { EmitLog(LogLevel::Debug, message); }
inline void LogInfo(std::string_view message) { EmitLog(LogLevel::Info, message); }
inline void LogWarning(std::string_view message) { EmitLog(LogLevel::Warning, message); }
inline void LogError(std::string_view message) { EmitLog(LogLevel::Error, message); }

void ReportCudaError(std::string_view action, cudaError_t err);

// Structured logging for cache events
void LogCacheInitialized(const GRIMTS::CacheConfig& config);
void LogCacheShutdown();
void LogCacheReset();
void LogCacheResize(std::size_t old_capacity, std::size_t new_capacity, bool success);
void LogCacheFault(std::string_view reason);
void LogTelemetrySummary(const GRIMTS::GuessCacheTelemetry& telemetry);

} // namespace GRIMTS::Logging