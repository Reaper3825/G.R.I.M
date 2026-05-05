//======================================================//
//  GuessCacheTraining.cu
//  GRIM-TS Guess Cache — Training Loop Integration
//======================================================//

#include "GuessCacheTraining.hpp"

#include "../../Shared/LogRecorder/LogRecorder.hpp"
#include "../../Shared/CudaAllocUtils.hpp"
#include "../../Shared/TrainingState/TrainingState_GPU.hpp"
#include "../../Shared/Batching/BatchPayload.hpp"

#include <vector>
#include <cmath>
#include <limits>
#include <sstream>
#include <cstdio>

using GRIM::CudaAlloc::cudaMallocOrThrow;
using GRIM::Logging::ModuleId;
using GRIM::Logging::EmitModuleInfo;
using GRIM::Logging::EmitModuleWarning;
using GRIM::Logging::EmitModuleError;

namespace GRIMTS::Training {

//======================================================//
//  GuessCacheScope RAII (Rule 20 ownership boundary)
//======================================================//

GuessCacheScope::OwnedBuffers::~OwnedBuffers() {
    release();
}

void GuessCacheScope::OwnedBuffers::release() {
    if (buffers.records) { cudaFree(buffers.records); buffers.records = nullptr; }
    if (buffers.keys) { cudaFree(buffers.keys); buffers.keys = nullptr; }
    if (buffers.size) { cudaFree(buffers.size); buffers.size = nullptr; }
    if (buffers.evict_cursor) { cudaFree(buffers.evict_cursor); buffers.evict_cursor = nullptr; }
    if (buffers.diversity_bloom) { cudaFree(buffers.diversity_bloom); buffers.diversity_bloom = nullptr; }
    if (buffers.calibration_offset) { cudaFree(buffers.calibration_offset); buffers.calibration_offset = nullptr; }
    if (buffers.slot_locks) { cudaFree(buffers.slot_locks); buffers.slot_locks = nullptr; }
    if (buffers.single_meta_buffer) { cudaFree(buffers.single_meta_buffer); buffers.single_meta_buffer = nullptr; }
    if (buffers.single_reward_buffer) { cudaFree(buffers.single_reward_buffer); buffers.single_reward_buffer = nullptr; }
    if (buffers.pinned_meta) { cudaFreeHost(buffers.pinned_meta); buffers.pinned_meta = nullptr; }
    if (buffers.pinned_rewards) { cudaFreeHost(buffers.pinned_rewards); buffers.pinned_rewards = nullptr; }

    buffers.bloom_words = 0;
    buffers.pinned_capacity = 0;
    buffers.capacity = 0;
    buffers.allocated = false;
}

void GuessCacheScope::OwnedBuffers::allocate(
    std::size_t capacity,
    bool enable_diversity,
    std::size_t diversity_bloom_bits,
    std::size_t pinned_buffer_size,
    cudaStream_t primary_stream) {

    if (buffers.allocated) {
        throw std::runtime_error("[GuessCacheScope::OwnedBuffers::allocate] buffers already allocated");
    }
    if (capacity == 0) {
        throw std::runtime_error("[GuessCacheScope::OwnedBuffers::allocate] capacity cannot be zero");
    }
    if (!primary_stream) {
        throw std::runtime_error("[GuessCacheScope::OwnedBuffers::allocate] primary_stream is nullptr");
    }

    cudaMallocOrThrow(&buffers.records, capacity * sizeof(GRIMTS::GuessRecord), "guess_cache_records");
    cudaMallocOrThrow(reinterpret_cast<void**>(&buffers.keys), capacity * sizeof(std::uint64_t), "guess_cache_keys");
    cudaMallocOrThrow(reinterpret_cast<void**>(&buffers.size), sizeof(unsigned int), "guess_cache_size");
    cudaMallocOrThrow(reinterpret_cast<void**>(&buffers.evict_cursor), sizeof(unsigned int), "guess_cache_evict_cursor");

    if (enable_diversity && diversity_bloom_bits > 0) {
        buffers.bloom_words = (diversity_bloom_bits + 31) / 32;
        cudaMallocOrThrow(reinterpret_cast<void**>(&buffers.diversity_bloom),
                          buffers.bloom_words * sizeof(std::uint32_t),
                          "guess_cache_diversity_bloom");
    }

    cudaMallocOrThrow(reinterpret_cast<void**>(&buffers.calibration_offset), sizeof(float), "guess_cache_calibration_offset");
    cudaMallocOrThrow(&buffers.single_meta_buffer, sizeof(GRIMTS::GuessMetadata), "guess_cache_single_meta");
    cudaMallocOrThrow(reinterpret_cast<void**>(&buffers.single_reward_buffer), sizeof(float), "guess_cache_single_reward");

    if (pinned_buffer_size > 0) {
        cudaError_t err = cudaMallocHost(&buffers.pinned_meta, pinned_buffer_size * sizeof(GRIMTS::GuessMetadata));
        if (err != cudaSuccess) {
            release();
            throw std::runtime_error(std::string("[GuessCacheScope::OwnedBuffers::allocate] cudaMallocHost pinned_meta failed: ") +
                                     cudaGetErrorString(err));
        }

        err = cudaMallocHost(&buffers.pinned_rewards, pinned_buffer_size * sizeof(float));
        if (err != cudaSuccess) {
            release();
            throw std::runtime_error(std::string("[GuessCacheScope::OwnedBuffers::allocate] cudaMallocHost pinned_rewards failed: ") +
                                     cudaGetErrorString(err));
        }
        buffers.pinned_capacity = pinned_buffer_size;
    }

    cudaMemsetAsync(buffers.size, 0, sizeof(unsigned int), primary_stream);
    cudaMemsetAsync(buffers.keys, 0xFF, capacity * sizeof(std::uint64_t), primary_stream);
    cudaMemsetAsync(buffers.records, 0, capacity * sizeof(GRIMTS::GuessRecord), primary_stream);
    cudaMemsetAsync(buffers.evict_cursor, 0, sizeof(unsigned int), primary_stream);
    if (buffers.diversity_bloom) {
        cudaMemsetAsync(buffers.diversity_bloom, 0, buffers.bloom_words * sizeof(std::uint32_t), primary_stream);
    }
    float zero_cal = 0.0f;
    cudaMemcpyAsync(buffers.calibration_offset, &zero_cal, sizeof(float), cudaMemcpyHostToDevice, primary_stream);

    buffers.capacity = capacity;
    buffers.allocated = true;

    fprintf(stdout, "[INFO] GuessCacheScope: buffers allocated. capacity=%zu, diversity=%s, bloom_bits=%zu, pinned=%zu\n",
            capacity, enable_diversity ? "ON" : "OFF", diversity_bloom_bits, pinned_buffer_size);
}

GuessCacheScope::GuessCacheScope(::GRIM::TrainingState& training_state,
                                 std::size_t capacity,
                                 bool enable_async)
    : training_state_(training_state), active_(false) {

    const bool enable_diversity = true;
    const std::size_t diversity_bloom_bits = 65536;
    const std::size_t pinned_buffer_size = enable_async ? 8192 : 0;

    // getPrimaryStream() throws if not initialized (Rule 20)
    cudaStream_t primary_stream = training_state_.stream_ctrl.getPrimaryStream();
    buffers_.allocate(capacity, enable_diversity, diversity_bloom_bits, pinned_buffer_size, primary_stream);

    // Build GRIMTS config
    GRIMTS::CacheConfig config{};
    config.initial_capacity = capacity;
    config.min_capacity = 4096;
    config.max_capacity = 262144;
    config.grow_threshold = 0.85f;
    config.shrink_threshold = 0.25f;
    config.evict_window = 32;
    config.enable_diversity_tracking = enable_diversity;
    config.diversity_bloom_bits = static_cast<int>(diversity_bloom_bits);
    config.enable_async_transfers = enable_async;
    config.pinned_buffer_size = pinned_buffer_size;
    config.enable_histograms = false;

    // Wire up GRIMTS logging to training log system
    GRIMTS::Logging::RegisterLogCallback([](GRIMTS::Logging::LogLevel level, std::string_view message) {
        switch (level) {
            case GRIMTS::Logging::LogLevel::Error:
                EmitModuleError(ModuleId::GuessCache, std::string(message), 0);
                break;
            case GRIMTS::Logging::LogLevel::Warning:
                EmitModuleWarning(ModuleId::GuessCache, std::string(message), 0);
                break;
            case GRIMTS::Logging::LogLevel::Info:
            case GRIMTS::Logging::LogLevel::Debug:
            default:
                EmitModuleInfo(ModuleId::GuessCache, std::string(message), 0);
                break;
        }
    });

    // Initialize GRIM-TS with scope-owned buffers
    active_ = GRIMTS::InitializeGuessCache(config, buffers_.buffers, primary_stream);
    if (active_) {
        GRIMTS::ResetGuessCache(primary_stream);
    } else {
        fprintf(stderr, "[ERROR] GuessCacheScope: GRIMTS::InitializeGuessCache failed!\n");
        buffers_.release();
    }
}

GuessCacheScope::~GuessCacheScope() {
    if (active_) {
        GRIMTS::ShutdownGuessCache();
        active_ = false;
    }
    // Clear logging callbacks to avoid dangling references
    GRIMTS::Logging::ClearLogCallbacks();
    buffers_.release();
}

//======================================================//
//  GuessCacheBatchBuffers
//======================================================//

GuessCacheBatchBuffers::~GuessCacheBatchBuffers() {
    release();
}

void GuessCacheBatchBuffers::release() {
    if (device_metadata_) { cudaFree(device_metadata_); device_metadata_ = nullptr; }
    if (device_rewards_)  { cudaFree(device_rewards_);  device_rewards_  = nullptr; }
    if (device_stats_)    { cudaFree(device_stats_);    device_stats_    = nullptr; }
    capacity_ = 0;
}

cudaError_t GuessCacheBatchBuffers::ensure(std::size_t capacity) {
    if (capacity == 0)          return cudaSuccess;
    if (capacity <= capacity_)  return cudaSuccess;

    release();
    capacity_ = capacity;

    cudaMallocOrThrow(reinterpret_cast<void**>(&device_metadata_), capacity * sizeof(GRIMTS::GuessMetadata),    "guess_cache_metadata");
    cudaMallocOrThrow(reinterpret_cast<void**>(&device_rewards_),  capacity * sizeof(float),                    "guess_cache_rewards");
    cudaMallocOrThrow(reinterpret_cast<void**>(&device_stats_),    capacity * sizeof(GRIMTS::GuessRewardStats), "guess_cache_stats");

    return cudaSuccess;
}

//======================================================//
//  initGuessCache
//======================================================//

std::unique_ptr<GuessCacheScope> initGuessCache(
    ::GRIM::TrainingState& training_state,
    bool guess_aux_enabled,
    bool single_stream_mode,
    int global_step,
    GuessCacheState& gc_state) {

    std::unique_ptr<GuessCacheScope> scope;

    if (!guess_aux_enabled) {
        gc_state.guess_cache_ready = false;
        EmitModuleInfo(ModuleId::GuessCache, "Guess cache disabled (guess_aux.enabled=false)", global_step);
        return scope;
    }

    scope = std::make_unique<GuessCacheScope>(
        training_state,
        kDefaultGuessCacheCapacity,
        !single_stream_mode);
    gc_state.guess_cache_ready = scope->active();

    if (gc_state.guess_cache_ready) {
        EmitModuleInfo(ModuleId::GuessCache,
            std::string("GPU cache ready (capacity=") + std::to_string(kDefaultGuessCacheCapacity) + ")",
            global_step);
        gc_state.batch_buffers = std::make_unique<GuessCacheBatchBuffers>();
    }

    return scope;
}

//======================================================//
//  resetGuessCacheForEpoch
//======================================================//

void resetGuessCacheForEpoch(
    ::GRIM::TrainingState& training_state,
    const GuessCacheState& gc_state) {

    if (!gc_state.guess_cache_ready || gc_state.guess_cache_faulted)
        return;

    // BUG FIX: Must pass stream — nullptr causes cudaMemsetAsync crash!
    cudaStream_t primary_stream = training_state.stream_ctrl.getPrimaryStream();
    GRIMTS::ResetGuessCache(primary_stream);
}

//======================================================//
//  updateGuessCacheFromBatch
//======================================================//

void updateGuessCacheFromBatch(
    ::GRIM::TrainingState& training_state,
    const ::GRIM::Batching::BatchPayload& payload,
    float batch_loss,
    int epoch_idx,
    int global_step,
    GuessCacheState& gc_state) {

    if (!gc_state.guess_cache_ready || gc_state.guess_cache_faulted)
        return;
    if (!std::isfinite(batch_loss))
        return;

    if (!gc_state.batch_buffers) {
        gc_state.batch_buffers = std::make_unique<GuessCacheBatchBuffers>();
    }

    const std::size_t guess_count = static_cast<std::size_t>(payload.batch_size);
    if (guess_count == 0)
        return;

    auto& buffers = *gc_state.batch_buffers;

    if (!training_state.stream_ctrl.isInitialized()) {
        gc_state.guess_cache_faulted = true;
        EmitModuleError(ModuleId::GuessCache,
                        "Guess cache update failed: stream controller not initialized",
                        global_step);
        return;
    }

    if (!training_state.cached_logits_tensor.data ||
        payload.max_seq_len <= 0) {
        EmitModuleWarning(ModuleId::GuessCache,
                          "Guess cache skipped: cached logits not ready for prediction-based pass",
                          global_step);
        return;
    }

    cudaStream_t stream = training_state.stream_ctrl.getPrimaryStream();
    const int batch_size     = static_cast<int>(guess_count);
    const int vocab_size     = payload.vocab_size;
    const int payload_seq_len = payload.max_seq_len;

    // --- Find last valid target position per sequence ---
    std::vector<int> guess_positions(batch_size, -1);
    for (int i = 0; i < batch_size; ++i) {
        const int seq_len = payload.seq_lengths[i];
        if (seq_len <= 0) continue;
        const int flat_start = i * payload.max_seq_len;
        int pos = seq_len - 1;
        while (pos >= 0 && payload.target_ids[flat_start + pos] < 0) {
            --pos;
        }
        if (pos >= 0 && pos < payload_seq_len) {
            guess_positions[i] = pos;
        }
    }

    // --- Copy prediction logits from device ---
    std::vector<float> pred_logits(static_cast<std::size_t>(batch_size) * vocab_size);
    cudaError_t err = cudaSuccess;
    for (int i = 0; i < batch_size; ++i) {
        const int pos = guess_positions[i];
        if (pos < 0) continue;
        const std::size_t offset =
            (static_cast<std::size_t>(i) * payload_seq_len + pos) * vocab_size;
        err = cudaMemcpyAsync(
            pred_logits.data() + static_cast<std::size_t>(i) * vocab_size,
            training_state.cached_logits_tensor.data + offset,
            static_cast<std::size_t>(vocab_size) * sizeof(float),
            cudaMemcpyDeviceToHost, stream);
        if (err != cudaSuccess) break;
    }
    if (err == cudaSuccess) {
        err = cudaStreamSynchronize(stream);
    }
    if (err != cudaSuccess) {
        gc_state.guess_cache_faulted = true;
        EmitModuleError(ModuleId::GuessCache,
                        std::string("Guess cache logit sync failed: ") + cudaGetErrorString(err),
                        global_step);
        return;
    }

    // --- Build prediction metadata + reward vectors ---
    std::vector<GRIMTS::GuessMetadata> pred_metadata;
    std::vector<GRIMTS::GuessMetadata> reward_metadata;
    std::vector<float> rewards;
    pred_metadata.reserve(guess_count);
    reward_metadata.reserve(guess_count);
    rewards.reserve(guess_count);

    const float reward = 1.0f / (1.0f + batch_loss);
    for (int i = 0; i < batch_size; ++i) {
        const int pos = guess_positions[i];
        if (pos < 0) continue;
        const int flat_start_i = i * payload.max_seq_len;
        const int target_token = payload.target_ids[flat_start_i + pos];
        if (target_token < 0 || target_token >= vocab_size) continue;

        const float* logits = pred_logits.data() + static_cast<std::size_t>(i) * vocab_size;
        float top1 = -std::numeric_limits<float>::infinity();
        float top2 = -std::numeric_limits<float>::infinity();
        int pred_token = 0;
        for (int v = 0; v < vocab_size; ++v) {
            const float logit = logits[v];
            if (logit > top1) {
                top2 = top1;
                top1 = logit;
                pred_token = v;
            } else if (logit > top2) {
                top2 = logit;
            }
        }

        const float margin = top1 - top2;
        const float confidence = 1.0f / (1.0f + std::exp(-margin));
        const float clamped_confidence = std::min(1.0f, std::max(0.0f, confidence));
        const std::uint64_t prompt_hash = GRIMTS::HashSignature(
            payload.input_ids.data() + flat_start_i,
            payload.seq_lengths[i] * sizeof(int));

        GRIMTS::GuessMetadata pred_meta{};
        pred_meta.prompt_hash     = prompt_hash;
        pred_meta.guess_hash      = GRIMTS::HashSignature(&pred_token, sizeof(int));
        pred_meta.confidence      = clamped_confidence;
        pred_meta.sequence_length = static_cast<std::uint16_t>(
            std::min<std::size_t>(payload.seq_lengths[i], std::numeric_limits<std::uint16_t>::max()));
        pred_meta.prompt_length   = static_cast<std::uint16_t>(
            std::min<std::size_t>(payload.seq_lengths[i], std::numeric_limits<std::uint16_t>::max()));
        pred_meta.epoch           = static_cast<std::uint32_t>(epoch_idx + 1);
        pred_metadata.push_back(pred_meta);

        GRIMTS::GuessMetadata reward_meta = pred_meta;
        reward_meta.guess_hash = GRIMTS::HashSignature(&target_token, sizeof(int));
        reward_metadata.push_back(reward_meta);
        rewards.push_back(reward);
    }

    if (pred_metadata.empty()) return;

    // --- Upload predictions to device cache ---
    err = buffers.ensure(pred_metadata.size());
    if (err != cudaSuccess) {
        gc_state.guess_cache_faulted = true;
        EmitModuleError(ModuleId::GuessCache,
                        std::string("Guess cache buffer allocation failed: ") + cudaGetErrorString(err),
                        global_step);
        return;
    }

    err = cudaMemcpyAsync(buffers.metadata(), pred_metadata.data(),
                          pred_metadata.size() * sizeof(GRIMTS::GuessMetadata),
                          cudaMemcpyHostToDevice, stream);
    if (err == cudaSuccess) {
        err = GRIMTS::CacheGuessBatchGPU(buffers.metadata(), pred_metadata.size(), stream);
    }
    if (err != cudaSuccess) {
        gc_state.guess_cache_faulted = true;
        EmitModuleError(ModuleId::GuessCache,
                        std::string("Guess cache insert failed: ") + cudaGetErrorString(err),
                        global_step);
        return;
    }

    // --- Upload rewards and apply ---
    err = cudaMemcpyAsync(buffers.metadata(), reward_metadata.data(),
                          reward_metadata.size() * sizeof(GRIMTS::GuessMetadata),
                          cudaMemcpyHostToDevice, stream);
    if (err == cudaSuccess) {
        err = cudaMemcpyAsync(buffers.rewards(), rewards.data(),
                              rewards.size() * sizeof(float),
                              cudaMemcpyHostToDevice, stream);
    }
    if (err != cudaSuccess) {
        gc_state.guess_cache_faulted = true;
        EmitModuleError(ModuleId::GuessCache,
                        std::string("Guess cache H2D copy failed: ") + cudaGetErrorString(err),
                        global_step);
        return;
    }

    err = GRIMTS::ApplyRewardBatchGPU(
        buffers.metadata(),
        buffers.rewards(),
        reward_metadata.size(),
        kGuessRewardMomentum,
        buffers.stats(),
        stream);
    if (err != cudaSuccess) {
        gc_state.guess_cache_faulted = true;
        EmitModuleError(ModuleId::GuessCache,
                        std::string("Guess cache reward update failed: ") + cudaGetErrorString(err),
                        global_step);
    }
}

//======================================================//
//  logGuessCacheTelemetry
//======================================================//

void logGuessCacheTelemetry(
    const GuessCacheState& gc_state,
    int global_step) {

    if (!gc_state.guess_cache_ready || gc_state.guess_cache_faulted)
        return;

    const GRIMTS::GuessCacheTelemetry telemetry = GRIMTS::GetCacheTelemetry(false);
    std::ostringstream cache_msg;
    cache_msg << "Telemetry: fill=" << (telemetry.fill_ratio * 100.0f) << "%"
              << " records=" << telemetry.total_records
              << " hits=" << telemetry.trends.total_hits
              << " misses=" << telemetry.trends.total_misses
              << " health=" << (telemetry.is_healthy ? "OK" : "DEGRADED");
    EmitModuleInfo(ModuleId::GuessCache, cache_msg.str(), global_step);
}

} // namespace GRIMTS::Training
