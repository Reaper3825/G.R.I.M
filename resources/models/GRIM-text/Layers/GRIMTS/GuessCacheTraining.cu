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
//  GuessCacheScope RAII (Rule 22 Compliant)
//======================================================//

GuessCacheScope::GuessCacheScope(::GRIM::TrainingState& training_state,
                                 std::size_t capacity,
                                 bool enable_async)
    : training_state_(training_state), active_(false), buffers_allocated_(false) {

    // RULE 22: Allocate buffers through TrainingState
    const bool enable_diversity = true;
    const std::size_t diversity_bloom_bits = 65536;
    const std::size_t pinned_buffer_size = enable_async ? 8192 : 0;

    training_state_.allocateGuessCacheBuffers(
            capacity, enable_diversity, diversity_bloom_bits, pinned_buffer_size);
    buffers_allocated_ = true;

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

    // RULE 22: Get stream from centralized controller
    // getPrimaryStream() throws if not initialized (Rule 20)
    cudaStream_t primary_stream = training_state_.stream_ctrl.getPrimaryStream();

    // Convert TrainingState::GuessCacheBuffers to GRIMTS::GuessCacheBuffers
    GRIMTS::GuessCacheBuffers grimts_buffers{};
    grimts_buffers.records            = training_state_.guess_cache_buffers.records;
    grimts_buffers.keys               = training_state_.guess_cache_buffers.keys;
    grimts_buffers.size               = training_state_.guess_cache_buffers.size;
    grimts_buffers.evict_cursor       = training_state_.guess_cache_buffers.evict_cursor;
    grimts_buffers.diversity_bloom    = training_state_.guess_cache_buffers.diversity_bloom;
    grimts_buffers.bloom_words        = training_state_.guess_cache_buffers.bloom_words;
    grimts_buffers.calibration_offset = training_state_.guess_cache_buffers.calibration_offset;
    grimts_buffers.single_meta_buffer = training_state_.guess_cache_buffers.single_meta_buffer;
    grimts_buffers.single_reward_buffer = training_state_.guess_cache_buffers.single_reward_buffer;
    grimts_buffers.pinned_meta        = training_state_.guess_cache_buffers.pinned_meta;
    grimts_buffers.pinned_rewards     = training_state_.guess_cache_buffers.pinned_rewards;
    grimts_buffers.pinned_capacity    = training_state_.guess_cache_buffers.pinned_capacity;
    grimts_buffers.capacity           = training_state_.guess_cache_buffers.capacity;
    grimts_buffers.allocated          = training_state_.guess_cache_buffers.allocated;

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

    // Initialize GRIM-TS with pre-allocated buffers
    active_ = GRIMTS::InitializeGuessCache(config, grimts_buffers, primary_stream);
    if (active_) {
        GRIMTS::ResetGuessCache(primary_stream);
    } else {
        fprintf(stderr, "[ERROR] GuessCacheScope: GRIMTS::InitializeGuessCache failed!\n");
        training_state_.freeGuessCacheBuffers();
        buffers_allocated_ = false;
    }
}

GuessCacheScope::~GuessCacheScope() {
    if (active_) {
        GRIMTS::ShutdownGuessCache();
        active_ = false;
    }
    // Clear logging callbacks to avoid dangling references
    GRIMTS::Logging::ClearLogCallbacks();
    // RULE 22: TrainingState owns the buffers, we allocated them so we free them
    if (buffers_allocated_) {
        training_state_.freeGuessCacheBuffers();
        buffers_allocated_ = false;
    }
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

    if (!training_state.autograd_intermediates.hasLogits() ||
        training_state.cached_batch_size != static_cast<int>(guess_count) ||
        training_state.cached_seq_len <= 0) {
        EmitModuleWarning(ModuleId::GuessCache,
                          "Guess cache skipped: cached logits not ready for prediction-based pass",
                          global_step);
        return;
    }

    cudaStream_t stream = training_state.stream_ctrl.getPrimaryStream();
    const int batch_size     = static_cast<int>(guess_count);
    const int vocab_size     = payload.vocab_size;
    const int cached_seq_len = training_state.cached_seq_len;

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
        if (pos >= 0 && pos < cached_seq_len) {
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
            (static_cast<std::size_t>(i) * cached_seq_len + pos) * vocab_size;
        err = cudaMemcpyAsync(
            pred_logits.data() + static_cast<std::size_t>(i) * vocab_size,
            training_state.autograd_intermediates.logits_tensor.data + offset,
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
