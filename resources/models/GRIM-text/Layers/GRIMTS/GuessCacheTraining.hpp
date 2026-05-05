#pragma once
//======================================================//
//  GuessCacheTraining.hpp
//  GRIM-TS Guess Cache — Training Loop Integration
//======================================================//
//
//  PURPOSE
//  =======
//  Encapsulates all guess cache lifecycle management and
//  per-batch update logic that was previously inlined in
//  Phase2_TrainingLoop.  The training loop calls into these
//  thin entry points; all CUDA/GRIMTS detail stays here.
//
//  Author: Austin Wadkins
//  Date: April 2025
//======================================================//

#include "GRIM-TS.hpp"
#include "../../GRIM/grim_language_model_cuda.hpp"

#include <memory>
#include <cstddef>
#include <cuda_runtime.h>

// Forward declarations — avoid pulling in heavy headers
namespace GRIM {
    struct TrainingState;
}
namespace GRIM::Batching {
    struct BatchPayload;
}

namespace GRIMTS::Training {

//----------------------------------------------------------------------
// Constants
//----------------------------------------------------------------------
constexpr std::size_t kDefaultGuessCacheCapacity = 16384;
constexpr float       kGuessRewardMomentum       = 0.85f;

//----------------------------------------------------------------------
// GuessCacheScope — RAII lifecycle (Rule 20 ownership boundary)
//----------------------------------------------------------------------
// GuessCacheScope owns the long-lived GRIM-TS cache buffers. TrainingState is
// borrowed only for the primary stream and model snapshots consumed by updates.

class GuessCacheScope {
public:
    explicit GuessCacheScope(::GRIM::TrainingState& training_state,
                             std::size_t capacity,
                             bool enable_async = true);
    ~GuessCacheScope();

    GuessCacheScope(const GuessCacheScope&) = delete;
    GuessCacheScope& operator=(const GuessCacheScope&) = delete;

    bool active() const { return active_; }

private:
    struct OwnedBuffers {
        GRIMTS::GuessCacheBuffers buffers{};

        OwnedBuffers() = default;
        ~OwnedBuffers();
        OwnedBuffers(const OwnedBuffers&) = delete;
        OwnedBuffers& operator=(const OwnedBuffers&) = delete;
        OwnedBuffers(OwnedBuffers&&) = delete;
        OwnedBuffers& operator=(OwnedBuffers&&) = delete;

        void allocate(std::size_t capacity,
                      bool enable_diversity,
                      std::size_t diversity_bloom_bits,
                      std::size_t pinned_buffer_size,
                      cudaStream_t primary_stream);
        void release();
    };

    ::GRIM::TrainingState& training_state_;
    OwnedBuffers buffers_;
    bool active_ = false;
};

//----------------------------------------------------------------------
// GuessCacheBatchBuffers — per-batch device scratch
//----------------------------------------------------------------------

class GuessCacheBatchBuffers {
public:
    GuessCacheBatchBuffers() = default;
    ~GuessCacheBatchBuffers();

    cudaError_t ensure(std::size_t capacity);

    GRIMTS::GuessMetadata*    metadata() const { return device_metadata_; }
    float*                    rewards()  const { return device_rewards_; }
    GRIMTS::GuessRewardStats* stats()    const { return device_stats_; }

private:
    void release();

    GRIMTS::GuessMetadata*    device_metadata_ = nullptr;
    float*                    device_rewards_   = nullptr;
    GRIMTS::GuessRewardStats* device_stats_     = nullptr;
    std::size_t               capacity_         = 0;
};

//----------------------------------------------------------------------
// Guess Cache state carried on TrainingLoopState
//----------------------------------------------------------------------

struct GuessCacheState {
    bool guess_cache_ready   = false;
    bool guess_cache_faulted = false;
    std::unique_ptr<GuessCacheBatchBuffers> batch_buffers;
};

//----------------------------------------------------------------------
// High-level entry points called from Phase2
//----------------------------------------------------------------------

/// Initialise the guess cache scope + batch buffers at the start of
/// training.  Returns the scope (caller keeps it alive for RAII
/// teardown) and populates `gc_state`.
std::unique_ptr<GuessCacheScope> initGuessCache(
    ::GRIM::TrainingState& training_state,
    bool guess_aux_enabled,
    bool single_stream_mode,
    int global_step,
    GuessCacheState& gc_state);

/// Reset the guess cache at the start of each epoch.
void resetGuessCacheForEpoch(
    ::GRIM::TrainingState& training_state,
    const GuessCacheState& gc_state);

/// After a successful forward+backward, extract logits and push
/// predictions + rewards into the guess cache.
void updateGuessCacheFromBatch(
    ::GRIM::TrainingState& training_state,
    const ::GRIM::Batching::BatchPayload& payload,
    float batch_loss,
    int epoch_idx,
    int global_step,
    GuessCacheState& gc_state);

/// Emit periodic telemetry for the guess cache (called at log intervals).
void logGuessCacheTelemetry(
    const GuessCacheState& gc_state,
    int global_step);

} // namespace GRIMTS::Training
