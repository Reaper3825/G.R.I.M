#pragma once
//======================================================//
//  Startup/Rng.hpp
//  Production-grade RNG system with hierarchical seeding.
//
//  Extracted from Phase1_Startup so the orchestrator does
//  not own the implementation. Phase1_Startup.hpp pulls
//  this header in so TrainingContext can still embed
//  RNGContext by value.
//======================================================//

#include <cstdint>
#include <random>

// Forward decl: TrainingLogger lives in training_logger.hpp, included by
// Phase1_Startup.hpp before this header so a forward decl is safe.
class TrainingLogger;

// AiConfigSnapshot lives behind Shared/HyperParameters/HyperParameters_GPU.hpp.
// Forward-declare in its owning namespace to avoid pulling that header
// transitively (and to keep this header circular-include-safe).
namespace GRIM { namespace Config { struct AiConfigSnapshot; } }

namespace GRIMText::Training {

struct TrainingContext;

/**
 * @brief Production-grade RNG context with reproducibility support.
 *
 * Hierarchical seeding strategy (PyTorch/JAX-style):
 *   base_seed: master seed from config (user-specified or time-based)
 *   data_seed = base_seed + 0    : CPU RNG for data shuffling
 *   init_seed = base_seed + 1000 : weight initialization (Xavier)
 *   cuda_seed = base_seed + 2000 : GPU dropout/sampling
 *
 * Move-only: owns a curandGenerator_t (opaque void* here to keep curand.h
 * out of headers).
 */
struct RNGContext {
    uint64_t base_seed = 0;
    uint64_t data_seed = 0;
    uint64_t init_seed = 0;
    uint64_t cuda_seed = 0;

    std::mt19937_64 data_rng;

    // curandGenerator_t (void* to avoid curand.h here)
    void* cuda_rng_generator = nullptr;

    bool deterministic = false;
    bool cuda_rng_initialized = false;

    RNGContext() = default;
    ~RNGContext();

    RNGContext(const RNGContext&) = delete;
    RNGContext& operator=(const RNGContext&) = delete;

    RNGContext(RNGContext&& other) noexcept;
    RNGContext& operator=(RNGContext&& other) noexcept;
};

namespace Internal {

/**
 * @brief Initialize RNG with hierarchical seeding from config.
 *        Reads training.config.seed (-1 = nondeterministic).
 */
RNGContext initializeRNG(const ::GRIM::Config::AiConfigSnapshot& config,
                         TrainingLogger& logger);

} // namespace Internal

void RngReady(::GRIMText::Training::TrainingContext& ctx);

} // namespace GRIMText::Training
