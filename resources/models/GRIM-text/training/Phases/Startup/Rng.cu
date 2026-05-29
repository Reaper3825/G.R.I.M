//======================================================//
//  Startup/Rng.cu
//  RNGContext lifetime + initializeRNG implementation.
//  Extracted from Phase1_Startup.cu.
//======================================================//

#include "Rng.hpp"
#include "../Phase1_Startup.hpp"   // LanguageModelConfig, TrainingLogger include chain

#include <chrono>
#include <sstream>
#include <stdexcept>
#include <string>

#ifdef USE_CUDA
#include <curand.h>
#endif

namespace GRIMText::Training {

//======================================================//
//  RNGContext lifetime
//======================================================//

RNGContext::~RNGContext() {
#ifdef USE_CUDA
    if (cuda_rng_initialized && cuda_rng_generator) {
        curandDestroyGenerator(static_cast<curandGenerator_t>(cuda_rng_generator));
        cuda_rng_generator = nullptr;
    }
#endif
}

RNGContext::RNGContext(RNGContext&& other) noexcept
    : base_seed(other.base_seed)
    , data_seed(other.data_seed)
    , init_seed(other.init_seed)
    , cuda_seed(other.cuda_seed)
    , data_rng(std::move(other.data_rng))
    , cuda_rng_generator(other.cuda_rng_generator)
    , deterministic(other.deterministic)
    , cuda_rng_initialized(other.cuda_rng_initialized)
{
    other.cuda_rng_generator = nullptr;
    other.cuda_rng_initialized = false;
}

RNGContext& RNGContext::operator=(RNGContext&& other) noexcept {
    if (this != &other) {
#ifdef USE_CUDA
        if (cuda_rng_initialized && cuda_rng_generator) {
            curandDestroyGenerator(static_cast<curandGenerator_t>(cuda_rng_generator));
        }
#endif
        base_seed = other.base_seed;
        data_seed = other.data_seed;
        init_seed = other.init_seed;
        cuda_seed = other.cuda_seed;
        data_rng = std::move(other.data_rng);
        cuda_rng_generator = other.cuda_rng_generator;
        deterministic = other.deterministic;
        cuda_rng_initialized = other.cuda_rng_initialized;

        other.cuda_rng_generator = nullptr;
        other.cuda_rng_initialized = false;
    }
    return *this;
}

//======================================================//
//  initializeRNG
//======================================================//

namespace Internal {

RNGContext initializeRNG(
    const ::GRIM::Config::AiConfigSnapshot& config,
    TrainingLogger& logger) {
    RNGContext ctx;
    const auto seed_hp = GRIM::HyperParameters::trainingSeedHP(config);

    logger.log("Initializing production-grade RNG system...");

    // Seed already loaded from ai_config.json on the single root config
    int64_t config_seed = seed_hp.seed;

    if (config_seed < 0) {
        auto now = std::chrono::high_resolution_clock::now();
        ctx.base_seed = static_cast<uint64_t>(now.time_since_epoch().count());
        ctx.deterministic = false;
        logger.log("✓ RNG mode: NON-DETERMINISTIC (random seed from timestamp)");
    } else {
        ctx.base_seed = static_cast<uint64_t>(config_seed);
        ctx.deterministic = true;
        logger.log("✓ RNG mode: DETERMINISTIC (seed=" + std::to_string(ctx.base_seed) + ")");
    }

    // Hierarchical seeding (PyTorch/JAX pattern)
    ctx.data_seed = ctx.base_seed + 0;
    ctx.init_seed = ctx.base_seed + 1000;
    ctx.cuda_seed = ctx.base_seed + 2000;

    ctx.data_rng = std::mt19937_64(ctx.data_seed);

#ifdef USE_CUDA
    // Rule 20: throw on failure, no silent degradation
    curandGenerator_t cuda_gen;
    curandStatus_t status = curandCreateGenerator(&cuda_gen, CURAND_RNG_PSEUDO_DEFAULT);
    if (status != CURAND_STATUS_SUCCESS) {
        throw std::runtime_error("FATAL: Failed to create CUDA RNG generator (curandStatus=" +
                                 std::to_string(status) + "). Training requires controlled GPU randomness.");
    }
    status = curandSetPseudoRandomGeneratorSeed(cuda_gen, ctx.cuda_seed);
    if (status != CURAND_STATUS_SUCCESS) {
        curandDestroyGenerator(cuda_gen);
        throw std::runtime_error("FATAL: Failed to set CUDA RNG seed (curandStatus=" +
                                 std::to_string(status) + "). Training requires controlled GPU randomness.");
    }
    ctx.cuda_rng_generator = static_cast<void*>(cuda_gen);
    ctx.cuda_rng_initialized = true;
    logger.log("✓ CUDA RNG initialized (seed=" + std::to_string(ctx.cuda_seed) + ")");
#else
    ctx.cuda_rng_generator = nullptr;
    ctx.cuda_rng_initialized = false;
    logger.log("⚠ CUDA RNG not available (USE_CUDA not defined)");
#endif

    std::ostringstream seed_log;
    seed_log << "RNG seed hierarchy:\n"
             << "  base_seed  = " << ctx.base_seed << " (master)\n"
             << "  data_seed  = " << ctx.data_seed << " (CPU shuffling)\n"
             << "  init_seed  = " << ctx.init_seed << " (weight initialization)\n"
             << "  cuda_seed  = " << ctx.cuda_seed << " (GPU dropout/sampling)\n"
             << "To reproduce this run, set ai_config.json: training.config.seed = " << ctx.base_seed;
    logger.log(seed_log.str());

    logger.log("✓ RNG system initialized");
    return ctx;
}

} // namespace Internal
} // namespace GRIMText::Training
