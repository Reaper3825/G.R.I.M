#pragma once

#ifdef __CUDACC__
#include <cuda_runtime.h>
#else
struct CUstream_st;
using cudaStream_t = CUstream_st*;
#endif

#include <cstdint>
#include <vector>

namespace GRIM {
struct OptimizerState;
struct FeedForwardParameterTensors;
namespace Config {
struct AiConfigSnapshot;
}
namespace HyperParameters {
struct EncoderLayerConstructionHP;
}
}

namespace GRIMText::Training::Startup {
struct GpuModelState;
}

namespace ParameterRegistry {
struct StartupParameterRegistry;
}

namespace GRIMText::Training::Startup::ModelRegistration {

#ifdef USE_CUDA

// Phase-1 startup tensor registration boundary.
//
// Ownership contract:
// - TrainingContext::parameter_registry owns migrated writable parameter bundles
//   and the durable ParameterGroup inventory declared in ParameterRegistry.hpp.
// - This module discovers trainable tensors, writes non-owning ParameterGroup
//   metadata, and binds externally owned optimizer moment tensors.
// - OptimizerState owns Adam/RAdam moment tensor storage; ParameterGroup entries
//   only borrow those tensors after bindOptimizerState().
void initializeFeedForwardParameterTensors(
    std::vector<GRIM::FeedForwardParameterTensors>& feed_forward_parameter_tensors,
    const GRIM::HyperParameters::EncoderLayerConstructionHP& encoder_hp,
    std::uint64_t weight_init_seed,
    cudaStream_t init_stream);

void buildParameterGroups(const GRIM::Config::AiConfigSnapshot& config,
                          GRIMText::Training::Startup::GpuModelState& gpu_model_state,
                          ::ParameterRegistry::StartupParameterRegistry& parameter_registry);
void bindOptimizerState(::ParameterRegistry::StartupParameterRegistry& parameter_registry,
                        GRIM::OptimizerState& optimizer_state,
                        cudaStream_t stream);

#endif // USE_CUDA

} // namespace GRIMText::Training::Startup::ModelRegistration
