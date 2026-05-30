#pragma once

#ifdef __CUDACC__
#include <cuda_runtime.h>
#else
struct CUstream_st;
using cudaStream_t = CUstream_st*;
#endif

namespace GRIM {
class LanguageModel;
struct OptimizerState;
}

namespace GRIMText::Training::Startup {
struct GpuModelState;
}

namespace GRIMText::Training::Startup::ModelRegistration::ParameterRegistry {
struct StartupParameterRegistry;
}

namespace GRIMText::Training::Startup::ModelRegistration {

#ifdef USE_CUDA

// Phase-1 startup tensor registration boundary.
//
// Ownership contract:
// - TrainingContext::parameter_registry owns migrated writable parameter bundles
//   declared in ParameterRegistry.hpp.
// - GRIM::LanguageModel owns the durable ParameterGroup vector only.
// - This module discovers trainable tensors, writes non-owning ParameterGroup
//   metadata, and binds externally owned optimizer moment tensors.
// - OptimizerState owns Adam/RAdam moment tensor storage; ParameterGroup entries
//   only borrow those tensors after bindOptimizerState().
void buildParameterGroups(GRIM::LanguageModel& model,
                          GRIMText::Training::Startup::GpuModelState& gpu_model_state,
                          GRIMText::Training::Startup::ModelRegistration::ParameterRegistry::StartupParameterRegistry& parameter_registry);
void bindOptimizerState(GRIM::LanguageModel& model,
                        GRIM::OptimizerState& optimizer_state,
                        cudaStream_t stream);

#endif // USE_CUDA

} // namespace GRIMText::Training::Startup::ModelRegistration
