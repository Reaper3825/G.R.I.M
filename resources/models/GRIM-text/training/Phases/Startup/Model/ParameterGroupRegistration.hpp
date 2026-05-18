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

namespace GRIM::HyperParameters {
struct ParameterRegistrationHP;
}

namespace GRIMText::Training::Startup::ModelRegistration {

#ifdef USE_CUDA

// Phase-1 startup tensor registration boundary.
//
// Ownership contract:
// - GRIM::LanguageModel owns the durable ParameterGroup vector and parameter tensors.
// - This module only discovers trainable tensors, writes non-owning ParameterGroup
//   metadata, and binds externally owned optimizer moment tensors.
// - OptimizerState owns Adam/RAdam moment tensor storage; ParameterGroup entries
//   only borrow those tensors after bindOptimizerState().
void buildParameterGroups(GRIM::LanguageModel& model,
                          const GRIM::HyperParameters::ParameterRegistrationHP& hp);
void buildParameterGroups(GRIM::LanguageModel& model);
void bindOptimizerState(GRIM::LanguageModel& model,
                        GRIM::OptimizerState& optimizer_state,
                        cudaStream_t stream);

#endif // USE_CUDA

} // namespace GRIMText::Training::Startup::ModelRegistration
