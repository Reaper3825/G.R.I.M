#pragma once

#include <string>

#ifdef __CUDACC__
#include <cuda_runtime.h>
#else
struct CUstream_st;
using cudaStream_t = CUstream_st*;
#endif

namespace GRIM::Config {
struct AiConfigSnapshot;
}

namespace ParameterRegistry {
struct StartupParameterRegistry;
}

namespace GRIM::Checkpoint {

// Strict, registry-driven model-parameter checkpoint boundary. The registry is
// the complete tensor inventory; config contributes only parameterless model
// semantics that must match at load time.
bool saveParameterCheckpoint(
    const Config::AiConfigSnapshot& config,
    const ::ParameterRegistry::StartupParameterRegistry& parameter_registry,
    cudaStream_t stream,
    const std::string& path);

bool loadParameterCheckpoint(
    const Config::AiConfigSnapshot& config,
    ::ParameterRegistry::StartupParameterRegistry& parameter_registry,
    cudaStream_t stream,
    const std::string& path);

} // namespace GRIM::Checkpoint
