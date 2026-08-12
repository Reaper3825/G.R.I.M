#pragma once

#include <cstdint>
#include <optional>
#include <string>

#include "../Shared/Curriculum/CurriculumMetadata.hpp"

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

struct LatestCurriculumCompletionRecord {
    CurriculumMetadata curriculum;
    std::uint64_t epochs_completed = 0;
};

// Read only the optional latest-curriculum training provenance from a registry
// checkpoint. Returns std::nullopt for checkpoints that predate the metadata.
std::optional<LatestCurriculumCompletionRecord>
readLatestCurriculumCompletion(const std::string& path);

// Strict, registry-driven model-parameter checkpoint boundary. The registry is
// the complete tensor inventory; config contributes only parameterless model
// semantics that must match at load time.
bool saveParameterCheckpoint(
    const Config::AiConfigSnapshot& config,
    const ::ParameterRegistry::StartupParameterRegistry& parameter_registry,
    cudaStream_t stream,
    const std::string& path,
    const std::optional<LatestCurriculumCompletionRecord>& latest_curriculum_completion = std::nullopt);

bool loadParameterCheckpoint(
    const Config::AiConfigSnapshot& config,
    ::ParameterRegistry::StartupParameterRegistry& parameter_registry,
    cudaStream_t stream,
    const std::string& path);

} // namespace GRIM::Checkpoint
