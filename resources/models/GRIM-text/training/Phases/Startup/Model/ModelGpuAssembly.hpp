#pragma once

#include <cstdint>

#include "ModelGpuState.hpp"

namespace GRIM {
namespace Config {
struct AiConfigSnapshot;
}
struct TrainingState;
struct GenerationState;
namespace PBM {
class PBMStateOwner;
}
}

namespace ParameterRegistry {
struct StartupParameterRegistry;
}

namespace GRIMText::Training::Startup {

#ifdef USE_CUDA

void initializeCuBLASHandle(::GRIM::TrainingState& training_state);
void initializePBM(const ::GRIM::Config::AiConfigSnapshot& model_cfg, ::GRIM::TrainingState& training_state, ::GRIM::PBM::PBMStateOwner& pbm_owner);
void assembleGpuModel(const ::GRIM::Config::AiConfigSnapshot& model_cfg, ::GRIM::TrainingState& training_state, const ::GRIM::PBM::PBMStateOwner& pbm_owner, GpuModelState& gpu_model_state, ::ParameterRegistry::StartupParameterRegistry& parameter_registry, std::uint64_t weight_init_seed);
void initializeTrainingRuntime(::GRIM::TrainingState& training_state, const ::GRIM::PBM::PBMStateOwner& pbm_owner);
void initializeInferenceRuntime(const ::GRIM::Config::AiConfigSnapshot& model_cfg, ::GRIM::TrainingState& training_state, ::GRIM::GenerationState& generation_state, const ::GRIM::PBM::PBMStateOwner& pbm_owner, const GpuModelState& gpu_model_state, const ::ParameterRegistry::StartupParameterRegistry& parameter_registry);

#endif // USE_CUDA

} // namespace GRIMText::Training::Startup
