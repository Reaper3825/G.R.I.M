#pragma once

#include <cstdint>

#include "ModelGpuState.hpp"

namespace GRIM {
class LanguageModel;
}

namespace GRIMText::Training::Startup {

#ifdef USE_CUDA

void initializeCuBLASHandle(::GRIM::LanguageModel& model);
void initializePBM(::GRIM::LanguageModel& model);
void assembleGpuModel(::GRIM::LanguageModel& model, GpuModelState& gpu_model_state, std::uint64_t weight_init_seed);
void initializeTrainingRuntime(::GRIM::LanguageModel& model);
void initializeInferenceRuntime(::GRIM::LanguageModel& model, const GpuModelState& gpu_model_state);

#endif // USE_CUDA

} // namespace GRIMText::Training::Startup
