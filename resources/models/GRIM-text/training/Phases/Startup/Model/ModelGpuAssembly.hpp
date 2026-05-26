#pragma once

#include <cstdint>

namespace GRIM {
class LanguageModel;
}

namespace GRIMText::Training::Startup {

#ifdef USE_CUDA

void initializeCuBLASHandle(::GRIM::LanguageModel& model);
void initializePBM(::GRIM::LanguageModel& model);
void assembleGpuModel(::GRIM::LanguageModel& model, std::uint64_t weight_init_seed);
void initializeTrainingRuntime(::GRIM::LanguageModel& model);
void initializeInferenceRuntime(::GRIM::LanguageModel& model);

#endif // USE_CUDA

} // namespace GRIMText::Training::Startup
