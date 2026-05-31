#pragma once

#include <string>

namespace GRIM {
class LanguageModel;
}

namespace GRIMText::Training::Startup {
struct GpuModelState;
}

namespace ParameterRegistry {
struct StartupParameterRegistry;
}

namespace GRIM {

struct TrainingState;

bool saveLanguageModelCheckpoint(
    LanguageModel& model,
    const GRIMText::Training::Startup::GpuModelState& gpu_model_state,
    const ::ParameterRegistry::StartupParameterRegistry& parameter_registry,
    const std::string& path);

bool loadLanguageModelCheckpoint(
    LanguageModel& model,
    const TrainingState& training_state,
    const GRIMText::Training::Startup::GpuModelState& gpu_model_state,
    ::ParameterRegistry::StartupParameterRegistry& parameter_registry,
    const std::string& path);

} // namespace GRIM
