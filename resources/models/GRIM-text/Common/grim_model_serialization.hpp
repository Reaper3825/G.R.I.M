#pragma once

#include <string>

namespace GRIM {
class LanguageModel;
}

namespace GRIMText::Training::Startup {
struct GpuModelState;
}

namespace GRIMText::Training::Startup::ModelRegistration::ParameterRegistry {
struct StartupParameterRegistry;
}

namespace GRIM {

bool saveLanguageModelCheckpoint(
    LanguageModel& model,
    const GRIMText::Training::Startup::GpuModelState& gpu_model_state,
    const GRIMText::Training::Startup::ModelRegistration::ParameterRegistry::StartupParameterRegistry& parameter_registry,
    const std::string& path);

bool loadLanguageModelCheckpoint(
    LanguageModel& model,
    const GRIMText::Training::Startup::GpuModelState& gpu_model_state,
    GRIMText::Training::Startup::ModelRegistration::ParameterRegistry::StartupParameterRegistry& parameter_registry,
    const std::string& path);

} // namespace GRIM
