#pragma once

namespace GRIMText::Training {

struct TrainingContext;

struct StartupValidationInputs {
    const TrainingContext& ctx;
};

void validateStartupOrThrow(const StartupValidationInputs& inputs);
void StartupValidated(TrainingContext& ctx);

} // namespace GRIMText::Training

