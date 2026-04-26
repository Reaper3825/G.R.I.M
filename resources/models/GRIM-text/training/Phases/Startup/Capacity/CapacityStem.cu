#include "CapacityStem.hpp"

#include "../../Phase1_Startup.hpp"

namespace GRIMText::Training {

void HyperparametersReady(TrainingContext& ctx) {
    GRIM::HyperParameters::validateTrainingHyperparameters(ctx.config.hyperparameters);
}

void CapacityStemReady(TrainingContext& ctx) {
    ctx.run_capacity = deriveRunCapacityOrThrow(ctx.config);
}

} // namespace GRIMText::Training

