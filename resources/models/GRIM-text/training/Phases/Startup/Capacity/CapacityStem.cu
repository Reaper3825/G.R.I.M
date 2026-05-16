#include "CapacityStem.hpp"

#include "../../Phase1_Startup.hpp"

namespace GRIMText::Training {

void HyperparametersReady(TrainingContext& ctx) {
    GRIM::HyperParameters::validateTrainingHyperparameters(ctx.config.hyperparameters);
    ctx.loss_config = GRIM::HyperParameters::lossConfigHP(ctx.config.hyperparameters);
    if (!ctx.loss_config.initialized) {
        throw std::runtime_error("FATAL: HyperparametersReady failed to initialize LossConfigHP");
    }
}

void CapacityStemReady(TrainingContext& ctx) {
    ctx.run_capacity = deriveRunCapacityOrThrow(ctx.config);
}

} // namespace GRIMText::Training

