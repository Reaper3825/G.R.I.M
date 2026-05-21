#include "CapacityStem.hpp"

#include "../../Phase1_Startup.hpp"

namespace GRIMText::Training {

void HyperparametersReady(TrainingContext& ctx) {
    GRIM::HyperParameters::validateTrainingHyperparameters(ctx.config.hyperparameters);
    // Validate the loss grouping against the authoritative hyperparameters
    // without storing a second config owner on TrainingContext.
    static_cast<void>(GRIM::HyperParameters::lossConfigHP(ctx.config.hyperparameters));
    static_cast<void>(GRIM::HyperParameters::trainingFixedShapeHP(ctx.config));
}

} // namespace GRIMText::Training

