#include "CapacityStem.hpp"

#include "../../Phase1_Startup.hpp"

namespace GRIMText::Training {

void HyperparametersReady(TrainingContext& ctx) {
    // Slice grouping payloads against the authoritative root config without
    // storing second config owners on TrainingContext.
    static_cast<void>(GRIM::HyperParameters::lossConfigHP(ctx.config));
    static_cast<void>(GRIM::HyperParameters::trainingFixedShapeHP(ctx.config));
}

} // namespace GRIMText::Training

