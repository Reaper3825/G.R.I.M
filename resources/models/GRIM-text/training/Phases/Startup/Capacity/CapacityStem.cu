#include "CapacityStem.hpp"

#include "../../Phase1_Startup.hpp"

namespace GRIMText::Training {

void HyperparametersReady(TrainingContext& ctx) {
    // Slice grouping payloads against the authoritative root config without
    // storing second config owners on TrainingContext.
    static_cast<void>(GRIM::HyperParameters::lossConfigHP(ctx.config));
    static_cast<void>(GRIM::HyperParameters::trainingFixedShapeHP(ctx.config));

    const auto execution_mode = GRIM::HyperParameters::snapshotExecutionMode(ctx.config);
    if (execution_mode == GRIM::HyperParameters::ModelExecutionMode::INFERENCE) {
        const auto paths_hp = GRIM::HyperParameters::pathsHP(ctx.config);
        ctx.requested_checkpoint_path = paths_hp.output_model_path;
    } else {
        ctx.requested_checkpoint_path.clear();
    }
}

} // namespace GRIMText::Training
