#include "EnvironmentReady.hpp"
#include "../Phase1_Startup.hpp"
#include "../../../Shared/DataLoader/CurriculumRegistry.hpp"

namespace GRIMText::Training {

void EnvironmentReady(TrainingContext& ctx) {
    using namespace GRIM::HyperParameters;
    if (snapshotExecutionMode(ctx.config) != ModelExecutionMode::TRAINING ||
        dataLoadingHP(ctx.config).training_stage != TrainingStage::RL)
        throw std::runtime_error("EnvironmentReady requires the RL startup path");
    if (ctx.environment_session)
        throw std::runtime_error("EnvironmentReady: environment was already prepared");
    const auto tokenizer_hp = tokenizerHP(ctx.config);
    const auto cache = std::filesystem::path(tokenizer_hp.data_path).parent_path();
    auto metadata = GRIM::LoadCurriculumMetadataFromRegistry(cache, tokenizer_hp.training_curriculum);
    if (metadata.training_stage != "rl")
        throw std::runtime_error("EnvironmentReady: selected curriculum training_stage must be rl");
    auto schedule = GRIM::loadScheduledGoals(cache, tokenizer_hp.training_curriculum,
        GRIM::Batching::parseCurriculumOrdering(trainingScheduleHP(ctx.config).batch_strategy),
        ctx.rng.data_seed);
    // Use the same config-relative resources/models/games registry as World Lab.
    // Assignment environment_id is authoritative; never substitute a named world.
    auto candidate = GRIM::Environment::prepareEnvironmentSession(
        std::move(schedule), resolveMappedPath("resources/models/games"));
    ctx.current_curriculum_metadata = std::move(metadata);
    ctx.environment_session = std::move(candidate);
    ctx.logging.logger->log("[Phase1] Environment ready | assignment=" +
        ctx.environment_session->currentAssignment().concept_block_id + " | environment=" +
        ctx.environment_session->currentAssignment().environment_id + " | schedule_position=0");
}

} // namespace GRIMText::Training
