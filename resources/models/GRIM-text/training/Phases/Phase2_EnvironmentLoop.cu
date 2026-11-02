#include "Phase2_EnvironmentLoop.hpp"
#include "../../Shared/Environment/EnvironmentInput.hpp"
#include "../../Shared/InferenceState/InferencePrefill.hpp"

#include <chrono>
#include <stdexcept>

namespace GRIMText::Training {

Phase2EnvironmentResult beginPhase2EnvironmentInteraction(TrainingContext& ctx) {
    using namespace GRIM::HyperParameters;
    if (snapshotExecutionMode(ctx.config) != ModelExecutionMode::TRAINING ||
        dataLoadingHP(ctx.config).training_stage != TrainingStage::RL)
        throw std::logic_error("Environment interaction requires the RL preparation path");
    if (!ctx.environment_episode || !ctx.environment_tokenizer)
        throw std::logic_error("Environment interaction requires a prepared episode and tokenizer");
    auto& generation_state = ctx.requireGenerationState("beginPhase2EnvironmentInteraction");
    if (generation_state.active_session)
        throw std::logic_error("Environment interaction cannot replace a live generation session");
    const auto& episode = *ctx.environment_episode;
    const auto input = GRIM::Environment::buildEnvironmentInferenceInput(
        episode.currentAssignment().goal, episode.latestResponse());
    Phase2EnvironmentResult result;
    result.episode_status = input.episode_status;
    if (result.episode_status != "active") return result;

    const auto encode_start = std::chrono::steady_clock::now();
    auto prefill = GRIM::Inference::buildInferencePrefill(ctx.config, *ctx.environment_tokenizer,
        input.prompt, &input.reasoning_state);
    const auto encode_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - encode_start).count();
    if (!prefill.concept_span_delimiters)
        throw std::logic_error("Environment interaction requires structured span boundaries");
    result.generation = executePhase2PayloadInference(ctx, *ctx.environment_tokenizer,
        prefill, generationHP(ctx.config));
    result.generation->encode_ms = encode_ms;
    return result;
}

} // namespace GRIMText::Training
