#include "Phase2_EnvironmentLoop.hpp"
#include "../../Shared/Environment/EnvironmentInput.hpp"
#include "../../Shared/Environment/EnvironmentBoundary.hpp"
#include "../../Shared/InferenceState/InferencePrefill.hpp"

#include <chrono>
#include <stdexcept>

namespace GRIMText::Training {

Phase2EnvironmentResult beginPhase2EnvironmentInteraction(TrainingContext& ctx) {
    using namespace GRIM::HyperParameters;
    if (snapshotExecutionMode(ctx.config) != ModelExecutionMode::TRAINING ||
        dataLoadingHP(ctx.config).training_stage != TrainingStage::RL)
        throw std::logic_error("Environment interaction requires the RL preparation path");
    if (!ctx.environment_session || !ctx.environment_tokenizer)
        throw std::logic_error("Environment interaction requires a prepared episode and tokenizer");
    auto& generation_state = ctx.requireGenerationState("beginPhase2EnvironmentInteraction");
    if (generation_state.active_session)
        throw std::logic_error("Environment interaction cannot replace a live generation session");
    const auto& episode = *ctx.environment_session;
    if (episode.taskEpisode().state != GRIM::Episodes::EpisodeState::Collecting)
        throw std::logic_error("Environment Goal attempt is already submitted or invalid");
    const auto input = GRIM::Environment::buildEnvironmentInferenceInput(
        *episode.taskEpisode().goal, episode.latestResponse());
    Phase2EnvironmentResult result;
    result.episode_status = input.episode_status;

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

Phase2EnvironmentRunResult executePhase2EnvironmentInteraction(TrainingContext& ctx) {
    if (ctx.environment_session &&
        ctx.environment_session->taskEpisode().state != GRIM::Episodes::EpisodeState::Collecting)
        throw std::logic_error("Environment Goal attempt is already submitted or invalid");
    Phase2EnvironmentRunResult run;
    try {
        run.interaction = beginPhase2EnvironmentInteraction(ctx);
        if (!run.interaction.generation)
            throw std::logic_error("Environment attempt began without generation");
        auto& episode = *ctx.environment_session;
        auto& tokenizer = *ctx.environment_tokenizer;
        std::optional<std::size_t> previous_close;
        for (;;) {
            auto& generated = *run.interaction.generation;
            if (!generated.generated_sequence)
                throw std::logic_error("Environment inference returned no trajectory");
            const auto& sequence = *generated.generated_sequence;
            if (sequence.return_reason != GRIM::Inference::InferenceReturnReason::SpanBoundary) {
                if (!sequence.finished)
                    throw std::logic_error("Environment inference returned without a pause or termination");
                using Reason = GRIM::Inference::InferenceReturnReason;
                using End = GRIM::Episodes::AttemptEndReason;
                const auto end = sequence.return_reason == Reason::EOS ? End::Eos :
                    sequence.return_reason == Reason::ContextLimit ? End::ContextLimit :
                    sequence.return_reason == Reason::ExplicitFinish ? End::ExplicitFinish :
                    End::SampledTokenLimit;
                episode.finishAttempt(sequence, generated.prompt_token_count, end);
                run.stop_reason = "generation_terminal";
                return run; // Incomplete attempts fail, even if the world reached its Goal.
            }
            const auto boundary = GRIM::Environment::inspectEnvironmentBoundary(
                sequence, generated.prompt_token_count, previous_close);
            previous_close = boundary.end - 1;
            if (boundary.phase == "answer") {
                episode.finishAttempt(sequence, generated.prompt_token_count,
                    GRIM::Episodes::AttemptEndReason::AnswerClosed, boundary.end - 1);
                run.stop_reason = "attempt_submitted";
                ctx.requireGenerationState("executePhase2EnvironmentInteraction").resetSession();
                return run;
            }
            if (boundary.phase == "execute") {
                // Decode exactly one command from this Execute, excluding tags.
                // Answer is a Goal-attempt submission and never reaches the worker.
                GRIM::Tokenizer::DecodeRequest request(sequence.token_ids, sequence.atom_entry_ids,
                    sequence.context_atom_table.get(), sequence.token_numeric_values,
                    sequence.token_atom_mask, boundary.begin + 1);
                request.token_count = request.atom_entry_count = request.token_numeric_count =
                    request.token_atom_mask_count = boundary.end - boundary.begin - 2;
                request.lenient_invalid_utf8 = false;
                episode.submitCommand(boundary.end - 1, tokenizer.decode(request));
                const auto input = GRIM::Environment::buildEnvironmentInferenceInput(
                    *episode.taskEpisode().goal, episode.latestResponse());
                run.interaction.episode_status = input.episode_status;
                appendPhase2RuntimeContext(ctx, tokenizer, "\n\n" + input.prompt + "\n\n");
            }
            // Determine/Define/Update only resume. Execute feedback, including
            // terminal world observations, precedes the model's final Answer.
            run.interaction.generation = resumePhase2PayloadInference(ctx, tokenizer);
        }
    } catch (const std::invalid_argument&) {
        // Invalid model syntax is a failed attempt, not an infrastructure sample
        // to exclude. Transport uncertainty already invalidates its session.
        if (ctx.environment_session && run.interaction.generation &&
            run.interaction.generation->generated_sequence &&
            ctx.environment_session->taskEpisode().state == GRIM::Episodes::EpisodeState::Collecting) {
            try {
                ctx.environment_session->finishAttempt(
                    *run.interaction.generation->generated_sequence,
                    run.interaction.generation->prompt_token_count,
                    GRIM::Episodes::AttemptEndReason::InvalidOutput);
            } catch (...) {
                ctx.environment_session->invalidateAttempt();
            }
        } else if (ctx.environment_session) {
            ctx.environment_session->invalidateAttempt();
        }
        ctx.requireGenerationState("executePhase2EnvironmentInteraction").resetSession();
        throw;
    } catch (...) {
        // Preserve episode action receipts (including unknown outcomes), while
        // preventing accidental continuation from an inconsistent model session.
        if (ctx.environment_session) ctx.environment_session->invalidateAttempt();
        ctx.requireGenerationState("executePhase2EnvironmentInteraction").resetSession();
        throw;
    }
}


Phase2EnvironmentLoopResult executePhase2EnvironmentLoop(TrainingContext& ctx) {
    if (!ctx.environment_session || !ctx.environment_tokenizer)
        throw std::logic_error("Environment loop requires prepared curriculum examples and tokenizer");
    auto& session = *ctx.environment_session;
    if (session.scheduleComplete())
        throw std::logic_error("Environment curriculum is already complete");
    Phase2EnvironmentLoopResult result;
    do {
        try {
            executePhase2EnvironmentInteraction(ctx);
        } catch (const std::invalid_argument&) {
            // A structurally invalid model continuation can already have a
            // committed failed attempt. Count it and continue; never swallow
            // transport or initialization failures as model failures.
            const auto& episode = session.taskEpisode();
            if (episode.state != GRIM::Episodes::EpisodeState::AttemptSubmitted ||
                !episode.attempts[0] ||
                episode.attempts[0]->end_reason != GRIM::Episodes::AttemptEndReason::InvalidOutput)
                throw;
        }
        const auto& episode = session.taskEpisode();
        const auto& attempt = episode.attempts[0];
        if (!attempt || episode.state == GRIM::Episodes::EpisodeState::InfrastructureInvalid)
            throw std::logic_error("Environment example ended without a valid attempt record");
        ++result.examples_completed;
        if (attempt->grade.verdict == GRIM::Episodes::GradeVerdict::Passed)
            ++result.examples_succeeded;
        else
            ++result.examples_failed;
        result.commands += episode.actions.size();
        // EOS releases the real session too; explicitly reset here to guarantee
        // no KV state, pending span, or token budget crosses example boundaries.
        ctx.requireGenerationState("executePhase2EnvironmentLoop").resetSession();
    } while (session.advanceExample());
    return result;
}

} // namespace GRIMText::Training
