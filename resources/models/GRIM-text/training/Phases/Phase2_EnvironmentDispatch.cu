#include "Phase2_EnvironmentDispatch.hpp"
#include "../../Shared/Environment/EnvironmentBoundary.hpp"
#include "../../Shared/Environment/EnvironmentInput.hpp"
#include <stdexcept>
#include <utility>

namespace GRIMText::Training {
Phase2EnvironmentRunResult executePhase2EnvironmentInteraction(TrainingContext& ctx) {
    Phase2EnvironmentRunResult run;
    try {
        run.interaction = beginPhase2EnvironmentInteraction(ctx);
        if (!run.interaction.generation) {
            run.stop_reason = "episode_terminal";
            return run;
        }
        auto& episode = *ctx.environment_episode;
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
                run.stop_reason = "generation_terminal";
                return run; // EOS/limits are not Goal completion; do not dispatch.
            }
            const auto boundary = GRIM::Environment::inspectEnvironmentBoundary(
                sequence, generated.prompt_token_count, previous_close);
            previous_close = boundary.end - 1;
            if (boundary.phase == "answer") {
                // Decode only this Answer's content, excluding both delimiters.
                // Cumulative continuation text can include older actions/feedback.
                GRIM::Tokenizer::DecodeRequest request(sequence.token_ids, sequence.atom_entry_ids,
                    sequence.context_atom_table.get(), sequence.token_numeric_values,
                    sequence.token_atom_mask, boundary.begin + 1);
                request.token_count = request.atom_entry_count = request.token_numeric_count =
                    request.token_atom_mask_count = boundary.end - boundary.begin - 2;
                request.lenient_invalid_utf8 = false;
                episode.submitAnswer(boundary.end - 1, tokenizer.decode(request));
                const auto input = GRIM::Environment::buildEnvironmentInferenceInput(
                    episode.currentAssignment().goal, episode.latestResponse());
                run.interaction.episode_status = input.episode_status;
                if (input.episode_status != "active") {
                    run.stop_reason = "episode_terminal";
                    ctx.requireGenerationState("executePhase2EnvironmentInteraction").resetSession();
                    return run;
                }
                appendPhase2RuntimeContext(ctx, tokenizer, "\n\n" + input.prompt + "\n\n");
            }
            // DDE/Update are observed without touching the environment. Answer
            // feedback is queued before this same-session resume.
            run.interaction.generation = resumePhase2PayloadInference(ctx, tokenizer);
        }
    } catch (...) {
        // Preserve episode action receipts (including unknown outcomes), while
        // preventing accidental continuation from an inconsistent model session.
        ctx.requireGenerationState("executePhase2EnvironmentInteraction").resetSession();
        throw;
    }
}
}
