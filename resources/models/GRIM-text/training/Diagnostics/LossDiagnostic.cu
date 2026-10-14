//======================================================//
//  LossDiagnostic.cu
//  Loss equation/statistics, observed baseline tracking, and token validation.
//======================================================//

#include "LossDiagnostic.hpp"

#include "../Phases/Phase2_TrainingLoop.hpp"
#include "../../Shared/Batching/BatchPayload.hpp"
#include "../../Shared/LogRecorder/LogTypes.hpp"
#include "../../Shared/LogRecorder/BatchLogTape.hpp"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <sstream>
#include <string>
#include <stdexcept>

namespace GRIM::Diagnostics {

namespace {

std::string formatScalar(float value, int precision = 4) {
    std::ostringstream oss;
    if (std::isfinite(value)) {
        if (value == 0.0f) value = 0.0f;
        oss << std::fixed << std::setprecision(precision) << value;
    } else if (std::isnan(value)) {
        oss << "nan";
    } else {
        oss << (value > 0 ? "inf" : "-inf");
    }
    return oss.str();
}

void updateLossBaselineAndValidateTokens(
    GRIMText::Training::TrainingLoopState& state,
    const GRIM::Batching::BatchPayload& payload,
    float loss,
    int batch_idx)
{
    // Adaptive loss tracking - record baseline and minimum
    // NOTE: Loss alone is NOT sufficient to skip batches. Skipping hard examples
    // based on loss biases the model away from difficult regions of the data manifold,
    // causing apparent early improvement but later generalization failure.
    // Only skip on CONFIRMED corruption: invalid tokens, NaNs, or gradient spikes.
    if (state.initial_loss == 0.0f) {
        state.initial_loss = loss;
        state.min_observed_loss = loss;
        return;
    }

    state.warmup_batches++;

    // Track minimum observed loss
    if (loss < state.min_observed_loss) {
        state.min_observed_loss = loss;
    }

    // Check for CONFIRMED corruption: invalid token IDs
    // NaN/Inf loss is already handled by the explicit Phase2 forward/loss/backward path upstream.
    // Loss-only skipping removes hard examples and destroys generalization
    bool has_invalid_tokens = false;

    // Scan for token IDs outside vocab range (actual corruption) — using flat payload
    for (int s = 0; s < payload.batch_size && !has_invalid_tokens; ++s) {
        const int flat_start = s * payload.max_seq_len;
        const int len = payload.seq_lengths[s];
        for (int t = 0; t < len; ++t) {
            if (payload.input_ids[flat_start + t] < 0 ||
                payload.input_ids[flat_start + t] >= payload.vocab_size) {
                has_invalid_tokens = true;
                break;
            }
        }
    }
    if (!payload.EnableAtomIdentification) {
        for (int s = 0; s < payload.batch_size && !has_invalid_tokens; ++s) {
            const int flat_start = s * payload.max_seq_len;
            const int len = payload.seq_lengths[s];
            for (int t = 0; t < len; ++t) {
                const int tid = payload.target_ids[flat_start + t];
                // targets can be -1 for masked positions, but not other negatives or OOB
                if (tid < -1 || tid >= payload.vocab_size) {
                    has_invalid_tokens = true;
                    break;
                }
            }
        }
    }

    // Rule 20: Data corruption = CRASH. Fix the data pipeline, don't silently skip.
    if (has_invalid_tokens) {
        throw std::runtime_error(
            "DATA CORRUPTION: batch " + std::to_string(batch_idx + 1) +
            " contains token IDs outside vocab range [0, " + std::to_string(payload.vocab_size) +
            ") — fix data pipeline at " + std::string(__FILE__) + ":" + std::to_string(__LINE__));
    }

    // Train-loss spike detection is owned by TelemetryLattice. Do NOT add a
    // second [LossMonitor] HIGH_LOSS log line here — duplicate scalar spike
    // paths drift from the telemetry-owned definition.
}

} // namespace

void runLossDiagnostic(
    GRIMText::Training::TrainingContext& ctx,
    GRIMText::Training::TrainingLoopState& state,
    const GRIM::Batching::BatchPayload& payload,
    const GRIMText::Training::BatchResult& result,
    int batch_idx)
{
    const bool initializing_baseline = state.initial_loss == 0.0f;
    updateLossBaselineAndValidateTokens(state, payload, result.loss, batch_idx);

    // ========================================================================
    // EQUATION LOGGING: Per-batch loss computation trace (Issue #120 debug)
    // ========================================================================
    {
        const bool atom_mode = payload.EnableAtomIdentification;
        const int valid_tokens_eq = atom_mode
            ? payload.atom_insertion_valid_gap_count
            : payload.lm_valid_tokens;

        std::ostringstream eq;
        eq << "[BATCH_LOSS] total = "
           << (atom_mode ? "atom_bce" : "text_ce");
        if (result.local_atom_retrieval_queries > 0) {
            eq << " + " << result.weight_local_atom_retrieval
               << " * local_atom_retrieval_ce";
        }
        eq << "\n";
        eq << "  valid_tokens=" << valid_tokens_eq << " vocab_size=" << payload.vocab_size << "\n";
        eq << "  ACTUAL total = " << result.loss << "\n";
        eq << "  ACTUAL primary = " << result.text_loss << "\n";
        eq << "  ACTUAL local_atom_retrieval_raw = "
           << result.local_atom_retrieval_loss_raw << "\n";
        eq << "  ACTUAL local_atom_retrieval_weighted = "
           << result.local_atom_retrieval_loss << "\n";
        EQ_LOG(ctx.logging.tape.get(), GRIM::Logging::LogGroup::Loss, GRIM::Logging::LogPhase::LOSS_COMPUTATION, -1, "BATCH_LOSS", eq.str().c_str());
    }

    {
        const bool atom_mode = payload.EnableAtomIdentification;
        const int valid_tokens = atom_mode
            ? payload.atom_insertion_valid_gap_count
            : payload.lm_valid_tokens;
        const int total_tokens = atom_mode
            ? payload.atomInsertionGapRowCount()
            : payload.total_tokens;
        const int masked_tokens = std::max(total_tokens - valid_tokens, 0);
        const float text_loss_sum = (valid_tokens > 0)
            ? result.text_loss * static_cast<float>(valid_tokens)
            : 0.0f;
        std::ostringstream loss_stats;
        loss_stats << "[LossStats] batch=" << (batch_idx + 1)
                   << " loss_mean=" << formatScalar(result.loss, 4)
                   << " text_loss_sum=" << formatScalar(text_loss_sum, 4)
                   << " primary_loss=" << formatScalar(result.text_loss, 4)
                   << " task=" << (atom_mode ? "atom_insertion" : "language_model")
                   << " local_atom_retrieval_raw="
                   << formatScalar(result.local_atom_retrieval_loss_raw, 4)
                   << " local_atom_retrieval_weighted="
                   << formatScalar(result.local_atom_retrieval_loss, 4)
                   << " local_atom_retrieval_weight="
                   << formatScalar(result.weight_local_atom_retrieval, 4)
                   << " local_atom_queries=" << result.local_atom_retrieval_queries
                   << " local_atom_reference_targets="
                   << result.local_atom_reference_targets
                   << " selector=" << formatScalar(result.selector_loss, 4)
                   << " valid_tokens=" << valid_tokens
                   << " masked_tokens=" << masked_tokens
                   << " total_tokens=" << total_tokens
                   << " initial_loss=" << formatScalar(state.initial_loss)
                   << " min_observed_loss=" << formatScalar(state.min_observed_loss)
                   << " warmup_batches=" << state.warmup_batches;
        if (initializing_baseline) {
            const bool checkpoint_loaded = !ctx.loaded_checkpoint_path.empty();
            loss_stats << " model_source="
                       << (checkpoint_loaded ? "checkpoint" : "random_initialization");
            if (checkpoint_loaded) {
                loss_stats << " checkpoint_path=\"" << ctx.loaded_checkpoint_path << "\"";
            }
        }
        ctx.logging.logger->log(loss_stats.str());
    }
}

} // namespace GRIM::Diagnostics
