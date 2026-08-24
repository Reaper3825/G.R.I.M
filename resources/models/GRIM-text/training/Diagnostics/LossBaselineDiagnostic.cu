//======================================================//
//  LossBaselineDiagnostic.cu
//  Adaptive loss baseline tracking + invalid-token
//  validation. Originally inline in Phase2_TrainingLoop.cu
//  processBatch.
//======================================================//

#include "LossBaselineDiagnostic.hpp"

#include "../Phases/Phase2_TrainingLoop.hpp"
#include "../../Shared/Batching/BatchPayload.hpp"

#include <cmath>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <string>

namespace GRIM::Diagnostics {

namespace {

std::string formatScalar(float value, int precision = 4) {
    if (!std::isfinite(value)) {
        if (std::isnan(value)) return "nan";
        return value < 0.0f ? "-inf" : "inf";
    }
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(precision) << value;
    return oss.str();
}

} // namespace

void runLossBaselineAndTokenValidation(
    GRIMText::Training::TrainingContext& ctx,
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
        // Record the actual initialization source selected during Phase1. Do not
        // infer checkpoint loading from loss magnitude: focal scaling and an
        // LM-head unigram prior can both put a fresh model well below ln(V).
        const float plain_ce_random_reference = payload.EnableAtomIdentification
            ? std::log(2.0f)
            : std::log(static_cast<float>(payload.vocab_size));
        const bool checkpoint_loaded = !ctx.loaded_checkpoint_path.empty();
        const std::string source_note = checkpoint_loaded
            ? "model_source=checkpoint checkpoint_path=\"" + ctx.loaded_checkpoint_path + "\""
            : "model_source=random_initialization";
        ctx.logging.logger->log(
            "[LossBaseline] Initial optimized loss=" + formatScalar(state.initial_loss) +
            " " + source_note +
            " plain_ce_random_reference=" + formatScalar(plain_ce_random_reference));
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
    (void)loss;
}

} // namespace GRIM::Diagnostics
