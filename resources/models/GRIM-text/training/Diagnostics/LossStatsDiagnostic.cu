//======================================================//
//  LossStatsDiagnostic.cu
//  Implementation of runLossStatsDiagnostic.
//  Lifted verbatim from Phase2_TrainingLoop.cu processBatch
//  (BATCH_LOSS equation + LossStats summary line).
//======================================================//

#include "LossStatsDiagnostic.hpp"

#include "../Phases/Phase2_TrainingLoop.hpp"
#include "../../Shared/Batching/BatchPayload.hpp"
#include "../../Shared/LogRecorder/LogTypes.hpp"
#include "../../Shared/LogRecorder/BatchLogTape.hpp"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <sstream>
#include <string>

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

} // namespace

void runLossStatsDiagnostic(
    GRIMText::Training::TrainingContext& ctx,
    const GRIM::Batching::BatchPayload& payload,
    const GRIMText::Training::BatchResult& result,
    int batch_idx)
{
    // ========================================================================
    // EQUATION LOGGING: Per-batch loss computation trace (Issue #120 debug)
    // ========================================================================
    {
        const int valid_tokens_eq = payload.lm_valid_tokens;
        const float expected_random_loss = std::log(static_cast<float>(payload.vocab_size));

        std::ostringstream eq;
        eq << "[BATCH_LOSS] loss = -sum(log(p_target)) / valid_tokens\n";
        eq << "  valid_tokens=" << valid_tokens_eq << " vocab_size=" << payload.vocab_size << "\n";
        eq << "  EXPECTED loss (random) = ln(" << payload.vocab_size << ") = " << expected_random_loss << "\n";
        eq << "  ACTUAL loss = " << result.loss << "\n";
        EQ_LOG(ctx.logging.tape.get(), GRIM::Logging::LogGroup::Loss, GRIM::Logging::LogPhase::LOSS_COMPUTATION, -1, "BATCH_LOSS", eq.str().c_str());
    }

    {
        const int valid_tokens = payload.lm_valid_tokens;
        const int total_tokens = payload.total_tokens;
        const int masked_tokens = std::max(total_tokens - valid_tokens, 0);
        const float loss_sum = (valid_tokens > 0)
            ? result.loss * static_cast<float>(valid_tokens)
            : 0.0f;
        std::ostringstream loss_stats;
        loss_stats << "[LossStats] batch=" << (batch_idx + 1)
                   << " loss_mean=" << formatScalar(result.loss, 4)
                   << " loss_sum=" << formatScalar(loss_sum, 4)
                   << " text_ce=" << formatScalar(result.text_loss, 4)
                   << " mtp=" << formatScalar(result.mtp_loss, 4)
                   << " selector=" << formatScalar(result.selector_loss, 4)
                   << " execution=" << formatScalar(result.execution_loss, 4)
                   << " valid_tokens=" << valid_tokens
                   << " masked_tokens=" << masked_tokens
                   << " total_tokens=" << total_tokens;
        ctx.logging.logger->log(loss_stats.str());
    }
}

} // namespace GRIM::Diagnostics
