//======================================================//
//  MtpDiagnostic.cu
//  Lifted verbatim from Phase2_TrainingLoop.cu — the
//  MTP per-head diagnostic and Lk/L0 healthy-range
//  monitor block from runEpoch's log-interval scope.
//  Behavior: identical. No logic, gating, ordering, or
//  log-string changes vs. the original inline block.
//======================================================//

#include "MtpDiagnostic.hpp"

#include "../Phases/Phase2_TrainingLoop.hpp"

#include <sstream>
#include <algorithm>
#include <cstddef>

namespace GRIM::Diagnostics {

void runMtpDiagnostic(
    GRIMText::Training::TrainingContext& ctx,
    const GRIMText::Training::BatchResult& batch_result)
{
    namespace Internal = ::GRIMText::Training::Internal;
    const auto& hp = ctx.config.hyperparameters;

    auto& ts = ctx.model->getTrainingState();
    if (ts.mtp_diagnostics.valid && !ts.mtp_diagnostics.head_loss.empty()) {
        const float L0 = ts.mtp_diagnostics.L0_main > 0.0f ? ts.mtp_diagnostics.L0_main : batch_result.loss;
        std::ostringstream mtp_log;
        for (size_t i = 0; i < ts.mtp_diagnostics.head_loss.size(); ++i) {
            const float Lk = ts.mtp_diagnostics.head_loss[i];
            const float acc = i < ts.mtp_diagnostics.head_acc.size() ? ts.mtp_diagnostics.head_acc[i] : 0.0f;
            const float ratio = (L0 > 0.0f) ? (Lk / L0) : 0.0f;
            mtp_log << "[MTP_EQUATION] head_k=" << (i + 1) << ": loss=" << Internal::formatScalar(Lk, 4)
                    << " acc=" << Internal::formatScalar(acc, 2) << "%"
                    << " loss_ratio=" << Internal::formatScalar(ratio, 4) << " ";
        }
        mtp_log << "alpha_effective=" << Internal::formatScalar(ts.mtp_diagnostics.alpha_effective, 4)
                << " L_total=" << Internal::formatScalar(ts.mtp_diagnostics.L_total, 4);
        ctx.logging.logger->log(mtp_log.str());
        // MTP Monitor: Lk/L0 with healthy-range indication (configurable via log_ratio_monitor)
        if (hp.mtp_log_ratio_monitor) {
            static const float kHealthyLow[] = { 1.1f, 1.3f, 1.5f, 1.6f };
            static const float kHealthyHigh[] = { 1.3f, 1.6f, 2.0f, 2.2f };
            std::ostringstream mon;
            mon << "[MTP_Monitor]";
            for (size_t i = 0; i < ts.mtp_diagnostics.head_loss.size(); ++i) {
                const float ratio = (L0 > 0.0f) ? (ts.mtp_diagnostics.head_loss[i] / L0) : 0.0f;
                const int k = static_cast<int>(i) + 1;
                const size_t idx = std::min(static_cast<size_t>(k - 1), static_cast<size_t>(4));
                const float lo = kHealthyLow[idx];
                const float hi = kHealthyHigh[idx];
                const bool ok = (ratio >= lo && ratio <= hi);
                mon << " k=" << k << ": Lk/L0=" << Internal::formatScalar(ratio, 3)
                    << " (healthy " << Internal::formatScalar(lo, 1) << "-" << Internal::formatScalar(hi, 1)
                    << (ok ? " OK" : " OUT_OF_RANGE") << ")";
            }
            ctx.logging.logger->log(mon.str());
        }
    }
}

} // namespace GRIM::Diagnostics
