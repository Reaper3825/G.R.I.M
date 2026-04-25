//======================================================//
//  LossSpikeDiagnostic.cu
//  Per-sequence breakdown when loss spikes far above the
//  captured baseline (initial) loss — originally extracted
//  from Phase2_TrainingLoop.cu.
//======================================================//

#include "LossSpikeDiagnostic.hpp"

#include "../Phases/Phase2_TrainingLoop.hpp"
#include "../../Shared/Loss/LossSignals/LossSignals.hpp"

#include <sstream>
#include <algorithm>
#include <cstddef>

namespace GRIM::Diagnostics {

void runLossSpikeDiagnostic(
    GRIMText::Training::TrainingContext& ctx,
    const GRIM::Batching::BatchPayload& payload,
    float loss,
    int batch_idx,
    const GRIM::Loss::LossSignalBus& bus)
{
    const auto& signals = bus.latest();
    if (!signals.baseline_spike) {
        return;
    }
    const float baseline_loss   = signals.baseline_loss;
    const float multiplier      = bus.config().baseline_spike_multiplier;
    const float spike_threshold = baseline_loss * multiplier;

    std::ostringstream spike_diag;
    spike_diag << "[LossDiag] SPIKE DETECTED loss=" << loss
               << " baseline=" << baseline_loss
               << " threshold=" << spike_threshold
               << " (=" << multiplier << "x baseline)"
               << " batch=" << (batch_idx + 1) << "\n";
    spike_diag << "  Sequences: ";
    size_t max_seq_in_batch = 0;
    for (int i = 0; i < payload.batch_size; ++i) {
        spike_diag << payload.seq_ids[i] << "(len=" << payload.seq_lengths[i] << ")";
        max_seq_in_batch = std::max(max_seq_in_batch, static_cast<size_t>(payload.seq_lengths[i]));
        if (i + 1 < payload.batch_size) spike_diag << ", ";
    }
    const bool stability_seq_override = ctx.config.hyperparameters.stability_overrides_enabled
                                        && ctx.config.hyperparameters.stability_override_max_seq_len > 0;
    const int config_seq_len_limit = stability_seq_override
        ? ctx.config.hyperparameters.stability_override_max_seq_len
        : ctx.config.hyperparameters.max_seq_len;
    const int effective_seq_len_limit = config_seq_len_limit > 0
        ? config_seq_len_limit
        : ctx.model->getConfig().max_seq_len;
    spike_diag << "\n  MAX_SEQ_LEN=" << max_seq_in_batch;
    spike_diag << " d_model=" << ctx.model->getConfig().d_model;
    spike_diag << " limit=" << effective_seq_len_limit;
    if (stability_seq_override) {
        spike_diag << " (stability_override)";
    }
    if (effective_seq_len_limit > 0 &&
        max_seq_in_batch >= static_cast<size_t>(effective_seq_len_limit)) {
        spike_diag << " *** BOUNDARY CROSSED (seq_len >= " << effective_seq_len_limit
                   << (stability_seq_override ? " = stability_override.max_seq_len" : " = max_seq_len")
                   << ") ***";
    }
    spike_diag << "\n  First 10 targets per seq: ";
    for (int s = 0; s < payload.batch_size; ++s) {
        spike_diag << "[";
        const int flat_start = s * payload.max_seq_len;
        const int len = payload.seq_lengths[s];
        for (int t = 0; t < std::min(10, len); ++t) {
            spike_diag << payload.target_ids[flat_start + t];
            if (t + 1 < std::min(10, len)) spike_diag << ",";
        }
        spike_diag << "] ";
    }
    spike_diag << "\n  Loss indicates model p_t ~ exp(-" << loss << ") -> EXTREME WRONG CONFIDENCE";
    spike_diag << "\n  HYPOTHESIS: Position embedding corrupted for pos >= " << effective_seq_len_limit;
    spike_diag << "\n  Check: Is attention collapsing? Are embeddings for these tokens corrupted?";
    ctx.logging.logger->log(spike_diag.str());
}

} // namespace GRIM::Diagnostics
