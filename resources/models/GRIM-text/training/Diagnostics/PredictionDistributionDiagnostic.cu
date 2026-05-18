//======================================================//
//  PredictionDistributionDiagnostic.cu
//  Lifted verbatim from Phase2_TrainingLoop.cu — the
//  BATCH_TARGET_DIST log block and the post-loss
//  BATCH_PRED_DIST + [LogitTrace][PostLoss] block.
//  Behavior: identical. No logic, gating, ordering, or
//  log-string changes vs. the original inline blocks.
//======================================================//

#include "PredictionDistributionDiagnostic.hpp"
#include "DiagnosticGates.hpp"

#include "../Phases/Phase2_TrainingLoop.hpp"
#include "../../Shared/TrainingState/TrainingState_GPU.hpp"
#include "../../Shared/LogRecorder/LogRecorder.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iomanip>
#include <limits>
#include <map>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include <cuda_runtime.h>

namespace GRIM::Diagnostics {

using GRIM::Logging::ModuleId;
using GRIM::Logging::EmitModuleInfo;

void runTargetDistributionLog(
    GRIMText::Training::TrainingContext& ctx,
    const GRIM::Batching::BatchPayload& payload,
    int batch_idx)
{
    // Log target distribution using flat payload target_ids
    std::map<int, int> target_counts;
    int total_valid = 0;
    int total_tokens_td = 0;
    for (int s = 0; s < payload.batch_size; ++s) {
        const int flat_start = s * payload.max_seq_len;
        const int len = payload.seq_lengths[s];
        for (int t = 0; t < len; ++t) {
            const int tid = payload.target_ids[flat_start + t];
            total_tokens_td++;
            if (tid >= 0) {
                target_counts[tid]++;
                total_valid++;
            }
        }
    }

    // Find top-10 most common targets
    std::vector<std::pair<int, int>> sorted_targets(target_counts.begin(), target_counts.end());
    std::sort(sorted_targets.begin(), sorted_targets.end(),
              [](const auto& a, const auto& b) { return a.second > b.second; });

    std::ostringstream target_info;
    target_info << "BATCH_TARGET_DIST batch=" << (batch_idx + 1)
                << " total_tokens=" << total_tokens_td
                << " valid=" << total_valid
                << " unique=" << target_counts.size()
                << " top10=[";
    for (size_t i = 0; i < std::min(sorted_targets.size(), size_t(10)); ++i) {
        target_info << "tid=" << sorted_targets[i].first
                    << ":" << sorted_targets[i].second;
        if (i + 1 < std::min(sorted_targets.size(), size_t(10))) target_info << ", ";
    }
    target_info << "]";
    EmitModuleInfo(ModuleId::ForwardPass, target_info.str(), ctx.global_step);
}

void runPredictionDistributionAndLogitTrace(
    GRIMText::Training::TrainingContext& ctx,
    const GRIM::Batching::BatchPayload& payload,
    float loss,
    int batch_idx)
{
    namespace Internal = ::GRIMText::Training::Internal;
    const bool logit_trace_enabled = shouldLogLogitTrace(ctx, batch_idx);

    // Log model predictions (what it predicts vs targets) - uses ForwardPass module for filtering
    // GUARDED: Blocking cudaMemcpy drains GPU pipeline - only run on diagnostic sync interval
    if (shouldSyncDiagnostics(ctx, batch_idx)) {
        const auto& ts = ctx.model->getTrainingState();
        cudaStream_t stream = ts.stream_ctrl.getPrimaryStream();
        if (ts.cached_logits_tensor.data && payload.batch_size > 0 && payload.max_seq_len > 0) {
            const int total_tokens = payload.total_tokens;
            const int vocab_size = static_cast<int>(ctx.data_info.actual_vocab_size);
            const int sample_positions = total_tokens;
            const size_t logit_bytes = static_cast<size_t>(sample_positions) * vocab_size * sizeof(float);
            std::vector<float> logit_sample(sample_positions * vocab_size);
            cudaMemcpyAsync(logit_sample.data(), ts.cached_logits_tensor.data, logit_bytes, cudaMemcpyDeviceToHost, stream);
            cudaStreamSynchronize(stream);
            // Count argmax predictions
            std::map<int, int> pred_counts;
            for (int pos = 0; pos < sample_positions; ++pos) {
                float max_logit = -std::numeric_limits<float>::infinity();
                int argmax = 0;
                for (int v = 0; v < vocab_size; ++v) {
                    float logit = logit_sample[pos * vocab_size + v];
                    if (logit > max_logit) {
                        max_logit = logit;
                        argmax = v;
                    }
                }
                pred_counts[argmax]++;
            }

            // Sort by frequency
            std::vector<std::pair<int, int>> sorted_preds(pred_counts.begin(), pred_counts.end());
            std::sort(sorted_preds.begin(), sorted_preds.end(),
                      [](const auto& a, const auto& b) { return a.second > b.second; });

            std::ostringstream pred_info;
            pred_info << "BATCH_PRED_DIST batch=" << batch_idx
                      << " sampled_pos=" << sample_positions
                      << " unique_preds=" << pred_counts.size()
                      << " top10=[";
            for (size_t i = 0; i < std::min(sorted_preds.size(), size_t(10)); ++i) {
                pred_info << "tid=" << sorted_preds[i].first
                          << ":" << sorted_preds[i].second;
                if (i + 1 < std::min(sorted_preds.size(), size_t(10))) pred_info << ", ";
            }
            pred_info << "]";
            EmitModuleInfo(ModuleId::ForwardPass, pred_info.str(), ctx.global_step);

            if (logit_trace_enabled && sample_positions > 0) {
                int debug_pos = -1;
                int debug_b = -1;
                int debug_t = -1;
                int target_token = -1;
                for (int pos = 0; pos < sample_positions; ++pos) {
                    const int b = pos / payload.max_seq_len;
                    const int t = pos % payload.max_seq_len;
                    if (b < payload.batch_size &&
                        t < payload.seq_lengths[b]) {
                        const int candidate = payload.target_ids[b * payload.max_seq_len + t];
                        if (candidate >= 0 && candidate < vocab_size) {
                            debug_pos = pos;
                            debug_b = b;
                            debug_t = t;
                            target_token = candidate;
                            break;
                        }
                    }
                }
                if (debug_pos < 0) {
                    debug_pos = 0;
                    debug_b = 0;
                    debug_t = 0;
                    if (payload.batch_size > 0 && payload.seq_lengths[0] > 0) {
                        target_token = payload.target_ids[0];
                    }
                }

                const float* logits = logit_sample.data() + debug_pos * vocab_size;
                float max_logit = -std::numeric_limits<float>::infinity();
                int argmax = 0;
                for (int v = 0; v < vocab_size; ++v) {
                    const float logit = logits[v];
                    if (logit > max_logit) {
                        max_logit = logit;
                        argmax = v;
                    }
                }

                double sum_exp = 0.0;
                for (int v = 0; v < vocab_size; ++v) {
                    sum_exp += std::exp(static_cast<double>(logits[v] - max_logit));
                }
                if (sum_exp <= 0.0) {
                    sum_exp = 1.0;
                }

                double p_t = -1.0;
                if (target_token >= 0 && target_token < vocab_size) {
                    p_t = std::exp(static_cast<double>(logits[target_token] - max_logit)) / sum_exp;
                }

                std::ostringstream trace_msg;
                trace_msg << std::fixed << std::setprecision(6);
                trace_msg << "[LogitTrace][PostLoss] source=cached_logits"
                          << " batch=" << (batch_idx + 1)
                          << " pos=" << debug_pos
                          << " b=" << debug_b
                          << " t=" << debug_t
                          << " target=" << target_token;
                if (p_t >= 0.0) {
                    trace_msg << " p_t=" << p_t;
                } else {
                    trace_msg << " p_t=N/A";
                }
                // sum_exp = Σ exp(logit[v] - max_logit), NOT Σ exp(logit[v])!
                // This is the SHIFTED partition function. Low values (e.g. 5.5) mean
                // the distribution is PEAKED (only ~5 tokens have significant mass).
                // logsumexp = log(sum_exp) + max_logit = log(Σ exp(logit[v]))
                const double logsumexp = std::log(sum_exp) + static_cast<double>(max_logit);
                trace_msg << " max_logit=" << max_logit
                          << " argmax=" << argmax
                          << " sum_exp_shifted=" << sum_exp
                          << " logsumexp=" << logsumexp
                          << " logit_range=" << Internal::formatScalar(max_logit - static_cast<float>(*std::min_element(logits, logits + vocab_size)), 4)
                          << " loss=" << Internal::formatScalar(loss, 4);
                ctx.logging.logger->log(trace_msg.str());
            }
        }
    }
}

} // namespace GRIM::Diagnostics
