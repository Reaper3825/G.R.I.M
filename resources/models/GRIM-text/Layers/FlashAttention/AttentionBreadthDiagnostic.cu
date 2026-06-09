#include "AttentionDiagnostics.hpp"

#include "../../Shared/LogRecorder/BatchLogTape.hpp"
#include "../../Shared/VerboseLogging.hpp"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

bool shouldEmitAttentionBreadthDiagnostics() {
    auto* tape = GRIM::Logging::getGlobalTape();
    return tape &&
           tape->accepts(GRIM::Logging::LogLevel::Info, GRIM::Logging::LogGroup::Attention) &&
           !tape->skipThisPass();
}

struct LinearStats {
    float min_val = 0.0f;
    float max_val = 0.0f;
    float rms = 0.0f;
};

LinearStats computeLinearStats(const std::vector<float>& values, const char* context) {
    if (values.empty()) {
        throw std::runtime_error(std::string(context) + ": values is empty");
    }
    LinearStats stats{};
    stats.min_val = values.front();
    stats.max_val = values.front();
    double sum_sq = 0.0;
    for (float value : values) {
        if (!std::isfinite(value)) {
            throw std::runtime_error(std::string(context) + ": encountered non-finite value");
        }
        stats.min_val = std::min(stats.min_val, value);
        stats.max_val = std::max(stats.max_val, value);
        sum_sq += static_cast<double>(value) * static_cast<double>(value);
    }
    stats.rms = std::sqrt(static_cast<float>(sum_sq / static_cast<double>(values.size())));
    return stats;
}

struct AttentionBreadthStats {
    float alpha_sum = 0.0f;
    float alpha_min = 0.0f;
    float alpha_max = 0.0f;
    float entropy = 0.0f;
    float normalized_entropy = 0.0f;
    float effective_support = 0.0f;
    float participation_ratio = 0.0f;
    float uniform_fraction = 0.0f;
    float mean_source_index = 0.0f;
    float normalized_mean_source_index = 0.0f;
};

AttentionBreadthStats computeAttentionBreadthStats(const std::vector<float>& causal_scores_row,
                                                   float lse_value) {
    if (causal_scores_row.empty()) {
        throw std::runtime_error("computeAttentionBreadthStats: causal_scores_row is empty");
    }
    if (!std::isfinite(lse_value)) {
        throw std::runtime_error("computeAttentionBreadthStats: lse_value is non-finite");
    }

    AttentionBreadthStats stats{};
    stats.alpha_min = std::numeric_limits<float>::infinity();

    double alpha_sum = 0.0;
    double alpha_sq_sum = 0.0;
    double entropy = 0.0;
    double weighted_source_sum = 0.0;

    for (size_t u = 0; u < causal_scores_row.size(); ++u) {
        const float score = causal_scores_row[u];
        if (!std::isfinite(score)) {
            throw std::runtime_error("computeAttentionBreadthStats: causal_scores_row contains non-finite score");
        }
        const float shifted = score - lse_value;
        const float alpha = std::exp(shifted);
        if (!std::isfinite(alpha)) {
            throw std::runtime_error("computeAttentionBreadthStats: alpha is non-finite after exp(score - lse)");
        }
        stats.alpha_min = std::min(stats.alpha_min, alpha);
        stats.alpha_max = std::max(stats.alpha_max, alpha);
        alpha_sum += alpha;
        alpha_sq_sum += static_cast<double>(alpha) * static_cast<double>(alpha);
        if (alpha > 0.0f) {
            entropy -= static_cast<double>(alpha) * static_cast<double>(shifted);
        }
        weighted_source_sum += static_cast<double>(u) * static_cast<double>(alpha);
    }

    if (!std::isfinite(alpha_sum) || alpha_sum <= 0.0) {
        throw std::runtime_error("computeAttentionBreadthStats: alpha_sum is invalid");
    }
    const double mass_error = std::fabs(alpha_sum - 1.0);
    if (mass_error > 5.0e-3) {
        throw std::runtime_error(
            "computeAttentionBreadthStats: alpha row sums to " + std::to_string(alpha_sum) +
            " (expected ~1.0). Pass the FULL causal score row and matching exact lse value.");
    }
    if (!std::isfinite(alpha_sq_sum) || alpha_sq_sum <= 0.0) {
        throw std::runtime_error("computeAttentionBreadthStats: alpha_sq_sum is invalid");
    }

    const float prefix_len = static_cast<float>(causal_scores_row.size());
    const float max_entropy = causal_scores_row.size() > 1
        ? std::log(prefix_len)
        : 0.0f;

    stats.alpha_sum = static_cast<float>(alpha_sum);
    stats.entropy = static_cast<float>(entropy);
    stats.normalized_entropy = max_entropy > 0.0f
        ? std::clamp(stats.entropy / max_entropy, 0.0f, 1.0f)
        : 0.0f;
    stats.effective_support = std::exp(stats.entropy);
    stats.participation_ratio = 1.0f / static_cast<float>(alpha_sq_sum);
    stats.uniform_fraction = stats.participation_ratio / prefix_len;
    stats.mean_source_index = static_cast<float>(weighted_source_sum / alpha_sum);
    stats.normalized_mean_source_index = causal_scores_row.size() > 1
        ? stats.mean_source_index / static_cast<float>(causal_scores_row.size() - 1)
        : 0.0f;
    return stats;
}

std::string interpretBreadth(const AttentionBreadthStats& stats, size_t prefix_len) {
    std::ostringstream interpretation;
    if (stats.normalized_entropy > 0.90f && stats.alpha_max < 0.20f) {
        interpretation << "[ANOMALY] broad prefix-averaging risk: attention is close to uniform over "
                       << stats.effective_support << " of " << prefix_len
                       << " positions; this can reduce token distinctness";
    } else if (stats.normalized_entropy > 0.75f && stats.uniform_fraction > 0.50f) {
        interpretation << "[WARNING] broad attention row: over half of the causal prefix participates meaningfully";
    } else if (stats.alpha_max > 0.85f) {
        interpretation << "sharp / selective row";
    } else {
        interpretation << "moderate breadth";
    }
    return interpretation.str();
}

}  // namespace

namespace GRIM::FlashAttentionDiagnostics {

void emitAttentionBreadthDiagnostic(const std::vector<float>& causal_scores_row,
                                                float lse_value,
                                                int query_index,
                                                int layer_idx,
                                                int head_idx) {
    if (!shouldEmitAttentionBreadthDiagnostics()) {
        return;
    }

    auto* tape = GRIM::Logging::getGlobalTape();
    if (!tape) {
          throw std::runtime_error("emitAttentionBreadthDiagnostic: global tape is not available");
    }

     const LinearStats score_stats = computeLinearStats(causal_scores_row, "emitAttentionBreadthDiagnostic scores");
    const AttentionBreadthStats breadth_stats = computeAttentionBreadthStats(causal_scores_row, lse_value);

     std::ostringstream log_line;
     log_line << std::fixed << std::setprecision(6);
     log_line << "alpha=exp(score-lse)"
                 << " layer=" << layer_idx
                 << " head=" << head_idx
                 << " query_index=" << query_index
                 << " prefix_len=" << causal_scores_row.size()
                 << " lse=" << lse_value
                 << " score[min=" << score_stats.min_val
                 << " max=" << score_stats.max_val
                 << " rms=" << score_stats.rms << ']'
                 << " alpha[sum=" << breadth_stats.alpha_sum
                 << " min=" << breadth_stats.alpha_min
                 << " max=" << breadth_stats.alpha_max << ']'
                 << " breadth[entropy=" << breadth_stats.entropy
                 << " normalized_entropy=" << breadth_stats.normalized_entropy
                 << " effective_support=" << breadth_stats.effective_support << '/' << causal_scores_row.size()
                 << " participation_ratio=" << breadth_stats.participation_ratio
                 << " uniform_fraction=" << breadth_stats.uniform_fraction << ']'
                 << " source_center[mean_source_index=" << breadth_stats.mean_source_index
                 << " normalized_mean_source_index=" << breadth_stats.normalized_mean_source_index << ']'
                 << " interpretation=\"" << interpretBreadth(breadth_stats, causal_scores_row.size()) << "\"";

     GRIM::Logging::LogEntry entry{};
     entry.level = GRIM::Logging::LogLevel::Info;
     entry.group = GRIM::Logging::LogGroup::Attention;
     entry.phase = GRIM::Logging::LogPhase::FLASH_ATTENTION_FWD;
     entry.layer_idx = static_cast<int16_t>(layer_idx);
     entry.global_step = tape->currentStep();
     entry.batch_idx = tape->currentBatch();
     entry.setTag("ATTN_BREADTH");
     entry.setMessageView(log_line.str());
     entry.primary = breadth_stats.normalized_entropy;
     entry.secondary = breadth_stats.uniform_fraction;
     tape->record(entry);
}

}  // namespace GRIM::FlashAttentionDiagnostics