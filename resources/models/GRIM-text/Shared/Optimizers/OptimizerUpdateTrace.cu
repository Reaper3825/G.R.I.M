#ifndef USE_CUDA
#define USE_CUDA
#endif
//======================================================//
//  OptimizerUpdateTrace.cu
//  Optimizer-boundary adaptive update magnitude diagnostics.
//======================================================//

#include "OptimizerUpdateTrace.hpp"

#include "../HyperParameters/HyperParameters_GPU.hpp"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <sstream>
#include <stdexcept>

namespace GRIM {
namespace {

struct OptimizerTraceFormula {
    float beta1 = 0.0f;
    float beta2 = 0.0f;
    float epsilon = 0.0f;
    float radam_rectifier = 1.0f;
    bool radam_rectified = false;
};

OptimizerTraceFormula resolveTraceFormula(const HyperParameters::OptimizerUpdateHP& hp,
                                          int optimizer_step) {
    if (optimizer_step < 0) {
        throw std::runtime_error("[computeOptimizerUpdateTrace] optimizer_step must be >= 0, got " +
                                 std::to_string(optimizer_step));
    }

    OptimizerTraceFormula formula{};
    switch (hp.kind) {
        case HyperParameters::OptimizerKind::UNSPECIFIED:
            throw std::runtime_error("[computeOptimizerUpdateTrace] optimizer kind is UNSPECIFIED");
        case HyperParameters::OptimizerKind::ADAMW:
            formula.beta1 = HyperParameters::ADAMW_BETA1;
            formula.beta2 = HyperParameters::ADAMW_BETA2;
            formula.epsilon = HyperParameters::ADAMW_EPSILON;
            return formula;
        case HyperParameters::OptimizerKind::RADAMW:
            formula.beta1 = hp.beta1;
            formula.beta2 = hp.beta2;
            formula.epsilon = hp.epsilon;
            break;
    }

    if (!(formula.beta1 > 0.0f && formula.beta1 < 1.0f)) {
        throw std::runtime_error("[computeOptimizerUpdateTrace] beta1 must be in (0,1), got " +
                                 std::to_string(formula.beta1));
    }
    if (!(formula.beta2 > 0.0f && formula.beta2 < 1.0f)) {
        throw std::runtime_error("[computeOptimizerUpdateTrace] beta2 must be in (0,1), got " +
                                 std::to_string(formula.beta2));
    }
    if (!(formula.epsilon > 0.0f) || !std::isfinite(formula.epsilon)) {
        throw std::runtime_error("[computeOptimizerUpdateTrace] epsilon must be finite and > 0, got " +
                                 std::to_string(formula.epsilon));
    }

    if (hp.kind == HyperParameters::OptimizerKind::RADAMW) {
        const int iteration = optimizer_step + 1;
        const float rho_inf = 2.0f / (1.0f - formula.beta2) - 1.0f;
        const float beta2_t = std::pow(formula.beta2, static_cast<float>(iteration));
        const float rho_t = rho_inf -
            2.0f * static_cast<float>(iteration) * beta2_t / (1.0f - beta2_t);
        formula.radam_rectified = rho_t > 4.0f;
        if (formula.radam_rectified) {
            const float num = (rho_t - 4.0f) * (rho_t - 2.0f) * rho_inf;
            const float den = (rho_inf - 4.0f) * (rho_inf - 2.0f) * rho_t;
            if (!(num > 0.0f) || !(den > 0.0f)) {
                throw std::runtime_error("[computeOptimizerUpdateTrace] invalid RAdam rectifier terms");
            }
            formula.radam_rectifier = std::sqrt(num / den);
            if (!std::isfinite(formula.radam_rectifier)) {
                throw std::runtime_error("[computeOptimizerUpdateTrace] RAdam rectifier is non-finite");
            }
        }
    }

    return formula;
}

float adaptiveUpdateForSample(float m_state,
                              float v_state,
                              float effective_lr,
                              float inv_bc1,
                              float inv_bc2,
                              const OptimizerTraceFormula& formula,
                              HyperParameters::OptimizerKind kind) {
    if (!std::isfinite(m_state) || !std::isfinite(v_state)) {
        throw std::runtime_error("[computeOptimizerUpdateTrace] sampled optimizer moment is non-finite");
    }
    if (v_state < 0.0f) {
        throw std::runtime_error("[computeOptimizerUpdateTrace] sampled second moment is negative");
    }

    const float m_hat = m_state * inv_bc1;
    if (kind == HyperParameters::OptimizerKind::RADAMW && !formula.radam_rectified) {
        return effective_lr * m_hat;
    }

    const float v_hat = v_state * inv_bc2;
    if (v_hat < 0.0f || !std::isfinite(v_hat)) {
        throw std::runtime_error("[computeOptimizerUpdateTrace] derived v_hat is invalid");
    }

    if (kind == HyperParameters::OptimizerKind::RADAMW) {
        return effective_lr * formula.radam_rectifier * m_hat / (std::sqrt(v_hat) + formula.epsilon);
    }

    return effective_lr * m_hat / std::sqrt(v_hat + formula.epsilon);
}

void appendComponentValues(std::ostringstream& oss,
                           const OptimizerUpdateTraceMetrics& metrics,
                           const float* values,
                           bool tied_embeddings) {
    if (tied_embeddings) {
        if (metrics.has_data[1]) oss << " emb_lm_tied=" << values[1];
    } else {
        if (metrics.has_data[0]) oss << " emb=" << values[0];
        if (metrics.has_data[1]) oss << " lm=" << values[1];
    }
    for (int t = 2; t < kOptimizerUpdateTraceTypeCount; ++t) {
        if (metrics.has_data[t]) {
            oss << " " << OptimizerUpdateTraceMetrics::typeName(t) << "=" << values[t];
        }
    }
}

} // namespace

const char* OptimizerUpdateTraceMetrics::typeName(int type_idx) {
    switch (static_cast<ParamGroupType>(type_idx)) {
        case ParamGroupType::EMBEDDING: return "emb";
        case ParamGroupType::LM_HEAD: return "lm";
        case ParamGroupType::ATTENTION: return "attn";
        case ParamGroupType::FFN: return "ffn";
        case ParamGroupType::RMSNORM: return "rmsnorm";
        case ParamGroupType::SCRATCHBLOCK: return "sb";
        case ParamGroupType::MTP: return "mtp";
        case ParamGroupType::EXECUTION_BLOCK: return "exec";
        case ParamGroupType::SLOT_SELECTOR: return "slot_selector";
        case ParamGroupType::COUNT: return "?";
    }
    return "?";
}

OptimizerUpdateTraceMetrics computeOptimizerUpdateTrace(
    const std::vector<ParameterGroup>& groups,
    const HyperParameters::OptimizerUpdateHP& hp,
    float learning_rate,
    int optimizer_step,
    cudaStream_t stream
) {
    if (groups.empty()) {
        throw std::runtime_error("[computeOptimizerUpdateTrace] parameter groups are empty");
    }
    if (!std::isfinite(learning_rate) || learning_rate < 0.0f) {
        throw std::runtime_error("[computeOptimizerUpdateTrace] learning_rate must be finite and >= 0, got " +
                                 std::to_string(learning_rate));
    }
    if (!stream) {
        throw std::runtime_error("[computeOptimizerUpdateTrace] stream is NULL");
    }

    OptimizerUpdateTraceMetrics result{};
    const OptimizerTraceFormula formula = resolveTraceFormula(hp, optimizer_step);
    const int iteration = optimizer_step + 1;
    const float bc1 = 1.0f - std::pow(formula.beta1, static_cast<float>(iteration));
    const float bc2 = 1.0f - std::pow(formula.beta2, static_cast<float>(iteration));
    if (bc1 <= 0.0f || bc2 <= 0.0f || !std::isfinite(bc1) || !std::isfinite(bc2)) {
        throw std::runtime_error("[computeOptimizerUpdateTrace] invalid bias correction");
    }
    const float inv_bc1 = 1.0f / bc1;
    const float inv_bc2 = 1.0f / bc2;

    bool any_sampled = false;
    bool type_sampled[kOptimizerUpdateTraceTypeCount] = {};
    const bool embedding_frozen = (hp.embedding_freeze_after_step >= 0) &&
                                  (optimizer_step >= hp.embedding_freeze_after_step);

    for (const auto& group : groups) {
        const int ti = static_cast<int>(group.type);
        if (ti < 0 || ti >= kOptimizerUpdateTraceTypeCount) {
            throw std::runtime_error("[computeOptimizerUpdateTrace] invalid ParamGroupType on group '" +
                                     group.name + "'");
        }
        if (type_sampled[ti]) continue;
        if (embedding_frozen && group.stats_bucket == ParamStatsBucket::EMBEDDING) continue;
        if (!group.weights() || !group.m_state() || !group.v_state() || group.size() == 0) continue;

        const int total = static_cast<int>(group.size());
        const int n = std::min(total, kOptimizerUpdateTraceSampleSize);
        if (n <= 0) continue;
        const int stride = (total > n) ? (total / n) : 1;
        const float effective_lr = learning_rate * group.lr_multiplier;

        float h_params[kOptimizerUpdateTraceSampleSize];
        float h_m[kOptimizerUpdateTraceSampleSize];
        float h_v[kOptimizerUpdateTraceSampleSize];

        if (stride <= 1) {
            cudaMemcpyAsync(h_params, group.weights(), n * sizeof(float), cudaMemcpyDeviceToHost, stream);
            cudaMemcpyAsync(h_m, group.m_state(), n * sizeof(float), cudaMemcpyDeviceToHost, stream);
            cudaMemcpyAsync(h_v, group.v_state(), n * sizeof(float), cudaMemcpyDeviceToHost, stream);
        } else {
            for (int s = 0; s < n; ++s) {
                const int idx = s * stride;
                cudaMemcpyAsync(&h_params[s], group.weights() + idx, sizeof(float), cudaMemcpyDeviceToHost, stream);
                cudaMemcpyAsync(&h_m[s], group.m_state() + idx, sizeof(float), cudaMemcpyDeviceToHost, stream);
                cudaMemcpyAsync(&h_v[s], group.v_state() + idx, sizeof(float), cudaMemcpyDeviceToHost, stream);
            }
        }
        const cudaError_t sync_err = cudaStreamSynchronize(stream);
        if (sync_err != cudaSuccess) {
            throw std::runtime_error(std::string("[computeOptimizerUpdateTrace] stream synchronize failed: ") +
                                     cudaGetErrorString(sync_err));
        }

        float update_sq_sum = 0.0f;
        float param_sq_sum = 0.0f;
        for (int i = 0; i < n; ++i) {
            if (!std::isfinite(h_params[i])) {
                throw std::runtime_error("[computeOptimizerUpdateTrace] sampled parameter is non-finite");
            }
            const float adaptive_update = adaptiveUpdateForSample(
                h_m[i], h_v[i], effective_lr, inv_bc1, inv_bc2, formula, hp.kind);
            update_sq_sum += adaptive_update * adaptive_update;
            param_sq_sum += h_params[i] * h_params[i];
        }

        result.adaptive_update_rms[ti] = std::sqrt(update_sq_sum / static_cast<float>(n));
        result.param_rms[ti] = std::sqrt(param_sq_sum / static_cast<float>(n));
        result.update_over_param[ti] = (result.param_rms[ti] > 1e-12f)
            ? (result.adaptive_update_rms[ti] / result.param_rms[ti])
            : 0.0f;
        result.element_count[ti] = n;
        result.has_data[ti] = true;
        type_sampled[ti] = true;
        any_sampled = true;
    }

    result.valid = any_sampled;
    return result;
}

std::vector<std::string> formatOptimizerUpdateTraceLines(
    const OptimizerUpdateTraceMetrics& metrics,
    int optimizer_step,
    bool tied_embeddings
) {
    std::vector<std::string> lines;
    if (!metrics.valid) return lines;

    const int iteration = optimizer_step + 1;

    std::ostringstream upd;
    upd << std::scientific << std::setprecision(10);
    upd << "[OptimizerUpdateTrace] COMPONENTS(adaptive_update_rms) optimizer_step=" << optimizer_step
        << " iteration=" << iteration;
    appendComponentValues(upd, metrics, metrics.adaptive_update_rms, tied_embeddings);
    lines.push_back(upd.str());

    std::ostringstream rel;
    rel << std::scientific << std::setprecision(10);
    rel << "[OptimizerUpdateTrace] COMPONENTS(adaptive_update/param) optimizer_step=" << optimizer_step
        << " iteration=" << iteration;
    appendComponentValues(rel, metrics, metrics.update_over_param, tied_embeddings);
    lines.push_back(rel.str());

    const int ffn_idx = static_cast<int>(ParamGroupType::FFN);
    if (metrics.has_data[ffn_idx] && metrics.adaptive_update_rms[ffn_idx] > 1e-15f) {
        std::ostringstream ratio;
        ratio << "[OptimizerUpdateTrace] RATIOS(vs_ffn) optimizer_step=" << optimizer_step
              << " iteration=" << iteration;
        for (int t = 0; t < kOptimizerUpdateTraceTypeCount; ++t) {
            if (!metrics.has_data[t] || t == ffn_idx) continue;
            if (tied_embeddings && t == static_cast<int>(ParamGroupType::EMBEDDING)) continue;
            const float value = metrics.adaptive_update_rms[t] / metrics.adaptive_update_rms[ffn_idx];
            ratio << " " << OptimizerUpdateTraceMetrics::typeName(t)
                  << "=" << std::fixed << std::setprecision(10) << value << "x";
        }
        lines.push_back(ratio.str());
    }

    return lines;
}

} // namespace GRIM
