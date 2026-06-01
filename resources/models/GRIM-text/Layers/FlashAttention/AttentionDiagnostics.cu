#include "AttentionDiagnostics.hpp"

#include "../../Shared/LogRecorder/LogRecorder.hpp"
#include "../../Shared/LogRecorder/BatchLogTape.hpp"
#include "../../Shared/VerboseLogging.hpp"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

struct FlashAttentionDiagLog {
    static void info(std::string_view msg, std::uint64_t step = 0) {
        GRIM::Logging::EmitModuleInfo(GRIM::Logging::ModuleId::Activations, msg, step);
    }
    static void error(std::string_view msg, std::uint64_t step = 0) {
        GRIM::Logging::EmitModuleError(GRIM::Logging::ModuleId::Activations, msg, step);
    }
};

bool shouldEmitAttentionDiagnostics() {
    if constexpr (!GRIM::VerboseLogging::ENABLE_FA_EQUATION_DIAGNOSTICS) {
        return false;
    }
    auto* tape = GRIM::Logging::getGlobalTape();
    return tape && tape->accepts(GRIM::Logging::LogLevel::Debug);
}

void requireCudaSuccess(cudaError_t err, const char* context) {
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string(context) + ": " + cudaGetErrorString(err));
    }
}

float bf16ToFloatHost(const __nv_bfloat16& value) {
    std::uint16_t raw = 0;
    std::memcpy(&raw, &value, sizeof(raw));
    const std::uint32_t bits = static_cast<std::uint32_t>(raw) << 16;
    float out = 0.0f;
    std::memcpy(&out, &bits, sizeof(out));
    return out;
}

float halfToFloatHost(const __half& value) {
    return __half2float(value);
}

template<typename StorageT>
float storageToFloatHost(const StorageT& value);

template<>
float storageToFloatHost(const __nv_bfloat16& value) {
    return bf16ToFloatHost(value);
}

template<>
float storageToFloatHost(const __half& value) {
    return halfToFloatHost(value);
}

template<typename StorageT>
std::vector<float> convertSamplesToFloat(const std::vector<StorageT>& raw) {
    std::vector<float> out(raw.size());
    for (size_t i = 0; i < raw.size(); ++i) {
        out[i] = storageToFloatHost(raw[i]);
    }
    return out;
}

template<typename StorageT>
std::vector<float> copyLinearDeviceSamplesAsFloat(const void* src,
                                                  size_t count,
                                                  const char* context) {
    if (!src && count > 0) {
        throw std::runtime_error(std::string(context) + ": src is NULL");
    }
    std::vector<StorageT> raw(count);
    if (count > 0) {
        requireCudaSuccess(cudaMemcpy(raw.data(), src, count * sizeof(StorageT), cudaMemcpyDeviceToHost),
                           context);
    }
    return convertSamplesToFloat(raw);
}

template<typename StorageT>
std::vector<float> copyMatrixPrefixAsFloat(const void* src,
                                           int src_cols,
                                           int rows_to_copy,
                                           int cols_to_copy,
                                           const char* context) {
    if (!src) {
        throw std::runtime_error(std::string(context) + ": src is NULL");
    }
    if (src_cols <= 0 || rows_to_copy <= 0 || cols_to_copy <= 0 || cols_to_copy > src_cols) {
        throw std::runtime_error(std::string(context) + ": invalid strided copy geometry");
    }
    std::vector<StorageT> raw(static_cast<size_t>(rows_to_copy) * static_cast<size_t>(cols_to_copy));
    requireCudaSuccess(
        cudaMemcpy2D(raw.data(),
                     static_cast<size_t>(cols_to_copy) * sizeof(StorageT),
                     src,
                     static_cast<size_t>(src_cols) * sizeof(StorageT),
                     static_cast<size_t>(cols_to_copy) * sizeof(StorageT),
                     rows_to_copy,
                     cudaMemcpyDeviceToHost),
        context);
    return convertSamplesToFloat(raw);
}

std::vector<float> copyLinearSamplesAsFloat(const void* src,
                                            size_t count,
                                            bool is_bf16,
                                            const char* context) {
    if (is_bf16) {
        return copyLinearDeviceSamplesAsFloat<__nv_bfloat16>(src, count, context);
    }
    return copyLinearDeviceSamplesAsFloat<__half>(src, count, context);
}

std::vector<float> copyMatrixPrefixAsFloat(const void* src,
                                           int src_cols,
                                           int rows_to_copy,
                                           int cols_to_copy,
                                           bool is_bf16,
                                           const char* context) {
    if (is_bf16) {
        return copyMatrixPrefixAsFloat<__nv_bfloat16>(src, src_cols, rows_to_copy, cols_to_copy, context);
    }
    return copyMatrixPrefixAsFloat<__half>(src, src_cols, rows_to_copy, cols_to_copy, context);
}

struct LinearStats {
    int nan_count = 0;
    int inf_count = 0;
    int zero_count = 0;
    int finite_count = 0;
    float first = 0.0f;
    float max_abs = 0.0f;
    float min_val = 0.0f;
    float max_val = 0.0f;
    float rms = 0.0f;
};

LinearStats computeLinearStats(const std::vector<float>& values) {
    LinearStats stats{};
    if (values.empty()) {
        return stats;
    }
    stats.first = values.front();
    bool saw_finite = false;
    double sum_sq = 0.0;
    for (float value : values) {
        if (std::isnan(value)) {
            ++stats.nan_count;
            continue;
        }
        if (std::isinf(value)) {
            ++stats.inf_count;
            continue;
        }
        if (!saw_finite) {
            stats.min_val = value;
            stats.max_val = value;
            saw_finite = true;
        } else {
            stats.min_val = std::min(stats.min_val, value);
            stats.max_val = std::max(stats.max_val, value);
        }
        if (value == 0.0f) {
            ++stats.zero_count;
        }
        stats.max_abs = std::max(stats.max_abs, std::fabs(value));
        sum_sq += static_cast<double>(value) * static_cast<double>(value);
        ++stats.finite_count;
    }
    if (stats.finite_count > 0) {
        stats.rms = std::sqrt(static_cast<float>(sum_sq / static_cast<double>(stats.finite_count)));
    }
    return stats;
}

bool isBoundarySequence(int seqlen) {
    return seqlen >= 920 || seqlen >= 1840;
}

std::vector<float> copyAlibiSlopesOrZeros(const float* slopes, int n_heads, const char* context) {
    std::vector<float> host(static_cast<size_t>(std::max(n_heads, 0)), 0.0f);
    if (slopes && n_heads > 0) {
        requireCudaSuccess(cudaMemcpy(host.data(), slopes, static_cast<size_t>(n_heads) * sizeof(float), cudaMemcpyDeviceToHost),
                           context);
    }
    return host;
}

void logSlopeRange(const std::vector<float>& slopes,
                   int call_count,
                   int seqlen,
                   int batch,
                   int n_heads) {
    if (slopes.empty()) {
        std::fprintf(stderr,
                     "[FA-FWD-ALIBI] call=%d seqlen=%d batch=%d n_heads=%d | slope_range=[0.000000, 0.000000] [no alibi]\n",
                     call_count, seqlen, batch, n_heads);
        return;
    }
    float min_slope = slopes.front();
    float max_slope = slopes.front();
    for (float slope : slopes) {
        min_slope = std::min(min_slope, slope);
        max_slope = std::max(max_slope, slope);
    }
    std::fprintf(stderr,
                 "[FA-FWD-ALIBI] call=%d seqlen=%d batch=%d n_heads=%d | slope_range=[%.6f, %.6f]",
                 call_count, seqlen, batch, n_heads, min_slope, max_slope);
    if (isBoundarySequence(seqlen)) {
        std::fprintf(stderr, " [*** BOUNDARY_SEQ seqlen=%d ***] slopes=[", seqlen);
        for (int h = 0; h < n_heads; ++h) {
            std::fprintf(stderr, "%.6f%s", slopes[static_cast<size_t>(h)], h < n_heads - 1 ? "," : "");
        }
        std::fprintf(stderr, "]");
    }
    std::fprintf(stderr, "\n");
}

void logLseSummary(const std::vector<float>& h_lse,
                   const char* header_tag,
                   const char* summary_tag,
                   const char* perhead_tag,
                   int batch,
                   int seqlen,
                   int n_heads,
                   bool include_call_count,
                   int call_count) {
    if (h_lse.empty()) {
        return;
    }

    if (include_call_count) {
        std::fprintf(stderr,
                     "%s call=%d seqlen=%d | ",
                     header_tag,
                     call_count,
                     seqlen);
    } else {
        std::fprintf(stderr,
                     "%s seqlen=%d batch=%d n_heads=%d total_elems=%zu\n",
                     header_tag,
                     seqlen,
                     batch,
                     n_heads,
                     h_lse.size());
    }

    float global_min = 0.0f;
    float global_max = 0.0f;
    bool saw_finite = false;
    int nan_count = 0;
    int inf_count = 0;
    double sum = 0.0;
    size_t finite_count = 0;
    for (float value : h_lse) {
        if (std::isnan(value)) {
            ++nan_count;
            continue;
        }
        if (std::isinf(value)) {
            ++inf_count;
            continue;
        }
        if (!saw_finite) {
            global_min = value;
            global_max = value;
            saw_finite = true;
        } else {
            global_min = std::min(global_min, value);
            global_max = std::max(global_max, value);
        }
        sum += value;
        ++finite_count;
    }
    const float mean = finite_count > 0 ? static_cast<float>(sum / static_cast<double>(finite_count)) : 0.0f;

    std::fprintf(stderr,
                 "%snan=%d inf=%d range=[%.4f, %.4f] mean=%.4f\n",
                 include_call_count ? "" : summary_tag,
                 nan_count,
                 inf_count,
                 global_min,
                 global_max,
                 mean);
    if (include_call_count) {
        // For backward we already printed the tag prefix above.
        std::fflush(stderr);
    }

    const bool has_anomaly = global_max > 50.0f || nan_count > 0 || inf_count > 0;
    if (!has_anomaly && !isBoundarySequence(seqlen)) {
        return;
    }

    if (include_call_count) {
        std::fprintf(stderr,
                     "%s %s%s per-head stats:\n",
                     perhead_tag,
                     has_anomaly ? "*** LSE ANOMALY *** " : "",
                     isBoundarySequence(seqlen) ? "(BOUNDARY_SEQ)" : "");
    } else {
        std::fprintf(stderr,
                     "%s %s head_stats:\n",
                     perhead_tag,
                     has_anomaly ? "*** ANOMALY ***" : "(boundary sequence)");
    }

    for (int h = 0; h < n_heads; ++h) {
        float head_min = std::numeric_limits<float>::max();
        float head_max = std::numeric_limits<float>::lowest();
        int head_nan = 0;
        int head_inf = 0;
        double head_sum = 0.0;
        size_t head_count = 0;
        for (int b = 0; b < batch; ++b) {
            for (int s = 0; s < seqlen; ++s) {
                const size_t idx = static_cast<size_t>(b) * static_cast<size_t>(n_heads) * static_cast<size_t>(seqlen)
                                 + static_cast<size_t>(h) * static_cast<size_t>(seqlen)
                                 + static_cast<size_t>(s);
                const float value = h_lse[idx];
                if (std::isnan(value)) {
                    ++head_nan;
                    continue;
                }
                if (std::isinf(value)) {
                    ++head_inf;
                    continue;
                }
                head_min = std::min(head_min, value);
                head_max = std::max(head_max, value);
                head_sum += value;
                ++head_count;
            }
        }
        const float head_mean = head_count > 0 ? static_cast<float>(head_sum / static_cast<double>(head_count)) : 0.0f;
        if (include_call_count) {
            const char* flag = head_max > 50.0f ? " *** EXPLOSION ***" : "";
            std::fprintf(stderr,
                         "    head[%d]: range=[%.4f, %.4f] mean=%.4f nan=%d inf=%d%s\n",
                         h, head_min, head_max, head_mean, head_nan, head_inf, flag);
        } else {
            std::fprintf(stderr,
                         "    head[%d]: range=[%.4f, %.4f] mean=%.4f nan=%d inf=%d\n",
                         h, head_min, head_max, head_mean, head_nan, head_inf);
        }
    }
}

void logBackwardStrideLayout(const GRIM::FlashAttentionDiagnostics::BackwardStrideLayout& strides) {
    char stride_msg[512];
    std::snprintf(stride_msg, sizeof(stride_msg),
                  "[FlashAttention] bwd strides: q(b=%lld r=%lld h=%lld) k(b=%lld r=%lld h=%lld) v(b=%lld r=%lld h=%lld) "
                  "o(b=%lld r=%lld h=%lld) do(b=%lld r=%lld h=%lld) dq(b=%lld r=%lld h=%lld) dk(b=%lld r=%lld h=%lld) dv(b=%lld r=%lld h=%lld)",
                  strides.q.batch,
                  strides.q.row,
                  strides.q.head,
                  strides.k.batch,
                  strides.k.row,
                  strides.k.head,
                  strides.v.batch,
                  strides.v.row,
                  strides.v.head,
                  strides.o.batch,
                  strides.o.row,
                  strides.o.head,
                  strides.dO.batch,
                  strides.dO.row,
                  strides.dO.head,
                  strides.dQ.batch,
                  strides.dQ.row,
                  strides.dQ.head,
                  strides.dK.batch,
                  strides.dK.row,
                  strides.dK.head,
                  strides.dV.batch,
                  strides.dV.row,
                  strides.dV.head);
    FlashAttentionDiagLog::info(stride_msg);
}

std::atomic<int> g_fwd_call_count{0};
std::atomic<int> g_bwd_call_count{0};

}  // namespace

namespace GRIM::FlashAttentionDiagnostics {

void emitForwardPreKernelDiagnostics(const ForwardDiagnosticRequest& request) {
    if (!shouldEmitAttentionDiagnostics()) {
        return;
    }
    requireCudaSuccess(cudaStreamSynchronize(request.stream), "emitForwardPreKernelDiagnostics: sync");

    const int call_count = g_fwd_call_count.fetch_add(1, std::memory_order_relaxed) + 1;
    const std::vector<float> h_slopes = copyAlibiSlopesOrZeros(request.alibi_slopes,
                                                               request.n_heads,
                                                               "emitForwardPreKernelDiagnostics: copy alibi slopes");
    logSlopeRange(h_slopes, call_count, request.seqlen, request.batch, request.n_heads);

    const size_t q_elems = static_cast<size_t>(request.batch) * request.n_heads * request.seqlen * request.head_dim;
    const size_t kv_elems = static_cast<size_t>(request.batch) * request.n_kv_heads * request.seqlen * request.head_dim;
    const size_t sample_size = 100;
    const std::vector<float> h_q = copyLinearSamplesAsFloat(request.q,
                                                            std::min(q_elems, sample_size),
                                                            request.is_bf16,
                                                            "emitForwardPreKernelDiagnostics: copy q sample");
    const std::vector<float> h_k = copyLinearSamplesAsFloat(request.k,
                                                            std::min(kv_elems, sample_size),
                                                            request.is_bf16,
                                                            "emitForwardPreKernelDiagnostics: copy k sample");
    const std::vector<float> h_v = copyLinearSamplesAsFloat(request.v,
                                                            std::min(kv_elems, sample_size),
                                                            request.is_bf16,
                                                            "emitForwardPreKernelDiagnostics: copy v sample");
    const LinearStats q_stats = computeLinearStats(h_q);
    const LinearStats k_stats = computeLinearStats(h_k);
    const LinearStats v_stats = computeLinearStats(h_v);
    std::fprintf(stderr,
                 "[FA-FWD-IN] Q: nan=%d inf=%d first=%.6f | K: nan=%d inf=%d first=%.6f | V: nan=%d inf=%d first=%.6f\n",
                 q_stats.nan_count, q_stats.inf_count, q_stats.first,
                 k_stats.nan_count, k_stats.inf_count, k_stats.first,
                 v_stats.nan_count, v_stats.inf_count, v_stats.first);

    const int sample_tokens = request.seqlen;
    if (sample_tokens <= 0 || request.head_dim <= 0) {
        return;
    }

    const std::vector<float> h_q_sample = copyMatrixPrefixAsFloat(request.q,
                                                                  request.n_heads * request.head_dim,
                                                                  sample_tokens,
                                                                  request.head_dim,
                                                                  request.is_bf16,
                                                                  "emitForwardPreKernelDiagnostics: copy q head0");
    const std::vector<float> h_k_sample = copyMatrixPrefixAsFloat(request.k,
                                                                  request.n_kv_heads * request.head_dim,
                                                                  sample_tokens,
                                                                  request.head_dim,
                                                                  request.is_bf16,
                                                                  "emitForwardPreKernelDiagnostics: copy k head0");

    float q_rms = 0.0f;
    float k_rms = 0.0f;
    float q_min = std::numeric_limits<float>::max();
    float q_max = std::numeric_limits<float>::lowest();
    float k_min = std::numeric_limits<float>::max();
    float k_max = std::numeric_limits<float>::lowest();
    for (float value : h_q_sample) {
        q_rms += value * value;
        q_min = std::min(q_min, value);
        q_max = std::max(q_max, value);
    }
    for (float value : h_k_sample) {
        k_rms += value * value;
        k_min = std::min(k_min, value);
        k_max = std::max(k_max, value);
    }
    q_rms = std::sqrt(q_rms / static_cast<float>(std::max<size_t>(h_q_sample.size(), 1)));
    k_rms = std::sqrt(k_rms / static_cast<float>(std::max<size_t>(h_k_sample.size(), 1)));

    std::vector<float> q_row_rms(static_cast<size_t>(sample_tokens), 0.0f);
    std::vector<float> k_row_rms(static_cast<size_t>(sample_tokens), 0.0f);
    for (int token = 0; token < sample_tokens; ++token) {
        float q_norm_sq = 0.0f;
        float k_norm_sq = 0.0f;
        for (int d = 0; d < request.head_dim; ++d) {
            q_norm_sq += h_q_sample[static_cast<size_t>(token) * request.head_dim + d]
                       * h_q_sample[static_cast<size_t>(token) * request.head_dim + d];
            k_norm_sq += h_k_sample[static_cast<size_t>(token) * request.head_dim + d]
                       * h_k_sample[static_cast<size_t>(token) * request.head_dim + d];
        }
        q_row_rms[static_cast<size_t>(token)] = std::sqrt(q_norm_sq / static_cast<float>(request.head_dim));
        k_row_rms[static_cast<size_t>(token)] = std::sqrt(k_norm_sq / static_cast<float>(request.head_dim));
    }

    float q_rms_mean = 0.0f;
    float k_rms_mean = 0.0f;
    for (int token = 0; token < sample_tokens; ++token) {
        q_rms_mean += q_row_rms[static_cast<size_t>(token)];
        k_rms_mean += k_row_rms[static_cast<size_t>(token)];
    }
    q_rms_mean /= static_cast<float>(sample_tokens);
    k_rms_mean /= static_cast<float>(sample_tokens);

    std::vector<float> scores_sample(static_cast<size_t>(sample_tokens) * static_cast<size_t>(sample_tokens), 0.0f);
    float score_min = std::numeric_limits<float>::max();
    float score_max = std::numeric_limits<float>::lowest();
    double score_sum_sq = 0.0;
    const float slope0 = h_slopes.empty() ? 0.0f : h_slopes.front();
    for (int qi = 0; qi < sample_tokens; ++qi) {
        for (int ki = 0; ki <= qi; ++ki) {
            float dot = 0.0f;
            for (int d = 0; d < request.head_dim; ++d) {
                dot += h_q_sample[static_cast<size_t>(qi) * request.head_dim + d]
                     * h_k_sample[static_cast<size_t>(ki) * request.head_dim + d];
            }
            const float score = dot * request.softmax_scale;
            const float alibi_bias = slope0 * static_cast<float>(ki - qi);
            const float final_score = score + alibi_bias;
            scores_sample[static_cast<size_t>(qi) * static_cast<size_t>(sample_tokens) + static_cast<size_t>(ki)] = final_score;
            score_min = std::min(score_min, final_score);
            score_max = std::max(score_max, final_score);
            score_sum_sq += static_cast<double>(final_score) * static_cast<double>(final_score);
        }
    }

    const int valid_scores = (sample_tokens * (sample_tokens + 1)) / 2;
    const float score_rms = std::sqrt(static_cast<float>(score_sum_sq / std::max(valid_scores, 1)));
    std::vector<float> expected_lse(static_cast<size_t>(sample_tokens), 0.0f);
    for (int qi = 0; qi < sample_tokens; ++qi) {
        float max_score_row = std::numeric_limits<float>::lowest();
        for (int ki = 0; ki <= qi; ++ki) {
            max_score_row = std::max(max_score_row,
                                     scores_sample[static_cast<size_t>(qi) * static_cast<size_t>(sample_tokens) + static_cast<size_t>(ki)]);
        }
        double sum_exp = 0.0;
        for (int ki = 0; ki <= qi; ++ki) {
            sum_exp += std::exp(static_cast<double>(scores_sample[static_cast<size_t>(qi) * static_cast<size_t>(sample_tokens) + static_cast<size_t>(ki)] - max_score_row));
        }
        expected_lse[static_cast<size_t>(qi)] = static_cast<float>(std::log(sum_exp) + max_score_row);
    }

    float expected_lse_min = expected_lse.front();
    float expected_lse_max = expected_lse.front();
    double expected_lse_sum = 0.0;
    for (float value : expected_lse) {
        expected_lse_min = std::min(expected_lse_min, value);
        expected_lse_max = std::max(expected_lse_max, value);
        expected_lse_sum += value;
    }
    const float expected_lse_mean = static_cast<float>(expected_lse_sum / static_cast<double>(expected_lse.size()));
    const float expected_score_magnitude = q_rms_mean * k_rms_mean * request.head_dim * request.softmax_scale;
    const float default_scale = 1.0f / std::sqrt(static_cast<float>(request.head_dim));

    std::fprintf(stderr, "\n[ATTN_SCORE_EQUATION] FLASH_ATTENTION_FWD: score = (Q @ K^T) / sqrt(head_dim) + alibi_bias\n");
    std::fprintf(stderr,
                 "  Q (sample %d tokens, head 0): shape=[%d,%d] min=%.4f max=%.4f rms=%.4f\n",
                 sample_tokens, sample_tokens, request.head_dim, q_min, q_max, q_rms);
    std::fprintf(stderr,
                 "  K (sample %d tokens, head 0): shape=[%d,%d] min=%.4f max=%.4f rms=%.4f\n",
                 sample_tokens, sample_tokens, request.head_dim, k_min, k_max, k_rms);
    std::fprintf(stderr,
                 "  Q_row_rms: mean=%.4f | K_row_rms: mean=%.4f\n",
                 q_rms_mean,
                 k_rms_mean);
    std::fprintf(stderr,
                 "  scale = %.6f (default 1/sqrt(%d) = %.6f)\n",
                 request.softmax_scale,
                 request.head_dim,
                 default_scale);
    std::fprintf(stderr,
                 "  alibi_slope[head0] = %.6f (max_distance=%d -> max_bias=%.4f)\n",
                 slope0,
                 request.seqlen - 1,
                 slope0 * static_cast<float>(request.seqlen - 1));
    std::fprintf(stderr,
                 "  EXPECTED score = Q_row_rms * K_row_rms * head_dim * scale = %.4f * %.4f * %d * %.6f ≈ %.4f\n",
                 q_rms_mean,
                 k_rms_mean,
                 request.head_dim,
                 request.softmax_scale,
                 expected_score_magnitude);
    std::fprintf(stderr,
                 "  ACTUAL score (FULL SEQUENCE %d positions, %d samples): min=%.4f max=%.4f rms=%.4f\n",
                 sample_tokens,
                 valid_scores,
                 score_min,
                 score_max,
                 score_rms);
    std::fprintf(stderr,
                 "  EXPECTED LSE (from sampled scores): min=%.4f max=%.4f mean=%.4f\n",
                 expected_lse_min,
                 expected_lse_max,
                 expected_lse_mean);
    std::fprintf(stderr, "  (Normal LSE for random init: ~6-10, depending on seqlen)\n");
    if (expected_score_magnitude > 20.0f) {
        std::fprintf(stderr,
                     "  *** ANOMALY: Expected score magnitude %.2f >> 20! Q/K vectors too large! ***\n",
                     expected_score_magnitude);
        std::fprintf(stderr,
                     "      For head_dim=%d, Q_rms and K_rms should each be ~%.3f for score~1.0\n",
                     request.head_dim,
                     1.0f / std::pow(static_cast<float>(request.head_dim), 0.25f));
    }
    if (score_max > 100.0f) {
        std::fprintf(stderr,
                     "  *** ANOMALY: score_max=%.2f >> 100! This will cause LSE explosion! ***\n",
                     score_max);
    }
    if (expected_lse_max > 50.0f) {
        std::fprintf(stderr,
                     "  *** ANOMALY: expected_lse_max=%.2f >> 50! Softmax will saturate! ***\n",
                     expected_lse_max);
    }
    std::fprintf(stderr, "\n");
}

void emitForwardPostKernelDiagnostics(const ForwardDiagnosticRequest& request) {
    if (!shouldEmitAttentionDiagnostics()) {
        return;
    }
    requireCudaSuccess(cudaStreamSynchronize(request.stream), "emitForwardPostKernelDiagnostics: sync");

    const size_t out_elems = static_cast<size_t>(request.batch) * request.n_heads * request.seqlen * request.head_dim;
    const size_t out_sample_size = std::min(out_elems, static_cast<size_t>(200));
    const std::vector<float> h_out = copyLinearSamplesAsFloat(request.out,
                                                              out_sample_size,
                                                              request.is_bf16,
                                                              "emitForwardPostKernelDiagnostics: copy out sample");
    const LinearStats out_stats = computeLinearStats(h_out);
    std::fprintf(stderr,
                 "[FA-FWD-OUT] output(%s→fp32): nan=%d zero=%d/%zu first=%.6f\n",
                 request.is_bf16 ? "bf16" : "fp16",
                 out_stats.nan_count,
                 out_stats.zero_count,
                 out_sample_size,
                 out_stats.first);

    const size_t lse_elems = static_cast<size_t>(request.batch) * request.n_heads * request.seqlen;
    std::vector<float> h_lse(lse_elems);
    requireCudaSuccess(cudaMemcpy(h_lse.data(), request.softmax_lse, lse_elems * sizeof(float), cudaMemcpyDeviceToHost),
                       "emitForwardPostKernelDiagnostics: copy lse");
    logLseSummary(h_lse,
                  "[FA-FWD-LSE]",
                  "[FA-FWD-LSE-SUMMARY] ",
                  "[FA-FWD-LSE-PERHEAD]",
                  request.batch,
                  request.seqlen,
                  request.n_heads,
                  false,
                  0);
}

void emitBackwardPreKernelDiagnostics(const BackwardDiagnosticRequest& request) {
    logBackwardStrideLayout(request.strides);
    if (!shouldEmitAttentionDiagnostics()) {
        return;
    }
    requireCudaSuccess(cudaStreamSynchronize(request.stream), "emitBackwardPreKernelDiagnostics: sync");

    const int call_count = g_bwd_call_count.fetch_add(1, std::memory_order_relaxed) + 1;
    if (request.alibi_slopes) {
        float alibi0 = 0.0f;
        requireCudaSuccess(cudaMemcpy(&alibi0, request.alibi_slopes, sizeof(float), cudaMemcpyDeviceToHost),
                           "emitBackwardPreKernelDiagnostics: copy alibi slope");
        char slope_msg[256];
        std::snprintf(slope_msg,
                      sizeof(slope_msg),
                      "[FlashAttention] flash_attn_bwd_ex called with alibi_slopes[0]=%f (n_heads=%d, n_kv_heads=%d)",
                      alibi0,
                      request.n_heads,
                      request.n_kv_heads);
        FlashAttentionDiagLog::info(slope_msg);
    }

    const size_t lse_elems = static_cast<size_t>(request.batch) * request.n_heads * request.seqlen;
    std::vector<float> h_lse(lse_elems);
    requireCudaSuccess(cudaMemcpy(h_lse.data(), request.softmax_lse, lse_elems * sizeof(float), cudaMemcpyDeviceToHost),
                       "emitBackwardPreKernelDiagnostics: copy saved lse");
    logLseSummary(h_lse,
                  "[FA-BWD-SAVED-LSE]",
                  "",
                  "[FA-BWD-SAVED-LSE-PERHEAD]",
                  request.batch,
                  request.seqlen,
                  request.n_heads,
                  true,
                  call_count);

    const size_t grad_elems = static_cast<size_t>(request.batch) * request.n_heads * request.seqlen * request.head_dim;
    const size_t grad_sample_size = std::min(grad_elems, static_cast<size_t>(10000));
    const std::vector<float> h_dout = copyLinearSamplesAsFloat(request.dout,
                                                               grad_sample_size,
                                                               request.is_bf16,
                                                               "emitBackwardPreKernelDiagnostics: copy grad_output sample");
    const LinearStats dout_stats = computeLinearStats(h_dout);
    std::fprintf(stderr,
                 "[FA-BWD-IN] grad_output (%s): nan=%d inf=%d max=%.10f rms=%.10f first=%.10f\n",
                 request.is_bf16 ? "BF16" : "FP16",
                 dout_stats.nan_count,
                 dout_stats.inf_count,
                 dout_stats.max_abs,
                 dout_stats.rms,
                 dout_stats.first);
}

void emitBackwardPostKernelDiagnostics(const BackwardDiagnosticRequest& request) {
    if (!shouldEmitAttentionDiagnostics()) {
        return;
    }
    requireCudaSuccess(cudaStreamSynchronize(request.stream), "emitBackwardPostKernelDiagnostics: sync");

    const size_t tensor_elems = static_cast<size_t>(request.batch) * request.seqlen * request.n_heads * request.head_dim;
    const size_t sample_size = std::min(tensor_elems, static_cast<size_t>(10000));
    const std::vector<float> h_dq = copyLinearSamplesAsFloat(request.dq,
                                                             sample_size,
                                                             request.is_bf16,
                                                             "emitBackwardPostKernelDiagnostics: copy dq sample");
    const std::vector<float> h_dk = copyLinearSamplesAsFloat(request.dk,
                                                             sample_size,
                                                             request.is_bf16,
                                                             "emitBackwardPostKernelDiagnostics: copy dk sample");
    const std::vector<float> h_dv = copyLinearSamplesAsFloat(request.dv,
                                                             sample_size,
                                                             request.is_bf16,
                                                             "emitBackwardPostKernelDiagnostics: copy dv sample");
    const LinearStats dq_stats = computeLinearStats(h_dq);
    const LinearStats dk_stats = computeLinearStats(h_dk);
    const LinearStats dv_stats = computeLinearStats(h_dv);

    std::fprintf(stderr,
                 "[FA-BWD-OUT] dQ (%s): nan=%d inf=%d max=%.10f rms=%.10f first=%.10f\n",
                 request.is_bf16 ? "BF16" : "FP16",
                 dq_stats.nan_count,
                 dq_stats.inf_count,
                 dq_stats.max_abs,
                 dq_stats.rms,
                 dq_stats.first);
    std::fprintf(stderr,
                 "[FA-BWD-OUT] actual tensor elems: dq=%zu dk=%zu dv=%zu (sampled %zu each) head_dim=%d\n",
                 tensor_elems,
                 tensor_elems,
                 tensor_elems,
                 sample_size,
                 request.head_dim);
    std::fprintf(stderr,
                 "[FA-BWD-OUT] dK (%s): nan=%d inf=%d max=%.10f rms=%.10f first=%.10f (n_heads=%d buffer)\n",
                 request.is_bf16 ? "BF16" : "FP16",
                 dk_stats.nan_count,
                 dk_stats.inf_count,
                 dk_stats.max_abs,
                 dk_stats.rms,
                 dk_stats.first,
                 request.n_heads);
    std::fprintf(stderr,
                 "[FA-BWD-OUT] dV (%s): nan=%d inf=%d max=%.10f rms=%.10f first=%.10f (n_heads=%d buffer)\n",
                 request.is_bf16 ? "BF16" : "FP16",
                 dv_stats.nan_count,
                 dv_stats.inf_count,
                 dv_stats.max_abs,
                 dv_stats.rms,
                 dv_stats.first,
                 request.n_heads);
}

}  // namespace GRIM::FlashAttentionDiagnostics