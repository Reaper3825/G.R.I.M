//======================================================//
//  LMHeadGemmDiagnostics.cu
//  Optional LM-head GEMM equation tracing isolated from production kernels.
//======================================================//

#include "LMHeadGemmDiagnostics.hpp"

#include "AutogradQKVDiagnostics.hpp"
#include "../LogRecorder/BatchLogTape.hpp"
#include "../VerboseLogging.hpp"

#include <algorithm>
#include <cstddef>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct MatrixSampleStats {
    float min_val = 0.0f;
    float max_val = 0.0f;
    float rms = 0.0f;
    float row_mean_abs_max = 0.0f;
    float row_mean_rms = 0.0f;
    int finite_count = 0;
    int nan_count = 0;
    int inf_count = 0;
};

void requireCudaSuccess(cudaError_t err, const char* context) {
    if (!context || !*context) {
        throw std::runtime_error("requireCudaSuccess: context is NULL or empty");
    }
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string(context) + ": " + cudaGetErrorString(err));
    }
}

bool shouldLogLmHeadGemmEquation() {
    if (!GRIM::VerboseLogging::ENABLE_GEMM_EQUATION_LOGS) {
        return false;
    }
    auto* tape = GRIM::Logging::getGlobalTape();
    return tape && tape->accepts(GRIM::Logging::LogLevel::Debug) && !tape->skipThisPass();
}

bool isLmHeadWeightName(const char* name) {
    return name && std::strstr(name, "lm_head") != nullptr;
}

const char* requireText(const char* text, const char* context) {
    if (!context || !*context) {
        throw std::runtime_error("requireText: context is NULL or empty");
    }
    if (!text || !*text) {
        throw std::runtime_error(std::string(context) + ": text is NULL or empty");
    }
    return text;
}

const char* boolText(bool value) {
    if (value) {
        return "true";
    }
    return "false";
}

MatrixSampleStats computeMatrixSampleStats(const std::vector<float>& values, int rows, int cols) {
    if (rows <= 0 || cols <= 0) {
        throw std::runtime_error("computeMatrixSampleStats: rows and cols must be positive");
    }
    if (values.size() != static_cast<size_t>(rows) * cols) {
        throw std::runtime_error("computeMatrixSampleStats: values size does not match rows*cols");
    }

    MatrixSampleStats stats{};
    double sum_sq = 0.0;
    double row_mean_sq_sum = 0.0;
    bool saw_finite = false;

    for (int r = 0; r < rows; ++r) {
        double row_sum = 0.0;
        int row_finite = 0;
        for (int c = 0; c < cols; ++c) {
            const float v = values[static_cast<size_t>(r) * cols + c];
            if (std::isnan(v)) {
                ++stats.nan_count;
                continue;
            }
            if (std::isinf(v)) {
                ++stats.inf_count;
                continue;
            }
            if (!saw_finite) {
                stats.min_val = v;
                stats.max_val = v;
                saw_finite = true;
            } else {
                stats.min_val = std::min(stats.min_val, v);
                stats.max_val = std::max(stats.max_val, v);
            }
            sum_sq += static_cast<double>(v) * static_cast<double>(v);
            row_sum += static_cast<double>(v);
            ++row_finite;
            ++stats.finite_count;
        }
        if (row_finite > 0) {
            const double row_mean = row_sum / static_cast<double>(row_finite);
            const double row_mean_abs = std::abs(row_mean);
            stats.row_mean_abs_max = std::max(stats.row_mean_abs_max, static_cast<float>(row_mean_abs));
            row_mean_sq_sum += row_mean * row_mean;
        }
    }

    if (stats.finite_count > 0) {
        stats.rms = static_cast<float>(std::sqrt(sum_sq / static_cast<double>(stats.finite_count)));
        stats.row_mean_rms = static_cast<float>(std::sqrt(row_mean_sq_sum / static_cast<double>(rows)));
    } else {
        stats.min_val = std::numeric_limits<float>::quiet_NaN();
        stats.max_val = std::numeric_limits<float>::quiet_NaN();
        stats.rms = std::numeric_limits<float>::quiet_NaN();
        stats.row_mean_abs_max = std::numeric_limits<float>::quiet_NaN();
        stats.row_mean_rms = std::numeric_limits<float>::quiet_NaN();
    }
    return stats;
}

std::vector<float> copyMatrixPrefix2D(const float* src,
                                      int src_cols,
                                      int rows_to_copy,
                                      int cols_to_copy,
                                      cudaStream_t stream,
                                      const char* context) {
    if (!src) {
        throw std::runtime_error(std::string(requireText(context, "copyMatrixPrefix2D")) + ": src is NULL");
    }
    if (src_cols <= 0 || rows_to_copy <= 0 || cols_to_copy <= 0 || cols_to_copy > src_cols) {
        throw std::runtime_error(std::string(requireText(context, "copyMatrixPrefix2D")) +
                                 ": invalid matrix copy dimensions");
    }
    std::vector<float> host(static_cast<size_t>(rows_to_copy) * cols_to_copy);
    requireCudaSuccess(cudaMemcpy2DAsync(
                           host.data(),
                           static_cast<size_t>(cols_to_copy) * sizeof(float),
                           src,
                           static_cast<size_t>(src_cols) * sizeof(float),
                           static_cast<size_t>(cols_to_copy) * sizeof(float),
                           rows_to_copy,
                           cudaMemcpyDeviceToHost,
                           stream),
                       context);
    return host;
}

}  // namespace

namespace GRIM::autograd {

void logLmHeadGemmForwardEquation(const Tensor& lm_input,
                                  const Tensor& effective_weights,
                                  const Tensor& logits,
                                  bool center_hidden_states,
                                  bool project_out_pc1,
                                  bool used_centered_weights,
                                  int total_tokens,
                                  int d_model,
                                  int vocab_size,
                                  cudaStream_t stream) {
    if (!shouldLogLmHeadGemmEquation()) {
        return;
    }
    if (!lm_input.data) {
        throw std::runtime_error("logLmHeadGemmForwardEquation: lm_input.data is NULL");
    }
    if (!effective_weights.data) {
        throw std::runtime_error("logLmHeadGemmForwardEquation: effective_weights.data is NULL");
    }
    if (!logits.data) {
        throw std::runtime_error("logLmHeadGemmForwardEquation: logits.data is NULL");
    }
    if (!stream) {
        throw std::runtime_error("logLmHeadGemmForwardEquation: stream is NULL");
    }
    if (total_tokens <= 0 || d_model <= 0 || vocab_size <= 0) {
        throw std::runtime_error("logLmHeadGemmForwardEquation: invalid tensor dimensions");
    }

    const int hidden_rows = std::min(total_tokens, 128);
    const int weight_rows = std::min(vocab_size, 512);
    const int logit_rows = std::min(total_tokens, 64);
    const int logit_cols = std::min(vocab_size, 256);

    std::vector<float> h_hidden = copyMatrixPrefix2D(
        lm_input.data, d_model, hidden_rows, d_model, stream,
        "logLmHeadGemmForwardEquation: copy lm_input sample");
    std::vector<float> h_weights = copyMatrixPrefix2D(
        effective_weights.data, d_model, weight_rows, d_model, stream,
        "logLmHeadGemmForwardEquation: copy effective_weights sample");
    std::vector<float> h_logits = copyMatrixPrefix2D(
        logits.data, vocab_size, logit_rows, logit_cols, stream,
        "logLmHeadGemmForwardEquation: copy logits sample");
    requireCudaSuccess(cudaStreamSynchronize(stream), "logLmHeadGemmForwardEquation: synchronize samples");

    const MatrixSampleStats hidden_stats = computeMatrixSampleStats(h_hidden, hidden_rows, d_model);
    const MatrixSampleStats weight_stats = computeMatrixSampleStats(h_weights, weight_rows, d_model);
    const MatrixSampleStats logit_stats = computeMatrixSampleStats(h_logits, logit_rows, logit_cols);

    const float expected_logit_rms = std::sqrt(static_cast<float>(d_model)) * hidden_stats.rms * weight_stats.rms;
    float ratio = std::numeric_limits<float>::quiet_NaN();
    if (expected_logit_rms > 1e-12f) {
        ratio = logit_stats.rms / expected_logit_rms;
    }

    std::string lm_input_expr = "RMSNorm(h)";
    if (center_hidden_states) {
        lm_input_expr = "center_columns_by_sequence_lengths(RMSNorm(h))";
    }
    if (project_out_pc1) {
        lm_input_expr = "project_out_pc1(" + lm_input_expr + ")";
    }
    std::string w_eff_expr = "W_lm";
    if (used_centered_weights) {
        w_eff_expr = "center_rows(W_lm)";
    }

    std::ostringstream eq;
    eq.setf(std::ios::fixed);
    eq.precision(8);
    eq << "[LM_HEAD_GEMM_EQUATION] logits = lm_input @ W_eff^T\n";
    eq << "  ORDER: lm_input=" << lm_input_expr << "; W_eff=" << w_eff_expr << "\n";
    eq << "  GEOMETRY: lm_input_actual_shape=[" << total_tokens << "," << d_model << "]"
       << " W_eff_actual_shape=[" << vocab_size << "," << d_model << "]"
       << " logits_actual_shape=[" << total_tokens << "," << vocab_size << "]"
       << " lm_input_sample_shape=[" << hidden_rows << "," << d_model << "]"
       << " W_eff_sample_shape=[" << weight_rows << "," << d_model << "]"
       << " logits_sample_shape=[" << logit_rows << "," << logit_cols << "]"
       << " lm_input_rows_clamped=" << boolText(hidden_rows != total_tokens)
       << " W_eff_rows_clamped=" << boolText(weight_rows != vocab_size)
       << " logits_rows_clamped=" << boolText(logit_rows != total_tokens)
       << " logits_cols_clamped=" << boolText(logit_cols != vocab_size) << "\n";
    eq << "  INPUT (lm_input sample): sample_shape=[" << hidden_rows << "," << d_model << "]"
       << " min=" << hidden_stats.min_val << " max=" << hidden_stats.max_val
       << " rms=" << hidden_stats.rms
       << " row_mean_abs_max=" << hidden_stats.row_mean_abs_max
       << " nan=" << hidden_stats.nan_count << " inf=" << hidden_stats.inf_count << "\n";
    eq << "  WEIGHT (W_eff sample): sample_shape=[" << weight_rows << "," << d_model << "]"
       << " min=" << weight_stats.min_val << " max=" << weight_stats.max_val
       << " rms=" << weight_stats.rms
       << " row_mean_abs_max=" << weight_stats.row_mean_abs_max
       << " nan=" << weight_stats.nan_count << " inf=" << weight_stats.inf_count << "\n";
    eq << "  EXPECTED sample_logit_rms ≈ sqrt(d_model) * rms(lm_input) * rms(W_eff) = "
       << expected_logit_rms << "\n";
    eq << "  ACTUAL (logits prefix sample): sample_shape=[" << logit_rows << "," << logit_cols << "]"
       << " min=" << logit_stats.min_val << " max=" << logit_stats.max_val
       << " rms=" << logit_stats.rms << " ratio=" << ratio
       << " nan=" << logit_stats.nan_count << " inf=" << logit_stats.inf_count << "\n";
    if (used_centered_weights && weight_stats.row_mean_abs_max > 1e-4f) {
        eq << "  [ANOMALY] W_eff row means are not near zero after center_rows(W_lm): max_abs="
           << weight_stats.row_mean_abs_max << "\n";
    }
    if (std::isfinite(ratio) && ratio > 3.0f) {
        eq << "  [ANOMALY] actual/expected logit RMS ratio=" << ratio
           << " suggests h↔W alignment or hidden-state collapse at the LM GEMM boundary\n";
    }
    if (hidden_stats.nan_count > 0 || hidden_stats.inf_count > 0 ||
        weight_stats.nan_count > 0 || weight_stats.inf_count > 0 ||
        logit_stats.nan_count > 0 || logit_stats.inf_count > 0) {
        eq << "  [ANOMALY] non-finite value observed in LM-head GEMM sample\n";
    }

    const std::string body = eq.str();
    std::fprintf(stderr, "%s", body.c_str());
    std::fflush(stderr);
    EQ_LOG(GRIM::Logging::getGlobalTape(), GRIM::Logging::LogGroup::LMHead,
           GRIM::Logging::LogPhase::LM_HEAD_PROJECTION, -1,
           "LM_HEAD_GEMM_EQUATION", body.c_str());
}

void logLmHeadGemmBackwardEquation(const Tensor& grad_output,
                                   const float* grad_lm_input,
                                   const float* grad_w_eff,
                                   bool lm_input_requires_grad,
                                   bool w_eff_requires_grad,
                                   const char* lm_input_name,
                                   const char* w_eff_name,
                                   int M,
                                   int K,
                                   int N,
                                   bool transpose_b,
                                   cudaStream_t stream) {
    if (!shouldLogLmHeadGemmEquation() || !isLmHeadWeightName(w_eff_name)) {
        return;
    }
    requireText(lm_input_name, "logLmHeadGemmBackwardEquation: lm_input_name");
    requireText(w_eff_name, "logLmHeadGemmBackwardEquation: w_eff_name");

    std::ostringstream eq;
    eq << "[LM_HEAD_GEMM_BACKWARD_EQUATION] logits = lm_input @ W_eff^T\n";
    eq << "  grad_lm_input = grad_logits @ W_eff\n";
    eq << "  grad_W_eff = grad_logits^T @ lm_input\n";
    eq << "  SHAPES: grad_logits=[" << M << "," << N << "]"
       << " W_eff=[" << N << "," << K << "]"
       << " grad_lm_input=[" << M << "," << K << "]"
       << " grad_W_eff=[" << N << "," << K << "]"
         << " transpose_b=" << boolText(transpose_b) << "\n";
     eq << "  NAMES: A=" << lm_input_name << " B=" << w_eff_name << "\n";
    const std::string body = eq.str();
    std::fprintf(stderr, "%s", body.c_str());
    std::fflush(stderr);
    EQ_LOG(GRIM::Logging::getGlobalTape(), GRIM::Logging::LogGroup::LMHead,
           GRIM::Logging::LogPhase::LM_HEAD_BACKWARD, -1,
           "LM_HEAD_GEMM_BACKWARD_EQUATION", body.c_str());

    logGradFlowTensorStats("LM_HEAD_GEMM_BWD grad_logits", grad_output.data,
                           grad_output.numel(), stream, true);
    if (lm_input_requires_grad && grad_lm_input) {
        logGradFlowTensorStats("LM_HEAD_GEMM_BWD grad_lm_input_pre_centering",
                               grad_lm_input,
                               static_cast<std::size_t>(M) * K,
                               stream,
                               true);
    }
    if (w_eff_requires_grad && grad_w_eff) {
        logGradFlowTensorStats("LM_HEAD_GEMM_BWD grad_W_eff_pre_center_rows",
                               grad_w_eff,
                               static_cast<std::size_t>(N) * K,
                               stream,
                               true);
    }
}

}  // namespace GRIM::autograd