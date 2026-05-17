//======================================================//
//  AutogradQKVDiagnostics.cu
//  QKV projection equation diagnostics and NaN/Inf scanners.
//======================================================//

#include "AutogradQKVDiagnostics.hpp"
#include "../CudaAllocUtils.hpp"
#include "../LogRecorder/BatchLogTape.hpp"
#include "../LogRecorder/LogRecorder.hpp"

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using GRIM::CudaAlloc::cudaMallocOrThrow;

const char* requireTag(const char* tag, const char* context) {
    if (!context || !*context) {
        throw std::runtime_error("requireTag: context is NULL or empty");
    }
    if (!tag || !*tag) {
        throw std::runtime_error(std::string(context) + ": diagnostic tag is NULL or empty");
    }
    return tag;
}

void checkCuda(cudaError_t err, const char* context) {
    if (!context || !*context) {
        throw std::runtime_error("checkCuda: context is NULL or empty");
    }
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string(context) + ": " + cudaGetErrorString(err));
    }
}

struct NonFiniteStats {
    int nan_count;
    int inf_count;
    int first_nan_idx;
    int first_inf_idx;
    float first_nan_val;
    float first_inf_val;
};

struct GradFlowBlockStats {
    float sum_sq;
    float max_abs;
    int nan_count;
    int inf_count;
};

__global__ void scanNonFiniteKernel(const float* data, int count, NonFiniteStats* stats) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) {
        return;
    }
    const float v = data[idx];
    if (isnan(v)) {
        atomicAdd(&stats->nan_count, 1);
        const int old = atomicCAS(&stats->first_nan_idx, -1, idx);
        if (old == -1) {
            stats->first_nan_val = v;
        }
    } else if (isinf(v)) {
        atomicAdd(&stats->inf_count, 1);
        const int old = atomicCAS(&stats->first_inf_idx, -1, idx);
        if (old == -1) {
            stats->first_inf_val = v;
        }
    }
}

__global__ void gradFlowStatsKernel(const float* data,
                                    std::size_t count,
                                    GradFlowBlockStats* partials) {
    __shared__ float s_sum_sq[256];
    __shared__ float s_max_abs[256];
    __shared__ int s_nan[256];
    __shared__ int s_inf[256];

    const int tid = threadIdx.x;
    const std::size_t start = static_cast<std::size_t>(blockIdx.x) * blockDim.x + tid;
    const std::size_t stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;

    float local_sum_sq = 0.0f;
    float local_max_abs = 0.0f;
    int local_nan = 0;
    int local_inf = 0;

    for (std::size_t i = start; i < count; i += stride) {
        const float v = data[i];
        if (isnan(v)) {
            ++local_nan;
        } else if (isinf(v)) {
            ++local_inf;
        } else {
            const float av = fabsf(v);
            local_sum_sq += v * v;
            local_max_abs = fmaxf(local_max_abs, av);
        }
    }

    s_sum_sq[tid] = local_sum_sq;
    s_max_abs[tid] = local_max_abs;
    s_nan[tid] = local_nan;
    s_inf[tid] = local_inf;
    __syncthreads();

    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (tid < offset) {
            s_sum_sq[tid] += s_sum_sq[tid + offset];
            s_max_abs[tid] = fmaxf(s_max_abs[tid], s_max_abs[tid + offset]);
            s_nan[tid] += s_nan[tid + offset];
            s_inf[tid] += s_inf[tid + offset];
        }
        __syncthreads();
    }

    if (tid == 0) {
        partials[blockIdx.x] = {s_sum_sq[0], s_max_abs[0], s_nan[0], s_inf[0]};
    }
}

std::string formatNonFiniteError(const char* tag, int count, const NonFiniteStats& out) {
    const char* checked_tag = requireTag(tag, "formatNonFiniteError");
    std::ostringstream message;
    message << "[QKV_NONFINITE] FATAL " << checked_tag
            << " count=" << count
            << " nan=" << out.nan_count
            << " inf=" << out.inf_count;
    if (out.nan_count > 0) {
        message << " first_nan_idx=" << out.first_nan_idx
                << " first_nan_val=" << out.first_nan_val;
    }
    if (out.inf_count > 0) {
        message << " first_inf_idx=" << out.first_inf_idx
                << " first_inf_val=" << out.first_inf_val;
    }
    return message.str();
}

void checkNonFiniteStats(const char* tag,
                         const float* data,
                         int count,
                         cudaStream_t stream) {
    const char* checked_tag = requireTag(tag, "checkNonFiniteStats");
    if (!stream) {
        throw std::runtime_error("checkNonFiniteStats: stream is NULL");
    }
    if (!data) {
        throw std::runtime_error(std::string("checkNonFiniteStats: ") + checked_tag + " data is NULL");
    }
    if (count <= 0) {
        throw std::runtime_error(std::string("checkNonFiniteStats: ") + checked_tag +
                                 " count must be > 0, got " + std::to_string(count));
    }

    NonFiniteStats init{};
    init.first_nan_idx = -1;
    init.first_inf_idx = -1;

    NonFiniteStats* d_stats = nullptr;
    cudaMallocOrThrow(reinterpret_cast<void**>(&d_stats), sizeof(NonFiniteStats), "QKV_debug_stats");

    cudaError_t err = cudaMemcpyAsync(d_stats, &init, sizeof(init), cudaMemcpyHostToDevice, stream);
    if (err != cudaSuccess) {
        cudaFree(d_stats);
        throw std::runtime_error(std::string("checkNonFiniteStats: ") + checked_tag +
                                 " cudaMemcpyAsync H2D failed: " + cudaGetErrorString(err));
    }

    constexpr int kThreads = 256;
    const int blocks = (count + kThreads - 1) / kThreads;
    scanNonFiniteKernel<<<blocks, kThreads, 0, stream>>>(data, count, d_stats);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        cudaFree(d_stats);
        throw std::runtime_error(std::string("checkNonFiniteStats: ") + checked_tag +
                                 " scanNonFiniteKernel launch failed: " + cudaGetErrorString(err));
    }

    NonFiniteStats out{};
    err = cudaMemcpyAsync(&out, d_stats, sizeof(out), cudaMemcpyDeviceToHost, stream);
    if (err != cudaSuccess) {
        cudaFree(d_stats);
        throw std::runtime_error(std::string("checkNonFiniteStats: ") + checked_tag +
                                 " cudaMemcpyAsync D2H failed: " + cudaGetErrorString(err));
    }

    err = cudaStreamSynchronize(stream);
    if (err != cudaSuccess) {
        cudaFree(d_stats);
        throw std::runtime_error(std::string("checkNonFiniteStats: ") + checked_tag +
                                 " cudaStreamSynchronize failed: " + cudaGetErrorString(err));
    }

    cudaFree(d_stats);

    if (out.nan_count == 0 && out.inf_count == 0) {
        return;
    }

    const std::string message = formatNonFiniteError(checked_tag, count, out);
    GRIM::Logging::EmitModuleError(GRIM::Logging::ModuleId::Attention, message, 0);
    throw std::runtime_error(message);
}

}  // anonymous namespace

namespace GRIM::autograd {

int qkvDebugLevel() {
    static int level = []() {
        const char* raw = std::getenv("GRIM_DEBUG_QKV");
        if (!raw || !*raw) {
            return 0;
        }
        return std::atoi(raw);
    }();
    return level;
}

int gradFlowDebugLevel() {
    static int level = []() {
        const char* raw = std::getenv("GRIM_DEBUG_GRADFLOW");
        if (!raw || !*raw) {
            return 0;
        }
        return std::atoi(raw);
    }();
    return level;
}

void logGradFlowTensorStats(const char* tag,
                            const float* data,
                            std::size_t count,
                            cudaStream_t stream,
                            bool force) {
    const int level = gradFlowDebugLevel();
    if (level <= 0 && !force) {
        return;
    }
    const char* checked_tag = requireTag(tag, "logGradFlowTensorStats");
    if (!stream) {
        throw std::runtime_error("logGradFlowTensorStats: stream is NULL");
    }
    if (!data) {
        throw std::runtime_error(std::string("logGradFlowTensorStats: ") + checked_tag + " data is NULL");
    }
    if (count == 0) {
        throw std::runtime_error(std::string("logGradFlowTensorStats: ") + checked_tag + " count is zero");
    }

    constexpr int kThreads = 256;
    constexpr int kMaxBlocks = 4096;
    const int blocks_needed = static_cast<int>((count + kThreads - 1) / kThreads);
    const int blocks = std::max(1, std::min(kMaxBlocks, blocks_needed));

    GradFlowBlockStats* d_partials = nullptr;
    cudaMallocOrThrow(reinterpret_cast<void**>(&d_partials),
                      static_cast<std::size_t>(blocks) * sizeof(GradFlowBlockStats),
                      "GradFlowStats_partials");

    gradFlowStatsKernel<<<blocks, kThreads, 0, stream>>>(data, count, d_partials);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        cudaFree(d_partials);
        throw std::runtime_error(std::string("logGradFlowTensorStats: ") + checked_tag +
                                 " gradFlowStatsKernel launch failed: " + cudaGetErrorString(err));
    }

    std::vector<GradFlowBlockStats> h_partials(static_cast<std::size_t>(blocks));
    err = cudaMemcpyAsync(h_partials.data(), d_partials,
                          static_cast<std::size_t>(blocks) * sizeof(GradFlowBlockStats),
                          cudaMemcpyDeviceToHost, stream);
    if (err != cudaSuccess) {
        cudaFree(d_partials);
        throw std::runtime_error(std::string("logGradFlowTensorStats: ") + checked_tag +
                                 " cudaMemcpyAsync D2H failed: " + cudaGetErrorString(err));
    }

    float first_values[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    const std::size_t first_count = std::min<std::size_t>(count, 4);
    err = cudaMemcpyAsync(first_values, data, first_count * sizeof(float), cudaMemcpyDeviceToHost, stream);
    if (err != cudaSuccess) {
        cudaFree(d_partials);
        throw std::runtime_error(std::string("logGradFlowTensorStats: ") + checked_tag +
                                 " first-value copy failed: " + cudaGetErrorString(err));
    }

    err = cudaStreamSynchronize(stream);
    cudaFree(d_partials);
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("logGradFlowTensorStats: ") + checked_tag +
                                 " stream synchronization failed: " + cudaGetErrorString(err));
    }

    double sum_sq = 0.0;
    float max_abs = 0.0f;
    int nan_count = 0;
    int inf_count = 0;
    for (const auto& p : h_partials) {
        sum_sq += static_cast<double>(p.sum_sq);
        max_abs = std::max(max_abs, p.max_abs);
        nan_count += p.nan_count;
        inf_count += p.inf_count;
    }

    const std::size_t finite_count = count - static_cast<std::size_t>(nan_count + inf_count);
    const float rms = finite_count > 0
        ? static_cast<float>(std::sqrt(sum_sq / static_cast<double>(finite_count)))
        : std::numeric_limits<float>::quiet_NaN();

    float threshold = 1000.0f;
    if (const char* raw = std::getenv("GRIM_GRADFLOW_THRESHOLD")) {
        const float parsed = std::atof(raw);
        if (std::isfinite(parsed) && parsed > 0.0f) {
            threshold = parsed;
        }
    }
    const bool anomalous = nan_count > 0 || inf_count > 0 || max_abs >= threshold || rms >= threshold;
    if (!force && level < 2 && !anomalous) {
        return;
    }

    std::fprintf(stderr,
                 "[GRADFLOW] %s count=%zu finite=%zu rms=%.10e max_abs=%.10e nan=%d inf=%d first=[%.10e,%.10e,%.10e,%.10e]\n",
                 checked_tag,
                 count,
                 finite_count,
                 rms,
                 max_abs,
                 nan_count,
                 inf_count,
                 first_values[0],
                 first_values[1],
                 first_values[2],
                 first_values[3]);
    std::fflush(stderr);
}

void checkQKVTensorFinite(const char* tag,
                          const Tensor& tensor,
                          cudaStream_t stream) {
    const char* checked_tag = requireTag(tag, "checkQKVTensorFinite");
    if (!tensor.data) {
        throw std::runtime_error(std::string("checkQKVTensorFinite: ") + checked_tag + " tensor.data is NULL");
    }
    const int count = static_cast<int>(tensor.numel());
    checkNonFiniteStats(checked_tag, tensor.data, count, stream);
}

void logQKVProjectionEquation(const Tensor& ln1_out,
                              const Tensor& W_qkv,
                              const Tensor& b_qkv,
                              const Tensor& qkv_out,
                              const Batching::BatchPayload& payload,
                              const GRIM::HyperParameters::EncoderSelfAttentionHP& hp,
                              cudaStream_t stream,
                              int layer_idx) {
    auto* tape = GRIM::Logging::getGlobalTape();
    if (!(tape && tape->accepts(GRIM::Logging::LogLevel::Debug)) || tape->skipThisPass()) {
        return;
    }
    if (!stream) {
        throw std::runtime_error("logQKVProjectionEquation: stream is NULL");
    }
    if (!ln1_out.data) {
        throw std::runtime_error("logQKVProjectionEquation: ln1_out.data is NULL");
    }
    if (!W_qkv.data) {
        throw std::runtime_error("logQKVProjectionEquation: W_qkv.data is NULL");
    }
    if (!qkv_out.data) {
        throw std::runtime_error("logQKVProjectionEquation: qkv_out.data is NULL");
    }
    if (hp.use_bias && !b_qkv.data) {
        throw std::runtime_error("logQKVProjectionEquation: hp.use_bias=true but b_qkv.data is NULL");
    }
    ln1_out.shape.require("logQKVProjectionEquation ln1_out");
    W_qkv.shape.require("logQKVProjectionEquation W_qkv");
    qkv_out.shape.require("logQKVProjectionEquation qkv_out");
    if (b_qkv.data) {
        b_qkv.shape.require("logQKVProjectionEquation b_qkv");
        if (static_cast<int>(b_qkv.numel()) != hp.qkv_dim) {
            throw std::runtime_error("logQKVProjectionEquation: b_qkv numel mismatch for hp.qkv_dim");
        }
    }
    if (!ln1_out.shape.is_2d_layout() || !W_qkv.shape.is_2d_layout() || !qkv_out.shape.is_2d_layout()) {
        throw std::runtime_error("logQKVProjectionEquation: ln1_out, W_qkv, and qkv_out must be 2D tensors");
    }

    const int qkv_dim_local = hp.qkv_dim;
    const int d_model_local = hp.d_model;
    const int num_heads_local = hp.num_heads;
    const int head_dim_local = hp.head_dim;
    const int n_sample = std::min(64, payload.total_tokens);
    if (n_sample <= 0) {
        throw std::runtime_error("logQKVProjectionEquation: payload.total_tokens must be > 0");
    }

    const auto ln1_shape = ln1_out.shape.as_2d();
    const auto w_shape = W_qkv.shape.as_2d();
    const auto qkv_shape = qkv_out.shape.as_2d();
    if (ln1_shape.rows != payload.total_tokens || qkv_shape.rows != payload.total_tokens ||
        ln1_shape.cols != d_model_local || qkv_shape.cols != qkv_dim_local ||
        w_shape.rows != qkv_dim_local || w_shape.cols != d_model_local) {
        throw std::runtime_error("logQKVProjectionEquation: tensor shape mismatch for QKV projection equation");
    }

    std::vector<float> h_ln1_sample(static_cast<size_t>(n_sample) * d_model_local);
    const int w_sample_cols = std::min(64, d_model_local);
    std::vector<float> h_wqkv_sample(static_cast<size_t>(qkv_dim_local) * w_sample_cols);
    std::vector<float> h_qkv_sample(static_cast<size_t>(n_sample) * qkv_dim_local);
    std::vector<float> h_bias_sample(qkv_dim_local);

    checkCuda(cudaMemcpyAsync(h_ln1_sample.data(), ln1_out.data,
                              h_ln1_sample.size() * sizeof(float), cudaMemcpyDeviceToHost, stream),
              "logQKVProjectionEquation: copy ln1_out sample");
    checkCuda(cudaMemcpy2DAsync(
        h_wqkv_sample.data(),
        static_cast<size_t>(w_sample_cols) * sizeof(float),
        W_qkv.data,
        static_cast<size_t>(d_model_local) * sizeof(float),
        static_cast<size_t>(w_sample_cols) * sizeof(float),
        qkv_dim_local,
        cudaMemcpyDeviceToHost,
        stream),
        "logQKVProjectionEquation: copy W_qkv sample");
    checkCuda(cudaMemcpyAsync(h_qkv_sample.data(), qkv_out.data,
                              h_qkv_sample.size() * sizeof(float), cudaMemcpyDeviceToHost, stream),
              "logQKVProjectionEquation: copy qkv_out sample");
    if (b_qkv.data) {
        checkCuda(cudaMemcpyAsync(h_bias_sample.data(), b_qkv.data,
                                  h_bias_sample.size() * sizeof(float), cudaMemcpyDeviceToHost, stream),
                  "logQKVProjectionEquation: copy b_qkv");
    }
    checkCuda(cudaStreamSynchronize(stream), "logQKVProjectionEquation: synchronize samples");

    float ln1_min = FLT_MAX, ln1_max = -FLT_MAX;
    double ln1_sum_sq = 0.0;
    double ln1_row_rms_sum = 0.0;
    for (int t = 0; t < n_sample; ++t) {
        double row_sum_sq = 0.0;
        for (int d = 0; d < d_model_local; ++d) {
            const float v = h_ln1_sample[t * d_model_local + d];
            ln1_min = fminf(ln1_min, v);
            ln1_max = fmaxf(ln1_max, v);
            ln1_sum_sq += v * v;
            row_sum_sq += v * v;
        }
        ln1_row_rms_sum += sqrtf(static_cast<float>(row_sum_sq / d_model_local));
    }
    const float ln1_rms = sqrtf(static_cast<float>(ln1_sum_sq / (n_sample * d_model_local)));
    const float ln1_row_rms_mean = static_cast<float>(ln1_row_rms_sum / n_sample);

    float wqkv_min = FLT_MAX, wqkv_max = -FLT_MAX;
    double wqkv_sum_sq = 0.0;
    double wqkv_row_rms_sum = 0.0;
    double wq_row_rms_sum = 0.0;
    for (int row = 0; row < qkv_dim_local; ++row) {
        double row_sum_sq = 0.0;
        for (int col = 0; col < w_sample_cols; ++col) {
            const float v = h_wqkv_sample[row * w_sample_cols + col];
            wqkv_min = fminf(wqkv_min, v);
            wqkv_max = fmaxf(wqkv_max, v);
            wqkv_sum_sq += v * v;
            row_sum_sq += v * v;
        }
        const float row_rms_scaled = sqrtf(static_cast<float>(row_sum_sq / w_sample_cols));
        wqkv_row_rms_sum += row_rms_scaled;
        if (row < d_model_local) {
            wq_row_rms_sum += row_rms_scaled;
        }
    }
    const float wqkv_rms = sqrtf(static_cast<float>(wqkv_sum_sq / (qkv_dim_local * w_sample_cols)));
    const float wqkv_row_rms_mean = static_cast<float>(wqkv_row_rms_sum / qkv_dim_local);
    const float wq_row_rms_mean = static_cast<float>(wq_row_rms_sum / d_model_local);

    float qkv_min = FLT_MAX, qkv_max = -FLT_MAX;
    double qkv_sum_sq = 0.0;
    double qkv_row_rms_sum = 0.0;
    double qkv_head_row_rms_sum = 0.0;
    for (int t = 0; t < n_sample; ++t) {
        double row_sum_sq = 0.0;
        for (int d = 0; d < d_model_local; ++d) {
            const float v = h_qkv_sample[t * qkv_dim_local + d];
            qkv_min = fminf(qkv_min, v);
            qkv_max = fmaxf(qkv_max, v);
            qkv_sum_sq += v * v;
            row_sum_sq += v * v;
        }
        qkv_row_rms_sum += sqrtf(static_cast<float>(row_sum_sq / d_model_local));
        for (int h = 0; h < num_heads_local; ++h) {
            double head_sum_sq = 0.0;
            const int head_base = t * qkv_dim_local + h * head_dim_local;
            for (int d = 0; d < head_dim_local; ++d) {
                const float v = h_qkv_sample[head_base + d];
                head_sum_sq += v * v;
            }
            qkv_head_row_rms_sum += sqrtf(static_cast<float>(head_sum_sq / head_dim_local));
        }
    }
    const float qkv_rms = sqrtf(static_cast<float>(qkv_sum_sq / (n_sample * d_model_local)));
    const float qkv_row_rms_mean = static_cast<float>(qkv_row_rms_sum / n_sample);
    const float qkv_head_row_rms_mean = static_cast<float>(qkv_head_row_rms_sum / (n_sample * num_heads_local));

    float bias_min = 0.0f, bias_max = 0.0f, bias_rms = 0.0f;
    if (b_qkv.data) {
        double bias_sum_sq = 0.0;
        for (int i = 0; i < qkv_dim_local; ++i) {
            const float v = h_bias_sample[i];
            bias_min = fminf(bias_min, v);
            bias_max = fmaxf(bias_max, v);
            bias_sum_sq += v * v;
        }
        bias_rms = sqrtf(static_cast<float>(bias_sum_sq / qkv_dim_local));
    }

    const float expected_q_elem_rms =
        ln1_row_rms_mean * wq_row_rms_mean * sqrtf(static_cast<float>(d_model_local));
    const float expected_q_full_row_rms = expected_q_elem_rms;
    const float expected_q_head_row_rms = expected_q_elem_rms;
    const float actual_q_full_row_rms = qkv_row_rms_mean;
    const float actual_q_head_row_rms = qkv_head_row_rms_mean;
    const float target_q_head_row_rms = 1.0f;
    const float target_q_full_row_rms = 1.0f;

    fprintf(stderr, "\n[QKV_EQUATION] ENCODER_LAYER_%d: qkv_out = ln1_out @ W_qkv^T + b_qkv\n", layer_idx);
    fprintf(stderr, "  ln1_out (sample %d tokens): shape=[%d,%d] min=%.10f max=%.10f rms=%.10f\n",
            n_sample, n_sample, d_model_local, ln1_min, ln1_max, ln1_rms);
    fprintf(stderr, "  ln1_out row_rms: mean=%.10f\n", ln1_row_rms_mean);
    fprintf(stderr, "  W_qkv (sample %d cols): shape=[%d,%d] min=%.10f max=%.10f rms=%.10f\n",
            w_sample_cols, qkv_dim_local, d_model_local, wqkv_min, wqkv_max, wqkv_rms);
    fprintf(stderr, "  W_q row_rms (scaled to d_model): mean=%.10f\n", wq_row_rms_mean);
    fprintf(stderr, "  W_qkv row_rms (all rows, scaled): mean=%.10f\n", wqkv_row_rms_mean);
    if (b_qkv.data) {
        fprintf(stderr, "  b_qkv: shape=[%d] min=%.10f max=%.10f rms=%.10f\n",
                qkv_dim_local, bias_min, bias_max, bias_rms);
    } else {
        fprintf(stderr, "  b_qkv: [nullptr]\n");
    }
    fprintf(stderr, "  EXPECTED qkv_elem_rms = ln1_row_rms * wq_row_rms * sqrt(d_model)\n");
    fprintf(stderr, "                        = %.8f * %.8f * sqrt(%d) = %.8f\n",
            ln1_row_rms_mean, wq_row_rms_mean, d_model_local, expected_q_elem_rms);
    fprintf(stderr, "  ACTUAL qkv_out (Q portion): min=%.10f max=%.10f rms=%.10f\n",
            qkv_min, qkv_max, qkv_rms);
    fprintf(stderr, "  EXPECTED Q row_rms (full/head): %.8f / %.8f\n",
            expected_q_full_row_rms, expected_q_head_row_rms);
    fprintf(stderr, "  ACTUAL   Q row_rms (full/head): %.8f / %.8f\n",
            actual_q_full_row_rms, actual_q_head_row_rms);
    fprintf(stderr, "  TARGET   Q row_rms (full/head): %.8f / %.8f (healthy attention: elem_rms≈1.0)\n",
            target_q_full_row_rms, target_q_head_row_rms);
    fprintf(stderr, "  INFLATION(full/head): %.8fx / %.8fx\n",
            actual_q_full_row_rms / target_q_full_row_rms,
            actual_q_head_row_rms / target_q_head_row_rms);

    std::ostringstream eq;
    eq << "[QKV_PROJECTION_EQUATION] qkv_out = ln1_out @ W_qkv^T + b_qkv\n";
    eq << "  INPUT (ln1_out): rms=" << ln1_rms << " row_rms=" << ln1_row_rms_mean << "\n";
    eq << "  WEIGHT (W_qkv): rms=" << wqkv_rms << " q_row_rms=" << wq_row_rms_mean << "\n";
    eq << "  EXPECTED qkv_elem_rms = ln1_row_rms * wq_row_rms * sqrt(d_model)\n";
    eq << "                         = " << ln1_row_rms_mean << " * " << wq_row_rms_mean
       << " * sqrt(" << d_model_local << ")\n";
    eq << "                         = " << expected_q_elem_rms << "\n";
    eq << "  ACTUAL Q row_rms (full/head): " << actual_q_full_row_rms << " / " << actual_q_head_row_rms << "\n";
    eq << "  TARGET Q row_rms (full/head): " << target_q_full_row_rms << " / " << target_q_head_row_rms << "\n";
    const float inflation_ratio_eq = actual_q_head_row_rms / target_q_head_row_rms;
    eq << "  INFLATION (full/head): " << (actual_q_full_row_rms / target_q_full_row_rms)
       << "x / " << inflation_ratio_eq << "x\n";
    if (inflation_ratio_eq > 5.0f) {
        eq << "  [ANOMALY] per-head q_row_rms=" << actual_q_head_row_rms
           << " is " << inflation_ratio_eq << "x larger than target=" << target_q_head_row_rms << "\n";
    }
    if (ln1_row_rms_mean > 50.0f) {
        eq << "  [ANOMALY] ln1_out row_rms=" << ln1_row_rms_mean << " >> expected ~1.0\n";
    }
    if (wq_row_rms_mean > 5.0f) {
        eq << "  [ANOMALY] W_q row_rms=" << wq_row_rms_mean << " >> expected ~0.036\n";
    }
    EQ_LOG(tape, GRIM::Logging::LogGroup::Attention, GRIM::Logging::LogPhase::QKV_PROJECTION, layer_idx,
           "QKV_PROJECTION_EQUATION", eq.str().c_str());
}

}  // namespace GRIM::autograd
