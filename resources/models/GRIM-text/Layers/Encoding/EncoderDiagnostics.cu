//======================================================//
//  EncoderDiagnostics.cu
//  Rule 21 diagnostics for encoder-layer residual output
//======================================================//

#include "EncoderDiagnostics.hpp"

#include "../../Shared/LogRecorder/BatchLogTape.hpp"

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdio>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

#define ENC_DIAG_CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        char msg[512]; \
        std::snprintf(msg, sizeof(msg), "CUDA ERROR at %s:%d - %s: %s", \
                 __FILE__, __LINE__, #call, cudaGetErrorString(err)); \
        throw std::runtime_error(msg); \
    } \
} while(0)

bool shouldLogEncoderResidualDiagnostic() {
    auto* tape = GRIM::Logging::getGlobalTape();
    return tape
        && tape->accepts(GRIM::Logging::LogLevel::Debug)
        && !tape->skipThisPass();
}

void requireTensor2D(const GRIM::Tensor& tensor,
                     const char* name,
                     int rows,
                     int cols,
                     const char* context) {
    if (!tensor.data) {
        throw std::runtime_error(std::string(context) + ": " + name + ".data is NULL");
    }
    tensor.shape.require((std::string(context) + " " + name).c_str());
    if (!tensor.shape.is_2d_layout()) {
        throw std::runtime_error(std::string(context) + ": " + name + " must be a 2D [total_tokens,d_model] tensor");
    }
    const auto dims = tensor.shape.as_2d();
    if (dims.rows != rows || dims.cols != cols) {
        throw std::runtime_error(std::string(context) + ": " + name + " shape mismatch. expected=[" +
                                 std::to_string(rows) + "," + std::to_string(cols) + "] got=[" +
                                 std::to_string(dims.rows) + "," + std::to_string(dims.cols) + "]");
    }
}

void requireLayerScaleTensor(const GRIM::Tensor& gamma,
                             const char* name,
                             int d_model,
                             const char* context) {
    if (!gamma.data) {
        throw std::runtime_error(std::string(context) + ": " + name + " is NULL while LayerScale is enabled");
    }
    gamma.shape.require((std::string(context) + " " + name).c_str());
    if (!gamma.shape.is_2d_layout()) {
        throw std::runtime_error(std::string(context) + ": " + name + " must be a 2D [1,d_model] gamma vector");
    }
    const auto dims = gamma.shape.as_2d();
    if (dims.rows != 1 || dims.cols != d_model) {
        throw std::runtime_error(std::string(context) + ": " + name + " must have shape [1,d_model]. expected=[1," +
                                 std::to_string(d_model) + "] got=[" + std::to_string(dims.rows) + "," +
                                 std::to_string(dims.cols) + "]");
    }
}

std::vector<float> copyTensorToHost(const GRIM::Tensor& tensor,
                                    const char* name,
                                    int rows,
                                    int cols,
                                    cudaStream_t stream,
                                    const char* context) {
    requireTensor2D(tensor, name, rows, cols, context);
    const size_t count = static_cast<size_t>(rows) * static_cast<size_t>(cols);
    std::vector<float> host(count);
    ENC_DIAG_CUDA_CHECK(cudaMemcpy(host.data(), tensor.data,
                                   count * sizeof(float), cudaMemcpyDeviceToHost));
    (void)stream;
    return host;
}

std::vector<float> copyLayerScaleToHost(const GRIM::Tensor& gamma,
                                        const char* name,
                                        int d_model,
                                        const char* context) {
    requireLayerScaleTensor(gamma, name, d_model, context);
    std::vector<float> host(static_cast<size_t>(d_model));
    ENC_DIAG_CUDA_CHECK(cudaMemcpy(host.data(), gamma.data,
                                   host.size() * sizeof(float), cudaMemcpyDeviceToHost));
    return host;
}

struct RowStats {
    float min = 0.0f;
    float max = 0.0f;
    float mean = 0.0f;
    float rms = 0.0f;
};

struct AggregateStats {
    float row_rms_min = 0.0f;
    float row_rms_max = 0.0f;
    float row_rms_mean = 0.0f;
    int row_rms_min_row = -1;
    int row_rms_max_row = -1;
    float scalar_min = 0.0f;
    float scalar_max = 0.0f;
    float scalar_mean = 0.0f;
};

std::vector<RowStats> computeRowStats(const std::vector<float>& values,
                                      int rows,
                                      int cols,
                                      const char* name) {
    if (static_cast<size_t>(rows) * static_cast<size_t>(cols) != values.size()) {
        throw std::runtime_error(std::string("EncoderDiagnostics::computeRowStats size mismatch for ") + name);
    }
    std::vector<RowStats> stats(static_cast<size_t>(rows));
    for (int row = 0; row < rows; ++row) {
        float row_min = FLT_MAX;
        float row_max = -FLT_MAX;
        double sum = 0.0;
        double sum_sq = 0.0;
        const size_t base = static_cast<size_t>(row) * static_cast<size_t>(cols);
        for (int col = 0; col < cols; ++col) {
            const float v = values[base + static_cast<size_t>(col)];
            row_min = std::min(row_min, v);
            row_max = std::max(row_max, v);
            sum += v;
            sum_sq += static_cast<double>(v) * static_cast<double>(v);
        }
        RowStats& row_stats = stats[static_cast<size_t>(row)];
        row_stats.min = row_min;
        row_stats.max = row_max;
        row_stats.mean = static_cast<float>(sum / cols);
        row_stats.rms = std::sqrt(static_cast<float>(sum_sq / cols));
    }
    return stats;
}

AggregateStats aggregateRows(const std::vector<RowStats>& rows,
                             const char* name) {
    if (rows.empty()) {
        throw std::runtime_error(std::string("EncoderDiagnostics::aggregateRows called with empty rows for ") + name);
    }
    AggregateStats aggregate{};
    aggregate.row_rms_min = FLT_MAX;
    aggregate.row_rms_max = -FLT_MAX;
    aggregate.scalar_min = FLT_MAX;
    aggregate.scalar_max = -FLT_MAX;
    double rms_sum = 0.0;
    double scalar_mean_sum = 0.0;
    for (size_t row = 0; row < rows.size(); ++row) {
        const RowStats& stats = rows[row];
        if (stats.rms < aggregate.row_rms_min) {
            aggregate.row_rms_min = stats.rms;
            aggregate.row_rms_min_row = static_cast<int>(row);
        }
        if (stats.rms > aggregate.row_rms_max) {
            aggregate.row_rms_max = stats.rms;
            aggregate.row_rms_max_row = static_cast<int>(row);
        }
        aggregate.scalar_min = std::min(aggregate.scalar_min, stats.min);
        aggregate.scalar_max = std::max(aggregate.scalar_max, stats.max);
        rms_sum += stats.rms;
        scalar_mean_sum += stats.mean;
    }
    aggregate.row_rms_mean = static_cast<float>(rms_sum / rows.size());
    aggregate.scalar_mean = static_cast<float>(scalar_mean_sum / rows.size());
    return aggregate;
}

struct LayerScaleDiagStats {
    float min = 0.0f;
    float max = 0.0f;
    float mean = 0.0f;
    float rms = 0.0f;
};

LayerScaleDiagStats computeLayerScaleStats(const std::vector<float>& gamma,
                                           const char* name) {
    if (gamma.empty()) {
        throw std::runtime_error(std::string("EncoderDiagnostics::computeLayerScaleStats empty gamma for ") + name);
    }
    LayerScaleDiagStats stats{};
    stats.min = FLT_MAX;
    stats.max = -FLT_MAX;
    double sum = 0.0;
    double sum_sq = 0.0;
    for (float v : gamma) {
        stats.min = std::min(stats.min, v);
        stats.max = std::max(stats.max, v);
        sum += v;
        sum_sq += static_cast<double>(v) * static_cast<double>(v);
    }
    stats.mean = static_cast<float>(sum / gamma.size());
    stats.rms = std::sqrt(static_cast<float>(sum_sq / gamma.size()));
    return stats;
}

double computeSampledAverageCosine(const std::vector<float>& output,
                                   const std::vector<RowStats>& output_rows,
                                   int total_tokens,
                                   int d_model,
                                   int* pair_count) {
    if (!pair_count) {
        throw std::runtime_error("EncoderDiagnostics::computeSampledAverageCosine pair_count is NULL");
    }
    const int sample_pairs = std::min(30, total_tokens / 2);
    if (sample_pairs <= 0) {
        *pair_count = 0;
        return 0.0;
    }
    const int stride = std::max(1, total_tokens / sample_pairs);
    double cos_sum = 0.0;
    int num_pairs = 0;

    for (int i = 0; i < total_tokens && num_pairs < sample_pairs; i += stride) {
        const int j = (i + total_tokens / 2) % total_tokens;
        if (i == j || output_rows[static_cast<size_t>(i)].rms < 1e-8f ||
            output_rows[static_cast<size_t>(j)].rms < 1e-8f) {
            continue;
        }

        double dot = 0.0;
        const size_t i_base = static_cast<size_t>(i) * static_cast<size_t>(d_model);
        const size_t j_base = static_cast<size_t>(j) * static_cast<size_t>(d_model);
        for (int d = 0; d < d_model; ++d) {
            dot += static_cast<double>(output[i_base + static_cast<size_t>(d)]) *
                   static_cast<double>(output[j_base + static_cast<size_t>(d)]);
        }
        cos_sum += dot / (static_cast<double>(output_rows[static_cast<size_t>(i)].rms) *
                          static_cast<double>(output_rows[static_cast<size_t>(j)].rms) *
                          static_cast<double>(d_model));
        ++num_pairs;
    }

    *pair_count = num_pairs;
    return (num_pairs > 0) ? cos_sum / num_pairs : 0.0;
}

bool isValidPayloadRow(const GRIM::Batching::BatchPayload& payload, int row) {
    if (payload.max_seq_len <= 0) {
        throw std::runtime_error("EncoderDiagnostics: payload.max_seq_len must be > 0");
    }
    if (static_cast<int>(payload.seq_lengths.size()) != payload.batch_size) {
        throw std::runtime_error("EncoderDiagnostics: payload.seq_lengths size (" +
                                 std::to_string(payload.seq_lengths.size()) +
                                 ") != payload.batch_size (" + std::to_string(payload.batch_size) + ")");
    }
    const int batch_row = row / payload.max_seq_len;
    const int seq_pos = row % payload.max_seq_len;
    if (batch_row < 0 || batch_row >= payload.batch_size) {
        throw std::runtime_error("EncoderDiagnostics: row " + std::to_string(row) +
                                 " maps outside payload.batch_size=" + std::to_string(payload.batch_size));
    }
    const int seq_len = payload.seq_lengths[static_cast<size_t>(batch_row)];
    return seq_pos < seq_len;
}

void emitRowStackLogs(GRIM::Logging::BatchLogTape* tape,
                      int layer_idx,
                      const GRIM::Batching::BatchPayload& payload,
                      const std::vector<RowStats>& input_rows,
                      const std::vector<RowStats>& attn_raw_rows,
                      const std::vector<RowStats>& attn_rows,
                      const std::vector<RowStats>& residual_rows,
                      const std::vector<RowStats>& ffn_raw_rows,
                      const std::vector<RowStats>& ffn_rows,
                      const std::vector<RowStats>& output_rows) {
    if (!tape) {
        throw std::runtime_error("EncoderDiagnostics::emitRowStackLogs tape is NULL");
    }
    const int total_tokens = payload.total_tokens;
    for (int row = 0; row < total_tokens; ++row) {
        const int batch_row = row / payload.max_seq_len;
        const int seq_pos = row % payload.max_seq_len;
        const bool valid = isValidPayloadRow(payload, row);
        const RowStats& in = input_rows[static_cast<size_t>(row)];
        const RowStats& attn_raw = attn_raw_rows[static_cast<size_t>(row)];
        const RowStats& attn = attn_rows[static_cast<size_t>(row)];
        const RowStats& res = residual_rows[static_cast<size_t>(row)];
        const RowStats& ffn_raw = ffn_raw_rows[static_cast<size_t>(row)];
        const RowStats& ffn = ffn_rows[static_cast<size_t>(row)];
        const RowStats& out = output_rows[static_cast<size_t>(row)];
        BATCH_LOG(tape, GRIM::Logging::LogLevel::Debug,
                  GRIM::Logging::LogGroup::Attention,
                  GRIM::Logging::LogPhase::RESIDUAL_POST_ATTN,
                  layer_idx,
                  "LAYER_ROW_STACK_EQUATION",
                  "row=%d batch_row=%d seq_pos=%d valid=%d input[min=%.6g max=%.6g mean=%.6g rms=%.6g] attn_raw[min=%.6g max=%.6g mean=%.6g rms=%.6g] attn_branch[min=%.6g max=%.6g mean=%.6g rms=%.6g] residual1[min=%.6g max=%.6g mean=%.6g rms=%.6g] ffn_raw[min=%.6g max=%.6g mean=%.6g rms=%.6g] ffn_branch[min=%.6g max=%.6g mean=%.6g rms=%.6g] output[min=%.6g max=%.6g mean=%.6g rms=%.6g]",
                  row, batch_row, seq_pos, valid ? 1 : 0,
                  in.min, in.max, in.mean, in.rms,
                  attn_raw.min, attn_raw.max, attn_raw.mean, attn_raw.rms,
                  attn.min, attn.max, attn.mean, attn.rms,
                  res.min, res.max, res.mean, res.rms,
                  ffn_raw.min, ffn_raw.max, ffn_raw.mean, ffn_raw.rms,
                  ffn.min, ffn.max, ffn.mean, ffn.rms,
                  out.min, out.max, out.mean, out.rms);
    }
}

} // namespace

namespace GRIM::EncoderDiagnostics {

void emitLayerResidualDiagnostic(const LayerResidualDiagnosticRequest& request) {
    auto* tape = GRIM::Logging::getGlobalTape();
    if (!shouldLogEncoderResidualDiagnostic()) {
        return;
    }
    if (!request.stream) {
        throw std::runtime_error("EncoderDiagnostics::emitLayerResidualDiagnostic: stream is NULL");
    }
    if (request.layer_idx < 0) {
        throw std::runtime_error("EncoderDiagnostics::emitLayerResidualDiagnostic: layer_idx must be >= 0");
    }
    const int total_tokens = request.payload.total_tokens;
    const int d_model = request.hp.d_model;
    if (total_tokens <= 0 || d_model <= 0) {
        throw std::runtime_error("EncoderDiagnostics::emitLayerResidualDiagnostic: invalid shape total_tokens=" +
                                 std::to_string(total_tokens) + " d_model=" + std::to_string(d_model));
    }
    if (request.payload.batch_size <= 0 || request.payload.max_seq_len <= 0) {
        throw std::runtime_error("EncoderDiagnostics::emitLayerResidualDiagnostic: invalid payload batch_size=" +
                                 std::to_string(request.payload.batch_size) + " max_seq_len=" +
                                 std::to_string(request.payload.max_seq_len));
    }
    if (request.payload.batch_size * request.payload.max_seq_len != total_tokens) {
        throw std::runtime_error("EncoderDiagnostics::emitLayerResidualDiagnostic: payload.total_tokens must equal batch_size*max_seq_len. total_tokens=" +
                                 std::to_string(total_tokens) + " batch_size=" + std::to_string(request.payload.batch_size) +
                                 " max_seq_len=" + std::to_string(request.payload.max_seq_len));
    }

    ENC_DIAG_CUDA_CHECK(cudaStreamSynchronize(request.stream));

    constexpr const char* kContext = "EncoderDiagnostics::emitLayerResidualDiagnostic";
    std::vector<float> h_input = copyTensorToHost(request.input, "input", total_tokens, d_model, request.stream, kContext);
    std::vector<float> h_attn_raw = copyTensorToHost(request.attention_raw, "attention_raw", total_tokens, d_model, request.stream, kContext);
    std::vector<float> h_attn = copyTensorToHost(request.attention_branch, "attention_branch", total_tokens, d_model, request.stream, kContext);
    std::vector<float> h_residual = copyTensorToHost(request.residual1, "residual1", total_tokens, d_model, request.stream, kContext);
    std::vector<float> h_ffn_raw = copyTensorToHost(request.ffn_raw, "ffn_raw", total_tokens, d_model, request.stream, kContext);
    std::vector<float> h_ffn = copyTensorToHost(request.ffn_branch, "ffn_branch", total_tokens, d_model, request.stream, kContext);
    std::vector<float> h_output = copyTensorToHost(request.output, "output", total_tokens, d_model, request.stream, kContext);

    std::vector<RowStats> input_rows = computeRowStats(h_input, total_tokens, d_model, "input");
    std::vector<RowStats> attn_raw_rows = computeRowStats(h_attn_raw, total_tokens, d_model, "attention_raw");
    std::vector<RowStats> attn_rows = computeRowStats(h_attn, total_tokens, d_model, "attention_branch");
    std::vector<RowStats> residual_rows = computeRowStats(h_residual, total_tokens, d_model, "residual1");
    std::vector<RowStats> ffn_raw_rows = computeRowStats(h_ffn_raw, total_tokens, d_model, "ffn_raw");
    std::vector<RowStats> ffn_rows = computeRowStats(h_ffn, total_tokens, d_model, "ffn_branch");
    std::vector<RowStats> output_rows = computeRowStats(h_output, total_tokens, d_model, "output");

    const AggregateStats input_agg = aggregateRows(input_rows, "input");
    const AggregateStats attn_raw_agg = aggregateRows(attn_raw_rows, "attention_raw");
    const AggregateStats attn_agg = aggregateRows(attn_rows, "attention_branch");
    const AggregateStats residual_agg = aggregateRows(residual_rows, "residual1");
    const AggregateStats ffn_raw_agg = aggregateRows(ffn_raw_rows, "ffn_raw");
    const AggregateStats ffn_agg = aggregateRows(ffn_rows, "ffn_branch");
    const AggregateStats output_agg = aggregateRows(output_rows, "output");

    int num_pairs = 0;
    const double avg_cos = computeSampledAverageCosine(h_output, output_rows, total_tokens, d_model, &num_pairs);

    LayerScaleDiagStats ls1_stats{};
    LayerScaleDiagStats ls2_stats{};
    if (request.hp.use_layer_scale) {
        if (!request.layer_scale1 || !request.layer_scale2) {
            throw std::runtime_error("EncoderDiagnostics::emitLayerResidualDiagnostic: LayerScale enabled but gamma pointers are NULL");
        }
        std::vector<float> h_ls1 = copyLayerScaleToHost(*request.layer_scale1, "layer_scale1", d_model, kContext);
        std::vector<float> h_ls2 = copyLayerScaleToHost(*request.layer_scale2, "layer_scale2", d_model, kContext);
        ls1_stats = computeLayerScaleStats(h_ls1, "layer_scale1");
        ls2_stats = computeLayerScaleStats(h_ls2, "layer_scale2");
    }

    std::ostringstream eq;
    eq << "[LAYER_COSINE_EQUATION] layer=" << request.layer_idx
       << ": residual1[t,d] = input[t,d] + gamma1[d] * attn[t,d]; output[t,d] = residual1[t,d] + gamma2[d] * ffn[t,d]\n";
    eq << "  OUTPUT h_L" << request.layer_idx << ": shape=[" << total_tokens << ", " << d_model
       << "] row_rms_range=[" << output_agg.row_rms_min << ", " << output_agg.row_rms_max << "]\n";
    if (request.hp.use_layer_scale) {
        eq << "  LAYERSCALE: gamma1[min=" << ls1_stats.min << " max=" << ls1_stats.max
           << " mean=" << ls1_stats.mean << " rms=" << ls1_stats.rms << "]"
           << " gamma2[min=" << ls2_stats.min << " max=" << ls2_stats.max
           << " mean=" << ls2_stats.mean << " rms=" << ls2_stats.rms << "]\n";
    } else {
        eq << "  LAYERSCALE: disabled\n";
    }
    eq << "  ACTUAL avg_cos=" << avg_cos << " (pairs=" << num_pairs
       << ") [|avg_cos|->1 = collapse, near 0 = diverse]";
    EQ_LOG(tape, GRIM::Logging::LogGroup::Attention,
           GRIM::Logging::LogPhase::RESIDUAL_POST_ATTN,
           request.layer_idx,
           "LAYER_COSINE_EQUATION",
           eq.str().c_str());

    BATCH_LOG(tape, GRIM::Logging::LogLevel::Debug,
              GRIM::Logging::LogGroup::Attention,
              GRIM::Logging::LogPhase::RESIDUAL_POST_ATTN,
              request.layer_idx,
              "LAYER_COSINE_STACK_EQUATION",
                            "rows=%d d_model=%d input[row_rms_mean=%.6g mean=%.6g] attn_raw[row_rms_mean=%.6g mean=%.6g] attn_branch[row_rms_mean=%.6g mean=%.6g] residual1[row_rms_mean=%.6g mean=%.6g] ffn_raw[row_rms_mean=%.6g mean=%.6g] ffn_branch[row_rms_mean=%.6g mean=%.6g] output[min=%.6g max=%.6g mean=%.6g row_rms_mean=%.6g]",
              total_tokens, d_model,
                            input_agg.row_rms_mean, input_agg.scalar_mean,
                            attn_raw_agg.row_rms_mean, attn_raw_agg.scalar_mean,
                            attn_agg.row_rms_mean, attn_agg.scalar_mean,
                            residual_agg.row_rms_mean, residual_agg.scalar_mean,
                            ffn_raw_agg.row_rms_mean, ffn_raw_agg.scalar_mean,
                            ffn_agg.row_rms_mean, ffn_agg.scalar_mean,
              output_agg.scalar_min, output_agg.scalar_max, output_agg.scalar_mean, output_agg.row_rms_mean);

    emitRowStackLogs(tape, request.layer_idx, request.payload,
                                         input_rows, attn_raw_rows, attn_rows, residual_rows, ffn_raw_rows, ffn_rows, output_rows);

    if (std::fabs(avg_cos) > 0.8) {
        BATCH_LOG(tape, GRIM::Logging::LogLevel::Warning,
                  GRIM::Logging::LogGroup::Attention,
                  GRIM::Logging::LogPhase::RESIDUAL_POST_ATTN,
                  request.layer_idx,
                  "LAYER_COSINE_ANOMALY",
                  "Layer %d |avg_cos|=%.6g HIGH - possible mode collapse; output_rms_max=%.6g row=%d",
                  request.layer_idx, std::fabs(avg_cos), output_agg.row_rms_max, output_agg.row_rms_max_row);
    }
}

} // namespace GRIM::EncoderDiagnostics

#undef ENC_DIAG_CUDA_CHECK