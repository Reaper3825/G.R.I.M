//======================================================//
//  LogitScaleDiagnostic.cu
//  Per-batch LM-valid logit statistics and logit-scale
//  equation diagnostics. Originally lifted from
//  Phase2_TrainingLoop.cu's "TRAINING SIGNAL: Logit
//  Statistics" scope.
//======================================================//

#include "LogitScaleDiagnostic.hpp"
#include "RhoDiagnostic.hpp"
#include "LMHeadWeightStats.hpp"

#include "../Phases/Phase2_TrainingLoop.hpp"
#include "../../Layers/LMHead/lm_head_GPU.hpp"
#include "../../Shared/TrainingState/TrainingState_GPU.hpp"
#include "../../Shared/LogRecorder/LogRecorder.hpp"
#include "../../Shared/Telemetry/TelemetryUpdate.hpp"

#include <vector>
#include <map>
#include <sstream>
#include <iomanip>
#include <cmath>
#include <algorithm>
#include <utility>
#include <limits>
#include <stdexcept>
#include <string>
#include <cstdint>

#include <cuda_runtime.h>

namespace GRIM::Diagnostics {

namespace {

const char* cudaMemoryTypeName(cudaMemoryType type)
{
    switch (type) {
        case cudaMemoryTypeHost:    return "host";
        case cudaMemoryTypeDevice:  return "device";
        case cudaMemoryTypeManaged: return "managed";
        case cudaMemoryTypeUnregistered: return "unregistered";
        default:                    return "unknown";
    }
}

void validateLmHeadWeightsConsumerBoundaryOrThrow(
    GRIMText::Training::TrainingContext& ctx,
    const float* lm_head_weights,
    const float* embedding_weights,
    bool tie_embeddings,
    int vocab_size,
    int d_model,
    int batch_idx,
    const char* consumer,
    const char* stage)
{
    if (!lm_head_weights) {
        throw std::runtime_error(
            std::string("[DEBUG-LM-WEIGHTS] ") + consumer + " " + stage +
            ": LM-head weights pointer is NULL at " + __FILE__ + ":" +
            std::to_string(__LINE__));
    }
    if (vocab_size <= 0) {
        throw std::runtime_error(
            std::string("[DEBUG-LM-WEIGHTS] ") + consumer + " " + stage +
            ": vocab_size must be > 0, got " + std::to_string(vocab_size));
    }
    if (d_model <= 0) {
        throw std::runtime_error(
            std::string("[DEBUG-LM-WEIGHTS] ") + consumer + " " + stage +
            ": d_model must be > 0, got " + std::to_string(d_model));
    }
    if (tie_embeddings) {
        if (!embedding_weights) {
            throw std::runtime_error(
                std::string("[DEBUG-LM-WEIGHTS] ") + consumer + " " + stage +
                ": tie_embeddings=true but embedding weights pointer is NULL at " +
                __FILE__ + ":" + std::to_string(__LINE__));
        }
        if (lm_head_weights != embedding_weights) {
            throw std::runtime_error(
                std::string("[DEBUG-LM-WEIGHTS] ") + consumer + " " + stage +
                ": tie_embeddings=true but LM-head/embedding pointers do not alias. lm_head=" +
                std::to_string(reinterpret_cast<uintptr_t>(lm_head_weights)) +
                " embedding=" + std::to_string(reinterpret_cast<uintptr_t>(embedding_weights)));
        }
    }

    cudaPointerAttributes attrs{};
    cudaError_t attr_err = cudaPointerGetAttributes(&attrs, lm_head_weights);
    if (attr_err != cudaSuccess) {
        throw std::runtime_error(
            std::string("[DEBUG-LM-WEIGHTS] ") + consumer + " " + stage +
            ": cudaPointerGetAttributes failed for LM-head weights ptr=" +
            std::to_string(reinterpret_cast<uintptr_t>(lm_head_weights)) +
            " error=" + cudaGetErrorString(attr_err) + " at " + __FILE__ + ":" +
            std::to_string(__LINE__));
    }
    if (attrs.type != cudaMemoryTypeDevice && attrs.type != cudaMemoryTypeManaged) {
        throw std::runtime_error(
            std::string("[DEBUG-LM-WEIGHTS] ") + consumer + " " + stage +
            ": LM-head weights ptr=" + std::to_string(reinterpret_cast<uintptr_t>(lm_head_weights)) +
            " resolved to unsupported memory type " + cudaMemoryTypeName(attrs.type) +
            " at " + __FILE__ + ":" + std::to_string(__LINE__));
    }

    std::ostringstream oss;
    oss << "[DEBUG-LM-WEIGHTS] consumer=" << consumer
        << " stage=" << stage
        << " batch=" << (batch_idx + 1)
        << " ptr=" << static_cast<const void*>(lm_head_weights)
        << " vocab_size=" << vocab_size
        << " d_model=" << d_model
        << " tie_embeddings=" << (tie_embeddings ? "true" : "false")
        << " aliased_embedding=" << ((lm_head_weights == embedding_weights) ? "true" : "false")
        << " mem_type=" << cudaMemoryTypeName(attrs.type)
        << " attr_device=" << attrs.device;
    ctx.logging.logger->log(oss.str());
}

std::vector<int> buildLmValidPositionsOrThrow(
    const GRIM::Batching::BatchPayload& payload,
    const char* caller)
{
    payload.validate(caller);

    if (payload.total_tokens != payload.batch_size * payload.max_seq_len) {
        throw std::runtime_error(
            std::string(caller) + ": payload.total_tokens (" + std::to_string(payload.total_tokens) +
            ") != payload.batch_size * payload.max_seq_len (" +
            std::to_string(payload.batch_size) + " * " + std::to_string(payload.max_seq_len) +
            ") at " + __FILE__ + ":" + std::to_string(__LINE__));
    }

    if (static_cast<int>(payload.target_ids.size()) != payload.total_tokens) {
        throw std::runtime_error(
            std::string(caller) + ": payload.target_ids.size() (" +
            std::to_string(payload.target_ids.size()) + ") != payload.total_tokens (" +
            std::to_string(payload.total_tokens) + ") at " + __FILE__ + ":" +
            std::to_string(__LINE__));
    }

    std::vector<int> lm_valid_positions;
    lm_valid_positions.reserve(static_cast<size_t>(payload.lm_valid_tokens));

    int counted_actual = 0;
    for (int b = 0; b < payload.batch_size; ++b) {
        const int len = payload.seq_lengths[static_cast<size_t>(b)];
        if (len < 0 || len > payload.max_seq_len) {
            throw std::runtime_error(
                std::string(caller) + ": seq_lengths[" + std::to_string(b) + "]=" +
                std::to_string(len) + " outside [0, max_seq_len=" +
                std::to_string(payload.max_seq_len) + "] at " + __FILE__ + ":" +
                std::to_string(__LINE__));
        }

        counted_actual += len;
        const int flat_start = b * payload.max_seq_len;
        const int flat_end = flat_start + len;
        for (int t = 0; t < len; ++t) {
            const int pos = flat_start + t;
            if (pos < 0 || pos >= payload.total_tokens) {
                throw std::runtime_error(
                    std::string(caller) + ": real position " + std::to_string(pos) +
                    " outside [0, total_tokens=" + std::to_string(payload.total_tokens) +
                    ") at " + __FILE__ + ":" + std::to_string(__LINE__));
            }
            if (payload.target_ids[static_cast<size_t>(pos)] >= 0) {
                lm_valid_positions.push_back(pos);
            }
        }
        for (int pos = flat_end; pos < flat_start + payload.max_seq_len; ++pos) {
            if (payload.target_ids[static_cast<size_t>(pos)] >= 0) {
                throw std::runtime_error(
                    std::string(caller) + ": target_ids[" + std::to_string(pos) +
                    "] is valid inside padding for row " + std::to_string(b) +
                    " at " + __FILE__ + ":" + std::to_string(__LINE__));
            }
        }
    }

    if (counted_actual != payload.actual_tokens) {
        throw std::runtime_error(
            std::string(caller) + ": counted actual tokens " + std::to_string(counted_actual) +
            " != payload.actual_tokens " + std::to_string(payload.actual_tokens) +
            " at " + __FILE__ + ":" + std::to_string(__LINE__));
    }
    if (static_cast<int>(lm_valid_positions.size()) != payload.lm_valid_tokens) {
        throw std::runtime_error(
            std::string(caller) + ": counted LM-valid target positions " +
            std::to_string(lm_valid_positions.size()) + " != payload.lm_valid_tokens " +
            std::to_string(payload.lm_valid_tokens) + " at " + __FILE__ + ":" +
            std::to_string(__LINE__));
    }
    if (lm_valid_positions.empty()) {
        throw std::runtime_error(
            std::string(caller) + ": no LM-valid target positions in payload at " +
            __FILE__ + ":" + std::to_string(__LINE__));
    }

    return lm_valid_positions;
}

} // namespace

void runLogitScaleDiagnostic(
    GRIMText::Training::TrainingContext& ctx,
    ::ParameterRegistry::StartupParameterRegistry& parameter_registry,
    const GRIM::Batching::BatchPayload& payload,
    const GRIM::Tensor& logits_tensor,
    const GRIM::Tensor& lm_head_input_tensor,
    int batch_idx)
{
    namespace Internal = ::GRIMText::Training::Internal;
    auto& embedding_layer = ctx.gpu_model.requireEmbeddingLayer("runLogitScaleDiagnostic");
    auto& lm_head_layer = ctx.gpu_model.requireLmHeadLayer("runLogitScaleDiagnostic");
    auto& lm_head_parameters = parameter_registry.requireLmHeadParameters("runLogitScaleDiagnostic");
    // ========================================================================
    // TRAINING SIGNAL: Logit Statistics (argmax distribution, confidence)
    // ========================================================================
    {
        const auto& ts = ctx.requireTrainingState("runLogitScaleDiagnostic");
        if (payload.batch_size > 0 && payload.max_seq_len > 0) {
            if (!logits_tensor.data) {
                throw std::runtime_error(
                    "runLogitScaleDiagnostic: live logits tensor is NULL inside active autograd scope");
            }
            if (!lm_head_input_tensor.data) {
                throw std::runtime_error(
                    "runLogitScaleDiagnostic: live LM-head input tensor is NULL inside active autograd scope");
            }
            const int total_tokens = payload.total_tokens;
            const int vocab_size = payload.vocab_size;
            const int d_model = GRIM::HyperParameters::snapshotTrainingConfigField<int>(ctx.config, "d_model");
            const std::vector<int> lm_valid_positions =
                buildLmValidPositionsOrThrow(payload, "runLogitScaleDiagnostic");
            const int sample_positions = static_cast<int>(lm_valid_positions.size());

                    if (vocab_size != static_cast<int>(ctx.data_info.actual_vocab_size)) {
                throw std::runtime_error(
                    "runLogitScaleDiagnostic: payload.vocab_size (" + std::to_string(vocab_size) +
                        ") != ctx.data_info.actual_vocab_size (" +
                        std::to_string(ctx.data_info.actual_vocab_size) + ") at " + __FILE__ + ":" +
                    std::to_string(__LINE__));
            }

            const auto& logits_shape = logits_tensor.shape.require("runLogitScaleDiagnostic logits_tensor");
            if (!logits_shape.is_2d_layout()) {
                throw std::runtime_error("runLogitScaleDiagnostic: logits tensor must be a 2D logits buffer");
            }
            const auto logits_dims = logits_shape.as_2d();
            if (logits_dims.rows < total_tokens) {
                throw std::runtime_error(
                    "runLogitScaleDiagnostic: logits rows (" +
                    std::to_string(logits_dims.rows) + ") < payload.total_tokens (" +
                    std::to_string(total_tokens) + ") at " + __FILE__ + ":" +
                    std::to_string(__LINE__));
            }
            if (logits_dims.cols != vocab_size) {
                throw std::runtime_error(
                    "runLogitScaleDiagnostic: logits cols (" +
                    std::to_string(logits_dims.cols) + ") != payload.vocab_size (" +
                    std::to_string(vocab_size) + ") at " + __FILE__ + ":" +
                    std::to_string(__LINE__));
            }
            
            // Copy the full rectangular logits buffer once, but compute every
            // statistic below only over BatchPayload LM-valid target positions.
            // payload.lm_valid_tokens is the invariant count; target_ids[pos] >= 0
            // gives the concrete row identities needed for non-contiguous batches.
            const size_t logit_bytes = static_cast<size_t>(total_tokens) * vocab_size * sizeof(float);
            std::vector<float> logit_sample(static_cast<size_t>(total_tokens) * vocab_size);
            cudaError_t logits_copy_err = cudaMemcpy(
                logit_sample.data(), logits_tensor.data, logit_bytes, cudaMemcpyDeviceToHost);
            if (logits_copy_err != cudaSuccess) {
                throw std::runtime_error(
                    std::string("runLogitScaleDiagnostic: cudaMemcpy logits failed: ") +
                    cudaGetErrorString(logits_copy_err) + " at " + __FILE__ + ":" +
                    std::to_string(__LINE__));
            }
            
            // Compute argmax predictions and logit statistics
            std::map<int, int> argmax_counts;
            float logit_mean = 0.0f;
            double logit_sum = 0.0;
            float logit_max = -std::numeric_limits<float>::infinity();
            float logit_min = std::numeric_limits<float>::infinity();
            float max_logit_per_pos_sum = 0.0f;
            float margin_sum = 0.0f;  // Top-2 margin: logit[max] - logit[second]
            
            for (const int pos : lm_valid_positions) {
                float pos_max = -std::numeric_limits<float>::infinity();
                float pos_second = -std::numeric_limits<float>::infinity();
                int pos_argmax = 0;
                
                for (int v = 0; v < vocab_size; ++v) {
                    float logit = logit_sample[static_cast<size_t>(pos) * vocab_size + v];
                    logit_sum += static_cast<double>(logit);
                    logit_max = std::max(logit_max, logit);
                    logit_min = std::min(logit_min, logit);
                    
                    // Rule 20: Check for inf/nan IMMEDIATELY when detected (issue #142)
                    if (!std::isfinite(logit)) {
                        throw std::runtime_error(
                            "[LogitSignal] FATAL: Logit is inf/nan at batch " + 
                            std::to_string(batch_idx + 1) + 
                            ", pos=" + std::to_string(pos) + 
                            ", vocab=" + std::to_string(v) + 
                            " (value=" + std::to_string(logit) + 
                            "). Forward pass produced non-finite values. " +
                            "Check: (1) LM head matmul overflow, (2) hidden state explosion, (3) weight norm explosion."
                        );
                    }
                    
                    if (logit > pos_max) {
                        pos_second = pos_max;  // Previous max becomes second
                        pos_max = logit;
                        pos_argmax = v;
                    } else if (logit > pos_second) {
                        pos_second = logit;
                    }
                }
                argmax_counts[pos_argmax]++;
                max_logit_per_pos_sum += pos_max;
                margin_sum += (pos_max - pos_second);  // margin = max - second
            }
            
            logit_mean = static_cast<float>(logit_sum / (static_cast<double>(sample_positions) * vocab_size));
            float avg_max_logit = max_logit_per_pos_sum / sample_positions;
            float avg_margin = margin_sum / sample_positions;  // Average top-2 margin
            
            // Find top argmax predictions
            std::vector<std::pair<int, int>> sorted_argmax(argmax_counts.begin(), argmax_counts.end());
            std::sort(sorted_argmax.begin(), sorted_argmax.end(),
                      [](const auto& a, const auto& b) { return a.second > b.second; });
            
            const int total_argmax = sample_positions;
            const int top1_count = sorted_argmax.empty() ? 0 : sorted_argmax[0].second;
            int top5_count = 0;
            for (size_t i = 0; i < std::min(sorted_argmax.size(), size_t(5)); ++i) {
                top5_count += sorted_argmax[i].second;
            }
            const float top1_frac = (total_argmax > 0) ? static_cast<float>(top1_count) / total_argmax : 0.0f;
            const float top5_frac = (total_argmax > 0) ? static_cast<float>(top5_count) / total_argmax : 0.0f;

            std::ostringstream logit_stats;
            logit_stats << "[LogitSignal] batch=" << (batch_idx + 1)
                        << " logit_mean=" << Internal::formatScalar(logit_mean, 4)
                        << " logit_max=" << Internal::formatScalar(logit_max, 4)
                        << " logit_min=" << Internal::formatScalar(logit_min, 4)
                        << " avg_max_logit=" << Internal::formatScalar(avg_max_logit, 4)
                        << " top2_margin=" << Internal::formatScalar(avg_margin, 4)
                        << " argmax_top1_frac=" << Internal::formatScalar(top1_frac, 4)
                        << " argmax_top5_frac=" << Internal::formatScalar(top5_frac, 4)
                        << " unique_argmax=" << argmax_counts.size()
                        << " top_argmax=[";
            for (size_t i = 0; i < std::min(sorted_argmax.size(), size_t(5)); ++i) {
                logit_stats << "tok" << sorted_argmax[i].first << ":" << sorted_argmax[i].second;
                if (i + 1 < std::min(sorted_argmax.size(), size_t(5))) logit_stats << ",";
            }
            logit_stats << "]";
            ctx.logging.logger->log(logit_stats.str());
            
            // ================================================================
            // LOGIT SCALE EQUATION DIAGNOSTIC (Rule 21)
            //
            // logit[v] = Σ_d hidden[t,d] × W[v,d] = h · W[v]^T
            // |logit[v]| ≤ ||h|| × ||W[v]||
            // logit_range = logit_max - logit_min
            // logit_std = sqrt(Var(logits))
            //
            // EXPECTED: logit_std ≈ sqrt(d_model) × h_rms_rms × W_rms_rms,
            // over the same LM-valid target rows used by CE loss.
            // Track trends over time in stable metrics (logit_std, max_logit,
            // top2_margin, argmax concentration) rather than a fixed range cutoff.
            //
            // This diagnostic traces:
            //   1. Logit std/range across LM-valid target positions×vocab
            //   2. Hidden state norms at LM head input for the same rows
            //   3. Weight norms for top-5 + random-10 vocab tokens
            //   4. Expected vs actual logit magnitude
            // ================================================================
            {
                // --- Logit statistics ---
                const float logit_range = logit_max - logit_min;
                // Compute logit variance (already have logit_mean from above)
                double logit_var_sum = 0.0;
                for (const int pos : lm_valid_positions) {
                    for (int v = 0; v < vocab_size; ++v) {
                        const double diff = static_cast<double>(logit_sample[static_cast<size_t>(pos) * vocab_size + v]) -
                                            static_cast<double>(logit_mean);
                        logit_var_sum += diff * diff;
                    }
                }
                const float logit_std = std::sqrt(static_cast<float>(logit_var_sum / (static_cast<double>(sample_positions) * vocab_size)));
                
                // Rule 20: FAIL LOUD on NaN/inf logits (issue #142)
                if (!std::isfinite(logit_max) || !std::isfinite(logit_min)) {
                    throw std::runtime_error(
                        "[LOGIT_SCALE_EQUATION] FATAL: Logits contain inf/nan at batch " + 
                        std::to_string(batch_idx + 1) + 
                        " (logit_max=" + std::to_string(logit_max) + 
                        ", logit_min=" + std::to_string(logit_min) + 
                        ", logit_mean=" + std::to_string(logit_mean) + 
                        "). Root cause: Numerical explosion in forward pass. " +
                        "Check: (1) hidden state norms, (2) weight norms, (3) gradient explosion in previous batch."
                    );
                }
                if (!std::isfinite(logit_std)) {
                    throw std::runtime_error(
                        "[LOGIT_SCALE_EQUATION] FATAL: logit_std is nan/inf at batch " + 
                        std::to_string(batch_idx + 1) + 
                        " (logit_std=" + std::to_string(logit_std) + 
                        ", logit_mean=" + std::to_string(logit_mean) + 
                        "). Variance computation produced NaN from inf logits."
                    );
                }
                
                // --- Per-position logit range ---
                float per_pos_range_sum = 0.0f;
                float per_pos_range_max = 0.0f;
                for (const int pos : lm_valid_positions) {
                    float pos_max = -std::numeric_limits<float>::infinity();
                    float pos_min = std::numeric_limits<float>::infinity();
                    for (int v = 0; v < vocab_size; ++v) {
                        const float l = logit_sample[static_cast<size_t>(pos) * vocab_size + v];
                        pos_max = std::max(pos_max, l);
                        pos_min = std::min(pos_min, l);
                    }
                    const float pos_range = pos_max - pos_min;
                    per_pos_range_sum += pos_range;
                    per_pos_range_max = std::max(per_pos_range_max, pos_range);
                }
                const float avg_per_pos_range = per_pos_range_sum / sample_positions;
                
                // --- Hidden state norms at LM head input ---
                // The live LM-head input tensor is centered when centering is enabled.
                const float* h_src = lm_head_input_tensor.data;
                const auto& h_shape = lm_head_input_tensor.shape.require("runLogitScaleDiagnostic lm_head_input_tensor");
                if (!h_shape.is_2d_layout()) {
                    throw std::runtime_error("runLogitScaleDiagnostic: LM-head input tensor must be a 2D hidden-state buffer");
                }
                const auto h_dims = h_shape.as_2d();
                if (h_dims.rows < total_tokens) {
                    throw std::runtime_error(
                        "runLogitScaleDiagnostic: LM-head input rows (" +
                        std::to_string(h_dims.rows) + ") < payload.total_tokens (" +
                        std::to_string(total_tokens) + ") at " + __FILE__ + ":" +
                        std::to_string(__LINE__));
                }
                if (h_dims.cols != d_model) {
                    throw std::runtime_error(
                        "runLogitScaleDiagnostic: LM-head input cols (" +
                        std::to_string(h_dims.cols) + ") != d_model (" +
                        std::to_string(d_model) + ") at " + __FILE__ + ":" +
                        std::to_string(__LINE__));
                }

                const size_t h_bytes = static_cast<size_t>(total_tokens) * d_model * sizeof(float);
                std::vector<float> h_sample(static_cast<size_t>(total_tokens) * d_model);
                cudaError_t h_copy_err = cudaMemcpy(h_sample.data(), h_src, h_bytes, cudaMemcpyDeviceToHost);
                if (h_copy_err != cudaSuccess) {
                    throw std::runtime_error(
                        std::string("runLogitScaleDiagnostic: cudaMemcpy hidden failed: ") +
                        cudaGetErrorString(h_copy_err) + " at " + __FILE__ + ":" +
                        std::to_string(__LINE__));
                }

                float h_rms_max = -std::numeric_limits<float>::infinity();
                float h_rms_min = std::numeric_limits<float>::infinity();
                double h_rms_sum = 0.0;
                double h_rms_sq_sum = 0.0;
                for (const int pos : lm_valid_positions) {
                    double sum_sq = 0.0;
                    for (int d = 0; d < d_model; ++d) {
                        const double val = static_cast<double>(h_sample[static_cast<size_t>(pos) * d_model + d]);
                        sum_sq += val * val;
                    }
                    const float rms = static_cast<float>(std::sqrt(sum_sq / d_model));
                    h_rms_sum += rms;
                    h_rms_sq_sum += static_cast<double>(rms) * rms;
                    h_rms_max = std::max(h_rms_max, rms);
                    h_rms_min = std::min(h_rms_min, rms);
                }
                const float h_rms_mean = static_cast<float>(h_rms_sum / sample_positions);
                const float h_rms_rms = static_cast<float>(std::sqrt(h_rms_sq_sum / sample_positions));
                
                // Rule 20: Verify hidden state norms are finite (issue #142)
                if (!std::isfinite(h_rms_mean) || !std::isfinite(h_rms_rms) ||
                    !std::isfinite(h_rms_max) || !std::isfinite(h_rms_min)) {
                    throw std::runtime_error(
                        "[HIDDEN_STATE_NORMS] FATAL: Hidden state RMS contain inf/nan at batch " + 
                        std::to_string(batch_idx + 1) + 
                        " (h_rms_mean=" + std::to_string(h_rms_mean) + 
                        ", h_rms_rms=" + std::to_string(h_rms_rms) + 
                        ", h_rms_max=" + std::to_string(h_rms_max) + 
                        ", h_rms_min=" + std::to_string(h_rms_min) + 
                        "). Encoder output exploded. " +
                        "Check: (1) attention gradient explosion, (2) FFN activation overflow, (3) RMSNorm inverse explosion."
                    );
                }
                
                // --- Effective weight norm statistics (same W_eff semantics as LMHeadLayer::forward) ---
                // Issue #138 FIX: Compute E[||W_eff||²] (mean of squared norms) for correct expected logit_std.
                // Old code used E[||W||] (mean of norms) which underestimates by Jensen's inequality
                // when ||W|| distribution is skewed.
                const float* lm_head_weights = lm_head_parameters.weights.data;
                const float* embedding_weights = embedding_layer.tokenWeights().data;
                if (!lm_head_weights) {
                    throw std::runtime_error("runLogitScaleDiagnostic: LM-head weights are NULL");
                }
                if (!embedding_weights) {
                    throw std::runtime_error("runLogitScaleDiagnostic: embedding weights are NULL");
                }
                if (GRIM::HyperParameters::snapshotTrainingConfigField<bool>(ctx.config, "tie_embeddings")) {
                    if (lm_head_weights != embedding_weights) {
                        throw std::runtime_error("Tied embeddings: lm_head_weights and embedding_weights must alias the same buffer.");
                    }
                }
                const bool tie_embeddings =
                    GRIM::HyperParameters::snapshotTrainingConfigField<bool>(ctx.config, "tie_embeddings");

                float w_rms_mean = 0.0f, w_rms_sq_mean = 0.0f, w_rms_max = 0.0f;
                int w_rms_max_tok = -1;
                const auto& lm_head_hp = lm_head_layer.hp();
                const bool use_centered_weights = lm_head_hp.center_hidden_states;
                const bool use_token_type_gate = GRIM::kEnableLmHeadTokenTypeGateExperiment;
                // Issue #138 / Apr 2026 follow-up: replace the 500-row host-side
                // sampled CPU loop with a full-vocab on-device warp-shuffle
                // reduction (Diagnostics/LMHeadWeightStats.{cu,hpp}). Result is
                // exact over the entire vocab — no sampling miss — and costs
                // one kernel launch + 16-byte D2H instead of `vocab_sample`
                // synchronous cudaMemcpys.
                {
                    // SSoT: vocab_size from BatchPayload (validated by payload.validate()),
                    // d_model from ModelArchitecture (validated at config load).
                    cudaStream_t diag_stream = ts.stream_ctrl.getPrimaryStream();
                    validateLmHeadWeightsConsumerBoundaryOrThrow(
                        ctx,
                        lm_head_weights,
                        embedding_weights,
                        tie_embeddings,
                        payload.vocab_size,
                        d_model,
                        batch_idx,
                        "computeLMHeadWeightStats",
                        "PRE");
                    GRIM::Diagnostics::LMHeadWeightStats stats =
                        GRIM::Diagnostics::computeLMHeadWeightStats(
                            lm_head_weights,
                            payload.vocab_size,
                            d_model,
                            diag_stream,
                            use_centered_weights,
                            use_token_type_gate);
                    validateLmHeadWeightsConsumerBoundaryOrThrow(
                        ctx,
                        lm_head_weights,
                        embedding_weights,
                        tie_embeddings,
                        payload.vocab_size,
                        d_model,
                        batch_idx,
                        "computeLMHeadWeightStats",
                        "POST");
                    w_rms_mean    = stats.w_rms_mean;
                    w_rms_sq_mean = stats.w_rms_quadmean * stats.w_rms_quadmean;
                    w_rms_max     = stats.w_rms_max;
                    w_rms_max_tok = stats.w_rms_max_tok;
                }
                
                // --- Expected logit magnitude ---
                // Issue #138 FIX: Correct formula for dot product variance.
                // Var(h·W) = d × Var(h_i) × Var(W_j) = h_rms² × E[||W||²]
                // Old code used h_rms × E[||W||] which underestimates due to Jensen's inequality.
                // Correct over a row population: logit_std ≈ h_rms_rms × sqrt(E[||W||²])
                // = sqrt(d_model) × sqrt(E_t[rms(h_t)^2]) × sqrt(E_v[rms(W_v)^2]).
                const float w_rms_rms = std::sqrt(w_rms_sq_mean);  // sqrt(E[rms²])
                const float expected_logit_std = std::sqrt(static_cast<float>(d_model)) * h_rms_rms * w_rms_rms;

                // Publish raw W_rms to telemetry lattice (stream 47). CSV logger picks it up.
                // This is the W_rms term that enters logit_std = sqrt(d_model) × h_rms_rms × W_rms_rms.
                ctx.telemetry.last_obs[(int)GRIM::Telemetry::MetricStream::LM_HEAD_W_RMS_RMS] = w_rms_rms;
                const float logit_std_ratio = (expected_logit_std > 1e-10f) ? logit_std / expected_logit_std : 0.0f;

                // ============================================================
                // h↔W ALIGNMENT DIAGNOSTICS (Issue #149, telemetry streams 39-44)
                //
                // Detects the LM-head leak channel: grad_W[v] += (p_v - y_v) * h_t
                // accumulating across batches makes every W row drift along the
                // dominant h direction. This produces a coherent (h-aligned) bias
                // in W that inflates logit_std beyond the random-baseline formula
                // sqrt(d) * h_rms * W_rms (which assumes uncorrelated h, W).
                //
                // Random baseline (uncorrelated unit vectors in R^d):
                //   RMS(cos(h, W_v)) ≈ 1 / sqrt(d_model)        (≈ 0.0361 at d=768)
                // logit_std_ratio² ≈ 1 + d_model · cos_hW_rms²  (correlation correction)
                //
                // ALL ACCUMULATORS USE DOUBLE PRECISION (Rule 21: numerical precision).
                // Reads full-row data once; cost is ~500 D2H copies (already paid by
                // existing W sampling above, plus h_sample copy above). LOGIT_SCALE
                // cadence is already low-frequency so this is negligible overhead.
                // ============================================================
                float hw_cos_rms = 0.0f;
                float hw_cos_signed_mean = 0.0f;
                float hw_cos_abs_max = 0.0f;
                float hw_hbar_wbar_cos = 0.0f;
                float hw_h_dc_mean = 0.0f;
                float hw_h_dc_abs_max = 0.0f;
                // Unigram-frequency-direction collapse detector (Issue #150, streams 45-46).
                // Empirical e_uf_dir = normalize(Σ_t E[input_ids[t]]) — since positions are
                // sampled from the empirical unigram distribution, this is a Monte-Carlo
                // estimator of Σ_v p(v)·E[v].
                float unigram_dir_cos_abs_mean    = 0.0f;
                float unigram_dir_cos_signed_mean = 0.0f;
                {
                    // (1) Per-position ||h_t||² and Σ_d h[t,d] (DC component) — double accum.
                    // Arrays are rectangular so they can be indexed by payload flat row id;
                    // only LM-valid positions are populated/consumed.
                    std::vector<double> h_norm_sq(total_tokens, 0.0);
                    std::vector<double> h_dc(total_tokens, 0.0);
                    for (const int t : lm_valid_positions) {
                        const float* row = &h_sample[static_cast<size_t>(t) * d_model];
                        double sum_sq = 0.0;
                        double sum   = 0.0;
                        for (int d = 0; d < d_model; ++d) {
                            const double v = static_cast<double>(row[d]);
                            sum_sq += v * v;
                            sum   += v;
                        }
                        h_norm_sq[t] = sum_sq;
                        h_dc[t]      = sum / static_cast<double>(d_model);
                    }
                    // h_bar = mean_t h_t (per-dimension), accumulated in double.
                    std::vector<double> h_bar(d_model, 0.0);
                    for (const int t : lm_valid_positions) {
                        const float* row = &h_sample[static_cast<size_t>(t) * d_model];
                        for (int d = 0; d < d_model; ++d) {
                            h_bar[d] += static_cast<double>(row[d]);
                        }
                    }
                    const double inv_T = 1.0 / static_cast<double>(sample_positions);
                    for (int d = 0; d < d_model; ++d) h_bar[d] *= inv_T;

                    // h DC summary stats.
                    double dc_sum = 0.0;
                    double dc_abs_max = 0.0;
                    for (const int t : lm_valid_positions) {
                        dc_sum     += h_dc[t];
                        const double a = std::abs(h_dc[t]);
                        if (a > dc_abs_max) dc_abs_max = a;
                    }
                    hw_h_dc_mean    = static_cast<float>(dc_sum * inv_T);
                    hw_h_dc_abs_max = static_cast<float>(dc_abs_max);

                    // (3) Stream sampled W rows; compute cos(h_t, W_v) pairs with double accum.
                    //     Strided sample of up to 500 rows across the full vocab — matches
                    //     the cadence of the (now-replaced) host-side W_rms sampling loop.
                    const int w_sample_count = 500;
                    const int hw_stride = std::max(1, vocab_size / w_sample_count);
                    std::vector<float>  w_row_buf(d_model);
                    std::vector<double> w_bar(d_model, 0.0);

                    validateLmHeadWeightsConsumerBoundaryOrThrow(
                        ctx,
                        lm_head_weights,
                        embedding_weights,
                        tie_embeddings,
                        vocab_size,
                        d_model,
                        batch_idx,
                        "HW_ALIGNMENT",
                        "PRE");

                    long double cos_sq_sum  = 0.0L;  // long double for RMS over up to 500*sample_positions terms
                    long double cos_signed  = 0.0L;
                    double      cos_abs_max_d = 0.0;
                    int64_t     pair_count  = 0;
                    int64_t     w_sampled   = 0;

                    for (int tok = 0; tok < vocab_size && w_sampled < w_sample_count;
                         tok += hw_stride, ++w_sampled)
                    {
                        const size_t row_offset = static_cast<size_t>(tok) * d_model;
                        cudaError_t w_copy_err = cudaMemcpy(w_row_buf.data(),
                                                            lm_head_weights + row_offset,
                                                            d_model * sizeof(float),
                                                            cudaMemcpyDeviceToHost);
                        if (w_copy_err != cudaSuccess) {
                            throw std::runtime_error(
                                std::string("[HW_ALIGNMENT] cudaMemcpy W row failed: ") +
                                cudaGetErrorString(w_copy_err) + " tok=" + std::to_string(tok) +
                                " at " + __FILE__ + ":" + std::to_string(__LINE__));
                        }

                        // ||W_v||² in double.
                        double w_norm_sq = 0.0;
                        for (int d = 0; d < d_model; ++d) {
                            const double v = static_cast<double>(w_row_buf[d]);
                            w_norm_sq += v * v;
                            w_bar[d]  += v;
                        }
                        if (w_norm_sq <= 0.0 || !std::isfinite(w_norm_sq)) continue;
                        const double inv_w_norm = 1.0 / std::sqrt(w_norm_sq);

                        for (const int t : lm_valid_positions) {
                            if (h_norm_sq[t] <= 0.0 || !std::isfinite(h_norm_sq[t])) continue;
                            const float* h_row = &h_sample[static_cast<size_t>(t) * d_model];
                            // dot(h_t, W_v) in double.
                            double dot = 0.0;
                            for (int d = 0; d < d_model; ++d) {
                                dot += static_cast<double>(h_row[d]) * static_cast<double>(w_row_buf[d]);
                            }
                            const double inv_h_norm = 1.0 / std::sqrt(h_norm_sq[t]);
                            const double cos_tv = dot * inv_h_norm * inv_w_norm;
                            cos_signed += static_cast<long double>(cos_tv);
                            cos_sq_sum += static_cast<long double>(cos_tv) * static_cast<long double>(cos_tv);
                            const double a = std::abs(cos_tv);
                            if (a > cos_abs_max_d) cos_abs_max_d = a;
                            ++pair_count;
                        }
                    }

                    if (pair_count > 0) {
                        const long double inv_n = 1.0L / static_cast<long double>(pair_count);
                        hw_cos_rms         = static_cast<float>(std::sqrt(static_cast<double>(cos_sq_sum * inv_n)));
                        hw_cos_signed_mean = static_cast<float>(static_cast<double>(cos_signed * inv_n));
                        hw_cos_abs_max     = static_cast<float>(cos_abs_max_d);
                    }

                    validateLmHeadWeightsConsumerBoundaryOrThrow(
                        ctx,
                        lm_head_weights,
                        embedding_weights,
                        tie_embeddings,
                        vocab_size,
                        d_model,
                        batch_idx,
                        "HW_ALIGNMENT",
                        "POST");

                    // (4) Rank-1 DC channel: cos(h_bar, W_bar).
                    if (w_sampled > 0) {
                        const double inv_W = 1.0 / static_cast<double>(w_sampled);
                        double hbar_dot_wbar = 0.0;
                        double hbar_norm_sq  = 0.0;
                        double wbar_norm_sq  = 0.0;
                        for (int d = 0; d < d_model; ++d) {
                            const double w = w_bar[d] * inv_W;
                            const double h = h_bar[d];
                            hbar_dot_wbar += h * w;
                            hbar_norm_sq  += h * h;
                            wbar_norm_sq  += w * w;
                        }
                        if (hbar_norm_sq > 0.0 && wbar_norm_sq > 0.0) {
                            hw_hbar_wbar_cos = static_cast<float>(
                                hbar_dot_wbar / std::sqrt(hbar_norm_sq * wbar_norm_sq));
                        }
                    }

                    // ============================================================
                    // UNIGRAM-FREQUENCY-DIRECTION COLLAPSE DETECTOR (Issue #150)
                    //
                    // Build empirical e_uf_dir = normalize(Σ_t E[input_ids[t]]) using
                    // ONLY the LM-valid target positions used by this logit-scale diagnostic.
                    // Then compute mean_t |cos(h_t, e_uf_dir)| and signed mean.
                    //
                    // High |cos| during steps 0–600 confirms hidden states are
                    // collapsing toward the dominant-token direction (representation
                    // bug, not optimizer bug). Random baseline ≈ sqrt(2/(π·d)) ≈ 0.029.
                    // ============================================================
                    {
                        // (1) Tally unique token counts at LM-valid positions in the
                        //     same flattened layout as h_sample (pos = b*seq_len + t).
                        std::map<int, int> tok_counts;
                        std::vector<int> pos_to_tok(total_tokens, -1);
                        for (const int pos : lm_valid_positions) {
                            const int tok = payload.input_ids[static_cast<size_t>(pos)];
                            if (tok < 0 || tok >= vocab_size) {
                                throw std::runtime_error(
                                    "[UNIGRAM_DIR] FATAL: input token out of vocab at batch " +
                                    std::to_string(batch_idx + 1) + ", pos=" +
                                    std::to_string(pos) + ", tok=" + std::to_string(tok) +
                                    ", vocab_size=" + std::to_string(vocab_size));
                            }
                            pos_to_tok[pos] = tok;
                            ++tok_counts[tok];
                        }

                        if (!tok_counts.empty()) {
                            // (2) Accumulate e_uf_dir = Σ count(tok) · E[tok] in double.
                            std::vector<double> e_uf_dir(d_model, 0.0);
                            std::vector<float>  e_row(d_model);
                            int64_t total_valid = 0;
                            for (const auto& kv : tok_counts) {
                                const int tok = kv.first;
                                const int cnt = kv.second;
                                cudaError_t e_copy_err = cudaMemcpy(
                                    e_row.data(),
                                    embedding_weights + static_cast<size_t>(tok) * d_model,
                                    d_model * sizeof(float),
                                    cudaMemcpyDeviceToHost);
                                if (e_copy_err != cudaSuccess) {
                                    throw std::runtime_error(
                                        std::string("[UNIGRAM_DIR] cudaMemcpy embedding row failed: ") +
                                        cudaGetErrorString(e_copy_err) + " tok=" + std::to_string(tok) +
                                        " at " + __FILE__ + ":" + std::to_string(__LINE__));
                                }
                                for (int d = 0; d < d_model; ++d) {
                                    e_uf_dir[d] += static_cast<double>(cnt) * static_cast<double>(e_row[d]);
                                }
                                total_valid += cnt;
                            }

                            // (3) Normalize e_uf_dir.
                            double e_norm_sq = 0.0;
                            for (int d = 0; d < d_model; ++d) e_norm_sq += e_uf_dir[d] * e_uf_dir[d];
                            if (e_norm_sq > 0.0 && std::isfinite(e_norm_sq) && total_valid > 0) {
                                const double inv_e_norm = 1.0 / std::sqrt(e_norm_sq);

                                // (4) Per-position cos(h_t, e_uf_dir); mean abs / signed.
                                long double cos_signed_sum  = 0.0L;
                                long double cos_abs_sum     = 0.0L;
                                int64_t     cos_count       = 0;
                                for (const int pos : lm_valid_positions) {
                                    if (pos_to_tok[pos] < 0) continue;             // skip pad
                                    if (h_norm_sq[pos] <= 0.0 || !std::isfinite(h_norm_sq[pos])) continue;
                                    const float* h_row = &h_sample[static_cast<size_t>(pos) * d_model];
                                    double dot = 0.0;
                                    for (int d = 0; d < d_model; ++d) {
                                        dot += static_cast<double>(h_row[d]) * e_uf_dir[d];
                                    }
                                    const double inv_h_norm = 1.0 / std::sqrt(h_norm_sq[pos]);
                                    const double cos_te     = dot * inv_h_norm * inv_e_norm;
                                    cos_signed_sum += static_cast<long double>(cos_te);
                                    cos_abs_sum    += static_cast<long double>(std::abs(cos_te));
                                    ++cos_count;
                                }

                                if (cos_count > 0) {
                                    const long double inv_n = 1.0L / static_cast<long double>(cos_count);
                                    unigram_dir_cos_abs_mean    = static_cast<float>(static_cast<double>(cos_abs_sum    * inv_n));
                                    unigram_dir_cos_signed_mean = static_cast<float>(static_cast<double>(cos_signed_sum * inv_n));
                                }
                            }
                        }
                    }

                    // Rule 20: fail loud on NaN/Inf in alignment metrics.
                    if (!std::isfinite(hw_cos_rms) || !std::isfinite(hw_cos_signed_mean) ||
                        !std::isfinite(hw_cos_abs_max) || !std::isfinite(hw_hbar_wbar_cos) ||
                        !std::isfinite(hw_h_dc_mean) || !std::isfinite(hw_h_dc_abs_max) ||
                        !std::isfinite(unigram_dir_cos_abs_mean) ||
                        !std::isfinite(unigram_dir_cos_signed_mean))
                    {
                        throw std::runtime_error(
                            "[HW_ALIGNMENT] FATAL: h↔W alignment metrics contain NaN/Inf at batch " +
                            std::to_string(batch_idx + 1));
                    }
                }

                // Publish to telemetry lattice (streams 39-46). CSV logger picks these up.
                ctx.telemetry.last_obs[39] = hw_cos_rms;          // HW_COS_RMS
                ctx.telemetry.last_obs[40] = hw_cos_signed_mean;  // HW_COS_SIGNED_MEAN
                ctx.telemetry.last_obs[41] = hw_cos_abs_max;      // HW_COS_ABS_MAX
                ctx.telemetry.last_obs[42] = hw_hbar_wbar_cos;    // HW_HBAR_WBAR_COS
                ctx.telemetry.last_obs[43] = hw_h_dc_mean;        // HW_H_DC_MEAN
                ctx.telemetry.last_obs[44] = hw_h_dc_abs_max;     // HW_H_DC_ABS_MAX
                ctx.telemetry.last_obs[45] = unigram_dir_cos_abs_mean;    // UNIGRAM_DIR_COS_ABS_MEAN
                ctx.telemetry.last_obs[46] = unigram_dir_cos_signed_mean; // UNIGRAM_DIR_COS_SIGNED_MEAN

                struct LogitTrendState {
                    bool initialized = false;
                    float ema_logit_std = 0.0f;
                    float ema_logit_max = 0.0f;
                    float ema_top2_margin = 0.0f;
                    float ema_top1_frac = 0.0f;
                };
                static LogitTrendState trend_state;
                const float trend_alpha = 0.10f;
                if (!trend_state.initialized) {
                    trend_state.initialized = true;
                    trend_state.ema_logit_std = logit_std;
                    trend_state.ema_logit_max = logit_max;
                    trend_state.ema_top2_margin = avg_margin;
                    trend_state.ema_top1_frac = top1_frac;
                } else {
                    trend_state.ema_logit_std = trend_alpha * logit_std + (1.0f - trend_alpha) * trend_state.ema_logit_std;
                    trend_state.ema_logit_max = trend_alpha * logit_max + (1.0f - trend_alpha) * trend_state.ema_logit_max;
                    trend_state.ema_top2_margin = trend_alpha * avg_margin + (1.0f - trend_alpha) * trend_state.ema_top2_margin;
                    trend_state.ema_top1_frac = trend_alpha * top1_frac + (1.0f - trend_alpha) * trend_state.ema_top1_frac;
                }
                
                std::ostringstream scale_eq;
                scale_eq << std::fixed << std::setprecision(6);
                scale_eq << "[LOGIT_SCALE_EQUATION] logit[v] = h · W[v]^T, logit_range = max - min\n";
                scale_eq << "  POPULATION: lm_valid_tokens=" << sample_positions
                         << " total_tokens=" << total_tokens
                         << " masked_tokens=" << (total_tokens - sample_positions) << "\n";
                scale_eq << "  LOGIT STATS: std=" << logit_std << " range=" << logit_range
                         << " avg_per_pos_range=" << avg_per_pos_range
                         << " max_per_pos_range=" << per_pos_range_max << "\n";
                scale_eq << "  HIDDEN (LM input): h_rms_mean=" << h_rms_mean
                         << " h_rms_rms=" << h_rms_rms
                         << " h_rms_max=" << h_rms_max << " h_rms_min=" << h_rms_min << "\n";
                    scale_eq << "  WEIGHTS (LM head effective W_eff): W_rms_mean=" << w_rms_mean
                         << " W_rms_rms=" << w_rms_rms
                         << " W_rms_max=" << w_rms_max << " (tok=" << w_rms_max_tok << ")"
                         << " d_model=" << d_model << "\n";
                    scale_eq << "  WEIGHT TRANSFORM: "
                         << (use_centered_weights && use_token_type_gate
                               ? "center_rows_by_token_type_gate(W_lm)"
                               : use_centered_weights
                                   ? "center_rows(W_lm)"
                                   : use_token_type_gate
                                       ? "type_gate_rows_by_token_type(W_lm)"
                                       : "W_lm")
                         << "\n";
                    scale_eq << "  EXPECTED logit_std = sqrt(d_model) × h_rms_rms × W_eff_rms_rms\n";
                scale_eq << "                      = sqrt(" << d_model << ") × " << h_rms_rms
                         << " × " << w_rms_rms << "\n";
                scale_eq << "                      = " << expected_logit_std << "\n";
                scale_eq << "  ACTUAL logit_std = " << logit_std
                         << " ratio(actual/expected)=" << logit_std_ratio << "\n";
                scale_eq << "  TREND (EMA α=" << trend_alpha << ")"
                         << " logit_std_delta=" << (logit_std - trend_state.ema_logit_std)
                         << " max_logit_delta=" << (logit_max - trend_state.ema_logit_max)
                         << " top2_margin_delta=" << (avg_margin - trend_state.ema_top2_margin)
                         << " top1_frac_delta=" << (top1_frac - trend_state.ema_top1_frac) << "\n";
                if (w_rms_max > 2.0f) {
                    scale_eq << "  [ANOMALY] WEIGHT_RMS_EXPLOSION: W_rms_max=" << w_rms_max
                             << " (tok=" << w_rms_max_tok << ") >> 2.0. Weight decay too weak or gradient bias.\n";
                }
                if (logit_std_ratio > 3.0f) {
                    scale_eq << "  [ANOMALY] LOGIT_STD_MISMATCH: actual/expected=" << logit_std_ratio
                             << " >> 3.0. Possible hidden-weight alignment or missing 1/sqrt(d) scaling.\n";
                }
                ctx.logging.logger->log(scale_eq.str());
            }
            
            // ================================================================
            // LM HEAD DIAGNOSTICS: Row norms ||W[v]|| for top predicted tokens
            // ================================================================
            const float* lm_head_weights_for_norms = lm_head_parameters.weights.data;
            if (!lm_head_weights_for_norms) {
                throw std::runtime_error("[LMHeadNorm] LM-head weights are NULL");
            }
            {
                const bool tie_embeddings =
                    GRIM::HyperParameters::snapshotTrainingConfigField<bool>(ctx.config, "tie_embeddings");
                const float* embedding_weights = embedding_layer.tokenWeights().data;
                validateLmHeadWeightsConsumerBoundaryOrThrow(
                    ctx,
                    lm_head_weights_for_norms,
                    embedding_weights,
                    tie_embeddings,
                    vocab_size,
                    d_model,
                    batch_idx,
                    "LMHeadNorm",
                    "PRE");
                // Copy LM head rows for top-5 predicted tokens
                std::ostringstream lm_stats;
                lm_stats << "[LMHeadNorm] batch=" << (batch_idx + 1) << " rms(W[v])=[";
                
                std::vector<float> row_buffer(d_model);
                for (size_t i = 0; i < std::min(sorted_argmax.size(), size_t(5)); ++i) {
                    int tok_id = sorted_argmax[i].first;
                    // Copy row [tok_id, :] from W [vocab_size, d_model]
                    const size_t row_offset = static_cast<size_t>(tok_id) * d_model;
                    cudaError_t norm_copy_err = cudaMemcpy(
                        row_buffer.data(),
                        lm_head_weights_for_norms + row_offset,
                        d_model * sizeof(float), cudaMemcpyDeviceToHost);
                    if (norm_copy_err != cudaSuccess) {
                        throw std::runtime_error(
                            std::string("[LMHeadNorm] cudaMemcpy W row failed: ") +
                            cudaGetErrorString(norm_copy_err) + " tok=" + std::to_string(tok_id) +
                            " at " + __FILE__ + ":" + std::to_string(__LINE__));
                    }
                    
                    // Compute RMS of row
                    float sum_sq = 0.0f;
                    for (int d = 0; d < d_model; ++d) {
                        sum_sq += row_buffer[d] * row_buffer[d];
                    }
                    float row_rms = std::sqrt(sum_sq / d_model);
                    lm_stats << "tok" << tok_id << ":" << Internal::formatScalar(row_rms, 10);
                    if (i + 1 < std::min(sorted_argmax.size(), size_t(5))) lm_stats << ",";
                }
                lm_stats << "]";
                ctx.logging.logger->log(lm_stats.str());
                validateLmHeadWeightsConsumerBoundaryOrThrow(
                    ctx,
                    lm_head_weights_for_norms,
                    embedding_weights,
                    tie_embeddings,
                    vocab_size,
                    d_model,
                    batch_idx,
                    "LMHeadNorm",
                    "POST");
            }
        }
    }
}

} // namespace GRIM::Diagnostics
