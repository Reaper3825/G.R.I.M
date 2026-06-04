//======================================================//
//  SpecialTokenDiagnostic.cu
//  Lifted verbatim from Phase2_TrainingLoop.cu — the
//  "DIAGNOSTIC: Issue #142 - Special Token Weight &
//  Gradient Verification" scope.
//  Behavior: identical. No logic, gating, ordering, or
//  log-string changes vs. the original inline block.
//======================================================//

#include "SpecialTokenDiagnostic.hpp"
#include "DiagnosticGates.hpp"

#include "../Phases/Phase2_TrainingLoop.hpp"
#include "../../Shared/LogRecorder/LogRecorder.hpp"

#include <vector>
#include <sstream>
#include <iomanip>
#include <cmath>
#include <stdexcept>
#include <string>

#include <cuda_runtime.h>

namespace GRIM::Diagnostics {

void runSpecialTokenDiagnostic(
    GRIMText::Training::TrainingContext& ctx,
    ::ParameterRegistry::StartupParameterRegistry& parameter_registry,
    const GRIM::Batching::BatchPayload& payload,
    int batch_idx)
{
    namespace Internal = ::GRIMText::Training::Internal;
    if (shouldSyncDiagnostics(ctx, batch_idx) && ctx.logging.tape && ctx.logging.tape->accepts(GRIM::Logging::LogLevel::Debug)) {
        auto& embedding_parameters = parameter_registry.requireEmbeddingParameters("runSpecialTokenDiagnostic");
        auto& lm_head_parameters = parameter_registry.requireLmHeadParameters("runSpecialTokenDiagnostic");
        const int d_model = GRIM::HyperParameters::snapshotTrainingConfigField<int>(ctx.config, "d_model");
        const float* weights_ptr = lm_head_parameters.weights.data;
        // Issue #150: When tied=no, LM head and embedding are DIFFERENT tensors.
        // Read gradients from the SAME layer as weights (LM head) so rms(W) and
        // rms(grad) refer to the same parameter. Previously read embedding grads,
        // which showed PAD scatter-add accumulation (~1754 positions) as 76x spike
        // vs BOS/EOS — misleading because that gradient doesn't affect LM head.
        const bool weights_tied = embedding_parameters.token_weights.data == lm_head_parameters.weights.data;
        const float* grads_ptr = weights_tied
            ? embedding_parameters.token_weights.grad_data()   // tied: same tensor, either pointer works
            : lm_head_parameters.weights.grad_data();      // untied: use LM head's own gradients

        if (weights_ptr) {
                constexpr int SPECIAL_IDS[] = {
                    GRIM::Tokenizer::UNK_TOKEN_ID,   // 0
                    GRIM::Tokenizer::PAD_TOKEN_ID,    // 1
                    GRIM::Tokenizer::BOS_TOKEN_ID,    // 2
                    GRIM::Tokenizer::EOS_TOKEN_ID     // 3
                };
                constexpr const char* SPECIAL_NAMES[] = {"UNK", "PAD", "BOS", "EOS"};
                constexpr int NUM_SPECIALS = 4;

                std::vector<float> row_buf(d_model);
                std::ostringstream diag;
                diag << std::fixed << std::setprecision(8);
                diag << "[SPECIAL_TOKEN_EQUATION] batch=" << (batch_idx + 1)
                     << " W_special health: logit[v] = h · W[v]^T\n";

                // Also sample a few content token norms for comparison baseline
                double content_norm_sum = 0.0;
                int content_norm_count = 0;
                constexpr int CONTENT_SAMPLE_IDS[] = {512, 1000, 5000, 10000, 25000, 40000};
                for (int cid : CONTENT_SAMPLE_IDS) {
                    if (cid >= payload.vocab_size) continue;
                    const size_t off = static_cast<size_t>(cid) * d_model;
                    cudaMemcpy(row_buf.data(), weights_ptr + off,
                               d_model * sizeof(float), cudaMemcpyDeviceToHost);
                    double sq = 0.0;
                    for (int d = 0; d < d_model; ++d) sq += static_cast<double>(row_buf[d]) * row_buf[d];
                    content_norm_sum += std::sqrt(sq / d_model);
                    content_norm_count++;
                }
                const double content_norm_mean = (content_norm_count > 0)
                    ? content_norm_sum / content_norm_count : 0.0;

                for (int s = 0; s < NUM_SPECIALS; ++s) {
                    const int tok_id = SPECIAL_IDS[s];
                    const size_t row_offset = static_cast<size_t>(tok_id) * d_model;

                    // Weight row
                    cudaMemcpy(row_buf.data(), weights_ptr + row_offset,
                               d_model * sizeof(float), cudaMemcpyDeviceToHost);
                    double w_sq = 0.0, w_sum = 0.0;
                    for (int d = 0; d < d_model; ++d) {
                        w_sq += static_cast<double>(row_buf[d]) * row_buf[d];
                        w_sum += row_buf[d];
                    }
                    const float w_rms = static_cast<float>(std::sqrt(w_sq / d_model));
                    const float w_mean = static_cast<float>(w_sum / d_model);

                    // Gradient row (may be null if not yet computed)
                    float g_rms = 0.0f, g_sum = 0.0f;
                    bool has_grad = false;
                    if (grads_ptr) {
                        cudaMemcpy(row_buf.data(), grads_ptr + row_offset,
                                   d_model * sizeof(float), cudaMemcpyDeviceToHost);
                        double g_sq = 0.0, gs = 0.0;
                        bool any_nonzero = false;
                        for (int d = 0; d < d_model; ++d) {
                            g_sq += static_cast<double>(row_buf[d]) * row_buf[d];
                            gs += row_buf[d];
                            if (row_buf[d] != 0.0f) any_nonzero = true;
                        }
                        g_rms = static_cast<float>(std::sqrt(g_sq / d_model));
                        g_sum = static_cast<float>(gs);
                        has_grad = any_nonzero;
                    }

                    // Count appearances as INPUT token in this batch
                    // (special tokens only appear as input if BOS is prepended, etc.)
                    // We don't have input_ids readily available here, so skip input count.

                    diag << "  " << SPECIAL_NAMES[s] << "(id=" << tok_id << "): "
                         << "rms(W)=" << w_rms
                         << " w_mean=" << w_mean;
                    if (grads_ptr) {
                        diag << " rms(grad)=" << g_rms
                             << " grad_sum=" << g_sum
                             << (has_grad ? "" : " [ZERO_GRAD]");
                    } else {
                        diag << " [NO_GRAD_BUFFER]";
                    }

                    // Anomaly: special token weight RMS diverging from content tokens
                    if (content_norm_mean > 0.0 && w_rms > 3.0f * content_norm_mean) {
                        diag << " [ANOMALY] rms(W)=" << w_rms
                             << " >> content_mean=" << Internal::formatScalar(static_cast<float>(content_norm_mean), 6);
                    }
                    if (w_rms < 1e-6f) {
                        diag << " [ANOMALY] NEAR_ZERO_WEIGHT";
                    }
                    if (!std::isfinite(w_rms) || !std::isfinite(g_rms)) {
                        throw std::runtime_error(
                            "[SPECIAL_TOKEN_EQUATION] Non-finite special token weight/grad: "
                            + std::string(SPECIAL_NAMES[s]) + " rms(W)=" + std::to_string(w_rms)
                            + " rms(grad)=" + std::to_string(g_rms)
                            + " at batch " + std::to_string(batch_idx + 1));
                    }
                    diag << "\n";
                }
                diag << "  content_baseline: rms(W)_mean=" << Internal::formatScalar(static_cast<float>(content_norm_mean), 6)
                     << " (sampled " << content_norm_count << " tokens)";

                ctx.logging.logger->log(diag.str());
                EQ_LOG(ctx.logging.tape.get(), GRIM::Logging::LogGroup::Embedding, GRIM::Logging::LogPhase::GRADIENT_CLIP, 0, "SPECIAL_TOKEN_EQUATION", diag.str().c_str());
            }
    }
}

} // namespace GRIM::Diagnostics
