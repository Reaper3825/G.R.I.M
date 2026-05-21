//======================================================//
//  PostClipParamGradEmbLmEquation.cu
//  Post-clip embedding/LM parameter-gradient equation.
//
//  Boundary: this module belongs to the gradient-clipping diagnostic boundary.
//  It is not a TensorContract GradFn diagnostic and owns no cached token or
//  cross-microbatch state. The token-frequency line is explicitly current-
//  payload context for the completed parameter-gradient buffer.
//======================================================//

#include "PostClipParamGradEmbLmEquation.hpp"

#include "../Phases/Phase2_TrainingLoop.hpp"
#include "../../Layers/Embedding/Embedding_GPU.hpp"
#include "../../Layers/LMHead/lm_head_GPU.hpp"
#include "../../Shared/Batching/BatchPayload.hpp"
#include "../../Shared/LogRecorder/BatchLogTape.hpp"
#include "../../Shared/LogRecorder/LogTypes.hpp"
#include "../../Shared/UnigramByte/TokenLayout.hpp"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <cuda_runtime.h>

namespace GRIM::Diagnostics {

namespace {

struct PostClipParamGradEmbLmEquationDiag {
    bool valid = false;
    bool tied_embeddings = false;

    float grad_rms = 0.0f;
    float mean_row_rms = 0.0f;
    float max_row_rms = 0.0f;
    int max_row_token = -1;
    float spike_ratio = 0.0f;

    static constexpr int kTopK = 5;
    int top_tokens[kTopK] = {};
    float top_rms[kTopK] = {};

    int num_active_rows = 0;
    int total_vocab = 0;
    float active_ratio = 0.0f;

    int current_payload_most_frequent_token = -1;
    int current_payload_most_frequent_count = 0;
    int current_payload_pad_slots_skipped = 0;
    int current_payload_real_tokens_counted = 0;

    float prev_emb_rms = 0.0f;
    float curr_emb_rms = 0.0f;
    float emb_rms_delta = 0.0f;
};

void cudaCheck(cudaError_t err, const char* where) {
    if (err != cudaSuccess) {
        throw std::runtime_error(
            std::string("[") + kPostClipParamGradEmbLmEquationOp + "] CUDA error at " +
            where + ": " + cudaGetErrorString(err));
    }
}

PostClipParamGradEmbLmEquationDiag computePostClipParamGradEmbLmEquation(
    const GRIM::EmbeddingLayer* embedding_layer,
    const GRIM::LMHeadLayer* lm_head_layer,
    const GRIM::Batching::BatchPayload& payload,
    int d_model,
    int vocab_size,
    bool tied_embeddings,
    float prev_emb_rms,
    float curr_emb_rms,
    cudaStream_t stream)
{
    if (!embedding_layer) {
        throw std::runtime_error(
            std::string("[") + kPostClipParamGradEmbLmEquationOp +
            "] embedding_layer is NULL");
    }
    if (payload.input_ids.empty()) {
        throw std::runtime_error(
            std::string("[") + kPostClipParamGradEmbLmEquationOp +
            "] BatchPayload.input_ids is empty at enabled diagnostic boundary");
    }
    if (static_cast<int>(payload.input_ids.size()) != payload.total_tokens) {
        throw std::runtime_error(
            std::string("[") + kPostClipParamGradEmbLmEquationOp +
            "] BatchPayload.input_ids.size()=" + std::to_string(payload.input_ids.size()) +
            " != payload.total_tokens=" + std::to_string(payload.total_tokens));
    }
    if (static_cast<int>(payload.seq_lengths.size()) != payload.batch_size) {
        throw std::runtime_error(
            std::string("[") + kPostClipParamGradEmbLmEquationOp +
            "] BatchPayload.seq_lengths.size()=" + std::to_string(payload.seq_lengths.size()) +
            " != payload.batch_size=" + std::to_string(payload.batch_size));
    }
    if (d_model <= 0) {
        throw std::runtime_error(
            std::string("[") + kPostClipParamGradEmbLmEquationOp +
            "] d_model must be > 0, got " + std::to_string(d_model));
    }
    if (vocab_size <= 0) {
        throw std::runtime_error(
            std::string("[") + kPostClipParamGradEmbLmEquationOp +
            "] vocab_size must be > 0, got " + std::to_string(vocab_size));
    }
    if (!lm_head_layer) {
        throw std::runtime_error(
            std::string("[") + kPostClipParamGradEmbLmEquationOp +
            "] lm_head_layer is NULL");
    }
    if (!stream) {
        throw std::runtime_error(
            std::string("[") + kPostClipParamGradEmbLmEquationOp +
            "] stream is NULL — caller MUST provide the active training stream");
    }

    const float* embedding_grad_ptr = embedding_layer->tokenWeights().grad_data();
    if (!embedding_grad_ptr) {
        throw std::runtime_error(
            std::string("[") + kPostClipParamGradEmbLmEquationOp +
            "] embedding tokenWeights grad_data is NULL after clipping measurement");
    }

    const float* lm_grad_ptr = lm_head_layer->weights().grad_data();
    if (!lm_grad_ptr) {
        throw std::runtime_error(
            std::string("[") + kPostClipParamGradEmbLmEquationOp +
            "] lm_head weights grad_data is NULL after clipping measurement");
    }
    if (tied_embeddings && lm_grad_ptr != embedding_grad_ptr) {
        throw std::runtime_error(
            std::string("[") + kPostClipParamGradEmbLmEquationOp +
            "] tied-embedding invariant violated: LM head grad buffer does not alias embedding grad buffer");
    }
    if (!tied_embeddings && lm_grad_ptr == embedding_grad_ptr) {
        throw std::runtime_error(
            std::string("[") + kPostClipParamGradEmbLmEquationOp +
            "] untied-embedding invariant violated: LM head and embedding grad buffers unexpectedly alias");
    }

    PostClipParamGradEmbLmEquationDiag diag{};
    diag.tied_embeddings = tied_embeddings;
    diag.total_vocab = vocab_size;
    diag.prev_emb_rms = prev_emb_rms;
    diag.curr_emb_rms = curr_emb_rms;
    diag.emb_rms_delta = curr_emb_rms - prev_emb_rms;

    std::unordered_map<int, int> token_freq;
    for (int b = 0; b < payload.batch_size; ++b) {
        const int seq_len = payload.seq_lengths[static_cast<size_t>(b)];
        if (seq_len < 0 || seq_len > payload.max_seq_len) {
            throw std::runtime_error(
                std::string("[") + kPostClipParamGradEmbLmEquationOp +
                "] invalid BatchPayload.seq_lengths[" + std::to_string(b) + "]=" +
                std::to_string(seq_len) + " for max_seq_len=" + std::to_string(payload.max_seq_len));
        }
        diag.current_payload_pad_slots_skipped += payload.max_seq_len - seq_len;
        const int row_offset = b * payload.max_seq_len;
        for (int t = 0; t < seq_len; ++t) {
            const int tok = payload.input_ids[static_cast<size_t>(row_offset + t)];
            if (tok == GRIM::Tokenizer::PAD_TOKEN_ID) {
                ++diag.current_payload_pad_slots_skipped;
                continue;
            }
            if (tok >= 0 && tok < vocab_size) {
                token_freq[tok]++;
                ++diag.current_payload_real_tokens_counted;
            }
        }
    }

    for (const auto& [tok, count] : token_freq) {
        if (count > diag.current_payload_most_frequent_count) {
            diag.current_payload_most_frequent_count = count;
            diag.current_payload_most_frequent_token = tok;
        }
    }

    const size_t grad_size = static_cast<size_t>(vocab_size) * static_cast<size_t>(d_model);
    std::vector<float> h_embedding_grad(grad_size);
    cudaCheck(cudaMemcpyAsync(
        h_embedding_grad.data(), embedding_grad_ptr, grad_size * sizeof(float), cudaMemcpyDeviceToHost, stream),
        tied_embeddings ? "cudaMemcpyAsync(shared embedding/LM grad D2H)"
                         : "cudaMemcpyAsync(embedding grad D2H)");

    std::vector<float> h_lm_grad;
    if (!tied_embeddings) {
        h_lm_grad.resize(grad_size);
        cudaCheck(cudaMemcpyAsync(
            h_lm_grad.data(), lm_grad_ptr, grad_size * sizeof(float), cudaMemcpyDeviceToHost, stream),
            "cudaMemcpyAsync(lm grad D2H)");
    }
    cudaCheck(cudaStreamSynchronize(stream), tied_embeddings
        ? "cudaStreamSynchronize(shared embedding/LM grad D2H)"
        : "cudaStreamSynchronize(embedding+lm grad D2H)");

    std::vector<float> row_rms_vals(vocab_size, 0.0f);
    double total_sum_sq = 0.0;
    size_t total_count = 0;
    for (int v = 0; v < vocab_size; ++v) {
        const float* embedding_row =
            h_embedding_grad.data() + static_cast<size_t>(v) * static_cast<size_t>(d_model);
        const float* lm_row = tied_embeddings
            ? embedding_row
            : h_lm_grad.data() + static_cast<size_t>(v) * static_cast<size_t>(d_model);
        double row_sq = 0.0;
        for (int d = 0; d < d_model; ++d) {
            row_sq += static_cast<double>(lm_row[d]) * static_cast<double>(lm_row[d]);
            if (!tied_embeddings) {
                row_sq += static_cast<double>(embedding_row[d]) * static_cast<double>(embedding_row[d]);
            }
        }
        const double row_count = static_cast<double>(d_model) * (tied_embeddings ? 1.0 : 2.0);
        row_rms_vals[v] = static_cast<float>(std::sqrt(row_sq / row_count));
        total_sum_sq += row_sq;
        total_count += static_cast<size_t>(d_model) * static_cast<size_t>(tied_embeddings ? 1 : 2);
        if (row_rms_vals[v] > 1e-10f) {
            diag.num_active_rows++;
        }
    }

    diag.grad_rms = total_count > 0
        ? static_cast<float>(std::sqrt(total_sum_sq / static_cast<double>(total_count)))
        : 0.0f;
    diag.active_ratio = static_cast<float>(diag.num_active_rows) / static_cast<float>(vocab_size);

    std::vector<std::pair<float, int>> rms_idx(vocab_size);
    for (int v = 0; v < vocab_size; ++v) {
        rms_idx[v] = {row_rms_vals[v], v};
    }
    const int top_count = std::min(vocab_size, PostClipParamGradEmbLmEquationDiag::kTopK);
    std::partial_sort(
        rms_idx.begin(), rms_idx.begin() + top_count,
        rms_idx.end(), [](const auto& a, const auto& b) { return a.first > b.first; });

    for (int k = 0; k < top_count; ++k) {
        diag.top_tokens[k] = rms_idx[k].second;
        diag.top_rms[k] = rms_idx[k].first;
    }

    diag.max_row_rms = rms_idx[0].first;
    diag.max_row_token = rms_idx[0].second;

    if (diag.num_active_rows > 0) {
        double sum_rms = 0.0;
        for (int v = 0; v < vocab_size; ++v) {
            if (row_rms_vals[v] > 1e-10f) {
                sum_rms += row_rms_vals[v];
            }
        }
        diag.mean_row_rms = static_cast<float>(sum_rms / static_cast<double>(diag.num_active_rows));
    }

    diag.spike_ratio = diag.mean_row_rms > 1e-10f
        ? diag.max_row_rms / diag.mean_row_rms
        : 0.0f;

    diag.valid = true;
    return diag;
}

std::string formatPostClipParamGradEmbLmEquation(
    const PostClipParamGradEmbLmEquationDiag& diag,
    int batch_idx)
{
    if (!diag.valid) {
        throw std::runtime_error(
            std::string("[") + kPostClipParamGradEmbLmEquationOp +
            "] formatter received invalid diagnostic state");
    }

    std::ostringstream oss;
    oss << std::fixed << std::setprecision(6);

    oss << "[" << kPostClipParamGradEmbLmEquationOp << "] accum_boundary_batch="
        << (batch_idx + 1);
    if (diag.tied_embeddings) {
        oss << " WEIGHT_GRAD: grad_W_tied = grad_lm + grad_emb (direct accumulation)\n";
        oss << "  EQUATION: grad_lm[v] = centered^T @ grad_logits[:,v] (dense matmul)\n";
        oss << "            grad_emb[tok] += grad_encoder[t] * emb_scale (sparse atomicAdd)\n";
        oss << "            postclip_param_grad_tied = grad_lm + grad_emb (same embedding/LM parameter buffer)\n";
    } else {
        oss << " WEIGHT_GRAD: untied grad_lm + grad_emb inspected together after component clipping\n";
        oss << "  EQUATION: grad_lm[v] = centered^T @ grad_logits[:,v] (dense matmul)\n";
        oss << "            grad_emb[tok] += grad_encoder[t] * emb_scale (sparse atomicAdd)\n";
        oss << "            postclip_param_grad_joint[v] = concat(grad_lm[v], grad_emb[v]) (host-side row aggregation across untied buffers)\n";
    }
    oss << "  GRADIENT BUFFER: rms=" << diag.grad_rms
        << " active_rows=" << diag.num_active_rows << "/" << diag.total_vocab
        << " active_ratio=" << std::setprecision(4) << diag.active_ratio << "\n";
    oss << std::setprecision(6);
    oss << "  ROW RMS: mean=" << diag.mean_row_rms
        << " max=" << diag.max_row_rms << " (tok=" << diag.max_row_token << ")"
        << " spike_ratio=" << std::setprecision(2) << diag.spike_ratio << "x\n";
    oss << std::setprecision(6);
    oss << "  TOP-5 ROWS: ";
    for (int k = 0; k < PostClipParamGradEmbLmEquationDiag::kTopK; ++k) {
        if (k > 0) oss << ", ";
        oss << "tok" << diag.top_tokens[k] << "=" << diag.top_rms[k];
    }
    oss << "\n";
    oss << "  CURRENT PAYLOAD SCATTER CONTEXT: most_frequent_real_nonpad=tok"
        << diag.current_payload_most_frequent_token
        << " (count=" << diag.current_payload_most_frequent_count << ")"
        << " real_nonpad_count=" << diag.current_payload_real_tokens_counted
        << " pad_slots_skipped=" << diag.current_payload_pad_slots_skipped << "\n";
    oss << "  GRAD WINDOW TREND: prev_emb_rms=" << diag.prev_emb_rms
        << " curr_emb_rms=" << diag.curr_emb_rms
        << " delta=" << std::showpos << diag.emb_rms_delta << std::noshowpos << "\n";

    if (diag.spike_ratio > 10.0f) {
        oss << "  [ANOMALY] SPIKE_RATIO=" << std::setprecision(1) << diag.spike_ratio
            << "x > 10x — single token row dominates post-clip parameter gradient. "
            << "Likely cause: frequent current-payload token (tok" << diag.max_row_token
            << ") with high embedding scatter contribution"
            << (diag.tied_embeddings
                ? " OR concentrated LM head gradient into the shared buffer.\n"
                : " OR concentrated LM head gradient in the untied projection weights.\n");
    }
    if (diag.emb_rms_delta > diag.prev_emb_rms * 0.5f && diag.prev_emb_rms > 0.01f) {
        oss << "  [ANOMALY] EMB_RMS_SPIKE: delta=" << diag.emb_rms_delta
            << " > 50% of prev=" << diag.prev_emb_rms
            << " — non-deterministic gradient magnitude jump.\n";
    }
    if (diag.active_ratio < 0.1f) {
        oss << "  [ANOMALY] SPARSE_GRADS: only " << std::setprecision(1)
            << (diag.active_ratio * 100.0f)
            << "% of vocab rows have non-zero post-clip grads — active rows concentrate update pressure.\n";
    }

    return oss.str();
}

} // namespace

void runPostClipParamGradEmbLmEquation(
    GRIMText::Training::TrainingContext& ctx,
    GRIMText::Training::TrainingLoopState& state,
    const GRIM::Batching::BatchPayload& payload,
    float emb_rms_pre,
    int batch_idx,
    bool sync_diag,
    cudaStream_t stream)
{
    const bool enabled = sync_diag && ctx.logging.tape &&
        ctx.logging.tape->accepts(GRIM::Logging::LogLevel::Debug);
    if (!enabled) {
        return;
    }

    const float prev_emb_rms = state.diagnostics.has_prev_emb_rms
        ? state.diagnostics.prev_emb_rms
        : emb_rms_pre;

    const PostClipParamGradEmbLmEquationDiag diag = computePostClipParamGradEmbLmEquation(
        ctx.model->getEmbeddingLayer(),
        ctx.model->getLmHeadLayer(),
        payload,
        ctx.model_config.d_model,
        static_cast<int>(ctx.data_info.actual_vocab_size),
        ctx.model_config.tie_embeddings,
        prev_emb_rms,
        emb_rms_pre,
        stream);

    const std::string line = formatPostClipParamGradEmbLmEquation(diag, batch_idx);
    ctx.logging.logger->log(line);

    EQ_LOG(
        ctx.logging.tape.get(),
        GRIM::Logging::LogGroup::Embedding,
        GRIM::Logging::LogPhase::GRADIENT_CLIP,
        0,
        kPostClipParamGradEmbLmEquationOp,
        line.c_str());

    state.diagnostics.prev_emb_rms = emb_rms_pre;
    state.diagnostics.has_prev_emb_rms = true;
}

} // namespace GRIM::Diagnostics