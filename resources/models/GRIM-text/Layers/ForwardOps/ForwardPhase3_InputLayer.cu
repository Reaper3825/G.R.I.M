#include "ForwardPhase3_InputLayer.hpp"
#include "ForwardDiagnostics.cuh"

#include "../../Layers/Embedding/Embedding_GPU.hpp"
#include "../../Layers/ScratchBlock/ScratchBlock_GPU.hpp"

#include <cuda_runtime.h>
#include <chrono>
#include <vector>
#include <cmath>
#include <cstdio>
#include <algorithm>

namespace GRIM {
namespace Forward {

namespace {
static bool g_order_log_enabled = false;
}  // namespace

ForwardStatus executePhase3_InputLayer(ForwardContext& ctx) {
    auto phase_start = std::chrono::high_resolution_clock::now();
    fprintf(stderr, "[PHASE_TIMING] Phase3 (Input) START\n");
    
    FWD_INFO("[ForwardPhase3] START mode=" << modeToString(ctx.mode)
                                          << " batch=" << ctx.batch_size
                                          << " seq=" << ctx.seq_len);

    ForwardStatus validation = ctx.validate();
    if (validation != ForwardStatus::SUCCESS) {
        ctx.error_message = "Phase 3 context validation failed";
        FWD_ERROR("[ForwardPhase3] Context validation failed: " << statusToString(validation));
        return validation;
    }

    FWD_CHECK_PTR(ctx, ctx.embedding_runtime, "embedding_runtime", -1);
    FWD_CHECK_PTR(ctx, ctx.training_state, "training_state", -1);

    auto* ts = ctx.training_state;
    const auto* cfg = ctx.config;
    cudaStream_t stream = ctx.stream;

    const bool wants_full_sequence = (ctx.mode == ForwardMode::TrainingFull ||
                                      ctx.mode == ForwardMode::Prefill ||
                                      ctx.mode == ForwardMode::DecodeFull);

    if (wants_full_sequence) {
        FWD_CHECK_PTR(ctx, ts->cached_token_ids, "cached_token_ids", -1);
        FWD_CHECK_PTR(ctx, ts->cached_embeddings, "cached_embeddings", -1);

        if (ctx.mode == ForwardMode::DecodeFull) {
            if (ctx.query_pos < 0) {
                FWD_FAIL_LOUD(ctx, ForwardStatus::INVALID_STATE, "query_pos < 0 for DecodeFull", -1);
            }
            cudaMemcpyAsync(ts->cached_token_ids + ctx.query_pos,
                            &ctx.new_token,
                            sizeof(int),
                            cudaMemcpyHostToDevice,
                            stream);
        } else if (!ctx.tokens_on_device) {
            if (!ctx.host_tokens) {
                FWD_FAIL_LOUD(ctx, ForwardStatus::NULL_POINTER, "host_tokens is null", -1);
            }
            cudaMemcpyAsync(ts->cached_token_ids,
                            ctx.host_tokens,
                            static_cast<size_t>(ctx.total_tokens) * sizeof(int),
                            cudaMemcpyHostToDevice,
                            stream);
        }

        auto emb_start = std::chrono::high_resolution_clock::now();
        if (!embeddingRuntimeForward(
                ctx.embedding_runtime,
                ts->cached_token_ids,
                nullptr,
                ctx.batch_size,
                ctx.seq_len,
                ts->cached_embeddings)) {
            FWD_FAIL_LOUD(ctx, ForwardStatus::INVALID_STATE, "embeddingRuntimeForward failed", -1);
        }
        auto emb_end = std::chrono::high_resolution_clock::now();
        auto emb_ms = std::chrono::duration<double, std::milli>(emb_end - emb_start).count();
        fprintf(stderr, "[VOCAB_TIMING] Phase3: Embedding lookup complete: %.2f ms\n", emb_ms);
        FWD_CHECK_CUDA(ctx, cudaGetLastError(), "embeddingRuntimeForward", -1);

        // === DIAGNOSTIC: Log h = tok_emb[token_id] + pos_emb[pos] components ===
        {
            const int d_model = cfg->d_model;
            const int seq_len = ctx.seq_len;
            
            // Get the embedding weights from runtime
            const float* d_tok_emb = ctx.embedding_runtime->weights.token_embeddings;
            const float* d_pos_emb = ctx.embedding_runtime->weights.position_embeddings;
            
            // Copy token IDs
            constexpr int kLogPositions = 5;
            const int positions_to_log = std::min(kLogPositions, ctx.total_tokens);
            std::vector<int> h_token_ids(positions_to_log);
            cudaMemcpyAsync(h_token_ids.data(), ts->cached_token_ids,
                           positions_to_log * sizeof(int),
                           cudaMemcpyDeviceToHost, ctx.stream);
            
            // Copy the final h (output)
            std::vector<float> h_final(positions_to_log * d_model);
            cudaMemcpyAsync(h_final.data(), ts->cached_embeddings, 
                           positions_to_log * d_model * sizeof(float),
                           cudaMemcpyDeviceToHost, ctx.stream);
            cudaStreamSynchronize(ctx.stream);
            
            fprintf(stderr, "[h_COMPONENTS] === h = tok_emb[token_id] + pos_emb[pos] ===\n");
            
            for (int t = 0; t < positions_to_log; ++t) {
                const int token_id = h_token_ids[t];
                const int pos = t % seq_len;  // Position within sequence
                
                // Copy tok_emb[token_id] and pos_emb[pos] to host
                std::vector<float> tok_vec(d_model), pos_vec(d_model);
                cudaMemcpyAsync(tok_vec.data(), d_tok_emb + static_cast<size_t>(token_id) * d_model,
                               d_model * sizeof(float), cudaMemcpyDeviceToHost, ctx.stream);
                cudaMemcpyAsync(pos_vec.data(), d_pos_emb + static_cast<size_t>(pos) * d_model,
                               d_model * sizeof(float), cudaMemcpyDeviceToHost, ctx.stream);
                cudaStreamSynchronize(ctx.stream);
                
                // Compute stats for each component
                double tok_sum = 0.0, pos_sum = 0.0, h_sum = 0.0;
                for (int i = 0; i < d_model; ++i) {
                    tok_sum += tok_vec[i];
                    pos_sum += pos_vec[i];
                    h_sum += h_final[t * d_model + i];
                }
                double tok_mean = tok_sum / d_model;
                double pos_mean = pos_sum / d_model;
                double h_mean = h_sum / d_model;
                
                // Compute raw_h = tok + pos (BEFORE RMSNorm)
                std::vector<float> raw_h(d_model);
                double raw_sq_sum = 0.0;
                for (int i = 0; i < d_model; ++i) {
                    raw_h[i] = tok_vec[i] + pos_vec[i];
                    raw_sq_sum += raw_h[i] * raw_h[i];
                }
                double raw_rms = std::sqrt(raw_sq_sum / d_model);
                double inv_rms = 1.0 / (raw_rms + 1e-5);  // Match GPU kernel eps=1e-5f
                
                // Compute expected normalized (what RMSNorm SHOULD produce)
                std::vector<float> expected_norm(d_model);
                double expected_sum = 0.0;
                for (int i = 0; i < d_model; ++i) {
                    expected_norm[i] = raw_h[i] * inv_rms;
                    expected_sum += expected_norm[i];
                }
                
                fprintf(stderr, "[h_COMPONENTS] pos=%d token_id=%d\n", t, token_id);
                fprintf(stderr, "[h_COMPONENTS]   tok_emb[%d]: sum=%.6f mean=%.9f\n", token_id, tok_sum, tok_mean);
                fprintf(stderr, "[h_COMPONENTS]   pos_emb[%d]: sum=%.6f mean=%.9f\n", pos, pos_sum, pos_mean);
                fprintf(stderr, "[h_COMPONENTS]   raw_h (tok+pos BEFORE RMSNorm): sum=%.6f rms=%.6f inv_rms=%.6f\n", 
                        tok_sum + pos_sum, raw_rms, inv_rms);
                fprintf(stderr, "[h_COMPONENTS]   expected_norm (raw_h * inv_rms): sum=%.6f\n", expected_sum);
                fprintf(stderr, "[h_COMPONENTS]   h_final (AFTER RMSNorm):         sum=%.6f mean=%.9f\n", h_sum, h_mean);
                fprintf(stderr, "[h_COMPONENTS]   VERIFY: expected_norm_sum=%.6f vs h_sum=%.6f diff=%.9f\n", 
                        expected_sum, h_sum, expected_sum - h_sum);
                fprintf(stderr, "[h_COMPONENTS]   tok_emb[0:3]=[%.6f,%.6f,%.6f,%.6f]\n",
                        tok_vec[0], tok_vec[1], tok_vec[2], tok_vec[3]);
                fprintf(stderr, "[h_COMPONENTS]   pos_emb[0:3]=[%.6f,%.6f,%.6f,%.6f]\n",
                        pos_vec[0], pos_vec[1], pos_vec[2], pos_vec[3]);
                fprintf(stderr, "[h_COMPONENTS]   raw_h[0:3]=[%.6f,%.6f,%.6f,%.6f]\n",
                        raw_h[0], raw_h[1], raw_h[2], raw_h[3]);
                fprintf(stderr, "[h_COMPONENTS]   expected[0:3]=[%.6f,%.6f,%.6f,%.6f]\n",
                        expected_norm[0], expected_norm[1], expected_norm[2], expected_norm[3]);
                fprintf(stderr, "[h_COMPONENTS]   h_final[0:3]=[%.6f,%.6f,%.6f,%.6f]\n",
                        h_final[t*d_model+0], h_final[t*d_model+1], h_final[t*d_model+2], h_final[t*d_model+3]);
            }
        }
        // === END DIAGNOSTIC ===

        if (g_order_log_enabled) {
            fprintf(stderr, "[ORDER] ForwardPhase3.embedding_done batch=%d seq=%d tokens=%d\n",
                    ctx.batch_size, ctx.seq_len, ctx.total_tokens);
        }

        const bool do_quant = ctx.enable_activation_quantization &&
            cfg->activation_quantization.enabled &&
            cfg->activation_quantization.apply_to_embeddings;
        if (g_order_log_enabled) {
            fprintf(stderr, "[ORDER] ForwardPhase3.quant_check batch=%d seq=%d tokens=%d enabled=%d\n",
                    ctx.batch_size, ctx.seq_len, ctx.total_tokens, do_quant ? 1 : 0);
        }
        if (do_quant) {
            const size_t elements = static_cast<size_t>(ctx.total_tokens) *
                                    static_cast<size_t>(cfg->d_model);
            if (elements > 0 && ctx.model) {
                ctx.model->applyActivationQuantization(ts->cached_embeddings, elements);
            }
            if (g_order_log_enabled) {
                fprintf(stderr, "[ORDER] ForwardPhase3.quant_done batch=%d seq=%d tokens=%d\n",
                        ctx.batch_size, ctx.seq_len, ctx.total_tokens);
            }
        }

        const bool do_scratch = ctx.enable_scratch_block &&
            ctx.scratch_block && ctx.scratch_block->isEnabled();
        if (g_order_log_enabled) {
            fprintf(stderr, "[ORDER] ForwardPhase3.scratch_check batch=%d seq=%d tokens=%d enabled=%d\n",
                    ctx.batch_size, ctx.seq_len, ctx.total_tokens, do_scratch ? 1 : 0);
        }
        if (do_scratch) {
            FWD_CHECK_PTR(ctx, ctx.token_numeric_values, "token_numeric_values", -1);
            FWD_CHECK_PTR(ctx, ctx.token_numeric_mask, "token_numeric_mask", -1);
            if (g_order_log_enabled) {
                fprintf(stderr, "[ORDER] ForwardPhase3.scratch_start batch=%d seq=%d tokens=%d\n",
                        ctx.batch_size, ctx.seq_len, ctx.total_tokens);
            }
            ScratchBlockForwardArgs sb_args{};
            sb_args.input = ts->cached_embeddings;
            sb_args.output = ts->cached_embeddings;
            sb_args.total_tokens = ctx.total_tokens;
            sb_args.seq_len = ctx.seq_len;
            sb_args.token_ids = ts->cached_token_ids;
            sb_args.token_numeric_values = ctx.token_numeric_values;
            sb_args.token_numeric_mask = ctx.token_numeric_mask;
            // GRMT v4: text features for atom injection
            sb_args.token_text_features = ctx.token_text_features;
            sb_args.token_text_mask = ctx.token_text_mask;
            sb_args.stream = stream;
            sb_args.cache_atom_embeddings = ts->cached_scratch_block_embeddings;
            sb_args.cache_atom_positions = ts->cached_scratch_block_positions;
            sb_args.cache_atom_types = ts->cached_scratch_block_types;
            sb_args.cache_num_atoms = ts->cached_scratch_block_num_atoms;

            ctx.scratch_block->forward(sb_args);
            FWD_CHECK_CUDA(ctx, cudaGetLastError(), "ScratchBlock forward", -1);
            
            // Issue #37: Track W[277] alignment after ScratchBlock
            // Tensor API: use .data field
            constexpr int kToken277 = 277;  // SPACE token
            if (ts->lm_head_weights.data && kToken277 < cfg->vocab_size) {
                const float* w277 = ts->lm_head_weights.data + static_cast<size_t>(kToken277) * cfg->d_model;
                FWD_DIAG_TOKEN277_ALIGNMENT("after_scratchblock", 
                    ts->cached_embeddings, w277, ctx.total_tokens, cfg->d_model, ctx.stream);
            }
            
            if (g_order_log_enabled) {
                fprintf(stderr, "[ORDER] ForwardPhase3.scratch_done batch=%d seq=%d tokens=%d\n",
                        ctx.batch_size, ctx.seq_len, ctx.total_tokens);
            }
        }
    } else if (ctx.mode == ForwardMode::DecodeIncremental) {
        FWD_CHECK_PTR(ctx, ts->single_token_embedding, "single_token_embedding", -1);
        FWD_CHECK_PTR(ctx, ts->single_token_hidden, "single_token_hidden", -1);

        if (!embeddingRuntimeForwardSingle(
                ctx.embedding_runtime,
                ctx.new_token,
                ctx.query_pos,
                ts->single_token_embedding)) {
            FWD_FAIL_LOUD(ctx, ForwardStatus::INVALID_STATE, "embeddingRuntimeForwardSingle failed", -1);
        }
        FWD_CHECK_CUDA(ctx, cudaGetLastError(), "embeddingRuntimeForwardSingle", -1);

        cudaMemcpyAsync(ts->single_token_hidden,
                        ts->single_token_embedding,
                        static_cast<size_t>(cfg->d_model) * sizeof(float),
                        cudaMemcpyDeviceToDevice,
                        stream);
    } else {
        FWD_FAIL_LOUD(ctx, ForwardStatus::UNSUPPORTED_MODE, "Unsupported forward mode", -1);
    }

    ctx.phase3_status = ForwardStatus::SUCCESS;
    auto phase_end = std::chrono::high_resolution_clock::now();
    auto phase_ms = std::chrono::duration<double, std::milli>(phase_end - phase_start).count();
    fprintf(stderr, "[PHASE_TIMING] Phase3 (Input) COMPLETE: %.2f ms\n", phase_ms);
    FWD_INFO("[ForwardPhase3] COMPLETE");
    return ForwardStatus::SUCCESS;
}

} // namespace Forward
} // namespace GRIM
