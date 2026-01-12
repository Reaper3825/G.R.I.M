#include "ForwardPhase3_InputLayer.hpp"
#include "ForwardDiagnostics.cuh"

#include "../../Layers/Embedding/Embedding_GPU.hpp"
#include "../../Layers/ScratchBlock/ScratchBlock_GPU.hpp"

#include <cuda_runtime.h>
#include <chrono>

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

        // === DIAGNOSTIC: Log embedding output ===
        // Expected: Xavier init means var ≈ 1/d_model, mean ≈ 0
        // Position embeddings (RoPE+ALiBi) are added inside embedding layer
        {
            const size_t emb_elements = static_cast<size_t>(ctx.total_tokens) * static_cast<size_t>(cfg->d_model);
            const float expected_var = 1.0f / cfg->d_model;  // Xavier init
            FWD_DIAG_BUFFER_EXPECTED("embeddings (after lookup + position)",
                ts->cached_embeddings, emb_elements,
                0.0f, expected_var * 2,  // Some variance from position embeddings added
                -1.0f, 1.0f,             // Should be small values
                ctx.stream);
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
