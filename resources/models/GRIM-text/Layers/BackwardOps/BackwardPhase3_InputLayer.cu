/**
 * @file BackwardPhase3_InputLayer.cu
 * @brief Phase 3: Input Layer Backward Pass Implementation
 *
 * This is the final phase of the backward pass, handling:
 * 1. ScratchBlock backward (if enabled)
 * 2. Embedding backward (scatter-add to vocab buffer)
 * 3. Position embedding backward (Issue #36 fix)
 *
 * After this phase, all gradients are computed and ready for the optimizer.
 *
 * ISSUE #38 REVERT (Jan 20, 2026):
 * The previous "fix" SKIPPED embedding backward when tie_embeddings=true.
 * THIS WAS WRONG and caused ~11,000x smaller gradients!
 *
 * Root cause of the confusion:
 * - LM head backward (Phase 1) computes: grad_W = h_centered.T @ grad_logits
 *   This is the OUTPUT projection gradient based on which predictions were wrong.
 * - Embedding backward computes: grad_E[token_id] += grad_encoder_output[position]
 *   This is the INPUT embedding gradient based on which input tokens contributed.
 *
 * These are DIFFERENT gradients that BOTH contribute to the tied weight update!
 * Skipping embedding backward removes ~99% of the gradient signal.
 *
 * Evidence after fix:
 *   PyTorch baseline emb_lm_tied norm: ~5.5
 *   GRIM before fix: ~0.0005 (11,000x smaller!)
 *   GRIM after fix: should match PyTorch baseline
 *
 * The "centering" concern was a red herring - embedding backward uses atomicAdd
 * which ACCUMULATES into the buffer, properly combining both gradient sources.
 *
 * EMBEDDING BACKWARD (always runs):
 * - Uses atomic scatter-add: grad_embeddings[token_id] += grad_position
 * - This accumulates gradients for tokens that appear multiple times
 */

#include "BackwardPhase3_InputLayer.hpp"
#include "../../Shared/ScratchBlock/ScratchBlock_GPU.hpp"
#include "../../Layers/Embedding/Embedding_GPU.hpp"
#include "../../Shared/TensorContract/TensorContract_GPU.hpp"
#include "../../Shared/Gradients/GradStatsCollector.hpp"
#include "../../Shared/LogRecorder/LogRecorder.hpp"

#include <cuda_runtime.h>
#include <sstream>
#include <iomanip>
#include <vector>

// Direct kernel declaration (grim_language_model_gpu.cu implementation)
extern "C" {
    void launchResidualAdd(const float* input, const float* residual,
                          float* output, int total_size, cudaStream_t stream);
}

namespace GRIM {
namespace Backward {

//======================================================//
//  Logging Setup
//======================================================//

namespace {
constexpr auto kModule = GRIM::Logging::ModuleId::BackwardPass;

#define P3_INFO(msg) do { std::ostringstream _oss; _oss << "[Phase3] " << msg; GRIM::Logging::EmitModuleInfo(kModule, _oss.str()); } while (0)
#define P3_WARN(msg) do { std::ostringstream _oss; _oss << "[Phase3] " << msg; GRIM::Logging::EmitModuleWarning(kModule, _oss.str()); } while (0)
#define P3_ERROR(msg) do { std::ostringstream _oss; _oss << "[Phase3] " << msg; GRIM::Logging::EmitModuleError(kModule, _oss.str()); } while (0)

inline void queueGradStats(const char* name,
    int layer,
    const float* grad_ptr,
    size_t size,
    float explosion_threshold,
    cudaStream_t stream) {
    if (!grad_ptr || size == 0) {
        return;
    }
    GRIM::GradStats::enqueue(name, layer, grad_ptr, size, explosion_threshold, stream);
}

} // anonymous namespace

//======================================================//
//  Phase 3 Entry Point
//======================================================//

BackwardStatus executePhase3_InputLayer(BackwardContext& ctx) {
    P3_INFO("START batch=" << ctx.batch_size << " seq=" << ctx.seq_len);
    
    //--------------------------------------------------//
    // Validation
    //--------------------------------------------------//
    
    BackwardStatus validation = ctx.validate();
    if (validation != BackwardStatus::SUCCESS) {
        ctx.error_message = "Phase 3 context validation failed";
        P3_ERROR("Context validation failed: " << statusToString(validation));
        return validation;
    }
    
    BWD_CHECK_PTR(ctx, ctx.current_grad, "current_grad (from Phase 2)", -1);
    
    //--------------------------------------------------//
    // Step 1: ScratchBlock Backward (if enabled)
    //--------------------------------------------------//
    
    if (ctx.scratch_block && ctx.scratch_block->isEnabled()) {
        BackwardStatus sb_status = computeScratchBlockBackward(ctx);
        if (sb_status != BackwardStatus::SUCCESS) {
            return sb_status;
        }
    } else {
        P3_INFO("ScratchBlock not enabled, skipping");
    }
    
    //--------------------------------------------------//
    // Step 2: Embedding Backward
    //--------------------------------------------------//
    
    BackwardStatus emb_status = computeEmbeddingBackward(ctx);
    if (emb_status != BackwardStatus::SUCCESS) {
        return emb_status;
    }
    
    ctx.phase3_status = BackwardStatus::SUCCESS;
    P3_INFO("COMPLETE - all gradients computed");
    return BackwardStatus::SUCCESS;
}

//======================================================//
//  ScratchBlock Backward
//======================================================//

BackwardStatus computeScratchBlockBackward(BackwardContext& ctx) {
    P3_INFO("Computing ScratchBlock backward...");
    
    auto* ts = ctx.training_state;
    
    BWD_CHECK_PTR(ctx, ctx.scratch_block, "scratch_block", -1);
    BWD_CHECK_PTR(ctx, ts->cached_embeddings, "cached_embeddings", -1);
    BWD_CHECK_PTR(ctx, ts->cached_token_ids, "cached_token_ids", -1);
    
    //--------------------------------------------------//
    // Prepare ScratchBlock forward args for backward
    // (backward uses same struct to access cached state)
    //--------------------------------------------------//
    
    ScratchBlockForwardArgs sb_args{};
    // TensorView: [total_tokens, d_model] BSM layout - in-place operation
    sb_args.input = TensorContract::TensorView::make_BSM(
        ts->cached_embeddings, ctx.total_tokens, ctx.config->d_model, "sb_bwd_input");
    sb_args.output = TensorContract::TensorView::make_BSM(
        ts->cached_embeddings, ctx.total_tokens, ctx.config->d_model, "sb_bwd_output");
    sb_args.total_tokens = ctx.total_tokens;
    sb_args.seq_len = ctx.seq_len;
    sb_args.token_ids = ts->cached_token_ids;
    sb_args.token_numeric_values = ts->cached_token_numeric_values;
    sb_args.token_numeric_mask = ts->cached_token_numeric_mask;
    sb_args.stream = ctx.training_state->stream_ctrl.getPrimaryStream();
    
    // Cached atom information from forward pass - TensorView for cache
    if (ts->cached_scratch_block_embeddings) {
        sb_args.cache_atom_embeddings = TensorContract::TensorView::make_BSM(
            ts->cached_scratch_block_embeddings,
            ctx.scratch_block->config().max_atoms,
            ctx.scratch_block->config().atom_embedding_dim,
            "sb_cache_embeddings");
    }
    sb_args.cache_atom_positions = ts->cached_scratch_block_positions;
    sb_args.cache_atom_types = ts->cached_scratch_block_types;
    sb_args.cache_num_atoms = ts->cached_scratch_block_num_atoms;
    
    //--------------------------------------------------// 
    // Execute ScratchBlock backward
    //--------------------------------------------------//
    
    // ScratchBlock::backward() modifies current_grad in-place
    ctx.scratch_block->backward(sb_args, ctx.current_grad, ctx.current_grad);
    
    BWD_CHECK_CUDA(ctx, cudaGetLastError(), "ScratchBlock backward", -1);
    
    //--------------------------------------------------//
    // Gradient validation
    //--------------------------------------------------//
    
    if (ctx.enable_grad_checks) {
        queueGradStats(
            "grad_after_scratchblock",
            -1,
            ctx.current_grad,
            static_cast<size_t>(ctx.total_tokens) * ctx.config->d_model,
            ctx.explosion_threshold,
            ctx.training_state->stream_ctrl.getPrimaryStream());
    }
    
    return BackwardStatus::SUCCESS;
}

//======================================================//
//  Embedding Backward
//======================================================//

BackwardStatus computeEmbeddingBackward(BackwardContext& ctx) {
    P3_INFO("Computing Embedding backward...");
    
    const auto* cfg = ctx.config;
    auto* ts = ctx.training_state;
    
    BWD_CHECK_PTR(ctx, ctx.embedding_runtime, "embedding_runtime", -1);
    BWD_CHECK_PTR(ctx, ts->cached_token_ids, "cached_token_ids", -1);
    BWD_CHECK_PTR(ctx, ts->embedding_grads(), "embedding_grads", -1);
    
    //--------------------------------------------------//
    // Gradient validation before embedding backward
    //--------------------------------------------------//
    
    if (ctx.enable_grad_checks) {
        queueGradStats(
            "grad_before_embedding",
            -1,
            ctx.current_grad,
            static_cast<size_t>(ctx.total_tokens) * cfg->d_model,
            ctx.explosion_threshold,
            ctx.training_state->stream_ctrl.getPrimaryStream());
    }
    
    //--------------------------------------------------//
    // Launch embedding backward kernel
    // Uses atomic scatter-add: embedding_grads[token_id] += grad
    //
    // ISSUE #38 REVERT (Jan 20, 2026):
    // The previous "fix" SKIPPED embedding backward for tied weights.
    // THIS WAS COMPLETELY WRONG - it removed ~99% of the gradient signal!
    //
    // In a transformer with weight tying:
    // - LM head backward: grad_W = h.T @ grad_logits (OUTPUT projection gradient)
    // - Embedding backward: grad_E[token_id] += grad (INPUT token gradient)
    //
    // Both gradients are DIFFERENT and must BOTH be computed!
    // atomicAdd naturally combines them in the shared buffer.
    //
    // Evidence of the bug:
    //   PyTorch emb_lm_tied norm: ~5.5
    //   GRIM with skip: ~0.0005 (11,000x smaller!)
    //   Weights were essentially FROZEN (changed by 0.000001 per step)
    //--------------------------------------------------//
    
    // ALWAYS run embedding backward (for both tied and untied weights)
    launchEmbeddingBackward(
        ctx.current_grad,
        ts->cached_token_ids,
        ts->embedding_grads(),
        ctx.batch_size,
        ctx.seq_len,
        cfg->d_model,
        cfg->vocab_size,
        ctx.training_state->stream_ctrl.getPrimaryStream());
    
    BWD_CHECK_CUDA(ctx, cudaGetLastError(), "Embedding backward", -1);
    
    //--------------------------------------------------//
    // Issue #36 FIX: Position embedding backward
    // PyTorch baseline uses trainable position embeddings (nn.Embedding).
    // GRIM was missing this backward pass → position embeddings were frozen!
    // This scatters gradients to position_embedding_grads[position] for each token.
    //--------------------------------------------------//
    
    if (ts->position_embedding_grads() && cfg->max_seq_len > 0) {
        launchPositionEmbeddingBackward(
            ctx.current_grad,
            ts->position_embedding_grads(),
            ctx.batch_size,
            ctx.seq_len,
            cfg->d_model,
            cfg->max_seq_len,
            ctx.training_state->stream_ctrl.getPrimaryStream());
        
        BWD_CHECK_CUDA(ctx, cudaGetLastError(), "Position embedding backward", -1);
        P3_INFO("Position embedding gradients computed (Issue #36 FIX)");
    }
    
    //--------------------------------------------------//
    // Gradient combination for tied weights
    //
    // With tie_embeddings=true, embedding_grads == lm_head_weight_grads (aliased).
    // After this point, the buffer contains the SUM of:
    //   1. LM head GEMM gradient: grad_W = h_centered.T @ grad_logits
    //   2. Embedding scatter-add gradient: grad_E[token_id] += grad[position]
    //
    // Both gradients are important and should combine naturally via atomicAdd.
    // No explicit merge is needed - they write to the same buffer.
    //--------------------------------------------------//
    
    if (cfg->tie_embeddings) {
        P3_INFO("Tied embeddings: combined LM head + embedding gradients in shared buffer");
    }
    
    // === ISSUE #39 DIAGNOSTIC: Check grad_W[277] at END of Phase3 ===
    // This verifies nothing else corrupted the gradient after LM head backward
    {
        static int s_p3_diag = 0;
        constexpr int kToken277 = 277;
        if (false && ++s_p3_diag <= 10 && ts->embedding_grads()) { // DISABLED
            cudaStreamSynchronize(ctx.training_state->stream_ctrl.getPrimaryStream());
            const size_t row_offset = static_cast<size_t>(kToken277) * cfg->d_model;
            std::vector<float> grad_row(cfg->d_model);
            cudaMemcpy(grad_row.data(), ts->embedding_grads() + row_offset,
                       cfg->d_model * sizeof(float), cudaMemcpyDeviceToHost);
            double sum = 0.0;
            for (int i = 0; i < cfg->d_model; ++i) sum += grad_row[i];
            P3_INFO("[Issue39-P3-DIAG] END of Phase3 call=" << s_p3_diag 
                    << " embedding_grads[277].sum=" << std::fixed << std::setprecision(9) << sum);
        }
    }
    
    //--------------------------------------------------//
    // Final gradient validation (deferred to batch flush)
    //--------------------------------------------------//
    
    if (ctx.enable_grad_checks && ts->embedding_grads()) {
        queueGradStats(
            "embedding_grads",
            -1,
            ts->embedding_grads(),
            ts->embedding_weights.numel(),
            ctx.explosion_threshold,
            ctx.training_state->stream_ctrl.getPrimaryStream());
    }
    
    //--------------------------------------------------//
    // All operations complete (no sync needed - stream ordering guarantees correctness)
    //--------------------------------------------------//
    
    return BackwardStatus::SUCCESS;
}

} // namespace Backward
} // namespace GRIM
