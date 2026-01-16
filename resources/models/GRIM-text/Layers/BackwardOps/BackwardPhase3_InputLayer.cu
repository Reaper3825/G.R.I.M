/**
 * @file BackwardPhase3_InputLayer.cu
 * @brief Phase 3: Input Layer Backward Pass Implementation
 *
 * This is the final phase of the backward pass, handling:
 * 1. ScratchBlock backward (if enabled)
 * 2. Embedding backward (scatter-add to vocab buffer) - ONLY when tie_embeddings=false
 * 3. Position embedding backward (Issue #36 fix)
 *
 * After this phase, all gradients are computed and ready for the optimizer.
 *
 * ISSUE #38 FIX (Jan 2026):
 * When tie_embeddings=true, we SKIP embedding backward entirely!
 * 
 * Reason: With weight tying, embedding_grads and lm_head_weight_grads are aliased
 * (same pointer). The LM head backward (Phase 1) computes the correct gradient using
 * CENTERED hidden states (Issue #37 fix). If we ran embedding backward, it would
 * atomicAdd NON-CENTERED gradients from encoder output, corrupting the centered
 * gradient and causing the gradient sign flip bug (mode collapse to token 277).
 * 
 * Evidence:
 *   Token277GradW (after LM head backward): mean=0.000000 ✅
 *   Token277Trace (after embedding backward): mean=0.000009 ⚠️ POLLUTED!
 *
 * EMBEDDING BACKWARD (only when tie_embeddings=false):
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
    sb_args.input = ts->cached_embeddings;
    sb_args.output = ts->cached_embeddings;  // Same buffer (in-place in forward)
    sb_args.total_tokens = ctx.total_tokens;
    sb_args.seq_len = ctx.seq_len;
    sb_args.token_ids = ts->cached_token_ids;
    sb_args.token_numeric_values = ts->cached_token_numeric_values;
    sb_args.token_numeric_mask = ts->cached_token_numeric_mask;
    sb_args.stream = ctx.training_state->stream_ctrl.getPrimaryStream();
    
    // Cached atom information from forward pass
    sb_args.cache_atom_embeddings = ts->cached_scratch_block_embeddings;
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
    // ISSUE #38 FIX: When tie_embeddings=true, SKIP embedding backward!
    //
    // Reason: With weight tying, embedding_grads == lm_head_weight_grads (aliased).
    // The LM head backward (Phase 1) already computes the correct gradient using
    // CENTERED hidden states (Issue #37 fix). If we run embedding backward here,
    // it would atomicAdd NON-CENTERED gradients to the same buffer, corrupting
    // the centered gradient and causing the gradient sign flip bug.
    //
    // Evidence from training logs:
    //   Token277GradW (LM head backward): mean=0.000000 ✅ (centered, correct)
    //   Token277Trace (after embedding backward): mean=0.000009 ⚠️ (polluted!)
    //
    // Solution: Only run embedding backward when NOT using weight tying.
    // When tie_embeddings=true, the LM head gradient IS the embedding gradient.
    //--------------------------------------------------//
    
    if (cfg->tie_embeddings) {
        // SKIP embedding backward - LM head gradient is already correct and centered!
        P3_INFO("Issue #38: Skipping embedding backward (tie_embeddings=true, using centered LM head gradient)");
    } else {
        // Run embedding backward only when weights are NOT tied
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
    }
    
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
    // Handle tie_embeddings gradient path
    //
    // ISSUE #38: With the fix above (skipping embedding backward for tied weights),
    // we no longer need to merge buffers. The gradient flow is now:
    //
    // tie_embeddings=true:
    //   - LM head backward computes grad_W using CENTERED hidden states
    //   - Embedding backward is SKIPPED (would pollute with non-centered grad)
    //   - No merge needed - lm_head_weight_grads IS the final gradient
    //
    // tie_embeddings=false:
    //   - LM head backward computes grad_W for separate lm_head_weight_grads
    //   - Embedding backward computes grad for separate embedding_grads
    //   - Both are used independently by optimizer
    //--------------------------------------------------//
    
    if (cfg->tie_embeddings) {
        // With Issue #38 fix, embedding backward was skipped.
        // The lm_head_weight_grads (aliased to embedding_grads) contains
        // the correct centered gradient from LM head backward.
        P3_INFO("Issue #38: Using centered LM head gradient for tied embeddings (no merge needed)");
    }
    
    // === ISSUE #39 DIAGNOSTIC: Check grad_W[277] at END of Phase3 ===
    // This verifies nothing else corrupted the gradient after LM head backward
    {
        static int s_p3_diag = 0;
        constexpr int kToken277 = 277;
        if (++s_p3_diag <= 10 && ts->embedding_grads()) {
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
