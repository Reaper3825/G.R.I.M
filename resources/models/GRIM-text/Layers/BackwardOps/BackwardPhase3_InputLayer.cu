/**
 * @file BackwardPhase3_InputLayer.cu
 * @brief Phase 3: Input Layer Backward Pass Implementation
 *
 * This is the final phase of the backward pass, handling:
 * 1. ScratchBlock backward (if enabled)
 * 2. Embedding backward (scatter-add to vocab buffer)
 * 3. Tie embeddings (merge with LM head gradients if enabled)
 *
 * After this phase, all gradients are computed and ready for the optimizer.
 *
 * EMBEDDING BACKWARD:
 * - Uses atomic scatter-add: grad_embeddings[token_id] += grad_position
 * - This accumulates gradients for tokens that appear multiple times
 * - When tie_embeddings=true, these gradients must be added to LM head weight gradients
 */

#include "BackwardPhase3_InputLayer.hpp"
#include "../../Shared/ScratchBlock/ScratchBlock_GPU.hpp"
#include "../../Layers/Embedding/Embedding_GPU.hpp"
#include "../../Shared/TensorContract/TensorContract_GPU.hpp"
#include "../../Shared/Gradients/GradStatsCollector.hpp"
#include "../../Shared/LogRecorder/LogRecorder.hpp"

#include <cuda_runtime.h>
#include <sstream>

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
    BWD_CHECK_PTR(ctx, ts->embedding_grads, "embedding_grads", -1);
    
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
    //--------------------------------------------------//
    
    launchEmbeddingBackward(
        ctx.current_grad,
        ts->cached_token_ids,
        ts->embedding_grads,
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
    
    if (ts->position_embedding_grads && cfg->max_seq_len > 0) {
        launchPositionEmbeddingBackward(
            ctx.current_grad,
            ts->position_embedding_grads,
            ctx.batch_size,
            ctx.seq_len,
            cfg->d_model,
            cfg->max_seq_len,
            ctx.training_state->stream_ctrl.getPrimaryStream());
        
        BWD_CHECK_CUDA(ctx, cudaGetLastError(), "Position embedding backward", -1);
        P3_INFO("Position embedding gradients computed (Issue #36 FIX)");
    }
    
    //--------------------------------------------------//
    // Handle tie_embeddings: merge embedding grads with LM head grads
    // ONLY if they are separate buffers! When tie_embeddings=true,
    // embedding_grads == lm_head_weight_grads (aliased) so embedding
    // backward already accumulated into the shared buffer via atomicAdd.
    //--------------------------------------------------//
    
    if (cfg->tie_embeddings && ts->lm_head_weight_grads) {
        // Check if buffers are aliased (same pointer)
        if (ts->embedding_grads == ts->lm_head_weight_grads) {
            // ALIASED: embedding backward already wrote to shared buffer.
            // No merge needed - gradients are already combined!
            P3_INFO("Tied embedding grads - buffers aliased, skipping merge (atomicAdd already accumulated)");
        } else {
            // NOT aliased (shouldn't happen with tie_embeddings=true, but be safe)
            P3_INFO("Merging embedding grads with LM head grads (separate buffers)");
            
            const size_t total_size = static_cast<size_t>(cfg->vocab_size) * cfg->d_model;
            
            launchResidualAdd(
                ts->embedding_grads,       // src1: embedding gradients
                ts->lm_head_weight_grads,  // src2: LM head gradients (from Phase 1)
                ts->lm_head_weight_grads,  // dst: combined gradients
                total_size,
                ctx.training_state->stream_ctrl.getPrimaryStream());
            
            BWD_CHECK_CUDA(ctx, cudaGetLastError(), "Tie embeddings merge", -1);
        }
    }
    
    //--------------------------------------------------//
    // Final gradient validation (deferred to batch flush)
    //--------------------------------------------------//
    
    if (ctx.enable_grad_checks && ts->embedding_grads && ts->embedding_grad_size > 0) {
        queueGradStats(
            "embedding_grads",
            -1,
            ts->embedding_grads,
            static_cast<size_t>(ts->embedding_grad_size),
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
