/**
 * @file BackwardOps_Orchestrator.cu
 * @brief Backward Pass Orchestrator Implementation
 *
 * This file coordinates the 3-phase backward pass and provides:
 * - Context initialization from LanguageModel
 * - Phase orchestration with error handling
 * - Comprehensive diagnostics and logging
 *
 * PHASE FLOW:
 *   initBackwardContext() → executeBackward()
 *        │
 *        ├─→ Phase 1: Output Layer → sets ctx.current_grad = grad_encoder_out
 *        │
 *        ├─→ Phase 2: Encoder → loops layers N-1 → 0, updates ctx.current_grad
 *        │
 *        └─→ Phase 3: Input Layer → final gradients to embedding_grads
 *
 * ERROR HANDLING:
 * - Each phase returns BackwardStatus
 * - On error: ctx.error_message and ctx.error_layer are set
 * - Gradient explosion checked at every phase boundary
 */

#include "BackwardOps_Orchestrator.hpp"
#include "../../Layers/Encoding/Encoding_GPU.hpp"
#include "../../Shared/ScratchBlock/ScratchBlock_GPU.hpp"
#include "../../Layers/Embedding/Embedding_GPU.hpp"
#include "../../Shared/LogRecorder/LogRecorder.hpp"

#include <cuda_runtime.h>
#include <chrono>
#include <sstream>
#include <iomanip>

namespace GRIM {
namespace Backward {

//======================================================//
//  Logging Setup
//======================================================//

namespace {
constexpr auto kModule = GRIM::Logging::ModuleId::BackwardPass;

#define ORCH_INFO(msg) do { std::ostringstream _oss; _oss << "[BackwardOrch] " << msg; GRIM::Logging::EmitModuleInfo(kModule, _oss.str()); } while (0)
#define ORCH_WARN(msg) do { std::ostringstream _oss; _oss << "[BackwardOrch] " << msg; GRIM::Logging::EmitModuleWarning(kModule, _oss.str()); } while (0)
#define ORCH_ERROR(msg) do { std::ostringstream _oss; _oss << "[BackwardOrch] " << msg; GRIM::Logging::EmitModuleError(kModule, _oss.str()); } while (0)

using Clock = std::chrono::high_resolution_clock;
using Duration = std::chrono::duration<double, std::milli>;

} // anonymous namespace

//======================================================//
//  Orchestrator Entry Point
//======================================================//

BackwardStatus executeBackward(BackwardContext& ctx) {
    auto start_time = Clock::now();
    
    ORCH_INFO("========== BACKWARD PASS START ==========");
    ORCH_INFO("batch=" << ctx.batch_size << " seq=" << ctx.seq_len 
              << " tokens=" << ctx.total_tokens 
              << " call_id=" << ctx.backward_call_id);
    
    //--------------------------------------------------//
    // Initial Validation
    //--------------------------------------------------//
    
    BackwardStatus validation = ctx.validate();
    if (validation != BackwardStatus::SUCCESS) {
        ORCH_ERROR("Context validation failed: " << statusToString(validation));
        ctx.error_message = "Backward context validation failed";
        return validation;
    }
    
    //--------------------------------------------------//
    // Phase 1: Output Layer Backward
    //--------------------------------------------------//
    
    auto phase1_start = Clock::now();
    
    BackwardStatus phase1_status = executePhase1_OutputLayer(ctx);
    
    auto phase1_end = Clock::now();
    Duration phase1_time = phase1_end - phase1_start;
    
    if (phase1_status != BackwardStatus::SUCCESS) {
        ORCH_ERROR("Phase 1 (Output Layer) FAILED: " << statusToString(phase1_status));
        ORCH_ERROR("  Error: " << ctx.error_message);
        return phase1_status;
    }
    
    ORCH_INFO("Phase 1 COMPLETE in " << std::fixed << std::setprecision(2) 
              << phase1_time.count() << "ms");
    
    //--------------------------------------------------//
    // Phase 2: Encoder Backward
    //--------------------------------------------------//
    
    auto phase2_start = Clock::now();
    
    BackwardStatus phase2_status = executePhase2_Encoder(ctx);
    
    auto phase2_end = Clock::now();
    Duration phase2_time = phase2_end - phase2_start;
    
    if (phase2_status != BackwardStatus::SUCCESS) {
        ORCH_ERROR("Phase 2 (Encoder) FAILED: " << statusToString(phase2_status));
        ORCH_ERROR("  Error: " << ctx.error_message);
        if (ctx.error_layer >= 0) {
            ORCH_ERROR("  Failed at layer: " << ctx.error_layer);
        }
        return phase2_status;
    }
    
    ORCH_INFO("Phase 2 COMPLETE in " << std::fixed << std::setprecision(2) 
              << phase2_time.count() << "ms (" << ctx.config->num_layers << " layers)");
    
    //--------------------------------------------------//
    // Phase 3: Input Layer Backward
    //--------------------------------------------------//
    
    auto phase3_start = Clock::now();
    
    BackwardStatus phase3_status = executePhase3_InputLayer(ctx);
    
    auto phase3_end = Clock::now();
    Duration phase3_time = phase3_end - phase3_start;
    
    if (phase3_status != BackwardStatus::SUCCESS) {
        ORCH_ERROR("Phase 3 (Input Layer) FAILED: " << statusToString(phase3_status));
        ORCH_ERROR("  Error: " << ctx.error_message);
        return phase3_status;
    }
    
    ORCH_INFO("Phase 3 COMPLETE in " << std::fixed << std::setprecision(2) 
              << phase3_time.count() << "ms");
    
    //--------------------------------------------------//
    // Summary
    //--------------------------------------------------//
    
    auto end_time = Clock::now();
    Duration total_time = end_time - start_time;
    
    ORCH_INFO("========== BACKWARD PASS COMPLETE ==========");
    ORCH_INFO("Total: " << std::fixed << std::setprecision(2) << total_time.count() << "ms");
    ORCH_INFO("  Phase 1 (Output):  " << std::setw(8) << phase1_time.count() << "ms");
    ORCH_INFO("  Phase 2 (Encoder): " << std::setw(8) << phase2_time.count() << "ms");
    ORCH_INFO("  Phase 3 (Input):   " << std::setw(8) << phase3_time.count() << "ms");
    
    return BackwardStatus::SUCCESS;
}

//======================================================//
//  Context Initialization
//======================================================//

BackwardContext initBackwardContext(
    LanguageModel& model,
    int batch_size,
    int seq_len,
    float loss,
    bool accumulate,
    float grad_scale) {
    
    // Extract components from LanguageModel
    const LanguageModelConfig* config = &model.getConfig();
    TrainingState* training_state = &model.getTrainingState();
    
    // Get GPU encoder
    GPUGrimEncoder* gpu_encoder = &model.getGpuEncoder();
    
    // Get ScratchBlock layer (may be nullptr)
    ScratchBlockLayer* scratch_block = model.getScratchBlockLayer();
    
    // Get embedding runtime
    EmbeddingRuntime* embedding_runtime = &model.getGpuEmbedder();
    
    // Get handles
    cublasHandle_t cublas_handle = training_state->cublas_handle;
    cudaStream_t stream = training_state->stream_ctrl.getPrimaryStream();
    
    return initBackwardContextRaw(
        config,
        training_state,
        gpu_encoder,
        scratch_block,
        embedding_runtime,
        cublas_handle,
        stream,
        batch_size,
        seq_len,
        accumulate,
        grad_scale,
        0);  // step = 0 for legacy wrapper
}

BackwardContext initBackwardContextRaw(
    const LanguageModelConfig* config,
    TrainingState* training_state,
    GPUGrimEncoder* gpu_encoder,
    ScratchBlockLayer* scratch_block,
    EmbeddingRuntime* embedding_runtime,
    cublasHandle_t cublas_handle,
    cudaStream_t stream,
    int batch_size,
    int seq_len,
    bool accumulate,
    float grad_scale,
    uint64_t step) {
    
    BackwardContext ctx{};
    
    //--------------------------------------------------//
    // Model Configuration
    //--------------------------------------------------//
    
    ctx.config = config;
    ctx.training_state = training_state;
    
    //--------------------------------------------------//
    // Batch Information
    //--------------------------------------------------//
    
    ctx.batch_size = batch_size;
    ctx.seq_len = seq_len;
    ctx.total_tokens = batch_size * seq_len;
    
    //--------------------------------------------------//
    // Gradient Flow State
    //--------------------------------------------------//
    
    ctx.current_grad = nullptr;  // Will be set by Phase 1
    ctx.grad_scale = (grad_scale > 0.0f) ? grad_scale : 1.0f;
    ctx.accumulate = accumulate;
    
    //--------------------------------------------------//
    // cuBLAS Constants
    //--------------------------------------------------//
    
    ctx.alpha = 1.0f;
    ctx.beta_zero = 0.0f;
    // BUG FIX: beta_accum MUST be conditional on accumulate flag!
    // First micro-batch (accumulate=false) -> beta=0.0 to OVERWRITE garbage
    // Subsequent micro-batches (accumulate=true) -> beta=1.0 to ACCUMULATE
    ctx.beta_accum = accumulate ? 1.0f : 0.0f;
    
    //--------------------------------------------------//
    // External Components
    //--------------------------------------------------//
    
    ctx.gpu_encoder = gpu_encoder;
    ctx.scratch_block = scratch_block;
    ctx.embedding_runtime = embedding_runtime;
    ctx.cublas_handle = cublas_handle;
    
    // CRITICAL: Rebind cuBLAS stream - NumericHead::forward may have changed it during forward pass
    cudaStream_t primary_stream = training_state->stream_ctrl.getPrimaryStream();
    cublasSetStream(cublas_handle, primary_stream);
    
    //--------------------------------------------------//
    // Diagnostics
    //--------------------------------------------------//
    
    // Increment call counter for deterministic diagnostics
    static uint64_t s_backward_call_counter = 0;
    ctx.backward_call_id = ++s_backward_call_counter;
    
    ctx.step = step;  // Set training step for layer logging
    ctx.enable_grad_checks = true;  // Deferred GPU stats; no per-kernel syncs.
    ctx.enable_layer_logging = false;  // Keep layer logging off unless needed.
    ctx.explosion_threshold = 1e6f; // Gradient norm threshold
    
    //--------------------------------------------------//
    // Initialize phase statuses
    //--------------------------------------------------//
    
    ctx.phase1_status = BackwardStatus::SUCCESS;
    ctx.phase2_status = BackwardStatus::SUCCESS;
    ctx.phase3_status = BackwardStatus::SUCCESS;
    ctx.error_message.clear();
    ctx.error_layer = -1;
    
    ORCH_INFO("Context initialized: batch=" << batch_size << " seq=" << seq_len
              << " accumulate=" << accumulate << " grad_scale=" << grad_scale
              << " call_id=" << ctx.backward_call_id);
    
    return ctx;
}

//======================================================//
//  Diagnostic Functions
//======================================================//

std::string getBackwardErrorReport(const BackwardContext& ctx) {
    std::ostringstream oss;
    oss << "========== BACKWARD PASS ERROR REPORT ==========\n";
    
    oss << "Call ID: " << ctx.backward_call_id << "\n";
    oss << "Batch: " << ctx.batch_size << " x " << ctx.seq_len 
        << " = " << ctx.total_tokens << " tokens\n";
    
    oss << "\nPhase Status:\n";
    oss << "  Phase 1 (Output):  " << statusToString(ctx.phase1_status) << "\n";
    oss << "  Phase 2 (Encoder): " << statusToString(ctx.phase2_status) << "\n";
    oss << "  Phase 3 (Input):   " << statusToString(ctx.phase3_status) << "\n";
    
    if (!ctx.error_message.empty()) {
        oss << "\nError Message: " << ctx.error_message << "\n";
    }
    
    if (ctx.error_layer >= 0) {
        oss << "Error Layer: " << ctx.error_layer << "\n";
    }
    
    oss << "\nConfiguration:\n";
    if (ctx.config) {
        oss << "  d_model: " << ctx.config->d_model << "\n";
        oss << "  num_layers: " << ctx.config->num_layers << "\n";
        oss << "  num_heads: " << ctx.config->num_heads << "\n";
        oss << "  d_ff: " << ctx.config->d_ff << "\n";
        oss << "  vocab_size: " << ctx.config->vocab_size << "\n";
    }
    
    oss << "================================================\n";
    return oss.str();
}

void logBackwardSummary(const BackwardContext& ctx) {
    ORCH_INFO("========== BACKWARD PASS SUMMARY ==========");
    ORCH_INFO("Call ID: " << ctx.backward_call_id);
    ORCH_INFO("Batch: " << ctx.batch_size << " x " << ctx.seq_len 
              << " = " << ctx.total_tokens << " tokens");
    
    ORCH_INFO("Phase Status:");
    ORCH_INFO("  Phase 1: " << statusToString(ctx.phase1_status));
    ORCH_INFO("  Phase 2: " << statusToString(ctx.phase2_status));
    ORCH_INFO("  Phase 3: " << statusToString(ctx.phase3_status));
    
    if (!ctx.error_message.empty()) {
        ORCH_WARN("Error: " << ctx.error_message);
    }
    
    if (ctx.error_layer >= 0) {
        ORCH_WARN("Error Layer: " << ctx.error_layer);
    }
    
    ORCH_INFO("============================================");
}

} // namespace Backward
} // namespace GRIM
