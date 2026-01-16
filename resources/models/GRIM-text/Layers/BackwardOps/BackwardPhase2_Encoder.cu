/**
 * @file BackwardPhase2_Encoder.cu
 * @brief Phase 2: Encoder Backward Pass Implementation
 *
 * This is the CRITICAL phase where gradient explosion typically occurs.
 * All operations have fail-loud checks and comprehensive logging.
 *
 * ARCHITECTURE:
 * - d_model = 768
 * - num_layers = 12 (loop from 11 → 0)
 * - num_heads = 12, num_kv_heads = 4 (GQA: 3:1 ratio)
 * - d_ff = 3072
 * - head_dim = 64
 *
 * TENSOR LAYOUTS:
 * - Sequence tensors: [total_tokens, d_model] row-major (BSM format)
 * - Attention tensors: [batch, heads, seq, head_dim] (BHSD format)
 * - Weight tensors: Row-major (cuBLAS uses col-major interpretation)
 *
 * CRITICAL NOTES:
 * 1. Flash Attention v2 backward uses BF16 BSHD inputs/outputs and workspace buffers
 * 2. GQA: grad_Q uses num_heads, grad_K/V use num_kv_heads
 * 3. Residual connections split gradient into two paths that merge
 */

#include "BackwardPhase2_Encoder.hpp"
#include "../../Common/grim_scale_buffer.hpp"
#include "../../Shared/TensorContract/TensorContract_GPU.hpp"
#include "../../Shared/Gradients/GradStatsCollector.hpp"
#include "../../Shared/LogRecorder/LogRecorder.hpp"
#include "../../Shared/HyperParameters/HyperParameters_GPU.hpp"
#include "../FlashAttention/Flash_Attention_Kernal.hpp"
#include "../LayernNorm/RMSNorm_Kernel_GPU.hpp"
#include "../Encoding/Encoding_GPU.hpp"
#include "../../Shared/Activations/GELU/GELU.hpp"
#include "../../Shared/TensorConversion/TensorConversion.hpp"
#include "../../Shared/PBM/PositionalBiasMethod.hpp"  // RoPE backward

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cmath>
#include <cstdint>
#include <sstream>

// Direct kernel declarations (grim_language_model_gpu.cu implementations)
extern "C" {
    void launchResidualAdd(const float* input, const float* residual,
                          float* output, int total_size, cudaStream_t stream);
}
#include <algorithm>

// External kernel declarations (C++ linkage - can throw exceptions)
void launchBiasSumGradient(const float* grad_output, float* grad_bias,
                          int total_tokens, int hidden_dim, cudaStream_t stream);

namespace GRIM {
namespace Backward {

static_assert(!GRIM::HyperParameters::QK_NORMALIZATION_ENABLED,
              "FlashAttention v2 backward does not support QK normalization.");
static_assert(GRIM::HyperParameters::SOFTMAX_TEMPERATURE == 1.0f,
              "FlashAttention v2 backward requires softmax_temperature=1.0f.");

//======================================================//
//  Issue #43 FIX: Activation Centering for Weight Gradients
//======================================================//
// Encoder weight gradients use cached activations (ln1_output, ffn_input, etc.)
// that have NON-ZERO MEAN. This creates systematic gradient bias:
//   grad_W[i,j] = Σ_t (activation[t,i] × grad[t,j])
//              = Σ_t ((centered[t,i] + mean_t) × grad[t,j])
//              = Σ_t (centered[t,i] × grad[t,j]) + mean × Σ_t grad[t,j]
//                                                  ^^^^^^^^^^^^^^^^^^^
//                                                  BIAS TERM (non-zero!)
// This bias causes encoder to output hidden states aligned with W[277] → collapse
//
// FIX: Center activations BEFORE using them in weight gradient GEMMs.
//======================================================//

__global__ void centerActivationsKernel(
    const float* __restrict__ input,   // [total_tokens, dim]
    float* __restrict__ output,        // [total_tokens, dim]
    int dim,
    int total_tokens
) {
    const int token_idx = blockIdx.x;
    if (token_idx >= total_tokens) return;
    
    const float* in_row = input + static_cast<size_t>(token_idx) * dim;
    float* out_row = output + static_cast<size_t>(token_idx) * dim;
    
    // Compute mean using all threads in block via reduction
    __shared__ float s_sum;
    if (threadIdx.x == 0) s_sum = 0.0f;
    __syncthreads();
    
    // Each thread sums a subset of elements
    float local_sum = 0.0f;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        local_sum += in_row[i];
    }
    
    // Warp reduction
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        local_sum += __shfl_down_sync(0xffffffff, local_sum, offset);
    }
    
    // First thread in each warp adds to shared memory
    if ((threadIdx.x & (warpSize - 1)) == 0) {
        atomicAdd(&s_sum, local_sum);
    }
    __syncthreads();
    
    const float mean = s_sum / static_cast<float>(dim);
    
    // Subtract mean from each element
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        out_row[i] = in_row[i] - mean;
    }
}

inline void centerActivations(
    const float* input,
    float* output,
    int dim,
    int total_tokens,
    cudaStream_t stream
) {
    if (total_tokens == 0 || dim == 0) return;
    
    constexpr int kBlockSize = 256;
    centerActivationsKernel<<<total_tokens, kBlockSize, 0, stream>>>(
        input, output, dim, total_tokens);
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("[Issue43-centerActivations] kernel launch failed: ") + 
                                 cudaGetErrorString(err));
    }
}

//======================================================//
//  Logging Setup
//======================================================//

namespace {
constexpr auto kModule = GRIM::Logging::ModuleId::BackwardPass;

#define P2_INFO(msg) do { std::ostringstream _oss; _oss << "[Phase2] " << msg; GRIM::Logging::EmitModuleInfo(kModule, _oss.str()); } while (0)
#define P2_WARN(msg) do { std::ostringstream _oss; _oss << "[Phase2] " << msg; GRIM::Logging::EmitModuleWarning(kModule, _oss.str()); } while (0)
#define P2_ERROR(msg) do { std::ostringstream _oss; _oss << "[Phase2] " << msg; GRIM::Logging::EmitModuleError(kModule, _oss.str()); } while (0)

// Diagnostic flag (can be disabled for production)
constexpr bool kEnableLayerDiagnostics = false;

//======================================================//
//  GPU-side Gradient Statistics (deferred, batch flush)
//======================================================//

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

inline cudaMemoryType getPointerMemoryType(const cudaPointerAttributes& attr) {
#if CUDART_VERSION >= 10000
    return attr.type;
#else
    return attr.memoryType;
#endif
}

inline const char* memoryTypeName(cudaMemoryType type) {
    switch (type) {
        case cudaMemoryTypeHost:
            return "host";
        case cudaMemoryTypeDevice:
            return "device";
        case cudaMemoryTypeManaged:
            return "managed";
        default:
            return "unknown";
    }
}

inline bool logPointerDiagnostics(const char* name, const void* ptr, size_t expected_bytes) {
    if (!ptr) {
        P2_ERROR("Pointer diagnostics: " << name << " is null");
        return false;
    }

    cudaPointerAttributes attr{};
    cudaError_t attr_err = cudaPointerGetAttributes(&attr, ptr);
    if (attr_err == cudaSuccess) {
        const cudaMemoryType mem_type = getPointerMemoryType(attr);
        P2_INFO("  " << name << " ptr=" << ptr
                      << " attrs: mem_type=" << memoryTypeName(mem_type)
                      << " device=" << attr.device
                      << " expected_bytes=" << expected_bytes);
    } else {
        P2_WARN("  " << name << " attrs: cudaPointerGetAttributes failed: "
                      << cudaGetErrorString(attr_err));
        cudaGetLastError();
    }
    return true;
}

} // anonymous namespace

//======================================================//
//  Phase 2 Entry Point
//======================================================//

BackwardStatus executePhase2_Encoder(BackwardContext& ctx) {
    P2_INFO("START num_layers=" << ctx.config->num_layers 
            << " batch=" << ctx.batch_size << " seq=" << ctx.seq_len);
    
    //--------------------------------------------------//
    // Validation
    //--------------------------------------------------//
    
    BackwardStatus validation = ctx.validate();
    if (validation != BackwardStatus::SUCCESS) {
        ctx.error_message = "Phase 2 context validation failed";
        P2_ERROR("Context validation failed: " << statusToString(validation));
        return validation;
    }
    
    BWD_CHECK_PTR(ctx, ctx.current_grad, "current_grad (from Phase 1)", -1);
    BWD_CHECK_PTR(ctx, ctx.gpu_encoder, "gpu_encoder", -1);
    
    const auto* cfg = ctx.config;
    
    // DIAGNOSTIC: Queue current_grad stats at START of Phase 2 (no sync).
    if (kEnableLayerDiagnostics || ctx.enable_grad_checks) {
        queueGradStats(
            "current_grad_start",
            -1,
            ctx.current_grad,
            static_cast<size_t>(ctx.total_tokens) * cfg->d_model,
            ctx.explosion_threshold,
            ctx.training_state->stream_ctrl.getPrimaryStream());
    }
    
    // Validate encoder has correct number of layers
    if (ctx.gpu_encoder->getNumLayers() != cfg->num_layers) {
        std::ostringstream oss;
        oss << "GPU encoder layer count mismatch: got " << ctx.gpu_encoder->getNumLayers()
            << " expected " << cfg->num_layers;
        BWD_FAIL_LOUD(ctx, BackwardStatus::INVALID_STATE, oss.str(), -1);
    }
    
    //--------------------------------------------------//
    // Main Loop: Layer N-1 → 0
    //--------------------------------------------------//
    
    for (int layer = cfg->num_layers - 1; layer >= 0; --layer) {
        P2_INFO("--- Layer " << layer << " ---");
        
        // DIAGNOSTIC: Queue current_grad stats BEFORE this layer's backward (no sync).
        if (kEnableLayerDiagnostics || ctx.enable_grad_checks) {
            queueGradStats(
                "current_grad_pre_layer",
                layer,
                ctx.current_grad,
                static_cast<size_t>(ctx.total_tokens) * cfg->d_model,
                ctx.explosion_threshold,
                ctx.training_state->stream_ctrl.getPrimaryStream());
        }
        
        BackwardStatus layer_status = executeLayerBackward(ctx, layer);
        
        if (layer_status != BackwardStatus::SUCCESS) {
            ctx.error_layer = layer;
            P2_ERROR("Layer " << layer << " failed: " << statusToString(layer_status));
            return layer_status;
        }
        
        // DIAGNOSTIC: Queue current_grad stats AFTER this layer's backward (no sync).
        if (kEnableLayerDiagnostics || ctx.enable_grad_checks) {
            queueGradStats(
                "current_grad_post_layer",
                layer,
                ctx.current_grad,
                static_cast<size_t>(ctx.total_tokens) * cfg->d_model,
                ctx.explosion_threshold,
                ctx.training_state->stream_ctrl.getPrimaryStream());
        }
    }
    
    ctx.phase2_status = BackwardStatus::SUCCESS;
    P2_INFO("COMPLETE - gradient ready for Phase 3");
    return BackwardStatus::SUCCESS;
}

//======================================================//
//  Per-Layer Backward
//======================================================//

BackwardStatus executeLayerBackward(BackwardContext& ctx, int layer) {
    const auto* cfg = ctx.config;
    auto* ts = ctx.training_state;
    
    //--------------------------------------------------//
    // Get Layer Components
    //--------------------------------------------------//
    
    auto* enc = ctx.gpu_encoder->getLayer(layer);
    BWD_CHECK_PTR(ctx, enc, "encoder_layer", layer);
    
    // Get cached activations for this layer
    float* ln2_output = ts->cached_ln2_outputs[layer];
    float* ffn_pre_gelu = ts->cached_ffn_pre_gelu[layer];
    float* ffn_output = ts->cached_ffn_outputs[layer];
    float* ln1_output = ts->cached_ln1_outputs[layer];
    float* attn_output = ts->cached_attn_outputs[layer];
    float* layer_input = (layer > 0) ? ts->cached_layer_outputs[layer - 1] 
                                     : ts->cached_embeddings;
    
    // Validate cached activations
    BWD_CHECK_PTR(ctx, ln2_output, "cached_ln2_output", layer);
    BWD_CHECK_PTR(ctx, ffn_pre_gelu, "cached_ffn_pre_gelu", layer);
    BWD_CHECK_PTR(ctx, ffn_output, "cached_ffn_output", layer);
    BWD_CHECK_PTR(ctx, ln1_output, "cached_ln1_output", layer);
    BWD_CHECK_PTR(ctx, attn_output, "cached_attn_output", layer);
    BWD_CHECK_PTR(ctx, layer_input, "layer_input", layer);
    
    // Get temp buffers for intermediate gradients
    float* grad_ffn_input = ts->grad_ffn_input;
    float* grad_ffn_hidden = ts->grad_ffn_hidden;
    float* grad_attn_input = ts->grad_attn_input;
    
    BWD_CHECK_PTR(ctx, grad_ffn_input, "grad_ffn_input", layer);
    BWD_CHECK_PTR(ctx, grad_ffn_hidden, "grad_ffn_hidden", layer);
    BWD_CHECK_PTR(ctx, grad_attn_input, "grad_attn_input", layer);
    
    const int total_tokens = ctx.total_tokens;
    
    //--------------------------------------------------//
    // Step 1: Copy current_grad for FFN path (residual split)
    //--------------------------------------------------//
    
    BWD_CHECK_CUDA(ctx, cudaMemcpyAsync(
        grad_ffn_input, ctx.current_grad,
        static_cast<size_t>(total_tokens) * cfg->d_model * sizeof(float),
        cudaMemcpyDeviceToDevice, ctx.training_state->stream_ctrl.getPrimaryStream()),
        "Copy current_grad to grad_ffn_input", layer);
    
    //--------------------------------------------------//
    // Step 2: Reconstruct residual1 = attn_output + layer_input
    //--------------------------------------------------//
    
    float* residual1 = ts->cached_residual1_outputs[layer];
    BWD_CHECK_PTR(ctx, residual1, "cached_residual1", layer);
    
    launchResidualAdd(
        attn_output, layer_input, residual1,
        static_cast<size_t>(total_tokens) * cfg->d_model,
        ctx.training_state->stream_ctrl.getPrimaryStream());
    
    //--------------------------------------------------//
    // Step 3: FFN Backward (MUST come before RMSNorm2 backward!)
    // Forward: residual1 -> RMSNorm2 -> FFN -> output
    // Backward: grad_output -> FFN_bwd -> RMSNorm2_bwd -> grad_residual1
    //--------------------------------------------------//
    
    BackwardStatus ffn_status = computeFFNBackward(
        ctx, layer,
        grad_ffn_input,   // gradient input (d_model) - this is grad w.r.t. layer output
        ln2_output,       // cached FFN input (ln2 output = RMSNorm2 output)
        ffn_output);      // cached GELU output (ffn_hidden)
    
    if (ffn_status != BackwardStatus::SUCCESS) {
        return ffn_status;
    }
    
    // Log FFN layer stats (deferred to batch flush)
    // DISABLED: if (ctx.enable_layer_logging && false) {
    //     queueGradStats(
    //         "ffn_bwd_post_ffn",
    //         layer,
    //         grad_ffn_input,
    //         static_cast<size_t>(total_tokens) * cfg->d_model,
    //         ctx.explosion_threshold,
    //         ctx.training_state->stream_ctrl.getPrimaryStream());
    // }
    
    //--------------------------------------------------//
    // Step 4: RMSNorm2 Backward (receives gradient from FFN backward)
    //--------------------------------------------------//
    
    BackwardStatus rms2_status = computeRMSNormBackward( 
        ctx, layer,
        grad_ffn_input,           // grad_output from FFN backward
        residual1,                // cached input (residual1)
        enc->getRMS2Gamma(),      // gamma weights
        enc->getRMS2GammaGrad(),  // grad_gamma output
        grad_ffn_input,           // grad_input (in-place)
        static_cast<size_t>(total_tokens) * cfg->d_model,
        2);  // norm_type = 2 for RMSNorm2
    
    if (rms2_status != BackwardStatus::SUCCESS) {
        return rms2_status;
    }
    
    // CRITICAL DEBUG: Synchronize stream after RMSNorm2 to isolate crash location
    // If crash happens before this log, it's in RMSNorm2 kernel
    // If crash happens after, it's in ResidualAdd or cudaMemcpyAsync
    cudaStreamSynchronize(ctx.training_state->stream_ctrl.getPrimaryStream());
    P2_INFO("RMSNorm2 layer" << layer << " kernel completed successfully");
    
    // Log RMSNorm2 layer stats (deferred to batch flush)
    // DISABLED: if (ctx.enable_layer_logging && false) {
    //     queueGradStats(
    //         "rms2_bwd_post_norm",
    //         layer,
    //         grad_ffn_input,
    //         static_cast<size_t>(total_tokens) * cfg->d_model,
    //         ctx.explosion_threshold,
    //         ctx.training_state->stream_ctrl.getPrimaryStream());
    // }
    
    //--------------------------------------------------//
    // Step 5: Residual Add (FFN path + residual path)
    //--------------------------------------------------//
    
    // grad_ffn_input now contains gradient from RMSNorm2 backward (w.r.t. residual1)
    // current_grad contains gradient from residual path (skip connection)
    // Combine them into grad_attn_input (this is gradient w.r.t. residual1)
    P2_INFO("Before ResidualAdd layer" << layer << " grad_ffn_input=" << (void*)grad_ffn_input 
            << " current_grad=" << (void*)ctx.current_grad << " grad_attn_input=" << (void*)grad_attn_input
            << " size=" << (static_cast<size_t>(total_tokens) * cfg->d_model));
    
    launchResidualAdd(
        grad_ffn_input, ctx.current_grad, grad_attn_input,
        static_cast<size_t>(total_tokens) * cfg->d_model,
        ctx.training_state->stream_ctrl.getPrimaryStream());
    
    P2_INFO("After ResidualAdd layer" << layer);
    
    // Update current_grad to gradient w.r.t. residual1
    P2_INFO("Before cudaMemcpyAsync layer" << layer);
    BWD_CHECK_CUDA(ctx, cudaMemcpyAsync(
        ctx.current_grad, grad_attn_input,
        static_cast<size_t>(total_tokens) * cfg->d_model * sizeof(float),
        cudaMemcpyDeviceToDevice, ctx.training_state->stream_ctrl.getPrimaryStream()),
        "Copy grad_attn_input to current_grad", layer);
    
    P2_INFO("After cudaMemcpyAsync layer" << layer);
    
    //--------------------------------------------------//
    // Step 6: Attention Backward (MUST come before RMSNorm1 backward!)
    // Forward: layer_input -> RMSNorm1 -> Attention -> + layer_input -> residual1
    // Backward: grad_residual1 -> Attention_bwd -> RMSNorm1_bwd -> grad_layer_input
    //--------------------------------------------------//
    
    BackwardStatus attn_status = computeAttentionBackward(
        ctx, layer,
        grad_attn_input,  // gradient input (this is grad w.r.t. attention output)
        ln1_output);      // cached RMSNorm1 output
    
    if (attn_status != BackwardStatus::SUCCESS) {
        return attn_status;
    }
    
    // Log Attention layer stats (deferred to batch flush)
    // DISABLED: if (ctx.enable_layer_logging && false) {
    //     float* grad_qkv_input = ts->grad_qkv_input;
    //     if (grad_qkv_input) {
    //         queueGradStats(
    //             "attn_bwd_post_attn",
    //             layer,
    //             grad_qkv_input,
    //             static_cast<size_t>(total_tokens) * cfg->d_model,
    //             ctx.explosion_threshold,
    //             ctx.training_state->stream_ctrl.getPrimaryStream());
    //     }
    // }
    
    //--------------------------------------------------//
    // Step 7: RMSNorm1 Backward (receives gradient from Attention backward)
    //--------------------------------------------------//
    
    // grad_qkv_input contains gradient from attention (w.r.t. RMSNorm1 output)
    float* grad_qkv_input = ts->grad_qkv_input;
    BWD_CHECK_PTR(ctx, grad_qkv_input, "grad_qkv_input", layer);
    
    BackwardStatus rms1_status = computeRMSNormBackward(
        ctx, layer,
        grad_qkv_input,           // grad_output from Attention backward
        layer_input,              // cached input
        enc->getRMS1Gamma(),      // gamma weights
        enc->getRMS1GammaGrad(),  // grad_gamma output
        grad_qkv_input,           // grad_input (in-place)
        static_cast<size_t>(total_tokens) * cfg->d_model,
        1);  // norm_type = 1 for RMSNorm1
    
    if (rms1_status != BackwardStatus::SUCCESS) {
        return rms1_status;
    }
    
    // Log RMSNorm1 layer stats (deferred to batch flush)
    // DISABLED: if (ctx.enable_layer_logging && false) {
    //     queueGradStats(
    //         "rms1_bwd_pre_attn",
    //         layer,
    //         grad_qkv_input,
    //         static_cast<size_t>(total_tokens) * cfg->d_model,
    //         ctx.explosion_threshold,
    //         ctx.training_state->stream_ctrl.getPrimaryStream());
    // }
    
    //--------------------------------------------------//
    // Step 8: Final Residual Add (Attention path + residual path)
    //--------------------------------------------------//
    
    // grad_qkv_input now contains gradient from RMSNorm1 backward (w.r.t. layer_input)
    // current_grad contains gradient from residual path
    launchResidualAdd(
        grad_qkv_input, ctx.current_grad, ctx.current_grad,
        static_cast<size_t>(total_tokens) * cfg->d_model,
        ctx.training_state->stream_ctrl.getPrimaryStream());
    
    return BackwardStatus::SUCCESS;
}

//======================================================//
//  RMSNorm Backward
//======================================================//

BackwardStatus computeRMSNormBackward(
    BackwardContext& ctx,
    int layer,
    const float* grad_output,
    const float* cached_input,
    const float* gamma,
    float* grad_gamma,
    float* grad_input,
    size_t size,
    int norm_type) {
    
    const auto* cfg = ctx.config;
    const int total_tokens = ctx.total_tokens;
    constexpr float eps = GRIM::HyperParameters::EPSILON_RMSNORM;
    
    BWD_CHECK_PTR(ctx, grad_output, "rms_grad_output", layer);
    BWD_CHECK_PTR(ctx, cached_input, "rms_cached_input", layer);
    BWD_CHECK_PTR(ctx, gamma, "rms_gamma", layer);
    
    // DEBUG: Log grad_gamma pointer status (no sync needed - just pointer logging)
    P2_INFO("RMSNorm" << norm_type << " layer" << layer << " grad_gamma=" 
            << (grad_gamma ? "VALID" : "NULL") << " ptr=" << (void*)grad_gamma
            << " total_tokens=" << total_tokens << " d_model=" << cfg->d_model);
    
    // DEBUG: Queue incoming gradient stats ONLY (skip forward pass caches - they have expected zeros at padding)
    // REMOVED: rms1_cached_input/rms2_cached_input - forward pass cache, NOT a gradient
    // REMOVED: rms1_gamma/rms2_gamma - forward pass weights, NOT a gradient
    if (ctx.enable_grad_checks) {
        queueGradStats(
            norm_type == 1 ? "rms1_grad_output" : "rms2_grad_output",
            layer,
            grad_output,
            size,
            ctx.explosion_threshold,
            ctx.training_state->stream_ctrl.getPrimaryStream());
    }
    
    // Launch RMSNorm backward kernel
    launchRMSNormBackward(
        cached_input,
        grad_output,
        gamma,
        grad_input,
        grad_gamma,  // May be nullptr if not tracking gamma gradients
        total_tokens, cfg->d_model, eps,
        ctx.training_state->stream_ctrl.getPrimaryStream());
    
    // CUDA error check (non-blocking)
    BWD_CHECK_CUDA(ctx, cudaPeekAtLastError(), 
                   (std::string("RMSNorm") + std::to_string(norm_type) + " backward").c_str(), 
                   layer);
    
    // Gradient validation (deferred to batch flush)
    if (ctx.enable_grad_checks) {
        queueGradStats(
            norm_type == 1 ? "grad_after_rms1" : "grad_after_rms2",
            layer,
            grad_input,
            size,
            ctx.explosion_threshold,
            ctx.training_state->stream_ctrl.getPrimaryStream());
        if (grad_gamma) {
            queueGradStats(
                norm_type == 1 ? "rms1_gamma_grad" : "rms2_gamma_grad",
                layer,
                grad_gamma,
                static_cast<size_t>(cfg->d_model),
                ctx.explosion_threshold,
                ctx.training_state->stream_ctrl.getPrimaryStream());
        }
    }
    
    return BackwardStatus::SUCCESS;
}

//======================================================//
//  FFN Backward
//======================================================//

BackwardStatus computeFFNBackward(
    BackwardContext& ctx,
    int layer,
    const float* grad_ffn_output,
    const float* cached_ffn_input,
    const float* cached_ffn_hidden) {
    
    const auto* cfg = ctx.config;
    auto* ts = ctx.training_state;
    const int total_tokens = ctx.total_tokens;
    
    auto* enc = ctx.gpu_encoder->getLayer(layer);
    BWD_CHECK_PTR(ctx, enc, "encoder_layer_ffn", layer);
    
    float* grad_ffn_hidden = ts->grad_ffn_hidden;
    float* grad_ffn_input_out = ts->grad_ffn_input;
    
    // Get FFN weight pointers
    float* W1 = enc->getFFNW1();
    float* W2 = enc->getFFNW2();
    BWD_CHECK_PTR(ctx, W1, "FFN_W1", layer);
    BWD_CHECK_PTR(ctx, W2, "FFN_W2", layer);
    
    const float alpha = ctx.alpha;
    const float beta_zero = ctx.beta_zero;
    const float beta_accum = ctx.beta_accum;
    
    //--------------------------------------------------//
    // Issue #43 FIX: Get centering scratch buffer
    //--------------------------------------------------//
    float* centered_scratch = ts->centered_activation_scratch;
    BWD_CHECK_PTR(ctx, centered_scratch, "centered_activation_scratch", layer);
    
    //--------------------------------------------------//
    // Step 4.1: Compute grad_W2 = ffn_hidden^T @ grad_ffn_output
    //--------------------------------------------------//
    
    // Issue #43 FIX: Center cached_ffn_hidden before weight gradient GEMM
    // This eliminates systematic bias from non-zero mean activations
    centerActivations(
        cached_ffn_hidden,
        centered_scratch,  // Output: centered activations
        cfg->d_ff,
        total_tokens,
        ctx.training_state->stream_ctrl.getPrimaryStream());
    
    // ffn_hidden (post_gelu): [total_tokens, d_ff] row-major
    // grad_ffn_output: [total_tokens, d_model] row-major
    // W2: stored as [d_model, d_ff] row-major
    // grad_W2: must match W2 layout [d_model, d_ff] row-major
    //
    // Math: grad_W2 = post_gelu^T @ grad_output = [d_ff, tokens] @ [tokens, d_model] = [d_ff, d_model]
    // But W2 is [d_model, d_ff], so we need grad_W2 = grad_output^T @ post_gelu
    // cuBLAS col-major: C[M,N] = A[M,K] @ B^T[K,N]
    // We want C = [d_model, d_ff] col-major = [d_ff, d_model] row-major... NO!
    // 
    // CORRECT: Match Feed_Forward_GPU.cu exactly
    // M=d_ff, N=d_model, K=tokens
    // C[d_ff, d_model] col-major = [d_model, d_ff] row-major (matches W2!)
    
    BWD_CHECK_CUBLAS(ctx, cublasSgemm(
        ctx.cublas_handle,
        CUBLAS_OP_N, CUBLAS_OP_T,
        cfg->d_ff, cfg->d_model, total_tokens,  // FIXED: swapped d_model <-> d_ff
        &alpha,
        centered_scratch, cfg->d_ff,            // Issue #43 FIX: use CENTERED activation
        grad_ffn_output, cfg->d_model,          // FIXED: B = grad_output [d_model, tokens] col-major
        &beta_accum,
        ts->ffn_w2_grads[layer], cfg->d_ff),    // FIXED: ldc = d_ff
        "grad_W2 computation", layer);
    
    //--------------------------------------------------//
    // Step 4.2: Compute grad_b2 (if using bias)
    //--------------------------------------------------//
    
    if (ts->ffn_b2_grads[layer]) {
        launchBiasSumGradient(
            const_cast<float*>(grad_ffn_output),
            ts->ffn_b2_grads[layer],
            total_tokens, cfg->d_model,
            ctx.training_state->stream_ctrl.getPrimaryStream());
    }
    
    //--------------------------------------------------//
    // Step 4.3: Propagate gradient through W2
    // grad_ffn_hidden = W2^T @ grad_ffn_output
    //--------------------------------------------------//
    
    // W2: [d_ff, d_model] → W2^T: [d_model, d_ff]
    BWD_CHECK_CUBLAS(ctx, cublasSgemm(
        ctx.cublas_handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        cfg->d_ff, total_tokens, cfg->d_model,
        &alpha,
        W2, cfg->d_ff,
        grad_ffn_output, cfg->d_model,
        &beta_zero,
        grad_ffn_hidden, cfg->d_ff),
        "grad through W2", layer);
    
    if (ctx.enable_grad_checks) {
        queueGradStats(
            "grad_ffn_hidden_after_W2",
            layer,
            grad_ffn_hidden,
            static_cast<size_t>(total_tokens) * cfg->d_ff,
            ctx.explosion_threshold,
            ctx.training_state->stream_ctrl.getPrimaryStream());
    }
    
    //--------------------------------------------------//
    // Step 4.4: GELU Backward
    //--------------------------------------------------//
    
    // cached_ffn_pre_gelu (from ts) is the input to GELU
    float* ffn_pre_gelu = ts->cached_ffn_pre_gelu[layer];
    BWD_CHECK_PTR(ctx, ffn_pre_gelu, "ffn_pre_gelu", layer);
    
    GELUBackwardArgs gelu_args{};
    gelu_args.input = ffn_pre_gelu;
    gelu_args.grad_output = grad_ffn_hidden;
    gelu_args.grad_input = grad_ffn_hidden;  // In-place
    gelu_args.elements = static_cast<size_t>(total_tokens) * cfg->d_ff;
    gelu_args.stream = ctx.training_state->stream_ctrl.getPrimaryStream();
    
    launchGeluBackward(gelu_args);
    
    BWD_CHECK_CUDA(ctx, cudaPeekAtLastError(), "GELU backward", layer);
    
    if (ctx.enable_grad_checks) {
        queueGradStats(
            "grad_ffn_hidden_after_GELU",
            layer,
            grad_ffn_hidden,
            static_cast<size_t>(total_tokens) * cfg->d_ff,
            ctx.explosion_threshold,
            ctx.training_state->stream_ctrl.getPrimaryStream());
    }
    
    //--------------------------------------------------//
    // Step 4.5: Compute grad_W1 = ffn_input^T @ grad_ffn_hidden
    //--------------------------------------------------//
    
    // Issue #43 FIX: Center cached_ffn_input before weight gradient GEMM
    // Reuse centered_scratch - it's large enough for d_model dimensions too
    centerActivations(
        cached_ffn_input,
        centered_scratch,  // Output: centered activations (reusing buffer)
        cfg->d_model,
        total_tokens,
        ctx.training_state->stream_ctrl.getPrimaryStream());
    
    // ln2_output (ffn_input): [total_tokens, d_model] row-major
    // grad_ffn_hidden: [total_tokens, d_ff] row-major
    // W1: stored as [d_ff, d_model] row-major
    // grad_W1: must match W1 layout [d_ff, d_model] row-major
    //
    // CORRECT: Match Feed_Forward_GPU.cu exactly
    // M=d_model, N=d_ff, K=tokens
    // C[d_model, d_ff] col-major = [d_ff, d_model] row-major (matches W1!)
    
    BWD_CHECK_CUBLAS(ctx, cublasSgemm(
        ctx.cublas_handle,
        CUBLAS_OP_N, CUBLAS_OP_T,
        cfg->d_model, cfg->d_ff, total_tokens,  // FIXED: swapped d_ff <-> d_model
        &alpha,
        centered_scratch, cfg->d_model,         // Issue #43 FIX: use CENTERED activation
        grad_ffn_hidden, cfg->d_ff,             // FIXED: B = grad_hidden [d_ff, tokens] col-major
        &beta_accum,
        ts->ffn_w1_grads[layer], cfg->d_model), // FIXED: ldc = d_model
        "grad_W1 computation", layer);
    
    //--------------------------------------------------//
    // Step 4.6: Compute grad_b1 (if using bias)
    //--------------------------------------------------//
    
    if (ts->ffn_b1_grads[layer]) {
        launchBiasSumGradient(
            grad_ffn_hidden,
            ts->ffn_b1_grads[layer],
            total_tokens, cfg->d_ff,
            ctx.training_state->stream_ctrl.getPrimaryStream());
    }
    
    //--------------------------------------------------//
    // Step 4.7: Propagate gradient through W1
    // grad_ffn_input = W1^T @ grad_ffn_hidden
    //--------------------------------------------------//
    
    BWD_CHECK_CUBLAS(ctx, cublasSgemm(
        ctx.cublas_handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        cfg->d_model, total_tokens, cfg->d_ff,
        &alpha,
        W1, cfg->d_model,
        grad_ffn_hidden, cfg->d_ff,
        &beta_zero,
        grad_ffn_input_out, cfg->d_model),
        "grad through W1", layer);
    
    if (ctx.enable_grad_checks) {
        queueGradStats(
            "grad_ffn_input_after_W1",
            layer,
            grad_ffn_input_out,
            static_cast<size_t>(total_tokens) * cfg->d_model,
            ctx.explosion_threshold,
            ctx.training_state->stream_ctrl.getPrimaryStream());
    }
    
    return BackwardStatus::SUCCESS;
}

//======================================================//
//  Attention Backward
//======================================================//

BackwardStatus computeAttentionBackward(
    BackwardContext& ctx,
    int layer,
    const float* grad_attn_output,
    const float* cached_ln1_output) {
    
    P2_INFO("computeAttentionBackward START layer" << layer);
    
    const auto* cfg = ctx.config;
    auto* ts = ctx.training_state;
    const int total_tokens = ctx.total_tokens;
    const int batch_size = ctx.batch_size;
    const int seq_len = ctx.seq_len;
    
    P2_INFO("Attention dims: batch=" << batch_size << " seq=" << seq_len << " tokens=" << total_tokens);
    
    auto* enc = ctx.gpu_encoder->getLayer(layer);
    auto* enc_attn = enc;  // Same object in our architecture
    BWD_CHECK_PTR(ctx, enc_attn, "encoder_layer_attn", layer);
    
    P2_INFO("Got encoder layer " << layer << " ptr=" << (void*)enc);
    
    // GQA dimensions
    const int num_heads = cfg->num_heads;
    const int num_kv_heads = ts->num_kv_heads;
    const int head_dim = cfg->head_dim;  // Use pre-computed value from config
    
    P2_INFO("GQA config: num_heads=" << num_heads << " num_kv_heads=" << num_kv_heads << " head_dim=" << head_dim);
    
    // Validate GQA configuration using TensorContract
    TensorContract::GQADims gqa_dims{num_heads, num_kv_heads, head_dim};
    if (!gqa_dims.is_valid()) {
        BWD_FAIL_LOUD(ctx, BackwardStatus::INVALID_STATE, 
                      "TensorContract GQA validation failed", layer);
    }
    if (num_heads % num_kv_heads != 0) {
        std::ostringstream oss;
        oss << "Invalid GQA config: num_heads=" << num_heads 
            << " not divisible by num_kv_heads=" << num_kv_heads;
        BWD_FAIL_LOUD(ctx, BackwardStatus::INVALID_STATE, oss.str(), layer);
    }
    
    const int total_qkv_dim = gqa_dims.total_qkv_dim();
    
    const size_t q_size = static_cast<size_t>(batch_size) * num_heads * seq_len * head_dim;
    const size_t kv_size = static_cast<size_t>(batch_size) * num_kv_heads * seq_len * head_dim;
    
    const float alpha = ctx.alpha;
    const float beta_zero = ctx.beta_zero;
    const float beta_accum = ctx.beta_accum;
    
    // Get cached attention output for weight gradients
    float* cached_attn_bhsd = ts->cached_attn_bhsd[layer];
    BWD_CHECK_PTR(ctx, cached_attn_bhsd, "cached_attn_bhsd", layer);
    
    P2_INFO("Cached attention buffers: cached_attn_bhsd=" << (void*)cached_attn_bhsd);
    
    //--------------------------------------------------//
    // Step 7.1: W_o Backward
    // grad_attn_out_reshaped = W_o^T @ grad_attn_output
    //--------------------------------------------------//
    
    P2_INFO("Starting W_o backward");
    
    float* W_o = enc_attn->getAttnWo();
    BWD_CHECK_PTR(ctx, W_o, "W_o", layer);
    
    // Get temp buffer for attention output gradient
    float* grad_attn_out_flat = ts->grad_attn_out_before_proj;
    BWD_CHECK_PTR(ctx, grad_attn_out_flat, "grad_attn_out_before_proj", layer);
    
    P2_INFO("Temp buffers: grad_attn_out_flat=" << (void*)grad_attn_out_flat << " W_o=" << (void*)W_o);
    
    // Reshape cached attention output (BHSD) into a flat [tokens, d_model] view for W_o gradients.
    // Store in grad_attn_out_flat temporarily for weight gradient computation.
    {
        P2_INFO("Before TensorContract convert BHSD->BSM");
        auto attn_bhsd_view = TensorContract::TensorView::make_BHSD(
            cached_attn_bhsd, batch_size, num_heads, seq_len, head_dim, "cached_attn_bhsd");
        auto attn_flat_view = TensorContract::TensorView::make_BSM(
            grad_attn_out_flat, total_tokens, cfg->d_model, "attn_out_flat");
        P2_INFO("Calling TensorContract::convert");
        TensorContract::convert(attn_bhsd_view, attn_flat_view,
                                ctx.training_state->stream_ctrl.getPrimaryStream());
        P2_INFO("After TensorContract convert");
        BWD_CHECK_CUDA(ctx, cudaPeekAtLastError(), "Reshape attn_out for W_o grad", layer);
        P2_INFO("After BWD_CHECK_CUDA");
    }
    
    //--------------------------------------------------//
    // Issue #43 FIX: Center cached attention output before W_o weight gradient
    //--------------------------------------------------//
    float* centered_scratch = ts->centered_activation_scratch;
    BWD_CHECK_PTR(ctx, centered_scratch, "centered_activation_scratch", layer);
    
    centerActivations(
        grad_attn_out_flat,     // Input: cached attention output (flattened)
        centered_scratch,        // Output: centered activation
        cfg->d_model,
        total_tokens,
        ctx.training_state->stream_ctrl.getPrimaryStream());
    
    P2_INFO("Before W_o weight gradient GEMM");
    
    // W_o: [d_model, d_model] row-major
    // grad_attn_output: [total_tokens, d_model]
    // grad_attn_out_flat: [total_tokens, d_model] (currently holds cached activation)
    
    // CRITICAL: Compute weight gradient FIRST, before we overwrite grad_attn_out_flat with input gradient!
    // grad_W_o = cached_attn_output^T @ grad_attn_output
    // BUG FIX Dec 24, 2025: Previously computed input grad first, which overwrote the activation
    // before it was used for weight gradient. This caused W_o gradients to be near-zero.
    
    // LOG: Capture inputs to W_o weight gradient computation
    P2_INFO("Checking grad_checks flag: " << ctx.enable_grad_checks);
    if (ctx.enable_grad_checks) {
        P2_INFO("Queueing grad stats for wo_incoming_grad");
        queueGradStats(
            "wo_incoming_grad",
            layer,
            grad_attn_output,
            static_cast<size_t>(total_tokens) * cfg->d_model,
            ctx.explosion_threshold,
            ctx.training_state->stream_ctrl.getPrimaryStream());
        P2_INFO("Queueing grad stats for wo_cached_activation");
        queueGradStats(
            "wo_cached_activation",
            layer,
            grad_attn_out_flat,
            static_cast<size_t>(total_tokens) * cfg->d_model,
            ctx.explosion_threshold,
            ctx.training_state->stream_ctrl.getPrimaryStream());
        P2_INFO("Grad stats queued");
    }
    
    P2_INFO("About to call cublasSgemm for W_o weight gradient");
    P2_INFO("Grad buffer pointer check: attn_out_weight_grads[" << layer << "]=" << (void*)ts->attn_out_weight_grads[layer]);
    
    // Validate ALL buffer pointers before GEMM
    P2_INFO("Validating input buffers:");
    P2_INFO("  grad_attn_out_flat=" << (void*)grad_attn_out_flat);
    P2_INFO("  grad_attn_output=" << (void*)grad_attn_output);
    P2_INFO("  W_o=" << (void*)W_o);
    
    if (!ts->attn_out_weight_grads[layer]) {
        BWD_FAIL_LOUD(ctx, BackwardStatus::INVALID_STATE, "attn_out_weight_grads is NULL", layer);
    }
    if (!grad_attn_out_flat) {
        BWD_FAIL_LOUD(ctx, BackwardStatus::INVALID_STATE, "grad_attn_out_flat is NULL", layer);
    }
    if (!grad_attn_output) {
        BWD_FAIL_LOUD(ctx, BackwardStatus::INVALID_STATE, "grad_attn_output is NULL", layer);
    }
    if (!W_o) {
        BWD_FAIL_LOUD(ctx, BackwardStatus::INVALID_STATE, "W_o is NULL", layer);
    }
    
    P2_INFO("GEMM params: M=" << cfg->d_model << " N=" << cfg->d_model << " K=" << total_tokens 
            << " lda=" << cfg->d_model << " ldb=" << cfg->d_model << " ldc=" << cfg->d_model);
    
    P2_INFO("Calling cublasSgemm for W_o weight gradient...");
    BWD_CHECK_CUBLAS(ctx, cublasSgemm(
        ctx.cublas_handle,
        CUBLAS_OP_N, CUBLAS_OP_T,
        cfg->d_model, cfg->d_model, total_tokens,
        &alpha,
        centered_scratch, cfg->d_model,          // Issue #43 FIX: use CENTERED activation
        grad_attn_output, cfg->d_model,          // B = gradient from layer above
        &beta_accum,
        ts->attn_out_weight_grads[layer], cfg->d_model),
        "W_o backward weight grad", layer);
    
    P2_INFO("cublasSgemm for W_o weight gradient COMPLETED successfully");
    
    // NOW compute input gradient (this overwrites grad_attn_out_flat, which is fine since
    // we've already used the activation for the weight gradient above)
    // grad_attn_input = W_o^T @ grad_attn_output
    
    P2_INFO("Starting W_o input gradient GEMM...");
    P2_INFO("GEMM params (input): M=" << cfg->d_model << " N=" << total_tokens << " K=" << cfg->d_model);
    BWD_CHECK_CUBLAS(ctx, cublasSgemm(
        ctx.cublas_handle,
        CUBLAS_OP_T, CUBLAS_OP_N,
        cfg->d_model, total_tokens, cfg->d_model,
        &alpha,
        W_o, cfg->d_model,
        grad_attn_output, cfg->d_model,
        &beta_zero,
        grad_attn_out_flat, cfg->d_model),
        "W_o backward input grad", layer);
    
    P2_INFO("W_o input gradient GEMM COMPLETED successfully");
    
    // W_o bias gradient (if using bias)
    if (ts->attn_out_bias_grads[layer]) {
        P2_INFO("Computing W_o bias gradient...");
        launchBiasSumGradient(
            const_cast<float*>(grad_attn_output),
            ts->attn_out_bias_grads[layer],
            total_tokens, cfg->d_model,
            ctx.training_state->stream_ctrl.getPrimaryStream());
        P2_INFO("W_o bias gradient COMPLETED");
    } else {
        P2_INFO("Skipping W_o bias gradient (use_bias=false)");
    }
    
    //--------------------------------------------------//
    // Step 7.2: Reshape for Flash Attention
    // grad_attn_out_flat [total_tokens, d_model] → [batch, heads, seq, head_dim]
    //--------------------------------------------------//
    
    P2_INFO("Starting reshape for Flash Attention backward");
    float* grad_attn_out_reshaped = ts->grad_attn_out_reshaped;
    BWD_CHECK_PTR(ctx, grad_attn_out_reshaped, "grad_attn_out_reshaped", layer);
    
    P2_INFO("Creating TensorContract views...");
    // Use TensorContract for type-safe reshape
    auto src_view = TensorContract::TensorView::make_BSM(
        grad_attn_out_flat, total_tokens, cfg->d_model, "grad_attn_flat");
    auto dst_view = TensorContract::TensorView::make_BHSD(
        grad_attn_out_reshaped, batch_size, num_heads, seq_len, head_dim, "grad_attn_bhsd");
    
    P2_INFO("Calling TensorContract::convert for reshape...");
    TensorContract::convert(src_view, dst_view, ctx.training_state->stream_ctrl.getPrimaryStream());
    BWD_CHECK_CUDA(ctx, cudaPeekAtLastError(), "Reshape grad_attn for Flash", layer);
    P2_INFO("Reshape COMPLETED successfully");
    
    //--------------------------------------------------//
    // Step 7.3: Flash Attention Backward (v2, BF16 path)
    //--------------------------------------------------//

    P2_INFO("Starting Flash Attention backward validation...");
    if (!cfg->use_flash_attention) {
        BWD_FAIL_LOUD(ctx, BackwardStatus::INVALID_STATE,
                      "Flash Attention disabled - no backward path available", layer);
    }
    if (head_dim != 32 && head_dim != 64) {
        BWD_FAIL_LOUD(ctx, BackwardStatus::INVALID_STATE,
                      "FlashAttention v2 requires head_dim=32 or 64", layer);
    }
    if (!ts->pbm_initialized || !ts->pbm_spec.valid || !ts->pbm_spec.alibi_slopes) {
        BWD_FAIL_LOUD(ctx, BackwardStatus::INVALID_STATE,
                      "ALiBi slopes are NULL - PBM hybrid requires ALiBi + RoPE", layer);
    }
    P2_INFO("Flash Attention config validated");
    if (ts->pbm_spec.num_heads != num_heads) {
    std::ostringstream oss;
    oss << "ALiBi num_heads mismatch: pbm_spec="
        << ts->pbm_spec.num_heads
        << " encoder="
        << num_heads;

    BWD_FAIL_LOUD(
        ctx,
        BackwardStatus::INVALID_STATE,
        oss.str().c_str(),
        layer
    );
}
if (ts->pbm_spec.num_kv_heads != num_kv_heads) {
    std::ostringstream oss;
    oss << "ALiBi num_kv_heads mismatch: pbm_spec="
        << ts->pbm_spec.num_kv_heads
        << " encoder="
        << num_kv_heads;

    BWD_FAIL_LOUD(
        ctx,
        BackwardStatus::INVALID_STATE,
        oss.str().c_str(),
        layer
    );
}

    cudaStream_t stream = ctx.training_state->stream_ctrl.getPrimaryStream();

    float* grad_Q = ts->grad_q;
    float* grad_K = ts->grad_k;
    float* grad_V = ts->grad_v;
    
    BWD_CHECK_PTR(ctx, grad_Q, "grad_Q", layer);
    BWD_CHECK_PTR(ctx, grad_K, "grad_K", layer);
    BWD_CHECK_PTR(ctx, grad_V, "grad_V", layer);
    
    float* cached_Q = ts->cached_Q[layer];
    float* cached_K = ts->cached_K[layer];
    float* cached_V = ts->cached_V[layer];
    float* cached_softmax_lse = ts->cached_softmax_lse[layer];
    
    // DIAGNOSTIC Issue #36: Verify cache contents at backward time
    static int bwd_read_count = 0;
    bwd_read_count++;
    if (bwd_read_count <= 24) {  // First 2 backward passes * 12 layers
        CUDA_CHECK(cudaStreamSynchronize(stream));
        std::vector<float> h_q(q_size);
        CUDA_CHECK(cudaMemcpy(h_q.data(), cached_Q, q_size * sizeof(float), cudaMemcpyDeviceToHost));
        double sum_sq = 0.0;
        for (std::size_t i = 0; i < q_size; ++i) sum_sq += h_q[i] * h_q[i];
        float rms = static_cast<float>(std::sqrt(sum_sq / q_size));
        fprintf(stderr, "[BWD_CACHE_READ_VERIFY] count=%d layer=%d cached_Q_rms=%.6f ptr=%p\n",
                bwd_read_count, layer, rms, (void*)cached_Q);
    }
    
    BWD_CHECK_PTR(ctx, cached_Q, "cached_Q", layer);
    BWD_CHECK_PTR(ctx, cached_K, "cached_K", layer);
    BWD_CHECK_PTR(ctx, cached_V, "cached_V", layer);
    BWD_CHECK_PTR(ctx, cached_softmax_lse, "cached_softmax_lse", layer);
    
    if (ctx.enable_grad_checks) {
        queueGradStats("cached_Q_check", layer, cached_Q, q_size, ctx.explosion_threshold, stream);
        queueGradStats("cached_K_check", layer, cached_K, kv_size, ctx.explosion_threshold, stream);
        queueGradStats("cached_V_check", layer, cached_V, kv_size, ctx.explosion_threshold, stream);
        queueGradStats("cached_attn_bhsd_check", layer, cached_attn_bhsd, q_size, ctx.explosion_threshold, stream);
    }

    __nv_bfloat16* fa_q = ts->fa_q_bf16;
    __nv_bfloat16* fa_k = ts->fa_k_bf16;
    __nv_bfloat16* fa_v = ts->fa_v_bf16;
    __nv_bfloat16* fa_out = ts->fa_out_bf16;
    __nv_bfloat16* fa_dout = ts->fa_dout_bf16;
    __nv_bfloat16* fa_dq = ts->fa_dq_bf16;
    __nv_bfloat16* fa_dk = ts->fa_dk_bf16;
    __nv_bfloat16* fa_dv = ts->fa_dv_bf16;

    BWD_CHECK_PTR(ctx, fa_q, "fa_q_bf16", layer);
    BWD_CHECK_PTR(ctx, fa_k, "fa_k_bf16", layer);
    BWD_CHECK_PTR(ctx, fa_v, "fa_v_bf16", layer);
    BWD_CHECK_PTR(ctx, fa_out, "fa_out_bf16", layer);
    BWD_CHECK_PTR(ctx, fa_dout, "fa_dout_bf16", layer);
    BWD_CHECK_PTR(ctx, fa_dq, "fa_dq_bf16", layer);
    BWD_CHECK_PTR(ctx, fa_dk, "fa_dk_bf16", layer);
    BWD_CHECK_PTR(ctx, fa_dv, "fa_dv_bf16", layer);
    BWD_CHECK_PTR(ctx, ts->fa_dq_accum, "fa_dq_accum", layer);
    BWD_CHECK_PTR(ctx, ts->fa_dsoftmax_sum, "fa_dsoftmax_sum", layer);

    if (ts->fa_q_bf16_elems < q_size || ts->fa_kv_bf16_elems < kv_size) {
        BWD_FAIL_LOUD(ctx, BackwardStatus::INVALID_STATE,
                      "BF16 scratch buffers too small for current batch", layer);
    }

    const size_t dq_accum_required = flash_attn_dq_accum_bytes(batch_size, seq_len, num_heads, head_dim);
    const size_t dsoftmax_required = flash_attn_dsoftmax_sum_bytes(batch_size, seq_len, num_heads);
    if (ts->fa_dq_accum_bytes < dq_accum_required || ts->fa_dsoftmax_sum_bytes < dsoftmax_required) {
        BWD_FAIL_LOUD(ctx, BackwardStatus::INVALID_STATE,
                      "FlashAttention workspace too small for current batch", layer);
    }

    const size_t expected_total_tokens = static_cast<size_t>(batch_size) * seq_len;
    if (expected_total_tokens != static_cast<size_t>(total_tokens)) {
        P2_WARN("Token count mismatch: total_tokens=" << total_tokens
                << " expected=" << expected_total_tokens);
    }
    P2_INFO("Flash Attention cache sizing:");
    P2_INFO("  max_cached_batch=" << ts->max_cached_batch
            << " max_cached_seq_len=" << ts->max_cached_seq_len
            << " max_cached_tokens=" << ts->max_cached_tokens);
    P2_INFO("  cached_batch_size=" << ts->cached_batch_size
            << " cached_seq_len=" << ts->cached_seq_len
            << " cached_valid_tokens=" << ts->cached_valid_tokens);
    if (batch_size > ts->max_cached_batch || seq_len > ts->max_cached_seq_len) {
        std::ostringstream oss;
        oss << "Batch/seq exceed cache: batch=" << batch_size
            << " seq=" << seq_len
            << " max_batch=" << ts->max_cached_batch
            << " max_seq=" << ts->max_cached_seq_len;
        BWD_FAIL_LOUD(ctx, BackwardStatus::INVALID_STATE, oss.str(), layer);
    }
    if (static_cast<size_t>(total_tokens) > ts->max_cached_tokens) {
        std::ostringstream oss;
        oss << "total_tokens exceed cache: total_tokens=" << total_tokens
            << " max_cached_tokens=" << ts->max_cached_tokens;
        BWD_FAIL_LOUD(ctx, BackwardStatus::INVALID_STATE, oss.str(), layer);
    }

    const size_t softmax_lse_elems = static_cast<size_t>(batch_size) *
                                     static_cast<size_t>(num_heads) *
                                     static_cast<size_t>(seq_len);
    const size_t max_softmax_lse_elems = static_cast<size_t>(ts->max_cached_batch) *
                                         static_cast<size_t>(num_heads) *
                                         static_cast<size_t>(ts->max_cached_seq_len);
    if (softmax_lse_elems > max_softmax_lse_elems) {
        std::ostringstream oss;
        oss << "softmax_lse exceeds cache: elems=" << softmax_lse_elems
            << " max_elems=" << max_softmax_lse_elems;
        BWD_FAIL_LOUD(ctx, BackwardStatus::INVALID_STATE, oss.str(), layer);
    }

    P2_INFO("Pointer diagnostics (expected bytes for current batch):");
    const size_t q_bytes = q_size * sizeof(float);
    const size_t kv_bytes = kv_size * sizeof(float);
    const size_t q_bf16_bytes = q_size * sizeof(__nv_bfloat16);
    const size_t kv_bf16_bytes = kv_size * sizeof(__nv_bfloat16);
    const size_t softmax_lse_bytes = softmax_lse_elems * sizeof(float);
    const size_t alibi_bytes = static_cast<size_t>(num_heads) * sizeof(float);
    bool range_ok = true;
    range_ok &= logPointerDiagnostics("cached_Q", cached_Q, q_bytes);
    range_ok &= logPointerDiagnostics("cached_K", cached_K, kv_bytes);
    range_ok &= logPointerDiagnostics("cached_V", cached_V, kv_bytes);
    range_ok &= logPointerDiagnostics("cached_attn_bhsd", cached_attn_bhsd, q_bytes);
    range_ok &= logPointerDiagnostics("grad_attn_out_reshaped", grad_attn_out_reshaped, q_bytes);
    range_ok &= logPointerDiagnostics("cached_softmax_lse", cached_softmax_lse, softmax_lse_bytes);
    range_ok &= logPointerDiagnostics("alibi_slopes", ts->pbm_spec.alibi_slopes, alibi_bytes);
    range_ok &= logPointerDiagnostics("fa_q_bf16", fa_q, q_bf16_bytes);
    range_ok &= logPointerDiagnostics("fa_k_bf16", fa_k, kv_bf16_bytes);
    range_ok &= logPointerDiagnostics("fa_v_bf16", fa_v, kv_bf16_bytes);
    range_ok &= logPointerDiagnostics("fa_out_bf16", fa_out, q_bf16_bytes);
    range_ok &= logPointerDiagnostics("fa_dout_bf16", fa_dout, q_bf16_bytes);
    range_ok &= logPointerDiagnostics("fa_dq_bf16", fa_dq, q_bf16_bytes);
    range_ok &= logPointerDiagnostics("fa_dk_bf16", fa_dk, kv_bf16_bytes);
    range_ok &= logPointerDiagnostics("fa_dv_bf16", fa_dv, kv_bf16_bytes);
    range_ok &= logPointerDiagnostics("fa_dq_accum", ts->fa_dq_accum, dq_accum_required);
    range_ok &= logPointerDiagnostics("fa_dsoftmax_sum", ts->fa_dsoftmax_sum, dsoftmax_required);
    if (!range_ok) {
        BWD_FAIL_LOUD(ctx, BackwardStatus::INVALID_STATE,
                      "Flash Attention pointer diagnostics failed (allocation too small)", layer);
    }

    P2_INFO("Converting cached Q/K/V/Out to BF16...");
    TensorConversion::convert_BHSD_to_BSHD_bf16(cached_Q, fa_q,
                                                batch_size, num_heads, seq_len, head_dim, stream);
    TensorConversion::convert_BHSD_to_BSHD_bf16(cached_K, fa_k,
                                                batch_size, num_kv_heads, seq_len, head_dim, stream);
    TensorConversion::convert_BHSD_to_BSHD_bf16(cached_V, fa_v,
                                                batch_size, num_kv_heads, seq_len, head_dim, stream);
    TensorConversion::convert_BHSD_to_BSHD_bf16(cached_attn_bhsd, fa_out,
                                                batch_size, num_heads, seq_len, head_dim, stream);
    TensorConversion::convert_BHSD_to_BSHD_bf16(grad_attn_out_reshaped, fa_dout,
                                                batch_size, num_heads, seq_len, head_dim, stream);
    BWD_CHECK_CUDA(ctx, cudaPeekAtLastError(), "BF16 conversions", layer);
    P2_INFO("BF16 conversions COMPLETED");

    P2_INFO("Calling flash_attn_bwd_ex...");
    P2_INFO("Flash Attention params:");
    P2_INFO("  batch_size=" << batch_size << " seq_len=" << seq_len);
    P2_INFO("  num_heads=" << num_heads << " num_kv_heads=" << num_kv_heads << " head_dim=" << head_dim);
    P2_INFO("  causal_mask=" << (cfg->causal_mask ? "true" : "false"));
    P2_INFO("Buffer pointers:");
    P2_INFO("  fa_q=" << (void*)fa_q << " fa_k=" << (void*)fa_k << " fa_v=" << (void*)fa_v);
    P2_INFO("  fa_out=" << (void*)fa_out << " fa_dout=" << (void*)fa_dout);
    P2_INFO("  cached_softmax_lse=" << (void*)cached_softmax_lse);
    P2_INFO("  alibi_slopes=" << (void*)ts->pbm_spec.alibi_slopes);
    P2_INFO("  fa_dq=" << (void*)fa_dq << " fa_dk=" << (void*)fa_dk << " fa_dv=" << (void*)fa_dv);
    P2_INFO("  fa_dq_accum=" << (void*)ts->fa_dq_accum << " fa_dsoftmax_sum=" << (void*)ts->fa_dsoftmax_sum);
    P2_INFO("Buffer sizes:");
    P2_INFO("  q/out size=" << q_size << " kv size=" << kv_size);
    P2_INFO("  fa_q_bf16_elems=" << ts->fa_q_bf16_elems << " fa_kv_bf16_elems=" << ts->fa_kv_bf16_elems);
    P2_INFO("  dq_accum_bytes=" << ts->fa_dq_accum_bytes << " dsoftmax_bytes=" << ts->fa_dsoftmax_sum_bytes);
    {
        auto is_aligned = [](const void* ptr, size_t alignment) {
            return (reinterpret_cast<std::uintptr_t>(ptr) & (alignment - 1)) == 0;
        };
        P2_INFO("Validation:");
        P2_INFO("  dq_accum_required=" << dq_accum_required << " dsoftmax_required=" << dsoftmax_required);
        P2_INFO("  ptr_align_16: fa_q=" << is_aligned(fa_q, 16)
                                   << " fa_k=" << is_aligned(fa_k, 16)
                                   << " fa_v=" << is_aligned(fa_v, 16)
                                   << " fa_out=" << is_aligned(fa_out, 16)
                                   << " fa_dout=" << is_aligned(fa_dout, 16));
        P2_INFO("  ptr_align_16: fa_dq=" << is_aligned(fa_dq, 16)
                                   << " fa_dk=" << is_aligned(fa_dk, 16)
                                   << " fa_dv=" << is_aligned(fa_dv, 16)
                                   << " fa_dq_accum=" << is_aligned(ts->fa_dq_accum, 16)
                                   << " fa_dsoftmax_sum=" << is_aligned(ts->fa_dsoftmax_sum, 16));
        P2_INFO("  ptr_align_16: cached_softmax_lse=" << is_aligned(cached_softmax_lse, 16)
                                   << " alibi_slopes=" << is_aligned(ts->pbm_spec.alibi_slopes, 16));
        P2_INFO("  alias_checks: q==k=" << (fa_q == fa_k)
                                   << " q==v=" << (fa_q == fa_v)
                                   << " k==v=" << (fa_k == fa_v)
                                   << " q==out=" << (fa_q == fa_out)
                                   << " out==dout=" << (fa_out == fa_dout));
    }
    if (ctx.enable_grad_checks) {
        P2_INFO("Pre-flash_attn_bwd_ex CUDA check...");
        BWD_CHECK_CUDA(ctx, cudaGetLastError(), "Pre Flash Attention backward", layer);
        P2_INFO("Synchronizing stream before flash_attn_bwd_ex...");
        BWD_CHECK_CUDA(ctx, cudaStreamSynchronize(stream), "Pre Flash Attention backward sync", layer);
        P2_INFO("Pre-flash_attn_bwd_ex stream sync OK");
    }
    
    flash_attn_bwd_ex(
        fa_q,
        fa_k,
        fa_v,
        fa_out,
        fa_dout,
        cached_softmax_lse,
        ts->pbm_spec.alibi_slopes,
        fa_dq,
        fa_dk,
        fa_dv,
        ts->fa_dq_accum,
        ts->fa_dsoftmax_sum,
        batch_size,
        seq_len,
        num_heads,
        num_kv_heads,
        head_dim,
        cfg->causal_mask,
        true,
        stream);

    P2_INFO("flash_attn_bwd_ex COMPLETED successfully");
    BWD_CHECK_CUDA(ctx, cudaPeekAtLastError(), "Flash Attention backward", layer);
    if (ctx.enable_grad_checks) {
        P2_INFO("Synchronizing stream after flash_attn_bwd_ex for error check...");
        BWD_CHECK_CUDA(ctx, cudaStreamSynchronize(stream), "Flash Attention backward sync", layer);
        P2_INFO("flash_attn_bwd_ex stream sync OK");
    }

    P2_INFO("Converting BF16 gradients back to FP32...");
    TensorConversion::convert_BSHD_bf16_to_BHSD(fa_dq, grad_Q,
                                                batch_size, seq_len, num_heads, head_dim, stream);
    TensorConversion::convert_BSHD_bf16_to_BHSD(fa_dk, grad_K,
                                                batch_size, seq_len, num_kv_heads, head_dim, stream);
    TensorConversion::convert_BSHD_bf16_to_BHSD(fa_dv, grad_V,
                                                batch_size, seq_len, num_kv_heads, head_dim, stream);
    P2_INFO("BF16->FP32 conversions COMPLETED");
    
    // Issue #42 DIAGNOSTIC: Log dQ/dK/dV gradient statistics after Flash Attention backward
    // These are the raw gradients that will flow to encoder weights via W_qkv backward
    {
        cudaStreamSynchronize(stream);  // Ensure conversions complete
        
        std::vector<float> dq_host(q_size);
        std::vector<float> dk_host(kv_size);
        std::vector<float> dv_host(kv_size);
        
        cudaMemcpy(dq_host.data(), grad_Q, q_size * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(dk_host.data(), grad_K, kv_size * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(dv_host.data(), grad_V, kv_size * sizeof(float), cudaMemcpyDeviceToHost);
        
        // Compute RMS and mean for each gradient
        auto computeStats = [layer](const std::vector<float>& v, const char* name) {
            double sum = 0.0, sum_sq = 0.0;
            float max_abs = 0.0f;
            for (float val : v) {
                sum += val;
                sum_sq += val * val;
                max_abs = std::max(max_abs, std::fabsf(val));
            }
            float mean = static_cast<float>(sum / v.size());
            float rms = static_cast<float>(std::sqrt(sum_sq / v.size()));
            P2_INFO("[FlashAttnGrad] layer=" << layer << " " << name 
                    << ": size=" << v.size() << " mean=" << mean 
                    << " rms=" << rms << " max_abs=" << max_abs);
        };
        
        computeStats(dq_host, "dQ");
        computeStats(dk_host, "dK");
        computeStats(dv_host, "dV");
    }
    
    //--------------------------------------------------//
    // Step 7.3b: CRITICAL - RoPE Backward (inverse rotation)
    //
    // During forward, Q and K were rotated by RoPE angles θ.
    // Flash Attention backward computed gradients in ROTATED space.
    // We MUST apply inverse rotation (-θ) to transform gradients
    // back to ORIGINAL (unrotated) space before W_qkv backward.
    //
    // Without this, gradients are in wrong coordinate space and
    // appear as ~400x smaller noise (root cause of gradient vanishing).
    //--------------------------------------------------//
    
    if (!ts->pbm_initialized || !ts->pbm_spec.valid || !ts->pbm_spec.rope_inv_freq) {
        BWD_FAIL_LOUD(ctx, BackwardStatus::INVALID_STATE,
                      "RoPE inv_freq is NULL - cannot apply RoPE backward", layer);
    }
    
    PBM::launchRoPERotationGQA_backward(
        grad_Q,                          // [batch, num_heads, seq, head_dim] - in-place
        grad_K,                          // [batch, num_kv_heads, seq, head_dim] - in-place
        ts->pbm_spec.rope_inv_freq,      // Inverse frequencies [rotary_dim/2]
        batch_size,
        num_heads,                       // Q head count
        num_kv_heads,                    // K head count (GQA)
        seq_len,
        head_dim,
        ts->pbm_spec.rotary_dim,
        stream
    );
    
    BWD_CHECK_CUDA(ctx, cudaPeekAtLastError(), "RoPE backward (inverse rotation)", layer);
    
    // Verify gradients after Flash Attention + RoPE backward (deferred to batch flush)
    if (ctx.enable_grad_checks) {
        queueGradStats("grad_Q", layer, grad_Q, q_size, ctx.explosion_threshold, stream);
        queueGradStats("grad_K", layer, grad_K, kv_size, ctx.explosion_threshold, stream);
        queueGradStats("grad_V", layer, grad_V, kv_size, ctx.explosion_threshold, stream);
    }
    
    //--------------------------------------------------//
    // Step 7.4: Merge Q, K, V gradients into fused format
    //--------------------------------------------------//
    
    float* grad_qkv_concat = ts->grad_qkv_concat;
    BWD_CHECK_PTR(ctx, grad_qkv_concat, "grad_qkv_concat", layer);
    
    // Use existing gqa_dims from line 646 (already validated)
    
    auto view_grad_Q = TensorContract::TensorView::make_BHSD(
        grad_Q, batch_size, num_heads, seq_len, head_dim, "grad_Q");
    auto view_grad_K = TensorContract::TensorView::make_BHSD(
        grad_K, batch_size, num_kv_heads, seq_len, head_dim, "grad_K");
    auto view_grad_V = TensorContract::TensorView::make_BHSD(
        grad_V, batch_size, num_kv_heads, seq_len, head_dim, "grad_V");
    auto view_grad_qkv = TensorContract::TensorView::make_QKV_FUSED(
        grad_qkv_concat, total_tokens, total_qkv_dim, "grad_qkv_fused");
    
    TensorContract::merge_qkv_grads_gqa(
        view_grad_Q, view_grad_K, view_grad_V,
        view_grad_qkv, gqa_dims,
        ctx.training_state->stream_ctrl.getPrimaryStream());
    
    BWD_CHECK_CUDA(ctx, cudaPeekAtLastError(), "Merge QKV gradients", layer);
    
    //--------------------------------------------------//
    // Step 7.5: QKV Weight Gradient
    // grad_W_qkv = grad_qkv^T @ ln1_output
    //--------------------------------------------------//
    
    // Issue #43 FIX: Center cached_ln1_output before weight gradient GEMM
    // Note: centered_scratch is already declared earlier in computeAttentionBackward
    centerActivations(
        cached_ln1_output,
        centered_scratch,        // Output: centered activation
        cfg->d_model,
        total_tokens,
        ctx.training_state->stream_ctrl.getPrimaryStream());
    
    BWD_CHECK_CUBLAS(ctx, cublasSgemm(
        ctx.cublas_handle,
        CUBLAS_OP_N, CUBLAS_OP_T,
        cfg->d_model, total_qkv_dim, total_tokens,
        &alpha,
        centered_scratch, cfg->d_model,          // Issue #43 FIX: use CENTERED activation
        grad_qkv_concat, total_qkv_dim,
        &beta_accum,
        ts->attn_qkv_weight_grads[layer], cfg->d_model),
        "QKV weight gradient", layer);
    
    //--------------------------------------------------//
    // Step 7.6: QKV Bias Gradient (if using bias)
    //--------------------------------------------------//
    
    if (ts->attn_qkv_bias_grads[layer]) {
        launchBiasSumGradient(
            grad_qkv_concat,
            ts->attn_qkv_bias_grads[layer],
            total_tokens, total_qkv_dim,
            ctx.training_state->stream_ctrl.getPrimaryStream());
    }
    
    //--------------------------------------------------//
    // Step 7.7: Propagate gradient through QKV projection
    // grad_qkv_input = grad_qkv @ W_qkv
    //--------------------------------------------------//
    
    float* W_qkv = enc_attn->getAttnWqkv();
    BWD_CHECK_PTR(ctx, W_qkv, "W_qkv", layer);
    
    float* grad_qkv_input = ts->grad_qkv_input;
    BWD_CHECK_PTR(ctx, grad_qkv_input, "grad_qkv_input", layer);
    
    // W_qkv: [total_qkv_dim, d_model] row-major
    // grad_qkv_concat: [total_tokens, total_qkv_dim]
    // grad_qkv_input: [total_tokens, d_model]
    
    BWD_CHECK_CUBLAS(ctx, cublasSgemm(
        ctx.cublas_handle,
        CUBLAS_OP_T, CUBLAS_OP_N,
        cfg->d_model, total_tokens, total_qkv_dim,
        &alpha,
        W_qkv, total_qkv_dim,
        grad_qkv_concat, total_qkv_dim,
        &beta_zero,
        grad_qkv_input, cfg->d_model),
        "QKV projection backward", layer);
    
    if (ctx.enable_grad_checks) {
        queueGradStats(
            "grad_qkv_input",
            layer,
            grad_qkv_input,
            static_cast<size_t>(total_tokens) * cfg->d_model,
            ctx.explosion_threshold,
            ctx.training_state->stream_ctrl.getPrimaryStream());
    }
    
    return BackwardStatus::SUCCESS;
}

} // namespace Backward
} // namespace GRIM

