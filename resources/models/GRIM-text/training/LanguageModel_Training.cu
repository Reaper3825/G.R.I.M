#define USE_CUDA

//======================================================//
//  LanguageModel_Training.cu - Training Methods
//======================================================//
//
//  This file contains LanguageModel training methods that
//  integrate with the 3-phase backward pass architecture.
//
//  Methods in this file:
//  - backward() - Wrapper that calls 3-phase orchestrator
//  - updateWeights() - AdamW optimizer step
//  - computeGradNorm() - L2 norm of all gradients
//  - scaleGradients() - Gradient clipping helper
//  - zeroGrad() - Zero all gradient buffers
//  - setSequenceLossWeights() / clearSequenceLossWeights()
//  - recordGradientClip() - Gradient metrics tracking
//  - resetOptimizerMoments() - Reset optimizer state
//  - configureUpdateProbe() / disableUpdateProbe() - Debug probes
//  - buildParameterGroups() - Initialize parameter groups
//
//======================================================//

#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <cstdlib>
#include <set>
#include <mutex>
#include <fstream>
#include <sstream>
#include <vector>

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "../GRIM/grim_language_model_cuda.hpp"
#include "../Layers/Encoding/Encoding_GPU.hpp"
#include "../Layers/Embedding/Embedding_GPU.hpp"
#include "../Shared/GradNorm/GradNormGPU.hpp"
#include "../Common/grim_scale_buffer.hpp"
#include "../Shared/LogRecorder/LogRecorder.hpp"
#include "../Shared/TensorContract/TensorContract_GPU.hpp"
#include "../Shared/Optimizers/AdamW/AdamW_Kernal_GPU.hpp"
#include "../../../control/ai_config_paths.hpp"
#include "../Shared/HyperParameters/HyperParameters_GPU.hpp"
#include "../Shared/Loss/ComputeLoss/AutogradLoss.hpp"
#include "Autograd/AutogradTraining.hpp"  // Pure autograd backward - replaces legacy 3-phase system

extern "C" {
    void launchXavierInit(float* weights, int size, float stddev,
                         unsigned int seed, cudaStream_t stream);
}

namespace GRIM {

#ifdef USE_CUDA

//======================================================//
//  Logging Setup
//======================================================//

namespace {
constexpr auto kBwdModule = GRIM::Logging::ModuleId::BackwardPass;

#define BWD_INFO(msg) do { std::ostringstream _oss; _oss << msg; GRIM::Logging::EmitModuleInfo(kBwdModule, _oss.str()); } while (0)
#define BWD_WARN(msg) do { std::ostringstream _oss; _oss << msg; GRIM::Logging::EmitModuleWarning(kBwdModule, _oss.str()); } while (0)
#define BWD_ERROR(msg) do { std::ostringstream _oss; _oss << msg; GRIM::Logging::EmitModuleError(kBwdModule, _oss.str()); } while (0)
std::string g_gradcheck_log_path;
std::mutex g_gradcheck_mutex;

struct NonFiniteScanResult {
    int index;
    float value;
    int is_nan;
    int is_inf;
};

__global__ void scanNonFiniteKernel(const float* __restrict__ data,
                                    size_t size,
                                    NonFiniteScanResult* __restrict__ out) {
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t stride = blockDim.x * gridDim.x;
    for (size_t i = idx; i < size; i += stride) {
        float v = data[i];
        if (!isfinite(v)) {
            if (atomicCAS(&out->index, -1, static_cast<int>(i)) == -1) {
                out->value = v;
                out->is_nan = isnan(v) ? 1 : 0;
                out->is_inf = isinf(v) ? 1 : 0;
            }
            return;
        }
    }
}

bool endsWith(const std::string& value, const std::string& suffix) {
    return value.size() >= suffix.size() &&
           value.compare(value.size() - suffix.size(), suffix.size(), suffix) == 0;
}

bool getGroupShape(const ParameterGroup& group,
                   const LanguageModelConfig& cfg,
                   int num_kv_heads,
                   int* rows,
                   int* cols) {
    if (!rows || !cols) {
        return false;
    }
    int r = 0;
    int c = 0;
    if (group.name == "embedding" ||
        group.name == "embedding_lm_head_tied" ||
        group.name == "lm_head_weight") {
        r = cfg.vocab_size;
        c = cfg.d_model;
    } else if (endsWith(group.name, "_qkv_weight")) {
        if (cfg.num_heads <= 0 || cfg.d_model <= 0 || num_kv_heads <= 0) {
            return false;
        }
        const int head_dim = cfg.head_dim;  // Use pre-computed value from config
        if (head_dim <= 0) {
            return false;
        }
        const int kv_dim = num_kv_heads * head_dim;
        const int total_qkv_dim = cfg.d_model + 2 * kv_dim;
        r = total_qkv_dim;
        c = cfg.d_model;
    } else if (endsWith(group.name, "_wo_weight")) {
        r = cfg.d_model;
        c = cfg.d_model;
    } else if (endsWith(group.name, "_ffn_w1")) {
        r = cfg.d_model;
        c = cfg.d_ff;
    } else if (endsWith(group.name, "_ffn_w2")) {
        r = cfg.d_ff;
        c = cfg.d_model;
    } else if (endsWith(group.name, "_rms1_gamma") ||
               endsWith(group.name, "_rms2_gamma")) {
        r = cfg.d_model;
        c = 1;
    } else if (group.name == "scratch_block_atom_type_embeddings") {
        constexpr int kNumAtomTypes = 16;
        r = kNumAtomTypes;
        c = cfg.scratch_block_atom_embedding_dim;
    } else if (group.name == "scratch_block_atom_projection") {
        r = cfg.scratch_block_atom_embedding_dim;
        c = cfg.d_model;
    } else if (group.name == "scratch_block_text_feature_projection") {
        constexpr int kTextFeatureDim = 16;
        r = kTextFeatureDim;
        c = cfg.d_model;
    } else if (group.name == "numeric_head_weight") {
        r = cfg.d_model;
        c = 1;
    } else {
        return false;
    }

    if (r <= 0 || c <= 0) {
        return false;
    }
    const size_t expected = static_cast<size_t>(r) * static_cast<size_t>(c);
    if (expected != group.size) {
        return false;
    }
    *rows = r;
    *cols = c;
    return true;
}

bool scanFirstNonFinite(const float* data,
                        size_t size,
                        NonFiniteScanResult* out,
                        cudaStream_t stream) {
    if (!data || size == 0 || !out) {
        return false;
    }
    static NonFiniteScanResult* d_result = nullptr;
    if (!d_result) {
        cudaError_t alloc_err = cudaMalloc(&d_result, sizeof(NonFiniteScanResult));
        if (alloc_err != cudaSuccess) {
            fprintf(stderr, "[computeGradNorm] FATAL: scan buffer alloc failed: %s\n",
                    cudaGetErrorString(alloc_err));
            return false;
        }
    }
    NonFiniteScanResult init{};
    init.index = -1;
    init.value = 0.0f;
    init.is_nan = 0;
    init.is_inf = 0;
    cudaMemcpyAsync(d_result, &init, sizeof(init), cudaMemcpyHostToDevice, stream);

    const int threads = 256;
    size_t blocks = (size + threads - 1) / threads;
    if (blocks > 1024) {
        blocks = 1024;
    }
    scanNonFiniteKernel<<<static_cast<int>(blocks), threads, 0, stream>>>(data, size, d_result);
    cudaError_t kernel_err = cudaGetLastError();
    if (kernel_err != cudaSuccess) {
        fprintf(stderr, "[computeGradNorm] FATAL: scan kernel launch failed: %s\n",
                cudaGetErrorString(kernel_err));
        return false;
    }

    cudaMemcpyAsync(out, d_result, sizeof(*out), cudaMemcpyDeviceToHost, stream);
    cudaError_t sync_err = cudaStreamSynchronize(stream);
    if (sync_err != cudaSuccess) {
        fprintf(stderr, "[computeGradNorm] FATAL: scan sync failed: %s\n",
                cudaGetErrorString(sync_err));
        return false;
    }

    return out->index >= 0;
}
}

void setGradCheckLogPath(const std::string& path) {
    std::lock_guard<std::mutex> lock(g_gradcheck_mutex);
    g_gradcheck_log_path = path;
}

//======================================================//
//  zeroGrad - Zero all gradient buffers
//======================================================//

void LanguageModel::zeroGrad() {
        BWD_INFO("[zeroGrad] Zeroing all gradient buffers");
    
    if (!training_state_.initialized) {
        BWD_INFO("[zeroGrad] TrainingState not initialized; initializing TrainingState...");
        initTrainingState();
    }
    
    const auto& cfg = config_;
    
    // GQA: Compute dimensions for gradient zeroing
    const int head_dim = cfg.head_dim;  // Use pre-computed value from config
    const int num_kv_heads = training_state_.num_kv_heads;
    const int kv_dim = num_kv_heads * head_dim;
    const int total_qkv_dim = cfg.d_model + 2 * kv_dim;
    
    // Zero embedding gradients (Tensor API)
    if (float* emb_grad = training_state_.embedding_weights.grad_data()) {
        cudaMemsetAsync(emb_grad, 0, 
                       cfg.vocab_size * cfg.d_model * sizeof(float),
                       training_state_.stream_ctrl.getPrimaryStream());
    }
    
    // Issue #36 FIX: Zero position embedding gradients
    if (float* pos_grad = training_state_.position_embedding_weights.grad_data()) {
        cudaMemsetAsync(pos_grad, 0,
                       cfg.max_seq_len * cfg.d_model * sizeof(float),
                       training_state_.stream_ctrl.getPrimaryStream());
    }
    
    // Zero LM head gradients (Tensor API)
    if (float* lm_weight_grad = training_state_.lm_head_weights.grad_data()) {
        cudaMemsetAsync(lm_weight_grad, 0,
                       cfg.vocab_size * cfg.d_model * sizeof(float),
                       training_state_.stream_ctrl.getPrimaryStream());
    }
    // ISSUE #59: Use grad_data() accessor
    if (float* lm_bias_grad = training_state_.lm_head_bias.grad_data()) {
        cudaMemsetAsync(lm_bias_grad, 0,
                       cfg.vocab_size * sizeof(float),
                       training_state_.stream_ctrl.getPrimaryStream());
    }

    // Zero numeric head gradients (Tensor API)
    // ISSUE #59: Use grad_data() accessor
    if (float* num_weight_grad = training_state_.numeric_head_weights.grad_data()) {
        cudaMemsetAsync(num_weight_grad, 0,
                       cfg.d_model * sizeof(float),
                       training_state_.stream_ctrl.getPrimaryStream());
    }
    if (float* num_bias_grad = training_state_.numeric_head_bias.grad_data()) {
        cudaMemsetAsync(num_bias_grad, 0,
                       sizeof(float),
                       training_state_.stream_ctrl.getPrimaryStream());
    }
    
    // Zero encoder layer gradients (using encoder's Tensor.grad per AUTOGRAD MIGRATION)
    GPUGrimEncoder* gpu_encoder = &getGpuEncoder();
    for (int layer = 0; layer < cfg.num_layers; ++layer) {
        GPUEncoderLayer* enc = gpu_encoder ? gpu_encoder->getLayer(layer) : nullptr;
        if (!enc) continue;
        
        // RMSNorm gamma gradients
        if (float* grad = enc->getRMS1GammaGrad()) {
            cudaMemsetAsync(grad, 0, cfg.d_model * sizeof(float), training_state_.stream_ctrl.getPrimaryStream());
        }
        if (float* grad = enc->getRMS2GammaGrad()) {
            cudaMemsetAsync(grad, 0, cfg.d_model * sizeof(float), training_state_.stream_ctrl.getPrimaryStream());
        }
        
        // Attention weight/bias gradients (via encoder's Tensor.grad)
        if (float* grad = enc->getAttnWqkvGrad()) {
            size_t qkv_weight_size = static_cast<size_t>(total_qkv_dim) * cfg.d_model;
            cudaMemsetAsync(grad, 0, qkv_weight_size * sizeof(float), training_state_.stream_ctrl.getPrimaryStream());
        }
        if (float* grad = enc->getAttnBqkvGrad()) {
            cudaMemsetAsync(grad, 0, total_qkv_dim * sizeof(float), training_state_.stream_ctrl.getPrimaryStream());
        }
        if (float* grad = enc->getAttnWoGrad()) {
            cudaMemsetAsync(grad, 0, cfg.d_model * cfg.d_model * sizeof(float), training_state_.stream_ctrl.getPrimaryStream());
        }
        if (float* grad = enc->getAttnBoGrad()) {
            cudaMemsetAsync(grad, 0, cfg.d_model * sizeof(float), training_state_.stream_ctrl.getPrimaryStream());
        }
        
        // FFN weight/bias gradients (via encoder's Tensor.grad)
        if (float* grad = enc->getFFNW1Grad()) {
            cudaMemsetAsync(grad, 0, cfg.d_model * cfg.d_ff * sizeof(float), training_state_.stream_ctrl.getPrimaryStream());
        }
        if (float* grad = enc->getFFNB1Grad()) {
            cudaMemsetAsync(grad, 0, cfg.d_ff * sizeof(float), training_state_.stream_ctrl.getPrimaryStream());
        }
        if (float* grad = enc->getFFNW2Grad()) {
            cudaMemsetAsync(grad, 0, cfg.d_ff * cfg.d_model * sizeof(float), training_state_.stream_ctrl.getPrimaryStream());
        }
        if (float* grad = enc->getFFNB2Grad()) {
            cudaMemsetAsync(grad, 0, cfg.d_model * sizeof(float), training_state_.stream_ctrl.getPrimaryStream());
        }
    }
    
    // Zero ScratchBlock gradients (owned by ScratchBlockLayer, not TrainingState per Rule 22)
    if (scratch_block_layer_ && scratch_block_layer_->isEnabled()) {
        constexpr int NUM_ATOM_TYPES = HyperParameters::NUM_ATOM_TYPES;
        constexpr int kTextFeatureDim = 16;
        const int atom_embedding_dim = cfg.scratch_block_atom_embedding_dim;
        
        if (float* atom_type_grad = scratch_block_layer_->getAtomTypeEmbeddingsGrad()) {
            cudaMemsetAsync(atom_type_grad, 0,
                           NUM_ATOM_TYPES * atom_embedding_dim * sizeof(float),
                           training_state_.stream_ctrl.getPrimaryStream());
        }
        if (float* atom_proj_grad = scratch_block_layer_->getAtomProjectionGrad()) {
            cudaMemsetAsync(atom_proj_grad, 0,
                           atom_embedding_dim * cfg.d_model * sizeof(float),
                           training_state_.stream_ctrl.getPrimaryStream());
        }
        // Text feature projection gradient [16 x d_model]
        if (float* text_proj_grad = scratch_block_layer_->getTextFeatureProjectionGrad()) {
            cudaMemsetAsync(text_proj_grad, 0,
                           kTextFeatureDim * cfg.d_model * sizeof(float),
                           training_state_.stream_ctrl.getPrimaryStream());
        }
    }
    
    // Zeroing is enqueued on the primary stream; no host sync needed here.
}

//======================================================//
//  backward - 3-Phase Backward Pass Wrapper
//======================================================//

void LanguageModel::backward(float loss, bool accumulate, float grad_scale, uint64_t step) {
    if (!training_state_.initialized) {
        BWD_ERROR("[backward] ERROR: Training state not initialized!");
        return;
    }

    ++backward_call_count_;
    last_grad_scale_ = (grad_scale > 0.0f) ? grad_scale : 1.0f;
    
    const auto& cfg = config_;
    const int batch_size = training_state_.cached_batch_size;
    const int seq_len = training_state_.cached_seq_len;

    BWD_INFO("[backward] START batch_size=" << batch_size << " seq_len=" << seq_len 
             << " d_model=" << cfg.d_model << " num_heads=" << cfg.num_heads 
             << " num_kv_heads=" << training_state_.num_kv_heads);
    
    if (batch_size <= 0 || seq_len <= 0) {
        BWD_ERROR("[backward] FATAL: Invalid batch dimensions: batch_size=" << batch_size << " seq_len=" << seq_len);
        return;
    }

    // ═══════════════════════════════════════════════════════════════════════════
    //  PURE AUTOGRAD BACKWARD (January 2026)
    //  
    //  Uses TensorContract autograd for ENTIRE backward pass:
    //  - loss.backward() propagates through entire computation graph
    //  - Gradients automatically flow to all parameter Tensors
    //  - No raw float* gradient manipulation required
    //  
    //  The legacy 3-phase backward system has been DELETED.
    // ═══════════════════════════════════════════════════════════════════════════
    
    cudaStream_t stream = training_state_.stream_ctrl.getPrimaryStream();
    
    // RULE 20: Fail loud - loss tensor MUST have grad_fn from forward pass
    if (!training_state_.loss_tensor.grad_fn) {
        throw std::runtime_error("[backward] loss_tensor.grad_fn is NULL - forward pass (computeLossBatch) MUST be called first!");
    }
    
    BWD_INFO("[backward] Executing pure autograd backward pass");
    
    // Zero gradients if not accumulating
    // The autograd system's executeAutogradBackward handles this, but we also need to zero
    // any raw gradient buffers that might still be referenced by the optimizer
    // 
    // beta_accum decision:
    //   accumulate=false → zeroGrad() called → beta_accum effectively 0 (start fresh)
    //   accumulate=true  → skip zeroGrad()   → beta_accum effectively 1 (add to existing)
    const float effective_beta = accumulate ? 1.0f : 0.0f;
    BWD_INFO("[backward] beta_accum decision: accumulate=" << (accumulate ? "true" : "false") 
             << " → effective_beta=" << effective_beta 
             << (accumulate ? " (ACCUMULATING to existing gradients)" : " (OVERWRITING with fresh gradients)"));
    
    if (!accumulate) {
        // Autograd will zero Tensor.grad fields, but we also zero legacy buffers
        // for any code paths that still expect them
        zeroGrad();  
    }
    
    // Issue #87 DEBUG: Verify grad buffer addresses match between zeroGrad and autograd
    BWD_INFO("[backward] GRAD BUFFER VERIFICATION:");
    BWD_INFO("[backward]   ts.embedding_weights.grad_data() = " << (void*)training_state_.embedding_weights.grad_data());
    BWD_INFO("[backward]   ts.lm_head_weights.grad_data()   = " << (void*)training_state_.lm_head_weights.grad_data());
    BWD_INFO("[backward]   (both should be IDENTICAL if weight tying is correct)");
    
    // Call backward on the loss tensor from forward pass
    // The grad_fn chain is already attached - this propagates through entire graph:
    // loss → logits → encoder_output → encoder_layers → embeddings
    training_state_.loss_tensor.backward(nullptr);
    
    // ═══════════════════════════════════════════════════════════════════════════
    //  ISSUE #71 FIX: NumericHead backward was NEVER CALLED!
    //  
    //  The numeric head forward creates ctx.numeric_head_output with a grad_fn.
    //  The numeric loss kernel (launchNumericLoss) computes grad_predictions and
    //  stores them in training_state_.grad_numeric_tensor.data. BUT this gradient
    //  was never backpropagated through the NumericHead autograd graph!
    //  
    //  Result: numeric_head_weights received ZERO gradients.
    //  
    //  Fix: Call numeric_head_output.backward() with the gradients from launchNumericLoss.
    // ═══════════════════════════════════════════════════════════════════════════
    if (config_.numeric_head_enabled && 
        training_state_.autograd_ctx && 
        training_state_.autograd_ctx->numeric_head_output.data &&
        training_state_.autograd_ctx->numeric_head_output.grad_fn &&
        training_state_.grad_numeric_tensor.data) {
        
        BWD_INFO("[backward] Issue #71 FIX: Calling numeric_head_output.backward() with numeric loss gradients");
        
        // Create a Tensor wrapper around the gradient from launchNumericLoss
        // This gradient is d(numeric_loss)/d(predictions) already computed by the kernel
        Tensor numeric_grad;
        numeric_grad.data = training_state_.grad_numeric_tensor.data;
        numeric_grad.shape = training_state_.autograd_ctx->numeric_head_output.shape;
        numeric_grad.owns_data = false;  // grad_numeric_tensor owns the memory
        numeric_grad.stream = stream;
        
        // Call backward on the numeric head output with the loss gradients
        // This propagates through NumericHeadGradFn::apply() to compute:
        //   - grad_numeric_head_weights = encoder_output^T @ grad_predictions
        //   - grad_encoder_output += weights @ grad_predictions (accumulated to encoder path)
        training_state_.autograd_ctx->numeric_head_output.backward(&numeric_grad);
        
        BWD_INFO("[backward] Issue #71: numeric_head_output.backward() completed");
    }
    
    // ISSUE #62: Sync stream after backward to ensure all async CUDA operations complete
    // before any GradFn destructors run (which might free buffers still in use)
    cudaStreamSynchronize(stream);
    // NOTE: Removed "[backward] Stream synced" fprintf - runs every batch, no value
    
    // Issue #60 DEBUG: Log gradient attribution for debugging token 277 mode collapse
    // This shows LM head vs embedding backward contributions separately
    if (training_state_.debug_gradient_attribution) {
        training_state_.logGradientAttribution(static_cast<int>(step), stream);
    }
    
    // Apply gradient scaling if needed
    if (grad_scale != 1.0f) {
        // Scale all parameter gradients by grad_scale
        const float scale = grad_scale;
        auto* ts = &training_state_;
        
        // Helper lambda to scale a Tensor's gradient
        // ISSUE #59: Use has_grad() and grad_data() accessors
        auto scaleGrad = [stream, scale](Tensor& t) {
            if (t.has_grad() && t.numel() > 0) {
                GRIM::scaleDeviceBuffer(t.grad_data(), t.numel(), scale, stream);
            }
        };
        
        scaleGrad(ts->embedding_weights);
        scaleGrad(ts->position_embedding_weights);
        scaleGrad(ts->lm_head_weights);
        scaleGrad(ts->lm_head_bias);
        scaleGrad(ts->final_rms_gamma);
        
        // NOTE: Encoder layer gradients are in encoder's internal Tensors.
        // The optimizer accesses them via enc->getAttnWqkvGrad() etc.
        // Gradient scaling for encoder is handled by the autograd system.
    }
    
    // ALWAYS clear sequence weights after backward
    training_state_.sequence_weight_count = 0;
    
    // Issue #47: Clear autograd context to free GPU memory from intermediate tensors
    // The next computeLossBatch() will create a new context anyway
    if (training_state_.autograd_ctx) {
        training_state_.autograd_ctx->clearIntermediates();
    }
    
    BWD_INFO("[backward] AUTOGRAD COMPLETE");
}


//======================================================//
//  buildParameterGroups - Initialize Parameter Groups for Optimizer
//======================================================//

void LanguageModel::buildParameterGroups() {
    // Clear parameter group metadata (optimizer states are managed by TrainingState)
    parameter_groups_.clear();
    
    const auto& cfg = config_;
    const int head_dim = cfg.head_dim;  // Use pre-computed value from config
    const int num_kv_heads = training_state_.num_kv_heads;
    const int kv_dim = num_kv_heads * head_dim;
    const int total_qkv_dim = cfg.d_model + 2 * kv_dim;
    
    // Validate GQA dimensions using TensorContract
    TensorContract::GQADims gqa_dims{cfg.num_heads, num_kv_heads, head_dim};
    if (!gqa_dims.is_valid()) {
        BWD_ERROR("[buildParameterGroups] TensorContract GQA validation failed!");
    }
    const int expected_qkv = gqa_dims.total_qkv_dim();
    if (expected_qkv != total_qkv_dim) {
        BWD_ERROR("[buildParameterGroups] QKV dimension mismatch: computed=" << total_qkv_dim 
                  << " expected=" << expected_qkv);
    }

    GPUGrimEncoder* gpu_encoder = &getGpuEncoder();
    if (!gpu_encoder) {
        BWD_ERROR("[buildParameterGroups] GPU encoder not initialized");
        return;
    }

    // Embedding weights - SKIP when tied (handled by LM head group below)
    // When tie_embeddings=true, embedding grad buffer == lm_head grad buffer (same pointer)
    // Registering both would double-count gradients and corrupt optimizer state
    EmbeddingRuntime* embedding_runtime = &getGpuEmbedder();
    float* emb_grads = training_state_.embedding_weights.grad_data();
    if (!cfg.tie_embeddings && emb_grads && embedding_runtime) {
        ParameterGroup emb_group;
        emb_group.name = "embedding";
        emb_group.weights = embedding_runtime->token_buffer;
        emb_group.grads = emb_grads;
        emb_group.size = cfg.vocab_size * cfg.d_model;  // Tensor API - use config directly
        emb_group.m_state = nullptr;
        emb_group.v_state = nullptr;
        emb_group.type = ParamGroupType::EMBEDDING;
        parameter_groups_.push_back(emb_group);
    }

    // ==========================================================================
    // Issue #36 FIX: Position embeddings MUST be trainable to match PyTorch
    // PyTorch baseline: pos_emb = nn.Embedding(seq_len, d_model) - gets gradients
    // GRIM was missing this → position embeddings were FROZEN at random init!
    // ==========================================================================
    float* pos_emb_grads = training_state_.position_embedding_weights.grad_data();
    if (pos_emb_grads && embedding_runtime && embedding_runtime->position_buffer) {
        ParameterGroup pos_emb_group;
        pos_emb_group.name = "position_embedding";
        pos_emb_group.weights = embedding_runtime->position_buffer;
        pos_emb_group.grads = pos_emb_grads;
        pos_emb_group.size = cfg.max_seq_len * cfg.d_model;  // Tensor API - use config directly
        pos_emb_group.m_state = nullptr;
        pos_emb_group.v_state = nullptr;
        pos_emb_group.type = ParamGroupType::EMBEDDING;  // Same treatment as token embeddings
        parameter_groups_.push_back(pos_emb_group);
        std::cout << "✅ Position embeddings registered with optimizer (Issue #36 FIX)" << std::endl;
    }

    // LM head weights (includes tied embedding grads when tie_embeddings=true)
    float* lm_weight_grads = training_state_.lm_head_weights.grad_data();
    if (lm_weight_grads && training_state_.lm_head_weights.data) {
        ParameterGroup lm_group;
        lm_group.name = cfg.tie_embeddings ? "embedding_lm_head_tied" : "lm_head_weight";
        lm_group.weights = training_state_.lm_head_weights.data;  // Tensor API
        lm_group.grads = lm_weight_grads;
        lm_group.size = cfg.vocab_size * cfg.d_model;
        lm_group.m_state = nullptr;
        lm_group.v_state = nullptr;
        // When tie_embeddings=true: This buffer serves as BOTH lm_head AND embedding weights
        //   - Receives dense gradients from LM head backward pass (output layer)
        //   - Receives sparse gradients from embedding backward pass (input layer, accumulated)
        //   - Gradient norm reported under LM_HEAD type (physical buffer name)
        //   - Gradient norm reported under LM_HEAD type (physical buffer name)
        // When tie_embeddings=false: This buffer is ONLY the lm_head projection weights
        //   - Embedding weights are separate and registered as EMBEDDING type above
        lm_group.type = ParamGroupType::LM_HEAD;
        parameter_groups_.push_back(lm_group);
    }

    // Numeric head weights (Tensor API)
    // ISSUE #59: Use has_grad() and grad_data() accessors
    if (cfg.numeric_head_enabled && training_state_.numeric_head_weights.has_grad() &&
        training_state_.numeric_head_weights.data) {
        ParameterGroup numeric_group;
        numeric_group.name = "numeric_head_weight";
        numeric_group.weights = training_state_.numeric_head_weights.data;  // Tensor API
        numeric_group.grads = training_state_.numeric_head_weights.grad_data();    // Tensor API
        numeric_group.size = cfg.d_model;
        numeric_group.m_state = nullptr;
        numeric_group.v_state = nullptr;
        numeric_group.type = ParamGroupType::NUMERIC_HEAD;
        parameter_groups_.push_back(numeric_group);
    }
    
    // Learned loss weighting parameters (log-variance for text and numeric losses)
    if (training_state_.log_var_text.has_grad() && training_state_.log_var_text.data) {
        ParameterGroup log_var_text_group;
        log_var_text_group.name = "log_var_text";
        log_var_text_group.weights = training_state_.log_var_text.data;
        log_var_text_group.grads = training_state_.log_var_text.grad_data();
        log_var_text_group.size = 1;
        log_var_text_group.m_state = nullptr;
        log_var_text_group.v_state = nullptr;
        log_var_text_group.type = ParamGroupType::NUMERIC_HEAD;  // Reuse type for scalar params
        parameter_groups_.push_back(log_var_text_group);
    }
    if (training_state_.log_var_numeric.has_grad() && training_state_.log_var_numeric.data) {
        ParameterGroup log_var_numeric_group;
        log_var_numeric_group.name = "log_var_numeric";
        log_var_numeric_group.weights = training_state_.log_var_numeric.data;
        log_var_numeric_group.grads = training_state_.log_var_numeric.grad_data();
        log_var_numeric_group.size = 1;
        log_var_numeric_group.m_state = nullptr;
        log_var_numeric_group.v_state = nullptr;
        log_var_numeric_group.type = ParamGroupType::NUMERIC_HEAD;  // Reuse type for scalar params
        parameter_groups_.push_back(log_var_numeric_group);
    }

    // ========================================================================
    // Issue #66 FIX: Verify gradient buffer initialization before registration
    // ========================================================================
    // CRITICAL: Encoder layers MUST call ensure_grad() during initialization
    // to allocate gradient buffers. If gradients are nullptr here, we'll crash
    // when optimizer tries to update them. Add diagnostic logging to catch this.
    
    // Per-layer parameters
    for (int layer = 0; layer < cfg.num_layers; ++layer) {
        GPUEncoderLayer* enc = gpu_encoder->getLayer(layer);
        if (!enc) {
            BWD_WARN("[buildParameterGroups] Layer " << layer << " is NULL - skipping");
            continue;
        }

        // QKV weights - uses encoder's Tensor.grad (autograd migration)
        float* qkv_grad_ptr = enc->getAttnWqkvGrad();
        float* qkv_weight_ptr = enc->getAttnWqkv();
        

        
        if (qkv_grad_ptr && qkv_weight_ptr) {
            ParameterGroup qkv_group;
            qkv_group.name = "layer" + std::to_string(layer) + "_qkv_weight";
            qkv_group.weights = qkv_weight_ptr;
            qkv_group.grads = qkv_grad_ptr;
            qkv_group.size = total_qkv_dim * cfg.d_model;
            qkv_group.m_state = nullptr;
            qkv_group.v_state = nullptr;
            qkv_group.type = ParamGroupType::ATTENTION;
            qkv_group.layer_index = layer;
            qkv_group.upsilon = HyperParameters::UPSILON_BASE * 
                sqrtf(static_cast<float>(HyperParameters::UPSILON_REFERENCE_LAYERS) / static_cast<float>(layer + 1));
            parameter_groups_.push_back(qkv_group);
        } else {
            // ISSUE #110: This warning explains WHY encoder groups are being skipped
            fprintf(stderr, "[buildParameterGroups] WARNING: Layer %d QKV SKIPPED! weights=%p grads=%p (not initialized)\n",
                    layer, (void*)qkv_weight_ptr, (void*)qkv_grad_ptr);
        }

        // Issue #97 FIX: b_qkv bias - previously NOT in optimizer (frozen at init!)
        float* bqkv_grad_ptr = enc->getAttnBqkvGrad();
        float* bqkv_weight_ptr = enc->getAttnBqkv();
        if (bqkv_grad_ptr && bqkv_weight_ptr) {
            ParameterGroup bqkv_group;
            bqkv_group.name = "layer" + std::to_string(layer) + "_qkv_bias";
            bqkv_group.weights = bqkv_weight_ptr;
            bqkv_group.grads = bqkv_grad_ptr;
            bqkv_group.size = total_qkv_dim;  // b_qkv: [total_qkv_dim]
            bqkv_group.m_state = nullptr;
            bqkv_group.v_state = nullptr;
            bqkv_group.type = ParamGroupType::ATTENTION;
            bqkv_group.layer_index = layer;
            bqkv_group.upsilon = HyperParameters::UPSILON_BASE * 
                sqrtf(static_cast<float>(HyperParameters::UPSILON_REFERENCE_LAYERS) / static_cast<float>(layer + 1));
            parameter_groups_.push_back(bqkv_group);
        }

        // Wo weights - uses encoder's Tensor.grad (autograd migration)
        float* wo_grad_ptr = enc->getAttnWoGrad();
        float* wo_weight_ptr = enc->getAttnWo();
        if (wo_grad_ptr && wo_weight_ptr) {
            ParameterGroup wo_group;
            wo_group.name = "layer" + std::to_string(layer) + "_wo_weight";
            wo_group.weights = wo_weight_ptr;
            wo_group.grads = wo_grad_ptr;
            wo_group.size = cfg.d_model * cfg.d_model;
            wo_group.m_state = nullptr;
            wo_group.v_state = nullptr;
            wo_group.type = ParamGroupType::ATTENTION;
            wo_group.layer_index = layer;
            wo_group.upsilon = HyperParameters::UPSILON_BASE * 
                sqrtf(static_cast<float>(HyperParameters::UPSILON_REFERENCE_LAYERS) / static_cast<float>(layer + 1));
            parameter_groups_.push_back(wo_group);
        } else {
            BWD_WARN("[buildParameterGroups] Layer " << layer << " Wo: weights=" << (void*)wo_weight_ptr 
                     << " grads=" << (void*)wo_grad_ptr << " (skipping - not initialized)");
        }

        // Issue #97 FIX: b_o bias - previously NOT in optimizer (frozen at init!)
        float* bo_grad_ptr = enc->getAttnBoGrad();
        float* bo_weight_ptr = enc->getAttnBo();
        if (bo_grad_ptr && bo_weight_ptr) {
            ParameterGroup bo_group;
            bo_group.name = "layer" + std::to_string(layer) + "_wo_bias";
            bo_group.weights = bo_weight_ptr;
            bo_group.grads = bo_grad_ptr;
            bo_group.size = cfg.d_model;  // b_o: [d_model]
            bo_group.m_state = nullptr;
            bo_group.v_state = nullptr;
            bo_group.type = ParamGroupType::ATTENTION;
            bo_group.layer_index = layer;
            bo_group.upsilon = HyperParameters::UPSILON_BASE * 
                sqrtf(static_cast<float>(HyperParameters::UPSILON_REFERENCE_LAYERS) / static_cast<float>(layer + 1));
            parameter_groups_.push_back(bo_group);
        }

        // FFN W1 - uses encoder's Tensor.grad (autograd migration)
        float* w1_grad_ptr = enc->getFFNW1Grad();
        float* w1_weight_ptr = enc->getFFNW1();
        if (w1_grad_ptr && w1_weight_ptr) {
            ParameterGroup w1_group;
            w1_group.name = "layer" + std::to_string(layer) + "_ffn_w1";
            w1_group.weights = w1_weight_ptr;
            w1_group.grads = w1_grad_ptr;
            w1_group.size = cfg.d_model * cfg.d_ff;
            w1_group.m_state = nullptr;
            w1_group.v_state = nullptr;
            w1_group.type = ParamGroupType::FFN;
            w1_group.layer_index = layer;
            w1_group.upsilon = HyperParameters::UPSILON_BASE * 
                sqrtf(static_cast<float>(HyperParameters::UPSILON_REFERENCE_LAYERS) / static_cast<float>(layer + 1));
            parameter_groups_.push_back(w1_group);
        } else {
            BWD_WARN("[buildParameterGroups] Layer " << layer << " W1: weights=" << (void*)w1_weight_ptr 
                     << " grads=" << (void*)w1_grad_ptr << " (skipping - not initialized)");
        }

        // Issue #97 FIX: b1 bias - previously NOT in optimizer (frozen at init!)
        float* b1_grad_ptr = enc->getFFNB1Grad();
        float* b1_weight_ptr = enc->getFFNB1();
        if (b1_grad_ptr && b1_weight_ptr) {
            ParameterGroup b1_group;
            b1_group.name = "layer" + std::to_string(layer) + "_ffn_b1";
            b1_group.weights = b1_weight_ptr;
            b1_group.grads = b1_grad_ptr;
            b1_group.size = cfg.d_ff;  // b1: [d_ff]
            b1_group.m_state = nullptr;
            b1_group.v_state = nullptr;
            b1_group.type = ParamGroupType::FFN;
            b1_group.layer_index = layer;
            b1_group.upsilon = HyperParameters::UPSILON_BASE * 
                sqrtf(static_cast<float>(HyperParameters::UPSILON_REFERENCE_LAYERS) / static_cast<float>(layer + 1));
            parameter_groups_.push_back(b1_group);
        }

        // FFN W2 - uses encoder's Tensor.grad (autograd migration)
        float* w2_grad_ptr = enc->getFFNW2Grad();
        float* w2_weight_ptr = enc->getFFNW2();
        if (w2_grad_ptr && w2_weight_ptr) {
            ParameterGroup w2_group;
            w2_group.name = "layer" + std::to_string(layer) + "_ffn_w2";
            w2_group.weights = w2_weight_ptr;
            w2_group.grads = w2_grad_ptr;
            w2_group.size = cfg.d_ff * cfg.d_model;
            w2_group.m_state = nullptr;
            w2_group.v_state = nullptr;
            w2_group.type = ParamGroupType::FFN;
            w2_group.layer_index = layer;
            w2_group.upsilon = HyperParameters::UPSILON_BASE * 
                sqrtf(static_cast<float>(HyperParameters::UPSILON_REFERENCE_LAYERS) / static_cast<float>(layer + 1));
            parameter_groups_.push_back(w2_group);
        } else {
            BWD_WARN("[buildParameterGroups] Layer " << layer << " W2: weights=" << (void*)w2_weight_ptr 
                     << " grads=" << (void*)w2_grad_ptr << " (skipping - not initialized)");
        }

        // Issue #97 FIX: b2 bias - previously NOT in optimizer (frozen at init!)
        float* b2_grad_ptr = enc->getFFNB2Grad();
        float* b2_weight_ptr = enc->getFFNB2();
        if (b2_grad_ptr && b2_weight_ptr) {
            ParameterGroup b2_group;
            b2_group.name = "layer" + std::to_string(layer) + "_ffn_b2";
            b2_group.weights = b2_weight_ptr;
            b2_group.grads = b2_grad_ptr;
            b2_group.size = cfg.d_model;  // b2: [d_model]
            b2_group.m_state = nullptr;
            b2_group.v_state = nullptr;
            b2_group.type = ParamGroupType::FFN;
            b2_group.layer_index = layer;
            b2_group.upsilon = HyperParameters::UPSILON_BASE * 
                sqrtf(static_cast<float>(HyperParameters::UPSILON_REFERENCE_LAYERS) / static_cast<float>(layer + 1));
            parameter_groups_.push_back(b2_group);
        }

        // RMSNorm gamma 1
        float* rms1_grad_ptr = enc->getRMS1GammaGrad();
        float* rms1_weight_ptr = enc->getRMS1Gamma();
        if (rms1_grad_ptr && rms1_weight_ptr) {
            ParameterGroup rms1_group;
            rms1_group.name = "layer" + std::to_string(layer) + "_rms1_gamma";
            rms1_group.weights = rms1_weight_ptr;
            rms1_group.grads = rms1_grad_ptr;
            rms1_group.size = cfg.d_model;
            rms1_group.m_state = nullptr;
            rms1_group.v_state = nullptr;
            rms1_group.type = ParamGroupType::RMSNORM;
            rms1_group.layer_index = layer;
            rms1_group.upsilon = HyperParameters::UPSILON_BASE * 
                sqrtf(static_cast<float>(HyperParameters::UPSILON_REFERENCE_LAYERS) / static_cast<float>(layer + 1));
            parameter_groups_.push_back(rms1_group);
        } else {
            BWD_WARN("[buildParameterGroups] Layer " << layer << " RMS1: weights=" << (void*)rms1_weight_ptr 
                     << " grads=" << (void*)rms1_grad_ptr << " (skipping - not initialized)");
        }

        // RMSNorm gamma 2
        float* rms2_grad_ptr = enc->getRMS2GammaGrad();
        float* rms2_weight_ptr = enc->getRMS2Gamma();
        if (rms2_grad_ptr && rms2_weight_ptr) {
            ParameterGroup rms2_group;
            rms2_group.name = "layer" + std::to_string(layer) + "_rms2_gamma";
            rms2_group.weights = rms2_weight_ptr;
            rms2_group.grads = rms2_grad_ptr;
            rms2_group.size = cfg.d_model;
            rms2_group.m_state = nullptr;  // Will be set after TrainingState allocation
            rms2_group.v_state = nullptr;
            rms2_group.type = ParamGroupType::RMSNORM;
            rms2_group.layer_index = layer;
            rms2_group.upsilon = HyperParameters::UPSILON_BASE * 
                sqrtf(static_cast<float>(HyperParameters::UPSILON_REFERENCE_LAYERS) / static_cast<float>(layer + 1));
            parameter_groups_.push_back(rms2_group);
        } else {
            BWD_WARN("[buildParameterGroups] Layer " << layer << " RMS2: weights=" << (void*)rms2_weight_ptr 
                     << " grads=" << (void*)rms2_grad_ptr << " (skipping - not initialized)");
        }
    }

    // ScratchBlock atom embeddings and projection (Rule 22: use layer's internal weight/grad buffers)
    // These parameters are trainable when scratch_block is enabled - they learn atom type representations
    // that help the model reason about structural elements (numbers, URLs, emails, etc.)
    if (scratch_block_layer_ && scratch_block_layer_->isEnabled()) {
        // Atom type embeddings: [NUM_ATOM_TYPES, atom_embedding_dim]
        constexpr int NUM_ATOM_TYPES = HyperParameters::NUM_ATOM_TYPES;
        const int atom_embedding_dim = cfg.scratch_block_atom_embedding_dim;
        
        if (scratch_block_layer_->getAtomTypeEmbeddings() && 
            scratch_block_layer_->getAtomTypeEmbeddingsGrad()) {
            ParameterGroup atom_emb_group;
            atom_emb_group.name = "scratch_block_atom_type_embeddings";
            atom_emb_group.weights = scratch_block_layer_->getAtomTypeEmbeddings();
            atom_emb_group.grads = scratch_block_layer_->getAtomTypeEmbeddingsGrad();
            atom_emb_group.size = NUM_ATOM_TYPES * atom_embedding_dim;
            atom_emb_group.m_state = nullptr;
            atom_emb_group.v_state = nullptr;
            atom_emb_group.type = ParamGroupType::SCRATCHBLOCK;
            parameter_groups_.push_back(atom_emb_group);
        }
        
        // Atom projection: [atom_embedding_dim, d_model] = [64, 768] = 49152 params
        if (scratch_block_layer_->getAtomProjection() && 
            scratch_block_layer_->getAtomProjectionGrad()) {
            ParameterGroup atom_proj_group;
            atom_proj_group.name = "scratch_block_atom_projection";
            atom_proj_group.weights = scratch_block_layer_->getAtomProjection();
            atom_proj_group.grads = scratch_block_layer_->getAtomProjectionGrad();
            atom_proj_group.size = atom_embedding_dim * cfg.d_model;
            atom_proj_group.m_state = nullptr;
            atom_proj_group.v_state = nullptr;
            atom_proj_group.type = ParamGroupType::SCRATCHBLOCK;
            parameter_groups_.push_back(atom_proj_group);
        }
        
        // Text feature projection: [16, d_model] = [16, 768] = 12288 params
        // This is the VALUE encoding path - text_features encode atom semantics
        constexpr int kTextFeatureDim = 16;
        if (scratch_block_layer_->getTextFeatureProjection() && 
            scratch_block_layer_->getTextFeatureProjectionGrad()) {
            ParameterGroup text_proj_group;
            text_proj_group.name = "scratch_block_text_feature_projection";
            text_proj_group.weights = scratch_block_layer_->getTextFeatureProjection();
            text_proj_group.grads = scratch_block_layer_->getTextFeatureProjectionGrad();
            text_proj_group.size = kTextFeatureDim * cfg.d_model;
            text_proj_group.m_state = nullptr;
            text_proj_group.v_state = nullptr;
            text_proj_group.type = ParamGroupType::SCRATCHBLOCK;
            parameter_groups_.push_back(text_proj_group);
        }
        
        BWD_INFO("[buildParameterGroups] Registered ScratchBlock: "
                 << "atom_emb=" << (NUM_ATOM_TYPES * atom_embedding_dim)
                 << " atom_proj=" << (atom_embedding_dim * cfg.d_model)
                 << " text_proj=" << (kTextFeatureDim * cfg.d_model)
                 << " total=" << (NUM_ATOM_TYPES * atom_embedding_dim + atom_embedding_dim * cfg.d_model + kTextFeatureDim * cfg.d_model));
    }

    // Issue #33: Final RMSNorm layer (between encoder output and LM head)
    // This normalizes encoder output variance to prevent logit scale explosion.
    // Standard transformer architecture: embedding → encoder → FINAL_NORM → lm_head
    // Without this, variance grows unboundedly through residual connections.
    float* final_rms_grads = training_state_.final_rms_gamma.grad_data();
    if (training_state_.final_rms_gamma.data && final_rms_grads) {
        ParameterGroup final_rms_group;
        final_rms_group.name = "final_rms_gamma";
        final_rms_group.weights = training_state_.final_rms_gamma.data;  // Tensor API
        final_rms_group.grads = final_rms_grads;                         // Tensor API accessor
        final_rms_group.size = cfg.d_model;
        final_rms_group.m_state = nullptr;
        final_rms_group.v_state = nullptr;
        final_rms_group.type = ParamGroupType::RMSNORM;
        parameter_groups_.push_back(final_rms_group);
        BWD_INFO("[buildParameterGroups] Registered final_rms_gamma: size=" << cfg.d_model);
    }

    // Collect sizes for centralized optimizer state allocation
    std::vector<size_t> sizes;
    sizes.reserve(parameter_groups_.size());
    for (const auto& group : parameter_groups_) {
        sizes.push_back(group.size);
    }
    
    // Allocate optimizer states via TrainingState (centralized ownership)
    training_state_.allocateOptimizerStates(sizes);
    
    // Bind pointers back to parameter groups (groups hold pointers, NOT ownership)
    for (size_t i = 0; i < parameter_groups_.size(); ++i) {
        parameter_groups_[i].m_state = training_state_.optimizer_m_states[i].data;
        parameter_groups_[i].v_state = training_state_.optimizer_v_states[i].data;
    }
    
    // Note: Gradient norm computation moved to TrainingState::gradnorm_ctrl (GradNormController)
    // Old d_grad_norm_sums_/h_grad_norm_sums_ buffers removed per Rule 20 (no backwards compatibility)

    BWD_INFO("[buildParameterGroups] Built " << parameter_groups_.size() << " parameter groups");
    
    // ISSUE #110 DIAGNOSTIC: Count parameter groups by type to understand what was registered
    int emb_count = 0, attn_count = 0, ffn_count = 0, rms_count = 0, other_count = 0;
    for (const auto& pg : parameter_groups_) {
        switch (pg.type) {
            case ParamGroupType::EMBEDDING: emb_count++; break;
            case ParamGroupType::ATTENTION: attn_count++; break;
            case ParamGroupType::FFN: ffn_count++; break;
            case ParamGroupType::RMSNORM: rms_count++; break;
            default: other_count++; break;
        }
    }
    fprintf(stderr, "[buildParameterGroups] TOTAL: %zu groups (emb=%d, attn=%d, ffn=%d, rms=%d, other=%d)\n",
            parameter_groups_.size(), emb_count, attn_count, ffn_count, rms_count, other_count);
}

//======================================================//
//  configureUpdateProbe / disableUpdateProbe
//======================================================//

void LanguageModel::configureUpdateProbe(const std::string& group_name, size_t sample_elems) {
    update_probe_group_name_ = group_name;
    update_probe_sample_elems_ = sample_elems;
    update_probe_ready_ = false;
    
    // Ensure parameter groups are built before configuring probe
    if (parameter_groups_.empty()) {
        buildParameterGroups();
    }
    
    if (parameter_groups_.empty()) {
        update_probe_group_index_ = static_cast<size_t>(-1);
        update_probe_weights_before_.clear();
        update_probe_weights_after_.clear();
        update_probe_grad_sample_.clear();
        return;
    }

    update_probe_group_index_ = static_cast<size_t>(-1);
    for (size_t i = 0; i < parameter_groups_.size(); ++i) {
        if (parameter_groups_[i].name == update_probe_group_name_) {
            update_probe_group_index_ = i;
            break;
        }
    }

    if (update_probe_group_index_ == static_cast<size_t>(-1)) {
        BWD_WARN("[configureUpdateProbe] target group '" << update_probe_group_name_
                 << "' not found");
        return;
    }

    const auto& probe_group = parameter_groups_[update_probe_group_index_];
    if (update_probe_sample_elems_ == 0 || update_probe_sample_elems_ > probe_group.size) {
        update_probe_sample_elems_ = std::min<size_t>(probe_group.size, 2048);
    }
    update_probe_weights_before_.assign(update_probe_sample_elems_, 0.0f);
    update_probe_weights_after_.assign(update_probe_sample_elems_, 0.0f);
    update_probe_grad_sample_.assign(update_probe_sample_elems_, 0.0f);
}

void LanguageModel::disableUpdateProbe() {
    update_probe_group_name_.clear();
    update_probe_group_index_ = static_cast<size_t>(-1);
    update_probe_sample_elems_ = 0;
    update_probe_weights_before_.clear();
    update_probe_weights_after_.clear();
    update_probe_grad_sample_.clear();
    update_probe_ready_ = false;
}

//======================================================//
//  recordGradientClip
//======================================================//

void LanguageModel::recordGradientClip(float clip_threshold, bool clipped) {
    last_grad_clip_limit_ = clip_threshold;
    grad_metrics_.clip_threshold = clip_threshold;
    grad_metrics_.clipped = clipped;
}

//======================================================//
//  Token 277 (SPACE) Weight Tracking Diagnostic
//  Tracks the W[277] row in embedding/LM head weights before/after optimizer
//======================================================//
static int s_token277_track_count = 0;

static void logToken277WeightUpdate(
    const float* weights_gpu,     // embedding/LM head weights on GPU
    const float* grads_gpu,       // gradients for the weights
    const float* m_state_gpu,     // AdamW first moment
    const float* v_state_gpu,     // AdamW second moment
    int d_model,
    int vocab_size,
    int optimizer_step,
    const char* phase,            // "BEFORE" or "AFTER"
    cudaStream_t stream
) {
    constexpr int kToken277 = 277;
    // DISABLED for performance (User request to remove printf syncs)
    return;
    
    if (kToken277 >= vocab_size) return;
    
    ++s_token277_track_count;
    // Only log first 30 calls per run
    if (s_token277_track_count > 30) return;
    
    cudaStreamSynchronize(stream);
    
    // W[277] is at offset kToken277 * d_model in the weight matrix
    // Weight layout: [vocab_size, d_model] row-major
    const size_t row_offset = static_cast<size_t>(kToken277) * d_model;
    
    std::vector<float> w277(d_model);
    std::vector<float> g277(d_model);
    std::vector<float> m277(d_model);
    std::vector<float> v277(d_model);
    
    cudaMemcpy(w277.data(), weights_gpu + row_offset, d_model * sizeof(float), cudaMemcpyDeviceToHost);
    if (grads_gpu) {
        cudaMemcpy(g277.data(), grads_gpu + row_offset, d_model * sizeof(float), cudaMemcpyDeviceToHost);
    }
    if (m_state_gpu) {
        cudaMemcpy(m277.data(), m_state_gpu + row_offset, d_model * sizeof(float), cudaMemcpyDeviceToHost);
    }
    if (v_state_gpu) {
        cudaMemcpy(v277.data(), v_state_gpu + row_offset, d_model * sizeof(float), cudaMemcpyDeviceToHost);
    }
    
    // Compute statistics for W[277]
    float w_sum = 0, w_sq = 0, w_min = w277[0], w_max = w277[0];
    float g_sum = 0, g_sq = 0;
    float m_sum = 0, v_sum = 0;
    
    for (int i = 0; i < d_model; ++i) {
        w_sum += w277[i];
        w_sq += w277[i] * w277[i];
        w_min = std::min(w_min, w277[i]);
        w_max = std::max(w_max, w277[i]);
        g_sum += g277[i];
        g_sq += g277[i] * g277[i];
        m_sum += m277[i];
        v_sum += v277[i];
    }
    
    float w_norm = std::sqrt(w_sq);
    float w_mean = w_sum / d_model;
    float g_norm = std::sqrt(g_sq);
    float g_mean = g_sum / d_model;
    float m_mean = m_sum / d_model;
    float v_mean = v_sum / d_model;
    
    fprintf(stderr, "[Token277Weight] %s step=%d W[277]: norm=%.6f mean=%.6f min=%.6f max=%.6f\n",
            phase, optimizer_step, w_norm, w_mean, w_min, w_max);
    fprintf(stderr, "[Token277Weight] %s step=%d G[277]: norm=%.6f mean=%.6f (grad sum=%.6f)\n",
            phase, optimizer_step, g_norm, g_mean, g_sum);
    if (m_state_gpu && v_state_gpu) {
        fprintf(stderr, "[Token277Weight] %s step=%d M[277]_mean=%.6f V[277]_mean=%.6f\n",
                phase, optimizer_step, m_mean, v_mean);
    }
    
    // Log first 10 values for detailed inspection
    fprintf(stderr, "[Token277Weight] %s W[277][0:10]=[", phase);
    for (int i = 0; i < std::min(10, d_model); ++i) {
        fprintf(stderr, "%.4f%s", w277[i], (i < 9) ? ", " : "");
    }
    fprintf(stderr, "]\n");
    fprintf(stderr, "[Token277Weight] %s G[277][0:10]=[", phase);
    for (int i = 0; i < std::min(10, d_model); ++i) {
        fprintf(stderr, "%.4f%s", g277[i], (i < 9) ? ", " : "");
    }
    fprintf(stderr, "]\n");
}

//======================================================//
//  updateWeights - AdamW Optimizer Step
//======================================================//

void LanguageModel::updateWeights(float learning_rate,
                                  OptimizerState* optimizer_state,
                                  float weight_decay) {
    if (!training_state_.initialized) {
        BWD_ERROR("[updateWeights] ERROR: Training state not initialized!");
        return;
    }
    
    if (!optimizer_state) {
        BWD_ERROR("[updateWeights] ERROR: Optimizer state is null!");
        return;
    }
    
    // NOTE: step is incremented in training loop after this function returns
    // Do NOT increment here to avoid double-counting

    // Rebuild parameter groups if needed
    if (parameter_groups_.empty()) {
        buildParameterGroups();
    }

    const cudaStream_t stream = training_state_.stream_ctrl.getPrimaryStream();
    constexpr size_t kOptimizerIoSample = 10;
    static bool logged_optimizer_io = false;  // Static so we only log once per training run
    size_t sample_count = 0;
    std::string sample_group_name;
    std::array<float, kOptimizerIoSample> w_before{};
    std::array<float, kOptimizerIoSample> g_before{};
    std::array<float, kOptimizerIoSample> m_before{};
    std::array<float, kOptimizerIoSample> v_before{};
    std::array<float, kOptimizerIoSample> w_after{};
    std::array<float, kOptimizerIoSample> m_after{};
    std::array<float, kOptimizerIoSample> v_after{};
    
    // Sample weights before update (for update probe)
    // BUG FIX: Sample from token 277 (SPACE, ~11% of training data) instead of token 0-2
    // Token 0-2 are BYTE TOKENS which rarely appear, causing grad_rms=0.000000 artifacts
    // Token 277 is guaranteed to have meaningful gradients in every batch
    if (update_probe_group_index_ != static_cast<size_t>(-1) && 
        update_probe_group_index_ < parameter_groups_.size()) {
        const auto& probe_group = parameter_groups_[update_probe_group_index_];
        if (probe_group.weights && update_probe_sample_elems_ > 0) {
            // For embedding/LM head buffers [vocab_size, d_model], sample from token 277 row
            // Token 277 offset = 277 * d_model = 277 * 768 = 212736
            constexpr size_t kToken277Offset = 277 * 768;  // Token 277 (SPACE) row start
            const size_t sample_offset = (probe_group.type == ParamGroupType::LM_HEAD || 
                                          probe_group.type == ParamGroupType::EMBEDDING)
                                         ? std::min(kToken277Offset, 
                                                    static_cast<size_t>(probe_group.size) - update_probe_sample_elems_)
                                         : 0;
            
            cudaMemcpyAsync(update_probe_weights_before_.data(), 
                          probe_group.weights + sample_offset, 
                          update_probe_sample_elems_ * sizeof(float),
                          cudaMemcpyDeviceToHost, training_state_.stream_ctrl.getPrimaryStream());
            
            // Store offset for consistent sampling in post-update copy
            update_probe_sample_offset_ = sample_offset;
            
            // One-time log to verify token 277 sampling (only first step)
            if (optimizer_state->step == 0) {
                BWD_INFO("[UpdateProbe] Sampling from offset " << sample_offset 
                         << " (token " << (sample_offset / 768) << " of vocab_size=" 
                         << probe_group.size / 768 << ") for group '" << probe_group.name << "'");
            }
        }
    }
    
    // Issue #110 diagnostic: Log every group being processed on first step
    if (optimizer_state->step == 0) {
        fprintf(stderr, "\n[updateWeights] STEP 0: Processing %zu parameter groups...\n", 
                parameter_groups_.size());
    }
    
    for (size_t i = 0; i < parameter_groups_.size(); ++i) {
        auto& group = parameter_groups_[i];
        if (!group.weights || !group.grads || group.size == 0) continue;
        
        // Issue #110: Log EVERY encoder group being updated on first step
        if (optimizer_state->step == 0) {
            fprintf(stderr, "[updateWeights] Group %zu/%zu: '%s' size=%d weights=%p grads=%p m=%p v=%p\n",
                    i, parameter_groups_.size(), group.name.c_str(), group.size,
                    static_cast<const void*>(group.weights),
                    static_cast<const void*>(group.grads),
                    static_cast<const void*>(group.m_state),
                    static_cast<const void*>(group.v_state));
        }
        
        if (!group.m_state || !group.v_state) {
            BWD_ERROR("[updateWeights] FATAL: Missing optimizer state for group '"
                      << group.name << "' idx=" << i
                      << " size=" << group.size
                      << " weights=" << static_cast<const void*>(group.weights)
                      << " grads=" << static_cast<const void*>(group.grads)
                      << " m_state=" << static_cast<const void*>(group.m_state)
                      << " v_state=" << static_cast<const void*>(group.v_state)
                      << " step=" << optimizer_state->step);
            std::abort();
        }
        
        // === TOKEN 277 TRACKING: Log W[277] BEFORE optimizer for LM head group ===
        const bool is_lm_head_group = (group.type == ParamGroupType::LM_HEAD);
        if (is_lm_head_group) {
            logToken277WeightUpdate(
                group.weights, group.grads, group.m_state, group.v_state,
                config_.d_model, config_.vocab_size, optimizer_state->step,
                "BEFORE", stream);
        }
        
        if (!logged_optimizer_io) {
            sample_count = std::min(kOptimizerIoSample, static_cast<size_t>(group.size));
            sample_group_name = group.name;
            if (sample_count > 0) {
                cudaMemcpyAsync(w_before.data(), group.weights,
                                sample_count * sizeof(float),
                                cudaMemcpyDeviceToHost, stream);
                cudaMemcpyAsync(g_before.data(), group.grads,
                                sample_count * sizeof(float),
                                cudaMemcpyDeviceToHost, stream);
                cudaMemcpyAsync(m_before.data(), group.m_state,
                                sample_count * sizeof(float),
                                cudaMemcpyDeviceToHost, stream);
                cudaMemcpyAsync(v_before.data(), group.v_state,
                                sample_count * sizeof(float),
                                cudaMemcpyDeviceToHost, stream);
                cudaError_t sync_err = cudaStreamSynchronize(stream);
                if (sync_err != cudaSuccess) {
                    BWD_ERROR("[updateWeights] pre-sample sync failed: " << cudaGetErrorString(sync_err));
                }
            }
        }

        // Apply depth-aware upsilon (Υ) regularization to weight decay
        // Formula: Υ_l = 0.1 * sqrt(L_ref / L) where L is 1-indexed layer
        // Deeper layers get LESS regularization (smaller effective weight_decay)
        const float effective_weight_decay = weight_decay * group.upsilon;
        
        launchAdamWKernel(
            group.weights,
            group.grads,
            group.m_state,
            group.v_state,
            group.size,
            learning_rate,
            effective_weight_decay,
            optimizer_state->step,
            stream
        );
        
        // === TOKEN 277 TRACKING: Log W[277] AFTER optimizer for LM head group ===
        if (is_lm_head_group) {
            logToken277WeightUpdate(
                group.weights, group.grads, group.m_state, group.v_state,
                config_.d_model, config_.vocab_size, optimizer_state->step,
                "AFTER", stream);
        }

        if (!logged_optimizer_io && sample_count > 0) {
            cudaMemcpyAsync(w_after.data(), group.weights,
                            sample_count * sizeof(float),
                            cudaMemcpyDeviceToHost, stream);
            cudaMemcpyAsync(m_after.data(), group.m_state,
                            sample_count * sizeof(float),
                            cudaMemcpyDeviceToHost, stream);
            cudaMemcpyAsync(v_after.data(), group.v_state,
                            sample_count * sizeof(float),
                            cudaMemcpyDeviceToHost, stream);
            cudaError_t sync_err = cudaStreamSynchronize(stream);
            if (sync_err != cudaSuccess) {
                BWD_ERROR("[updateWeights] post-sample sync failed: " << cudaGetErrorString(sync_err));
            }

            const int iteration = static_cast<int>(optimizer_state->step) + 1;
            const float bias_correction2 =
                1.0f - std::pow(HyperParameters::ADAMW_BETA2, static_cast<float>(iteration));
            const float inv_bias_correction2 =
                (bias_correction2 > 0.0f) ? (1.0f / bias_correction2) : 1.0f;


            std::ostringstream io_oss;
            io_oss << "[OptIO] group='" << sample_group_name
                   << "' step=" << optimizer_state->step
                   << " lr=" << learning_rate
                   << " weight_decay=" << weight_decay
                   << " n=" << sample_count
                   << " w_before=[";
            for (size_t i = 0; i < sample_count; ++i) {
                if (i) io_oss << ", ";
                io_oss << w_before[i];
            }
            io_oss << "] grad=[";
            for (size_t i = 0; i < sample_count; ++i) {
                if (i) io_oss << ", ";
                io_oss << g_before[i];
            }
            io_oss << "] m_before=[";
            for (size_t i = 0; i < sample_count; ++i) {
                if (i) io_oss << ", ";
                io_oss << m_before[i];
            }
            io_oss << "] v_before=[";
            for (size_t i = 0; i < sample_count; ++i) {
                if (i) io_oss << ", ";
                io_oss << v_before[i];
            }
            io_oss << "] w_after=[";
            for (size_t i = 0; i < sample_count; ++i) {
                if (i) io_oss << ", ";
                io_oss << w_after[i];
            }
            io_oss << "] m_after=[";
            for (size_t i = 0; i < sample_count; ++i) {
                if (i) io_oss << ", ";
                io_oss << m_after[i];
            }
            io_oss << "] v_after=[";
            for (size_t i = 0; i < sample_count; ++i) {
                if (i) io_oss << ", ";
                io_oss << v_after[i];
            }
            io_oss << "] denom=[";
            for (size_t i = 0; i < sample_count; ++i) {
                if (i) io_oss << ", ";
                const float v_hat = v_after[i] * inv_bias_correction2;
                const float denom = std::sqrt(v_hat + HyperParameters::ADAMW_EPSILON);
                io_oss << denom;
            }
            io_oss << "]";
            std::cout << io_oss.str() << std::endl;
            logged_optimizer_io = true;
        
        }
    }
    
    // Issue #110: Log completion of all groups on first step
    if (optimizer_state->step == 0) {
        fprintf(stderr, "[updateWeights] STEP 0 COMPLETE: All %zu groups processed by optimizer!\n\n",
                parameter_groups_.size());
    }
    
    // //old_sync: cudaStreamSynchronize(training_state_.stream_ctrl.getPrimaryStream());
    // Sync removed - update probe disabled, next operation will implicitly sync if needed
    // DEPRECATED: probe-only sync to guarantee update probe samples observe weight updates.
    if (update_probe_group_index_ != static_cast<size_t>(-1) &&
        update_probe_group_index_ < parameter_groups_.size()) {
        cudaError_t sync_err = cudaStreamSynchronize(training_state_.stream_ctrl.getPrimaryStream());
        if (sync_err != cudaSuccess) {
            BWD_ERROR("[updateWeights] probe sync failed: " << cudaGetErrorString(sync_err));
        }
    }
    
    // Sample weights after update and compute probe metrics
    if (update_probe_group_index_ != static_cast<size_t>(-1) && 
        update_probe_group_index_ < parameter_groups_.size()) {
        const auto& probe_group = parameter_groups_[update_probe_group_index_];
        
        if (probe_group.weights && probe_group.grads && update_probe_sample_elems_ > 0) {
            // Copy weights after update (use same offset as pre-update copy)
            cudaMemcpy(update_probe_weights_after_.data(), 
                      probe_group.weights + update_probe_sample_offset_, 
                      update_probe_sample_elems_ * sizeof(float),
                      cudaMemcpyDeviceToHost);
            
            // Copy gradient sample (use same offset to sample from token 277 region)
            cudaMemcpy(update_probe_grad_sample_.data(), 
                      probe_group.grads + update_probe_sample_offset_, 
                      update_probe_sample_elems_ * sizeof(float),
                      cudaMemcpyDeviceToHost);

            // Log raw samples that feed update_rms/rel/max_abs
            {
                constexpr size_t kProbeValueSamples = 4;
                const size_t value_samples = std::min(kProbeValueSamples, update_probe_sample_elems_);
                std::ostringstream val_oss;
                val_oss << "[UpdateProbeValues] group='" << probe_group.name
                        << "' step=" << optimizer_state->step
                        << " w_before[0:" << value_samples << "]=[";
                for (size_t i = 0; i < value_samples; ++i) {
                    if (i) val_oss << ", ";
                    val_oss << update_probe_weights_before_[i];
                }
                val_oss << "] w_after[0:" << value_samples << "]=[";
                for (size_t i = 0; i < value_samples; ++i) {
                    if (i) val_oss << ", ";
                    val_oss << update_probe_weights_after_[i];
                }
                val_oss << "] grad[0:" << value_samples << "]=[";
                for (size_t i = 0; i < value_samples; ++i) {
                    if (i) val_oss << ", ";
                    val_oss << update_probe_grad_sample_[i];
                }
                val_oss << "] update[0:" << value_samples << "]=[";
                for (size_t i = 0; i < value_samples; ++i) {
                    if (i) val_oss << ", ";
                    val_oss << (update_probe_weights_after_[i] - update_probe_weights_before_[i]);
                }
                val_oss << "]";
                std::cout << val_oss.str() << std::endl;
            }

            // Log a small sample of optimizer state (m/v) for the probe group
            // BUG FIX: Use SAME offset as weights/grads for consistent sampling!
            // Previously was sampling m_state[0] while grad was from token 277 (offset 212736)
            if (probe_group.m_state && probe_group.v_state) {
                constexpr size_t kProbeStateSamples = 4;
                const size_t state_samples = std::min(kProbeStateSamples, update_probe_sample_elems_);
                std::array<float, kProbeStateSamples> m_sample{};
                std::array<float, kProbeStateSamples> v_sample{};
                // Use update_probe_sample_offset_ to read from SAME position as weights/grads
                cudaMemcpy(m_sample.data(), probe_group.m_state + update_probe_sample_offset_,
                           state_samples * sizeof(float), cudaMemcpyDeviceToHost);
                cudaMemcpy(v_sample.data(), probe_group.v_state + update_probe_sample_offset_,
                           state_samples * sizeof(float), cudaMemcpyDeviceToHost);

                std::ostringstream state_oss;
                state_oss << "[UpdateProbeState] group='" << probe_group.name
                          << "' step=" << optimizer_state->step
                          << " offset=" << update_probe_sample_offset_
                          << " (token " << (update_probe_sample_offset_ / 768) << ")"
                          << " m_state[0:" << state_samples << "]=[";                for (size_t i = 0; i < state_samples; ++i) {
                    if (i) state_oss << ", ";
                    state_oss << m_sample[i];
                }
                state_oss << "] v_state[0:" << state_samples << "]=[";
                for (size_t i = 0; i < state_samples; ++i) {
                    if (i) state_oss << ", ";
                    state_oss << v_sample[i];
                }
                state_oss << "]";
                std::cout << state_oss.str() << std::endl;
            }
            
            // Compute metrics
            float param_sum_sq = 0.0f, grad_sum_sq = 0.0f, update_sum_sq = 0.0f;
            float max_abs_update = 0.0f;
            
            for (size_t i = 0; i < update_probe_sample_elems_; ++i) {
                float w_before = update_probe_weights_before_[i];
                float w_after = update_probe_weights_after_[i];
                float grad = update_probe_grad_sample_[i];
                float update = w_after - w_before;
                
                param_sum_sq += w_after * w_after;
                grad_sum_sq += grad * grad;
                update_sum_sq += update * update;
                max_abs_update = std::max(max_abs_update, std::abs(update));
            }

            const float n = static_cast<float>(update_probe_sample_elems_);
            update_probe_result_.group_name = update_probe_group_name_;
            update_probe_result_.parameter_rms = std::sqrt(param_sum_sq / n);
            update_probe_result_.grad_rms = std::sqrt(grad_sum_sq / n);
            update_probe_result_.update_rms = std::sqrt(update_sum_sq / n);
            update_probe_result_.relative_update = update_probe_result_.parameter_rms > 1e-10f 
                ? update_probe_result_.update_rms / update_probe_result_.parameter_rms 
                : 0.0f;
            update_probe_result_.max_abs_update = max_abs_update;
            update_probe_result_.learning_rate = learning_rate;
            update_probe_result_.optimizer_step = optimizer_state->step;
            update_probe_result_.sample_size = static_cast<uint32_t>(update_probe_sample_elems_);

            update_probe_ready_ = true;
        }
    }
}

//======================================================//
//  resetOptimizerMoments
//======================================================//

void LanguageModel::resetOptimizerMoments() {
    if (parameter_groups_.empty()) {
        return;
    }

    cudaStream_t stream = training_state_.stream_ctrl.getPrimaryStream();
    
    for (auto& group : parameter_groups_) {
        if (group.m_state && group.size > 0) {
            cudaMemsetAsync(group.m_state, 0, group.size * sizeof(float), stream);
        }
        if (group.v_state && group.size > 0) {
            cudaMemsetAsync(group.v_state, 0, group.size * sizeof(float), stream);
        }
    }
}

void LanguageModel::scaleOptimizerMoments(float scale) {
    if (parameter_groups_.empty() || scale <= 0.0f) {
        return;
    }

    cudaStream_t stream = training_state_.stream_ctrl.getPrimaryStream();
    
    for (auto& group : parameter_groups_) {
        if (group.m_state && group.size > 0) {
            scaleDeviceBuffer(group.m_state, group.size, scale, stream);
        }
        if (group.v_state && group.size > 0) {
            scaleDeviceBuffer(group.v_state, group.size, scale, stream);
        }
    }
}

//======================================================//
//  dumpGradientValues - Debug: Dump first N gradient values for each parameter
//======================================================//

void LanguageModel::dumpGradientValues(int step, const std::string& filepath) {
    if (parameter_groups_.empty()) {
        buildParameterGroups();
    }
    
    cudaStream_t stream = training_state_.stream_ctrl.getPrimaryStream();
    cudaStreamSynchronize(stream);  // Ensure all gradients computed
    
    std::ofstream file(filepath, std::ios::app);
    file << "\n========== STEP " << step << " GRADIENT VALUES (GRIM-text) ==========\n";
    
    constexpr int NUM_TO_PRINT = 10;
    std::vector<float> host_buffer(NUM_TO_PRINT);
    
    for (const auto& group : parameter_groups_) {
        if (!group.grads || group.size == 0) continue;
        
        // Copy first N values to host
        size_t copy_count = std::min(group.size, (size_t)NUM_TO_PRINT);
        cudaMemcpy(host_buffer.data(), group.grads, copy_count * sizeof(float), cudaMemcpyDeviceToHost);
        
        // Compute stats on GPU (norm, min, max, mean)
        float norm = 0.0f;
        {
            // Simple norm computation for stats
            std::vector<float> full_buffer(group.size);
            cudaMemcpy(full_buffer.data(), group.grads, group.size * sizeof(float), cudaMemcpyDeviceToHost);
            double sum_sq = 0.0;
            float min_val = full_buffer[0], max_val = full_buffer[0];
            double sum = 0.0;
            for (size_t i = 0; i < group.size; ++i) {
                sum_sq += (double)full_buffer[i] * full_buffer[i];
                sum += full_buffer[i];
                if (full_buffer[i] < min_val) min_val = full_buffer[i];
                if (full_buffer[i] > max_val) max_val = full_buffer[i];
            }
            norm = std::sqrt((float)sum_sq);
            
            file << "\n[" << group.name << "] size=" << group.size 
                 << " norm=" << std::scientific << std::setprecision(6) << norm << "\n";
            file << "  first " << copy_count << " values: ";
            for (size_t i = 0; i < copy_count; ++i) {
                file << host_buffer[i] << " ";
            }
            file << "\n  min=" << min_val << " max=" << max_val << " mean=" << (sum / group.size) << "\n";
        }
    }
    
    file.close();
}

//======================================================//
//  computeGradNorm - GPU-Resident L2 Norm of All Gradients
//======================================================//
//
// ARCHITECTURE (Rule 22 Compliant):
// - All norm computation happens on GPU via GradNormController
// - Clipping decision made on device
// - Only final metrics copied to host (async, for logging)
// - Uses StreamController (no raw streams)
//
// FAIL LOUD:
// - Returns -1.0f on fatal error (logged with context)
// - NaN/Inf detection with explicit error codes
//======================================================//

float LanguageModel::computeGradNorm(bool sync_for_host_read) {
    if (parameter_groups_.empty()) {
        buildParameterGroups();
    }
    
    if (parameter_groups_.empty()) {
        fprintf(stderr, "[computeGradNorm] FATAL: parameter_groups_ is empty after buildParameterGroups()\n");
        return -1.0f;
    }
    
    cudaStream_t stream = training_state_.stream_ctrl.getPrimaryStream();
    
    // Initialize GradNormController on first use (lazy init for compatibility)
    if (!training_state_.gradnorm_ctrl.isInitialized()) {
        auto status = training_state_.gradnorm_ctrl.initialize(parameter_groups_.size() * 2, stream);
        if (status != GradNorm::GradNormStatus::SUCCESS) {
            fprintf(stderr, "[computeGradNorm] FATAL: GradNormController init failed: %s\n",
                    GradNorm::statusToString(status));
            return -1.0f;
        }
    }
    
    // Build arrays for GradNormController (separate grads, sizes, types)
    std::vector<float*> grads_ptrs;
    std::vector<size_t> sizes;
    std::vector<ParamGroupType> types;
    
    grads_ptrs.reserve(parameter_groups_.size());
    sizes.reserve(parameter_groups_.size());
    types.reserve(parameter_groups_.size());
    
    for (const auto& group : parameter_groups_) {
        grads_ptrs.push_back(group.grads);
        sizes.push_back(group.size);
        types.push_back(group.type);
    }
    
    // Compute norms on GPU (non-blocking)
    // Clipping is done separately in Phase2 via scaleGradients() after CPU-side decision
    auto status = training_state_.gradnorm_ctrl.computeNorms(
        grads_ptrs.data(),
        sizes.data(),
        types.data(),
        parameter_groups_.size(),
        stream
    );
    
    if (status != GradNorm::GradNormStatus::SUCCESS) {
        fprintf(stderr, "[computeGradNorm] FATAL: computeNorms failed: %s\n",
                GradNorm::statusToString(status));
        return -1.0f;
    }
    
    // PERFORMANCE FIX: Only sync when caller needs host-readable metrics
    // Most batches don't need this - skip the expensive GPU-CPU sync
    if (!sync_for_host_read) {
        // Kernel launched, norms computed on GPU, no CPU stall
        // Return cached value from last sync (persists across batches)
        return grad_metrics_ready_ ? grad_metrics_.total_norm : 0.0f;
    }
    
    // Caller needs metrics (for logging/clipping decisions) - async copy + sync
    status = training_state_.gradnorm_ctrl.asyncCopyToHost(stream);
    if (status != GradNorm::GradNormStatus::SUCCESS) {
        fprintf(stderr, "[computeGradNorm] FATAL: asyncCopyToHost failed: %s\n",
                GradNorm::statusToString(status));
        return -1.0f;
    }
    
    // SYNC POINT: Wait for async copy to complete before reading host metrics
    cudaError_t sync_err = cudaStreamSynchronize(stream);
    if (sync_err != cudaSuccess) {
        fprintf(stderr, "[computeGradNorm] FATAL: cudaStreamSynchronize failed: %s\n",
                cudaGetErrorString(sync_err));
        return -1.0f;
    }
    
    // Copy from pinned host buffer to grad_metrics_
    const auto& gm = training_state_.gradnorm_ctrl.getHostMetrics();
    grad_metrics_.embedding_norm = gm.embedding_norm;
    grad_metrics_.lm_head_norm = gm.lm_head_norm;
    grad_metrics_.numeric_head_norm = gm.numeric_head_norm;
    grad_metrics_.attention_norm = gm.attention_norm;
    grad_metrics_.ffn_norm = gm.ffn_norm;
    grad_metrics_.rmsnorm_norm = gm.rmsnorm_norm;
    grad_metrics_.scratchblock_norm = gm.scratchblock_norm;
    grad_metrics_.total_norm = gm.total_norm;
    grad_metrics_ready_ = true;
    
    // Check for NaN/Inf (fail loud)
    if (gm.has_nan) {
        GRIM::Logging::EmitModuleWarning(GRIM::Logging::ModuleId::Optimizer, "[computeGradNorm] WARNING: NaN detected in gradients!");
        if (gm.first_nan_group >= 0 &&
            static_cast<size_t>(gm.first_nan_group) < parameter_groups_.size()) {
            const auto& group = parameter_groups_[gm.first_nan_group];
            char buffer[256];
            snprintf(buffer, sizeof(buffer), "[computeGradNorm] WARNING: first NaN group=%d name=%s size=%zu norm=%g",
                     gm.first_nan_group, group.name.c_str(), group.size, gm.first_nan_value);
            GRIM::Logging::EmitModuleWarning(GRIM::Logging::ModuleId::Optimizer, buffer);
        } else {
            char buffer[256];
            snprintf(buffer, sizeof(buffer), "[computeGradNorm] WARNING: first NaN group=%d norm=%g",
                     gm.first_nan_group, gm.first_nan_value);
            GRIM::Logging::EmitModuleWarning(GRIM::Logging::ModuleId::Optimizer, buffer);
        }
    }
    if (gm.has_inf) {
        GRIM::Logging::EmitModuleWarning(GRIM::Logging::ModuleId::Optimizer, "[computeGradNorm] WARNING: Inf detected in gradients!");
        if (gm.first_inf_group >= 0 &&
            static_cast<size_t>(gm.first_inf_group) < parameter_groups_.size()) {
            const auto& group = parameter_groups_[gm.first_inf_group];
            char buffer[256];
            snprintf(buffer, sizeof(buffer), "[computeGradNorm] WARNING: first Inf group=%d name=%s size=%zu norm=%g",
                     gm.first_inf_group, group.name.c_str(), group.size, gm.first_inf_value);
            GRIM::Logging::EmitModuleWarning(GRIM::Logging::ModuleId::Optimizer, buffer);
        } else {
            char buffer[256];
            snprintf(buffer, sizeof(buffer), "[computeGradNorm] WARNING: first Inf group=%d norm=%g",
                     gm.first_inf_group, gm.first_inf_value);
            GRIM::Logging::EmitModuleWarning(GRIM::Logging::ModuleId::Optimizer, buffer);
        }
    }

    auto logFirstNonFinite = [&](int group_idx, const char* trigger) {
        if (group_idx < 0 || static_cast<size_t>(group_idx) >= parameter_groups_.size()) {
            char buffer[256];
            snprintf(buffer, sizeof(buffer), "[computeGradNorm] WARNING: %s group index out of range (%d)",
                     trigger, group_idx);
            GRIM::Logging::EmitModuleWarning(GRIM::Logging::ModuleId::Optimizer, buffer);
            return;
        }
        const auto& group = parameter_groups_[group_idx];
        if (!group.grads || group.size == 0) {
            char buffer[256];
            snprintf(buffer, sizeof(buffer), "[computeGradNorm] WARNING: %s group '%s' has no grad buffer",
                     trigger, group.name.c_str());
            GRIM::Logging::EmitModuleWarning(GRIM::Logging::ModuleId::Optimizer, buffer);
            return;
        }
        NonFiniteScanResult scan{};
        if (!scanFirstNonFinite(group.grads, group.size, &scan, stream)) {
            char buffer[256];
            snprintf(buffer, sizeof(buffer), "[computeGradNorm] WARNING: %s group '%s' scan found no non-finite values",
                     trigger, group.name.c_str());
            GRIM::Logging::EmitModuleWarning(GRIM::Logging::ModuleId::Optimizer, buffer);
            return;
        }
        const char* kind = scan.is_nan ? "NaN" : (scan.is_inf ? "Inf" : "NonFinite");
        int rows = 0;
        int cols = 0;
        char buffer[512];
        if (getGroupShape(group, config_, training_state_.num_kv_heads, &rows, &cols) &&
            scan.index >= 0) {
            const int row = scan.index / cols;
            const int col = scan.index - row * cols;
            snprintf(buffer, sizeof(buffer),
                    "[computeGradNorm] NON-FINITE sample group=%d name=%s idx=%d row=%d col=%d shape=%dx%d value=%g kind=%s trigger=%s",
                    group_idx, group.name.c_str(), scan.index, row, col, rows, cols, scan.value, kind, trigger);
        } else {
            snprintf(buffer, sizeof(buffer),
                    "[computeGradNorm] NON-FINITE sample group=%d name=%s idx=%d value=%g kind=%s trigger=%s",
                    group_idx, group.name.c_str(), scan.index, scan.value, kind, trigger);
        }
        GRIM::Logging::EmitModuleWarning(GRIM::Logging::ModuleId::Optimizer, buffer);
    };

    if (gm.has_nan) {
        logFirstNonFinite(gm.first_nan_group, "NaN");
    }
    if (gm.has_inf) {
        const int inf_group = gm.first_inf_group;
        if (inf_group != gm.first_nan_group) {
            logFirstNonFinite(inf_group, "Inf");
        }
    }
    
    return gm.total_norm;
}

//======================================================//
//  scaleGradients - Gradient Clipping Helper
//======================================================//

__global__ void scaleKernel(float* __restrict__ data, float scale, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        data[idx] *= scale;
    }
}

void LanguageModel::scaleGradients(float scale) {
    if (parameter_groups_.empty()) {
        buildParameterGroups();
    }
    
    const int threads = 256;
    
    for (auto& group : parameter_groups_) {
        if (!group.grads || group.size == 0) continue;
        
        const int blocks = (group.size + threads - 1) / threads;
        scaleKernel<<<blocks, threads, 0, training_state_.stream_ctrl.getPrimaryStream()>>>(
            group.grads, scale, static_cast<int>(group.size)
        );
    }
    
    // No sync needed - kernels are on same stream as optimizer step posssibly!
    // GPU will naturally serialize: scale kernels → optimizer kernels
}

//======================================================//
//  setSequenceLossWeights / clearSequenceLossWeights
//======================================================//
// NOTE: Currently unused (sample_weight=1.0 always) but INTENTIONALLY KEPT
// for future use cases:
//   - Curriculum learning: weight easy examples higher early, hard examples later
//   - Data quality weighting: upweight high-quality sources (academic papers, etc.)
//   - Importance sampling: variance reduction via per-sample weights
//   - Deduplication soft-weighting: downweight near-duplicate sequences
// When count=0, loss kernel defaults to sample_weight=1.0f (standard LLM training)
//======================================================//

void LanguageModel::setSequenceLossWeights(const std::vector<float>& weights) {
    if (weights.empty()) {
        training_state_.sequence_weight_count = 0;
        return;
    }
    
    size_t copy_size = std::min(weights.size(), 
                                static_cast<size_t>(training_state_.max_cached_tokens));
    
    cudaMemcpyAsync(
        training_state_.sequence_weights_tensor.data,
        weights.data(),
        copy_size * sizeof(float),
        cudaMemcpyHostToDevice,
        training_state_.stream_ctrl.getPrimaryStream()
    );
    
    training_state_.sequence_weight_count = static_cast<int>(copy_size);
}

void LanguageModel::clearSequenceLossWeights() {
    training_state_.sequence_weight_count = 0;
}

//======================================================//
//  clampGradients - Clamp individual gradient values
//======================================================//

__global__ void clampKernel(float* __restrict__ data, float min_val, float max_val, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        data[idx] = fminf(fmaxf(data[idx], min_val), max_val);
    }
}

void LanguageModel::clampGradients(float min_val, float max_val) {
    if (parameter_groups_.empty()) {
        buildParameterGroups();
    }
    
    const int threads = 256;
    
    for (auto& group : parameter_groups_) {
        if (!group.grads || group.size == 0) continue;
        
        const int blocks = (group.size + threads - 1) / threads;
        clampKernel<<<blocks, threads, 0, training_state_.stream_ctrl.getPrimaryStream()>>>(
            group.grads, min_val, max_val, static_cast<int>(group.size)
        );
    }
    
    cudaStreamSynchronize(training_state_.stream_ctrl.getPrimaryStream());
}

//======================================================//
//  Diagnostic Methods (stubs for unused features)
//======================================================//

void LanguageModel::dumpGradients(const std::string& path) {
    BWD_WARN("[dumpGradients] Not implemented in 3-phase architecture");
}

void LanguageModel::logEmbeddingDiagnostics(const std::string& tag) {
    BWD_WARN("[logEmbeddingDiagnostics] Not implemented in 3-phase architecture");
}

#endif // USE_CUDA

} // namespace GRIM
