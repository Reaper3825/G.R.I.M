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
#include "../Shared/GradNorm/GradNormGPU.hpp"
#include "../Common/grim_scale_buffer.hpp"
#include "../Shared/LogRecorder/LogRecorder.hpp"
#include "../Shared/TensorContract/TensorContract_GPU.hpp"
#include "../Shared/Optimizers/AdamW/AdamW_Kernal_GPU.hpp"
#include "../../../control/ai_config_paths.hpp"
#include "../Shared/HyperParameters/HyperParameters_GPU.hpp"
#include "../Shared/TrainingState/TrainingTensors.hpp"
#include "../Shared/Loss/ComputeLoss/AutogradLoss.hpp"
#include "Autograd/AutogradTraining.hpp"  // Pure autograd backward - replaces legacy 3-phase system

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
    if (expected != group.size()) {
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
    cudaStream_t stream = training_state_.stream_ctrl.getPrimaryStream();
    
    // ── Top-level parameter Tensors ──────────────────────────────────────────
    training_state_.tensors_->embedding_weights.zero_grad(stream);
    training_state_.tensors_->position_embedding_weights.zero_grad(stream);
    training_state_.tensors_->lm_head_weights.zero_grad(stream);
    training_state_.tensors_->lm_head_bias.zero_grad(stream);
    training_state_.tensors_->numeric_head_weights.zero_grad(stream);
    training_state_.tensors_->numeric_head_bias.zero_grad(stream);
    training_state_.tensors_->final_rms_gamma.zero_grad(stream);  // BUG FIX: was missing from zeroGrad!
    
    // ── Encoder layer Tensors ────────────────────────────────────────────────
    GPUGrimEncoder& gpu_encoder = getGpuEncoder();
    for (int layer = 0; layer < cfg.num_layers; ++layer) {
        GPUEncoderLayer* enc = gpu_encoder.getLayer(layer);
        if (!enc) continue;
        
        // RMSNorm gamma
        enc->rms1Gamma().zero_grad(stream);
        enc->rms2Gamma().zero_grad(stream);
        
        // Attention weights/biases
        enc->attnWqkv().zero_grad(stream);
        enc->attnBqkv().zero_grad(stream);
        enc->attnWo().zero_grad(stream);
        enc->attnBo().zero_grad(stream);
        
        // FFN weights/biases
        if (FeedForwardLayer* ffn = enc->getFfnLayer()) {
            ffn->W1().zero_grad(stream);
            ffn->b1().zero_grad(stream);
            ffn->W2().zero_grad(stream);
            ffn->b2().zero_grad(stream);
        }
        
        // LayerScale
        enc->layerScale1().zero_grad(stream);
        enc->layerScale2().zero_grad(stream);
    }
    
    // ── ScratchBlock gradients ─────────
    if (scratch_block_layer_ && scratch_block_layer_->isEnabled()) {
        scratch_block_layer_->atomTypeEmbeddings().zero_grad(stream);
        scratch_block_layer_->atomProjection().zero_grad(stream);
        scratch_block_layer_->textFeatureProjection().zero_grad(stream);
    }
    
    // Zeroing is enqueued on the primary stream; no host sync needed here.
}

//======================================================//
//  backward - 3-Phase Backward Pass Wrapper
//======================================================//

void LanguageModel::backward(float loss, bool accumulate, float grad_scale, uint64_t step) {
    if (!training_state_.initialized) {
        throw std::runtime_error("[backward] Training state not initialized - caller MUST call initTrainingState() first");
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
        throw std::runtime_error("[backward] Invalid batch dimensions: batch_size=" + std::to_string(batch_size) + " seq_len=" + std::to_string(seq_len));
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
    if (!training_state_.autograd_intermediates.loss_tensor.grad_fn) {
        throw std::runtime_error("[backward] loss_tensor.grad_fn is NULL - forward pass (computeLossBatch) MUST be called first!");
    }

    BWD_INFO("[backward] Delegating to Autograd::executeAutogradBackward()");

    const float effective_grad_scale = (grad_scale > 0.0f) ? grad_scale : 1.0f;
    Autograd::AutogradContext autograd_ctx = Autograd::initAutogradContext(
        &config_,
        &training_state_,
        &getGpuEncoder(),
        getScratchBlockLayer(),
        training_state_.cublas_handle,
        stream,
        batch_size,
        seq_len,
        effective_grad_scale,
        step,
        true
    );

    // ScratchBlock backward reads side-channel token buffers via context.
    autograd_ctx.token_numeric_values = training_state_.cached_token_numeric_values.data;
    autograd_ctx.token_numeric_mask = reinterpret_cast<const uint8_t*>(training_state_.cached_token_numeric_mask.data);
    autograd_ctx.token_text_features = reinterpret_cast<const uint16_t*>(training_state_.cached_token_text_features.data);
    autograd_ctx.token_text_mask = reinterpret_cast<const uint8_t*>(training_state_.cached_token_text_mask.data);

    Autograd::BackwardResult bwd_result = Autograd::executeAutogradBackward(autograd_ctx, accumulate);
    if (!bwd_result.success) {
        throw std::runtime_error("[backward] executeAutogradBackward failed: " + bwd_result.error_message);
    }
    BWD_INFO("[backward] Autograd backward complete: grad_norm=" << bwd_result.grad_norm);
    
    // Issue #60 DEBUG: Log gradient attribution for debugging token 277 mode collapse
    // This shows LM head vs embedding backward contributions separately
    if (training_state_.debug_gradient_attribution) {
        training_state_.logGradientAttribution(static_cast<int>(step), stream);
    }
    
    // ALWAYS clear sequence weights after backward
    training_state_.sequence_weight_count = 0;
    
    // Clear autograd intermediates to free GPU memory from intermediate tensors
    // The next computeLossBatch() will create new intermediates anyway
    training_state_.autograd_intermediates.clear();
    
    BWD_INFO("[backward] AUTOGRAD COMPLETE");
}


//======================================================//
//  buildParameterGroups - Initialize Parameter Groups for Optimizer
//======================================================//

void LanguageModel::buildParameterGroups() {
    fprintf(stderr, "[buildParameterGroups] ENTER\n"); fflush(stderr);
    // Clear parameter group metadata (optimizer states are managed by TrainingState)
    parameter_groups_.clear();
    
    const auto& cfg = config_;
    const int num_kv_heads = training_state_.num_kv_heads;
    fprintf(stderr, "[buildParameterGroups] cfg: num_layers=%d num_heads=%d head_dim=%d num_kv_heads=%d d_model=%d vocab=%u tie=%d numeric=%d\n",
            cfg.num_layers, cfg.num_heads, cfg.head_dim, num_kv_heads, cfg.d_model, cfg.vocab_size, (int)cfg.tie_embeddings, (int)cfg.numeric_head_enabled); fflush(stderr);
    
    // Validate GQA dimensions using TensorContract
    TensorContract::GQADims gqa_dims{cfg.num_heads, num_kv_heads, cfg.head_dim};
    if (!gqa_dims.is_valid()) {
        BWD_ERROR("[buildParameterGroups] TensorContract GQA validation failed!");
    }

    GPUGrimEncoder& gpu_encoder = getGpuEncoder();

    // ── Helper: Register a Tensor as a parameter group ───────────────────────
    // Eliminates manual size computation — uses Tensor::numel() directly.
    // Rule 20: tensor MUST have data and grad, otherwise skip with warning.
    auto registerTensor = [&](const std::string& name, Tensor& tensor,
                              ParamGroupType type, int layer = -1,
                              float wd_mult = 1.0f) {
        if (!tensor.data || !tensor.has_grad()) {
            throw std::runtime_error("[buildParameterGroups] " + name + " has no data or grad! data=" 
                                     + std::to_string(reinterpret_cast<uintptr_t>(tensor.data)) 
                                     + " has_grad=" + std::to_string(tensor.has_grad()) 
                                     + " layer=" + std::to_string(layer));
        }
        ParameterGroup group;
        group.name = name;
        group.tensor = &tensor;
        group.m_tensor = nullptr;
        group.v_tensor = nullptr;
        group.type = type;
        group.layer_index = layer;
        group.weight_decay_multiplier = wd_mult;
        if (layer >= 0) {
            group.upsilon = HyperParameters::UPSILON_BASE * 
                sqrtf(static_cast<float>(HyperParameters::UPSILON_REFERENCE_LAYERS) / static_cast<float>(layer + 1));
        }
        parameter_groups_.push_back(group);
    };

    // Helper: Register a bias or norm tensor (weight_decay = 0.0)
    // Standard AdamW practice: biases and normalization parameters should NOT have weight decay
    auto registerNonDecayTensor = [&](const std::string& name, Tensor& tensor,
                                      ParamGroupType type, int layer = -1) {
        registerTensor(name, tensor, type, layer, 0.0f);
    };

    // Helper: Try to register a bias tensor if it has been allocated (guards use_bias=false case)
    auto tryRegisterBias = [&](const std::string& name, Tensor& tensor,
                               ParamGroupType type, int layer = -1) {
        if (tensor.data && tensor.has_grad()) {
            registerNonDecayTensor(name, tensor, type, layer);
        } else {
            fprintf(stderr, "[buildParameterGroups] %s skipped (use_bias=false or not allocated)\n", name.c_str());
        }
    };

    // ── Top-level parameters ─────────────────────────────────────────────────
    
    // Embedding weights - SKIP when tied (handled by LM head group below)
    // When tie_embeddings=true, embedding grad buffer == lm_head grad buffer (same pointer)
    // Registering both would double-count gradients and corrupt optimizer state
    if (!cfg.tie_embeddings) {
        registerTensor("embedding", training_state_.tensors_->embedding_weights, ParamGroupType::EMBEDDING);
    }

    // Issue #113: Sinusoidal position embeddings are FIXED (not learned).
    // Only register if they have allocated data (i.e., learned position embeddings are in use).
    if (training_state_.tensors_->position_embedding_weights.data && training_state_.tensors_->position_embedding_weights.has_grad()) {
        registerTensor("position_embedding", training_state_.tensors_->position_embedding_weights, ParamGroupType::EMBEDDING);
    } else {
        fprintf(stderr, "[buildParameterGroups] position_embedding skipped (sinusoidal/fixed, not learned)\n");
    }

    // LM head weights (includes tied embedding grads when tie_embeddings=true)
    fprintf(stderr, "[buildParameterGroups] DIAG-A: about to register LM head\n"); fflush(stderr);
    {
        std::string lm_name = cfg.tie_embeddings ? "embedding_lm_head_tied" : "lm_head_weight";
        registerTensor(lm_name, training_state_.tensors_->lm_head_weights, ParamGroupType::LM_HEAD);
    }
    // LM head bias — no weight decay
    tryRegisterBias("lm_head_bias", training_state_.tensors_->lm_head_bias, ParamGroupType::LM_HEAD);
    fprintf(stderr, "[buildParameterGroups] DIAG-B: LM head done, registering numeric/log_var\n"); fflush(stderr);

    // Numeric head
    if (cfg.numeric_head_enabled) {
        registerTensor("numeric_head_weight", training_state_.tensors_->numeric_head_weights, ParamGroupType::NUMERIC_HEAD);
        // Numeric head bias — no weight decay
        tryRegisterBias("numeric_head_bias", training_state_.tensors_->numeric_head_bias, ParamGroupType::NUMERIC_HEAD);
    }
    
    // Learned loss weighting (log-variance for text and numeric losses) — no weight decay
    registerNonDecayTensor("log_var_text", training_state_.log_var_text, ParamGroupType::NUMERIC_HEAD);
    registerNonDecayTensor("log_var_numeric", training_state_.log_var_numeric, ParamGroupType::NUMERIC_HEAD);

    fprintf(stderr, "[buildParameterGroups] DIAG-C: numeric/log_var done, starting %d layers\n", cfg.num_layers); fflush(stderr);
    // ── Per-layer encoder parameters ─────────────────────────────────────────
    for (int layer = 0; layer < cfg.num_layers; ++layer) {
        GPUEncoderLayer* enc = gpu_encoder.getLayer(layer);
        if (!enc) {
            BWD_WARN("[buildParameterGroups] Layer " << layer << " is NULL - skipping");
            continue;
        }
        
        std::string prefix = "layer" + std::to_string(layer);
        
        // Attention weights/biases
        registerTensor(prefix + "_qkv_weight", enc->attnWqkv(), ParamGroupType::ATTENTION, layer);
        tryRegisterBias(prefix + "_qkv_bias",  enc->attnBqkv(), ParamGroupType::ATTENTION, layer);
        registerTensor(prefix + "_wo_weight",   enc->attnWo(),   ParamGroupType::ATTENTION, layer);
        tryRegisterBias(prefix + "_wo_bias",    enc->attnBo(),   ParamGroupType::ATTENTION, layer);
        
        // FFN weights/biases
        registerTensor(prefix + "_ffn_w1", enc->ffnW1(), ParamGroupType::FFN, layer);
        tryRegisterBias(prefix + "_ffn_b1", enc->ffnB1(), ParamGroupType::FFN, layer);
        registerTensor(prefix + "_ffn_w2", enc->ffnW2(), ParamGroupType::FFN, layer);
        tryRegisterBias(prefix + "_ffn_b2", enc->ffnB2(), ParamGroupType::FFN, layer);
        
        // RMSNorm gamma (pre-norm + sandwich post-residual) — no weight decay
        registerNonDecayTensor(prefix + "_rms1_gamma", enc->rms1Gamma(), ParamGroupType::RMSNORM, layer);
        registerNonDecayTensor(prefix + "_rms2_gamma", enc->rms2Gamma(), ParamGroupType::RMSNORM, layer);
        registerNonDecayTensor(prefix + "_rms_post_attn_gamma", enc->rmsPostAttnGamma(), ParamGroupType::RMSNORM, layer);
        registerNonDecayTensor(prefix + "_rms_post_ffn_gamma", enc->rmsPostFfnGamma(), ParamGroupType::RMSNORM, layer);
    }

    fprintf(stderr, "[buildParameterGroups] DIAG-D: all %d layers done, registering scratchblock/final_rms\n", cfg.num_layers); fflush(stderr);
    // ── ScratchBlock parameters ──
    fprintf(stderr, "[buildParameterGroups] DIAG-D0: scratch_block_layer_=%p\n", (void*)scratch_block_layer_.get()); fflush(stderr);
    if (scratch_block_layer_) {
        fprintf(stderr, "[buildParameterGroups] DIAG-D0a: isEnabled=%d\n", (int)scratch_block_layer_->isEnabled()); fflush(stderr);
    }
    if (scratch_block_layer_ && scratch_block_layer_->isEnabled()) {
        try {
            auto& ate = scratch_block_layer_->atomTypeEmbeddings();
            fprintf(stderr, "[buildParameterGroups] DIAG-D1: atom_type_embeddings data=%p grad=%d numel=%zu\n",
                    (void*)ate.data, (int)ate.has_grad(), ate.numel()); fflush(stderr);
            registerTensor("scratch_block_atom_type_embeddings", ate, ParamGroupType::SCRATCHBLOCK);
            fprintf(stderr, "[buildParameterGroups] DIAG-D1: registered OK\n"); fflush(stderr);

            auto& ap = scratch_block_layer_->atomProjection();
            fprintf(stderr, "[buildParameterGroups] DIAG-D2: atom_projection data=%p grad=%d numel=%zu\n",
                    (void*)ap.data, (int)ap.has_grad(), ap.numel()); fflush(stderr);
            registerTensor("scratch_block_atom_projection", ap, ParamGroupType::SCRATCHBLOCK);
            fprintf(stderr, "[buildParameterGroups] DIAG-D2: registered OK\n"); fflush(stderr);

            auto& tfp = scratch_block_layer_->textFeatureProjection();
            fprintf(stderr, "[buildParameterGroups] DIAG-D3: text_feature_projection data=%p grad=%d numel=%zu\n",
                    (void*)tfp.data, (int)tfp.has_grad(), tfp.numel()); fflush(stderr);
            registerTensor("scratch_block_text_feature_projection", tfp, ParamGroupType::SCRATCHBLOCK);
            fprintf(stderr, "[buildParameterGroups] DIAG-D3: registered OK\n"); fflush(stderr);
        } catch (const std::exception& ex) {
            fprintf(stderr, "[buildParameterGroups] DIAG-D-EXCEPTION in scratchblock: %s\n", ex.what()); fflush(stderr);
            throw;
        }
        fprintf(stderr, "[buildParameterGroups] DIAG-D4: scratchblock done\n"); fflush(stderr);
    } else {
        fprintf(stderr, "[buildParameterGroups] DIAG-D-SKIP: scratchblock not enabled\n"); fflush(stderr);
    }

    // Final RMSNorm (between encoder output and LM head) — no weight decay
    fprintf(stderr, "[buildParameterGroups] DIAG-D5: about to register final_rms_gamma data=%p grad=%d numel=%zu\n",
            (void*)training_state_.tensors_->final_rms_gamma.data, (int)training_state_.tensors_->final_rms_gamma.has_grad(),
            training_state_.tensors_->final_rms_gamma.numel()); fflush(stderr);
    registerNonDecayTensor("final_rms_gamma", training_state_.tensors_->final_rms_gamma, ParamGroupType::RMSNORM);
    fprintf(stderr, "[buildParameterGroups] DIAG-D5: registered OK\n"); fflush(stderr);

    fprintf(stderr, "[buildParameterGroups] DIAG-E: %zu groups registered, allocating optimizer states\n", parameter_groups_.size()); fflush(stderr);
    // Collect sizes for centralized optimizer state allocation
    std::vector<size_t> sizes;
    sizes.reserve(parameter_groups_.size());
    for (const auto& group : parameter_groups_) {
        sizes.push_back(group.size());
    }
    
    // Allocate optimizer states via TrainingState (centralized ownership)
    training_state_.allocateOptimizerStates(sizes);
    
    fprintf(stderr, "[buildParameterGroups] DIAG-F: optimizer states allocated (m=%zu v=%zu), binding tensors\n",
            training_state_.optimizer_m_states.size(), training_state_.optimizer_v_states.size()); fflush(stderr);
    // Bind optimizer Tensors to parameter groups (groups hold pointers, NOT ownership)
    for (size_t i = 0; i < parameter_groups_.size(); ++i) {
        parameter_groups_[i].m_tensor = &training_state_.optimizer_m_states[i];
        parameter_groups_[i].v_tensor = &training_state_.optimizer_v_states[i];
    }
    fprintf(stderr, "[buildParameterGroups] DIAG-G: all bindings done\n"); fflush(stderr);
    
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
    fprintf(stderr, "[configureUpdateProbe] ENTER group='%s' sample_elems=%zu\n", group_name.c_str(), sample_elems); fflush(stderr);
    update_probe_group_name_ = group_name;
    update_probe_sample_elems_ = sample_elems;
    update_probe_ready_ = false;
    
    // Ensure parameter groups are built before configuring probe
    if (parameter_groups_.empty()) {
        fprintf(stderr, "[configureUpdateProbe] parameter_groups_ is empty, calling buildParameterGroups()\n"); fflush(stderr);
        buildParameterGroups();
        fprintf(stderr, "[configureUpdateProbe] buildParameterGroups() returned, groups=%zu\n", parameter_groups_.size()); fflush(stderr);
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
    if (update_probe_sample_elems_ == 0 || update_probe_sample_elems_ > probe_group.size()) {
        update_probe_sample_elems_ = std::min<size_t>(probe_group.size(), 2048);
    }
    update_probe_weights_before_.assign(update_probe_sample_elems_, 0.0f);
    update_probe_weights_after_.assign(update_probe_sample_elems_, 0.0f);
    update_probe_grad_sample_.assign(update_probe_sample_elems_, 0.0f);
    fprintf(stderr, "[configureUpdateProbe] EXIT: group_index=%zu sample_elems=%zu probe_group_size=%zu\n",
            update_probe_group_index_, update_probe_sample_elems_,
            update_probe_group_index_ != static_cast<size_t>(-1) ? parameter_groups_[update_probe_group_index_].size() : 0);
    fflush(stderr);
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
//  updateWeights - AdamW Optimizer Step
//======================================================//

void LanguageModel::updateWeights(float learning_rate,
                                  OptimizerState* optimizer_state,
                                  float weight_decay) {
    if (!training_state_.initialized) {
        throw std::runtime_error("[updateWeights] Training state not initialized - caller MUST call initTrainingState() first");
    }
    
    if (!optimizer_state) {
        throw std::runtime_error("[updateWeights] optimizer_state is NULL - caller MUST provide valid OptimizerState");
    }

    // NOTE: step is incremented in training loop after this function returns
    // Do NOT increment here to avoid double-counting

    // Rebuild parameter groups if needed
    if (parameter_groups_.empty()) {
        buildParameterGroups();
    }

    const cudaStream_t stream = training_state_.stream_ctrl.getPrimaryStream();
    
    for (size_t i = 0; i < parameter_groups_.size(); ++i) {
        auto& group = parameter_groups_[i];
        if (!group.weights() || !group.grads() || group.size() == 0) continue;
        
        // Validate optimizer state exists before update
        if (!group.m_state() || !group.v_state()) {
            throw std::runtime_error(
                "[updateWeights] FATAL: Missing optimizer state for group '" + group.name +
                "' idx=" + std::to_string(i) +
                " size=" + std::to_string(group.size()) +
                " weights=" + std::to_string(reinterpret_cast<uintptr_t>(group.weights())) +
                " grads=" + std::to_string(reinterpret_cast<uintptr_t>(group.grads())) +
                " m_state=" + std::to_string(reinterpret_cast<uintptr_t>(group.m_state())) +
                " v_state=" + std::to_string(reinterpret_cast<uintptr_t>(group.v_state())) +
                " step=" + std::to_string(optimizer_state->step));
        }

        // Apply depth-aware upsilon (Υ) regularization to weight decay
        // Formula: Υ_l = 0.1 * sqrt(L_ref / L) where L is 1-indexed layer
        // Deeper layers get LESS regularization (smaller effective weight_decay)
        // weight_decay_multiplier = 0.0 for biases/norms (standard AdamW practice)
        const float effective_weight_decay = weight_decay * group.upsilon * group.weight_decay_multiplier;
        
        // Launch AdamW optimizer kernel to update this parameter group
        launchAdamWKernel(
            group,
            learning_rate,
            effective_weight_decay,
            optimizer_state->step,
            stream
        );
    }
}

//======================================================//
//  resetOptimizerMoments
//======================================================//

void LanguageModel::resetOptimizerMoments() {
    if (parameter_groups_.empty()) {
        throw std::runtime_error("[resetOptimizerMoments] parameter_groups_ is empty - buildParameterGroups() MUST be called first");
    }

    cudaStream_t stream = training_state_.stream_ctrl.getPrimaryStream();
    
    for (auto& group : parameter_groups_) {
        if (group.m_state() && group.size() > 0) {
            cudaMemsetAsync(group.m_state(), 0, group.size() * sizeof(float), stream);
        }
        if (group.v_state() && group.size() > 0) {
            cudaMemsetAsync(group.v_state(), 0, group.size() * sizeof(float), stream);
        }
    }
}

void LanguageModel::scaleOptimizerMoments(float scale) {
    if (parameter_groups_.empty()) {
        throw std::runtime_error("[scaleOptimizerMoments] parameter_groups_ is empty - buildParameterGroups() MUST be called first");
    }
    if (scale <= 0.0f) {
        throw std::runtime_error("[scaleOptimizerMoments] scale MUST be > 0.0f, got " + std::to_string(scale));
    }

    cudaStream_t stream = training_state_.stream_ctrl.getPrimaryStream();
    
    for (auto& group : parameter_groups_) {
        if (group.m_state() && group.size() > 0) {
            scaleDeviceBuffer(group.m_state(), group.size(), scale, stream);
        }
        if (group.v_state() && group.size() > 0) {
            scaleDeviceBuffer(group.v_state(), group.size(), scale, stream);
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
        if (!group.grads() || group.size() == 0) continue;
        
        // Copy first N values to host
        size_t copy_count = std::min(group.size(), (size_t)NUM_TO_PRINT);
        cudaMemcpy(host_buffer.data(), group.grads(), copy_count * sizeof(float), cudaMemcpyDeviceToHost);
        
        // Compute stats on GPU (norm, min, max, mean)
        float norm = 0.0f;
        {
            // Simple norm computation for stats
            std::vector<float> full_buffer(group.size());
            cudaMemcpy(full_buffer.data(), group.grads(), group.size() * sizeof(float), cudaMemcpyDeviceToHost);
            double sum_sq = 0.0;
            float min_val = full_buffer[0], max_val = full_buffer[0];
            double sum = 0.0;
            for (size_t i = 0; i < group.size(); ++i) {
                sum_sq += (double)full_buffer[i] * full_buffer[i];
                sum += full_buffer[i];
                if (full_buffer[i] < min_val) min_val = full_buffer[i];
                if (full_buffer[i] > max_val) max_val = full_buffer[i];
            }
            norm = std::sqrt((float)sum_sq);
            
            file << "\n[" << group.name << "] size=" << group.size() 
                 << " norm=" << std::scientific << std::setprecision(6) << norm << "\n";
            file << "  first " << copy_count << " values: ";
            for (size_t i = 0; i < copy_count; ++i) {
                file << host_buffer[i] << " ";
            }
            file << "\n  min=" << min_val << " max=" << max_val << " mean=" << (sum / group.size()) << "\n";
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
    
    // Issue #138: Record CUDA event BEFORE launching grad norm kernels.
    // This lets us measure how much time cudaStreamSynchronize spends draining
    // upstream backward kernels vs computing the actual grad norms.
    // Without this, the wall-clock timer misleadingly shows 3ms-53ms variance
    // that is actually backward pipeline drain time, not norm computation time.
    cudaEvent_t pre_norm_event = nullptr;
    if (sync_for_host_read) {
        cudaEventCreate(&pre_norm_event);
        cudaEventRecord(pre_norm_event, stream);
    }
    
    // Compute norms on GPU (non-blocking)
    // Clipping is done separately in Phase2 via scaleGradients() after CPU-side decision
    auto status = training_state_.gradnorm_ctrl.computeNorms(
        parameter_groups_.data(),
        parameter_groups_.size(),
        stream
    );
    
    if (status != GradNorm::GradNormStatus::SUCCESS) {
        fprintf(stderr, "[computeGradNorm] FATAL: computeNorms failed: %s\n",
                GradNorm::statusToString(status));
        if (pre_norm_event) cudaEventDestroy(pre_norm_event);
        return -1.0f;
    }
    
    // PERFORMANCE FIX: Only sync when caller needs host-readable metrics
    // Most batches don't need this - skip the expensive GPU-CPU sync
    if (!sync_for_host_read) {
        // Kernel launched, norms computed on GPU, no CPU stall
        // RULE 20: Require caller to have synced recently via sync_for_host_read=true
        // (e.g., every 10 batches). Do NOT silently return cached/stale data.
        if (!grad_metrics_ready_) {
            throw std::runtime_error(
                "[computeGradNorm] sync_for_host_read=false but no valid cached metrics! "
                "Caller MUST call with sync_for_host_read=true periodically (e.g., every 10 batches) "
                "to populate grad_metrics_. At line " + std::to_string(__LINE__));
        }
        return grad_metrics_.total_norm;
    }
    
    // Record event AFTER norm kernels launched, BEFORE sync
    cudaEvent_t post_norm_event = nullptr;
    cudaEventCreate(&post_norm_event);
    cudaEventRecord(post_norm_event, stream);
    
    // Caller needs metrics (for logging/clipping decisions) - async copy + sync
    status = training_state_.gradnorm_ctrl.asyncCopyToHost(stream);
    if (status != GradNorm::GradNormStatus::SUCCESS) {
        fprintf(stderr, "[computeGradNorm] FATAL: asyncCopyToHost failed: %s\n",
                GradNorm::statusToString(status));
        cudaEventDestroy(pre_norm_event);
        cudaEventDestroy(post_norm_event);
        return -1.0f;
    }
    
    // SYNC POINT: Wait for async copy to complete before reading host metrics
    cudaError_t sync_err = cudaStreamSynchronize(stream);
    if (sync_err != cudaSuccess) {
        fprintf(stderr, "[computeGradNorm] FATAL: cudaStreamSynchronize failed: %s\n",
                cudaGetErrorString(sync_err));
        cudaEventDestroy(pre_norm_event);
        cudaEventDestroy(post_norm_event);
        return -1.0f;
    }
    
    // Measure GPU-side timing: how long did the norm kernels themselves take?
    // pre_norm_event fires when GPU reaches our kernels (after backward drains)
    // post_norm_event fires when norm kernels are done
    float gpu_norm_ms = 0.0f;
    cudaEventElapsedTime(&gpu_norm_ms, pre_norm_event, post_norm_event);
    last_gpu_norm_ms_ = gpu_norm_ms;
    cudaEventDestroy(pre_norm_event);
    cudaEventDestroy(post_norm_event);
    
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
                     gm.first_nan_group, group.name.c_str(), group.size(), gm.first_nan_value);
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
                     gm.first_inf_group, group.name.c_str(), group.size(), gm.first_inf_value);
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
        if (!group.grads() || group.size() == 0) {
            char buffer[256];
            snprintf(buffer, sizeof(buffer), "[computeGradNorm] WARNING: %s group '%s' has no grad buffer",
                     trigger, group.name.c_str());
            GRIM::Logging::EmitModuleWarning(GRIM::Logging::ModuleId::Optimizer, buffer);
            return;
        }
        NonFiniteScanResult scan{};
        if (!scanFirstNonFinite(group.grads(), group.size(), &scan, stream)) {
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
        if (!group.grads() || group.size() == 0) continue;
        
        const int blocks = (group.size() + threads - 1) / threads;
        scaleKernel<<<blocks, threads, 0, training_state_.stream_ctrl.getPrimaryStream()>>>(
            group.grads(), scale, static_cast<int>(group.size())
        );
    }
    
    // No sync needed - kernels are on same stream as optimizer step posssibly!
    // GPU will naturally serialize: scale kernels → optimizer kernels
}

//======================================================//
//  scaleGradientsByType - Scale only a specific parameter group type
//  Issue #134: Used to clip numeric head independently from text params
//======================================================//

void LanguageModel::scaleGradientsByType(float scale, ParamGroupType type) {
    if (parameter_groups_.empty()) {
        buildParameterGroups();
    }
    
    const int threads = 256;
    
    for (auto& group : parameter_groups_) {
        if (!group.grads() || group.size() == 0) continue;
        if (group.type != type) continue;
        
        const int blocks = (group.size() + threads - 1) / threads;
        scaleKernel<<<blocks, threads, 0, training_state_.stream_ctrl.getPrimaryStream()>>>(
            group.grads(), scale, static_cast<int>(group.size())
        );
    }
}

//======================================================//
//  scaleGradientsExcludingType - Scale all param groups EXCEPT a specific type
//  Issue #134: Used to clip text params without affecting numeric head
//======================================================//

void LanguageModel::scaleGradientsExcludingType(float scale, ParamGroupType exclude_type) {
    if (parameter_groups_.empty()) {
        buildParameterGroups();
    }
    
    const int threads = 256;
    
    for (auto& group : parameter_groups_) {
        if (!group.grads() || group.size() == 0) continue;
        if (group.type == exclude_type) continue;
        
        const int blocks = (group.size() + threads - 1) / threads;
        scaleKernel<<<blocks, threads, 0, training_state_.stream_ctrl.getPrimaryStream()>>>(
            group.grads(), scale, static_cast<int>(group.size())
        );
    }
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
        if (!group.grads() || group.size() == 0) continue;
        
        const int blocks = (group.size() + threads - 1) / threads;
        clampKernel<<<blocks, threads, 0, training_state_.stream_ctrl.getPrimaryStream()>>>(
            group.grads(), min_val, max_val, static_cast<int>(group.size())
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
