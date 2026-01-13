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
#include "../Layers/BackwardOps/BackwardOps_Orchestrator.hpp"
#include "../Layers/BackwardOps/BackwardContext.hpp"
#include "../Layers/Embedding/Embedding_GPU.hpp"
#include "../Shared/GradNorm/GradNormGPU.hpp"
#include "../Common/grim_scale_buffer.hpp"
#include "../Shared/LogRecorder/LogRecorder.hpp"
#include "../Shared/TensorContract/TensorContract_GPU.hpp"
#include "../Shared/Optimizers/AdamW/AdamW_Kernal_GPU.hpp"
#include "../../../control/ai_config_paths.hpp"
#include "../Shared/HyperParameters/HyperParameters_GPU.hpp"

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
    
    // Zero embedding gradients
    if (training_state_.embedding_grads) {
        cudaMemsetAsync(training_state_.embedding_grads, 0, 
                       training_state_.embedding_grad_size * sizeof(float),
                       training_state_.stream_ctrl.getPrimaryStream());
    }
    
    // Zero LM head gradients
    if (training_state_.lm_head_weight_grads) {
        cudaMemsetAsync(training_state_.lm_head_weight_grads, 0,
                       cfg.vocab_size * cfg.d_model * sizeof(float),
                       training_state_.stream_ctrl.getPrimaryStream());
    }
    if (training_state_.lm_head_bias_grads) {
        cudaMemsetAsync(training_state_.lm_head_bias_grads, 0,
                       cfg.vocab_size * sizeof(float),
                       training_state_.stream_ctrl.getPrimaryStream());
    }

    if (training_state_.numeric_head_weight_grads) {
        cudaMemsetAsync(training_state_.numeric_head_weight_grads, 0,
                       cfg.d_model * sizeof(float),
                       training_state_.stream_ctrl.getPrimaryStream());
    }
    if (training_state_.numeric_head_bias_grads) {
        cudaMemsetAsync(training_state_.numeric_head_bias_grads, 0,
                       sizeof(float),
                       training_state_.stream_ctrl.getPrimaryStream());
    }
    
    // Zero encoder layer gradients
    for (int layer = 0; layer < cfg.num_layers; ++layer) {
        GPUGrimEncoder* gpu_encoder = &getGpuEncoder();
        GPUEncoderLayer* enc = gpu_encoder ? gpu_encoder->getLayer(layer) : nullptr;
        if (enc && enc->getRMS1GammaGrad()) {
            cudaMemsetAsync(enc->getRMS1GammaGrad(), 0, cfg.d_model * sizeof(float), training_state_.stream_ctrl.getPrimaryStream());
        }
        if (enc && enc->getRMS2GammaGrad()) {
            cudaMemsetAsync(enc->getRMS2GammaGrad(), 0, cfg.d_model * sizeof(float), training_state_.stream_ctrl.getPrimaryStream());
        }
        
        // GQA: QKV weight grad size is total_qkv_dim * d_model
        if (training_state_.attn_qkv_weight_grads[layer]) {
            size_t qkv_weight_size = static_cast<size_t>(total_qkv_dim) * cfg.d_model;
            cudaMemsetAsync(training_state_.attn_qkv_weight_grads[layer], 0,
                           qkv_weight_size * sizeof(float), training_state_.stream_ctrl.getPrimaryStream());
        }
        if (training_state_.attn_qkv_bias_grads[layer]) {
            cudaMemsetAsync(training_state_.attn_qkv_bias_grads[layer], 0,
                           total_qkv_dim * sizeof(float), training_state_.stream_ctrl.getPrimaryStream());
        }
        if (training_state_.attn_out_weight_grads[layer]) {
            cudaMemsetAsync(training_state_.attn_out_weight_grads[layer], 0,
                           cfg.d_model * cfg.d_model * sizeof(float), training_state_.stream_ctrl.getPrimaryStream());
        }
        if (training_state_.attn_out_bias_grads[layer]) {
            cudaMemsetAsync(training_state_.attn_out_bias_grads[layer], 0,
                           cfg.d_model * sizeof(float), training_state_.stream_ctrl.getPrimaryStream());
        }
        if (training_state_.ffn_w1_grads[layer]) {
            cudaMemsetAsync(training_state_.ffn_w1_grads[layer], 0,
                           cfg.d_model * cfg.d_ff * sizeof(float), training_state_.stream_ctrl.getPrimaryStream());
        }
        if (training_state_.ffn_b1_grads[layer]) {
            cudaMemsetAsync(training_state_.ffn_b1_grads[layer], 0,
                           cfg.d_ff * sizeof(float), training_state_.stream_ctrl.getPrimaryStream());
        }
        if (training_state_.ffn_w2_grads[layer]) {
            cudaMemsetAsync(training_state_.ffn_w2_grads[layer], 0,
                           cfg.d_ff * cfg.d_model * sizeof(float), training_state_.stream_ctrl.getPrimaryStream());
        }
        if (training_state_.ffn_b2_grads[layer]) {
            cudaMemsetAsync(training_state_.ffn_b2_grads[layer], 0,
                           cfg.d_model * sizeof(float), training_state_.stream_ctrl.getPrimaryStream());
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
    // BUGFIX: Don't reset grad_metrics_ready_ here - it should persist until next sync
    // Old code: grad_metrics_ready_ = false;  // This caused cached grad_norm to always return 0.0f
    
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

    // Initialize backward context for 3-phase orchestrator
    GPUGrimEncoder* gpu_encoder = &getGpuEncoder();
    if (!gpu_encoder) {
        BWD_ERROR("[backward] CRITICAL: GPU encoder is nullptr!");
        return;
    }

    EmbeddingRuntime* embedding_runtime = &getGpuEmbedder();

    Backward::BackwardContext ctx = Backward::initBackwardContextRaw(
        &config_,
        &training_state_,
        gpu_encoder,
        scratch_block_layer_.get(),
        embedding_runtime,
        training_state_.cublas_handle,
        training_state_.stream_ctrl.getPrimaryStream(),
        batch_size,
        seq_len,
        accumulate,
        grad_scale,
        step
    );

    // Execute 3-phase backward pass
    Backward::BackwardStatus status = Backward::executeBackward(ctx);

    // ALWAYS clear sequence weights after a backward attempt
    training_state_.sequence_weight_count = 0;

    if (status != Backward::BackwardStatus::SUCCESS) {
        BWD_ERROR("[backward] 3-phase backward FAILED: " << Backward::statusToString(status));
        BWD_ERROR("  Error: " << ctx.error_message);
        if (ctx.error_layer >= 0) {
            BWD_ERROR("  Error layer: " << ctx.error_layer);
        }
        BWD_ERROR(Backward::getBackwardErrorReport(ctx));
        return;
    }

    BWD_INFO("[backward] COMPLETE");
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
    // When tie_embeddings=true, embedding_grads == lm_head_weight_grads (same pointer)
    // Registering both would double-count gradients and corrupt optimizer state
    EmbeddingRuntime* embedding_runtime = &getGpuEmbedder();
    if (!cfg.tie_embeddings && training_state_.embedding_grads && embedding_runtime) {
        ParameterGroup emb_group;
        emb_group.name = "embedding";
        emb_group.weights = embedding_runtime->token_buffer;
        emb_group.grads = training_state_.embedding_grads;
        emb_group.size = training_state_.embedding_grad_size;
        emb_group.m_state = nullptr;
        emb_group.v_state = nullptr;
        emb_group.type = ParamGroupType::EMBEDDING;
        parameter_groups_.push_back(emb_group);
    }

    // LM head weights (includes tied embedding grads when tie_embeddings=true)
    if (training_state_.lm_head_weight_grads && training_state_.lm_head_weights) {
        ParameterGroup lm_group;
        lm_group.name = cfg.tie_embeddings ? "embedding_lm_head_tied" : "lm_head_weight";
        lm_group.weights = training_state_.lm_head_weights;
        lm_group.grads = training_state_.lm_head_weight_grads;
        lm_group.size = cfg.vocab_size * cfg.d_model;
        lm_group.m_state = nullptr;
        lm_group.v_state = nullptr;
        // When tie_embeddings=true: This buffer serves as BOTH lm_head AND embedding weights
        //   - Receives dense gradients from LM head backward pass (output layer)
        //   - Receives sparse gradients from embedding backward pass (input layer, accumulated)
        //   - Gradient norm reported under LM_HEAD type (physical buffer name)
        // When tie_embeddings=false: This buffer is ONLY the lm_head projection weights
        //   - Embedding weights are separate and registered as EMBEDDING type above
        lm_group.type = ParamGroupType::LM_HEAD;
        parameter_groups_.push_back(lm_group);
    }

    if (cfg.numeric_head_enabled && training_state_.numeric_head_weight_grads &&
        training_state_.numeric_head_weights) {
        ParameterGroup numeric_group;
        numeric_group.name = "numeric_head_weight";
        numeric_group.weights = training_state_.numeric_head_weights;
        numeric_group.grads = training_state_.numeric_head_weight_grads;
        numeric_group.size = cfg.d_model;
        numeric_group.m_state = nullptr;
        numeric_group.v_state = nullptr;
        numeric_group.type = ParamGroupType::NUMERIC_HEAD;
        parameter_groups_.push_back(numeric_group);
    }

    // Per-layer parameters
    for (int layer = 0; layer < cfg.num_layers; ++layer) {
        GPUEncoderLayer* enc = gpu_encoder->getLayer(layer);
        if (!enc) continue;

        // QKV weights
        if (training_state_.attn_qkv_weight_grads[layer] && enc->getAttnWqkv()) {
            ParameterGroup qkv_group;
            qkv_group.name = "layer" + std::to_string(layer) + "_qkv_weight";
            qkv_group.weights = enc->getAttnWqkv();
            qkv_group.grads = training_state_.attn_qkv_weight_grads[layer];
            qkv_group.size = total_qkv_dim * cfg.d_model;
            qkv_group.m_state = nullptr;
            qkv_group.v_state = nullptr;
            qkv_group.type = ParamGroupType::ATTENTION;
            parameter_groups_.push_back(qkv_group);
        }

        // Wo weights
        if (training_state_.attn_out_weight_grads[layer] && enc->getAttnWo()) {
            ParameterGroup wo_group;
            wo_group.name = "layer" + std::to_string(layer) + "_wo_weight";
            wo_group.weights = enc->getAttnWo();
            wo_group.grads = training_state_.attn_out_weight_grads[layer];
            wo_group.size = cfg.d_model * cfg.d_model;
            wo_group.m_state = nullptr;
            wo_group.v_state = nullptr;
            wo_group.type = ParamGroupType::ATTENTION;
            parameter_groups_.push_back(wo_group);
        }

        // FFN W1
        if (training_state_.ffn_w1_grads[layer] && enc->getFFNW1()) {
            ParameterGroup w1_group;
            w1_group.name = "layer" + std::to_string(layer) + "_ffn_w1";
            w1_group.weights = enc->getFFNW1();
            w1_group.grads = training_state_.ffn_w1_grads[layer];
            w1_group.size = cfg.d_model * cfg.d_ff;
            w1_group.m_state = nullptr;
            w1_group.v_state = nullptr;
            w1_group.type = ParamGroupType::FFN;
            parameter_groups_.push_back(w1_group);
        }

        // FFN W2
        if (training_state_.ffn_w2_grads[layer] && enc->getFFNW2()) {
            ParameterGroup w2_group;
            w2_group.name = "layer" + std::to_string(layer) + "_ffn_w2";
            w2_group.weights = enc->getFFNW2();
            w2_group.grads = training_state_.ffn_w2_grads[layer];
            w2_group.size = cfg.d_ff * cfg.d_model;
            w2_group.m_state = nullptr;
            w2_group.v_state = nullptr;
            w2_group.type = ParamGroupType::FFN;
            parameter_groups_.push_back(w2_group);
        }

        // RMSNorm gamma 1
        if (enc->getRMS1GammaGrad() && enc->getRMS1Gamma()) {
            ParameterGroup rms1_group;
            rms1_group.name = "layer" + std::to_string(layer) + "_rms1_gamma";
            rms1_group.weights = enc->getRMS1Gamma();
            rms1_group.grads = enc->getRMS1GammaGrad();
            rms1_group.size = cfg.d_model;
            rms1_group.m_state = nullptr;
            rms1_group.v_state = nullptr;
            rms1_group.type = ParamGroupType::RMSNORM;
            parameter_groups_.push_back(rms1_group);
        }

        // RMSNorm gamma 2
        if (enc->getRMS2GammaGrad() && enc->getRMS2Gamma()) {
            ParameterGroup rms2_group;
            rms2_group.name = "layer" + std::to_string(layer) + "_rms2_gamma";
            rms2_group.weights = enc->getRMS2Gamma();
            rms2_group.grads = enc->getRMS2GammaGrad();
            rms2_group.size = cfg.d_model;
            rms2_group.m_state = nullptr;  // Will be set after TrainingState allocation
            rms2_group.v_state = nullptr;
            rms2_group.type = ParamGroupType::RMSNORM;
            parameter_groups_.push_back(rms2_group);
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
    if (training_state_.final_rms_gamma && training_state_.final_rms_gamma_grads) {
        ParameterGroup final_rms_group;
        final_rms_group.name = "final_rms_gamma";
        final_rms_group.weights = training_state_.final_rms_gamma;
        final_rms_group.grads = training_state_.final_rms_gamma_grads;
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
        parameter_groups_[i].m_state = training_state_.optimizer_m_states[i];
        parameter_groups_[i].v_state = training_state_.optimizer_v_states[i];
    }
    
    // Note: Gradient norm computation moved to TrainingState::gradnorm_ctrl (GradNormController)
    // Old d_grad_norm_sums_/h_grad_norm_sums_ buffers removed per Rule 20 (no backwards compatibility)

    BWD_INFO("[buildParameterGroups] Built " << parameter_groups_.size() << " parameter groups");
}

//======================================================//
//  configureUpdateProbe / disableUpdateProbe
//======================================================//

void LanguageModel::configureUpdateProbe(const std::string& group_name, size_t sample_elems) {
    update_probe_group_name_ = group_name;
    update_probe_sample_elems_ = sample_elems;
    update_probe_ready_ = false;
    
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
    if (update_probe_group_index_ != static_cast<size_t>(-1) && 
        update_probe_group_index_ < parameter_groups_.size()) {
        const auto& probe_group = parameter_groups_[update_probe_group_index_];
        if (probe_group.weights && update_probe_sample_elems_ > 0) {
            cudaMemcpyAsync(update_probe_weights_before_.data(), 
                          probe_group.weights, 
                          update_probe_sample_elems_ * sizeof(float),
                          cudaMemcpyDeviceToHost, training_state_.stream_ctrl.getPrimaryStream());
        }
    }
    
    for (size_t i = 0; i < parameter_groups_.size(); ++i) {
        auto& group = parameter_groups_[i];
        if (!group.weights || !group.grads || group.size == 0) continue;
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

        launchAdamWKernel(
            group.weights,
            group.grads,
            group.m_state,
            group.v_state,
            group.size,
            learning_rate,
            weight_decay,
            optimizer_state->step,
            stream
        );

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
            // Copy weights after update
            cudaMemcpy(update_probe_weights_after_.data(), 
                      probe_group.weights, 
                      update_probe_sample_elems_ * sizeof(float),
                      cudaMemcpyDeviceToHost);
            
            // Copy gradient sample
            cudaMemcpy(update_probe_grad_sample_.data(), 
                      probe_group.grads, 
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
            if (probe_group.m_state && probe_group.v_state) {
                constexpr size_t kProbeStateSamples = 4;
                const size_t state_samples = std::min(kProbeStateSamples, probe_group.size);
                std::array<float, kProbeStateSamples> m_sample{};
                std::array<float, kProbeStateSamples> v_sample{};
                cudaMemcpy(m_sample.data(), probe_group.m_state,
                           state_samples * sizeof(float), cudaMemcpyDeviceToHost);
                cudaMemcpy(v_sample.data(), probe_group.v_state,
                           state_samples * sizeof(float), cudaMemcpyDeviceToHost);

                std::ostringstream state_oss;
                state_oss << "[UpdateProbeState] group='" << probe_group.name
                          << "' step=" << optimizer_state->step
                          << " m_state[0:" << state_samples << "]=[";
                for (size_t i = 0; i < state_samples; ++i) {
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
        training_state_.sequence_weights,
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
