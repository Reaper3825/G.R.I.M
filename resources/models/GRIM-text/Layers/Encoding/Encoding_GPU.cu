//======================================================//
//  Encoding_GPU.cu
//  PRODUCTION-READY Transformer Encoder Layer
//  
//  This implementation:
//    - Uses Flash Attention directly (NOT GPUMultiHeadAttention)
//    - Uses modular QKV projection from Layers/Attention/
//    - Uses RMSNorm (NOT LayerNorm - no beta parameter!)
//    - Validates EVERYTHING and screams on errors
//    - GQA-native from the start
//======================================================//

#include "Encoding_GPU.hpp"
#include "../../GRIM/grim_language_model_cuda.hpp"  // EncoderLayerCache definition
#include "../../Shared/TensorContract/TensorContract_GPU.hpp"
#include "../FlashAttention/Flash_Attention_Kernal.hpp"
#include "../FeedForward/Feed_Forward_GPU.hpp"
#include "../../Shared/PBM/PositionalBiasMethod.hpp"
#include "../../Shared/StreamController/StreamController_GPU.hpp"
#include "../../Shared/TensorConversion/TensorConversion.hpp"
#include <cuda_runtime.h>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>
#include <cmath>

//======================================================//
//  Issue #37 DIAGNOSTIC: Hidden State Alignment Tracker
//  Logs how hidden states align with W[277] at each stage
//  DISABLED by default - set kEnableHiddenAlignDiag = true to enable
//======================================================//
namespace {
    constexpr bool kEnableHiddenAlignDiag = false;  // Set true to enable [HiddenAlign] logs
    constexpr bool kEnableEncoderStepLogs = false;  // Set true to enable [EncoderFwd] step logs
    
    // Shared W[277] reference - set once per forward pass
    static const float* s_w277_ref = nullptr;
    static int s_w277_d_model = 0;
    static float s_w277_norm = 0.0f;
    static int s_layer_diag_count = 0;
    static constexpr int kMaxDiagLogs = 120;  // First 2 batches * 12 layers
    
    void setW277Reference(const float* lm_weights, int vocab_size, int d_model, cudaStream_t stream) {
        if constexpr (!kEnableHiddenAlignDiag) return;
        constexpr int kToken277 = 277;
        if (!lm_weights || kToken277 >= vocab_size) {
            s_w277_ref = nullptr;
            return;
        }
        // Copy W[277] row to static host buffer
        static std::vector<float> h_w277;
        h_w277.resize(d_model);
        s_w277_d_model = d_model;
        
        cudaStreamSynchronize(stream);
        cudaMemcpy(h_w277.data(), lm_weights + static_cast<size_t>(kToken277) * d_model,
                   d_model * sizeof(float), cudaMemcpyDeviceToHost);
        
        // Compute norm
        double sum_sq = 0.0;
        for (int i = 0; i < d_model; ++i) sum_sq += h_w277[i] * h_w277[i];
        s_w277_norm = static_cast<float>(std::sqrt(sum_sq));
        s_w277_ref = h_w277.data();
    }
    
    void logHiddenStateAlignment(const char* stage, int layer_idx, 
                                  const float* d_hidden, int total_tokens, int d_model,
                                  cudaStream_t stream) {
        if constexpr (!kEnableHiddenAlignDiag) return;
        if (!s_w277_ref || s_w277_d_model != d_model || s_layer_diag_count >= kMaxDiagLogs) return;
        
        cudaStreamSynchronize(stream);
        
        // Sample first 5 tokens
        const int num_sample = std::min(5, total_tokens);
        std::vector<float> h_hidden(static_cast<size_t>(num_sample) * d_model);
        cudaMemcpy(h_hidden.data(), d_hidden, h_hidden.size() * sizeof(float), cudaMemcpyDeviceToHost);
        
        // Compute alignment for each sampled token
        float total_dot = 0.0f, total_cos = 0.0f, total_h_norm = 0.0f;
        for (int t = 0; t < num_sample; ++t) {
            const float* h = h_hidden.data() + t * d_model;
            double dot = 0.0, h_norm_sq = 0.0;
            for (int i = 0; i < d_model; ++i) {
                dot += h[i] * s_w277_ref[i];
                h_norm_sq += h[i] * h[i];
            }
            float h_norm = static_cast<float>(std::sqrt(h_norm_sq));
            float cos_sim = (h_norm > 1e-8f && s_w277_norm > 1e-8f) 
                           ? static_cast<float>(dot) / (h_norm * s_w277_norm) : 0.0f;
            total_dot += static_cast<float>(dot);
            total_cos += cos_sim;
            total_h_norm += h_norm;
        }
        
        fprintf(stderr, "[HiddenAlign] layer=%d stage=%s tokens=%d | "
                "mean_dot=%.4f mean_cos=%.4f mean_h_norm=%.4f w277_norm=%.4f\n",
                layer_idx, stage, total_tokens,
                total_dot / num_sample, total_cos / num_sample, 
                total_h_norm / num_sample, s_w277_norm);
    }
    
    void resetLayerDiagCount() { s_layer_diag_count = 0; }
    void incrementLayerDiagCount() { ++s_layer_diag_count; }
}
//======================================================// 

// External kernel declaration (global scope - defined in BackwardKernels.cu)
void launchBiasSumGradient(const float* grad_output, float* grad_bias,
                          int total_tokens, int hidden_dim,
                          cudaStream_t stream);

namespace GRIM {

static_assert(!GRIM::HyperParameters::QK_NORMALIZATION_ENABLED,
              "FlashAttention v2 forward does not support QK normalization.");
static_assert(GRIM::HyperParameters::SOFTMAX_TEMPERATURE == 1.0f,
              "FlashAttention v2 forward requires softmax_temperature=1.0f.");

//======================================================//
//  CUDA Error Checking
//======================================================//

#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        char msg[512]; \
        snprintf(msg, sizeof(msg), "CUDA ERROR at %s:%d - %s: %s", \
                 __FILE__, __LINE__, #call, cudaGetErrorString(err)); \
        throw std::runtime_error(msg); \
    } \
} while(0)

#define CUBLAS_CHECK(call) do { \
    cublasStatus_t status = (call); \
    if (status != CUBLAS_STATUS_SUCCESS) { \
        char msg[256]; \
        snprintf(msg, sizeof(msg), "cuBLAS ERROR at %s:%d - %s: status=%d", \
                 __FILE__, __LINE__, #call, (int)status); \
        throw std::runtime_error(msg); \
    } \
} while(0)

//======================================================//
//  Extern declarations - use shared kernels from BackwardKernels.cu
//  Rule 20: No duplicate code, use centralized implementations
//======================================================//

// From BackwardKernels.cu
extern "C" void launchResidualAdd(
    const float* input,
    const float* residual, 
    float* output,
    int total_size,
    cudaStream_t stream
);

//======================================================//
//  Local kernels (not duplicated elsewhere)
//======================================================//

__global__ void fillOnesKernel(float* data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        data[idx] = 1.0f;
    }
}

__global__ void addBiasKernel(float* data, const float* bias, int tokens, int dim) {
    int token_idx = blockIdx.x;
    int dim_idx = threadIdx.x;
    if (token_idx < tokens && dim_idx < dim) {
        data[token_idx * dim + dim_idx] += bias[dim_idx];
    }
}

static void launchAddBias(float* data, const float* bias, int tokens, int dim, 
                          cudaStream_t stream) {
    dim3 grid(tokens);
    dim3 block(dim);
    if (dim > 1024) {
        // For large dims, use 2D grid
        grid = dim3(tokens, (dim + 255) / 256);
        block = dim3(256);
    }
    addBiasKernel<<<grid, block, 0, stream>>>(data, bias, tokens, dim);
    CUDA_CHECK(cudaGetLastError());
}

//======================================================//
//  RMSNorm Forward/Backward (calls into RMSNorm_Kernel_GPU.cu)
//======================================================//

// NOTE: RMSNorm forward/backward are declared in RMSNorm_Kernel_GPU.hpp
// Include that header if needed. The extern declarations below are REMOVED
// as backward pass is now handled via TensorView-based RMSNormBackwardParams.

//======================================================//
//  EncodingLayer Implementation
//======================================================//

EncodingLayer::EncodingLayer(const EncodingConfig& cfg) {
    setConfig(cfg);
}

EncodingLayer::~EncodingLayer() {
    freeWeights();
    // NOTE: config_.cublas_handle is NOT owned by EncodingLayer (Rule 22)
    // TrainingState owns it - do NOT destroy!
}

EncodingLayer::EncodingLayer(EncodingLayer&& other) noexcept
    : config_(other.config_)
    , weights_allocated_(other.weights_allocated_)
    , rms1_gamma_(std::move(other.rms1_gamma_))
    , rms2_gamma_(std::move(other.rms2_gamma_))
    , W_qkv_(std::move(other.W_qkv_))
    , b_qkv_(std::move(other.b_qkv_))
    , W_o_(std::move(other.W_o_))
    , b_o_(std::move(other.b_o_))
    , ffn_(std::move(other.ffn_))
    , workspace_(other.workspace_)
    , workspace_bytes_(other.workspace_bytes_)
{
    // Null out the moved-from object
    other.config_.cublas_handle = nullptr;
    other.weights_allocated_ = false;
    other.workspace_ = nullptr;
    other.workspace_bytes_ = 0;
}

EncodingLayer& EncodingLayer::operator=(EncodingLayer&& other) noexcept {
    if (this != &other) {
        freeWeights();
        // NOTE: config_.cublas_handle is NOT owned - do NOT destroy (Rule 22)
        
        config_ = other.config_;
        weights_allocated_ = other.weights_allocated_;
        config_.cublas_handle = other.config_.cublas_handle;
        rms1_gamma_ = std::move(other.rms1_gamma_);
        rms2_gamma_ = std::move(other.rms2_gamma_);
        W_qkv_ = std::move(other.W_qkv_);
        b_qkv_ = std::move(other.b_qkv_);
        W_o_ = std::move(other.W_o_);
        b_o_ = std::move(other.b_o_);
        ffn_ = std::move(other.ffn_);
        workspace_ = other.workspace_;
        workspace_bytes_ = other.workspace_bytes_;
        
        other.config_.cublas_handle = nullptr;
        other.weights_allocated_ = false;
        other.workspace_ = nullptr;
        other.workspace_bytes_ = 0;
    }
    return *this;
}

void EncodingLayer::setConfig(const EncodingConfig& cfg) {
    cfg.validate("EncodingLayer::setConfig");
    config_ = cfg;
}

void EncodingLayer::freeWeights() {
    // Tensor handles its own memory cleanup via destructor
    // Reset to empty tensors
    rms1_gamma_ = Tensor();
    rms2_gamma_ = Tensor();
    W_qkv_ = Tensor();
    b_qkv_ = Tensor();
    W_o_ = Tensor();
    b_o_ = Tensor();
    ffn_.reset();
    weights_allocated_ = false;
}

void EncodingLayer::allocateWeights() {
    config_.validate("EncodingLayer::allocateWeights");
    
    if (weights_allocated_) {
        throw std::runtime_error("EncodingLayer::allocateWeights: weights already allocated! "
                                 "Call freeWeights() first if you want to reallocate.");
    }
    
    const int d_model = config_.d_model;
    const int kv_dim = config_.kvDim();
    const int d_ff = config_.d_ff;
    
    // Use centralized cuBLAS handle (Rule 22: NO local handle creation)
    if (!config_.cublas_handle) {
        std::cerr << "ERROR in EncodingLayer::allocateWeights: config_.cublas_handle is NULL!" << std::endl;
        std::cerr << "  config_.stream = " << config_.stream << std::endl;
        std::cerr << "  d_model = " << d_model << ", num_heads = " << config_.num_heads << std::endl;
        throw std::runtime_error("EncodingLayer::allocateWeights: config_.cublas_handle is NULL. "
                                 "MUST pass training_state.cublas_handle per Rule 22.");
    }
    if (!config_.stream) {
        std::cerr << "ERROR in EncodingLayer::allocateWeights: config_.stream is NULL!" << std::endl;
        throw std::runtime_error("EncodingLayer::allocateWeights: config_.stream is NULL");
    }
    StreamController::fatalIfDefaultStream(config_.stream, "EncodingLayer::allocateWeights");
    // Rule 20: Don't store copy of handle, use config_.cublas_handle directly
    
    // Create shapes for weight tensors
    // RMSNorm gamma: 1D vector stored as [1, d_model] BSM
    TensorContract::Shape2D gamma_2d{1, d_model};
    TensorContract::TensorShape gamma_shape(TensorContract::Layout::BSM, gamma_2d);
    
    // Attention weights
    const int qkv_out_dim = d_model + 2 * kv_dim;
    TensorContract::Shape2D qkv_weight_2d{qkv_out_dim, d_model};
    TensorContract::Shape2D qkv_bias_2d{1, qkv_out_dim};
    TensorContract::Shape2D o_weight_2d{d_model, d_model};
    TensorContract::Shape2D o_bias_2d{1, d_model};
    
    TensorContract::TensorShape qkv_weight_shape(TensorContract::Layout::BSM, qkv_weight_2d);
    TensorContract::TensorShape qkv_bias_shape(TensorContract::Layout::BSM, qkv_bias_2d);
    TensorContract::TensorShape o_weight_shape(TensorContract::Layout::BSM, o_weight_2d);
    TensorContract::TensorShape o_bias_shape(TensorContract::Layout::BSM, o_bias_2d);
    
    // RMSNorm gamma - initialized to 1.0, then set requires_grad
    rms1_gamma_ = Tensor::zeros(gamma_shape, true, config_.stream);
    rms2_gamma_ = Tensor::zeros(gamma_shape, true, config_.stream);
    
    // Fill gamma with ones via kernel
    int threads = 256;
    int blocks = (d_model + threads - 1) / threads;
    fillOnesKernel<<<blocks, threads, 0, config_.stream>>>(rms1_gamma_.data, d_model);
    fillOnesKernel<<<blocks, threads, 0, config_.stream>>>(rms2_gamma_.data, d_model);
    
    // Xavier initialization for attention weights
    W_qkv_ = Tensor::xavier_uniform(qkv_weight_shape, true, config_.stream);
    b_qkv_ = Tensor::zeros(qkv_bias_shape, true, config_.stream);
    
    // W_o: [d_model, d_model] output projection
    W_o_ = Tensor::xavier_uniform(o_weight_shape, true, config_.stream);
    b_o_ = Tensor::zeros(o_bias_shape, true, config_.stream);
    
    // FFN layer
    FeedForwardConfig ffn_cfg;
    ffn_cfg.d_model = d_model;
    ffn_cfg.d_ff = d_ff;
    ffn_cfg.stream = config_.stream;
    ffn_cfg.cublas_handle = config_.cublas_handle;  // CRITICAL: Must pass handle to FFN (Rule 22)
    ffn_ = std::make_unique<FeedForwardLayer>(ffn_cfg);
    ffn_->ensureWeightStorage();
    
    // AUTOGRAD MIGRATION: Allocate gradient buffers for all trainable tensors
    // This replaces the legacy cudaMalloc in InitTrainingState.cu
    rms1_gamma_.ensure_grad();
    rms2_gamma_.ensure_grad();
    W_qkv_.ensure_grad();
    b_qkv_.ensure_grad();
    W_o_.ensure_grad();
    b_o_.ensure_grad();
    // FFN gradients allocated via ffn_->ensureWeightStorage() -> FFN's own ensure_grad() calls
    
    weights_allocated_ = true;
}

void EncodingLayer::useExternalWeights(
    Tensor& rms1_gamma,
    Tensor& rms2_gamma,
    Tensor& qkv_weight,
    Tensor& qkv_bias,
    Tensor& out_weight,
    Tensor& out_bias,
    Tensor& ffn_w1,
    Tensor& ffn_b1,
    Tensor& ffn_w2,
    Tensor& ffn_b2
) {
    // Rule 20: Fail loud if already allocated own weights (prevents confusion)
    if (weights_allocated_ && !using_external_weights_) {
        throw std::runtime_error("EncodingLayer::useExternalWeights: Cannot switch to external weights "
                                 "after allocating own weights. Use freeWeights() first or never call allocateWeights().");
    }
    
    config_.validate("EncodingLayer::useExternalWeights");
    
    const int d_model = config_.d_model;
    const int kv_dim = config_.kvDim();
    const int d_ff = config_.d_ff;
    const int qkv_out_dim = d_model + 2 * kv_dim;
    
    // Validate shapes
    if (rms1_gamma.numel() != d_model) {
        throw std::invalid_argument("useExternalWeights: rms1_gamma size mismatch. Expected " + 
                                    std::to_string(d_model) + ", got " + std::to_string(rms1_gamma.numel()));
    }
    if (rms2_gamma.numel() != d_model) {
        throw std::invalid_argument("useExternalWeights: rms2_gamma size mismatch");
    }
    if (qkv_weight.numel() != static_cast<std::size_t>(qkv_out_dim) * d_model) {
        throw std::invalid_argument("useExternalWeights: qkv_weight size mismatch. Expected " +
                                    std::to_string(static_cast<std::size_t>(qkv_out_dim) * d_model) + 
                                    ", got " + std::to_string(qkv_weight.numel()));
    }
    if (out_weight.numel() != static_cast<std::size_t>(d_model) * d_model) {
        throw std::invalid_argument("useExternalWeights: out_weight size mismatch");
    }
    if (ffn_w1.numel() != static_cast<std::size_t>(d_ff) * d_model) {
        throw std::invalid_argument("useExternalWeights: ffn_w1 size mismatch");
    }
    if (ffn_w2.numel() != static_cast<std::size_t>(d_model) * d_ff) {
        throw std::invalid_argument("useExternalWeights: ffn_w2 size mismatch");
    }
    
    // Create view Tensors that reference the external buffers
    // NOTE: These Tensors do NOT own the data (owns_data=false)
    // The grad pointers also come from the external Tensors
    // ISSUE #59: Use share_grad() for proper shared_ptr semantics
    
    // RMSNorm gammas
    rms1_gamma_ = Tensor::from_ptr(rms1_gamma.data, rms1_gamma.shape, false, true);
    rms1_gamma_.share_grad(rms1_gamma);
    rms1_gamma_.owns_data = false;
    
    rms2_gamma_ = Tensor::from_ptr(rms2_gamma.data, rms2_gamma.shape, false, true);
    rms2_gamma_.share_grad(rms2_gamma);
    rms2_gamma_.owns_data = false;
    
    // QKV projection
    W_qkv_ = Tensor::from_ptr(qkv_weight.data, qkv_weight.shape, false, true);
    W_qkv_.share_grad(qkv_weight);
    W_qkv_.owns_data = false;
    
    if (qkv_bias.data) {
        b_qkv_ = Tensor::from_ptr(qkv_bias.data, qkv_bias.shape, false, true);
        b_qkv_.share_grad(qkv_bias);
        b_qkv_.owns_data = false;
    }
    
    // Output projection
    W_o_ = Tensor::from_ptr(out_weight.data, out_weight.shape, false, true);
    W_o_.share_grad(out_weight);
    W_o_.owns_data = false;
    
    if (out_bias.data) {
        b_o_ = Tensor::from_ptr(out_bias.data, out_bias.shape, false, true);
        b_o_.share_grad(out_bias);
        b_o_.owns_data = false;
    }
    
    // FFN - need to create the layer and set its external weights
    if (!ffn_) {
        FeedForwardConfig ffn_cfg;
        ffn_cfg.d_model = d_model;
        ffn_cfg.d_ff = d_ff;
        ffn_cfg.stream = config_.stream;
        ffn_cfg.cublas_handle = config_.cublas_handle;
        ffn_ = std::make_unique<FeedForwardLayer>(ffn_cfg);
    }
    ffn_->useExternalWeights(ffn_w1, ffn_b1, ffn_w2, ffn_b2);
    
    weights_allocated_ = true;
    using_external_weights_ = true;
    
    fprintf(stderr, "[EncodingLayer] Using external weights: qkv=[%zu], W_o=[%zu], ffn_w1=[%zu], ffn_w2=[%zu]\n",
            W_qkv_.numel(), W_o_.numel(), ffn_w1.numel(), ffn_w2.numel());
}

void EncodingLayer::validateReady(const char* context) const {
    if (!weights_allocated_) {
        throw std::runtime_error(std::string(context) + 
            ": weights not allocated! Call allocateWeights() first.");
    }
    if (!config_.cublas_handle) {
        throw std::runtime_error(std::string(context) + 
            ": cuBLAS handle not initialized in config!");
    }
}

std::size_t EncodingLayer::requiredWorkspaceBytes(int total_tokens, int seq_len) const {
    config_.validate("EncodingLayer::requiredWorkspaceBytes");
    
    if (total_tokens <= 0 || seq_len <= 0) {
        throw std::invalid_argument("requiredWorkspaceBytes: total_tokens and seq_len must be > 0");
    }
    
    const int batch_size = total_tokens / seq_len;
    const int d_model = config_.d_model;
    const int kv_dim = config_.kvDim();
    const int num_heads = config_.num_heads;
    const int num_kv_heads = config_.effectiveKVHeads();
    const int head_dim = config_.headDim();
    const int d_ff = config_.d_ff;
    
    std::size_t bytes = 0;
    
    // RMSNorm intermediates
    bytes += total_tokens * d_model * sizeof(float);  // ln1_out
    bytes += total_tokens * d_model * sizeof(float);  // ln2_out
    
    // QKV projection outputs [tokens, dim]
    bytes += total_tokens * d_model * sizeof(float);  // Q [tokens, d_model]
    bytes += total_tokens * kv_dim * sizeof(float);   // K [tokens, kv_dim]
    bytes += total_tokens * kv_dim * sizeof(float);   // V [tokens, kv_dim]
    
    // QKV reshaped to BHSD for Flash Attention
    bytes += batch_size * num_heads * seq_len * head_dim * sizeof(float);      // Q_bhsd
    bytes += batch_size * num_kv_heads * seq_len * head_dim * sizeof(float);   // K_bhsd
    bytes += batch_size * num_kv_heads * seq_len * head_dim * sizeof(float);   // V_bhsd
    
    // Attention output BHSD
    bytes += batch_size * num_heads * seq_len * head_dim * sizeof(float);      // attn_out_bhsd
    
    // Attention output reshaped [tokens, d_model]
    bytes += total_tokens * d_model * sizeof(float);  // attn_out
    
    // Residual
    bytes += total_tokens * d_model * sizeof(float);  // residual1
    
    // FFN intermediates
    bytes += total_tokens * d_ff * sizeof(float);     // pre_gelu
    bytes += total_tokens * d_ff * sizeof(float);     // post_gelu
    bytes += total_tokens * d_model * sizeof(float);  // ffn_out
    
    return bytes;
}

void EncodingLayer::setWorkspace(float* workspace, std::size_t bytes) {
    workspace_ = workspace;
    workspace_bytes_ = bytes;
}

//======================================================//
//  Forward Pass - Autograd Implementation with ForwardIntermediates (Issue #56 Fix)
//======================================================//

Tensor EncodingLayer::forward(const Tensor& input, int seq_len, cudaStream_t stream,
                               ForwardIntermediates& intermediates) {
    if constexpr (kEnableEncoderStepLogs) {
        fprintf(stderr, "[EncoderFwd] START total_tokens=%d seq_len=%d\n", 
                input.shape.flat.rows, seq_len);
    }
    validateReady("EncodingLayer::forward");
    
    // CRITICAL: Set autograd cuBLAS handle before any autograd::matmul calls
    // The handle must be set per-call since it's thread_local
    autograd::set_autograd_cublas_handle(config_.cublas_handle);
    if constexpr (kEnableEncoderStepLogs) {
        fprintf(stderr, "[EncoderFwd] autograd cuBLAS handle set: %p\n", (void*)config_.cublas_handle);
    }
    
    // Validate input
    if (!input.data) {
        throw std::runtime_error("EncodingLayer::forward: input.data is NULL");
    }
    if (!stream) {
        throw std::runtime_error("EncodingLayer::forward: stream is NULL");
    }
    
    const int total_tokens = input.shape.flat.rows;
    const int d_model = config_.d_model;
    
    if (input.shape.flat.cols != d_model) {
        throw std::runtime_error("EncodingLayer::forward: input d_model mismatch. "
                                 "Expected " + std::to_string(d_model) + 
                                 ", got " + std::to_string(input.shape.flat.cols));
    }
    if (total_tokens % seq_len != 0) {
        throw std::runtime_error("EncodingLayer::forward: total_tokens not divisible by seq_len");
    }
    
    const int batch_size = total_tokens / seq_len;
    const int num_heads = config_.num_heads;
    const int num_kv_heads = config_.effectiveKVHeads();
    const int head_dim = config_.headDim();
    if constexpr (kEnableEncoderStepLogs) {
        fprintf(stderr, "[EncoderFwd] validated: batch=%d heads=%d kv_heads=%d head_dim=%d\n", 
                batch_size, num_heads, num_kv_heads, head_dim);
    }
    
    //--------------------------------------------------
    // 1. RMSNorm1: input -> ln1_out
    // Issue #56: Store in intermediates to keep autograd graph alive
    //--------------------------------------------------
    if constexpr (kEnableEncoderStepLogs) fprintf(stderr, "[EncoderFwd] Step 1: RMSNorm1...\n");
    intermediates.ln1_out = autograd::rms_norm(input, rms1_gamma_, config_.rms_epsilon, stream);
    if constexpr (kEnableEncoderStepLogs) fprintf(stderr, "[EncoderFwd] Step 1: RMSNorm1 DONE\n");
    
    //--------------------------------------------------
    // 2. QKV Projection: ln1_out @ W_qkv^T + b_qkv
    //    W_qkv is [total_qkv_dim, d_model] so we compute ln1_out @ W_qkv^T
    // Issue #56: Store in intermediates to keep autograd graph alive
    //--------------------------------------------------
    if constexpr (kEnableEncoderStepLogs) {
        fprintf(stderr, "[EncoderFwd] Step 2: QKV matmul...\n");
        fprintf(stderr, "[EncoderFwd] Step 2: ln1_out.data=%p shape=[%d,%d] W_qkv.data=%p shape=[%d,%d]\n",
                (void*)intermediates.ln1_out.data, intermediates.ln1_out.shape.flat.rows, intermediates.ln1_out.shape.flat.cols,
                (void*)W_qkv_.data, W_qkv_.shape.flat.rows, W_qkv_.shape.flat.cols);
        fflush(stderr);
    }
    // Use transpose_b=true since W_qkv is [qkv_dim, d_model] and we need [tokens, d_model] @ [d_model, qkv_dim]
    intermediates.qkv_out = autograd::matmul(intermediates.ln1_out, W_qkv_, stream, nullptr, nullptr, true);
    if constexpr (kEnableEncoderStepLogs) fprintf(stderr, "[EncoderFwd] Step 2: QKV matmul DONE, adding bias...\n");
    // Add bias: qkv_out = qkv_out + b_qkv (broadcast)
    // TODO: Need autograd::bias_add for proper gradient tracking
    launchFFNBiasAdd(intermediates.qkv_out.data, b_qkv_.data, total_tokens, 
                     config_.d_model + 2 * config_.kvDim(), stream);
    if constexpr (kEnableEncoderStepLogs) fprintf(stderr, "[EncoderFwd] Step 2: QKV bias DONE\n");
    
    //--------------------------------------------------
    // 3. Split QKV and reshape to BHSD for attention
    //    qkv_out is [total_tokens, d_model + 2*kv_dim]
    //    Q: [total_tokens, 0:d_model]
    //    K: [total_tokens, d_model:d_model+kv_dim]  
    //    V: [total_tokens, d_model+kv_dim:end]
    // Issue #56: Store Q, K, V in intermediates
    //--------------------------------------------------
    if constexpr (kEnableEncoderStepLogs) fprintf(stderr, "[EncoderFwd] Step 3: Split QKV...\n");
    const int kv_dim = config_.kvDim();
    
    // Create Tensor views for Q, K, V (slices of qkv_out)
    intermediates.Q = Tensor::empty(TensorContract::TensorShape::make_BSM(total_tokens, d_model), true, stream);
    intermediates.K = Tensor::empty(TensorContract::TensorShape::make_BSM(total_tokens, kv_dim), true, stream);
    intermediates.V = Tensor::empty(TensorContract::TensorShape::make_BSM(total_tokens, kv_dim), true, stream);
    
    // Extract Q, K, V from fused QKV output using TensorConvert
    // For GQA: qkv_out is [tokens, d_model + 2*kv_dim], not [tokens, 3*d_model]
    // We need a custom split that handles different Q vs K/V sizes
    {
        // Q is first d_model columns
        // K is next kv_dim columns  
        // V is final kv_dim columns
        const int total_qkv_dim = d_model + 2 * kv_dim;
        const float* src = intermediates.qkv_out.data;
        // Manual split via cudaMemcpy2D (row-major slicing)
        // Q: copy d_model elements starting at offset 0
        CUDA_CHECK(cudaMemcpy2DAsync(
            intermediates.Q.data, d_model * sizeof(float),          // dst, dpitch
            src, total_qkv_dim * sizeof(float),       // src, spitch
            d_model * sizeof(float), total_tokens,    // width, height
            cudaMemcpyDeviceToDevice, stream));
        // K: copy kv_dim elements starting at offset d_model
        CUDA_CHECK(cudaMemcpy2DAsync(
            intermediates.K.data, kv_dim * sizeof(float),
            src + d_model, total_qkv_dim * sizeof(float),
            kv_dim * sizeof(float), total_tokens,
            cudaMemcpyDeviceToDevice, stream));
        // V: copy kv_dim elements starting at offset d_model + kv_dim
        CUDA_CHECK(cudaMemcpy2DAsync(
            intermediates.V.data, kv_dim * sizeof(float),
            src + d_model + kv_dim, total_qkv_dim * sizeof(float),
            kv_dim * sizeof(float), total_tokens,
            cudaMemcpyDeviceToDevice, stream));
    }
    if constexpr (kEnableEncoderStepLogs) fprintf(stderr, "[EncoderFwd] Step 3: Split QKV DONE, reshaping to BHSD...\n");
    
    // Reshape to BHSD for Flash Attention
    // Issue #56: Store in intermediates
    intermediates.Q_bhsd = Tensor::empty(TensorContract::TensorShape::make_BHSD(batch_size, num_heads, seq_len, head_dim), true, stream);
    intermediates.K_bhsd = Tensor::empty(TensorContract::TensorShape::make_BHSD(batch_size, num_kv_heads, seq_len, head_dim), true, stream);
    intermediates.V_bhsd = Tensor::empty(TensorContract::TensorShape::make_BHSD(batch_size, num_kv_heads, seq_len, head_dim), true, stream);
    
    launchQKVReshapeToBHSD(intermediates.Q.data, intermediates.K.data, intermediates.V.data, 
                           intermediates.Q_bhsd.data, intermediates.K_bhsd.data, intermediates.V_bhsd.data,
                           batch_size, seq_len, num_heads, num_kv_heads, head_dim, stream);
    if constexpr (kEnableEncoderStepLogs) fprintf(stderr, "[EncoderFwd] Step 3b: Reshape to BHSD DONE\n");
    
    //--------------------------------------------------
    // 3b. Apply RoPE rotation to Q and K
    //--------------------------------------------------
    if constexpr (kEnableEncoderStepLogs) fprintf(stderr, "[EncoderFwd] Step 3c: RoPE rotation...\n");
    if (config_.pos_encoding && config_.pos_encoding->valid && 
        config_.pos_encoding->rope_inv_freq != nullptr && config_.pos_encoding->rotary_dim > 0) {
        PBM::launchRoPERotationGQA(
            intermediates.Q_bhsd.data, intermediates.K_bhsd.data,
            config_.pos_encoding->rope_inv_freq,
            batch_size, num_heads, num_kv_heads, seq_len, head_dim,
            config_.pos_encoding->rotary_dim, stream);
    } else {
        throw std::runtime_error("EncodingLayer::forward: RoPE not initialized");
    }
    if constexpr (kEnableEncoderStepLogs) fprintf(stderr, "[EncoderFwd] Step 3c: RoPE DONE\n");
    
    //--------------------------------------------------
    // 4. Flash Attention: Q, K, V -> attn_out
    //--------------------------------------------------
    if constexpr (kEnableEncoderStepLogs) fprintf(stderr, "[EncoderFwd] Step 4: Flash Attention...\n");
    
    // RULE 20: Fail loud if pos_encoding is NULL - GRIM requires hybrid PBM
    if (!config_.pos_encoding) {
        throw std::runtime_error(
            "EncodingLayer::forward: config_.pos_encoding is NULL - "
            "GRIM requires PBM (ALiBi+RoPE) for positional encoding");
    }
    if (!config_.pos_encoding->alibi_slopes) {
        throw std::runtime_error(
            "EncodingLayer::forward: config_.pos_encoding->alibi_slopes is NULL - "
            "PBM ALiBi slopes not initialized");
    }
    
    // Issue #56: Store attention output in intermediates
    intermediates.attn_out_bhsd = autograd::scaled_dot_product_attention(
        intermediates.Q_bhsd, intermediates.K_bhsd, intermediates.V_bhsd, 
        config_.pos_encoding->alibi_slopes, 0.0f, stream, nullptr);
    if constexpr (kEnableEncoderStepLogs) fprintf(stderr, "[EncoderFwd] Step 4: Flash Attention DONE\n");
    
    //--------------------------------------------------
    // 5. Reshape attention output: BHSD -> [tokens, d_model]
    // Issue #56: Store in intermediates
    //--------------------------------------------------
    if constexpr (kEnableEncoderStepLogs) fprintf(stderr, "[EncoderFwd] Step 5: Reshape from BHSD...\n");
    intermediates.attn_out = Tensor::empty(TensorContract::TensorShape::make_BSM(total_tokens, d_model), true, stream);
    launchReshapeFromBHSD(intermediates.attn_out_bhsd.data, intermediates.attn_out.data, 
                          batch_size, seq_len, num_heads, head_dim, stream);
    if constexpr (kEnableEncoderStepLogs) fprintf(stderr, "[EncoderFwd] Step 5: Reshape DONE\n");
    
    //--------------------------------------------------
    // 6. Output projection: attn_out @ W_o^T + b_o
    // Issue #56: Store in intermediates
    //--------------------------------------------------
    if constexpr (kEnableEncoderStepLogs) fprintf(stderr, "[EncoderFwd] Step 6: Output projection...\n");
    // W_o is [d_model, d_model], so W_o^T is also [d_model, d_model]
    // Use transpose_b=true to compute attn_out @ W_o^T
    intermediates.proj_out = autograd::matmul(intermediates.attn_out, W_o_, stream, nullptr, nullptr, true);
    launchFFNBiasAdd(intermediates.proj_out.data, b_o_.data, total_tokens, d_model, stream);
    if constexpr (kEnableEncoderStepLogs) fprintf(stderr, "[EncoderFwd] Step 6: Output projection DONE\n");
    
    //--------------------------------------------------
    // 7. Residual1: input + proj_out -> residual1
    // Issue #56: Store in intermediates
    //--------------------------------------------------
    if constexpr (kEnableEncoderStepLogs) fprintf(stderr, "[EncoderFwd] Step 7: Residual1...\n");
    intermediates.residual1 = autograd::add(input, intermediates.proj_out, stream);
    if constexpr (kEnableEncoderStepLogs) fprintf(stderr, "[EncoderFwd] Step 7: Residual1 DONE\n");
    
    //--------------------------------------------------
    // 8. RMSNorm2: residual1 -> ln2_out
    // Issue #56: Store in intermediates
    //--------------------------------------------------
    if constexpr (kEnableEncoderStepLogs) fprintf(stderr, "[EncoderFwd] Step 8: RMSNorm2...\n");
    intermediates.ln2_out = autograd::rms_norm(intermediates.residual1, rms2_gamma_, config_.rms_epsilon, stream);
    if constexpr (kEnableEncoderStepLogs) fprintf(stderr, "[EncoderFwd] Step 8: RMSNorm2 DONE\n");
    
    //--------------------------------------------------
    // 9. FFN: ln2_out -> ffn_out (already using autograd)
    // Issue #56: FFN also stores its intermediates in this same ForwardIntermediates
    // (ffn_linear1_out, ffn_gelu_out are written by FFN forward)
    //--------------------------------------------------
    if constexpr (kEnableEncoderStepLogs) fprintf(stderr, "[EncoderFwd] Step 9: FFN...\n");
    intermediates.ffn_out = ffn_->forward(intermediates.ln2_out, intermediates);
    if constexpr (kEnableEncoderStepLogs) fprintf(stderr, "[EncoderFwd] Step 9: FFN DONE\n");
    
    //--------------------------------------------------
    // 10. Residual2: residual1 + ffn_out -> output
    // Issue #56: The final output IS stored in intermediates too
    // for consistency, but we also return it
    //--------------------------------------------------
    if constexpr (kEnableEncoderStepLogs) fprintf(stderr, "[EncoderFwd] Step 10: Residual2...\n");
    intermediates.output = autograd::add(intermediates.residual1, intermediates.ffn_out, stream);
    if constexpr (kEnableEncoderStepLogs) fprintf(stderr, "[EncoderFwd] Step 10: Residual2 DONE - layer COMPLETE\n");
    
    // Return a non-owning view of the output
    // The actual Tensor lives in intermediates and stays alive until backward completes
    Tensor result = Tensor::from_ptr(
        intermediates.output.data,
        intermediates.output.shape,
        false,  // doesn't own data - intermediates.output owns it
        true    // requires_grad
    );
    result.is_leaf = false;
    result.grad_fn = intermediates.output.grad_fn;
    result.owns_grad_fn = false;  // Borrowed, intermediates.output owns it
    result.stream = stream;
    
    return result;
}

//======================================================//
// Issue #37 DIAGNOSTIC: Public wrappers for alignment tracking
//======================================================//
void setEncoderW277Reference(const float* lm_weights, int vocab_size, int d_model, cudaStream_t stream) {
    setW277Reference(lm_weights, vocab_size, d_model, stream);
}

void resetEncoderDiagCount() {
    resetLayerDiagCount();
}

} // namespace GRIM
