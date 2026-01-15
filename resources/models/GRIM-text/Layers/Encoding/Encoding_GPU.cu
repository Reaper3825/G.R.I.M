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
#include "../../Shared/TensorContract/TensorContract_GPU.hpp"
#include "../FlashAttention/Flash_Attention_Kernal.hpp"
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
//======================================================//
namespace {
    // Shared W[277] reference - set once per forward pass
    static const float* s_w277_ref = nullptr;
    static int s_w277_d_model = 0;
    static float s_w277_norm = 0.0f;
    static int s_layer_diag_count = 0;
    static constexpr int kMaxDiagLogs = 24;  // First 2 batches * 12 layers
    
    void setW277Reference(const float* lm_weights, int vocab_size, int d_model, cudaStream_t stream) {
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

// Declared in RMSNorm_Kernel_GPU.cu
extern void launchRMSNormForward(const float* input, const float* gamma,
                                  float* output, int tokens, int hidden_dim,
                                  float epsilon, cudaStream_t stream);

extern void launchRMSNormBackward(const float* grad_output, const float* input,
                                   const float* gamma, float* grad_input,
                                   float* grad_gamma, int tokens, int hidden_dim,
                                   float epsilon, cudaStream_t stream);

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
    , rms1_gamma_(other.rms1_gamma_)
    , rms2_gamma_(other.rms2_gamma_)
    , rms1_gamma_grad_(other.rms1_gamma_grad_)
    , rms2_gamma_grad_(other.rms2_gamma_grad_)
    , W_qkv_(other.W_qkv_)
    , b_qkv_(other.b_qkv_)
    , W_o_(other.W_o_)
    , b_o_(other.b_o_)
    , W_qkv_grad_(other.W_qkv_grad_)
    , b_qkv_grad_(other.b_qkv_grad_)
    , W_o_grad_(other.W_o_grad_)
    , b_o_grad_(other.b_o_grad_)
    , ffn_(std::move(other.ffn_))
    , workspace_(other.workspace_)
    , workspace_bytes_(other.workspace_bytes_)
{
    // Null out the moved-from object
    other.config_.cublas_handle = nullptr;
    other.rms1_gamma_ = nullptr;
    other.rms2_gamma_ = nullptr;
    other.rms1_gamma_grad_ = nullptr;
    other.rms2_gamma_grad_ = nullptr;
    other.W_qkv_ = nullptr;
    other.b_qkv_ = nullptr;
    other.W_o_ = nullptr;
    other.b_o_ = nullptr;
    other.W_qkv_grad_ = nullptr;
    other.b_qkv_grad_ = nullptr;
    other.W_o_grad_ = nullptr;
    other.b_o_grad_ = nullptr;
    other.weights_allocated_ = false;
}

EncodingLayer& EncodingLayer::operator=(EncodingLayer&& other) noexcept {
    if (this != &other) {
        freeWeights();
        // NOTE: config_.cublas_handle is NOT owned - do NOT destroy (Rule 22)
        
        config_ = other.config_;
        weights_allocated_ = other.weights_allocated_;
        config_.cublas_handle = other.config_.cublas_handle;
        rms1_gamma_ = other.rms1_gamma_;
        rms2_gamma_ = other.rms2_gamma_;
        rms1_gamma_grad_ = other.rms1_gamma_grad_;
        rms2_gamma_grad_ = other.rms2_gamma_grad_;
        W_qkv_ = other.W_qkv_;
        b_qkv_ = other.b_qkv_;
        W_o_ = other.W_o_;
        b_o_ = other.b_o_;
        W_qkv_grad_ = other.W_qkv_grad_;
        b_qkv_grad_ = other.b_qkv_grad_;
        W_o_grad_ = other.W_o_grad_;
        b_o_grad_ = other.b_o_grad_;
        ffn_ = std::move(other.ffn_);
        workspace_ = other.workspace_;
        workspace_bytes_ = other.workspace_bytes_;
        
        other.config_.cublas_handle = nullptr;
        other.rms1_gamma_ = nullptr;
        other.rms2_gamma_ = nullptr;
        other.rms1_gamma_grad_ = nullptr;
        other.rms2_gamma_grad_ = nullptr;
        other.W_qkv_ = nullptr;
        other.b_qkv_ = nullptr;
        other.W_o_ = nullptr;
        other.b_o_ = nullptr;
        other.W_qkv_grad_ = nullptr;
        other.b_qkv_grad_ = nullptr;
        other.W_o_grad_ = nullptr;
        other.b_o_grad_ = nullptr;
        other.weights_allocated_ = false;
    }
    return *this;
}

void EncodingLayer::setConfig(const EncodingConfig& cfg) {
    cfg.validate("EncodingLayer::setConfig");
    config_ = cfg;
}

void EncodingLayer::freeWeights() {
    if (rms1_gamma_) { cudaFree(rms1_gamma_); rms1_gamma_ = nullptr; }
    if (rms2_gamma_) { cudaFree(rms2_gamma_); rms2_gamma_ = nullptr; }
    if (rms1_gamma_grad_) { cudaFree(rms1_gamma_grad_); rms1_gamma_grad_ = nullptr; }
    if (rms2_gamma_grad_) { cudaFree(rms2_gamma_grad_); rms2_gamma_grad_ = nullptr; }
    if (W_qkv_) { cudaFree(W_qkv_); W_qkv_ = nullptr; }
    if (b_qkv_) { cudaFree(b_qkv_); b_qkv_ = nullptr; }
    if (W_o_) { cudaFree(W_o_); W_o_ = nullptr; }
    if (b_o_) { cudaFree(b_o_); b_o_ = nullptr; }
    if (W_qkv_grad_) { cudaFree(W_qkv_grad_); W_qkv_grad_ = nullptr; }
    if (b_qkv_grad_) { cudaFree(b_qkv_grad_); b_qkv_grad_ = nullptr; }
    if (W_o_grad_) { cudaFree(W_o_grad_); W_o_grad_ = nullptr; }
    if (b_o_grad_) { cudaFree(b_o_grad_); b_o_grad_ = nullptr; }
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
    
    // RMSNorm gamma (NO BETA - this is RMSNorm!)
    CUDA_CHECK(cudaMalloc(&rms1_gamma_, d_model * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&rms2_gamma_, d_model * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&rms1_gamma_grad_, d_model * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&rms2_gamma_grad_, d_model * sizeof(float)));
    
    // Initialize gamma to 1.0, gradients to 0.0
    int threads = 256;
    int blocks = (d_model + threads - 1) / threads;
    fillOnesKernel<<<blocks, threads, 0, config_.stream>>>(rms1_gamma_, d_model);
    fillOnesKernel<<<blocks, threads, 0, config_.stream>>>(rms2_gamma_, d_model);
    CUDA_CHECK(cudaMemsetAsync(rms1_gamma_grad_, 0, d_model * sizeof(float), config_.stream));
    CUDA_CHECK(cudaMemsetAsync(rms2_gamma_grad_, 0, d_model * sizeof(float), config_.stream));
    
    // Attention weights
    // W_qkv: [d_model + 2*kv_dim, d_model] for GQA-aware projection
    // For MHA (kv_dim = d_model): [3*d_model, d_model]
    // For GQA (kv_dim < d_model): smaller K,V projections
    const size_t qkv_weight_size = static_cast<size_t>(d_model + 2 * kv_dim) * d_model;
    const size_t qkv_bias_size = d_model + 2 * kv_dim;
    const size_t o_weight_size = static_cast<size_t>(d_model) * d_model;
    
    CUDA_CHECK(cudaMalloc(&W_qkv_, qkv_weight_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&b_qkv_, qkv_bias_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&W_o_, o_weight_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&b_o_, d_model * sizeof(float)));
    
    CUDA_CHECK(cudaMalloc(&W_qkv_grad_, qkv_weight_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&b_qkv_grad_, qkv_bias_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&W_o_grad_, o_weight_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&b_o_grad_, d_model * sizeof(float)));
    
    // Initialize attention weights to small random values (caller should re-initialize)
    // For now, zero them so any use without init is obvious
    CUDA_CHECK(cudaMemsetAsync(W_qkv_, 0, qkv_weight_size * sizeof(float), config_.stream));
    CUDA_CHECK(cudaMemsetAsync(b_qkv_, 0, qkv_bias_size * sizeof(float), config_.stream));
    CUDA_CHECK(cudaMemsetAsync(W_o_, 0, o_weight_size * sizeof(float), config_.stream));
    CUDA_CHECK(cudaMemsetAsync(b_o_, 0, d_model * sizeof(float), config_.stream));
    
    // Zero gradients
    CUDA_CHECK(cudaMemsetAsync(W_qkv_grad_, 0, qkv_weight_size * sizeof(float), config_.stream));
    CUDA_CHECK(cudaMemsetAsync(b_qkv_grad_, 0, qkv_bias_size * sizeof(float), config_.stream));
    CUDA_CHECK(cudaMemsetAsync(W_o_grad_, 0, o_weight_size * sizeof(float), config_.stream));
    CUDA_CHECK(cudaMemsetAsync(b_o_grad_, 0, d_model * sizeof(float), config_.stream));
    
    // FFN layer
    FeedForwardConfig ffn_cfg;
    ffn_cfg.d_model = d_model;
    ffn_cfg.d_ff = d_ff;
    ffn_cfg.stream = config_.stream;
    ffn_cfg.cublas_handle = config_.cublas_handle;  // CRITICAL: Must pass handle to FFN (Rule 22)
    ffn_ = std::make_unique<FeedForwardLayer>(ffn_cfg);
    ffn_->ensureWeightStorage();
    
    weights_allocated_ = true;
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
//  Forward Pass
//======================================================//

void EncodingLayer::forward(const EncodingForwardArgs& args) {
    args.validate("EncodingLayer::forward");
    validateReady("EncodingLayer::forward");
    
    const int total_tokens = args.total_tokens;
    const int seq_len = args.seq_len;
    const int batch_size = args.batchSize();
    const int d_model = config_.d_model;
    const int kv_dim = config_.kvDim();
    const int num_heads = config_.num_heads;
    const int num_kv_heads = config_.effectiveKVHeads();
    const int head_dim = config_.headDim();
    
    if (!args.stream) {
        throw std::runtime_error("EncodingLayer::forward: stream is NULL");
    }
    cudaStream_t stream = args.stream;
    
    // Validate workspace
    const std::size_t required = requiredWorkspaceBytes(total_tokens, seq_len);
    if (!workspace_ || workspace_bytes_ < required) {
        throw std::runtime_error("EncodingLayer::forward: insufficient workspace. "
                                 "Required: " + std::to_string(required) + 
                                 " bytes, provided: " + std::to_string(workspace_bytes_));
    }
    
    // Carve up workspace
    float* ptr = workspace_;
    auto bump = [&ptr](std::size_t count) -> float* {
        float* out = ptr;
        ptr += count;
        return out;
    };
    
    float* ln1_out = bump(total_tokens * d_model);
    float* ln2_out = bump(total_tokens * d_model);
    float* Q = bump(total_tokens * d_model);
    float* K = bump(total_tokens * kv_dim);
    float* V = bump(total_tokens * kv_dim);
    float* Q_bhsd = bump(batch_size * num_heads * seq_len * head_dim);
    float* K_bhsd = bump(batch_size * num_kv_heads * seq_len * head_dim);
    float* V_bhsd = bump(batch_size * num_kv_heads * seq_len * head_dim);
    float* attn_out_bhsd = bump(batch_size * num_heads * seq_len * head_dim);
    float* attn_out = bump(total_tokens * d_model);
    float* residual1 = bump(total_tokens * d_model);
    float* pre_gelu = bump(total_tokens * config_.d_ff);
    float* post_gelu = bump(total_tokens * config_.d_ff);
    float* ffn_out = bump(total_tokens * d_model);
    
    //--------------------------------------------------
    // 1. RMSNorm1: input -> ln1_out
    //--------------------------------------------------
    launchRMSNormForward(args.input, rms1_gamma_, ln1_out,
                         total_tokens, d_model, config_.rms_epsilon, stream);
    
    // REMOVED cache_ln1_input copy - backward uses layer input from previous layer's cache (redundant!)
    if (args.cache_ln1_out) {
        CUDA_CHECK(cudaMemcpyAsync(args.cache_ln1_out, ln1_out,
                                    total_tokens * d_model * sizeof(float),
                                    cudaMemcpyDeviceToDevice, stream));
    }
    
    //--------------------------------------------------
    // 2. QKV Projection: ln1_out -> Q, K, V
    //--------------------------------------------------
    QKVProjectionConfig proj_cfg;
    proj_cfg.d_model = d_model;
    proj_cfg.num_heads = num_heads;
    proj_cfg.num_kv_heads = num_kv_heads;
    proj_cfg.head_dim = head_dim;  // Use pre-computed value from config_.headDim()
    proj_cfg.batch_size = batch_size;
    proj_cfg.seq_len = seq_len;
    proj_cfg.stream = stream;
    proj_cfg.handle = config_.cublas_handle;
    
    QKVProjectionWeights proj_weights;
    proj_weights.W_qkv = W_qkv_;
    proj_weights.b_qkv = b_qkv_;
    
    if (config_.isGQA()) {
        // GQA: separate projections
        const size_t q_weight_size = static_cast<size_t>(d_model) * d_model;
        const size_t k_weight_size = static_cast<size_t>(kv_dim) * d_model;
        
        proj_weights.W_q = W_qkv_;
        proj_weights.W_k = W_qkv_ + q_weight_size;
        proj_weights.W_v = W_qkv_ + q_weight_size + k_weight_size;
        proj_weights.b_q = b_qkv_;
        proj_weights.b_k = b_qkv_ + d_model;
        proj_weights.b_v = b_qkv_ + d_model + kv_dim;
        
        launchGQAProjection(ln1_out, proj_weights, Q, K, V, proj_cfg);
    } else {
        // Rule 20: MHA is deprecated/legacy - always use GQA path
        throw std::runtime_error("EncodingLayer::forward: MHA fallback path DELETED! "
                                 "GQA projection is required (num_kv_heads <= num_heads). "
                                 "Got num_heads=" + std::to_string(num_heads) + 
                                 ", num_kv_heads=" + std::to_string(num_kv_heads) + 
                                 ". Verify config.isGQA() returns true.");
    }
    
    //--------------------------------------------------
    // 3. Reshape QKV to BHSD for Flash Attention
    //--------------------------------------------------
    launchQKVReshapeToBHSD(Q, K, V, Q_bhsd, K_bhsd, V_bhsd,
                           batch_size, seq_len, num_heads, num_kv_heads, head_dim, stream);
    
    //--------------------------------------------------
    // 3b. Apply RoPE rotation to Q and K (CRITICAL for selective attention!)
    //     RoPE rotates Q/K vectors based on position, enabling position-aware dot products
    //     PBM is always hybrid (ALiBi + RoPE) - validate all PBM state
    //--------------------------------------------------
    if (config_.pos_encoding && config_.pos_encoding->valid && 
        config_.pos_encoding->rope_inv_freq != nullptr && config_.pos_encoding->rotary_dim > 0) {
        
        // Use GQA-aware RoPE rotation that handles Q and K with different head counts
        PBM::launchRoPERotationGQA(
            Q_bhsd, K_bhsd,  // In-place rotation
            config_.pos_encoding->rope_inv_freq,
            batch_size,
            num_heads,       // Q head count
            num_kv_heads,    // K head count (fewer in GQA)
            seq_len,
            head_dim,
            config_.pos_encoding->rotary_dim,
            stream
        );
    } else {
        throw std::runtime_error("EncodingLayer::forward: RoPE not initialized - model requires positional encoding");
    }
    
    // Cache Q, K, V in BHSD format for backward
    // DIAGNOSTIC Issue #36: Verify cache write actually works
    static int cache_write_count = 0;
    const std::size_t q_size = static_cast<std::size_t>(batch_size) * num_heads * seq_len * head_dim;
    
    if (args.cache_q) {
        // SYNC before write to ensure Q_bhsd is computed
        CUDA_CHECK(cudaStreamSynchronize(stream));
        
        // Check Q_bhsd RMS BEFORE copy
        float pre_rms = 0.0f;
        {
            std::vector<float> h_q(q_size);
            CUDA_CHECK(cudaMemcpy(h_q.data(), Q_bhsd, q_size * sizeof(float), cudaMemcpyDeviceToHost));
            double sum_sq = 0.0;
            for (std::size_t i = 0; i < q_size; ++i) sum_sq += h_q[i] * h_q[i];
            pre_rms = static_cast<float>(std::sqrt(sum_sq / q_size));
        }
        
        // Do the actual copy
        CUDA_CHECK(cudaMemcpyAsync(args.cache_q, Q_bhsd, q_size * sizeof(float), cudaMemcpyDeviceToDevice, stream));
        
        // SYNC after write
        CUDA_CHECK(cudaStreamSynchronize(stream));
        
        // Check cache RMS AFTER copy
        float post_rms = 0.0f;
        {
            std::vector<float> h_cache(q_size);
            CUDA_CHECK(cudaMemcpy(h_cache.data(), args.cache_q, q_size * sizeof(float), cudaMemcpyDeviceToHost));
            double sum_sq = 0.0;
            for (std::size_t i = 0; i < q_size; ++i) sum_sq += h_cache[i] * h_cache[i];
            post_rms = static_cast<float>(std::sqrt(sum_sq / q_size));
        }
        
        cache_write_count++;
        if (cache_write_count <= 24) {  // First 2 batches * 12 layers
            fprintf(stderr, "[CACHE_WRITE_VERIFY] count=%d Q_bhsd_rms=%.6f cache_q_rms=%.6f match=%s\n",
                    cache_write_count, pre_rms, post_rms, 
                    (std::abs(pre_rms - post_rms) < 1e-5f) ? "YES" : "NO");
        }
    } else {
        // DIAGNOSTIC: Issue #36 - why is cache_q null?
        fprintf(stderr, "[CACHE_WRITE] WARNING: args.cache_q is NULL - cache not written!\n");
    }
    if (args.cache_k) {
        CUDA_CHECK(cudaMemcpyAsync(args.cache_k, K_bhsd,
                                    batch_size * num_kv_heads * seq_len * head_dim * sizeof(float),
                                    cudaMemcpyDeviceToDevice, stream));
    }
    if (args.cache_v) {
        CUDA_CHECK(cudaMemcpyAsync(args.cache_v, V_bhsd,
                                    batch_size * num_kv_heads * seq_len * head_dim * sizeof(float),
                                    cudaMemcpyDeviceToDevice, stream));
    }
    
    //--------------------------------------------------
    // 4. Flash Attention v2: Q/K/V -> attn_out (BF16 path)
    //--------------------------------------------------
    if (!config_.use_flash_attention) {
        throw std::runtime_error("EncodingLayer::forward: Flash Attention disabled - no fallback path available");
    }
    if (config_.qk_norm_enabled || config_.softmax_temperature != 1.0f) {
        throw std::runtime_error("EncodingLayer::forward: QK-norm/softmax temperature unsupported in FA v2 path");
    }
    if (head_dim != 32 && head_dim != 64) {
        throw std::runtime_error("EncodingLayer::forward: FlashAttention v2 requires head_dim=32 or 64");
    }
    if (num_heads <= 0 || num_kv_heads <= 0 || (num_heads % num_kv_heads) != 0) {
        throw std::runtime_error("EncodingLayer::forward: invalid head configuration for FlashAttention v2");
    }
    if (!args.cache_softmax_lse) {
        throw std::runtime_error("EncodingLayer::forward: softmax_lse buffer is NULL (required for FA v2)");
    }
    if (!args.fa_q_bf16 || !args.fa_k_bf16 || !args.fa_v_bf16 || !args.fa_out_bf16) {
        throw std::runtime_error("EncodingLayer::forward: BF16 scratch buffers are NULL (required for FA v2)");
    }

    const size_t q_elems = static_cast<size_t>(batch_size) *
                           static_cast<size_t>(num_heads) *
                           static_cast<size_t>(seq_len) *
                           static_cast<size_t>(head_dim);
    const size_t kv_elems = static_cast<size_t>(batch_size) *
                            static_cast<size_t>(num_kv_heads) *
                            static_cast<size_t>(seq_len) *
                            static_cast<size_t>(head_dim);
    if (args.fa_q_bf16_elems < q_elems || args.fa_kv_bf16_elems < kv_elems) {
        throw std::runtime_error("EncodingLayer::forward: BF16 scratch buffers too small for current batch");
    }

    if (!config_.pos_encoding || !config_.pos_encoding->valid || !config_.pos_encoding->alibi_slopes) {
        throw std::runtime_error("EncodingLayer::forward: ALiBi slopes are NULL or PBM invalid - PBM hybrid requires ALiBi + RoPE");
    }
    if (config_.pos_encoding->num_heads != num_heads) {
        throw std::runtime_error("EncodingLayer::forward: ALiBi num_heads mismatch: pbm_spec=" + 
                                 std::to_string(config_.pos_encoding->num_heads) + 
                                 " encoder=" + std::to_string(num_heads));
    }
    if (config_.pos_encoding->num_kv_heads != num_kv_heads) {
        throw std::runtime_error("EncodingLayer::forward: ALiBi num_kv_heads mismatch: pbm_spec=" + 
                                 std::to_string(config_.pos_encoding->num_kv_heads) + 
                                 " encoder=" + std::to_string(num_kv_heads));
    }

    TensorConversion::convert_BHSD_to_BSHD_bf16(Q_bhsd, args.fa_q_bf16,
                                                batch_size, num_heads, seq_len, head_dim, stream);
    TensorConversion::convert_BHSD_to_BSHD_bf16(K_bhsd, args.fa_k_bf16,
                                                batch_size, num_kv_heads, seq_len, head_dim, stream);
    TensorConversion::convert_BHSD_to_BSHD_bf16(V_bhsd, args.fa_v_bf16,
                                                batch_size, num_kv_heads, seq_len, head_dim, stream);

    flash_attn_fwd_ex(
        args.fa_q_bf16,
        args.fa_k_bf16,
        args.fa_v_bf16,
        args.fa_out_bf16,
        args.cache_softmax_lse,
        config_.pos_encoding->alibi_slopes,
        batch_size,
        seq_len,
        num_heads,
        num_kv_heads,
        head_dim,
        config_.causal_mask,
        true,
        stream);

    TensorConversion::convert_BSHD_bf16_to_BHSD(args.fa_out_bf16, attn_out_bhsd,
                                                batch_size, seq_len, num_heads, head_dim, stream);
    
    // Cache attention output BHSD for backward (cache_attn_bhsd and cache_attn_out are aliases)
    if (args.cache_attn_bhsd) {
        CUDA_CHECK(cudaMemcpyAsync(args.cache_attn_bhsd, attn_out_bhsd,
                                    batch_size * num_heads * seq_len * head_dim * sizeof(float),
                                    cudaMemcpyDeviceToDevice, stream));
    }
    
    //--------------------------------------------------
    // 5. Reshape attention output: BHSD -> [tokens, d_model]
    //--------------------------------------------------
    launchReshapeFromBHSD(attn_out_bhsd, attn_out, batch_size, seq_len, num_heads, head_dim, stream);
    
    //--------------------------------------------------
    // 6. Output projection: attn_out @ W_o^T + b_o
    //    Use ln2_out as temp to avoid GEMM input/output aliasing (UB in cuBLAS)
    //--------------------------------------------------
    const float alpha = 1.0f;
    const float beta = 0.0f;
    
    // CRITICAL: Always rebind stream before cuBLAS ops - NumericHead may have changed it
    cublasSetStream(config_.cublas_handle, stream);
    
    // attn_out [tokens, d_model] @ W_o^T [d_model, d_model] -> ln2_out (temp)
    CUBLAS_CHECK(cublasSgemm(config_.cublas_handle,
                             CUBLAS_OP_T, CUBLAS_OP_N,
                             d_model, total_tokens, d_model,
                             &alpha,
                             W_o_, d_model,
                             attn_out, d_model,
                             &beta,
                             ln2_out, d_model));  // Output to temp buffer
    
    launchAddBias(ln2_out, b_o_, total_tokens, d_model, stream);

    // Cache attention output after W_o projection for backward (BSM layout).
    if (args.cache_attn_output) {
        CUDA_CHECK(cudaMemcpyAsync(args.cache_attn_output, ln2_out,
                                    static_cast<std::size_t>(total_tokens) * d_model * sizeof(float),
                                    cudaMemcpyDeviceToDevice, stream));
    }
    
    //--------------------------------------------------
    // 7. Residual1: input + proj_out -> residual1
    //--------------------------------------------------
    launchResidualAdd(args.input, ln2_out, residual1, total_tokens * d_model, stream);
    
    // === Issue #37 DIAGNOSTIC: Track alignment AFTER attention ===
    static int s_attn_layer = 0;
    logHiddenStateAlignment("after_attn", s_attn_layer, residual1, total_tokens, d_model, stream);
    
    if (args.cache_residual1) {
        CUDA_CHECK(cudaMemcpyAsync(args.cache_residual1, residual1,
                                    total_tokens * d_model * sizeof(float),
                                    cudaMemcpyDeviceToDevice, stream));
    }
    
    //--------------------------------------------------
    // 8. RMSNorm2: residual1 -> ln2_out
    //--------------------------------------------------
    launchRMSNormForward(residual1, rms2_gamma_, ln2_out,
                         total_tokens, d_model, config_.rms_epsilon, stream);
    
    if (args.cache_ln2_out) {
        CUDA_CHECK(cudaMemcpyAsync(args.cache_ln2_out, ln2_out,
                                    total_tokens * d_model * sizeof(float),
                                    cudaMemcpyDeviceToDevice, stream));
    }
    
    //--------------------------------------------------
    // 9. FFN: ln2_out -> ffn_out
    //--------------------------------------------------
    FeedForwardForwardArgs ffn_args;
    ffn_args.input = ln2_out;
    ffn_args.output = ffn_out;
    ffn_args.total_tokens = total_tokens;
    ffn_args.stream = stream;
    // Rule 20: No fallback - if caller wants pre_gelu cached, they MUST provide buffer
    // If they don't provide buffer, we pass nullptr (FFN will decide if that's valid)
    ffn_args.cache_pre_gelu = args.cache_ffn_pre_gelu;
    ffn_args.cache_post_gelu = post_gelu;
    
    ffn_->forward(ffn_args, nullptr);
    
    // Cache post-GELU output for backward pass (required for grad_W2 computation)
    if (args.cache_ffn_output) {
        CUDA_CHECK(cudaMemcpyAsync(args.cache_ffn_output, post_gelu,
                                    static_cast<std::size_t>(total_tokens) * config_.d_ff * sizeof(float),
                                    cudaMemcpyDeviceToDevice, stream));
    }
    
    //--------------------------------------------------
    // 10. Residual2: residual1 + ffn_out -> output
    //--------------------------------------------------
    launchResidualAdd(residual1, ffn_out, args.output, total_tokens * d_model, stream);
    
    // === Issue #37 DIAGNOSTIC: Track alignment AFTER FFN (layer output) ===
    logHiddenStateAlignment("after_ffn", s_attn_layer, args.output, total_tokens, d_model, stream);
    s_attn_layer = (s_attn_layer + 1) % 12;  // Cycle through layers
    incrementLayerDiagCount();
    
    // Cache layer output for backward pass (used as layer_input in next layer's backward)
    if (args.cache_layer_output) {
        CUDA_CHECK(cudaMemcpyAsync(args.cache_layer_output, args.output,
                                    total_tokens * d_model * sizeof(float),
                                    cudaMemcpyDeviceToDevice, stream));
    }
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
