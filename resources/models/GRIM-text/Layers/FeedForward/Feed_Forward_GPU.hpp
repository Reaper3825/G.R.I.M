//======================================================//
//  Feed_Forward_GPU.hpp
//  GPU-accelerated FeedForward layer with CRTP design
//  
//  Two-layer MLP: Linear -> GELU -> Linear
//  Uses cuBLAS for matrix operations
//  
//  Uses shared components:
//    - Shared/Activations/GELU for activation
//    - Shared/GPUBuffer for memory management
//======================================================//

#pragma once

#ifdef USE_CUDA

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cstddef>
#include <memory>

#include "../grim_layer_gpu.hpp"
#include "../../Shared/GPUBuffer/GPUBuffer.hpp"
#include "../../Shared/Activations/GELU/GELU.hpp"

namespace GRIM {

//======================================================//
//  Configuration
//======================================================//

struct FeedForwardConfig {
    int d_model = 768;           // Input/output dimension
    int d_ff = 3072;             // Hidden (intermediate) dimension
    float dropout_rate = 0.0f;   // Dropout probability (currently unused) Note : Should be used already need to verify
    cudaStream_t stream = nullptr;
    cublasHandle_t cublas_handle = nullptr;  // Rule 22: MUST be training_state.cublas_handle
};

//======================================================//
//  Forward/Backward Arguments
//======================================================//

struct FeedForwardForwardArgs {
    const float* input = nullptr;         // [total_tokens, d_model]
    float* output = nullptr;              // [total_tokens, d_model]
    int total_tokens = 0;
    cudaStream_t stream = nullptr;
    
    // Optional caches for training backward pass
    float* cache_pre_gelu = nullptr;      // [total_tokens, d_ff] - pre-GELU activations
    float* cache_post_gelu = nullptr;     // [total_tokens, d_ff] - post-GELU activations
};


//======================================================//
//  FeedForwardLayer - CRTP Implementation
//======================================================//

class FeedForwardLayer final : public Layer<FeedForwardLayer, float> {
public:
    static constexpr LayerType layer_type = LayerType::kFeedForward;

    // Rule 20: Default constructor deleted - config with valid cublas_handle REQUIRED
    FeedForwardLayer() = delete;
    explicit FeedForwardLayer(const FeedForwardConfig& config);
    FeedForwardLayer(const Dimensions& dims, const FeedForwardConfig& config);
    ~FeedForwardLayer();

    // Prevent copy (cuBLAS handle)
    FeedForwardLayer(const FeedForwardLayer&) = delete;
    FeedForwardLayer& operator=(const FeedForwardLayer&) = delete;

    // Allow move
    FeedForwardLayer(FeedForwardLayer&& other) noexcept;
    FeedForwardLayer& operator=(FeedForwardLayer&& other) noexcept;

    //--------------------------------------------------
    // Configuration
    //--------------------------------------------------
    void setConfig(const FeedForwardConfig& cfg);
    const FeedForwardConfig& config() const noexcept { return config_; }

    //--------------------------------------------------
    // Weight Management
    //--------------------------------------------------
    void ensureWeightStorage();
    
    // Direct GPU weight accessors (for training integration)
    float* getW1() { return W1_.ptr(); }
    float* getB1() { return b1_.ptr(); }
    float* getW2() { return W2_.ptr(); }
    float* getB2() { return b2_.ptr(); }
    const float* getW1() const { return W1_.ptr(); }
    const float* getB1() const { return b1_.ptr(); }
    const float* getW2() const { return W2_.ptr(); }
    const float* getB2() const { return b2_.ptr(); }

    //--------------------------------------------------
    // Forward Pass
    //--------------------------------------------------
    // Primary GPU-native forward
    void forward(const FeedForwardForwardArgs& args,
                 LayerWorkspace<float>* workspace = nullptr);
    
    // Convenience: forwardGPU matching GPUFeedForwardNetwork interface
    void forwardGPU(const float* d_input, float* d_output, int total_tokens,
                    float* d_pre_gelu_cache, float* d_post_gelu_workspace);

    //--------------------------------------------------
    // NOTE: Backward Pass DELETED - Rule 20
    // Training uses BackwardPhase2_Encoder.cu::computeFFNBackward()
    //--------------------------------------------------

    //--------------------------------------------------
    // NOTE: CRTP Interface DELETED - Rule 20
    // Use forward(FeedForwardForwardArgs&) directly.
    // Training backward uses BackwardPhase2_Encoder.cu.
    //--------------------------------------------------

    //--------------------------------------------------
    // Workspace
    //--------------------------------------------------
    std::size_t requiredWorkspaceBytes(int total_tokens) const;

private:
    void initCublas();
    void destroyCublas();
    // DELETED: resolveStream() - Rule 20 (no stream fallback)

    FeedForwardConfig config_{};
    // Rule 22: Use config_.cublas_handle, do NOT store local copy

    // Weight buffers [GPU memory]
    GPUBuffer<float> W1_;    // [d_ff, d_model]
    GPUBuffer<float> b1_;    // [d_ff]
    GPUBuffer<float> W2_;    // [d_model, d_ff]
    GPUBuffer<float> b2_;    // [d_model]

    // Internal workspace buffers (allocated on demand)
    GPUBuffer<float> pre_gelu_buf_;     // [tokens, d_ff]
    GPUBuffer<float> post_gelu_buf_;    // [tokens, d_ff]
    // DELETED: grad_hidden_buf_ - only needed for backward (now in BackwardPhase2)
};

//======================================================//
//  FFN-specific kernel declarations
//  (Bias add/backward - not covered by GELU module)
//======================================================//

// Bias addition: tensor[i] += bias[i % features]
void launchFFNBiasAdd(float* tensor, const float* bias,
                      int total_tokens, int features,
                      cudaStream_t stream);

// Bias gradient: grad_bias = sum(grad_output, axis=0)
void launchFFNBiasBackward(const float* grad_output, float* grad_bias,
                           int total_tokens, int features,
                           cudaStream_t stream);

} // namespace GRIM

#endif // USE_CUDA
