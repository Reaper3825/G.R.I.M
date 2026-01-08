/**
 * @file GELU.hpp
 * @brief Gaussian Error Linear Unit (GELU) activation function for GPU
 *
 * GELU ACTIVATION:
 * GELU(x) = 0.5 * x * (1 + tanh[√(2/π) * (x + 0.044715 * x³)])
 *
 * This is the tanh approximation of GELU, which is faster than the
 * exact erf-based formula while maintaining high accuracy.
 *
 * MEMORY LAYOUT:
 * - Input/output tensors are treated as flat 1D arrays
 * - Element-wise operation: output[i] = GELU(input[i])
 * - No in-place operation support (input != output required)
 *
 * USAGE:
 * Forward: launchGeluForward() for inference
 * Backward: launchGeluBackward() for training (requires original input)
 */

#pragma once

#include <cstddef>
#include <cuda_runtime.h>

#include "../../../Layers/grim_layer_gpu.hpp"

namespace GRIM {

/**
 * @brief Configuration for GELU layer
 * @param elements Total number of elements to process
 * @param stream CUDA stream for async execution (nullptr = default stream)
 */
struct GELUConfig {
    std::size_t elements = 0;
    cudaStream_t stream = nullptr;
};

/**
 * @brief Arguments for GELU forward pass
 * @param input Input tensor [elements] (device memory)
 * @param output Output tensor [elements] (device memory, must differ from input)
 * @param elements Number of elements to process
 * @param stream CUDA stream for async execution
 */
struct GELUForwardArgs {
    const float* input = nullptr;
    float* output = nullptr;
    std::size_t elements = 0;
    cudaStream_t stream = nullptr;
};

/**
 * @brief Arguments for GELU backward pass
 * @param input Original forward pass input [elements] (device memory, required for derivative)
 * @param grad_output Gradient from next layer [elements] (device memory)
 * @param grad_input Computed gradient for previous layer [elements] (device memory)
 * @param elements Number of elements to process
 * @param stream CUDA stream for async execution
 */
struct GELUBackwardArgs {
    const float* input = nullptr;
    const float* grad_output = nullptr;
    float* grad_input = nullptr;
    std::size_t elements = 0;
    cudaStream_t stream = nullptr;
};

class GELULayer final : public Layer<GELULayer, float> {
public:
    static constexpr LayerType layer_type = LayerType::kFeedForward;

    GELULayer();
    explicit GELULayer(const GELUConfig& config);
    GELULayer(const Dimensions& dims, const GELUConfig& config);

    void setConfig(const GELUConfig& cfg) { config_ = cfg; }
    const GELUConfig& config() const noexcept { return config_; }

    void forward(const GELUForwardArgs& args,
                 LayerWorkspace<float>* workspace = nullptr);
    void backward(const GELUBackwardArgs& args,
                  LayerWorkspace<float>* workspace = nullptr);

    // Layer interface hooks
    void onConfigure(const Dimensions& dims);
    void forwardImpl(const LayerIO<float>& io, LayerWorkspace<float>* workspace);
    void backwardImpl(const LayerIO<float>& io, LayerWorkspace<float>* workspace);
    void applyGradientsImpl(value_type) {}

private:
    GELUConfig config_{};
    const float* last_input_ = nullptr;

    GELUConfig makeConfig(const GELUForwardArgs& args) const;
    GELUConfig makeConfig(const GELUBackwardArgs& args) const;
};

void launchGeluForward(const GELUForwardArgs& args);
void launchGeluBackward(const GELUBackwardArgs& args);

} // namespace GRIM

