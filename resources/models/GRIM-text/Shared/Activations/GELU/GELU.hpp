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
 * - Input/output tensors use TensorContract::TensorView with BSM layout
 * - Element-wise operation: output[i] = GELU(input[i])
 * - In-place operation supported for backward (grad_output == grad_input)
 *
 * USAGE:
 * Forward: launchGeluForward() for inference
 * Backward: launchGeluBackward() for training (requires original input)
 *
 * REFACTORED: Uses TensorContract::TensorView instead of raw float*
 * All tensor views validated via require() (Rule 20: Fail Loud)
 */

#pragma once

#include <cstddef>
#include <cuda_runtime.h>

#include "../../../Layers/grim_layer_gpu.hpp"
#include "../../TensorContract/TensorContract_GPU.hpp"

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

//======================================================//
//  GELUForwardArgs - Type-safe tensor views
//======================================================//
/**
 * @brief Arguments for GELU forward pass using TensorView
 * 
 * RULE 20: All views validated via require() - fail loud on NULL/invalid
 */
struct GELUForwardArgs {
    // Input tensors (const view)
    TensorContract::TensorView input;    // [elements] BSM layout (rows=1, cols=elements or rows=tokens, cols=d)
    
    // Output tensors (mutable view) - must differ from input
    TensorContract::TensorView output;   // [elements] same shape as input
    
    // Execution context
    cudaStream_t stream = nullptr;
    
    // RULE 20: Fail loud validation
    void validate(const char* context) const {
        input.require(context);
        output.require(context);
        
        if (input.size_elements() != output.size_elements()) {
            throw std::runtime_error(std::string(context) + 
                ": input/output element count mismatch (input=" + 
                std::to_string(input.size_elements()) + ", output=" + 
                std::to_string(output.size_elements()) + ")");
        }
        if (input.ptr == output.ptr) {
            throw std::runtime_error(std::string(context) + 
                ": GELU forward does not support in-place operation (input == output)");
        }
    }
    
    // Convenience: extract element count from tensor
    std::size_t elements() const { return input.size_elements(); }
};

//======================================================//
//  GELUBackwardArgs - Type-safe tensor views
//======================================================//
/**
 * @brief Arguments for GELU backward pass using TensorView
 * 
 * RULE 20: All views validated via require() - fail loud on NULL/invalid
 */
struct GELUBackwardArgs {
    // Input tensors (const views)
    TensorContract::TensorView input;        // Original forward input (required for derivative)
    TensorContract::TensorView grad_output;  // Gradient from next layer
    
    // Output tensors (mutable view)
    TensorContract::TensorView grad_input;   // Computed gradient (may be same as grad_output for in-place)
    
    // Execution context
    cudaStream_t stream = nullptr;
    
    // RULE 20: Fail loud validation
    void validate(const char* context) const {
        input.require(context);
        grad_output.require(context);
        grad_input.require(context);
        
        if (input.size_elements() != grad_output.size_elements()) {
            throw std::runtime_error(std::string(context) + 
                ": input/grad_output element count mismatch");
        }
        if (grad_output.size_elements() != grad_input.size_elements()) {
            throw std::runtime_error(std::string(context) + 
                ": grad_output/grad_input element count mismatch");
        }
    }
    
    // Convenience: extract element count from tensor
    std::size_t elements() const { return input.size_elements(); }
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

private:
    GELUConfig config_{};
    const float* last_input_ = nullptr;

    GELUConfig makeConfig(const GELUForwardArgs& args) const;
    GELUConfig makeConfig(const GELUBackwardArgs& args) const;
};

void launchGeluForward(const GELUForwardArgs& args);
void launchGeluBackward(const GELUBackwardArgs& args);

} // namespace GRIM

