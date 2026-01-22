/**
 * @file GELU.cu
 * @brief GPU implementation of GELU activation with tanh approximation
 *
 * ACTIVATION FORMULA:
 * GELU(x) = 0.5 * x * (1 + tanh[√(2/π) * (x + 0.044715 * x³)])
 *
 * DERIVATIVE FORMULA:
 * GELU'(x) = 0.5 * (1 + tanh(inner)) + 0.5 * x * (√(2/π) * (1 + 3*0.044715*x²)) * sech²(inner)
 * where inner = √(2/π) * (x + 0.044715 * x³)
 * and sech²(inner) = 1 - tanh²(inner)
 *
 * KERNEL DESIGN:
 * - 1D grid, 256 threads per block
 * - Each thread processes one element independently
 * - Uses built-in tanhf() for GPU-optimized approximation
 *
 * PERFORMANCE NOTES:
 * - Cubic term (x³) computed explicitly to avoid pow() overhead
 * - Derivative recomputes tanh (could cache if memory allows)
 */

#include "GELU.hpp"
#include "HyperParameters/HyperParameters_GPU.hpp"

#include <cmath>
#include <cstdio>  // For fprintf error reporting

namespace GRIM {

// Block size for GELU kernels (from HyperParameters)
static constexpr int kGeluBlockSize = HyperParameters::CUDA_BLOCK_SIZE_STANDARD;

namespace {

// Constants for GELU tanh approximation (from HyperParameters)
constexpr float kSqrt2OverPi = HyperParameters::GELU_SQRT_2_OVER_PI;
constexpr float kCubicCoeff = HyperParameters::GELU_CUBIC_COEFF;

/**
 * @brief GELU activation: 0.5 * x * (1 + tanh[√(2/π) * (x + 0.044715*x³)])
 */
__device__ inline float geluActivation(float x) {
    const float x3 = x * x * x;  // Explicit multiply faster than powf(x, 3)
    const float inner = kSqrt2OverPi * (x + kCubicCoeff * x3);
    return 0.5f * x * (1.0f + tanhf(inner));
}

/**
 * @brief GELU derivative for backward pass
 * Applies chain rule: ∂GELU/∂x
 * NOTE: Recomputes tanh(inner). Could cache from forward pass if memory allows.
 */
__device__ inline float geluDerivative(float x) {
    const float x2 = x * x;
    const float x3 = x2 * x;
    const float inner = kSqrt2OverPi * (x + kCubicCoeff * x3);
    const float tanh_inner = tanhf(inner);
    const float sech2 = 1.0f - tanh_inner * tanh_inner;  // sech²(y) = 1 - tanh²(y)
    const float term = kSqrt2OverPi * (1.0f + 3.0f * kCubicCoeff * x2);  // d(inner)/dx
    return 0.5f * (1.0f + tanh_inner) + 0.5f * x * term * sech2;
}

__global__ void GeluForwardKernel(const float* input, float* output, std::size_t elements) {
    const std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= elements) {
        return;
    }
    output[idx] = geluActivation(input[idx]);
}

/**
 * @brief GELU backward kernel: grad_input[i] = grad_output[i] * GELU'(input[i])
 * Grid: 1D, enough blocks to cover elements
 * Block: 256 threads
 * Requires original forward input for derivative computation
 */
__global__ void GeluBackwardKernel(const float* input,
                                   const float* grad_output,
                                   float* grad_input,
                                   std::size_t elements) {
    const std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= elements) {
        return;
    }
    const float x = input[idx];
    const float grad = grad_output[idx];
    grad_input[idx] = grad * geluDerivative(x);  // Chain rule
}

inline int computeGridSize(std::size_t elements, int block_size) {
    return static_cast<int>((elements + block_size - 1) / block_size);
}

} // namespace

GELULayer::GELULayer() = default;

GELULayer::GELULayer(const GELUConfig& config) : config_(config) {}

GELULayer::GELULayer(const Dimensions& dims, const GELUConfig& config)
    : Layer(dims), config_(config) {}

void GELULayer::forward(const GELUForwardArgs& args, LayerWorkspace<float>* /*workspace*/) {
    args.validate("GELULayer::forward");
    
    GELUConfig cfg = config_;
    cfg.elements = args.elements();
    cfg.stream = args.stream;
    config_ = cfg;
    
    last_input_ = args.input.ptr;  // Store for backward pass
    launchGeluForward(args);
}

void GELULayer::backward(const GELUBackwardArgs& args, LayerWorkspace<float>* /*workspace*/) {
    args.validate("GELULayer::backward");
    launchGeluBackward(args);
}

GELUConfig GELULayer::makeConfig(const GELUForwardArgs& args) const {
    GELUConfig cfg = config_;
    cfg.elements = args.elements();
    cfg.stream = args.stream;
    return cfg;
}

GELUConfig GELULayer::makeConfig(const GELUBackwardArgs& args) const {
    GELUConfig cfg = config_;
    cfg.elements = args.elements();
    cfg.stream = args.stream;
    return cfg;
}

void launchGeluForward(const GELUForwardArgs& args) {
    // RULE 20: Fail loud validation
    args.validate("launchGeluForward");

    const std::size_t elements = args.elements();
    const int grid = computeGridSize(elements, kGeluBlockSize);
    GeluForwardKernel<<<grid, kGeluBlockSize, 0, args.stream ? args.stream : nullptr>>>(
        args.input.ptr, args.output.ptr, elements);
    
    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("[GELU] GeluForwardKernel launch failed: ") +
                                cudaGetErrorString(err));
    }
}

void launchGeluBackward(const GELUBackwardArgs& args) {
    // RULE 20: Fail loud validation
    args.validate("launchGeluBackward");

    const std::size_t elements = args.elements();
    const int grid = computeGridSize(elements, kGeluBlockSize);
    GeluBackwardKernel<<<grid, kGeluBlockSize, 0, args.stream ? args.stream : nullptr>>>(
        args.input.ptr, args.grad_output.ptr, args.grad_input.ptr, elements);
    
    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("[GELU] GeluBackwardKernel launch failed: ") +
                                cudaGetErrorString(err));
    }
}

} // namespace GRIM

