#pragma once
//======================================================//
//  BiasAddGradFn.hpp
//  Backward node for broadcast bias-add: out[i,j] = input[i,j] + bias[j].
//  Forward: pass-through copy + per-feature bias broadcast.
//  Backward:
//    grad_input = grad_output           (pure pass-through)
//    grad_bias[j] = sum_i(grad_output[i,j])  (reduction over tokens)
//
//  Owns:
//    - struct BiasAddGradFn        (declared here)
//    - autograd::broadcast_add(...) (forward op; defined in BiasAddGradFn.cu)
//
//  Forward bias kernels (biasAddKernel/biasBackwardKernel) and launch
//  helpers live in the .cu under internal linkage — they are owned by the
//  autograd layer (not FFN-specific) and are only consumed by this op.
//
//  ISSUE #97: encoder biases (b_qkv, b_o, b1, b2) used to be frozen because
//  bias-add was a raw CUDA kernel with NO autograd tracking; this op fixes
//  that by giving bias-add a proper backward node.
//======================================================//

#include "TensorContract_GPU.hpp"

#include <memory>
#include <cstddef>
#include <cuda_runtime.h>

namespace GRIM {
namespace autograd {

struct BiasAddGradFn : public GradFn {
    bool input_requires_grad = false;
    bool bias_requires_grad = false;
    float* grad_input = nullptr;
    float* grad_bias = nullptr;
    std::shared_ptr<float> owned_grad_input;
    std::shared_ptr<float> owned_grad_bias;
    TensorContract::TensorShape input_shape;
    TensorContract::TensorShape bias_shape;
    std::shared_ptr<GradFn> input_grad_fn;
    std::size_t total_tokens = 0;
    std::size_t features = 0;

    BiasAddGradFn();

    void capture_inputs(Tensor& input, Tensor& bias,
                        int num_tokens, int num_features,
                        cudaStream_t stream);

    void apply(const Tensor& grad_output, cudaStream_t stream) override;
    void release_saved() override;
};

}  // namespace autograd
}  // namespace GRIM
