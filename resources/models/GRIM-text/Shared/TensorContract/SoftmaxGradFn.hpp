#pragma once
//======================================================//
//  SoftmaxGradFn.hpp
//  Backward node for softmax.
//======================================================//

#include "TensorContract_GPU.hpp"

#include <memory>
#include <cuda_runtime.h>

namespace GRIM {
namespace autograd {

struct SoftmaxGradFn : public GradFn {
    bool input_requires_grad = false;
    float* input_grad = nullptr;
    std::shared_ptr<float> owned_input_grad;
    TensorContract::TensorShape input_shape;
    std::shared_ptr<GradFn> input_grad_fn;
    float* saved_softmax = nullptr;
    int num_tokens = 0;
    int dim = 0;
    float inv_temperature = 1.0f;

    SoftmaxGradFn();
    ~SoftmaxGradFn() override;

    void capture_input(Tensor& x, cudaStream_t stream);
    void save(const float* softmax_output, int tokens_, int dim_, float inv_temp, cudaStream_t stream);
    void apply(const Tensor& grad_output, cudaStream_t stream) override;
    void release_saved() override;
};

}  // namespace autograd
}  // namespace GRIM
