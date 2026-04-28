#pragma once
//======================================================//
//  LogSoftmaxGradFn.hpp
//  Backward node for log_softmax.
//======================================================//

#include "../TensorContract_GPU.hpp"

#include <cuda_runtime.h>
#include <memory>

namespace GRIM {
namespace autograd {

struct LogSoftmaxGradFn : public GradFn {
    bool input_requires_grad = false;
    float* input_grad = nullptr;
    bool owns_input_grad = false;
    TensorContract::TensorShape input_shape;
    std::shared_ptr<GradFn> input_grad_fn;
    float* saved_log_softmax = nullptr;
    bool owns_saved_log_softmax = true;
    int num_tokens = 0;
    int dim = 0;

    LogSoftmaxGradFn();
    ~LogSoftmaxGradFn() override;

    void capture_input(Tensor& x);
    void save(const float* log_softmax_output, int tokens, int d, cudaStream_t stream, bool copy = true);
    void apply(const Tensor& grad_output, cudaStream_t stream) override;
    void release_saved() override;
};

}  // namespace autograd
}  // namespace GRIM
