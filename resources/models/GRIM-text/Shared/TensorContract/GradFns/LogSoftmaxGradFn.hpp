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
    std::shared_ptr<Tensor> input_gradient;
    float* saved_log_softmax = nullptr;
    bool owns_saved_log_softmax = true;
    int num_tokens = 0;
    int dim = 0;

    LogSoftmaxGradFn();
    ~LogSoftmaxGradFn() override;

    void capture_input(Tensor& x, cudaStream_t stream);
    void save(const float* log_softmax_output, int tokens, int d, cudaStream_t stream, bool copy = true);
    void apply_impl(const Tensor& grad_output,
                    cudaStream_t stream,
                    const Batching::BatchPayload* backward_payload,
                    const Batching::BatchDeviceBindings* backward_bindings) override;
    void release_saved() override;
};

}  // namespace autograd
}  // namespace GRIM
