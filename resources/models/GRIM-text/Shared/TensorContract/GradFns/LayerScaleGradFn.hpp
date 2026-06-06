#pragma once
//======================================================//
//  LayerScaleGradFn.hpp
//  Backward node for learnable per-channel residual scaling.
//======================================================//

#include "../TensorContract_GPU.hpp"

#include <cstddef>
#include <memory>
#include <cuda_runtime.h>

namespace GRIM {
namespace autograd {

struct LayerScaleGradFn : public GradFn {
    float* input_data = nullptr;
    float* input_grad = nullptr;
    const float* scale_data = nullptr;
    float* scale_grad = nullptr;
    std::shared_ptr<float> owned_input_data;
    std::shared_ptr<float> owned_input_grad;
    std::shared_ptr<float> owned_scale_data;
    std::shared_ptr<float> owned_scale_grad;
    std::shared_ptr<GradFn> input_grad_fn;
    std::shared_ptr<GradFn> scale_grad_fn;
    TensorContract::TensorShape input_shape;
    TensorContract::TensorShape scale_shape;
    std::size_t element_count = 0;
    int rows = 0;
    int cols = 0;

    LayerScaleGradFn();

    void capture_inputs(Tensor& input, Tensor& scale_param, cudaStream_t stream);
    void apply_impl(const Tensor& grad_output,
                    cudaStream_t stream,
                    const Batching::BatchPayload* backward_payload,
                    const Batching::BatchDeviceBindings* backward_bindings) override;
    void release_saved() override;
};

}  // namespace autograd
}  // namespace GRIM
