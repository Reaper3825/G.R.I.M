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
    const float* scale_data = nullptr;
    std::shared_ptr<float> owned_input_data;
    std::shared_ptr<float> owned_scale_data;
    std::shared_ptr<Tensor> input_gradient;
    std::shared_ptr<Tensor> scale_gradient;
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
