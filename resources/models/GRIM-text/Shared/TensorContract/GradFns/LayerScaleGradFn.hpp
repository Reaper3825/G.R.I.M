#pragma once
//======================================================//
//  LayerScaleGradFn.hpp
//  Backward node for learnable scalar multiplication.
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
    std::shared_ptr<float> owned_input_data;
    std::shared_ptr<float> owned_input_grad;
    std::shared_ptr<GradFn> input_grad_fn;
    TensorContract::TensorShape input_shape;
    std::size_t element_count = 0;

    float scale_value = 1.0f;
    float* scale_grad = nullptr;

    LayerScaleGradFn();

    void capture_inputs(Tensor& input, Tensor& scale_param, float cached_scale_value, cudaStream_t stream);
    void apply(const Tensor& grad_output, cudaStream_t stream) override;
    void release_saved() override;
};

}  // namespace autograd
}  // namespace GRIM
