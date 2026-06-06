#pragma once
//======================================================//
//  ReciprocalGradFn.hpp
//  Backward node for element-wise reciprocal: y = 1/x.
//  Uses saved output: d/dx (1/x) = -1/x^2 = -y^2.
//
//  Owns:
//    - struct ReciprocalGradFn        (declared here)
//    - kernel_reciprocal_forward      (defined in ReciprocalGradFn.cu)
//    - kernel_reciprocal_backward     (defined in ReciprocalGradFn.cu)
//    - autograd::reciprocal(...)      (forward op; defined in ReciprocalGradFn.cu)
//======================================================//

#include "../TensorContract_GPU.hpp"

#include <memory>
#include <cstddef>
#include <cuda_runtime.h>

namespace GRIM {
namespace autograd {

struct ReciprocalGradFn : public GradFn {
    bool input_requires_grad = false;
    float* input_grad = nullptr;
    std::shared_ptr<float> owned_input_grad;
    TensorContract::TensorShape input_shape;
    std::shared_ptr<GradFn> input_grad_fn;
    const float* cached_output = nullptr;          ///< Non-owning ref into result.data
    std::size_t cached_size = 0;

    ReciprocalGradFn();
    ~ReciprocalGradFn() override;

    void capture_input(Tensor& x, cudaStream_t stream);
    void save_output(const float* output_data, std::size_t size);
    void apply_impl(const Tensor& grad_output,
                    cudaStream_t stream,
                    const Batching::BatchPayload* backward_payload,
                    const Batching::BatchDeviceBindings* backward_bindings) override;
    void release_saved() override;
};

}  // namespace autograd
}  // namespace GRIM
