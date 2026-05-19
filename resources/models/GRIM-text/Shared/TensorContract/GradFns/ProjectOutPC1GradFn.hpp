#pragma once
//======================================================//
//  ProjectOutPC1GradFn.hpp
//  Backward node for projecting out the dominant PC1 direction.
//======================================================//

#include "../TensorContract_GPU.hpp"

#include <cstddef>
#include <memory>
#include <cuda_runtime.h>

namespace GRIM {
namespace autograd {

struct ProjectOutPC1GradFn : public GradFn {
    bool input_requires_grad = false;
    TensorContract::TensorShape input_shape;
    std::size_t element_count = 0;
    int num_rows = 0;
    int num_cols = 0;
    int power_iters = 0;

    float* input_grad = nullptr;
    std::shared_ptr<GradFn> input_grad_fn;
    std::shared_ptr<float> owned_input_grad;

    float* input_data_saved = nullptr;
    std::shared_ptr<float> owned_input_data;

    float* g_saved = nullptr;
    std::shared_ptr<float> owned_g;

    float* g_history_saved = nullptr;
    std::shared_ptr<float> owned_g_history;

    float* inv_norm_saved = nullptr;
    std::shared_ptr<float> owned_inv_norms;

    ProjectOutPC1GradFn();

    void capture_input(Tensor& input, int rows, int cols, int n_power_iters,
                       float* g_device, float* g_history_device, float* inv_norm_device,
                       cudaStream_t stream);
    void apply_impl(const Tensor& grad_output, cudaStream_t stream) override;
    void release_saved() override;
};

}  // namespace autograd
}  // namespace GRIM
