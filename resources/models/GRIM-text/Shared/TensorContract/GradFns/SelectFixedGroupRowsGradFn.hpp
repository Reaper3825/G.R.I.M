#pragma once
//======================================================//
//  SelectFixedGroupRowsGradFn.hpp
//  Primitive row selection within a fixed rectangular group layout.
//======================================================//

#include "../TensorContract_GPU.hpp"

#include <cuda_runtime.h>
#include <memory>

namespace GRIM::autograd {

struct SelectFixedGroupRowsGradFn final : public GradFn {
    std::shared_ptr<Tensor> input_gradient;
    int group_count = 0;
    int rows_per_group = 0;
    int row_offset = 0;
    int selected_rows_per_group = 0;
    int feature_count = 0;

    SelectFixedGroupRowsGradFn();

    void captureInput(Tensor& input, cudaStream_t stream);
    void apply_impl(
        const Tensor& grad_output,
        cudaStream_t stream,
        const Batching::BatchPayload* backward_payload,
        const Batching::BatchDeviceBindings* backward_bindings) override;
    void release_saved() override;
};

} // namespace GRIM::autograd
