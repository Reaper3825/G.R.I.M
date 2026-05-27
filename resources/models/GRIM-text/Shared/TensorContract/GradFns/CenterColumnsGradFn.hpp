#pragma once
//======================================================//
//  CenterColumnsGradFn.hpp
//  Backward node for column-wise centering.
//======================================================//

#include "../TensorContract_GPU.hpp"

#include <cstddef>
#include <memory>
#include <vector>
#include <cuda_runtime.h>

namespace GRIM {
namespace autograd {

struct CenterColumnsGradFn : public GradFn {
    bool input_requires_grad = false;
    TensorContract::TensorShape input_shape;
    std::size_t element_count = 0;
    int num_cols = 0;
    int num_rows = 0;
    int rows_per_group = 0;
    int group_count = 0;
    bool use_sequence_lengths = false;
    bool use_causal_prefix = false;
    std::vector<int> sequence_lengths;

    float* input_grad = nullptr;
    std::shared_ptr<GradFn> input_grad_fn;
    std::shared_ptr<float> owned_input_grad;
    bool input_is_leaf = false;
    float* leaf_grad_buf = nullptr;

    CenterColumnsGradFn();

    void capture_input(Tensor& input, int cols, int rows, int group_rows,
                       const std::vector<int>* sequence_lengths, int groups,
                       cudaStream_t stream,
                       bool causal_prefix = false);
    void apply_impl(const Tensor& grad_output, cudaStream_t stream) override;
    void release_saved() override;
};

}  // namespace autograd
}  // namespace GRIM
