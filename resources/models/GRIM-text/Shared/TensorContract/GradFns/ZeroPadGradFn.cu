//======================================================//
//  ZeroPadGradFn.cu
//  Row-offset zero-pad forward + autograd backward.
//======================================================//

#include "ZeroPadGradFn.hpp"
#include "../GradientAccumulation.hpp"
#include "../TensorContract_GPU.hpp"
#include "../../CudaAllocUtils.hpp"

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdint>
#include <stdexcept>
#include <string>

#define AG_TRACE(...) do { if (g_autograd_verbose) { fprintf(stderr, __VA_ARGS__); fflush(stderr); } } while(0)

void trackKernelLaunch(const char* kernel_name, cudaStream_t stream);

namespace GRIM {

using CudaAlloc::cudaMallocOrThrow;

namespace autograd {

ZeroPadGradFn::ZeroPadGradFn() {
    op_name = "zero_pad";
}

ZeroPadGradFn::~ZeroPadGradFn() {
    release_saved();
}

void ZeroPadGradFn::capture_input(Tensor& x, cudaStream_t stream, size_t offset_elems) {
    input_requires_grad = x.requires_grad;
    input_shape = x.shape;
    input_grad_fn = x.grad_fn;
    input_count = x.numel();
    offset_elements = offset_elems;

    if (input_requires_grad) {
        if (x.is_leaf) {
            x.ensure_grad();
            input_grad = x.grad_data();
        } else {
            float* buffer = nullptr;
            cudaMallocOrThrow(reinterpret_cast<void**>(&buffer), input_count * sizeof(float), "ZeroPadGradFn_input_grad");
            cudaMemsetAsync(buffer, 0, input_count * sizeof(float), stream);
            owned_input_grad = std::shared_ptr<float>(buffer, [](float* p) { queueForDeferredCleanup(p); });
            input_grad = owned_input_grad.get();
        }
    }
}

void ZeroPadGradFn::apply_impl(const Tensor& grad_output, cudaStream_t stream) {
    setCurrentGradFnOp("zero_pad", this);
    if (applied) return;
    applied = true;

    if (!input_requires_grad) return;
    if (!input_grad) {
        throw std::runtime_error("ZeroPadGradFn::apply: input_grad is NULL");
    }

    // grad_x += grad_output[offset_elements : offset_elements + input_count]
    accumulate_grad(input_grad,
                    grad_output.data + offset_elements,
                    input_count,
                    1.0f,
                    stream,
                    "ZeroPadGradFn::apply input_grad");
    trackKernelLaunch("zero_pad_backward", stream);

    if (input_grad_fn) {
        Tensor view;
        view.data = input_grad; view.shape = input_shape;
        view.owns_data = false; view.stream = stream;
        input_grad_fn->apply(view, stream);
    }
}

void ZeroPadGradFn::release_saved() {
    GradFn::release_saved();
    input_grad = nullptr;
    input_grad_fn.reset();
}

// ═══════════════════════════════════════════════════════════════════════════
// autograd::zero_pad — forward op
// No custom kernel: forward uses cudaMemsetAsync + cudaMemcpyAsync.
// ═══════════════════════════════════════════════════════════════════════════

Tensor zero_pad(const Tensor& x, int row_offset, int total_rows, cudaStream_t stream) {
    if (stream == nullptr || stream == 0) {
        throw std::runtime_error("autograd::zero_pad: stream is NULL - caller MUST provide valid stream");
    }
    if (!x.data) {
        throw std::runtime_error("autograd::zero_pad: input data is NULL");
    }
    if (!x.shape.is_2d_layout()) {
        throw std::runtime_error("autograd::zero_pad: input must be a 2D layout [rows, cols], got non-2D layout");
    }
    const int row_tokens = x.shape.flat.rows;
    const int cols = x.shape.flat.cols;
    if (row_offset < 0 || row_offset + row_tokens > total_rows) {
        throw std::runtime_error("autograd::zero_pad: offset=" + std::to_string(row_offset) +
                                 " + rows=" + std::to_string(row_tokens) +
                                 " > total_rows=" + std::to_string(total_rows));
    }

    Tensor result = Tensor::zeros(TensorContract::TensorShape::make_BSM(total_rows, cols), x.requires_grad, stream, "zero_pad_result");

    const size_t offset_elements = static_cast<size_t>(row_offset) * cols;
    const size_t slice_bytes = static_cast<size_t>(row_tokens) * cols * sizeof(float);
    cudaMemcpyAsync(result.data + offset_elements, x.data, slice_bytes,
                    cudaMemcpyDeviceToDevice, stream);

    if (x.requires_grad) {
        result.is_leaf = false;
        auto grad_fn = std::make_shared<ZeroPadGradFn>();
        grad_fn->capture_input(const_cast<Tensor&>(x), stream, offset_elements);
        result.grad_fn = grad_fn;
    }

    return result;
}

}  // namespace autograd
}  // namespace GRIM
