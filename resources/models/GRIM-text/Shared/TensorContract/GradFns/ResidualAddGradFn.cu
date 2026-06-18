//======================================================//
//  ResidualAddGradFn.cu
//  Residual / skip-connection add forward + autograd backward.
//
//  Forward: y = x + residual  (delegates to TensorContract::add)
//  Backward: both inputs receive grad_output unchanged
//    (d(x + residual)/dx = 1, d(x + residual)/d(residual) = 1)
//
//  Distinct from AddGradFn purely so "residual_add" surfaces in op_name
//  traces; gradient math is identical.
//======================================================//

#include "ResidualAddGradFn.hpp"
#include "../GradientAccumulation.hpp"
#include "../TensorContract_GPU.hpp"
#include "../../CudaAllocUtils.hpp"

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdint>
#include <stdexcept>
#include <string>

#define AG_TRACE(...) do { if (g_autograd_verbose) { fprintf(stderr, __VA_ARGS__); fflush(stderr); } } while(0)

namespace GRIM {

using CudaAlloc::cudaMallocOrThrow;

namespace autograd {

ResidualAddGradFn::ResidualAddGradFn() {
    op_name = "residual_add";
}

void ResidualAddGradFn::capture_inputs(Tensor& x, Tensor& r, cudaStream_t stream) {
    input_requires_grad = x.requires_grad;
    residual_requires_grad = r.requires_grad;
    input_shape = x.shape;
    residual_shape = r.shape;

    input_grad_fn = x.grad_fn;
    residual_grad_fn = r.grad_fn;
    register_input(x.grad_fn);
    register_input(r.grad_fn);

    element_count = x.numel();

    if (input_requires_grad) {
        if (x.is_leaf) {
            x.ensure_grad();
            input_grad = x.grad_data();
        } else {
            const size_t x_numel = x.numel();
            float* buffer = nullptr;
            cudaMallocOrThrow(reinterpret_cast<void**>(&buffer), x_numel * sizeof(float), "ResidualAddGradFn_input_grad");
            cudaMemsetAsync(buffer, 0, x_numel * sizeof(float), stream);
            owned_input_grad = std::shared_ptr<float>(buffer, [](float* p) { queueForDeferredCleanup(p); });
            input_grad = owned_input_grad.get();
        }
    }
    if (residual_requires_grad) {
        if (r.is_leaf) {
            r.ensure_grad();
            residual_grad = r.grad_data();
        } else {
            const size_t r_numel = r.numel();
            float* buffer = nullptr;
            cudaMallocOrThrow(reinterpret_cast<void**>(&buffer), r_numel * sizeof(float), "ResidualAddGradFn_residual_grad");
            cudaMemsetAsync(buffer, 0, r_numel * sizeof(float), stream);
            owned_residual_grad = std::shared_ptr<float>(buffer, [](float* p) { queueForDeferredCleanup(p); });
            residual_grad = owned_residual_grad.get();
        }
    }
}

void ResidualAddGradFn::apply_impl(const Tensor& grad_output,
                                   cudaStream_t stream,
                                   const Batching::BatchPayload* backward_payload,
                                   const Batching::BatchDeviceBindings* backward_bindings) {
    setCurrentGradFnOp("residual_add", this);

    if (applied) {
        return;
    }
    applied = true;

    const size_t count = grad_output.numel();

    if (input_requires_grad && input_grad) {
        accumulate_grad(input_grad, grad_output.data, count, 1.0f, stream, "ResidualAddGradFn::apply input_grad");
    }

    if (residual_requires_grad && residual_grad) {
        accumulate_grad(residual_grad, grad_output.data, count, 1.0f, stream, "ResidualAddGradFn::apply residual_grad");
    }

    if (input_requires_grad && input_grad_fn) {
        Tensor view;
        view.data = input_grad; view.shape = input_shape;
        view.owns_data = false; view.stream = stream;
        input_grad_fn->apply(view, stream, backward_payload, backward_bindings);
    }
    if (residual_requires_grad && residual_grad_fn && residual_grad_fn != input_grad_fn) {
        Tensor view;
        view.data = residual_grad; view.shape = residual_shape;
        view.owns_data = false; view.stream = stream;
        residual_grad_fn->apply(view, stream, backward_payload, backward_bindings);
    }
}

void ResidualAddGradFn::release_saved() {
    GradFn::release_saved();
    input_grad = nullptr;
    residual_grad = nullptr;
    input_grad_fn.reset();
    residual_grad_fn.reset();
}

Tensor residual_add(const Tensor& x, const Tensor& residual, cudaStream_t stream) {
    if (x.numel() != residual.numel()) {
        throw std::invalid_argument("autograd::residual_add: tensor size mismatch");
    }

    Tensor result = Tensor::empty(x.shape, x.requires_grad || residual.requires_grad, stream, "residual_add_result");

    // Forward: y = x + residual
    TensorContract::add(x, residual, result, stream);

    if (result.requires_grad) {
        result.is_leaf = false;
        auto grad_fn = std::make_shared<ResidualAddGradFn>();
        grad_fn->capture_inputs(const_cast<Tensor&>(x), const_cast<Tensor&>(residual), stream);
        result.grad_fn = grad_fn;
    }

    return result;
}

}  // namespace autograd
}  // namespace GRIM
