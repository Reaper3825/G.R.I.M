//======================================================//
//  AddGradFn.cu
//  Element-wise tensor add forward + autograd backward.
//  Forward delegates to TensorContract::add() (no local kernel).
//  Backward accumulates grad_output unchanged into both inputs.
//======================================================//

#include "AddGradFn.hpp"
#include "../GradientAccumulation.hpp"
#include "../TensorContract_GPU.hpp"
#include "../../CudaAllocUtils.hpp"

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdint>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>
#include <algorithm>

#define AG_TRACE(...) do { if (g_autograd_verbose) { fprintf(stderr, __VA_ARGS__); fflush(stderr); } } while(0)

namespace GRIM {

using CudaAlloc::cudaMallocOrThrow;

namespace autograd {

AddGradFn::AddGradFn() {
    op_name = "add";
}

void AddGradFn::capture_inputs(Tensor& a, Tensor& b, cudaStream_t stream) {
    a_requires_grad = a.requires_grad;
    b_requires_grad = b.requires_grad;
    a_shape = a.shape;
    b_shape = b.shape;

    a_grad_fn = a.grad_fn;
    b_grad_fn = b.grad_fn;

    element_count = a.numel();

    // ISSUE #56 FIX: Handle grad buffer ownership based on tensor type
    // Leaf tensors (weights) persist, safe to use their grad buffer directly
    // Non-leaf tensors (activations) are temporary, need owned buffer
    if (a_requires_grad) {
        a.ensure_grad();
        if (a.is_leaf) {
            grad_a = a.grad_data();
            AG_TRACE("[AddGradFn] Using persistent grad_a buffer (leaf): %p\n", (void*)grad_a);
        } else {
            const size_t a_numel = a.numel();
            float* buffer_a = nullptr;
            cudaMallocOrThrow(reinterpret_cast<void**>(&buffer_a), a_numel * sizeof(float), "AddGradFn_grad_a");
            cudaMemsetAsync(buffer_a, 0, a_numel * sizeof(float), stream);
            owned_grad_a = std::shared_ptr<float>(buffer_a, [](float* p) {
                queueForDeferredCleanup(p);
            });
            grad_a = owned_grad_a.get();
            AG_TRACE("[AddGradFn] Allocated owned grad_a buffer (non-leaf): %zu floats at %p\n", a_numel, (void*)grad_a);
        }
    }
    if (b_requires_grad) {
        b.ensure_grad();
        if (b.is_leaf) {
            grad_b = b.grad_data();
            AG_TRACE("[AddGradFn] Using persistent grad_b buffer (leaf): %p\n", (void*)grad_b);
        } else {
            const size_t b_numel = b.numel();
            float* buffer_b = nullptr;
            cudaMallocOrThrow(reinterpret_cast<void**>(&buffer_b), b_numel * sizeof(float), "AddGradFn_grad_b");
            cudaMemsetAsync(buffer_b, 0, b_numel * sizeof(float), stream);
            owned_grad_b = std::shared_ptr<float>(buffer_b, [](float* p) {
                queueForDeferredCleanup(p);
            });
            grad_b = owned_grad_b.get();
            AG_TRACE("[AddGradFn] Allocated owned grad_b buffer (non-leaf): %zu floats at %p\n", b_numel, (void*)grad_b);
        }
    }
}

void AddGradFn::capture_single_input(Tensor& a, cudaStream_t stream) {
    a_requires_grad = a.requires_grad;
    b_requires_grad = false;
    a_shape = a.shape;

    a_grad_fn = a.grad_fn;

    element_count = a.numel();

    if (a_requires_grad) {
        a.ensure_grad();
        if (a.is_leaf) {
            grad_a = a.grad_data();
        } else {
            const size_t a_numel = a.numel();
            float* buffer_a = nullptr;
            cudaMallocOrThrow(reinterpret_cast<void**>(&buffer_a), a_numel * sizeof(float), "AddGradFn_single_grad_a");
            cudaMemsetAsync(buffer_a, 0, a_numel * sizeof(float), stream);
            owned_grad_a = std::shared_ptr<float>(buffer_a, [](float* p) {
                queueForDeferredCleanup(p);
            });
            grad_a = owned_grad_a.get();
        }
    }
}

void AddGradFn::apply_impl(const Tensor& grad_output, cudaStream_t stream) {
    setCurrentGradFnOp("add", this);

    // ISSUE #49: Prevent infinite loops when grad_fn is shared by multiple ops
    if (applied) {
        return;
    }
    applied = true;

    if (!grad_output.data) {
        throw std::runtime_error("AddGradFn::apply: grad_output.data is NULL - backward called with null gradient");
    }

    const size_t count = grad_output.numel();

    if (a_requires_grad && grad_a) {
        accumulate_grad(grad_a, grad_output.data, count, 1.0f, stream, "AddGradFn::apply grad_a");
    }
    if (b_requires_grad && grad_b) {
        accumulate_grad(grad_b, grad_output.data, count, 1.0f, stream, "AddGradFn::apply grad_b");
    }

    // CONTINUE AUTOGRAD CHAIN using stored grad_fn pointers
    // For c = a + b: dc/da = 1, dc/db = 1, so upstream receives grad_output unchanged.
    // grad_a/grad_b are local accumulators (leaf buffers or owned intermediates) —
    // the chain must propagate the raw flowing gradient, not the accumulated buffer.
    if (a_requires_grad && a_grad_fn && a_grad_fn->op_name) {
        Tensor view;
        view.data = grad_output.data; view.shape = a_shape;
        view.owns_data = false; view.stream = stream;
        a_grad_fn->apply(view, stream);
    }

    if (b_requires_grad && b_grad_fn && b_grad_fn != a_grad_fn && b_grad_fn->op_name) {
        Tensor view;
        view.data = grad_output.data; view.shape = b_shape;
        view.owns_data = false; view.stream = stream;
        b_grad_fn->apply(view, stream);
    }
}

void AddGradFn::release_saved() {
    GradFn::release_saved();

    grad_a = nullptr;
    grad_b = nullptr;

    a_grad_fn.reset();
    b_grad_fn.reset();
}

Tensor add(const Tensor& a, const Tensor& b, cudaStream_t stream) {
    if (a.numel() != b.numel()) {
        throw std::invalid_argument("autograd::add: tensor size mismatch");
    }

    Tensor result = Tensor::empty(a.shape, a.requires_grad || b.requires_grad, stream, "add_result");

    // c = a + b — use TensorContract::add for the forward
    TensorContract::TensorView a_view(const_cast<float*>(a.data), a.shape);
    TensorContract::TensorView b_view(const_cast<float*>(b.data), b.shape);
    TensorContract::TensorView r_view(result.data, result.shape);
    TensorContract::add(a_view, b_view, r_view, stream);

    if (result.requires_grad) {
        result.is_leaf = false;
        auto grad_fn = std::make_shared<AddGradFn>();
        grad_fn->capture_inputs(const_cast<Tensor&>(a), const_cast<Tensor&>(b), stream);
        result.grad_fn = grad_fn;
    }

    return result;
}

}  // namespace autograd
}  // namespace GRIM
