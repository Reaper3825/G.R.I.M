//======================================================//
//  ConcatGradFn.cu
//  CUDA kernels and autograd node for row-wise concat.
//======================================================//

#include "ConcatGradFn.hpp"
#include "../AutogradEngine.hpp"
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

namespace {

constexpr int AUTOGRAD_BLOCK_SIZE = 256;

//========================================================================
// Concat Forward: out[i, 0:D1] = a[i,:], out[i, D1:D1+D2] = b[i,:]
//========================================================================
__global__ void kernel_concat_forward(
    float* __restrict__ output,
    const float* __restrict__ a,
    const float* __restrict__ b,
    int N, int D1, int D2
) {
    const int row = blockIdx.x;
    if (row >= N) return;
    const int D = D1 + D2;
    for (int j = threadIdx.x; j < D; j += blockDim.x) {
        if (j < D1)
            output[static_cast<size_t>(row) * D + j] = a[static_cast<size_t>(row) * D1 + j];
        else
            output[static_cast<size_t>(row) * D + j] = b[static_cast<size_t>(row) * D2 + (j - D1)];
    }
}

__global__ void kernel_concat_backward_a(
    float* __restrict__ grad_a,
    const float* __restrict__ grad_out,
    int N, int D1, int D_total
) {
    const int row = blockIdx.x;
    if (row >= N) return;
    for (int j = threadIdx.x; j < D1; j += blockDim.x)
        grad_a[static_cast<size_t>(row) * D1 + j] += grad_out[static_cast<size_t>(row) * D_total + j];
}

__global__ void kernel_concat_backward_b(
    float* __restrict__ grad_b,
    const float* __restrict__ grad_out,
    int N, int D1, int D2, int D_total
) {
    const int row = blockIdx.x;
    if (row >= N) return;
    for (int j = threadIdx.x; j < D2; j += blockDim.x)
        grad_b[static_cast<size_t>(row) * D2 + j] += grad_out[static_cast<size_t>(row) * D_total + (D1 + j)];
}

}  // anonymous namespace

namespace GRIM {
namespace autograd {

ConcatGradFn::ConcatGradFn() {
    op_name = "concat";
}

void ConcatGradFn::capture_inputs(Tensor& a, Tensor& b, cudaStream_t stream) {
    if (!stream) throw std::runtime_error("ConcatGradFn::capture: stream is NULL");
    a.require("ConcatGradFn::capture a");
    b.require("ConcatGradFn::capture b");
    a_requires_grad = a.requires_grad;
    b_requires_grad = b.requires_grad;
    a_shape = a.shape;
    b_shape = b.shape;
    auto capture = [&](Tensor& input, std::shared_ptr<GradFn>& producer, Tensor*& leaf) {
        producer.reset();
        leaf = nullptr;
        if (!input.requires_grad) return;
        if (input.is_leaf) {
            if (input.grad_fn) throw std::runtime_error("ConcatGradFn::capture: leaf has a producer");
            input.ensure_grad();
            leaf = input.grad_.get();
            if (!leaf) throw std::runtime_error("ConcatGradFn::capture: missing leaf gradient");
        } else {
            if (!input.grad_fn) throw std::runtime_error("ConcatGradFn::capture: non-leaf has no producer");
            producer = input.grad_fn;
            register_input(producer);
        }
    };
    capture(a, a_grad_fn, leaf_grad_a);
    capture(b, b_grad_fn, leaf_grad_b);
}

void ConcatGradFn::apply_impl(const Tensor& grad_output,
                              cudaStream_t stream,
                              const Batching::BatchPayload* backward_payload,
                              const Batching::BatchDeviceBindings* backward_bindings) {
    setCurrentGradFnOp("concat", this);
    if (applied) return;
    if (!stream) throw std::runtime_error("ConcatGradFn::apply: stream is NULL");
    grad_output.require("ConcatGradFn::apply grad_output");

    const int D_total = D1 + D2;
    const auto expected_output = TensorContract::TensorShape::make_BSM(rows, D_total);
    if (grad_output.shape.layout != expected_output.layout ||
        grad_output.shape.as_2d() != expected_output.as_2d()) {
        throw std::runtime_error(
            "ConcatGradFn::apply: grad_output shape does not match captured concat output");
    }

    auto destination = [&](bool required, const std::shared_ptr<GradFn>& producer,
                           Tensor* leaf, const TensorContract::TensorShape& shape) -> Tensor* {
        if (!required) return nullptr;
        if (producer) return &producer->gradient_destination(shape, stream);
        if (!leaf) throw std::runtime_error("ConcatGradFn::apply: missing leaf destination");
        leaf->require("ConcatGradFn::apply leaf");
        return leaf;
    };
    Tensor* grad_a = destination(a_requires_grad, a_grad_fn, leaf_grad_a, a_shape);
    Tensor* grad_b = destination(b_requires_grad, b_grad_fn, leaf_grad_b, b_shape);
    applied = true;

    if (a_requires_grad) {
        if (!grad_a) {
            throw std::runtime_error("ConcatGradFn::apply: grad_a Tensor is missing");
        }
        grad_a->require("ConcatGradFn::apply grad_a");
        kernel_concat_backward_a<<<rows, AUTOGRAD_BLOCK_SIZE, 0, stream>>>(
            grad_a->data, grad_output.data, rows, D1, D_total);
        const cudaError_t launch_status = cudaGetLastError();
        if (launch_status != cudaSuccess) {
            throw std::runtime_error(
                std::string("ConcatGradFn::apply: grad_a kernel launch failed: ") +
                cudaGetErrorString(launch_status));
        }
    }
    if (b_requires_grad) {
        if (!grad_b) {
            throw std::runtime_error("ConcatGradFn::apply: grad_b Tensor is missing");
        }
        grad_b->require("ConcatGradFn::apply grad_b");
        kernel_concat_backward_b<<<rows, AUTOGRAD_BLOCK_SIZE, 0, stream>>>(
            grad_b->data, grad_output.data, rows, D1, D2, D_total);
        const cudaError_t launch_status = cudaGetLastError();
        if (launch_status != cudaSuccess) {
            throw std::runtime_error(
                std::string("ConcatGradFn::apply: grad_b kernel launch failed: ") +
                cudaGetErrorString(launch_status));
        }
    }

    // Finish both writes before notifying each deduplicated producer edge.
    // The engine destination already contains these contributions.
    if (a_requires_grad && !a_grad_fn) leaf_grad_a->record_leaf_gradient_delivery();
    if (b_requires_grad && !b_grad_fn) leaf_grad_b->record_leaf_gradient_delivery();
    auto notify = [&](const std::shared_ptr<GradFn>& producer) {
        if (!producer) return;
        if (auto* engine = AutogradEngine::active()) {
            engine->contribute(producer.get());
        } else {
            producer->apply(producer->pending_gradient("ConcatGradFn::apply producer"),
                            stream, backward_payload, backward_bindings);
        }
    };
    notify(a_grad_fn);
    if (b_grad_fn != a_grad_fn) notify(b_grad_fn);
}

void ConcatGradFn::release_saved() {
    GradFn::release_saved();
    leaf_grad_a = nullptr;
    leaf_grad_b = nullptr;
    a_grad_fn.reset();
    b_grad_fn.reset();
}

Tensor concat(const Tensor& a, const Tensor& b, cudaStream_t stream) {
    if (!a.shape.is_2d_layout() || !b.shape.is_2d_layout()) {
        throw std::invalid_argument("autograd::concat: both inputs must be 2D");
    }
    const auto a_dims = a.shape.as_2d();
    const auto b_dims = b.shape.as_2d();
    if (a_dims.rows != b_dims.rows) {
        throw std::invalid_argument("autograd::concat: row count mismatch (a=" +
            std::to_string(a_dims.rows) + " b=" + std::to_string(b_dims.rows) + ")");
    }
    if (!a.data || !b.data) {
        throw std::invalid_argument("autograd::concat: null data pointer");
    }

    const int N = a_dims.rows;
    const int D1 = a_dims.cols;
    const int D2 = b_dims.cols;

    const bool needs_grad = a.requires_grad || b.requires_grad;
    auto shape = TensorContract::TensorShape::make_BSM(N, D1 + D2);
    Tensor result = Tensor::empty(shape, needs_grad, stream, "concat_result");

    kernel_concat_forward<<<N, AUTOGRAD_BLOCK_SIZE, 0, stream>>>(
        result.data, a.data, b.data, N, D1, D2);

    if (needs_grad) {
        result.is_leaf = false;
        auto grad_fn = std::make_shared<ConcatGradFn>();
        grad_fn->capture_inputs(const_cast<Tensor&>(a), const_cast<Tensor&>(b), stream);
        grad_fn->rows = N;
        grad_fn->D1 = D1;
        grad_fn->D2 = D2;
        result.grad_fn = grad_fn;
    }

    return result;
}

}  // namespace autograd
}  // namespace GRIM
