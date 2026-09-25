//======================================================//
//  ElementwiseMulGradFn.cu
//  Element-wise (Hadamard) multiply forward + autograd backward.
//
//  Forward:  y[i] = a[i] * b[i]
//  Backward:
//    grad_a[i] += grad_y[i] * b[i]
//    grad_b[i] += grad_y[i] * a[i]
//
//  Backward uses non-owning cached references to a/b — see header.
//======================================================//

#include "ElementwiseMulGradFn.hpp"
#include "../AutogradEngine.hpp"
#include "../TensorContract_GPU.hpp"

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdint>
#include <stdexcept>
#include <string>

#define AG_TRACE(...) do { if constexpr (GRIM::VerboseLogging::ENABLE_AUTOGRAD_TRACE_LOGS) { fprintf(stderr, __VA_ARGS__); fflush(stderr); } } while(0)

void trackKernelLaunch(const char* kernel_name, cudaStream_t stream);

namespace {

constexpr int AUTOGRAD_BLOCK_SIZE = 256;
constexpr int kMaxGridBlocks1DFallback = 65534;

inline int getMaxGridBlocks1D() {
    static int cached = -1;
    if (cached < 0) {
        int device = 0;
        if (cudaGetDevice(&device) != cudaSuccess) {
            cached = kMaxGridBlocks1DFallback;
        } else {
            int max_x = 0;
            if (cudaDeviceGetAttribute(&max_x, cudaDevAttrMaxGridDimX, device) != cudaSuccess) {
                cached = kMaxGridBlocks1DFallback;
            } else {
                cached = (max_x > 65534) ? 65534 : max_x;
            }
        }
    }
    return cached;
}

constexpr int kMaxGridDimY = 65535;

inline dim3 gridForCount(size_t count) {
    if (count == 0) return dim3(1, 1, 1);
    const int blocks = static_cast<int>((count + AUTOGRAD_BLOCK_SIZE - 1) / AUTOGRAD_BLOCK_SIZE);
    const int max1d = getMaxGridBlocks1D();
    if (blocks <= max1d) return dim3(blocks, 1, 1);
    int gy = (blocks + max1d - 1) / max1d;
    if (gy <= kMaxGridDimY) return dim3(max1d, gy, 1);
    int gx = (blocks + kMaxGridDimY - 1) / kMaxGridDimY;
    return dim3(gx, kMaxGridDimY, 1);
}

// Element-wise multiply forward: output = a ⊙ b
__global__ void kernel_elementwise_mul_forward(
    const float* __restrict__ a,
    const float* __restrict__ b,
    float* __restrict__ output,
    size_t count
) {
    const size_t block_idx = static_cast<size_t>(blockIdx.y) * gridDim.x + blockIdx.x;
    const size_t idx = block_idx * blockDim.x + threadIdx.x;
    if (idx < count) {
        output[idx] = a[idx] * b[idx];
    }
}

// Element-wise multiply backward (used for both grad_a and grad_b):
//   grad_self[i] += grad_output[i] * other[i]
__global__ void kernel_elementwise_mul_backward(
    const float* grad_output,
    const float* other,
    float* grad_self,
    size_t count
) {
    const size_t block_idx = static_cast<size_t>(blockIdx.y) * gridDim.x + blockIdx.x;
    const size_t idx = block_idx * blockDim.x + threadIdx.x;
    if (idx < count) {
        grad_self[idx] += grad_output[idx] * other[idx];
    }
}

}  // anonymous namespace

namespace GRIM {

namespace autograd {

ElementwiseMulGradFn::ElementwiseMulGradFn() {
    op_name = "elementwise_mul";
}

void ElementwiseMulGradFn::capture_inputs(Tensor& a, Tensor& b, cudaStream_t stream) {
    if (!stream) throw std::runtime_error("ElementwiseMulGradFn::capture: stream is NULL");
    if (a.numel() != b.numel()) throw std::runtime_error("ElementwiseMulGradFn::capture: size mismatch");
    a_requires_grad = a.requires_grad;
    b_requires_grad = b.requires_grad;
    a_shape = a.shape;
    b_shape = b.shape;
    auto capture = [&](Tensor& x, std::shared_ptr<GradFn>& producer, Tensor*& leaf) {
        x.require("ElementwiseMulGradFn::capture");
        producer.reset();
        leaf = nullptr;
        if (!x.requires_grad) return;
        if (x.is_leaf) {
            if (x.grad_fn) throw std::runtime_error("ElementwiseMulGradFn::capture: leaf has a producer");
            x.ensure_grad();
            leaf = x.grad_.get();
            if (!leaf) throw std::runtime_error("ElementwiseMulGradFn::capture: missing leaf gradient");
        } else {
            if (!x.grad_fn) throw std::runtime_error("ElementwiseMulGradFn::capture: non-leaf has no producer");
            producer = x.grad_fn;
            register_input(producer);
        }
    };
    capture(a, a_grad_fn, leaf_grad_a);
    capture(b, b_grad_fn, leaf_grad_b);
    cached_size = a.numel();
    cached_b = a_requires_grad ? b.data : nullptr;
    cached_a = b_requires_grad ? a.data : nullptr;
}

void ElementwiseMulGradFn::apply_impl(const Tensor& grad_output,
                                      cudaStream_t stream,
                                      const Batching::BatchPayload* backward_payload,
                                      const Batching::BatchDeviceBindings* backward_bindings) {
    setCurrentGradFnOp("elementwise_mul", this);
    if (applied) return;
    if (!stream) throw std::runtime_error("ElementwiseMulGradFn::apply: stream is NULL");
    grad_output.require("ElementwiseMulGradFn::apply grad_output");
    const size_t count = grad_output.numel();
    if (count != cached_size) throw std::runtime_error("ElementwiseMulGradFn::apply: gradient size mismatch");
    if ((a_requires_grad && !cached_b) || (b_requires_grad && !cached_a)) {
        throw std::runtime_error("ElementwiseMulGradFn::apply: missing cached input");
    }
    auto destination = [&](bool required, const std::shared_ptr<GradFn>& producer,
                           Tensor* leaf, const TensorContract::TensorShape& shape) -> Tensor* {
        if (!required) return nullptr;
        if (producer) return &producer->gradient_destination(shape, stream);
        if (!leaf) throw std::runtime_error("ElementwiseMulGradFn::apply: missing leaf destination");
        leaf->require("ElementwiseMulGradFn::apply leaf");
        return leaf;
    };
    Tensor* a_gradient = destination(a_requires_grad, a_grad_fn, leaf_grad_a, a_shape);
    Tensor* b_gradient = destination(b_requires_grad, b_grad_fn, leaf_grad_b, b_shape);
    applied = true;
    if (a_gradient) {
        kernel_elementwise_mul_backward<<<gridForCount(count), AUTOGRAD_BLOCK_SIZE, 0, stream>>>(
            grad_output.data, cached_b, a_gradient->data, count);
        trackKernelLaunch("kernel_elementwise_mul_backward_a", stream);
        if (!a_grad_fn) leaf_grad_a->record_leaf_gradient_delivery();
    }
    if (b_gradient) {
        kernel_elementwise_mul_backward<<<gridForCount(count), AUTOGRAD_BLOCK_SIZE, 0, stream>>>(
            grad_output.data, cached_a, b_gradient->data, count);
        trackKernelLaunch("kernel_elementwise_mul_backward_b", stream);
        if (!b_grad_fn) leaf_grad_b->record_leaf_gradient_delivery();
    }
    // Complete both writes before notifying each deduplicated producer edge.
    auto notify = [&](const std::shared_ptr<GradFn>& producer) {
        if (!producer) return;
        if (auto* engine = AutogradEngine::active()) {
            engine->contribute(producer.get());
        } else {
            producer->apply(producer->pending_gradient("ElementwiseMulGradFn::apply producer"),
                            stream, backward_payload, backward_bindings);
        }
    };
    notify(a_grad_fn);
    if (b_grad_fn != a_grad_fn) notify(b_grad_fn);
}

void ElementwiseMulGradFn::release_saved() {
    GradFn::release_saved();
    cached_a = nullptr;
    cached_b = nullptr;
    cached_size = 0;
    leaf_grad_a = nullptr;
    leaf_grad_b = nullptr;
    a_grad_fn.reset();
    b_grad_fn.reset();
}

Tensor elementwise_mul(const Tensor& a, const Tensor& b, cudaStream_t stream) {
    if (stream == nullptr || stream == 0) {
        throw std::runtime_error("autograd::elementwise_mul: stream is NULL");
    }
    if (a.numel() != b.numel()) {
        throw std::runtime_error("autograd::elementwise_mul: size mismatch a.numel()=" +
                                 std::to_string(a.numel()) + " b.numel()=" + std::to_string(b.numel()));
    }

    const bool needs_grad = a.requires_grad || b.requires_grad;
    Tensor result = Tensor::empty(a.shape, needs_grad, stream, "emul_result");

    const size_t count = a.numel();
    kernel_elementwise_mul_forward<<<gridForCount(count), AUTOGRAD_BLOCK_SIZE, 0, stream>>>(
        a.data, b.data, result.data, count);

    if (needs_grad) {
        result.is_leaf = false;
        auto grad_fn = std::make_shared<ElementwiseMulGradFn>();
        grad_fn->capture_inputs(const_cast<Tensor&>(a), const_cast<Tensor&>(b), stream);
        result.grad_fn = grad_fn;
    }

    return result;
}

}  // namespace autograd
}  // namespace GRIM
