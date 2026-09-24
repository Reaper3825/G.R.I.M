//======================================================//
//  AddGradFn.cu
//  Add delivers gradients directly to producer accumulators or leaf buffers.
//======================================================//

#include "AddGradFn.hpp"
#include "../AutogradEngine.hpp"
#include "../GradientAccumulation.hpp"

#include <stdexcept>

namespace GRIM::autograd {
namespace {

// Leaf storage remains owned by the input tensor. Non-leaf captures keep only
// the producer alive; its pending gradient is created on first delivery.
void captureAddInput(Tensor& input, std::shared_ptr<GradFn>& producer,
                     Tensor*& leaf_gradient, cudaStream_t stream) {
    if (!stream) throw std::runtime_error("AddGradFn::capture: stream is NULL");
    input.require("AddGradFn::capture");
    producer.reset();
    leaf_gradient = nullptr;
    if (!input.requires_grad) return;
    if (input.is_leaf) {
        if (input.grad_fn) throw std::runtime_error("AddGradFn::capture: leaf has a producer");
        input.ensure_grad();
        leaf_gradient = input.grad_.get();
        if (!leaf_gradient) throw std::runtime_error("AddGradFn::capture: missing leaf gradient");
    } else {
        if (!input.grad_fn) throw std::runtime_error("AddGradFn::capture: non-leaf has no producer");
        producer = input.grad_fn;
    }
}

} // namespace

AddGradFn::AddGradFn() { op_name = "add"; }

void AddGradFn::capture_inputs(Tensor& a, Tensor& b, cudaStream_t stream) {
    a_requires_grad = a.requires_grad;
    b_requires_grad = b.requires_grad;
    a_shape = a.shape;
    b_shape = b.shape;
    element_count = a.numel();
    captureAddInput(a, a_grad_fn, leaf_grad_a, stream);
    captureAddInput(b, b_grad_fn, leaf_grad_b, stream);
    register_input(a_grad_fn);
    register_input(b_grad_fn);
}

void AddGradFn::capture_single_input(Tensor& a, cudaStream_t stream) {
    a_requires_grad = a.requires_grad;
    b_requires_grad = false;
    a_shape = a.shape;
    element_count = a.numel();
    b_grad_fn.reset();
    leaf_grad_b = nullptr;
    captureAddInput(a, a_grad_fn, leaf_grad_a, stream);
    register_input(a_grad_fn);
}

void AddGradFn::apply_impl(const Tensor& grad_output,
                         cudaStream_t stream,
                         const Batching::BatchPayload* backward_payload,
                         const Batching::BatchDeviceBindings* backward_bindings) {
    setCurrentGradFnOp("add", this);
    if (applied) return;
    grad_output.require("AddGradFn::apply grad_output");
    if (!stream || grad_output.numel() != element_count) {
        throw std::runtime_error("AddGradFn::apply: invalid stream or gradient size");
    }
    applied = true;

    auto deliver = [&](bool required, const TensorContract::TensorShape& shape,
                       const std::shared_ptr<GradFn>& producer, Tensor* leaf) {
        if (!required) return;
        // Preserve the input's shape for same-numel views. This view borrows
        // grad_output only for delivery; receive_gradient copies into its own
        // accumulator and never adopts this pointer.
        Tensor contribution;
        contribution.data = grad_output.data;
        contribution.shape = shape;
        contribution.owns_data = false;
        contribution.stream = stream;
        if (producer) {
            producer->receive_gradient(contribution, stream);
        } else {
            if (!leaf) throw std::runtime_error("AddGradFn::apply: missing leaf destination");
            accumulate_grad(*leaf, contribution, 1.0f, stream, "AddGradFn::apply leaf");
            leaf->record_leaf_gradient_delivery();
        }
    };

    // Both mathematical inputs contribute, including add(x, x). Finish both
    // writes before issuing one notification per deduplicated producer edge.
    deliver(a_requires_grad, a_shape, a_grad_fn, leaf_grad_a);
    deliver(b_requires_grad, b_shape, b_grad_fn, leaf_grad_b);
    auto notify = [&](const std::shared_ptr<GradFn>& producer) {
        if (!producer) return;
        if (auto* engine = AutogradEngine::active()) {
            engine->contribute(producer.get());
        } else {
            // Preserve the legacy recursive path without re-accumulating the
            // contribution through apply() while an engine is active.
            producer->apply(producer->pending_gradient("AddGradFn::apply producer"),
                            stream, backward_payload, backward_bindings);
        }
    };
    notify(a_grad_fn);
    if (b_grad_fn != a_grad_fn) notify(b_grad_fn);
}

void AddGradFn::release_saved() {
    GradFn::release_saved();
    leaf_grad_a = nullptr;
    leaf_grad_b = nullptr;
    a_grad_fn.reset();
    b_grad_fn.reset();
}

Tensor add(const Tensor& a, const Tensor& b, cudaStream_t stream) {
    if (a.numel() != b.numel()) {
        throw std::invalid_argument("autograd::add: tensor size mismatch");
    }

    Tensor result = Tensor::empty(a.shape, a.requires_grad || b.requires_grad, stream, "add_result");

    // c = a + b — use TensorContract::add for the forward
    TensorContract::add(a, b, result, stream);

    if (result.requires_grad) {
        result.is_leaf = false;
        auto grad_fn = std::make_shared<AddGradFn>();
        grad_fn->capture_inputs(const_cast<Tensor&>(a), const_cast<Tensor&>(b), stream);
        result.grad_fn = grad_fn;
    }

    return result;
}

}  // namespace GRIM::autograd
