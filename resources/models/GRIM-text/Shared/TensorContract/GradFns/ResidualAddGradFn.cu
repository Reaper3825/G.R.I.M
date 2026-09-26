//======================================================//
//  ResidualAddGradFn.cu
//  Residual add delivers gradients directly to producer accumulators or leaf buffers.
//======================================================//

#include "ResidualAddGradFn.hpp"
#include "../AutogradEngine.hpp"
#include "../GradientAccumulation.hpp"

#include <stdexcept>

namespace GRIM::autograd {
namespace {

// Leaf storage remains owned by the input tensor. Non-leaf captures keep only
// the producer alive; its pending gradient is created on first delivery.
void captureResidualInput(Tensor& input, std::shared_ptr<GradFn>& producer,
                     Tensor*& leaf_gradient, cudaStream_t stream) {
    if (!stream) throw std::runtime_error("ResidualAddGradFn::capture: stream is NULL");
    input.require("ResidualAddGradFn::capture");
    producer.reset();
    leaf_gradient = nullptr;
    if (!input.requires_grad) return;
    if (input.is_leaf) {
        if (input.grad_fn) throw std::runtime_error("ResidualAddGradFn::capture: leaf has a producer");
        input.ensure_grad();
        leaf_gradient = input.grad_.get();
        if (!leaf_gradient) throw std::runtime_error("ResidualAddGradFn::capture: missing leaf gradient");
    } else {
        if (!input.grad_fn) throw std::runtime_error("ResidualAddGradFn::capture: non-leaf has no producer");
        producer = input.grad_fn;
    }
}

} // namespace

ResidualAddGradFn::ResidualAddGradFn() { op_name = "residual_add"; }

void ResidualAddGradFn::capture_inputs(Tensor& a, Tensor& b, cudaStream_t stream) {
    input_requires_grad = a.requires_grad;
    residual_requires_grad = b.requires_grad;
    input_shape = a.shape;
    residual_shape = b.shape;
    element_count = a.numel();
    captureResidualInput(a, input_grad_fn, leaf_input_gradient, stream);
    captureResidualInput(b, residual_grad_fn, leaf_residual_gradient, stream);
    register_input(input_grad_fn);
    register_input(residual_grad_fn);
}

void ResidualAddGradFn::apply_impl(const Tensor& grad_output,
                         cudaStream_t stream,
                         const Batching::BatchPayload* backward_payload,
                         const Batching::BatchDeviceBindings* backward_bindings) {
    setCurrentGradFnOp("residual_add", this);
    if (applied) return;
    grad_output.require("ResidualAddGradFn::apply grad_output");
    if (!stream || grad_output.numel() != element_count) {
        throw std::runtime_error("ResidualAddGradFn::apply: invalid stream or gradient size");
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
            if (!leaf) throw std::runtime_error("ResidualAddGradFn::apply: missing leaf destination");
            accumulate_grad(*leaf, contribution, 1.0f, stream, "ResidualAddGradFn::apply leaf");
            leaf->record_leaf_gradient_delivery();
        }
    };

    // Both mathematical inputs contribute, including residual_add(x, x). Finish both
    // writes before issuing one notification per deduplicated producer edge.
    deliver(input_requires_grad, input_shape, input_grad_fn, leaf_input_gradient);
    deliver(residual_requires_grad, residual_shape, residual_grad_fn, leaf_residual_gradient);
    auto notify = [&](const std::shared_ptr<GradFn>& producer) {
        if (!producer) return;
        if (auto* engine = AutogradEngine::active()) {
            engine->contribute(producer.get());
        } else {
            // Preserve the legacy recursive path without re-accumulating the
            // contribution through apply() while an engine is active.
            producer->apply(producer->pending_gradient("ResidualAddGradFn::apply producer"),
                            stream, backward_payload, backward_bindings);
        }
    };
    notify(input_grad_fn);
    if (residual_grad_fn != input_grad_fn) notify(residual_grad_fn);
}

void ResidualAddGradFn::release_saved() {
    GradFn::release_saved();
    leaf_input_gradient = nullptr;
    leaf_residual_gradient = nullptr;
    input_grad_fn.reset();
    residual_grad_fn.reset();
}

Tensor residual_add(const Tensor& a, const Tensor& b, cudaStream_t stream) {
    if (a.numel() != b.numel()) {
        throw std::invalid_argument("autograd::residual_add: tensor size mismatch");
    }

    Tensor result = Tensor::empty(a.shape, a.requires_grad || b.requires_grad, stream, "residual_add_result");

    // c = a + b — use TensorContract::add for the forward
    TensorContract::add(a, b, result, stream);

    if (result.requires_grad) {
        result.is_leaf = false;
        auto grad_fn = std::make_shared<ResidualAddGradFn>();
        grad_fn->capture_inputs(const_cast<Tensor&>(a), const_cast<Tensor&>(b), stream);
        result.grad_fn = grad_fn;
    }

    return result;
}

}  // namespace GRIM::autograd
