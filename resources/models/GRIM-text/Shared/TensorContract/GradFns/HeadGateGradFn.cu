#include "HeadGateGradFn.hpp"
#include "../AutogradEngine.hpp"
#include "../../TensorConversion/TensorConversion.hpp"
#include <limits>
#include <stdexcept>

void trackKernelLaunch(const char*, cudaStream_t);

namespace GRIM {

namespace autograd {

HeadGateGradFn::HeadGateGradFn() {
    op_name = "head_gate_bhsd_to_flat";
}

void HeadGateGradFn::capture_inputs(Tensor& s, Tensor& x, cudaStream_t stream) {
    if (!stream) throw std::runtime_error("HeadGateGradFn::capture: stream is NULL");
    scale_requires_grad = s.requires_grad;
    x_requires_grad = x.requires_grad;
    scale_shape = s.shape;
    x_shape = x.shape;
    auto capture = [&](Tensor& x, std::shared_ptr<GradFn>& producer, Tensor*& leaf) {
        x.require("HeadGateGradFn::capture");
        producer.reset();
        leaf = nullptr;
        if (!x.requires_grad) return;
        if (x.is_leaf) {
            if (x.grad_fn) throw std::runtime_error("HeadGateGradFn::capture: leaf has a producer");
            x.ensure_grad();
            leaf = x.grad_.get();
            if (!leaf) throw std::runtime_error("HeadGateGradFn::capture: missing leaf gradient");
        } else {
            if (!x.grad_fn) throw std::runtime_error("HeadGateGradFn::capture: non-leaf has no producer");
            producer = x.grad_fn;
            register_input(producer);
        }
    };
    capture(s, scale_grad_fn, leaf_scale_gradient);
    capture(x, x_grad_fn, leaf_x_gradient);
}

void HeadGateGradFn::set_cache_refs(const float* scale_data, const float* x_data, int b, int h, int s, int d) {
    if (!scale_data) throw std::runtime_error("HeadGateGradFn::set_cache_refs: scale_data is NULL");
    if (!x_data) throw std::runtime_error("HeadGateGradFn::set_cache_refs: x_data is NULL");
    cached_scale = scale_data;
    cached_x = x_data;
    batch = b; heads = h; seq = s; head_dim = d;
}

void HeadGateGradFn::apply_impl(const Tensor& grad_output,
                                       cudaStream_t stream,
                                       const Batching::BatchPayload* backward_payload,
                                       const Batching::BatchDeviceBindings* backward_bindings) {
    setCurrentGradFnOp("head_gate_bhsd_to_flat", this);
    if (applied) return;
    if (!stream) throw std::runtime_error("HeadGateGradFn::apply: stream is NULL");
    grad_output.require("HeadGateGradFn::apply grad_output");
    const size_t tokens = static_cast<size_t>(batch) * seq;
    if (batch <= 0 || heads <= 0 || seq <= 0 || head_dim <= 0 || !cached_scale || !cached_x ||
        scale_shape.total_elements() != tokens * heads ||
        x_shape.total_elements() != tokens * heads * head_dim ||
        !grad_output.shape.is_2d_layout() ||
        grad_output.shape.as_2d().rows != tokens ||
        grad_output.shape.as_2d().cols != static_cast<size_t>(heads) * head_dim) {
        throw std::runtime_error("HeadGateGradFn::apply: invalid cache or gradient geometry");
    }
    auto destination = [&](bool required, const std::shared_ptr<GradFn>& producer,
                           Tensor* leaf, const TensorContract::TensorShape& shape) -> Tensor* {
        if (!required) return nullptr;
        if (producer) return &producer->gradient_destination(shape, stream);
        if (!leaf) throw std::runtime_error("HeadGateGradFn::apply: missing leaf destination");
        leaf->require("HeadGateGradFn::apply leaf");
        return leaf;
    };
    Tensor* x_gradient = destination(x_requires_grad, x_grad_fn, leaf_x_gradient, x_shape);
    Tensor* scale_gradient = destination(scale_requires_grad, scale_grad_fn, leaf_scale_gradient, scale_shape);
    applied = true;
    TensorConversion::head_gate_BHSD_to_BSM_backward(
        grad_output.data, cached_x, cached_scale,
        x_gradient ? x_gradient->data : nullptr,
        scale_gradient ? scale_gradient->data : nullptr,
        batch, heads, seq, head_dim, stream);
    trackKernelLaunch("head_gate_BHSD_to_BSM_backward", stream);

    if (x_requires_grad && !x_grad_fn) leaf_x_gradient->record_leaf_gradient_delivery();
    if (scale_requires_grad && !scale_grad_fn) leaf_scale_gradient->record_leaf_gradient_delivery();
    // Complete both writes before notifying each deduplicated producer edge.
    auto notify = [&](const std::shared_ptr<GradFn>& producer) {
        if (!producer) return;
        if (auto* engine = AutogradEngine::active()) {
            engine->contribute(producer.get());
        } else {
            producer->apply(producer->pending_gradient("HeadGateGradFn::apply producer"),
                            stream, backward_payload, backward_bindings);
        }
    };
    notify(x_grad_fn);
    if (scale_grad_fn != x_grad_fn) notify(scale_grad_fn);
}

void HeadGateGradFn::release_saved() {
    GradFn::release_saved();
    cached_scale = nullptr;
    cached_x = nullptr;
    scale_grad_fn.reset();
    x_grad_fn.reset();
    leaf_scale_gradient = nullptr;
    leaf_x_gradient = nullptr;
}

Tensor head_gate_bhsd_to_flat(const Tensor& scale, const Tensor& x, cudaStream_t stream) {
    if (!stream) throw std::runtime_error("head_gate_bhsd_to_flat: stream is NULL");
    scale.require("head_gate_bhsd_to_flat scale");
    x.require("head_gate_bhsd_to_flat attention");
    if (x.shape.layout != TensorContract::Layout::BHSD || !scale.shape.is_2d_layout())
        throw std::runtime_error("head_gate_bhsd_to_flat: expected BHSD attention and [tokens,heads] gates");
    const auto dims = x.shape.as_4d();
    const auto gates = scale.shape.as_2d();
    const size_t tokens = static_cast<size_t>(dims.batch) * dims.seq;
    const size_t width = static_cast<size_t>(dims.heads) * dims.head_dim;
    if (dims.batch <= 0 || dims.heads <= 0 || dims.seq <= 0 || dims.head_dim <= 0 ||
        tokens > static_cast<size_t>(std::numeric_limits<int>::max()) ||
        width > static_cast<size_t>(std::numeric_limits<int>::max()) ||
        gates.rows != tokens || gates.cols != dims.heads)
        throw std::runtime_error("head_gate_bhsd_to_flat: gate geometry must match query heads");
    const bool needs_grad = scale.requires_grad || x.requires_grad;
    Tensor result = Tensor::empty(TensorContract::TensorShape::make_BSM(
        static_cast<int>(tokens), static_cast<int>(width)), needs_grad, stream, "head_gate_flat_result");
    TensorConversion::head_gate_BHSD_to_BSM(x.data, scale.data, result.data,
        dims.batch, dims.heads, dims.seq, dims.head_dim, stream);
    trackKernelLaunch("head_gate_BHSD_to_BSM", stream);
    if (needs_grad) {
        result.is_leaf = false;
        auto node = std::make_shared<HeadGateGradFn>();
        node->capture_inputs(const_cast<Tensor&>(scale), const_cast<Tensor&>(x), stream);
        node->set_cache_refs(scale.data, x.data, dims.batch, dims.heads, dims.seq, dims.head_dim);
        result.grad_fn = node;
    }
    return result;
}

} // namespace autograd
} // namespace GRIM
