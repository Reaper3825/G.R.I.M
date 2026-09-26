namespace {
using namespace GRIM;
using namespace GRIM::autograd;
cudaStream_t stream = reinterpret_cast<void*>(1);
auto flat_shape = TensorContract::TensorShape::make_BSM(4, 4);
auto head_shape = TensorContract::TensorShape::make_BHSD(2, 2, 2, 2);
Tensor flat() {
    auto t = Tensor::zeros(flat_shape, false, stream, "seed");
    for (int i = 0; i < 16; ++i) t.data[i] = float(i + 1);
    return t;
}
Tensor heads() { return Tensor::zeros(head_shape, true, stream, "input"); }
void expect(const float* values, float base, float scale) {
    for (int b = 0; b < 2; ++b) for (int h = 0; h < 2; ++h)
    for (int s = 0; s < 2; ++s) for (int d = 0; d < 2; ++d) {
        int dst = ((b * 2 + h) * 2 + s) * 2 + d;
        int src = ((b * 2 + s) * 2 + h) * 2 + d;
        assert(values[dst] == base + scale * (src + 1));
    }
}
struct Sink : GradFn {
    int calls = 0;
    std::vector<float> received;
    void apply_impl(const Tensor& incoming, cudaStream_t,
                    const Batching::BatchPayload*, const Batching::BatchDeviceBindings*) override {
        ++calls; received.assign(incoming.data, incoming.data + incoming.numel());
    }
};
Tensor reshape(Tensor& input) {
    auto out = Tensor::zeros(flat_shape, true, stream, "out");
    auto n = std::make_shared<ReshapeFromBHSDGradFn>();
    const auto before = allocations; n->capture_input(input, stream);
    if (!input.is_leaf) assert(allocations == before);
    n->batch_size = n->seq_len = n->num_heads = n->head_dim = 2;
    out.is_leaf = false; out.grad_fn = n; return out;
}
void backward(Tensor& out, bool engine) {
    Tensor seed = flat();
    if (engine) {
        out.grad_fn->receive_gradient(seed, stream);
        AutogradEngine scheduler(stream, nullptr, nullptr); scheduler.run(out.grad_fn.get());
    } else out.grad_fn->apply(seed, stream, nullptr, nullptr);
}
}
int main() {
    // Real geometry kernel: overwrite remains available; accumulation preserves data.
    {
        Tensor src = flat(), dst = heads(); std::fill_n(dst.data, 16, 7.0f);
        TensorConversion::convert_BSM_to_BHSD(src.data, dst.data, 2, 2, 2, 2, stream, false);
        expect(dst.data, 0, 1);
        TensorConversion::convert_BSM_to_BHSD(src.data, dst.data, 2, 2, 2, 2, stream, true);
        expect(dst.data, 0, 2);
        for (int i = 0; i < 16; ++i) assert(src.data[i] == i + 1);
    }
    for (bool engine : {false, true}) {
        Tensor leaf = heads(); leaf.ensure_grad(); std::fill_n(leaf.grad_->data, 16, 5.0f);
        std::fill_n(leaf.data, 16, 9.0f);
        for (int pass = 1; pass <= 2; ++pass) {
            Tensor out = reshape(leaf); backward(out, engine);
            expect(leaf.grad_->data, 5, float(pass)); assert(leaf.grad_->deliveries == pass);
            backward(out, false); expect(leaf.grad_->data, 5, float(pass));
        }
        for (int i = 0; i < 16; ++i) assert(leaf.data[i] == 9);
    }
    for (bool engine : {false, true}) {
        Tensor x = heads(); auto sink = std::make_shared<Sink>(); x.is_leaf = false; x.grad_fn = sink;
        Tensor out = reshape(x); assert(!sink->pending_gradient_ && !x.grad_);
        backward(out, engine); assert(sink->calls == 1);
        expect(sink->received.data(), 0, 1); expect(sink->pending_gradient_->data, 0, 1);
        auto n = std::static_pointer_cast<ReshapeFromBHSDGradFn>(out.grad_fn);
        n->release_saved(); assert(!n->input_producer && !n->leaf_input_gradient && !n->pending_gradient_);
        assert(n->engine_inputs_.empty());
    }
    {
        // Two consumers accumulate into the SAME BHSD producer, then notify once each.
        Tensor x = heads(); auto sink = std::make_shared<Sink>(); x.is_leaf = false; x.grad_fn = sink;
        Tensor a = reshape(x), b = reshape(x), out = add(a, b, stream);
        std::fill_n(sink->gradient_destination(head_shape, stream).data, 16, 5.0f);
        backward(out, true); assert(sink->calls == 1); expect(sink->received.data(), 5, 2);
    }
    {
        Tensor x = heads(); x.requires_grad = false;
        Tensor out = reshape(x); backward(out, true); assert(!x.grad_);
    }
    {
        // Same element count but wrong flat geometry must fail before destination allocation.
        Tensor x = heads(); auto sink = std::make_shared<Sink>(); x.is_leaf = false; x.grad_fn = sink;
        Tensor out = reshape(x);
        Tensor bad = Tensor::zeros(TensorContract::TensorShape::make_BSM(2, 8), false, stream, "bad");
        bool rejected = false;
        try { out.grad_fn->apply(bad, stream, nullptr, nullptr); }
        catch (const std::runtime_error&) { rejected = true; }
        assert(rejected && !sink->pending_gradient_ && sink->calls == 0 && !out.grad_fn->applied);
    }
    std::cout << "Reshape GradFn host routing and geometry tests passed\n";
}
