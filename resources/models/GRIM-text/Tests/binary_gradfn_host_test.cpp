// Appended to the existing host boundary by test_binary_gradfn_host.py.
namespace {
using namespace GRIM::autograd;
void capture(ElementwiseMulGradFn& n, Tensor& a, Tensor& b) { n.capture_inputs(a, b, stream); }
void capture(BiasAddGradFn& n, Tensor& a, Tensor& b) { n.capture_inputs(a, b, 1, 3, stream); }
void released(ElementwiseMulGradFn& n) {
    assert(!n.a_grad_fn && !n.b_grad_fn && !n.leaf_grad_a && !n.leaf_grad_b);
    assert(!n.cached_a && !n.cached_b);
}
void released(BiasAddGradFn& n) {
    assert(!n.input_grad_fn && !n.bias_grad_fn && !n.leaf_grad_input && !n.leaf_grad_bias);
}
template<class Node> Tensor binary(Tensor& a, Tensor& b) {
    Tensor out = tensor(); auto n = std::make_shared<Node>();
    const auto before = allocations;
    capture(*n, a, b);
    if (!a.is_leaf && !b.is_leaf) assert(before == allocations);
    out.grad_fn = n; out.is_leaf = false; return out;
}
template<class Node> void routing(float factor) {
    // Each input requires gradients independently; frozen inputs have no delivery.
    for (bool ar : {false, true}) for (bool br : {false, true}) {
        Tensor a = tensor(ar), b = tensor(br);
        std::fill_n(a.data, 3, 2.0f); std::fill_n(b.data, 3, 2.0f);
        if (ar) a.ensure_grad();
        if (br) b.ensure_grad();
        for (int pass = 1; pass <= 2; ++pass) {
            Tensor out = binary<Node>(a, b); backward(out);
            if (ar) { expect(a, factor * pass); assert(a.grad_->deliveries == pass); }
            else assert(!a.grad_);
            if (br) { expect(b, factor * pass); assert(b.grad_->deliveries == pass); }
            else assert(!b.grad_);
        }
        for (int i = 0; i < 3; ++i) assert(a.data[i] == 2 && b.data[i] == 2);
    }
    for (bool engine : {true, false}) {
        // Distinct producers receive their own destination, with no private capture allocation.
        Tensor la = tensor(), lb = tensor(); std::shared_ptr<Identity> pa, pb;
        Tensor a = intermediate(la, pa), b = intermediate(lb, pb);
        std::fill_n(a.data, 3, 2.0f); std::fill_n(b.data, 3, 2.0f);
        Tensor out = binary<Node>(a, b);
        assert(!pa->pending_gradient_ && !pb->pending_gradient_);
        destinations.clear(); backward(out, engine);
        expect(la, factor); expect(lb, factor);
        assert(pa->calls == 1 && pb->calls == 1);
        assert(destinations.size() == 2);
        assert(destinations[0] == pa->pending_gradient_->data);
        assert(destinations[1] == pb->pending_gradient_->data);
        backward(out, false); assert(pa->calls == 1 && pb->calls == 1);
        auto n = std::static_pointer_cast<Node>(out.grad_fn);
        n->release_saved(); released(*n); assert(!n->pending_gradient_);
    }
    for (bool engine : {true, false}) {
        // Same producer in both slots: sum BOTH terms before one notification.
        Tensor leaf = tensor(); std::shared_ptr<Identity> p;
        Tensor x = intermediate(leaf, p); std::fill_n(x.data, 3, 2.0f);
        Tensor out = binary<Node>(x, x);
        destinations.clear(); backward(out, engine);
        expect(leaf, 2 * factor); assert(p->calls == 1);
        assert(destinations.size() == 2 && destinations[0] == destinations[1]);
    }
    {
        // Repeated leaf receives both mathematical contributions.
        Tensor x = tensor(); std::fill_n(x.data, 3, 2.0f);
        Tensor out = binary<Node>(x, x); backward(out);
        expect(x, 2 * factor); assert(x.grad_->deliveries == 2);
    }
    {
        // Diamond with multiple consumers: producer waits for all contributions.
        Tensor leaf = tensor(); std::shared_ptr<Identity> p;
        Tensor x = intermediate(leaf, p); std::fill_n(x.data, 3, 2.0f);
        Tensor a = binary<Node>(x, x), b = binary<Node>(x, x);
        Tensor out = add(a, b, stream); backward(out);
        expect(leaf, 4 * factor); assert(p->calls == 1);
    }
    {
        // Mixed leaf/non-leaf destinations.
        Tensor leaf = tensor(), b = tensor(); std::shared_ptr<Identity> p;
        Tensor a = intermediate(leaf, p);
        std::fill_n(a.data, 3, 2.0f); std::fill_n(b.data, 3, 2.0f);
        Tensor out = binary<Node>(a, b); backward(out);
        expect(leaf, factor); expect(b, factor);
        assert(p->calls == 1 && b.grad_->deliveries == 1);
    }
    {
        Tensor a = tensor(), b = tensor(); Tensor out = binary<Node>(a, b);
        Tensor bad = Tensor::zeros({2}, false, stream, "bad");
        bool rejected = false;
        try { out.grad_fn->apply(bad, stream, nullptr, nullptr); }
        catch (const std::runtime_error&) { rejected = true; }
        assert(rejected && a.grad_->deliveries == 0 && b.grad_->deliveries == 0);
    }
}
void bias_reduction() {
    for (bool producer_bias : {false, true}) {
        Tensor input = Tensor::zeros({6}, true, stream, "input");
        Tensor leaf = tensor(); std::shared_ptr<Identity> p;
        Tensor bias = producer_bias ? intermediate(leaf, p) : tensor();
        auto n = std::make_shared<BiasAddGradFn>();
        n->capture_inputs(input, bias, 2, 3, stream);
        Tensor seed = Tensor::zeros({6}, false, stream, "seed");
        for (int i = 0; i < 6; ++i) seed.data[i] = float(i + 1);
        n->receive_gradient(seed, stream);
        AutogradEngine engine(stream, nullptr, nullptr); engine.run(n.get());
        for (int i = 0; i < 6; ++i) assert(input.grad_->data[i] == i + 1);
        const Tensor& dst = producer_bias ? *leaf.grad_ : *bias.grad_;
        assert(dst.data[0] == 5 && dst.data[1] == 7 && dst.data[2] == 9);
        if (producer_bias) assert(p->calls == 1);
        else assert(bias.grad_->deliveries == 1);
    }
}
}
int main() {
    routing<ElementwiseMulGradFn>(2);
    routing<BiasAddGradFn>(1);
    bias_reduction();
    std::cout << "Batch 2 binary GradFn host routing tests passed\n";
}
