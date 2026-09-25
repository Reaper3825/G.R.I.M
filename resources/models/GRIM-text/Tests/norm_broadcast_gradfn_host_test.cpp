// Production routing runs against CPU reference kernels in the Python runner.
namespace {
using namespace GRIM::autograd;
void capture(BroadcastRowMulGradFn& n, Tensor& a, Tensor& b) {
    n.capture_inputs(a, b, stream); n.set_cache_refs(a.data, b.data, 3, 1);
}
void capture(RMSNormGradFn& n, Tensor& a, Tensor& b) {
    n.capture_inputs(a, b, stream);
    // Stand in for the saved forward cache; no CUDA allocation in this host test.
    n.cached_input = a.data; n.cached_size = 3; n.d_model = 3; n.eps = 0;
}
float expected(BroadcastRowMulGradFn*, bool, int i) { return float(i + 1); }
float expected(RMSNormGradFn*, bool first, int i) { return first ? float(i - 1) : float(i + 1); }
void released(BroadcastRowMulGradFn& n) {
    assert(!n.scale_grad_fn && !n.x_grad_fn && !n.leaf_scale_gradient && !n.leaf_x_gradient);
    assert(!n.cached_scale && !n.cached_x);
}
void released(RMSNormGradFn& n) {
    assert(!n.input_grad_fn && !n.gamma_grad_fn && !n.leaf_input_gradient && !n.leaf_gamma_gradient);
    assert(!n.cached_input && !n.gamma_data);
}
void near(float actual, float wanted) { assert(std::abs(actual - wanted) < 1e-5f); }
template<class Node> Tensor operation(Tensor& a, Tensor& b) {
    Tensor out = tensor(); auto n = std::make_shared<Node>();
    const auto before = allocations; capture(*n, a, b);
    // Caller prepares leaf buffers; capture must allocate no gradient storage.
    assert(allocations == before);
    out.grad_fn = n; out.is_leaf = false; return out;
}
template<class Node> void routing() {
    for (bool engine : {false, true})
    for (bool ar : {false, true}) for (bool br : {false, true})
    for (bool ap : {false, true}) for (bool bp : {false, true}) {
        Tensor la = tensor(), lb = tensor(); std::shared_ptr<Identity> pa, pb;
        la.ensure_grad(); lb.ensure_grad();
        Tensor a = ap ? intermediate(la, pa) : la;
        Tensor b = bp ? intermediate(lb, pb) : lb;
        a.requires_grad = ar; b.requires_grad = br;
        std::fill_n(a.data, 3, 1.0f); std::fill_n(b.data, 3, 1.0f);
        std::fill_n(la.grad_->data, 3, 5.0f); std::fill_n(lb.grad_->data, 3, 5.0f);
        Tensor out = operation<Node>(a, b);
        if (pa) assert(!pa->pending_gradient_);
        if (pb) assert(!pb->pending_gradient_);
        destinations.clear(); backward(out, engine);
        assert(destinations.size() == size_t(ar) + size_t(br));
        for (int i = 0; i < 3; ++i) {
            near(la.grad_->data[i], 5 + (ar ? expected((Node*)nullptr, true, i) : 0));
            near(lb.grad_->data[i], 5 + (br ? expected((Node*)nullptr, false, i) : 0));
            assert(a.data[i] == 1 && b.data[i] == 1);
        }
        if (pa) assert(pa->calls == int(ar)); else assert(la.grad_->deliveries == int(ar));
        if (pb) assert(pb->calls == int(br)); else assert(lb.grad_->deliveries == int(br));
        if (ar) {
            float* dst = ap ? pa->pending_gradient_->data : la.grad_->data;
            assert(std::find(destinations.begin(), destinations.end(), dst) != destinations.end());
        }
        if (br) {
            float* dst = bp ? pb->pending_gradient_->data : lb.grad_->data;
            assert(std::find(destinations.begin(), destinations.end(), dst) != destinations.end());
        }
        auto writes = destinations.size(); backward(out, false); assert(destinations.size() == writes);
        auto n = std::static_pointer_cast<Node>(out.grad_fn); n->release_saved();
        released(*n); assert(!n->pending_gradient_ && n->engine_inputs_.empty());
    }
    for (bool engine : {false, true}) for (bool producer : {false, true}) {
        Tensor leaf = tensor(); leaf.ensure_grad(); std::shared_ptr<Identity> p;
        Tensor x = producer ? intermediate(leaf, p) : leaf;
        std::fill_n(x.data, 3, 1.0f);
        Tensor out = operation<Node>(x, x); destinations.clear(); backward(out, engine);
        assert(destinations.size() == 2 && destinations[0] == destinations[1]);
        for (int i = 0; i < 3; ++i)
            near(leaf.grad_->data[i], expected((Node*)nullptr, true, i) + expected((Node*)nullptr, false, i));
        if (producer) assert(p->calls == 1); else assert(leaf.grad_->deliveries == 2);
    }
    {
        Tensor leaf = tensor(); std::shared_ptr<Identity> p;
        Tensor x = intermediate(leaf, p); std::fill_n(x.data, 3, 1.0f);
        Tensor a = operation<Node>(x, x), b = operation<Node>(x, x);
        Tensor root = add(a, b, stream); backward(root); assert(p->calls == 1);
        for (int i = 0; i < 3; ++i)
            near(leaf.grad_->data[i], 2 * (expected((Node*)nullptr, true, i) + expected((Node*)nullptr, false, i)));
    }
    {
        Tensor a = tensor(), b = tensor(); a.ensure_grad(); b.ensure_grad();
        std::fill_n(a.data, 3, 1.0f); std::fill_n(b.data, 3, 1.0f);
        for (int pass = 1; pass <= 2; ++pass) {
            Tensor out = operation<Node>(a, b); backward(out);
            for (int i = 0; i < 3; ++i) {
                near(a.grad_->data[i], pass * expected((Node*)nullptr, true, i));
                near(b.grad_->data[i], pass * expected((Node*)nullptr, false, i));
            }
            assert(a.grad_->deliveries == pass && b.grad_->deliveries == pass);
        }
    }
    {
        Tensor a = tensor(), b = tensor(); a.ensure_grad(); b.ensure_grad();
        Tensor out = operation<Node>(a, b);
        Tensor bad = Tensor::zeros({2}, false, stream, "bad"); bool rejected = false;
        destinations.clear();
        try { out.grad_fn->apply(bad, stream, nullptr, nullptr); }
        catch (const std::runtime_error&) { rejected = true; }
        assert(rejected && destinations.empty() && !out.grad_fn->applied);
    }
}
void reductions() {
    // Nontrivial two-row broadcast reduction, both destinations non-leaf.
    Tensor ls = Tensor::zeros({2}, true, stream, "scale"); ls.ensure_grad();
    Tensor lx = Tensor::zeros({6}, true, stream, "x"); lx.ensure_grad();
    auto ps = std::make_shared<Identity>(ls), px = std::make_shared<Identity>(lx);
    Tensor s = ls, x = lx; s.is_leaf = x.is_leaf = false; s.grad_fn = ps; x.grad_fn = px;
    s.data[0] = 2; s.data[1] = 3; std::fill_n(x.data, 6, 1.0f);
    auto n = std::make_shared<BroadcastRowMulGradFn>(); n->capture_inputs(s, x, stream);
    n->set_cache_refs(s.data, x.data, 2, 3);
    Tensor seed = Tensor::zeros({6}, false, stream, "seed");
    for (int i = 0; i < 6; ++i) seed.data[i] = float(i + 1);
    n->receive_gradient(seed, stream);
    { AutogradEngine engine(stream, nullptr, nullptr); engine.run(n.get()); }
    near(ls.grad_->data[0], 6); near(ls.grad_->data[1], 15);
    for (int i = 0; i < 6; ++i) near(lx.grad_->data[i], (i + 1) * (i < 3 ? 2.0f : 3.0f));
    assert(ps->calls == 1 && px->calls == 1);

    // RMSNorm gamma reduction across two rows, including a non-leaf gamma.
    Tensor lg = tensor(); lg.ensure_grad(); auto pg = std::make_shared<Identity>(lg);
    Tensor gamma = lg; gamma.is_leaf = false; gamma.grad_fn = pg;
    std::fill_n(gamma.data, 3, 1.0f);
    Tensor input = Tensor::zeros({6}, true, stream, "input"); std::fill_n(input.data, 6, 1.0f);
    auto rms = std::make_shared<RMSNormGradFn>(); rms->capture_inputs(input, gamma, stream);
    rms->cached_input = input.data; rms->cached_size = 6; rms->d_model = 3; rms->eps = 0;
    rms->receive_gradient(seed, stream);
    { AutogradEngine engine(stream, nullptr, nullptr); engine.run(rms.get()); }
    for (int i = 0; i < 6; ++i) near(input.grad_->data[i], float(i % 3 - 1));
    near(lg.grad_->data[0], 5); near(lg.grad_->data[1], 7); near(lg.grad_->data[2], 9);
    assert(pg->calls == 1 && input.grad_->deliveries == 1);
}
}
int main() {
    routing<BroadcastRowMulGradFn>(); routing<RMSNormGradFn>(); reductions();
    std::cout << "Batch 3 norm/broadcast GradFn host routing tests passed\n";
}
