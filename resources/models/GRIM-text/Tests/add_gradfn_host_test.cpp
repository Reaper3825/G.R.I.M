// CPU test boundary for the real AddGradFn and AutogradEngine source, assembled
// by test_add_gradfn_host.py. CUDA math/storage is replaced; scheduling is not.
#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <deque>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>
using cudaStream_t = void*;
using cudaError_t = int;
constexpr int cudaSuccess = 0;
inline int cudaStreamSynchronize(cudaStream_t) { return 0; }
inline const char* cudaGetErrorString(int) { return "fake CUDA"; }

namespace GRIM {
namespace Batching { struct BatchPayload {}; struct BatchDeviceBindings {}; }
namespace VerboseLogging { constexpr bool ENABLE_GRADFLOW_LOGS = false; }
namespace TensorContract { struct TensorShape { size_t count = 0; }; }
namespace MemoryAccounting {
enum class Kind { EngineGradient };
inline void classify(void*, Kind) {}
}
struct GradFn;
inline size_t allocations = 0;
struct Tensor {
    float* data = nullptr;
    TensorContract::TensorShape shape;
    bool requires_grad = false, is_leaf = true, owns_data = false;
    cudaStream_t stream = nullptr;
    std::shared_ptr<GradFn> grad_fn;
    std::shared_ptr<Tensor> grad_;
    std::shared_ptr<std::vector<float>> storage;
    int deliveries = 0;
    size_t numel() const { return shape.count; }
    const Tensor& require(const char*) const {
        if (!data) throw std::runtime_error("missing test data");
        return *this;
    }
    static Tensor zeros(TensorContract::TensorShape s, bool grad, cudaStream_t stream, const char*) {
        ++allocations;
        Tensor t; t.shape = s; t.requires_grad = grad; t.stream = stream;
        t.storage = std::make_shared<std::vector<float>>(s.count, 0.0f);
        t.data = t.storage->data(); t.owns_data = true; return t;
    }
    static Tensor empty(TensorContract::TensorShape s, bool grad, cudaStream_t stream, const char* name) {
        return zeros(s, grad, stream, name);
    }
    void ensure_grad() {
        if (!grad_) grad_ = std::make_shared<Tensor>(zeros(shape, false, stream, "leaf"));
    }
    void record_leaf_gradient_delivery() { ++deliveries; }
};
namespace TensorContract {
inline void add(const Tensor& a, const Tensor& b, Tensor& out, cudaStream_t) {
    for (size_t i = 0; i < a.numel(); ++i) out.data[i] = a.data[i] + b.data[i];
}
}
namespace autograd {
inline void accumulate_grad(Tensor& dst, const Tensor& src, float scale, cudaStream_t, const char*) {
    if (dst.numel() != src.numel()) throw std::runtime_error("shape mismatch");
    for (size_t i = 0; i < dst.numel(); ++i) dst.data[i] += scale * src.data[i];
}
}
struct GradFn {
    const char* op_name = "test";
    bool applied = false;
    std::vector<GradFn*> engine_inputs_;
    std::shared_ptr<Tensor> pending_gradient_;
    virtual ~GradFn() = default;
    void register_input(const std::shared_ptr<GradFn>& producer) {
        if (producer && std::find(engine_inputs_.begin(), engine_inputs_.end(), producer.get()) == engine_inputs_.end())
            engine_inputs_.push_back(producer.get());
    }
    virtual void collect_input_edges(std::vector<GradFn*>& out) const { out = engine_inputs_; }
    void receive_gradient(const Tensor&, cudaStream_t);
    const Tensor& pending_gradient(const char*) const;
    void apply(const Tensor&, cudaStream_t, const Batching::BatchPayload*, const Batching::BatchDeviceBindings*);
    void run_backward(cudaStream_t, const Batching::BatchPayload*, const Batching::BatchDeviceBindings*);
    virtual void apply_impl(const Tensor&, cudaStream_t, const Batching::BatchPayload*, const Batching::BatchDeviceBindings*) = 0;
    virtual void release_saved() { pending_gradient_.reset(); engine_inputs_.clear(); }
};
inline void setCurrentGradFnOp(const char*, GradFn*) {}
inline void logTensorContractApplyGradOutputStats(GradFn&, const Tensor&, cudaStream_t) {}
} // namespace GRIM

// REAL_IMPLEMENTATIONS

namespace {
using namespace GRIM;
cudaStream_t stream = reinterpret_cast<void*>(1);
Tensor tensor(bool grad = true) { return Tensor::zeros({3}, grad, stream, "input"); }
struct Identity : GradFn {
    Tensor* leaf;
    int calls = 0;
    explicit Identity(Tensor& input) : leaf(input.grad_.get()) {}
    void apply_impl(const Tensor& incoming, cudaStream_t s, const Batching::BatchPayload*, const Batching::BatchDeviceBindings*) override {
        ++calls;
        autograd::accumulate_grad(*leaf, incoming, 1, s, "identity");
    }
};
Tensor intermediate(Tensor& leaf, std::shared_ptr<Identity>& node) {
    leaf.ensure_grad(); node = std::make_shared<Identity>(leaf);
    Tensor t = tensor(); t.is_leaf = false; t.grad_fn = node; return t;
}
void backward(Tensor& output, bool engine = true) {
    Tensor seed = tensor(false);
    seed.data[0] = 1; seed.data[1] = 2; seed.data[2] = 3;
    if (engine) {
        output.grad_fn->receive_gradient(seed, stream);
        autograd::AutogradEngine scheduler(stream, nullptr, nullptr);
        scheduler.run(output.grad_fn.get());
    } else output.grad_fn->apply(seed, stream, nullptr, nullptr);
}
void expect(const Tensor& leaf, float multiple) {
    for (size_t i = 0; i < 3; ++i) assert(leaf.grad_->data[i] == multiple * float(i + 1));
}
}
int main() {
    using namespace GRIM;
    // Distinct producers: capture allocates only the forward result.
    for (bool engine : {true, false}) {
        Tensor la = tensor(), lb = tensor();
        std::shared_ptr<Identity> na, nb;
        Tensor a = intermediate(la, na), b = intermediate(lb, nb);
        const auto before = allocations;
        Tensor out = autograd::add(a, b, stream);
        assert(allocations == before + 1);
        assert(!na->pending_gradient_ && !nb->pending_gradient_);
        backward(out, engine);
        expect(la, 1); expect(lb, 1);
        assert(na->calls == 1 && nb->calls == 1);
        assert(na->pending_gradient_->data != nb->pending_gradient_->data);
    }
    // Repeated intermediate: two additions but exactly one scheduler edge.
    for (bool engine : {true, false}) {
        Tensor leaf = tensor(); std::shared_ptr<Identity> node;
        Tensor x = intermediate(leaf, node);
        Tensor out = autograd::add(x, x, stream);
        backward(out, engine); expect(leaf, 2); assert(node->calls == 1);
    }
    // Diamond: a shared producer must wait for both different Add consumers.
    {
        Tensor leaf = tensor(); std::shared_ptr<Identity> node;
        Tensor x = intermediate(leaf, node), constant = tensor(false);
        Tensor a = autograd::add(x, constant, stream);
        Tensor b = autograd::add(x, x, stream);
        Tensor out = autograd::add(a, b, stream);
        backward(out); expect(leaf, 3); assert(node->calls == 1);
    }
    // Leaf identity and accumulation across successive microbatch graphs.
    {
        Tensor leaf = tensor();
        for (int i = 1; i <= 2; ++i) {
            Tensor out = autograd::add(leaf, leaf, stream);
            backward(out); expect(leaf, float(2 * i));
        }
        assert(leaf.grad_->deliveries == 4);
    }
    // Mixed leaf/producer and single-input capture.
    {
        Tensor la = tensor(), lb = tensor(); std::shared_ptr<Identity> node;
        Tensor x = intermediate(la, node);
        Tensor out = autograd::add(x, lb, stream);
        backward(out); expect(la, 1); expect(lb, 1);
    }
    {
        Tensor leaf = tensor(); std::shared_ptr<Identity> node;
        Tensor x = intermediate(leaf, node), out = tensor();
        auto add = std::make_shared<autograd::AddGradFn>();
        auto before = allocations;
        add->capture_single_input(x, stream);
        assert(allocations == before);
        out.grad_fn = add; backward(out); expect(leaf, 1);
        add->release_saved();
        assert(!add->a_grad_fn && !add->leaf_grad_a && !add->pending_gradient_);
    }
    std::cout << "AddGradFn host routing tests passed\n";
}
