"""CPU routing tests for batch 1: GELU, dropout, softmax and log-softmax.

Compile production capture/apply/release bodies and the real scheduler against
the Add test's host storage shim. Replace CUDA launches with a known 2*dy
contribution: these tests verify destinations and delivery, not CUDA math.
No training/runtime target is built or executed.
"""
from pathlib import Path
import re

from test_add_gradfn_host import compile_and_run, without_includes


def method(source, signature):
    start = source.index(signature)
    brace = source.index('{', start)
    depth = 1
    end = brace + 1
    while depth:
        depth += (source[end] == '{') - (source[end] == '}')
        end += 1
    return source[start:end]


def main():
    tests = Path(__file__).resolve().parent
    contract = tests.parent / 'Shared/TensorContract'
    fixture = (tests / 'add_gradfn_host_test.cpp').read_text(encoding='utf-8')
    boundary, helpers = fixture.split('// REAL_IMPLEMENTATIONS')
    helpers = helpers[:helpers.index('int main()')]
    boundary += r'''
#include <cstdint>
inline int cudaGetLastError() { return 0; }
inline void trackKernelLaunch(const char*, cudaStream_t) {}
inline void throwIfCudaFailed(int status, const char*) { assert(status == 0); }
namespace GRIM::MemoryAccounting { inline void free(void*) {} }
namespace GRIM::autograd {
inline bool tensorShapesEqual(const TensorContract::TensorShape& a,
                              const TensorContract::TensorShape& b) {
    return a.count == b.count && a.layout == b.layout;
}
inline float* last_destination = nullptr;
inline void contribution(const float* dy, float* dx, size_t n) {
    last_destination = dx;
    for (size_t i = 0; i < n; ++i) dx[i] += 2 * dy[i];
}
inline void kernel_gelu_backward(const float* dy, const float*, float* dx, size_t n) {
    contribution(dy, dx, n);
}
inline void kernel_dropout_backward(const float* dy, const std::uint8_t*, float* dx, float, size_t n) {
    contribution(dy, dx, n);
}
inline void kernel_softmax_backward(const float* dy, const float*, float* dx, int rows, int cols, float) {
    contribution(dy, dx, size_t(rows) * cols);
}
inline void kernel_log_softmax_backward(const float* dy, const float*, float* dx, int rows, int cols) {
    contribution(dy, dx, size_t(rows) * cols);
}
}
'''
    runtime = (contract / 'TensorContract_GPU.cu').read_text(encoding='utf-8')
    start = runtime.index('void GradFn::receive_gradient(')
    end = runtime.index('//======================================================//', start)
    parts = [boundary,
             without_includes(contract / 'AutogradEngine.hpp'),
             without_includes(contract / 'GradFns/AddGradFn.hpp'),
             without_includes(contract / 'AutogradEngine.cu'),
             'namespace GRIM {\n' + runtime[start:end] + '\n}',
             without_includes(contract / 'GradFns/AddGradFn.cu')]
    names = ('Gelu', 'Dropout', 'Softmax', 'LogSoftmax')
    for name in names:
        cls = name + 'GradFn'
        parts.append(without_includes(contract / f'GradFns/{cls}.hpp'))
        source = (contract / f'GradFns/{cls}.cu').read_text(encoding='utf-8')
        bodies = '\n'.join(method(source, signature) for signature in (
            f'{cls}::{cls}()', f'{cls}::~{cls}()',
            f'void {cls}::capture_input(', f'void {cls}::apply_impl(',
            f'void {cls}::release_saved('))
        bodies = re.sub(r'<<<.*?>>>', '', bodies, flags=re.S)
        parts.append('namespace GRIM::autograd {\n' + bodies + '\n}')
    parts.append(helpers)
    parts.append(r'''
namespace {
using namespace GRIM::autograd;
float saved[3] = {1, 1, 1};
std::uint8_t mask[3] = {1, 1, 1};
void prepare(GeluGradFn& n) { n.cached_input = saved; n.cached_size = 3; }
void prepare(DropoutGradFn& n) { n.saved_mask = mask; n.count = 3; }
void prepare(SoftmaxGradFn& n) { n.saved_softmax = saved; n.num_tokens = 1; n.dim = 3; }
void prepare(LogSoftmaxGradFn& n) { n.saved_log_softmax = saved; n.num_tokens = 1; n.dim = 3; }
template<class Node> Tensor unary(Tensor& x) {
    Tensor out = tensor();
    auto n = std::make_shared<Node>();
    const auto before = allocations;
    n->capture_input(x, stream);
    // A non-leaf capture must not allocate any gradient storage.
    if (!x.is_leaf) assert(allocations == before);
    prepare(*n); out.grad_fn = n; out.is_leaf = false;
    return out;
}
template<class Node> void check() {
    // Leaves: add to existing gradients across graphs, leaving input values alone.
    Tensor leaf = tensor(); leaf.ensure_grad();
    std::fill_n(leaf.data, 3, 7.0f);
    std::fill_n(leaf.grad_->data, 3, 5.0f);
    for (int pass = 1; pass <= 2; ++pass) {
        Tensor out = unary<Node>(leaf);
        backward(out);
        assert(last_destination == leaf.grad_->data);
        for (int i = 0; i < 3; ++i) {
            assert(leaf.data[i] == 7);
            assert(leaf.grad_->data[i] == 5 + 2 * pass * (i + 1));
        }
        assert(leaf.grad_->deliveries == pass);
        auto n = std::static_pointer_cast<Node>(out.grad_fn);
        n->release_saved();
        assert(!n->input_producer && !n->leaf_input_gradient && !n->pending_gradient_);
    }
    // Producer destination, both iterative and recursive execution.
    for (bool engine : {true, false}) {
        Tensor sink = tensor(); std::shared_ptr<Identity> producer;
        Tensor x = intermediate(sink, producer);
        Tensor out = unary<Node>(x);
        assert(!producer->pending_gradient_);
        backward(out, engine);
        assert(last_destination == producer->pending_gradient_->data);
        expect(sink, 2); assert(producer->calls == 1);
        // Reapplying the same node must not deliver twice.
        backward(out, false);
        expect(sink, 2); assert(producer->calls == 1);
    }
    // Two consumers share one producer: retain both contributions, schedule once.
    {
        Tensor sink = tensor(); std::shared_ptr<Identity> producer;
        Tensor x = intermediate(sink, producer);
        Tensor a = unary<Node>(x), b = unary<Node>(x);
        Tensor root = add(a, b, stream);
        assert(!producer->pending_gradient_);
        backward(root);
        expect(sink, 4); assert(producer->calls == 1);
        assert(last_destination == producer->pending_gradient_->data);
    }
    // Gradient size validation must reject malformed input before kernel delivery.
    {
        Tensor sink = tensor(); Tensor out = unary<Node>(sink);
        Tensor bad = Tensor::zeros({2}, false, stream, "bad");
        bool rejected = false;
        try { out.grad_fn->apply(bad, stream, nullptr, nullptr); }
        catch (const std::runtime_error&) { rejected = true; }
        assert(rejected && sink.grad_->deliveries == 0);
    }
}
}
int main() {
    check<GeluGradFn>();
    check<DropoutGradFn>();
    check<SoftmaxGradFn>();
    check<LogSoftmaxGradFn>();
    std::cout << "Batch 1 unary GradFn host routing tests passed\n";
}
''')
    compile_and_run('\n'.join(parts))


if __name__ == '__main__':
    main()
