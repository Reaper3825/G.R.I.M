"""Host tests for reshape destination routing and the production index mapping.

Execute the real GradFn and engine bodies with host storage. The production
conversion kernel is executed one simulated CUDA thread at a time. This checks
mapping, overwrite/accumulate semantics and routing, but not GPU execution.
"""
from pathlib import Path

from test_add_gradfn_host import compile_and_run, without_includes
from test_unary_gradfn_host import method


def main():
    tests = Path(__file__).resolve().parent
    contract = tests.parent / 'Shared/TensorContract'
    boundary = (tests / 'add_gradfn_host_test.cpp').read_text(encoding='utf-8').split('// REAL_IMPLEMENTATIONS')[0]
    old_shape = method(boundary, 'struct TensorShape')
    boundary = boundary.replace(old_shape, r'''struct TensorShape {
    struct Flat {
        int rows = 0, cols = 0;
        bool operator==(const Flat& b) const { return rows == b.rows && cols == b.cols; }
    } flat;
    struct Heads {
        int batch = 0, heads = 0, seq = 0, head_dim = 0;
        bool operator==(const Heads& b) const {
            return batch == b.batch && heads == b.heads && seq == b.seq && head_dim == b.head_dim;
        }
    } heads;
    size_t count = 0;
    int layout = 0;
    size_t total_elements() const { return count; }
    void require(const char*) const { if (!count) throw std::runtime_error("empty shape"); }
    bool is_2d_layout() const { return layout == 0; }
    bool is_4d() const { return layout == 1; }
    const Flat& as_2d() const { return flat; }
    const Heads& as_4d() const { return heads; }
    static TensorShape make_BSM(int r, int c) {
        TensorShape s; s.flat = {r,c}; s.count = size_t(r)*c; return s;
    }
    static TensorShape make_BHSD(int b, int h, int q, int d) {
        TensorShape s; s.heads = {b,h,q,d}; s.layout = 1; s.count = size_t(b)*h*q*d; return s;
    }
}''')
    conversion = (tests.parent / 'Shared/TensorConversion/TensorConversion.cu').read_text(encoding='utf-8')
    kernel = method(conversion, '__global__ void kernel_BSM_to_BHSD(')
    kernel = kernel.replace('__global__ ', '').replace('__restrict__ ', '')
    wrapper = method(conversion, 'void convert_BSM_to_BHSD(')
    launch = 'kernel_BSM_to_BHSD<<<blocks, BLOCK_SIZE, 0, stream>>>'
    assert launch in wrapper
    wrapper = wrapper.replace(launch, '''for (blockIdx.x = 0; blockIdx.x < blocks; ++blockIdx.x)
        for (threadIdx.x = 0; threadIdx.x < BLOCK_SIZE; ++threadIdx.x)
            kernel_BSM_to_BHSD''')
    geometry = '''namespace GRIM::TensorConversion {
constexpr int BLOCK_SIZE = 256;
struct Index { int x; };
inline Index blockIdx{0}, threadIdx{0}, blockDim{BLOCK_SIZE};
''' + kernel + '\n' + wrapper + '\n}'
    runtime = (contract / 'TensorContract_GPU.cu').read_text(encoding='utf-8')
    start = runtime.index('void GradFn::receive_gradient(')
    end = runtime.index('//======================================================//', start)
    attention = (contract / 'AutogradAttention.cu').read_text(encoding='utf-8')
    node = method(attention, 'struct ReshapeFromBHSDGradFn') + ';'
    parts = [boundary, 'inline int cudaGetLastError() { return 0; }',
             without_includes(contract / 'AutogradEngine.hpp'),
             without_includes(contract / 'GradFns/AddGradFn.hpp'),
             without_includes(contract / 'AutogradEngine.cu'),
             'namespace GRIM {\n' + runtime[start:end] + '\n}',
             without_includes(contract / 'GradFns/AddGradFn.cu'), geometry,
             'namespace GRIM::autograd {\n' + node + '\n}',
             (tests / 'reshape_gradfn_host_test.cpp').read_text(encoding='utf-8')]
    compile_and_run('\n'.join(parts))


if __name__ == '__main__':
    main()
