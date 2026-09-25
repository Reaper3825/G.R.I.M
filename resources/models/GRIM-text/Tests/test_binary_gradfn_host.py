"""Batch 2 destination routing tests, using production methods and scheduler.

CUDA launches use CPU multiply/reduction substitutes. No CUDA or training target
is built; this verifies delivery and accumulation, not GPU kernel execution.
"""
from pathlib import Path
import re

from test_add_gradfn_host import compile_and_run, without_includes
from test_unary_gradfn_host import method


def main():
    tests = Path(__file__).resolve().parent
    contract = tests.parent / 'Shared/TensorContract'
    fixture = (tests / 'add_gradfn_host_test.cpp').read_text(encoding='utf-8')
    boundary, helpers = fixture.split('// REAL_IMPLEMENTATIONS')
    helpers = helpers[:helpers.index('int main()')]
    boundary += r'''
inline void trackKernelLaunch(const char*, cudaStream_t) {}
namespace GRIM::autograd {
inline void logGradFlowTensorStats(const char*, const float*, size_t, cudaStream_t) {}
inline std::vector<float*> destinations;
inline void accumulate_grad(float* dst, const float* src, size_t n, float scale,
                            cudaStream_t, const char*) {
    destinations.push_back(dst);
    for (size_t i = 0; i < n; ++i) dst[i] += scale * src[i];
}
inline void kernel_elementwise_mul_backward(const float* dy, const float* other,
                                           float* dst, size_t n) {
    destinations.push_back(dst);
    for (size_t i = 0; i < n; ++i) dst[i] += dy[i] * other[i];
}
inline void launchBiasBackward(const float* dy, float* dst, int rows, int cols, cudaStream_t) {
    destinations.push_back(dst);
    for (int j = 0; j < cols; ++j)
        for (int i = 0; i < rows; ++i) dst[j] += dy[i * cols + j];
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
    for cls in ('ElementwiseMulGradFn', 'BiasAddGradFn'):
        parts.append(without_includes(contract / f'GradFns/{cls}.hpp'))
        source = (contract / f'GradFns/{cls}.cu').read_text(encoding='utf-8')
        bodies = '\n'.join(method(source, signature) for signature in (
            f'{cls}::{cls}()', f'void {cls}::capture_inputs(',
            f'void {cls}::apply_impl(', f'void {cls}::release_saved('))
        bodies = re.sub(r'<<<.*?>>>', '', bodies, flags=re.S)
        parts.append('namespace GRIM::autograd {\n' + bodies + '\n}')
    parts.append(helpers)
    parts.append((tests / 'binary_gradfn_host_test.cpp').read_text(encoding='utf-8'))
    compile_and_run('\n'.join(parts))


if __name__ == '__main__':
    main()
