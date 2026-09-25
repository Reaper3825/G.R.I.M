"""Batch 3 destination routing tests, using production methods and scheduler.

CUDA launches use CPU broadcast/RMSNorm reference substitutes. No CUDA or training target
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
#include <cmath>
constexpr int AUTOGRAD_BLOCK_SIZE = 256;
inline void trackKernelLaunch(const char*, cudaStream_t) {}
namespace GRIM::autograd {
inline std::vector<float*> destinations;
inline void kernel_broadcast_row_mul_backward_x(const float* dy, const float* scale,
                                                float* dst, int rows, int cols) {
    destinations.push_back(dst);
    for (int i = 0; i < rows * cols; ++i) dst[i] += dy[i] * scale[i / cols];
}
inline void kernel_broadcast_row_mul_backward_scale(const float* dy, const float* x,
                                                    float* dst, int rows, int cols) {
    destinations.push_back(dst);
    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < cols; ++j) dst[i] += dy[i * cols + j] * x[i * cols + j];
}
inline void kernel_rmsnorm_backward(const float* dy, const float* x, const float* gamma,
                                   float* dx, float* dg, int rows, int cols, float eps) {
    if (dx) destinations.push_back(dx);
    if (dg) destinations.push_back(dg);
    for (int r = 0; r < rows; ++r) {
        float sq = 0, dot = 0;
        for (int j = 0; j < cols; ++j) {
            int i = r * cols + j;
            sq += x[i] * x[i]; dot += dy[i] * gamma[j] * x[i];
        }
        float rms_sq = sq / cols + eps, inv = 1 / std::sqrt(rms_sq);
        for (int j = 0; j < cols; ++j) {
            int i = r * cols + j;
            if (dx) dx[i] += (dy[i] * gamma[j] - x[i] * dot / (cols * rms_sq)) * inv;
            if (dg) dg[j] += dy[i] * x[i] * inv;
        }
    }
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
    for cls in ('BroadcastRowMulGradFn', 'RMSNormGradFn'):
        parts.append(without_includes(contract / f'GradFns/{cls}.hpp'))
        source = (contract / f'GradFns/{cls}.cu').read_text(encoding='utf-8')
        bodies = '\n'.join(method(source, signature) for signature in (
            f'{cls}::{cls}()', f'void {cls}::capture_inputs(',
            f'void {cls}::apply_impl(', f'void {cls}::release_saved('))
        if cls == 'RMSNormGradFn':
            bodies += '\n' + method(source, 'RMSNormGradFn::~RMSNormGradFn()')
        else:
            bodies += '\n' + method(source, 'void BroadcastRowMulGradFn::set_cache_refs(')
        bodies = re.sub(r'<<<.*?>>>', '', bodies, flags=re.S)
        parts.append('namespace GRIM::autograd {\n' + bodies + '\n}')
    parts.append(helpers)
    parts.append((tests / 'norm_broadcast_gradfn_host_test.cpp').read_text(encoding='utf-8'))
    compile_and_run('\n'.join(parts))


if __name__ == '__main__':
    main()
