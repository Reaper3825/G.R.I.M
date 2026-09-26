"""Run production head-gate autograd/engine code with CPU tensor/conversion shims.

This checks ownership, shape validation, accumulation and graph scheduling.
test_head_gate_cuda.py separately checks the actual CUDA conversion kernels.
"""
from pathlib import Path

from test_add_gradfn_host import compile_and_run, without_includes
from test_unary_gradfn_host import method


def main():
    tests = Path(__file__).resolve().parent
    contract = tests.parent / 'Shared/TensorContract'
    fixture = (tests / 'add_gradfn_host_test.cpp').read_text(encoding='utf-8')
    boundary = fixture.split('// REAL_IMPLEMENTATIONS')[0]
    start = boundary.index('struct TensorShape {')
    end = boundary.index('\n};', start) + 3
    boundary = boundary[:start] + r'''
enum Layout { BSM, BHSD };
struct Dims2 {
    size_t rows, cols;
    bool operator!=(const Dims2& b) const { return rows != b.rows || cols != b.cols; }
    bool operator==(const Dims2& b) const { return !(*this != b); }
};
struct Dims4 {
    int batch, heads, seq, head_dim;
    bool operator!=(const Dims4& b) const {
        return batch != b.batch || heads != b.heads || seq != b.seq || head_dim != b.head_dim;
    }
    bool operator==(const Dims4& b) const { return !(*this != b); }
};
struct TensorShape {
    size_t count = 0;
    Layout layout = BSM;
    Dims2 flat{};
    Dims4 raw{};
    size_t total_elements() const { return count; }
    void require(const char*) const { if (!count) throw std::runtime_error("empty shape"); }
    bool is_2d_layout() const { return layout == BSM; }
    bool is_4d() const { return layout == BHSD; }
    Dims2 as_2d() const { return flat; }
    Dims4 as_4d() const { return raw; }
    static TensorShape make_BSM(int t, int m) {
        return {size_t(t)*m, BSM, {size_t(t),size_t(m)}, {}};
    }
    static TensorShape make_BHSD(int b,int h,int s,int d) {
        return {size_t(b)*h*s*d, BHSD, {}, {b,h,s,d}};
    }
};
''' + boundary[end:]
    boundary = boundary.replace('    void receive_gradient(',
                                '    void release_consumed_gradient(cudaStream_t) { pending_gradient_.reset(); }\n    void receive_gradient(')
    boundary += r'''
#include <cmath>
#include <limits>
inline void trackKernelLaunch(const char*, cudaStream_t) {}
namespace TensorConversion {
void head_gate_BHSD_to_BSM(const float* x,const float* g,float* y,
                          int B,int H,int S,int D,cudaStream_t) {
    for(int b=0;b<B;++b) for(int s=0;s<S;++s) for(int h=0;h<H;++h) for(int d=0;d<D;++d)
        y[((b*S+s)*H+h)*D+d] = x[((b*H+h)*S+s)*D+d]*g[(b*S+s)*H+h];
}
void head_gate_BHSD_to_BSM_backward(const float* dy,const float* x,const float* g,
                                   float* dx,float* dg,int B,int H,int S,int D,cudaStream_t) {
    for(int b=0;b<B;++b) for(int s=0;s<S;++s) for(int h=0;h<H;++h) for(int d=0;d<D;++d) {
        int i=((b*S+s)*H+h)*D+d, j=((b*H+h)*S+s)*D+d, k=(b*S+s)*H+h;
        if(dx) dx[j] += dy[i]*g[k];
        if(dg) dg[k] += dy[i]*x[j];
    }
}
}
'''
    runtime = (contract / 'TensorContract_GPU.cu').read_text(encoding='utf-8')
    start = runtime.index('void GradFn::receive_gradient(')
    end = runtime.index('//======================================================//', start)
    methods = runtime[start:end]
    # The host storage shim uses shared vectors, not cudaFreeAsync allocations.
    methods = methods.replace(method(methods, 'void GradFn::release_consumed_gradient('), '')
    parts = [boundary,
             without_includes(contract / 'AutogradEngine.hpp'),
             without_includes(contract / 'AutogradEngine.cu'),
             'namespace GRIM {\n' + methods + '\n}',
             without_includes(contract / 'GradFns/HeadGateGradFn.hpp'),
             without_includes(contract / 'GradFns/HeadGateGradFn.cu'),
             (tests / 'head_gate_gradfn_host_test.cpp').read_text(encoding='utf-8')]
    compile_and_run('\n'.join(parts))


if __name__ == '__main__':
    main()
