"""Host regression for production fused loss math and autograd routing.

No CUDA/training target is built. The math test executes the actual CUDA kernel
bodies with CPU threads emulating block barriers and warp shuffles. This checks
equations/indexing, not CUDA compilation, GPU synchronization or performance.
"""
from pathlib import Path
from test_add_gradfn_host import compile_and_run, without_includes
from test_unary_gradfn_host import method

ROOT = Path(__file__).resolve().parent.parent
LOSS = ROOT / 'Shared/Loss/ComputeLoss'
CONTRACT = ROOT / 'Shared/TensorContract'


def math_test():
    source = (LOSS / 'CrossEntropyNLL.cu').read_text(encoding='utf-8')
    kernels = '\n'.join(method(source, name) for name in (
        '__global__ void kernelCrossEntropyNLLForward(',
        '__device__ __forceinline__ float crossEntropyLogProbGradient(',
        '__global__ void kernelCrossEntropyLogitsBackward('))
    hp = (ROOT / 'Shared/HyperParameters/HyperparameterGroupings.hpp').read_text(encoding='utf-8')
    config = method(hp, 'struct LossConfigHP') + ';'
    fixture = Path(__file__).with_name('text_loss_math_host_test.cpp').read_text(encoding='utf-8')
    compile_and_run(fixture.replace('// LOSS_CONFIG', config).replace('// LOSS_KERNELS', kernels))


def routing_test():
    tests = Path(__file__).resolve().parent
    boundary = (tests / 'add_gradfn_host_test.cpp').read_text(encoding='utf-8').split('// REAL_IMPLEMENTATIONS')[0]
    boundary = boundary.replace('struct BatchPayload {};', 'struct BatchPayload { int total_tokens=1, vocab_size=3, lm_valid_tokens=1; std::vector<int> target_ids{0}; };')
    boundary = boundary.replace('size_t as_2d() const { return count; }', '''
    struct Dims { int rows, cols; bool operator==(const Dims& o) const { return rows==o.rows && cols==o.cols; } };
    Dims as_2d() const { return {1, int(count)}; }''')
    boundary = boundary.replace('enum class Kind { EngineGradient };', 'enum class Kind { EngineGradient, Saved };\ninline int freeAsync(void*, cudaStream_t) { return 0; }')
    boundary = boundary.replace('bool applied = false;', 'bool applied = false; bool released_ = false;')
    boundary = boundary.replace('virtual void release_saved() {', 'void release_consumed_gradient(cudaStream_t);\n    virtual void release_saved() { released_ = true;')
    runtime = (CONTRACT / 'TensorContract_GPU.cu').read_text(encoding='utf-8')
    start = runtime.index('void GradFn::receive_gradient(')
    end = runtime.index('//======================================================//', start)
    node = method((LOSS / 'AutogradLoss.cu').read_text(encoding='utf-8'), 'struct TextLossGradFn') + ';'
    fixture = (tests / 'text_loss_routing_host_test.cpp').read_text(encoding='utf-8')
    stubs, checks = fixture.split('// PRODUCTION_NODE')
    compile_and_run('\n'.join([
        boundary, without_includes(CONTRACT / 'AutogradEngine.hpp'),
        without_includes(CONTRACT / 'GradFns/AddGradFn.hpp'),
        without_includes(CONTRACT / 'AutogradEngine.cu'),
        'namespace GRIM {\n' + runtime[start:end] + '\n}',
        without_includes(CONTRACT / 'GradFns/AddGradFn.cu'), stubs,
        'namespace GRIM::autograd {\n' + node + '\n}', checks]))


if __name__ == '__main__':
    math_test()
    routing_test()
