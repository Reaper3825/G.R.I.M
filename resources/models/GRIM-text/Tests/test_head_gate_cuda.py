"""Build/run only a standalone head-gate kernel test, never a model runtime.

Extracts the unmodified production conversion kernels/wrappers to avoid linking
the trainer. Uses the installed CUDA/MSVC toolchain; installs no dependencies.
"""
from pathlib import Path
import os
import shutil
import subprocess
import tempfile

from test_unary_gradfn_host import method


def main():
    tests = Path(__file__).resolve().parent
    source = (tests.parent / 'Shared/TensorConversion/TensorConversion.cu').read_text(encoding='utf-8')
    hp = (tests.parent / 'Shared/HyperParameters/HyperParameters_GPU.hpp').read_text(encoding='utf-8')
    import re
    block_size = re.search(r'CUDA_BLOCK_SIZE_STANDARD\s*=\s*(\d+)', hp)[1]
    signatures = ['dim3 gridForCount(', '__device__ std::size_t globalLinearIndex()',
                  '__global__ void kernel_head_gate_flatten(',
                  '__global__ void kernel_head_gate_backward_input(',
                  '__global__ void kernel_head_gate_backward_gate(',
                  'void head_gate_BHSD_to_BSM(', 'void head_gate_BHSD_to_BSM_backward(']
    code = '''#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <vector>
#include <stdexcept>
#include <limits>
#include <algorithm>
namespace TensorConversion {
'''
    code += f'constexpr int BLOCK_SIZE = {block_size};\nconstexpr int kMaxGridDimY = 65535;\n'
    code += '\n'.join(method(source, name) for name in signatures) + '\n}\n'
    code += (tests / 'head_gate_cuda_test.cu').read_text(encoding='utf-8')
    out = Path(tempfile.mkdtemp(prefix='grim-head-gate-cuda-test-'))
    cu = out / 'test.cu'
    cu.write_text(code, encoding='utf-8')
    nvcc = shutil.which('nvcc')
    if not nvcc:
        raise RuntimeError('Installed nvcc is required for this CUDA test')
    env = os.environ.copy()
    args = [nvcc, '-std=c++17', '-arch=native']
    if os.name == 'nt':
        base = Path(os.environ['ProgramFiles(x86)'])
        compilers = sorted((base / 'Microsoft Visual Studio/2022').glob('*/VC/Tools/MSVC/*/bin/Hostx64/x64/cl.exe'))
        compiler = Path(shutil.which('cl') or compilers[-1])
        vc = compiler.parents[3]
        kit = base / 'Windows Kits/10'
        version = sorted((kit / 'Include').iterdir())[-1].name
        env['INCLUDE'] = ';'.join(map(str, [vc/'include'] + [kit/'Include'/version/x for x in ('ucrt','shared','um','winrt')]))
        env['LIB'] = ';'.join(map(str, [vc/'lib/x64',kit/'Lib'/version/'ucrt/x64',kit/'Lib'/version/'um/x64']))
        env['PATH'] = str(compiler.parent) + ';' + env['PATH']
        args += ['-ccbin', str(compiler.parent)]
    exe = out / ('test.exe' if os.name == 'nt' else 'test')
    subprocess.run(args+[str(cu),'-o',str(exe)],cwd=out,env=env,check=True)
    subprocess.run([str(exe)],cwd=out,env=env,check=True)


if __name__ == '__main__':
    main()
