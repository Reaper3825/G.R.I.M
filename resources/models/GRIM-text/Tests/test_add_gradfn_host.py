"""Compile only a CPU routing test, using production Add/engine method bodies.

No CUDA/training target is compiled or executed. The host shim substitutes
Tensor storage and accumulation kernels; it does not validate GPU execution.
"""
from pathlib import Path
import os
import re
import shutil
import subprocess
import tempfile


def without_includes(path):
    return re.sub(r'^\s*#(?:include|pragma once)[^\n]*', '', path.read_text(encoding='utf-8'), flags=re.M)


def main():
    tests = Path(__file__).resolve().parent
    contract = tests.parent / 'Shared/TensorContract'
    runtime = (contract / 'TensorContract_GPU.cu').read_text(encoding='utf-8')
    start = runtime.index('void GradFn::receive_gradient(')
    end = runtime.index('//======================================================//', start)
    implementations = '\n'.join([
        without_includes(contract / 'AutogradEngine.hpp'),
        without_includes(contract / 'GradFns/AddGradFn.hpp'),
        without_includes(contract / 'AutogradEngine.cu'),
        'namespace GRIM {\n' + runtime[start:end] + '\n}',
        without_includes(contract / 'GradFns/AddGradFn.cu'),
    ])
    source = (tests / 'add_gradfn_host_test.cpp').read_text(encoding='utf-8')
    source = source.replace('// REAL_IMPLEMENTATIONS', implementations)
    # Persistent temporary output directory; no repository/runtime build changes.
    out = Path(tempfile.mkdtemp(prefix='grim-add-gradfn-test-'))
    cpp = out / 'test.cpp'
    cpp.write_text(source, encoding='utf-8')
    env = os.environ.copy()
    compiler = shutil.which('cl') if os.name == 'nt' else shutil.which('c++')
    if os.name == 'nt' and not compiler:
        base = Path(os.environ['ProgramFiles(x86)'])
        candidates = sorted((base / 'Microsoft Visual Studio/2022').glob('*/VC/Tools/MSVC/*/bin/Hostx64/x64/cl.exe'))
        if not candidates:
            raise RuntimeError('No installed host C++ compiler found')
        compiler = str(candidates[-1])
        vc = Path(compiler).parents[3]
        kit = base / 'Windows Kits/10'
        version = sorted((kit / 'Include').iterdir())[-1].name
        env['INCLUDE'] = ';'.join(map(str, [vc / 'include'] + [kit / 'Include' / version / x for x in ('ucrt', 'shared', 'um', 'winrt')]))
        env['LIB'] = ';'.join(map(str, [vc / 'lib/x64', kit / 'Lib' / version / 'ucrt/x64', kit / 'Lib' / version / 'um/x64']))
        env['PATH'] = str(Path(compiler).parent) + ';' + env['PATH']
    if not compiler:
        raise RuntimeError('No installed host C++ compiler found')
    exe = out / ('test.exe' if os.name == 'nt' else 'test')
    args = ([compiler, '/nologo', '/std:c++17', '/EHsc', str(cpp), '/Fe:' + str(exe)]
            if os.name == 'nt' else [compiler, '-std=c++17', str(cpp), '-o', str(exe)])
    subprocess.run(args, cwd=out, env=env, check=True)
    subprocess.run([str(exe)], cwd=out, env=env, check=True)


if __name__ == '__main__':
    main()
