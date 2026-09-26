"""Round-trip head-gate config with the standalone compiler and production reader.

Requires the host-only ConfigCompiler project built in build/config-compiler.
Does not configure or build a training/runtime target.
"""
from pathlib import Path
import json
import os
import subprocess
import tempfile

from test_add_gradfn_host import compile_and_run


def main():
    tests = Path(__file__).resolve().parent
    repo = tests.parents[3]
    build = repo / 'build/config-compiler'
    compiler = build / ('Release/compile_model_config.exe' if os.name == 'nt' else 'compile_model_config')
    out = Path(tempfile.mkdtemp(prefix='grim-head-gate-config-test-'))
    config = json.loads((repo / 'model_config.json').read_text(encoding='utf-8'))
    artifacts = []
    for enabled in (False, True):
        config['attention_head_gate_enabled'] = enabled
        src, dst = out / f'{enabled}.json', out / f'{enabled}.grimcfg'
        src.write_text(json.dumps(config), encoding='utf-8')
        subprocess.run([str(compiler),'--input',str(src),'--output',str(dst)],check=True,capture_output=True,text=True)
        artifacts.append(dst)
    del config['attention_head_gate_enabled']
    src.write_text(json.dumps(config), encoding='utf-8')
    bad = subprocess.run([str(compiler),'--input',str(src),'--output',str(out/'invalid.grimcfg')],capture_output=True,text=True)
    assert bad.returncode != 0 and 'attention_head_gate_enabled' in bad.stderr
    for source in (repo/'resources/models/model_store').glob('*/model_config.json'):
        subprocess.run([str(compiler),'--input',str(source),'--output',str(out/(source.parent.name+'.grimcfg'))],check=True,capture_output=True,text=True)
    reader = tests.parent / 'Shared/ModelConfig/CompiledModelConfig.cpp'
    cpp = f'#include "{reader.as_posix()}"\n' + r'''
#include <algorithm>
#include <cassert>
#include <iostream>
int main() {
    using namespace GRIM::Config;
''' + f'''
    const auto off = loadCompiledModelConfig({json.dumps(str(artifacts[0]))});
    const auto on = loadCompiledModelConfig({json.dumps(str(artifacts[1]))});
''' + r'''
    assert(!off.features.attention.head_gate_enabled && on.features.attention.head_gate_enabled);
    assert(on.features.attention.residual_gate_enabled);
    assert(on.architecture.num_heads==12 && on.architecture.num_kv_heads==4);
    assert(on.schema_version==9 && on.semantic_version==11);
    const auto capability = CompiledModelCapability::AttentionHeadGate;
    assert(std::find(off.required_capabilities.begin(),off.required_capabilities.end(),capability)==off.required_capabilities.end());
    assert(std::find(on.required_capabilities.begin(),on.required_capabilities.end(),capability)!=on.required_capabilities.end());
    assert(off.integrity.model_compatibility_xxhash64!=on.integrity.model_compatibility_xxhash64);
    std::cout<<"Head gate config compiler/reader/compatibility tests passed\n";
}
'''
    # Compiler flags only for this process's host test; no toolchain mutation.
    if os.name != 'nt':
        raise RuntimeError('Set up equivalent header search paths for this host before running the reader test')
    previous = os.environ.get('CL')
    os.environ['CL'] = (previous or '') + f' /I"{build / "generated"}" /I"{repo / "vcpkg_installed/x64-windows/include"}"'
    try:
        compile_and_run(cpp)
    finally:
        if previous is None:
            os.environ.pop('CL', None)
        else:
            os.environ['CL'] = previous


if __name__ == '__main__':
    main()
