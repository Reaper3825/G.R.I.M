"""Run focused CPU policy tests and syntax checks without building a trainer."""
from pathlib import Path
import argparse
import os
import shutil
import subprocess
import tempfile


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--syntax', action='store_true')
    parser.add_argument('--session', action='store_true')
    args = parser.parse_args()
    tests = Path(__file__).resolve().parent
    root = tests.parents[3]
    shared = tests.parent / 'Shared'
    training = shared / 'UnigramByte/Training'
    env = os.environ.copy()
    compiler = shutil.which('cl')
    if not compiler:
        base = Path(os.environ['ProgramFiles(x86)'])
        compiler = str(sorted((base / 'Microsoft Visual Studio/2022').glob('*/VC/Tools/MSVC/*/bin/Hostx64/x64/cl.exe'))[-1])
        vc = Path(compiler).parents[3]
        kit = base / 'Windows Kits/10'
        version = sorted((kit / 'Include').iterdir())[-1].name
        env['INCLUDE'] = ';'.join(map(str, [vc / 'include'] + [kit / 'Include' / version / x for x in ('ucrt', 'shared', 'um', 'winrt')]))
        env['LIB'] = ';'.join(map(str, [vc / 'lib/x64', kit / 'Lib' / version / 'ucrt/x64', kit / 'Lib' / version / 'um/x64']))
        env['PATH'] = str(Path(compiler).parent) + ';' + env['PATH']
    includes = [root, shared, root / 'vcpkg_installed/x64-windows/include', Path(env['CUDA_PATH']) / 'include']
    flags = [compiler, '/nologo', '/std:c++17', '/EHsc', '/utf-8', '/DNOMINMAX', '/TP'] + ['/I' + str(p) for p in includes]
    with tempfile.TemporaryDirectory(prefix='grim-tokenizer-policy-') as out:
        if args.syntax:
            files = [shared / 'UnigramByte/UnigramTrainer.cu'] + [training / (stem + '.cu') for stem in (
                'SubwordMining', 'UnigramTrainingPrepass', 'UnigramTrainingValidation',
                'UnigramEMConvergence', 'UnigramVocabularyFitting',
                'UnigramCandidateAdmission', 'UnigramCharacterDiagnostics',
                'UnigramCorpusPreparation', 'UnigramExpectationStep',
                'UnigramForwardBackward', 'UnigramMaximizationStep', 'UnigramLikelihoodPruning')]
            subprocess.run(flags + ['/Zs'] + list(map(str, files)), cwd=out, env=env, check=True)
            print('Tokenizer training syntax checks passed')
        elif args.session:
            sources = [tests / 'tokenizer_training_session_host_test.cpp', shared / 'UnigramByte/TextUtils.cu'] + [training / (stem + '.cu') for stem in (
                'UnigramExpectationStep', 'UnigramForwardBackward', 'UnigramMaximizationStep',
                'UnigramLikelihoodPruning', 'UnigramVocabularyFitting',
                'UnigramEMConvergence', 'UnigramTrainingValidation')]
            exe = Path(out) / 'session_test.exe'
            subprocess.run(flags + list(map(str, sources)) + ['/Fe:' + str(exe)], cwd=out, env=env, check=True)
            subprocess.run([str(exe)], cwd=root, env=env, check=True)
        else:
            sources = [tests / 'tokenizer_training_policy_host_test.cpp', shared / 'UnigramByte/TextUtils.cu'] + [training / (stem + '.cu') for stem in (
                'UnigramTrainingValidation', 'UnigramEMConvergence', 'UnigramMaximizationStep')]
            exe = Path(out) / 'policy_test.exe'
            subprocess.run(flags + list(map(str, sources)) + ['/Fe:' + str(exe)], cwd=out, env=env, check=True)
            subprocess.run([str(exe), str(root / 'ai_config.json')], cwd=root, env=env, check=True)


if __name__ == '__main__':
    main()
