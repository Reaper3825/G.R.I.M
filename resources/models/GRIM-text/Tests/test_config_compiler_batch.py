"""Exercise batch CLI behavior using an already-built host-only compiler."""
from pathlib import Path
import json
import os
import subprocess
import tempfile


def main():
    repo = Path(__file__).resolve().parents[4]
    build = Path(os.environ.get('GRIM_CONFIG_COMPILER_BUILD', repo / 'build/config-compiler'))
    compiler = build / ('Release/compile_model_config.exe' if os.name == 'nt' else 'compile_model_config')
    compiler = compiler.resolve()
    source = (repo / 'model_config.json').read_text(encoding='utf-8')
    with tempfile.TemporaryDirectory(prefix='grim-config-batch-') as temporary:
        root = Path(temporary)
        store = root / 'model store'
        store.mkdir()
        for name in ('a', 'b', 'source-less'):
            (store / name).mkdir()
        for name in ('a', 'b'):
            (store / name / 'model_config.json').write_text(source, encoding='utf-8')
        # Nested presets are deliberately outside the discovery contract.
        nested = store / 'source-less/nested'
        nested.mkdir()
        (nested / 'model_config.json').write_text(source, encoding='utf-8')
        config = root / 'ai_config.json'

        def write_config(document):
            config.write_text(json.dumps(document), encoding='utf-8')

        def run(*args, ok=True, cwd=None):
            result = subprocess.run([str(compiler), *map(str, args)], cwd=cwd,
                                    capture_output=True, text=True)
            assert (result.returncode == 0) == ok, result.stdout + result.stderr
            return result

        write_config({'paths': {'grim_text': {'model_store': 'model store'}}})
        # Default config path and config-relative store resolution.
        result = run('--batch', cwd=root)
        assert '2 succeeded, 0 failed, 1 skipped' in result.stdout
        assert result.stdout.index(str(store / 'a/model.grimcfg')) < result.stdout.index(str(store / 'b/model.grimcfg'))
        assert not (nested / 'model.grimcfg').exists()
        single = root / 'single.grimcfg'
        run('--input', store / 'a/model_config.json', '--output', single)
        assert single.read_bytes() == (store / 'a/model.grimcfg').read_bytes()
        # A validation failure preserves its old artifact and allows later jobs.
        old = (store / 'a/model.grimcfg').read_bytes()
        (store / 'a/model_config.json').write_text('{}', encoding='utf-8')
        (store / 'b/model.grimcfg').write_bytes(b'old artifact')
        result = run('--batch', '--ai-config', config, ok=False, cwd=repo)
        assert '1 succeeded, 1 failed, 1 skipped' in result.stdout
        assert str(store / 'a/model_config.json') in result.stderr
        assert (store / 'a/model.grimcfg').read_bytes() == old
        assert (store / 'b/model.grimcfg').read_bytes() == single.read_bytes()
        (store / 'a/model_config.json').write_text(source, encoding='utf-8')
        # Absolute store paths and the legacy directive also work.
        write_config({'paths': {'grim_text': {'model_store': str(store)}}})
        run('--batch', '--ai-config', config)
        write_config({'training': {'config': {'grim_text_model_store': 'model store'}}})
        run('--batch', '--ai-config', config)
        for directive in ('', 'missing', 42):
            write_config({'paths': {'grim_text': {'model_store': directive}}})
            run('--batch', '--ai-config', config, ok=False)
        empty = root / 'empty'
        empty.mkdir()
        write_config({'paths': {'grim_text': {'model_store': str(empty)}}})
        assert 'no model_config.json inputs found' in run('--batch', '--ai-config', config, ok=False).stderr
        config.write_text('{', encoding='utf-8')
        run('--batch', '--ai-config', config, ok=False)
        run('--batch', '--input', store / 'a/model_config.json', ok=False)
        run('--batch', '--output', single, ok=False)
        run('--ai-config', config, ok=False)
        run('--batch', '--ai-config', ok=False)
        run('--help')
    print('Config compiler batch tests passed')


if __name__ == '__main__':
    main()
