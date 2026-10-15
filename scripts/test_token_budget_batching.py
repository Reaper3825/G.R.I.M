"""Compile/run focused host-only tests; never build or run a training target."""
from pathlib import Path
import os
import shutil
import subprocess
import tempfile


def main():
    repo = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    if os.name == "nt":
        compiler = shutil.which("cl")
        if not compiler:
            vswhere = Path(os.environ.get("ProgramFiles(x86)", "C:/Program Files (x86)")) / "Microsoft Visual Studio/Installer/vswhere.exe"
            install = subprocess.check_output([
                str(vswhere), "-latest", "-products", "*", "-requires",
                "Microsoft.VisualStudio.Component.VC.Tools.x86.x64", "-property", "installationPath"
            ], text=True).strip()
            if not install:
                raise RuntimeError("An installed MSVC toolchain is required")
            vcvars = Path(install) / "VC/Auxiliary/Build/vcvars64.bat"
            # Capture environment locally; never print environment variables.
            output = subprocess.check_output(
                f'call "{vcvars}" >nul && set', shell=True, text=True)
            for line in output.splitlines():
                if "=" in line and not line.startswith("="):
                    key, value = line.split("=", 1)
                    env[key.upper()] = value
            compiler = shutil.which("cl", path=env["PATH"])
        if not compiler:
            raise RuntimeError("MSVC initialization did not expose cl.exe")
        prefix = [compiler, "/nologo", "/std:c++17", "/EHsc", "/utf-8", "/W3", "/TP"]
    else:
        compiler = shutil.which("c++")
        if not compiler:
            raise RuntimeError("An installed C++ compiler is required")
        prefix = [compiler, "-std=c++17", "-x", "c++"]

    shared = repo / "resources/models/GRIM-text/Shared"
    includes = [repo / "scripts/token_budget_host_stubs", repo,
                repo / "vcpkg_installed/x64-windows/include"]
    tests = [
        ("token_budget", [repo / "scripts/test_token_budget_batching.cpp", shared / "Batching/Batching_GPU.cu"]),
        ("curriculum_ordering", [repo / "scripts/test_curriculum_ordering.cpp", shared / "Batching/Batching_GPU.cu", shared / "Batching/EpochBatching.cu"]),
        ("payload_validation", [repo / "resources/models/GRIM-text/Tests/batch_payload_lm_validation_host_test.cpp"]),
        ("token_budget_config", [repo / "scripts/test_token_budget_config.cpp"]),
        ("token_budget_payload", [repo / "scripts/test_token_budget_payload.cpp",
                                  shared / "Batching/BatchPayload.cu",
                                  shared / "Batching/LocalAtomSelectionData.cu",
                                  shared / "UnigramByte/SequenceLocalAtomTable.cu"]),
    ]
    with tempfile.TemporaryDirectory(prefix="grim-token-budget-") as temporary:
        work = Path(temporary)
        for name, sources in tests:
            executable = work / (name + (".exe" if os.name == "nt" else ""))
            if os.name == "nt":
                args = prefix + [f"/I{p}" for p in includes] + list(map(str, sources)) + [f"/Fe:{executable}"]
            else:
                args = prefix + [f"-I{p}" for p in includes] + list(map(str, sources)) + ["-o", str(executable)]
            result = subprocess.run(args, cwd=work, env=env, capture_output=True, text=True)
            if result.returncode:
                raise RuntimeError(result.stdout + result.stderr)
            subprocess.run([str(executable)], cwd=repo, env=env, check=True)


if __name__ == "__main__":
    main()
