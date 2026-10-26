# Focused host-only schema/storage/transport checks. No application or trainer.
$ErrorActionPreference = 'Stop'
$workspace = Split-Path $PSScriptRoot -Parent
$output = Join-Path $workspace '.codex-build/goal-evidence'
New-Item -ItemType Directory -Force -Path (Join-Path $output 'include/grim_concept') | Out-Null
Set-Content -LiteralPath (Join-Path $output '.gitignore') -Value '*' -Encoding ascii
$flatc = Join-Path $workspace 'vcpkg_installed/x64-windows/tools/flatbuffers/flatc.exe'
& $flatc --cpp --scoped-enums -o (Join-Path $output 'include/grim_concept') (Join-Path $workspace 'DataCollection/concept_block.fbs')
if ($LASTEXITCODE -ne 0) { throw 'Schema generation failed' }
$vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio/Installer/vswhere.exe'
$vs = & $vswhere -latest -products '*' -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
if (-not $vs) { throw 'An existing MSVC C++ toolchain is required.' }
$devcmd = Join-Path $vs 'Common7/Tools/VsDevCmd.bat'
$cudaInclude = Join-Path $env:CUDA_PATH 'include'
if (-not (Test-Path (Join-Path $cudaInclude 'cuda_runtime.h'))) { throw 'Existing CUDA headers required by span metadata are unavailable.' }
$cases = @(
    @('scripts/test_concept_block_flatbuffer_roundtrip.cpp', 'DataCollection/io/concept_block_io_flatbuffer.cpp'),
    @('scripts/test_reasoning_state_transport.cpp'),
    @('scripts/test_state_label_rendering.cpp'),
    @('tests/concept_section_labels_tests.cpp'),
    @('tests/concept_section_mask_tests.cpp', 'resources/models/GRIM-text/training/Phases/Startup/ConceptSupervision.cu', 'resources/models/GRIM-text/training/Phases/Startup/SlidingWindow.cu', 'resources/models/GRIM-text/Shared/UnigramByte/SequenceLocalAtomTable.cu')
)
foreach ($case in $cases) {
    $name = [IO.Path]::GetFileNameWithoutExtension($case[0])
    $arguments = @('/nologo','/std:c++20','/EHsc','/W3','/TP','/DNOMINMAX',('/I"' + $workspace + '"'),('/I"' + $output + '/include"'),('/I"' + $workspace + '/vcpkg_installed/x64-windows/include"'))
    $arguments += ('/I"' + $cudaInclude + '"')
    $arguments += $case | ForEach-Object { '"' + (Join-Path $workspace $_) + '"' }
    $arguments += ('/Fe:' + $name + '.exe')
    Set-Content -LiteralPath (Join-Path $output 'compile.rsp') -Value $arguments -Encoding ascii
    Set-Content -LiteralPath (Join-Path $output 'build.cmd') -Value @('@echo off',('call "' + $devcmd + '" -arch=x64 -host_arch=x64 >nul'),'if errorlevel 1 exit /b %errorlevel%','cl @compile.rsp') -Encoding ascii
    Push-Location $output
    try {
        & $env:ComSpec /d /c build.cmd
        if ($LASTEXITCODE -ne 0) { throw "Host-only compilation failed: $name" }
    } finally { Pop-Location }
    Push-Location $workspace
    try {
        & (Join-Path $output ($name + '.exe'))
        if ($LASTEXITCODE -ne 0) { throw "Host-only check failed: $name" }
    } finally { Pop-Location }
}
