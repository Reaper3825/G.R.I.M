$ErrorActionPreference = 'Stop'
$workspace = Split-Path $PSScriptRoot -Parent
$output = Join-Path $workspace '.codex-build/dataset-stream-tests'
New-Item -ItemType Directory -Force -Path $output | Out-Null
Set-Content -LiteralPath (Join-Path $output '.gitignore') -Value '*' -Encoding ascii
$generated = Join-Path $output 'generated/grim_concept'
New-Item -ItemType Directory -Force -Path $generated | Out-Null
$flatc = Join-Path $workspace 'vcpkg_installed/x64-windows/tools/flatbuffers/flatc.exe'
& $flatc --cpp --scoped-enums -o $generated (Join-Path $workspace 'DataCollection/concept_block.fbs')
if ($LASTEXITCODE -ne 0) { throw 'ConceptBlock schema generation failed.' }
$vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio/Installer/vswhere.exe'
$vs = & $vswhere -latest -products '*' -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
if (-not $vs) { throw 'An existing MSVC C++ toolchain is required.' }
$devcmd = Join-Path $vs 'Common7/Tools/VsDevCmd.bat'
$arguments = @('/nologo','/std:c++20','/EHsc','/W3','/DNOMINMAX','/DGRIM_BUILD_HOST',('/I"' + $workspace + '"'),('/I"' + $workspace + '/vcpkg_installed/x64-windows/include"'),('/I"' + $output + '/generated"'))
$arguments += @('tests/dataset_stream_tests.cpp','bootstrap/dataset_bootstrap.cpp','DataCollection/dataset_stream.cpp','DataCollection/dataset_catalog.cpp','DataCollection/dataset_target.cpp','DataCollection/dataset_creation_service.cpp','DataCollection/concept_generator.cpp','DataCollection/io/dataset_io_json.cpp','DataCollection/io/concept_block_io_flatbuffer.cpp') | ForEach-Object { '"' + (Join-Path $workspace $_) + '"' }
$arguments += '/Fe:dataset_stream_tests.exe'
Set-Content -LiteralPath (Join-Path $output 'compile.rsp') -Value $arguments -Encoding ascii
Set-Content -LiteralPath (Join-Path $output 'build.cmd') -Value @('@echo off',('call "' + $devcmd + '" -arch=x64 -host_arch=x64 >nul'),'if errorlevel 1 exit /b %errorlevel%','cl @compile.rsp','if errorlevel 1 exit /b %errorlevel%','dataset_stream_tests.exe') -Encoding ascii
Push-Location $output
try {
    & $env:ComSpec /d /c build.cmd
    if ($LASTEXITCODE -ne 0) { throw 'Dataset stream tests failed.' }
} finally { Pop-Location }
