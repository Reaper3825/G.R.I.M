param([switch]$CheckPanelSyntax)
$ErrorActionPreference = 'Stop'
$workspace = Split-Path $PSScriptRoot -Parent
$output = Join-Path $workspace '.codex-build/ui-observatory-tests'
New-Item -ItemType Directory -Force -Path $output | Out-Null
Set-Content -LiteralPath (Join-Path $output '.gitignore') -Value '*' -Encoding ascii
$vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio/Installer/vswhere.exe'
$vs = & $vswhere -latest -products '*' -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
if (-not $vs) { throw 'An existing MSVC C++ toolchain is required.' }
$devcmd = Join-Path $vs 'Common7/Tools/VsDevCmd.bat'
$common = @('/nologo','/std:c++20','/EHsc','/W3','/Zc:__cplusplus','/Zc:preprocessor','/DGRIM_BUILD_HOST','/DNOMINMAX')
$common += @('', '/helpers','/core','/deps/include','/vcpkg_installed/x64-windows/include') | ForEach-Object { '/I"' + $workspace + $_ + '"' }
$arguments = $common + @('"' + $workspace + '/tests/ui_observatory_report_tests.cpp"', '"' + $workspace + '/ui/observatory/observatory_report.cpp"', '/Fe:ui_observatory_report_tests.exe')
Set-Content -LiteralPath (Join-Path $output 'report.rsp') -Value $arguments -Encoding ascii
$commands = @('@echo off', ('call "' + $devcmd + '" -arch=x64 -host_arch=x64 >nul'), 'if errorlevel 1 exit /b %errorlevel%', 'cl @report.rsp', 'if errorlevel 1 exit /b %errorlevel%', 'ui_observatory_report_tests.exe', 'if errorlevel 1 exit /b %errorlevel%')
if ($CheckPanelSyntax) {
    # Validate the real config reader against freshly generated desktop-schema APIs.
    $flatc = Join-Path $workspace 'vcpkg_installed/x64-windows/tools/flatbuffers/flatc.exe'
    if (-not (Test-Path -LiteralPath $flatc)) { throw 'The existing FlatBuffers compiler is required for config-reader validation.' }
    $generated = Join-Path $output 'model_config'
    New-Item -ItemType Directory -Force -Path $generated | Out-Null
    & $flatc --cpp --gen-object-api --gen-mutable -o $generated (Join-Path $workspace 'resources/models/GRIM-text/training/schemas/grim_compiled_hyperparameters.fbs')
    if ($LASTEXITCODE -ne 0) { throw 'Config schema generation failed.' }
    $syntax = $common + '/Zs'
    $syntax += '/I"' + $generated + '"'
    # Match the desktop target's GLM contract (including Cesium's XYZW-only vectors).
    $syntax += @('/DGLM_FORCE_XYZW_ONLY','/DGLM_FORCE_EXPLICIT_CTOR','/DGLM_FORCE_INTRINSICS','/DGLM_ENABLE_EXPERIMENTAL')
    $syntax += @('ui/observatory/observatory_viewport.cpp','ui/observatory/ui_observatory_view.cpp','ui/ui_training_panel.cpp','resources/models/GRIM-text/Shared/ModelConfig/CompiledModelConfig.cpp') | ForEach-Object { '"' + (Join-Path $workspace $_) + '"' }
    Set-Content -LiteralPath (Join-Path $output 'panel-syntax.rsp') -Value $syntax -Encoding ascii
    $commands += @('cl @panel-syntax.rsp','if errorlevel 1 exit /b %errorlevel%')
}
Set-Content -LiteralPath (Join-Path $output 'report.cmd') -Value $commands -Encoding ascii
Push-Location $output
try {
    & $env:ComSpec /d /c report.cmd
    if ($LASTEXITCODE -ne 0) { throw 'Observatory host validation failed.' }
} finally { Pop-Location }
