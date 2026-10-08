$ErrorActionPreference = 'Stop'
$workspace = Split-Path $PSScriptRoot -Parent
$output = Join-Path $workspace '.codex-build/ui-observatory-tests'
New-Item -ItemType Directory -Force -Path $output | Out-Null
Set-Content -LiteralPath (Join-Path $output '.gitignore') -Value '*' -Encoding ascii
$vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio/Installer/vswhere.exe'
$vs = & $vswhere -latest -products '*' -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
if (-not $vs) { throw 'An existing MSVC C++ toolchain is required.' }
$devcmd = Join-Path $vs 'Common7/Tools/VsDevCmd.bat'
$sources = @('tests/ui_observatory_prerequisites_tests.cpp','ui/primitives/ui_slider.cpp','ui/primitives/ui_graph.cpp','ui/ui_orbit_camera.cpp','helpers/widget.cpp','helpers/vector2.cpp','ui/ui_focus_manager.cpp')
$argsFile = Join-Path $output 'compile.rsp'
$arguments = @('/nologo','/std:c++17','/EHsc','/W3','/DGRIM_BUILD_HOST','/DNOMINMAX',('/I"' + $workspace + '"'),('/I"' + $workspace + '/helpers"'),('/I"' + $workspace + '/core"'),('/I"' + $workspace + '/vcpkg_installed/x64-windows/include"'))
$arguments += $sources | ForEach-Object { '"' + (Join-Path $workspace $_) + '"' }
$arguments += '/Fe:ui_observatory_prerequisites_tests.exe'
Set-Content -LiteralPath $argsFile -Value $arguments -Encoding ascii
$buildScript = Join-Path $output 'build.cmd'
Set-Content -LiteralPath $buildScript -Value @(
    '@echo off',
    ('call "' + $devcmd + '" -arch=x64 -host_arch=x64 >nul'),
    'if errorlevel 1 exit /b %errorlevel%',
    'cl @compile.rsp'
) -Encoding ascii
Push-Location $output
try {
    & $env:ComSpec /d /c build.cmd
    if ($LASTEXITCODE -ne 0) { throw 'Host-only test compilation failed.' }
    & (Join-Path $output 'ui_observatory_prerequisites_tests.exe')
    if ($LASTEXITCODE -ne 0) { throw 'Host-only tests failed.' }
} finally { Pop-Location }
