$ErrorActionPreference = 'Stop'
$workspace = Split-Path $PSScriptRoot -Parent
$output = Join-Path $workspace '.codex-build/environment-preparation-tests'
New-Item -ItemType Directory -Force -Path $output | Out-Null
Set-Content -LiteralPath (Join-Path $output '.gitignore') -Value '*' -Encoding ascii
$vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio/Installer/vswhere.exe'
$vs = & $vswhere -latest -products '*' -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
if (-not $vs) { throw 'An existing MSVC C++ toolchain is required.' }
$devcmd = Join-Path $vs 'Common7/Tools/VsDevCmd.bat'
$arguments = @('/nologo','/std:c++20','/EHsc','/W3','/DNOMINMAX',('/I"' + $workspace + '"'),('/I"' + $workspace + '/vcpkg_installed/x64-windows/include"'))
$arguments += @('tests/environment_preparation_tests.cpp','resources/models/GRIM-text/Shared/Environment/EnvironmentEpisode.cpp','ui/world_lab/world_document.cpp') | ForEach-Object { '"' + (Join-Path $workspace $_) + '"' }
$arguments += '/Fe:environment_preparation_tests.exe'
Set-Content -LiteralPath (Join-Path $output 'compile.rsp') -Value $arguments -Encoding ascii
Set-Content -LiteralPath (Join-Path $output 'build.cmd') -Value @('@echo off',('call "' + $devcmd + '" -arch=x64 -host_arch=x64 >nul'),'if errorlevel 1 exit /b %errorlevel%','cl @compile.rsp','if errorlevel 1 exit /b %errorlevel%','environment_preparation_tests.exe') -Encoding ascii
Push-Location $output
try {
    & $env:ComSpec /d /c build.cmd
    if ($LASTEXITCODE -ne 0) { throw 'Environment preparation tests failed.' }
} finally { Pop-Location }
