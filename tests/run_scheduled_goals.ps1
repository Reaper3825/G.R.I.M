$ErrorActionPreference = 'Stop'
$workspace = Split-Path $PSScriptRoot -Parent
$output = Join-Path $workspace '.codex-build/scheduled-goal-tests'
New-Item -ItemType Directory -Force -Path $output | Out-Null
Set-Content -LiteralPath (Join-Path $output '.gitignore') -Value '*' -Encoding ascii
$vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio/Installer/vswhere.exe'
$vs = & $vswhere -latest -products '*' -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
if (-not $vs) { throw 'An existing MSVC C++ toolchain is required.' }
$devcmd = Join-Path $vs 'Common7/Tools/VsDevCmd.bat'
$arguments = @('/nologo','/std:c++20','/EHsc','/W3','/TP','/DNOMINMAX',('/I"' + $workspace + '"'),('/I"' + $workspace + '/vcpkg_installed/x64-windows/include"'))
$arguments += @('tests/scheduled_goal_tests.cpp','resources/models/GRIM-text/Shared/DataLoader/ConceptBlockCorpusReader.cu','resources/models/GRIM-text/Shared/DataLoader/CurriculumRegistry.cu') | ForEach-Object { '"' + (Join-Path $workspace $_) + '"' }
$arguments += '/Fe:scheduled_goal_tests.exe'
Set-Content -LiteralPath (Join-Path $output 'compile.rsp') -Value $arguments -Encoding ascii
Set-Content -LiteralPath (Join-Path $output 'build.cmd') -Value @('@echo off',('call "' + $devcmd + '" -arch=x64 -host_arch=x64 >nul'),'if errorlevel 1 exit /b %errorlevel%','cl @compile.rsp','if errorlevel 1 exit /b %errorlevel%','scheduled_goal_tests.exe') -Encoding ascii
Push-Location $output
try {
    & $env:ComSpec /d /c build.cmd
    if ($LASTEXITCODE -ne 0) { throw 'Scheduled Goal tests failed.' }
} finally { Pop-Location }
