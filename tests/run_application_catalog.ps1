$ErrorActionPreference = 'Stop'
$workspace = Split-Path $PSScriptRoot -Parent
$output = Join-Path $workspace '.codex-build/application-catalog-tests'
New-Item -ItemType Directory -Force -Path $output | Out-Null
Set-Content -LiteralPath (Join-Path $output '.gitignore') -Value '*' -Encoding ascii
$vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio/Installer/vswhere.exe'
$vs = & $vswhere -latest -products '*' -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
if (-not $vs) { throw 'An existing MSVC toolchain is required.' }
$devcmd = Join-Path $vs 'Common7/Tools/VsDevCmd.bat'
$common = @('/nologo','/std:c++20','/EHsc','/W4','/utf-8','/DNOMINMAX',('/I"' + $workspace + '"'),('/I"' + $workspace + '/vcpkg_installed/x64-windows/include"'))
$arguments = $common + (@('tests/application_catalog_tests.cpp','core/applications/application_catalog.cpp','MMO/Core/ToolRegistry.cpp') | ForEach-Object { '"' + (Join-Path $workspace $_) + '"' }) + '/Fe:application_catalog_tests.exe'
Set-Content -LiteralPath (Join-Path $output 'catalog.rsp') -Value $arguments -Encoding ascii
$smoke = $common + (@('tests/application_discovery_smoke.cpp','core/applications/application_backend_windows.cpp','core/applications/application_catalog.cpp','MMO/Core/ToolRegistry.cpp') | ForEach-Object { '"' + (Join-Path $workspace $_) + '"' }) + @('/Fe:application_discovery_smoke.exe','/link','shell32.lib','ole32.lib','uuid.lib')
Set-Content -LiteralPath (Join-Path $output 'smoke.rsp') -Value $smoke -Encoding ascii
Set-Content -LiteralPath (Join-Path $output 'test.cmd') -Value @('@echo off',('call "' + $devcmd + '" -arch=x64 -host_arch=x64 >nul'),'if errorlevel 1 exit /b %errorlevel%','cl @catalog.rsp','if errorlevel 1 exit /b %errorlevel%','application_catalog_tests.exe','if errorlevel 1 exit /b %errorlevel%','cl @smoke.rsp','if errorlevel 1 exit /b %errorlevel%','application_discovery_smoke.exe') -Encoding ascii
Push-Location $output
try { & $env:ComSpec /d /c test.cmd; if ($LASTEXITCODE -ne 0) { throw 'Application catalog tests failed.' } }
finally { Pop-Location }
