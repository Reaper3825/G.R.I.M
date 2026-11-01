$ErrorActionPreference = 'Stop'
$workspace = Split-Path $PSScriptRoot -Parent
$output = Join-Path $workspace '.codex-build/click-target-tests'
New-Item -ItemType Directory -Force -Path $output | Out-Null
Set-Content -LiteralPath (Join-Path $output '.gitignore') -Value '*' -Encoding ascii
$vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio/Installer/vswhere.exe'
$vs = & $vswhere -latest -products '*' -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
if (-not $vs) { throw 'An existing MSVC C++ toolchain is required.' }
$devcmd = Join-Path $vs 'Common7/Tools/VsDevCmd.bat'
$includes = @('', '/helpers', '/core', '/commands', '/ai', '/vcpkg_installed/x64-windows/include', '/vcpkg_installed/x64-windows/include/opencv4') | ForEach-Object { '/I"' + $workspace + $_ + '"' }
$common = @('/nologo','/std:c++20','/EHsc','/utf-8','/W3','/DGRIM_BUILD_HOST','/DNOMINMAX','/DWIN32_LEAN_AND_MEAN') + $includes
$arguments = $common + @('tests/click_target_tests.cpp','perception/click_target.cpp','commands/commands_click_on.cpp','commands/commands_perception.cpp','perception/input_command_service.cpp','ai/action_executor.cpp') | ForEach-Object { if ($_ -like '*.cpp') { '"' + (Join-Path $workspace $_) + '"' } else { $_ } }
$arguments += '/Fe:click_target_tests.exe'
$arguments += '/link', ('"' + (Join-Path $workspace 'vcpkg_installed/x64-windows/lib/opencv_core4.lib') + '"')
Set-Content -LiteralPath (Join-Path $output 'test.rsp') -Value $arguments -Encoding ascii
$syntax = $common + @('/Zs','perception/click_target_host.cpp','perception/perception_context.cpp','core/input/InputController.cpp','commands/commands_perception.cpp','ai/action_executor.cpp','plugins/core_plugin.cpp') | ForEach-Object { if ($_ -like '*.cpp') { '"' + (Join-Path $workspace $_) + '"' } else { $_ } }
Set-Content -LiteralPath (Join-Path $output 'syntax.rsp') -Value $syntax -Encoding ascii
Set-Content -LiteralPath (Join-Path $output 'build.cmd') -Value @('@echo off', ('call "' + $devcmd + '" -arch=x64 -host_arch=x64 >nul'), 'if errorlevel 1 exit /b %errorlevel%', ('set "PATH=' + (Join-Path $workspace 'vcpkg_installed/x64-windows/bin') + ';%PATH%"'), 'cl @test.rsp', 'if errorlevel 1 exit /b %errorlevel%', 'click_target_tests.exe', 'if errorlevel 1 exit /b %errorlevel%', 'cl @syntax.rsp') -Encoding ascii
Push-Location $output
try {
    & $env:ComSpec /d /c build.cmd
    if ($LASTEXITCODE -ne 0) { throw 'Click target tests or production syntax checks failed.' }
} finally { Pop-Location }
