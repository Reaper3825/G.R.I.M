$ErrorActionPreference = 'Stop'
$workspace = Split-Path $PSScriptRoot -Parent
$output = Join-Path $workspace '.codex-build/model-workspace-ui-syntax'
New-Item -ItemType Directory -Force -Path $output | Out-Null
Set-Content -LiteralPath (Join-Path $output '.gitignore') -Value '*' -Encoding ascii
$vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio/Installer/vswhere.exe'
$vs = & $vswhere -latest -products '*' -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
if (-not $vs) { throw 'An existing MSVC C++ toolchain is required.' }
$devcmd = Join-Path $vs 'Common7/Tools/VsDevCmd.bat'
$arguments = @('/nologo','/std:c++20','/EHsc','/W3','/wd4251','/utf-8','/TP','/DNOMINMAX',('/I"' + $workspace + '"'),('/I"' + $workspace + '/vcpkg_installed/x64-windows/include"'))
$sources = @('plugins/model_workspace_plugin/model_workspace_plugin.cpp','plugins/model_workspace_plugin/ui_training_panel.cpp') + @(Get-ChildItem -LiteralPath (Join-Path $workspace 'plugins/model_workspace_plugin/training_panel') -Filter '*.cpp' | ForEach-Object { 'plugins/model_workspace_plugin/training_panel/' + $_.Name }) 
$arguments += $sources | ForEach-Object { '"' + (Join-Path $workspace $_) + '"' }
$arguments += '/I"' + $workspace + '/helpers"'
$arguments += '/I"' + $workspace + '/external/bgfx.cmake/bgfx/include"'
$arguments += @('/Zs','/DGRIM_BUILD_PLUGIN',('/I' + (Join-Path $workspace 'ui')),('/I' + (Join-Path $workspace 'core')))
Set-Content -LiteralPath (Join-Path $output 'compile.rsp') -Value $arguments -Encoding ascii
$hostArguments = $arguments | Where-Object { $_ -notlike '*.cpp"' -and $_ -ne '/DGRIM_BUILD_PLUGIN' }
$hostArguments += @('/DGRIM_BUILD_HOST', ('"' + (Join-Path $workspace 'ui/model_workspace_services.cpp') + '"'))
Set-Content -LiteralPath (Join-Path $output 'host.rsp') -Value $hostArguments -Encoding ascii
Set-Content -LiteralPath (Join-Path $output 'build.cmd') -Value @('@echo off',('call "' + $devcmd + '" -arch=x64 -host_arch=x64 >nul'),'if errorlevel 1 exit /b %errorlevel%','cl @compile.rsp','if errorlevel 1 exit /b %errorlevel%','cl @host.rsp','if errorlevel 1 exit /b %errorlevel%') -Encoding ascii
Push-Location $output
try {
    & $env:ComSpec /d /c build.cmd
    if ($LASTEXITCODE -ne 0) { throw 'Model Workspace UI syntax check failed.' }
} finally { Pop-Location }


