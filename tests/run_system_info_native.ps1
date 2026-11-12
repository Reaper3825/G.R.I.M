$ErrorActionPreference = 'Stop'
$workspace = Split-Path $PSScriptRoot -Parent
$output = Join-Path $workspace '.codex-build/system-info-native-tests'
New-Item -ItemType Directory -Force -Path $output | Out-Null
Set-Content -LiteralPath (Join-Path $output '.gitignore') -Value '*' -Encoding ascii
$vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio/Installer/vswhere.exe'
$vs = & $vswhere -latest -products '*' -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
if (-not $vs) { throw 'An existing MSVC toolchain is required.' }
$devcmd = Join-Path $vs 'Common7/Tools/VsDevCmd.bat'
$common = @('/nologo','/std:c++20','/EHsc','/W4','/utf-8','/DNOMINMAX','/DWIN32_LEAN_AND_MEAN',('/I"' + $workspace + '"'),('/I"' + $workspace + '/vcpkg_installed/x64-windows/include"'))
$link = @('/link','dxgi.lib','ole32.lib','oleaut32.lib','wbemuuid.lib','wlanapi.lib','user32.lib','comsuppw.lib')
$native = '"' + (Join-Path $workspace 'plugins/system_info_plugin/hardware_detection.cpp') + '"'
$smoke = $common + @(('"' + (Join-Path $workspace 'tests/system_info_native_smoke.cpp') + '"'),$native,'/Fe:system_info_native_smoke.exe') + $link
Set-Content -LiteralPath (Join-Path $output 'smoke.rsp') -Value $smoke -Encoding ascii
$dll = $common + @('/LD','/DGRIM_BUILD_PLUGIN',$native)
$dll += @('plugins/system_info_plugin/system_info_plugin.cpp','plugins/system_info_plugin/system_info_provider.cpp','plugins/system_info_plugin/system_info_command.cpp','plugins/plugin_logging.cpp') | ForEach-Object { '"' + (Join-Path $workspace $_) + '"' }
$dll += @('/Fe:system_info_plugin.dll') + $link
Set-Content -LiteralPath (Join-Path $output 'native.rsp') -Value $dll -Encoding ascii
Set-Content -LiteralPath (Join-Path $output 'test.cmd') -Value @('@echo off',('call "' + $devcmd + '" -arch=x64 -host_arch=x64 >nul'),'if errorlevel 1 exit /b %errorlevel%','cl @native.rsp','if errorlevel 1 exit /b %errorlevel%','cl @smoke.rsp','if errorlevel 1 exit /b %errorlevel%','system_info_native_smoke.exe') -Encoding ascii
Push-Location $output
try { & $env:ComSpec /d /c test.cmd; if ($LASTEXITCODE -ne 0) { throw 'Native system information tests failed.' } }
finally { Pop-Location }
